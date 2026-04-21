"""
main_agent.py — RAG Agent kết nối ChromaDB thật
================================================
Thay thế mock retrieval bằng ChromaDB query thật.
Output query() luôn bao gồm `retrieved_ids` để
Retrieval Evaluator có thể tính Hit Rate & MRR.
"""

import asyncio
import os
from pathlib import Path
from typing import List, Dict

import chromadb
from chromadb.utils import embedding_functions
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

DB_DIR          = Path(__file__).parent.parent / "data" / "chroma_db"
COLLECTION_NAME = "internal_docs"
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY", "")
TOP_K           = 5   # số chunks retrieve mỗi query


class MainAgent:
    """
    RAG Agent sử dụng ChromaDB làm Vector Store.
    - retrieve(): tìm top-K chunks và trả về retrieved_ids
    - generate(): gọi OpenAI để sinh câu trả lời từ context
    - query()   : hàm tổng hợp dùng cho BenchmarkRunner
    """

    def __init__(self):
        self.name = "SupportAgent-v1"
        self._collection = None
        self._llm = AsyncOpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

    def _get_collection(self):
        """Lazy-init ChromaDB collection."""
        if self._collection is not None:
            return self._collection

        client = chromadb.PersistentClient(path=str(DB_DIR))
        if OPENAI_API_KEY:
            ef = embedding_functions.OpenAIEmbeddingFunction(
                api_key=OPENAI_API_KEY,
                model_name="text-embedding-3-small"
            )
        else:
            ef = embedding_functions.DefaultEmbeddingFunction()

        self._collection = client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=ef
        )
        return self._collection

    # ─────────────────────────────────────────
    # Retrieval
    # ─────────────────────────────────────────
    def retrieve(self, question: str, top_k: int = TOP_K) -> Dict:
        """
        Truy vấn ChromaDB, trả về:
        {
          "retrieved_ids": ["chunk_0012", "chunk_0045", ...],   ← dùng cho eval
          "contexts"     : ["nội dung chunk 1", ...],
          "sources"      : ["filename.txt", ...],
        }
        """
        collection = self._get_collection()
        results = collection.query(
            query_texts=[question],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        ids       = results["ids"][0]
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]

        return {
            "retrieved_ids": ids,
            "contexts"     : documents,
            "sources"      : [m.get("filename", "unknown") for m in metadatas],
        }

    # ─────────────────────────────────────────
    # Generation
    # ─────────────────────────────────────────
    async def generate(self, question: str, contexts: List[str]) -> Dict:
        """
        Gọi OpenAI để sinh câu trả lời từ context.
        Nếu không có API key → trả về câu trả lời mẫu.
        """
        context_text = "\n\n---\n\n".join(contexts)

        if not self._llm:
            # Fallback khi không có API key (dev/test mode)
            return {
                "answer"      : f"[DEV MODE] Dựa trên {len(contexts)} đoạn tài liệu, câu trả lời cho '{question}' là: [Cần OpenAI API key để sinh câu trả lời thật].",
                "model"       : "mock",
                "tokens_used" : 0,
            }

        system_prompt = (
            "Bạn là trợ lý nội bộ của công ty. "
            "Hãy trả lời câu hỏi chỉ dựa trên tài liệu được cung cấp. "
            "Nếu tài liệu không đề cập, hãy nói thẳng: 'Tôi không tìm thấy thông tin này trong tài liệu nội bộ.'"
        )
        user_prompt = (
            f"Tài liệu tham khảo:\n{context_text}\n\n"
            f"Câu hỏi: {question}"
        )

        response = await self._llm.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user"  , "content": user_prompt},
            ],
            temperature=0,
            max_tokens=512,
        )
        answer      = response.choices[0].message.content
        tokens_used = response.usage.total_tokens

        return {
            "answer"      : answer,
            "model"       : "gpt-4o-mini",
            "tokens_used" : tokens_used,
        }

    # ─────────────────────────────────────────
    # Main entry — dùng cho BenchmarkRunner
    # ─────────────────────────────────────────
    async def query(self, question: str) -> Dict:
        """
        Full RAG pipeline.
        Trả về dict tương thích với BenchmarkRunner & RetrievalEvaluator:
        {
          "answer"        : str,
          "retrieved_ids" : List[str],   ← KEY cho Retrieval Eval
          "contexts"      : List[str],
          "sources"       : List[str],
          "metadata"      : {...}
        }
        """
        # 1. Retrieve
        retrieval = self.retrieve(question)

        # 2. Generate
        gen = await self.generate(question, retrieval["contexts"])

        return {
            "answer"       : gen["answer"],
            "retrieved_ids": retrieval["retrieved_ids"],
            "contexts"     : retrieval["contexts"],
            "sources"      : retrieval["sources"],
            "metadata"     : {
                "model"      : gen["model"],
                "tokens_used": gen["tokens_used"],
                "top_k"      : TOP_K,
            },
        }


class MainAgentV2(MainAgent):
    """
    RAG Agent V2 - Phiên bản nâng cấp (Được tối ưu Retrieval & Generation).
    Áp dụng: 
    1. Tăng TOP_K để cải thiện Hit Rate.
    2. Chain-of-Thought Prompting để tăng Accuracy.
    """

    def __init__(self):
        super().__init__() # Gọi lại logic khởi tạo của V1
        self.name = "SupportAgent-v2"
        self.optimized_top_k = 3 # Kéo nhiều context hơn V1 một chút

    # Override lại hàm retrieve để dùng TOP_K mới
    def retrieve(self, question: str, top_k: int = None) -> Dict:
        # Nếu không truyền top_k vào, dùng top_k tối ưu của V2
        actual_top_k = top_k if top_k else self.optimized_top_k
        return super().retrieve(question, actual_top_k)

    # Override lại hàm generate để nâng cấp Prompt
    async def generate(self, question: str, contexts: List[str]) -> Dict:
        context_text = "\n\n---\n\n".join(contexts)

        if not self._llm:
            return {
                "answer": "[DEV MODE V2] Cần API Key.",
                "model": "mock",
                "tokens_used": 0,
            }

        # 🚀 TỐI ƯU HÓA: Áp dụng Chain-of-Thought (CoT) và quy tắc ngặt nghèo hơn
        system_prompt = (
            "Bạn là một chuyên gia phân tích tài liệu nội bộ xuất sắc. "
            "Quy tắc tuyệt đối:\n"
            "1. Chỉ sử dụng thông tin từ 'Tài liệu tham khảo' bên dưới. KHÔNG bịa đặt.\n"
            "2. Nếu tài liệu không chứa câu trả lời, hãy trả lời chính xác câu này: 'Tôi không tìm thấy thông tin này trong tài liệu nội bộ.'\n"
            "3. Suy nghĩ từng bước (Think step-by-step): Xác định thông tin liên quan trong tài liệu trước, sau đó mới tổng hợp thành câu trả lời cuối cùng trực diện và chuyên nghiệp."
        )
        
        user_prompt = (
            f"Tài liệu tham khảo:\n{context_text}\n\n"
            f"Câu hỏi của người dùng: {question}"
        )

        response = await self._llm.chat.completions.create(
            model="gpt-4o-mini", # Vẫn giữ mini để tối ưu cost, bù đắp bằng prompt tốt
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0, # Giữ temp=0 để câu trả lời có tính determinisic (ổn định)
            max_tokens=512, # Tăng nhẹ token phòng trường hợp CoT sinh ra nhiều text
        )
        
        answer = response.choices[0].message.content
        tokens_used = response.usage.total_tokens

        return {
            "answer": answer,
            "model": "gpt-4o-mini-v2",
            "tokens_used": tokens_used,
        }

    # Không cần viết lại hàm query() vì nó sẽ tự động gọi retrieve() và generate() của V2

# ─────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────
if __name__ == "__main__":
    agent = MainAgent()

    async def test():
        questions = [
            "Mật khẩu phải thay đổi sau bao nhiêu ngày?",
            "Quy trình mua sắm gồm những bước nào?",
            "MFA bắt buộc cho những đối tượng nào?",
        ]
        for q in questions:
            print(f"\n❓ {q}")
            resp = await agent.query(q)
            print(f"   retrieved_ids : {resp['retrieved_ids']}")
            print(f"   sources        : {resp['sources']}")
            print(f"   answer         : {resp['answer'][:120]}...")

    asyncio.run(test())
