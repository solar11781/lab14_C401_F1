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
from chromadb.errors import NotFoundError
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

DB_DIR = Path(os.getenv("CHROMA_DB_PATH", str(Path(__file__).parent.parent / "data" / "chroma_db")))
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "internal_docs")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GENERATION_MODEL = os.getenv("GENERATION_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
TOP_K = int(os.getenv("TOP_K", "5"))


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
                model_name=EMBEDDING_MODEL
            )
        else:
            ef = embedding_functions.DefaultEmbeddingFunction()

        try:
            self._collection = client.get_collection(
                name=COLLECTION_NAME,
                embedding_function=ef
            )
        except NotFoundError:
            self._collection = None
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
        if collection is None:
            return {
                "retrieved_ids": [],
                "retrieved_context": [],
                "contexts": [],
                "sources": [],
            }

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
            "retrieved_context": documents,
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
            model=GENERATION_MODEL,
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
            "model"       : GENERATION_MODEL,
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
            "retrieved_context": retrieval["retrieved_context"],
            "contexts"     : retrieval["contexts"],
            "sources"      : retrieval["sources"],
            "metadata"     : {
                "model"      : gen["model"],
                "tokens_used": gen["tokens_used"],
                "top_k"      : TOP_K,
            },
        }


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
