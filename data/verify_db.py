"""
verify_db.py — Database Verifier
==========================================
Script kiểm tra nhanh Vector DB sau khi build:
- Đếm số chunks đã index
- In 5 chunk sample với doc_id
- Thử query 3 câu hỏi mẫu, kiểm tra retrieved_ids hợp lý

Chạy: python data/verify_db.py
"""

import json
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
import os

load_dotenv()

DB_DIR          = Path(__file__).parent / "chroma_db"
MAPPING_FILE    = Path(__file__).parent / "doc_id_mapping.json"
COLLECTION_NAME = "internal_docs"
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY", "")


def get_collection():
    client = chromadb.PersistentClient(path=str(DB_DIR))
    if OPENAI_API_KEY:
        ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=OPENAI_API_KEY,
            model_name="text-embedding-3-small"
        )
    else:
        ef = embedding_functions.DefaultEmbeddingFunction()
    return client.get_collection(name=COLLECTION_NAME, embedding_function=ef)


def verify_mapping():
    """Kiểm tra doc_id_mapping.json."""
    print("\n── [1] Kiểm tra doc_id_mapping.json ──")
    if not MAPPING_FILE.exists():
        print("  ❌ Không tìm thấy doc_id_mapping.json! Hãy chạy build_vectordb.py trước.")
        return False

    with open(MAPPING_FILE, encoding="utf-8") as f:
        mapping = json.load(f)

    print(f"  ✅ Tổng số chunks trong mapping: {len(mapping)}")

    print("\n  📄 5 chunk mẫu:")
    for i, (doc_id, info) in enumerate(list(mapping.items())[:5]):
        preview = info["content"][:80].replace("\n", " ")
        print(f"  [{doc_id}] ({info['filename']}) → \"{preview}...\"")
    return True


def verify_chromadb_count(collection):
    """Kiểm tra số lượng document trong ChromaDB."""
    print("\n── [2] Kiểm tra ChromaDB collection ──")
    count = collection.count()
    print(f"  ✅ Số chunks trong ChromaDB: {count}")
    return count > 0


def verify_query(collection):
    """Thử query 3 câu hỏi mẫu và in kết quả retrieved_ids."""
    print("\n── [3] Test query (top 3 results) ──")
    test_questions = [
        "Yêu cầu độ dài mật khẩu tối thiểu là bao nhiêu?",
        "Quy trình xử lý purchase request như thế nào?",
        "Nhân viên mới cần làm gì trong ngày đầu tiên?",
    ]
    for q in test_questions:
        results = collection.query(
            query_texts=[q],
            n_results=3,
            include=["documents", "metadatas", "distances"]
        )
        ids       = results["ids"][0]
        distances = results["distances"][0]
        metas     = results["metadatas"][0]

        print(f"\n  ❓ \"{q}\"")
        for rank, (doc_id, dist, meta) in enumerate(zip(ids, distances, metas), 1):
            print(f"     #{rank} [{doc_id}] (score={1-dist:.3f}) ← {meta['filename']}")


def main():
    print("=" * 55)
    print("  VERIFY VECTOR DB — Lab Day 14")
    print("=" * 55)

    ok = verify_mapping()
    if not ok:
        return

    try:
        col = get_collection()
    except Exception as e:
        print(f"\n❌ Không thể kết nối ChromaDB: {e}")
        print("   → Hãy chạy: python data/build_vectordb.py")
        return

    if verify_chromadb_count(col):
        verify_query(col)
        print("\n✅ Tất cả kiểm tra đều PASS. DB sẵn sàng!")
    else:
        print("\n❌ ChromaDB rỗng. Hãy chạy lại build_vectordb.py.")


if __name__ == "__main__":
    main()
