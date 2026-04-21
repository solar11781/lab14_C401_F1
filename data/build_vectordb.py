# -*- coding: utf-8 -*-
"""
build_vectordb.py — Database Builder
================================================
Script này:
1. Đọc toàn bộ tài liệu trong data/docs/
2. Chunking bằng sliding window (chunk_size=300 ký tự, overlap=50)
3. Tạo embedding bằng OpenAI text-embedding-3-small
4. Lưu vào ChromaDB tại data/chroma_db/
5. Export doc_id_mapping.json để dùng cho ground_truth_ids

Chạy: python data/build_vectordb.py
"""

import sys
import os
import json
import re

# Fix UTF-8 output on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

# ─────────────────────────────────────────────
# Cấu hình
# ─────────────────────────────────────────────
DOCS_DIR    = Path(__file__).parent / "docs"
DB_DIR      = Path(__file__).parent / "chroma_db"
MAPPING_OUT = Path(__file__).parent / "doc_id_mapping.json"

CHUNK_SIZE  = 300   # ký tự
CHUNK_OVERLAP = 50
COLLECTION_NAME = "internal_docs"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")


# ─────────────────────────────────────────────
# 1. Text Chunker
# ─────────────────────────────────────────────
def split_into_chunks(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    """
    Chia văn bản thành các chunk bằng sliding window trên ký tự.
    Cố gắng cắt tại ranh giới dòng để giữ ngữ nghĩa.
    """
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(start + chunk_size, text_len)
        # Nếu chưa đến cuối, thử tìm ranh giới dòng để cắt gọn hơn
        if end < text_len:
            newline_pos = text.rfind("\n", start, end)
            if newline_pos > start + overlap:
                end = newline_pos + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        # Ensure start always advances by at least 1 to prevent infinite loop
        next_start = end - overlap
        if next_start <= start:
            next_start = start + 1
        start = next_start
        if start >= text_len:
            break
    return chunks


# ─────────────────────────────────────────────
# 2. Load tài liệu
# ─────────────────────────────────────────────
def load_documents(docs_dir: Path):
    """Đọc tất cả file .txt trong docs_dir, trả về list dict."""
    documents = []
    for filepath in sorted(docs_dir.glob("*.txt")):
        with open(filepath, encoding="utf-8") as f:
            content = f.read()
        # Trích source từ header nếu có
        source_match = re.search(r"^Source:\s*(.+)$", content, re.MULTILINE)
        source = source_match.group(1).strip() if source_match else filepath.name
        dept_match = re.search(r"^Department:\s*(.+)$", content, re.MULTILINE)
        department = dept_match.group(1).strip() if dept_match else "General"

        documents.append({
            "filename": filepath.name,
            "source": source,
            "department": department,
            "content": content,
        })
    print(f"[load_documents] Đã tải {len(documents)} tài liệu từ {docs_dir}")
    return documents


# ─────────────────────────────────────────────
# 3. Build ChromaDB
# ─────────────────────────────────────────────
def build_vectordb(documents):
    """
    Chunk từng tài liệu, tạo embedding và nạp vào ChromaDB.
    Trả về doc_id_mapping dict.
    """
    # Khởi tạo ChromaDB client
    client = chromadb.PersistentClient(path=str(DB_DIR))

    # Xóa collection cũ nếu tồn tại (re-build)
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"[build_vectordb] Đã xóa collection cũ '{COLLECTION_NAME}'")
    except Exception:
        pass

    # Embedding function
    if OPENAI_API_KEY:
        ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=OPENAI_API_KEY,
            model_name="text-embedding-3-small"
        )
        print("[build_vectordb] Sử dụng OpenAI text-embedding-3-small")
    else:
        # Fallback: dùng embedding mặc định của ChromaDB (không cần API key)
        ef = embedding_functions.DefaultEmbeddingFunction()
        print("[build_vectordb] CẢNH BÁO: Không có OPENAI_API_KEY → dùng default embedding (all-MiniLM-L6-v2)")

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"}
    )

    doc_id_mapping = {}
    chunk_counter = 0

    all_ids, all_docs, all_metas = [], [], []

    for doc in tqdm(documents, desc="Chunking & indexing"):
        chunks = split_into_chunks(doc["content"])
        for i, chunk_text in enumerate(chunks):
            chunk_counter += 1
            doc_id = f"chunk_{chunk_counter:04d}"   # e.g. "chunk_0001"

            doc_id_mapping[doc_id] = {
                "content"    : chunk_text,
                "source"     : doc["source"],
                "filename"   : doc["filename"],
                "department" : doc["department"],
                "chunk_index": i,
            }

            all_ids.append(doc_id)
            all_docs.append(chunk_text)
            all_metas.append({
                "source"    : doc["source"],
                "filename"  : doc["filename"],
                "department": doc["department"],
                "chunk_index": i,
            })

    # Nạp vào ChromaDB theo batch để tránh API rate limit
    BATCH = 50
    for start in range(0, len(all_ids), BATCH):
        collection.add(
            ids       = all_ids[start:start+BATCH],
            documents = all_docs[start:start+BATCH],
            metadatas = all_metas[start:start+BATCH],
        )

    print(f"[build_vectordb] ✅ Đã index {chunk_counter} chunks từ {len(documents)} tài liệu")
    return doc_id_mapping


# ─────────────────────────────────────────────
# 4. Export mapping
# ─────────────────────────────────────────────
def export_mapping(mapping: dict):
    with open(MAPPING_OUT, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)
    print(f"[export_mapping] ✅ Đã lưu {len(mapping)} entries → {MAPPING_OUT}")


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  BUILD VECTOR DB - Lab Day 14")
    print("=" * 55)
    docs    = load_documents(DOCS_DIR)
    mapping = build_vectordb(docs)
    export_mapping(mapping)
    print("\n[OK] Hoan tat!")
    print(f"   Vector DB  : {DB_DIR}")
    print(f"   ID Mapping : {MAPPING_OUT}")
    print("   Format doc_id: 'chunk_XXXX' (4 chu so, bat dau tu chunk_0001)")
