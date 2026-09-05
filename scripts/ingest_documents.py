#!/usr/bin/env python3
"""
RAG ingestion (Section 38). Loads sample synthetic payer policy documents,
chunks them, embeds them (via the configured EmbeddingProvider), and stores
them in document_chunks so the Appeal Copilot's retrieval step has a real
corpus to search from the moment the app starts.

Usage:
    python scripts/ingest_documents.py
    python scripts/ingest_documents.py --file path/to/other.json
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from app.core.database import Base, SessionLocal, engine
from app.models.domain import Document, DocumentChunk
from app.rag.chunking import chunk_text
from app.rag.embeddings import get_embedding_provider

DEFAULT_FILE = Path(__file__).resolve().parent.parent / "data" / "synthetic" / "sample_payer_policies.json"


def ingest(file_path: Path):
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    provider = get_embedding_provider()
    try:
        docs = json.loads(file_path.read_text())
        total_chunks = 0
        for d in docs:
            document = Document(title=d["title"], source_type=d.get("source_type", "payer_policy"), version=d.get("version", "1.0"))
            db.add(document)
            db.flush()

            chunks = chunk_text(d["text"])
            for i, chunk in enumerate(chunks):
                embedding = provider.embed(chunk)
                db.add(DocumentChunk(document_id=document.id, chunk_index=i, chunk_text=chunk, embedding=embedding))
                total_chunks += 1

        db.commit()
        print(f"Ingested {len(docs)} documents / {total_chunks} chunks from {file_path}")
    finally:
        db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default=str(DEFAULT_FILE))
    args = parser.parse_args()
    ingest(Path(args.file))
