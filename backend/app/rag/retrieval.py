"""
Retrieval layer. On Postgres this would use pgvector's <-> operator with an
IVFFlat/HNSW index for ANN search; on SQLite (dev/test) there's no native
vector index, so we fall back to an explicit in-Python cosine-similarity
scan over document_chunks. Functionally identical results at demo scale;
NOT how this should run in production at real corpus size (that's exactly
why Section 10 of the design doc recommends Postgres+pgvector for MVP).
"""
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.domain import Document, DocumentChunk
from app.rag.embeddings import cosine_similarity, get_embedding_provider


def retrieve(db: Session, query: str, top_k: int = 5, payer_id: str | None = None) -> list[dict]:
    provider = get_embedding_provider()
    query_vec = provider.embed(query)

    q = select(DocumentChunk, Document).join(Document, Document.id == DocumentChunk.document_id)
    if payer_id:
        q = q.where((Document.payer_id == payer_id) | (Document.payer_id.is_(None)))
    rows = db.execute(q).all()

    scored = []
    for chunk, doc in rows:
        if chunk.embedding is None:
            continue
        score = cosine_similarity(query_vec, chunk.embedding)
        scored.append((score, chunk, doc))

    scored.sort(key=lambda t: t[0], reverse=True)
    top = scored[:top_k]

    return [
        {
            "chunk_id": chunk.id,
            "document_id": doc.id,
            "document_title": doc.title,
            "similarity_score": round(float(score), 4),
            "text": chunk.chunk_text,
            "metadata": {"source_type": doc.source_type, "version": doc.version, "payer_id": doc.payer_id},
        }
        for score, chunk, doc in top
    ]
