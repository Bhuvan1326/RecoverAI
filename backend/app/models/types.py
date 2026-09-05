"""
Portable column types.

The app must run against SQLite (zero-dependency local/dev/test mode) AND
PostgreSQL+pgvector (Docker/production mode) without maintaining two schemas.
These TypeDecorators pick the right underlying representation at runtime.
"""
import json
import uuid

from sqlalchemy import JSON, String, TypeDecorator
from sqlalchemy.dialects.postgresql import UUID as PG_UUID


class GUID(TypeDecorator):
    """Platform-independent UUID: Postgres native UUID, SQLite CHAR(36)."""

    impl = String
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            return dialect.type_descriptor(PG_UUID(as_uuid=False))
        return dialect.type_descriptor(String(36))

    def process_bind_param(self, value, dialect):
        if value is None:
            return value
        return str(value)

    def process_result_value(self, value, dialect):
        if value is None:
            return value
        return str(value)


def new_uuid() -> str:
    return str(uuid.uuid4())


class Vector(TypeDecorator):
    """
    pgvector VECTOR on Postgres; JSON-encoded float list on SQLite.

    Retrieval on SQLite falls back to an in-Python cosine-similarity scan
    (see rag/retrieval) since SQLite has no native ANN index -- fine at
    portfolio/demo scale, explicitly NOT how this should run at real scale
    (that's what the Postgres+pgvector path with IVFFlat/HNSW is for).
    """

    impl = JSON
    cache_ok = True

    def __init__(self, dim: int = 1536, *args, **kwargs):
        self.dim = dim
        super().__init__(*args, **kwargs)

    def __repr__(self):
        # Without this, Alembic's autogenerate renders this TypeDecorator
        # as a bare `Vector()` (falling back to the class default dim of
        # 1536) instead of capturing the actual `dim` the column was
        # constructed with -- exactly the bug that caused the initial
        # migration to silently create a 1536-dim column for a field
        # declared Vector(384) in the ORM model. SQLite never caught this
        # (it stores embeddings as untyped JSON), so it was invisible until
        # tested against real pgvector, which enforces the declared
        # dimension. This repr is what Alembic actually calls when
        # rendering a column's type in a migration file.
        return f"Vector(dim={self.dim})"

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            from pgvector.sqlalchemy import Vector as PGVector

            return dialect.type_descriptor(PGVector(self.dim))
        return dialect.type_descriptor(JSON())

    def process_bind_param(self, value, dialect):
        if value is None:
            return value
        if dialect.name == "postgresql":
            return value
        return json.dumps(list(value))

    def process_result_value(self, value, dialect):
        if value is None:
            return value
        if dialect.name == "postgresql":
            return value
        if isinstance(value, str):
            return json.loads(value)
        return value
