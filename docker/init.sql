-- Enables the pgvector extension so document_chunks.embedding can use the
-- native VECTOR type (see backend/app/models/types.py -> Vector). Runs
-- automatically on first container start via docker-entrypoint-initdb.d.
CREATE EXTENSION IF NOT EXISTS vector;
