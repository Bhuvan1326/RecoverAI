.PHONY: install seed train ingest test lint dev-backend dev-frontend docker-up docker-down

install:
	python3 -m venv venv
	./venv/bin/pip install -r requirements.txt

seed:
	PYTHONPATH=backend:. ./venv/bin/python scripts/seed_database.py

train:
	PYTHONPATH=backend:. ./venv/bin/python -m ml.training.train_denial_model

ingest:
	PYTHONPATH=backend:. ./venv/bin/python scripts/ingest_documents.py

test:
	PYTHONPATH=backend:. ./venv/bin/pytest backend/tests -v

lint:
	./venv/bin/ruff check backend ml rag scripts
	cd frontend && npm run lint

dev-backend:
	PYTHONPATH=backend:. ./venv/bin/uvicorn app.main:app --app-dir backend --reload --port 8000

dev-frontend:
	cd frontend && BACKEND_URL=http://localhost:8000 npm run dev

docker-up:
	docker compose up --build

docker-down:
	docker compose down -v
