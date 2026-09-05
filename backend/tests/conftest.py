import os
import sys
from pathlib import Path

os.environ.setdefault("DATABASE_URL", "sqlite:///./test_recoverai.db")
os.environ.setdefault("ENVIRONMENT", "test")
os.environ.setdefault("CELERY_TASK_ALWAYS_EAGER", "true")

BACKEND_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_ROOT))

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.database import Base, get_db
from app.main import app


@pytest.fixture(scope="function")
def db_engine(tmp_path):
    db_path = tmp_path / "test.db"
    engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    yield engine
    Base.metadata.drop_all(bind=engine)


@pytest.fixture(scope="function")
def client(db_engine, monkeypatch):
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=db_engine)

    def override_get_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db

    # Celery tasks (app/workers/tasks.py) intentionally do NOT go through
    # FastAPI's dependency injection -- they open their own SessionLocal()
    # directly, correctly mirroring how a real worker process (separate
    # from the API process) connects to the database in production. For
    # that same code to reach THIS test's isolated per-function database
    # instead of a stale/nonexistent module-level default, monkeypatch the
    # module-level SessionLocal that app.core.database exports (and that
    # app.workers.tasks imports) to this test's TestingSessionLocal too.
    import app.core.database as database_module

    monkeypatch.setattr(database_module, "SessionLocal", TestingSessionLocal)

    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture
def admin_token(client):
    client.post("/auth/register", json={"email": "admin@example.com", "full_name": "Admin", "password": "TestPass123!", "role": "ADMIN"})
    r = client.post("/auth/login", data={"username": "admin@example.com", "password": "TestPass123!"})
    return r.json()["access_token"]


@pytest.fixture
def analyst_token(client, admin_token):
    client.post(
        "/auth/register",
        headers={"Authorization": f"Bearer {admin_token}"},
        json={"email": "analyst@example.com", "full_name": "Analyst", "password": "TestPass123!", "role": "ANALYST"},
    )
    r = client.post("/auth/login", data={"username": "analyst@example.com", "password": "TestPass123!"})
    return r.json()["access_token"]
