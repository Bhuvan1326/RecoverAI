"""
Application configuration.

All settings are environment-driven (12-factor). Sensible local-dev defaults
are provided so the app runs out of the box with `docker compose up`, but
NOTHING here is meant to be a production secret. See .env.example.
"""
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

# Anchored to the repo root regardless of the importing process's cwd.
# Without this, a SQLite URL like "sqlite:///./recoverai.db" resolves
# relative to whatever directory a given process happened to be launched
# from -- which silently breaks multi-process local dev (e.g. the API
# server and a separately-launched `celery worker` process each computing
# a DIFFERENT actual file path and therefore seeing different, empty
# databases) even though `alembic upgrade head` and Base.metadata.create_all()
# both "succeeded". Postgres in Docker never has this problem since its URL
# is a real network address, not a relative filesystem path -- this only
# matters for the zero-setup local/SQLite quickstart.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_DEFAULT_SQLITE_URL = f"sqlite:///{_REPO_ROOT / 'recoverai.db'}"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    APP_NAME: str = "RecoverAI"
    ENVIRONMENT: Literal["local", "docker", "test", "production"] = "local"

    # Database. Defaults to a local SQLite file so the backend is runnable
    # with zero external services for development/testing. Docker Compose
    # overrides this via DATABASE_URL to point at Postgres+pgvector.
    DATABASE_URL: str = _DEFAULT_SQLITE_URL

    # Auth
    JWT_SECRET_KEY: str = "dev-only-secret-change-me-in-production"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 8

    # CORS
    CORS_ORIGINS: list[str] = ["http://localhost:3000"]

    # LLM / Embedding provider abstraction (Section 39 of spec).
    # "mock" works with zero external services / API keys so the whole
    # application remains functional without any paid API access.
    LLM_PROVIDER: Literal["mock", "anthropic", "openai"] = "mock"
    EMBEDDING_PROVIDER: Literal["mock", "openai"] = "mock"
    ANTHROPIC_API_KEY: str | None = None
    OPENAI_API_KEY: str | None = None

    # Rate limiting
    RATE_LIMIT_PER_MINUTE: int = 120

    # Model artifact directory
    MODEL_ARTIFACT_DIR: str = "ml/models/artifacts"

    # Anomaly detection (Feature A / Phase 2)
    ANOMALY_CONTAMINATION: float = 0.05
    ANOMALY_RANDOM_STATE: int = 42

    # Drift detection (Feature D / Phase 2). Thresholds are configurable
    # engineering defaults, not universally "correct" statistical cutoffs --
    # see docs/architecture for the reasoning.
    DRIFT_PSI_WARNING: float = 0.10
    DRIFT_PSI_CRITICAL: float = 0.25

    # Celery / Redis (Feature E / Phase 2)
    REDIS_URL: str = "redis://localhost:6379/0"
    CELERY_TASK_ALWAYS_EAGER: bool = False  # True in tests -- runs tasks synchronously, no broker needed

    # Celery Beat schedule (production periodic jobs). Hours are UTC,
    # 24-hour clock. Configurable so a deployment can move these off
    # peak-traffic windows without a code change.
    DRIFT_CHECK_HOUR_UTC: int = 2
    DRIFT_CHECK_MINUTE_UTC: int = 0
    DRIFT_CHECK_WINDOW: int = 500  # claims considered "current" in each scheduled drift check

    MODEL_RETRAIN_DAY_OF_WEEK: str = "sun"  # celery crontab day_of_week format
    MODEL_RETRAIN_HOUR_UTC: int = 3
    MODEL_RETRAIN_MINUTE_UTC: int = 0


@lru_cache
def get_settings() -> Settings:
    return Settings()
