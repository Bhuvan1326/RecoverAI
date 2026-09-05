"""
Read-only system configuration view for the Settings page (admin-only).

CRITICAL: this endpoint must NEVER return secret values (API keys, JWT
signing key, database/broker connection strings -- Postgres URLs in Docker
embed a password, Redis URLs can too). Only non-secret operational
configuration is returned, and provider API keys are surfaced as booleans
("is a key configured") rather than the key values themselves.
"""
from fastapi import APIRouter, Depends

from app.core.config import get_settings
from app.core.deps import require_roles
from app.models.domain import User, UserRole

router = APIRouter(prefix="/settings", tags=["settings"])


@router.get("/system")
def get_system_settings(_user: User = Depends(require_roles(UserRole.ADMIN))):
    settings = get_settings()
    return {
        "app_name": settings.APP_NAME,
        "environment": settings.ENVIRONMENT,
        "auth": {
            "access_token_expire_minutes": settings.ACCESS_TOKEN_EXPIRE_MINUTES,
        },
        "rate_limiting": {
            "requests_per_minute": settings.RATE_LIMIT_PER_MINUTE,
        },
        "rag": {
            "llm_provider": settings.LLM_PROVIDER,
            "embedding_provider": settings.EMBEDDING_PROVIDER,
            "anthropic_api_key_configured": bool(settings.ANTHROPIC_API_KEY),
            "openai_api_key_configured": bool(settings.OPENAI_API_KEY),
        },
        "anomaly_detection": {
            "contamination": settings.ANOMALY_CONTAMINATION,
            "random_state": settings.ANOMALY_RANDOM_STATE,
        },
        "drift_detection": {
            "psi_warning_threshold": settings.DRIFT_PSI_WARNING,
            "psi_critical_threshold": settings.DRIFT_PSI_CRITICAL,
        },
        "celery": {
            "task_always_eager": settings.CELERY_TASK_ALWAYS_EAGER,
            "daily_drift_check_utc": f"{settings.DRIFT_CHECK_HOUR_UTC:02d}:{settings.DRIFT_CHECK_MINUTE_UTC:02d}",
            "drift_check_window": settings.DRIFT_CHECK_WINDOW,
            "weekly_model_retrain_utc": f"{settings.MODEL_RETRAIN_DAY_OF_WEEK} {settings.MODEL_RETRAIN_HOUR_UTC:02d}:{settings.MODEL_RETRAIN_MINUTE_UTC:02d}",
        },
        "cors_origins": settings.CORS_ORIGINS,
    }
