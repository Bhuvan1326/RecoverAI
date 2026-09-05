import logging
import time
from collections import defaultdict

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api import analytics, appeals, audit, auth, claims, jobs, model_monitoring, recovery_queue, workflow
from app.api import settings as settings_api
from app.api import users
from app.core.config import get_settings
from app.core.database import Base, engine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("recoverai")

settings = get_settings()

app = FastAPI(
    title="RecoverAI API",
    description=(
        "Agentic Revenue Recovery & Claims Denial Prevention Control Tower. "
        "Decision-support only -- synthetic/public data, no PHI, no autonomous "
        "claim/appeal submission. See /docs for the full OpenAPI schema."
    ),
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Minimal in-memory rate limiter (Section 40) -----------------------
_request_log: dict[str, list[float]] = defaultdict(list)


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    if settings.ENVIRONMENT == "test":
        # Rate limiting is real and enforced in every other environment;
        # disabled only for the automated test suite, which legitimately
        # fires hundreds of requests in tens of seconds against a single
        # shared TestClient "IP" -- that's a test-harness artifact, not the
        # abuse pattern this middleware exists to catch. See
        # test_rate_limit_middleware.py for a dedicated test that exercises
        # the real limiter logic directly (bypassing this early return).
        return await call_next(request)

    client_ip = request.client.host if request.client else "unknown"
    now = time.time()
    window = _request_log[client_ip]
    window[:] = [t for t in window if now - t < 60]
    if len(window) >= settings.RATE_LIMIT_PER_MINUTE:
        return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded. Try again shortly."})
    window.append(now)

    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    return response


# --- Error handling (Section 33): never leak stack traces --------------
@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(status_code=422, content={"detail": "Invalid request", "errors": exc.errors()})


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


@app.on_event("startup")
def on_startup():
    # Docker path: docker/entrypoint.sh runs `alembic upgrade head` before the
    # app process even starts (Section F3/F4) -- that's the source of truth.
    # For the zero-setup local/SQLite quickstart (README Section 5, no Docker,
    # no separate migration step), fall back to create_all so `uvicorn ...`
    # alone still works. This never runs against Postgres in the Docker path.
    if settings.DATABASE_URL.startswith("sqlite") and settings.ENVIRONMENT != "test":
        Base.metadata.create_all(bind=engine)


@app.get("/health")
def health():
    return {"status": "ok", "app": settings.APP_NAME, "environment": settings.ENVIRONMENT}


app.include_router(auth.router)
app.include_router(claims.router)
app.include_router(recovery_queue.router)
app.include_router(appeals.router)
app.include_router(workflow.router)
app.include_router(analytics.router)
app.include_router(model_monitoring.router)
app.include_router(audit.router)
app.include_router(jobs.router)
app.include_router(users.router)
app.include_router(settings_api.router)
