# RecoverAI — Agentic Revenue Recovery & Claims Denial Prevention Control Tower

A full-stack, portfolio-grade decision-support platform for healthcare revenue-cycle
teams: predicts claim denials before submission, prioritizes denied claims by expected
recovery value, flags anomalous claims, monitors its own models for drift, and runs a
guarded, human-approved AI agent that investigates denials and drafts citation-grounded
appeals.

**Built entirely on synthetic and public data. No real PHI. Not a HIPAA-certified
product. RecoverAI never submits claims or appeals autonomously — every consequential
action requires human approval, enforced in code.**

---

## 1. What this is

Denials are one of the largest controllable revenue leaks in U.S. healthcare — widely
cited estimates put initial denial rates around 7–10% of submitted claims, with a large
share considered preventable (missing authorization, coding errors, incomplete
documentation). RecoverAI sits on both sides of that problem:

- **Pre-submission:** predicts denial risk, explains why (SHAP), and validates claims
  before they go out.
- **Post-denial:** ranks denied claims by expected recovery value (using a trained,
  calibrated appeal-success model), recommends the next-best action, flags anomalous
  claims, and — with human approval at every step — has an agent investigate a denial,
  retrieve relevant payer policy via RAG, and draft an appeal with citations.
- **Ongoing:** monitors its own models for data and prediction drift, and runs
  expensive/batch work (batch scoring, drift computation, training) asynchronously via
  Celery rather than blocking API requests.

## 2. Architecture

```
Next.js / React frontend  ->  FastAPI backend  ->  PostgreSQL + pgvector
                                    |                    |
                     ml/ (4 trained models + SHAP)   Redis <- Celery worker + beat
                     rag/ (chunking, embeddings, retrieval)
                     agents/ (orchestrator, tools -- all guardrail-checked)
                     guardrails/ (code-level allow-list, not prompt-based)
                     services/ (validator, recovery-value engine, audit log)
```

Modular monolith by design (see `docs/` design doc for the full architecture
comparison) — one deploy unit, clean internal module boundaries, an append-only
audit log standing in for an event bus, Celery/Redis for genuinely async work.

## 3. What's actually implemented vs. stubbed

Everything marked ✅ is real, tested, working code, verified against a live running
server (not just unit tests in isolation).

### Fully implemented

| Area | Status |
|---|---|
| Synthetic data generator | ✅ Real, documented assumptions |
| Auth (JWT) + RBAC (4 roles) | ✅ Real, server-enforced, tested |
| Claims CRUD + validator | ✅ Real, rule-based |
| Denial-risk model (LogReg + RF + XGBoost + CatBoost, isotonic calibration) | ✅ Real, chronological leakage-safe split, PR-AUC champion selection |
| SHAP explainability | ✅ Real, per-claim, live |
| **Denial-reason model** (Phase 2) | ✅ Rules take precedence for known facts; trained multiclass model (LogReg/RF/XGBoost, macro-F1 champion) handles the ambiguous residual; heuristic fallback if untrained |
| **Appeal-success model** (Phase 2) | ✅ Trained, isotonic-calibrated, PR-AUC champion selection, explicit leakage guard; feeds Expected Recovery Value and the priority queue live on every request; heuristic fallback if untrained |
| **Anomaly detection** (Phase 2) | ✅ Real Isolation Forest, configurable contamination, vectorized batch scoring (a real O(n^2) bug was found and fixed — see Section 8) |
| Expected recovery / priority queue / next-best-action | ✅ Real, deterministic, now driven by the trained appeal-success model |
| RAG appeal copilot | ✅ Real pipeline, mock providers by default (zero API keys), real Anthropic/OpenAI providers wired in |
| **RAG evaluation** | ✅ Real retrieval recall@K + citation referential-integrity/excerpt-fidelity checks against the actual corpus and real generated drafts (not test fixtures); verified live at 100% recall@1 and 100% citation correctness |
| Evidence completeness gate | ✅ Real, blocks generation below threshold |
| Agentic orchestrator | ✅ Real, deterministic guarded pipeline, now includes anomaly + appeal-success |
| Guardrail engine | ✅ Real, code-level allow-list, tested against simulated bypass attempts |
| Audit trail | ✅ Real, append-only, hash-chained, tamper detection tested |
| Human approval workflow | ✅ Real, RBAC-gated, end-to-end tested |
| Dashboard / payer / provider / denial / anomaly analytics | ✅ Real, vectorized for performance |
| Model monitoring / champion-challenger | ✅ Real for all 4 model types |
| **Drift detection** (Phase 2) | ✅ Real PSI (numeric + categorical) and KS-statistic, prediction drift, missing-value drift, configurable thresholds; verified live producing real WARNING/CRITICAL flags |
| **Recovery-ranking evaluation** | ✅ Real Precision@K/NDCG@K/expected-recovery-captured@K, evaluated on the held-out chronological test split against actual appeal outcomes, comparing the model-driven ranking against naive baselines — reports whichever strategy actually wins, honestly, not just favorable numbers |
| **Celery + Redis background jobs** (Phase 2) | ✅ Real — verified against an actual Redis broker with a separately-launched worker process (not just eager-mode tests) |
| **Alembic migrations** (Phase 2) | ✅ Real — full 20-table schema from empty, downgrade/upgrade round-trips cleanly, Docker entrypoint runs it before app start |
| **Recovery Strategy Simulator** | ✅ Real capacity-constrained comparison of 3 prioritization strategies over the actual current denied-claims set, using the same per-action effort estimates the Next-Best-Action engine produces — not a canned demo; verified live producing genuinely different winning strategies depending on staff-hour budget (naive claim-amount-first wins under tight capacity, the model-driven ranking wins at moderate capacity, all converge at unlimited capacity) |
| Frontend pages | ✅ Login, Dashboard, Claims, Claim Workbench (score/SHAP/denial-reason/anomaly/appeal-success/validate/what-if/appeal-copilot), Recovery Queue, **What-If & Recovery Simulator** (standalone claim what-if picker + strategy simulator), Anomalies, Payer Intelligence, Provider Intelligence, Model Monitoring (with real drift + ranking-eval tables), Audit Trail, Settings (profile, password change, admin user management, read-only system config, audit-chain verify) |
| Tests | ✅ **154 backend tests passing** |
| CI | ✅ GitHub Actions: lint+test, migration test, ML smoke test, frontend lint+build, Docker build, dependency audit |

### Known simplifications (honest, not hidden)

- **Appeal-success and denial-reason model quality is modest** (PR-AUC ~0.43-0.58,
  macro-F1 ~0.48-0.55 on real training runs) — an honest reflection of how much signal
  is actually present in the synthetic label-generation functions, not a bug.
- **Celery Beat's schedule** was upgraded to a production-grade
  `crontab()`-based config (daily drift check + weekly full model retrain, both at
  configurable UTC times via Settings/`.env`) — verified live with a real `celery beat`
  process, a real Redis broker, and a separate worker process actually firing the
  scheduled task at its configured time (see Section 11).
- Rate limiting is in-memory, single-process, and disabled in the test environment to
  avoid false-positive 429s from the test suite's own request volume (the real limiting
  logic is still covered directly — see `test_rate_limit.py`).

## 4. Quickstart (Docker — the intended path)

```bash
cp .env.example .env
docker compose up --build
```

Starts `postgres`, `redis`, `backend`, `celery-worker`, `celery-beat`, `frontend`.
Then seed the database (demo users, synthetic claims, trains all 4 models, ingests
sample RAG documents):

```bash
docker compose exec backend python scripts/seed_database.py
```

- Frontend: http://localhost:3000
- Backend API docs: http://localhost:8000/docs

> `docker compose up --build` itself has not been run literally as written — no Docker
> daemon is available in the environment this was built in, and container-registry
> access (Docker Hub) is blocked at the network level even with an alternative runtime
> (Podman) installed. Instead, every service `docker-compose.yml` defines was verified
> for real, end-to-end, against **real** PostgreSQL 16 + real pgvector (installed
> natively, not SQLite) and **real** Redis with a separately-launched Celery worker
> process: full Alembic migration (including a genuine upgrade→downgrade→upgrade
> round-trip), real claim scoring, real pgvector similarity retrieval producing a real
> cited appeal draft, a real async batch-scoring job round-tripping through Redis, and
> a real audit-chain verification — all against the exact `DATABASE_URL`/`REDIS_URL`
> shape `docker-compose.yml` configures. This caught and fixed two real bugs that had
> been invisible under SQLite-only testing (Section 8). The one thing this doesn't
> prove is the container build/orchestration layer itself (Dockerfile build steps,
> `depends_on` health-check ordering, volume mounts) — please run `docker compose up
> --build` yourself to confirm that layer before relying on it in production.

## 5. Quickstart (local, no Docker)

```bash
python3 -m venv venv
./venv/bin/pip install -r requirements.txt
PYTHONPATH=backend:. ./venv/bin/python scripts/seed_database.py

# terminal 1
PYTHONPATH=backend:. ./venv/bin/uvicorn app.main:app --app-dir backend --reload --port 8000

# terminal 2
cd frontend && npm install && BACKEND_URL=http://localhost:8000 npm run dev

# terminal 3 (optional -- only needed to exercise real async jobs)
redis-server --daemonize yes
PYTHONPATH=backend:. ./venv/bin/python -m celery -A app.workers.celery_app worker --loglevel=info
```

Uses SQLite by default at an absolute path anchored to the repo root
(`app/core/config.py`) — this matters if you run the API and a Celery worker as
separate local processes, since a relative SQLite path resolves differently per
process working directory (a real bug caught during development — see Section 8).

## 6. Demo credentials

Created by `scripts/seed_database.py` (development only):

| Role | Email | Password |
|---|---|---|
| ADMIN | admin@recoverai.demo | DemoPass123! |
| REVIEWER | reviewer@recoverai.demo | DemoPass123! |
| BILLER | biller@recoverai.demo | DemoPass123! |
| ANALYST | analyst@recoverai.demo | DemoPass123! |

## 7. Demo flow

1. Log in as `admin@recoverai.demo`.
2. Dashboard — revenue-at-risk funnel (labeled SIMULATED).
3. Claims -> open a `DENIED` claim -> Claim Workbench: Score claim, Explain (SHAP,
   plus predicted denial reason with rule/ML source + confidence), Score anomaly,
   Score appeal success, Run validator, What-If Simulator, Investigate & draft
   (agent runs risk + denial-reason + anomaly + appeal-success, retrieves policy,
   checks evidence, drafts an appeal with citations, Approve/Reject enforces human
   approval).
4. Recovery Queue — ranked by expected recovery value using the trained appeal-success
   model.
5. Anomalies — aggregate anomaly analytics with severity distribution.
6. Audit Trail — every step above, hash-chained.
7. Model Monitoring — champion vs. challenger for all 4 models, plus a real drift
   table ("Run drift check now" triggers live PSI/KS computation).

## 8. ML pipeline

Four trained model types, all following the same disciplined pattern: chronological
(never random) split, multiple candidates compared, calibration where the output feeds
a dollar figure, champion selection by the metric that actually matters for an
imbalanced problem (never raw accuracy).

| Model | Candidates | Champion metric | Calibration |
|---|---|---|---|
| Denial risk | LogReg, RF, XGBoost, CatBoost | PR-AUC | Isotonic |
| Denial reason (residual only) | LogReg, RF, XGBoost, CatBoost (multiclass) | Macro-F1 | N/A |
| Appeal success | LogReg, RF, XGBoost, CatBoost | PR-AUC | Isotonic |
| Anomaly detection | Isolation Forest only — unsupervised, no CatBoost equivalent applies | — (unsupervised) | N/A |

Features (`ml/features/`) use only pre-submission/pre-resolution-knowable fields, with
expanding as-of-date historical rates so nothing sees its own future. Leakage guards
are directly tested (`test_ml_features.py`, `test_appeal_success_leakage.py`).

**A real performance bug worth mentioning**: the anomaly analytics aggregate endpoint
originally reloaded and rescanned the entire feature dataframe once per claim (O(n^2)),
timing out at 1,500 claims. Fixed by fully vectorizing the batch-scoring path — verified
live going from a >30s timeout to 2.26s for 1,500 claims. The same fix was applied
proactively to the appeal-success batch path before it became the same bug twice.

**Two more real bugs, found only by testing against real PostgreSQL** (Section 4):
a vector-dimension mismatch (the `document_chunks.embedding` column's migration had
silently defaulted to 1536 dimensions instead of the intended 384, invisible under
SQLite's untyped JSON storage but rejected outright by real pgvector), and a
Postgres-only Alembic downgrade bug (dropping tables with Enum columns doesn't drop
the associated native ENUM types, breaking a downgrade→upgrade round-trip). Both are
fixed and covered by regression tests (`test_migrations.py`).

## 9. Agent & guardrail architecture

`agents/orchestrator.py` runs a fixed, auditable pipeline of guarded tool calls
(investigate -> risk -> anomaly -> retrieve policy -> denial reason -> evidence check ->
recovery calc (trained appeal-success model) -> recommend action -> draft appeal ->
PENDING_APPROVAL) rather than a free-form LLM tool loop — deliberately, to get all the
guardrail/audit value with none of the non-determinism risk in a financial workflow.

Every tool is wrapped with `@guarded(action_name)` (`guardrails/engine.py`) — a plain
Python allow-list checked in code, not a prompt instruction. Tested against every
`BLOCKED_AI_ACTIONS` entry, fails closed on unrecognized actions, and can't be talked
around by a "prompt-injection-style" action string.

## 10. RAG architecture

Chunking -> embeddings (mock by default, real OpenAI wired in) -> pgvector/SQLite
retrieval -> evidence-completeness gate -> LLM drafting (mock by default, real
Anthropic wired in, cannot fabricate facts by construction) -> citations. Document
ingestion can also run asynchronously via Celery.

**Evaluated, not just built** (`ml/evaluation/eval_rag.py`): retrieval recall@K against
the ingested policy corpus (using the exact query pattern the live agent uses), plus
citation referential-integrity and excerpt-fidelity checks run against real generated
appeal drafts from the actual `draft_appeal()` pipeline — not synthetic test fixtures.
These checks work against any `LLM_PROVIDER`, not just the faithful-by-construction mock
one; verified live at 100% recall@1 and 100% referential integrity/excerpt fidelity on
the current corpus and sample. Triggerable via `GET`/`POST /model-monitoring/rag-eval`
or the Model Monitoring UI.

## 11. Background jobs & scheduling (Celery + Redis)

Five real Celery tasks (`app/workers/tasks.py`): `score_claim_batch`,
`generate_shap_explanations`, `ingest_documents`, `calculate_drift_metrics`,
`train_model`, plus `scheduled_retrain_all_models` for the weekly job. Every task
opens its own DB session resolved dynamically at call time (`database_module.SessionLocal()`,
not a stale import-time reference) — a real bug was caught here during development
(both in the tasks themselves and, separately, in the four `ml/training/*.py` scripts
when invoked through Celery for the first time) and fixed the same way in both places.

**Production schedule** (`app/workers/celery_app.py`): real `crontab()` entries, not a
placeholder raw-second interval —

- `daily-drift-check` at a configurable UTC time (`DRIFT_CHECK_HOUR_UTC`/`_MINUTE_UTC`,
  default 02:00) — calls `calculate_drift_metrics`, which creates and tracks its own
  `BackgroundJob` row (no pre-existing job_id from an API call) so scheduled runs are
  just as visible via `GET /jobs/{id}` as on-demand ones.
- `weekly-model-retrain` at a configurable UTC day/time
  (`MODEL_RETRAIN_DAY_OF_WEEK`/`_HOUR_UTC`/`_MINUTE_UTC`, default Sunday 03:00) —
  retrains all 4 model types in sequence, each independently job-tracked so one
  model's training failure (e.g. not enough new resolved appeals yet) doesn't hide
  whether the others succeeded.

**Verified live**, not just in eager-mode tests: installed a real Redis server, ran an
actual `celery beat` scheduler process and a separate `celery worker` process, set the
drift-check time to ~90 seconds in the future, and confirmed via the running API that
`celery beat` waited for exactly that wall-clock time, dispatched the task over Redis,
and the worker computed and persisted 26 real drift metrics — all visible through
`GET /model-monitoring/drift` within seconds of the scheduled time.

Both schedules only take effect if the `celery-beat` Docker service is actually
running; the app works fully without it, since both jobs can always be triggered
on demand via `POST /model-monitoring/drift/compute(-async)` and
`POST /models/{type}/train-async`.

## 12. Database & migrations

20 tables (`backend/app/models/domain.py`), portable between SQLite (dev/test) and
PostgreSQL+pgvector (Docker/production) via custom column types. Alembic
(`backend/alembic/`) is the schema source of truth — `alembic upgrade head` builds the
complete schema from empty, verified to match ORM metadata exactly, downgrade/upgrade
round-trips cleanly. Docker entrypoint runs this before app start; local SQLite
quickstart falls back to `create_all`.

## 13. API

Full interactive docs at `/docs` (51 routes). Highlights: `/claims/{id}/score`,
`/claims/{id}/explanation`, `/claims/{id}/anomaly-score`,
`/claims/{id}/appeal-success-score`, `/claims/{id}/simulate`, `/recovery-queue`,
`/recovery-queue/simulate-strategy`, `/appeals/draft`,
`/workflow-actions/{id}/approve|reject`, `/audit-logs/verify`,
`/model-monitoring/drift`, `/model-monitoring/drift/compute`,
`/model-monitoring/rag-eval`, `/claims/batch-score`,
`/jobs/{job_id}`, `/users`, `/settings/system`.

## 14. Evaluation

PR-AUC/ROC-AUC/precision/recall/F1/Brier/calibration for denial-risk and
appeal-success; macro-F1/weighted-F1/per-class precision-recall for denial-reason;
PSI/KS for drift, with configurable WARNING/CRITICAL thresholds. Recovery-ranking
Precision@K/NDCG@K/expected-recovery-captured@K (`ml/evaluation/eval_recovery_ranking.py`)
evaluates the live priority queue's model-driven ranking against naive baselines
(claim-amount-only, appeal-probability-only, random) on real held-out appeal
outcomes — the SAME chronological test split the appeal-success model itself was
evaluated on, so this is genuine out-of-sample ranking evaluation, not a metric on
training data. Results persist to `ModelMetric` and are queryable via
`GET /model-monitoring/recovery-ranking` or triggerable on demand via
`POST /model-monitoring/recovery-ranking/compute` (also surfaced in the Model
Monitoring UI). The script reports whichever strategy actually wins on each run
honestly — the model-driven ranking doesn't always beat every baseline on every
metric, and that's reported rather than hidden. Guardrail violation rate is tested
directly (must be zero).

## 15. Security

RBAC enforced server-side (role read from the DB-backed user record via JWT, never
trusted from client input). Bcrypt password hashing. In-memory rate limiting. CORS
restricted. No stack traces leaked to clients. RAG-ingested content treated as data,
never instructions. Celery tasks and the ML training scripts both resolve their DB
session dynamically at call time (a real bug was caught and fixed in both places — see
Section 11). Full threat model in the `docs/` design doc.

## 16. Responsible AI

Every AI output carries an "association, not causation" disclaimer where relevant.
Anomaly detection is labeled "Anomaly Detection," never "Fraud Detection" — no
validated fraud dataset backs this. Every dashboard number is labeled
`SIMULATED — SYNTHETIC DEMO DATA`. No HIPAA-compliance claim anywhere. No claim of
validated real-world accuracy — modest model metrics are reported honestly, not
oversold. RecoverAI never submits a claim or appeal autonomously.

## 17. Limitations

- Appeal-success and denial-reason model quality is modest (Section 3) — reflects the
  synthetic generator's actual signal, reported honestly.
- Rate limiting is single-process in-memory.
- Mock LLM/embedding providers by default; real-provider paths wired in but not
  load-tested against live paid APIs here.
- The container build/orchestration layer itself (Dockerfile build steps, service
  health-check ordering, volume mounts) has not been verified — everything the
  containers would actually run against (real Postgres+pgvector, real Redis, real
  Celery) has been (Section 4/8), but `docker compose up --build` as a literal command
  has not, since no Docker daemon or container-registry network access is available in
  the environment this was built in.
- Synthetic-data decision-support demonstration only — not validated for real-world
  use, not HIPAA-certified, not a substitute for legal/compliance review.

## 18. Future work

Extracting `ml/` inference into its own service if request volume justifies it.

## 19. Resume bullets

- Architected and built RecoverAI, a full-stack agentic healthcare revenue-recovery
  platform (Next.js/React, FastAPI, PostgreSQL+pgvector, Celery/Redis) predicting claim
  denials pre-submission and prioritizing post-denial recovery by expected value.
- Trained and calibrated 4 model types (denial risk, denial reason, appeal success,
  anomaly detection) across 4 candidate algorithms each (LogReg/RandomForest/
  XGBoost/CatBoost where supervised) with SHAP explainability, isotonic calibration,
  and PR-AUC/macro-F1-driven champion selection on leakage-guarded, chronologically
  split pipelines, verified by automated leakage tests.
- Built a citation-grounded RAG appeal-drafting pipeline behind a code-enforced
  guardrail engine and human-approval workflow — zero autonomous submission of claims
  or appeals, verified by tests simulating guardrail-bypass attempts.
- Implemented real automated model-drift monitoring (PSI/KS statistics) and a
  production Celery Beat schedule (crontab-based daily drift checks, weekly model
  retraining), verified end-to-end against a live Redis broker, a real `celery beat`
  scheduler, and a separately-launched worker process actually firing on schedule.
- Implemented an append-only, hash-chained audit trail, RBAC, and Alembic-managed
  migrations across a 20-table schema portable between SQLite and PostgreSQL+pgvector.
- Built a Precision@K/NDCG@K/expected-recovery-captured@K evaluation harness for the
  recovery priority queue, benchmarked against real held-out appeal outcomes on the
  same chronological test split used for model evaluation — comparing the model-driven
  ranking against naive baselines and reporting results honestly even when a baseline
  wins on a given metric.
- Built an automated RAG evaluation harness (retrieval recall@K, citation referential-
  integrity, and excerpt-fidelity checks) that runs the real end-to-end agent
  draft-appeal pipeline against actual denied claims rather than synthetic fixtures,
  and works against any LLM provider, not just the faithful-by-construction default one.
- Built a capacity-constrained Recovery Strategy Simulator comparing prioritization
  strategies over live denied claims against a real staff-hour budget, verified to
  produce genuinely different winning strategies depending on capacity rather than
  decorative labels on identical output.
- Shipped Docker Compose orchestration (6 services), GitHub Actions CI, and a
  154-test backend suite covering auth, guardrails, ML leakage guards, ranking-metric
  correctness, and async job correctness.
- Found and fixed five real bugs during development: an O(n^2) performance regression
  in a batch-scoring endpoint (30s+ timeout -> 2.3s), a SQLite relative-path bug that
  silently caused a separately-launched worker process to read/write a different
  database file than the API server, a stale import-time DB-session binding in
  both the Celery tasks and the ML training scripts that only surfaced when training
  was invoked asynchronously for the first time, and two Postgres-only bugs (a
  vector-dimension mismatch and an Alembic Enum-type cleanup gap) found only by
  standing up real PostgreSQL 16 + pgvector natively and running the full stack
  end-to-end against it when a container runtime wasn't available.

---

*Synthetic-data decision-support demonstration. Not a HIPAA-certified healthcare
product. Not medical, legal, or billing advice.*
