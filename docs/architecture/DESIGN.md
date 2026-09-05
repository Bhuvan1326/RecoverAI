# RecoverAI — Agentic Revenue Recovery & Claims Denial Prevention Control Tower
### Full Project Design Document (Portfolio-Grade, Solo-Developer, 4–6 Week MVP)

> **Positioning statement:** RecoverAI is an agentic healthcare revenue-recovery decision-support platform that predicts claim denials, identifies revenue leakage, estimates expected recovery value, prioritizes recovery opportunities, recommends next-best actions, and provides a grounded RAG appeal copilot — while keeping every consequential action under human approval. Built entirely on synthetic/public data. Never HIPAA-certified, never autonomous on money-moving actions.

---

## 1. Executive Project Definition

**Problem:** Denials are one of the largest controllable leaks in U.S. healthcare finance. Industry surveys cited in the RCM literature put initial denial rates around 7–10% of submitted claims (translating to hundreds of millions of claims/year), with providers spending roughly $25 and 16 staff-minutes reworking each one, and up to ~90% of denials considered preventable by RCM researchers. One widely cited estimate puts total annual losses from denied claims in the hundreds of billions of dollars nationally.

**What RecoverAI does:** it sits pre-submission (predict/prevent) and post-denial (prioritize/recover), wrapping ML risk scoring, explainability, an expected-value recovery engine, and a guarded LLM agent into one control tower, with humans approving every action that touches money, a claim, or a payer.

**What RecoverAI is not:** a claims clearinghouse, a certified HIPAA system, a fraud-determination engine, or an autonomous appeal-submission bot. It is a decision-support layer.

**Why it's a strong portfolio project:** it forces you to demonstrate tabular ML + calibration, SHAP explainability, hybrid rule/ML classification, ranking/expected-value optimization, RAG with citation discipline, a genuinely *guarded* agent (not a toy chatbot), an audit-grade data model, and production engineering (Docker, CI, monitoring) — all in one coherent narrative a non-technical interviewer can follow in two sentences and a technical interviewer can interrogate for an hour.

---

## 2. Industry & RCM Research (Grounding, Not Gospel)

Key figures that recur across the research/industry literature (use these to *justify design decisions*, not as guaranteed real-world outcomes for your synthetic demo):

- Initial denial rates are frequently cited around 7% of submitted claims, translating to well over 200 million denied claims annually in the U.S., with providers spending real staff time and dollars reworking each one.
- One Information Systems Frontiers study estimates roughly one in seven claims is denied and that denials cost U.S. hospitals on the order of hundreds of billions of dollars a year, motivating "Responsible AI" pre-submission denial prediction as a Design Science Research artifact.
- RCM analysts commonly treat the large majority of denials as preventable through better upstream data quality (eligibility, auth, coding, documentation) rather than payer misbehavior alone.
- Multiple applied papers (Power BI + Random Forest on CMS/Medicare data; XGBoost + Random Forest + Isolation Forest on synthetic RCM data) validate exactly the modeling stack this project proposes: gradient-boosted trees for denial-risk scoring plus an isolation-forest layer for anomalies.
- Payer-side AI adoption (automated clinical review, algorithmic necessity screening) is outpacing provider-side tooling, which is the market gap RecoverAI's narrative leans on: providers need equivalent AI leverage, but constrained by human approval.

**Takeaway for the pitch:** "Payers already use ML to deny; providers largely don't yet use ML to defend revenue. RecoverAI closes that asymmetry — under human control."

---

## 3. Revenue Leakage Analysis Framework

Model leakage as a funnel, and make every dashboard number traceable to this funnel:

```
Total Billed Charges
   → Preventable Leakage      (denials caused by fixable pre-submission issues: auth, eligibility, coding, docs)
   → Non-Preventable Leakage  (medical necessity disputes, timely filing misses, true non-coverage)
   → Recoverable Revenue      (denied but appeal-eligible with positive expected value)
   → Expected Recovery        (Recoverable Revenue × P(appeal success) × P(payment) − processing cost)
   → Write-off / Unrecoverable
```

Every synthetic dashboard number must be labeled **[SIMULATED]** and traceable to this funnel so it never reads as a real-world benchmark claim.

---

## 4. Feature Architecture (Mapping 22 Features → System Layers)

| Layer | Features |
|---|---|
| **Prevention layer** (pre-submission) | 1 Denial Prediction, 2 Denial Reason Prediction, 3 SHAP Explainability, 8 Claim Validator, 14 What-If Simulator |
| **Recovery intelligence layer** (post-denial) | 4 Expected Recovery Value, 5 Priority Queue, 6 Next-Best-Action, 15 Appeal Success Prediction, 13 Strategy Simulator |
| **Analytics layer** | 7 Revenue-at-Risk Dashboard, 9 Payer Intelligence, 10 Provider Intelligence, 11 Anomaly Detection, 12 Root-Cause Analytics |
| **Agentic/RAG layer** | 16 Appeal Copilot, 17 Evidence Completeness, 18 Recovery Orchestrator, 19 Guardrail Engine |
| **Trust & ops layer** | 20 Audit Trail, 21 Model Monitoring/Drift, 22 Champion vs Challenger |

This mapping is the backbone of your repo's module boundaries (`ml/`, `rag/`, `agents/`, `workflows/`) and of the roadmap in Section 20.

---

## 5. Architecture Comparison

| Criterion | A. Modular Monolith | B. Microservices | C. Event-Driven |
|---|---|---|---|
| Solo-dev feasibility | High | Low | Medium |
| Operational overhead | Low (1 deploy unit) | High (N services, service mesh) | Medium (broker + consumers) |
| Cost | Low | High | Medium |
| Time-to-MVP | Fast | Slow | Medium |
| Scalability ceiling | Medium (vertical + read replicas) | High | High |
| Refactor path | Extract services later via module boundaries | N/A (already extracted) | N/A (already event-native) |
| Debuggability | High (single process, single log stream) | Low | Medium |
| Portfolio narrative | "I know when *not* to over-engineer" | Risk of looking like resume-driven design | Good if you *also* show the monolith path |

## 6. Recommended Architecture — Modular Monolith with Internal Event Bus

```
                        Next.js / React (frontend)
                                   │  REST (JSON)
                                   ▼
                        FastAPI application (single deploy)
        ┌───────────────┬───────────────┬────────────────┬───────────────┐
        │  api/          │  services/     │  agents/        │  workflows/    │
        │ (routers)      │ (business      │ (orchestrator,  │ (guardrails,   │
        │                │  logic)        │  tools)         │  approvals)    │
        └───────────────┴───────────────┴────────────────┴───────────────┘
                │                │                │                │
        ┌───────┴────────────────┴────────────────┴────────────────┴───────┐
        │                     Internal event bus (in-process pub/sub,       │
        │                     backed by Postgres outbox table)              │
        └──────────────────────────────┬──────────────────────────────────┘
                                        ▼
                     PostgreSQL (+ pgvector) ── Redis/Celery (async jobs:
                     scoring batches, SHAP generation, drift jobs, embeddings)
```

Rationale: a single FastAPI service with clean internal module boundaries (`ml/`, `rag/`, `agents/`, `guardrails/`, `workflows/`) gets you 90% of the architectural *talking points* of microservices (separation of concerns, testability, swappable model registry) at 10% of the operational cost. An **internal outbox-pattern event log** (a Postgres table you publish domain events into — `claim.scored`, `claim.flagged`, `human.reviewed`, etc.) gives you the audit trail *and* a real story for "how would you evolve this to Kafka/microservices," without needing to run Kafka on a laptop.

---

## 7. Data Strategy

**Sources, ranked by fit:**

1. **CMS DE-SynPUF (Data Entrepreneurs' Synthetic Public Use File)** — CMS's own synthetic Medicare claims (beneficiary, inpatient, outpatient, carrier, Part D), explicitly built for third-party app development with no real-PHI risk. CMS is explicit that DE-SynPUF has been imputed/suppressed/coarsened specifically so it has *no inferential research validity* about real Medicare populations — which is exactly the disclaimer RecoverAI should surface everywhere it touches this data. Available directly from CMS, via a GitHub relational mirror, and via an OMOP-CDM AWS Registry of Open Data build.
2. **Your own synthetic claim generator** (Section 3 of the original brief) for denial-specific fields DE-SynPUF doesn't have natively (denial reason, appeal outcome, authorization status) — DE-SynPUF gives you *realistic claim shape and cost distributions*; your generator layers denial/appeal semantics on top.
3. **Public payer policy text / CMS LCD-NCD summaries / synthetic internal templates** for the RAG corpus (Feature 16) — never scrape or store real payer proprietary manuals; use publicly published coverage-determination summaries and your own synthetic "payer policy" documents.

**Rules:**
- Chronological (not random) train/val/test splits — a claim's *submission date* determines its split, so no future payer behavior leaks into training.
- Denial-prediction features must be **knowable at submission time** (eligibility snapshot, auth status, coding, documentation completeness, payer/provider history *up to that date*). Anything only known post-adjudication (actual denial reason, payment amount, appeal outcome) is excluded from the pre-submission model and reserved as the *label* or for post-hoc models (denial-reason, appeal-success).
- Every synthetic table gets a `is_synthetic BOOLEAN DEFAULT TRUE` and a `data_source` column so provenance is queryable, not just documented in a README.

---

## 8. Database Design (PostgreSQL)

Core entities (abbreviated DDL — flesh out constraints/indexes per table when you build):

```sql
-- Reference entities
payers(id PK, name, payer_type, is_synthetic, created_at)
providers(id PK, npi, name, specialty, is_synthetic, created_at)

-- Claims
claims(
  id PK, claim_number, provider_id FK, payer_id FK, patient_ref,
  claim_amount NUMERIC, submission_date, status,
  eligibility_status, authorization_status, timely_filing_deadline,
  is_synthetic, data_source, created_at, updated_at
)
claim_lines(id PK, claim_id FK, procedure_code, diagnosis_code, modifiers, line_amount, units)

-- Outcomes
denial_events(id PK, claim_id FK, denial_reason_code, denial_date, denial_source, raw_reason_text)
appeal_events(id PK, denial_event_id FK, appeal_date, outcome, recovered_amount, decision_date)
recovery_outcomes(id PK, claim_id FK, expected_recovery, actual_recovery, closed_at)

-- ML / AI
model_versions(id PK, model_name, model_type, version_tag, trained_at, metrics_json, is_champion)
model_predictions(id PK, claim_id FK, model_version_id FK, prediction_type, score, explanation_json, created_at)
rag_retrievals(id PK, appeal_draft_id FK, document_chunk_id FK, similarity_score, rank)
documents(id PK, title, source_type, version, uploaded_at, is_synthetic)
document_chunks(id PK, document_id FK, chunk_text, embedding VECTOR(1536), chunk_index)

-- Workflow & governance
workflow_actions(id PK, claim_id FK, action_type, recommended_by, status, created_at)
human_reviews(id PK, workflow_action_id FK, reviewer, decision, decided_at, notes)
audit_logs(id PK, actor_type, actor_id, event_type, claim_id FK NULL, payload_json, prev_hash, hash, created_at)
```

Design notes:
- `audit_logs` is **append-only**; `prev_hash`/`hash` gives you a cheap hash-chain (tamper-evident, not blockchain — say this explicitly in interviews) so a modified row is detectable.
- `document_chunks.embedding` uses `pgvector`'s `vector` type with an IVFFlat or HNSW index for retrieval.
- Every table that can hold synthetic content carries `is_synthetic` + `data_source` for provenance.

---

## 9. Machine Learning System Design

### 9.1 Denial Prediction (Feature 1)
| Model | Interpretability | Training time | Calibration out-of-box | MVP fit |
|---|---|---|---|---|
| Logistic Regression | High | Fast | Good (native) | Strong baseline / champion challenger |
| Random Forest | Medium | Fast | Poor (needs Platt/Isotonic) | Good baseline |
| **XGBoost** | Medium (w/ SHAP) | Fast–Medium | Good with `binary:logistic` + calibration | **Recommended MVP champion** |
| LightGBM | Medium | Fastest on large data | Similar to XGB | Good challenger |
| CatBoost | Medium | Medium | Strong native handling of categoricals | Strong challenger, esp. payer/procedure categoricals |
| TabNet | Low | Slow, data-hungry | Needs work | Skip for MVP — dataset size and interview ROI don't justify it |

**Recommendation:** XGBoost as champion (best documented fit in the applied literature you're modeling this on — several papers explicitly pair XGBoost + Random Forest for this exact problem), CatBoost as challenger for categorical-heavy payer/procedure features, Logistic Regression as an always-on interpretable baseline for the Champion/Challenger dashboard (Feature 22). Calibrate the champion with isotonic or Platt scaling and report Brier score — denial *probabilities* feed directly into the Expected Recovery Value formula, so miscalibration corrupts every downstream dollar figure.

### 9.2 Denial Reason Prediction (Feature 2)
Hybrid: deterministic rule layer first (missing-auth flag, eligibility-fail flag, timely-filing breach — these are *known facts*, not predictions) → multiclass classifier (XGBoost/LightGBM multiclass, or simple multinomial logistic for MVP) only for the *ambiguous residual* (coding mismatch vs. medical necessity vs. documentation gaps, which genuinely require pattern learning). Report macro-F1 (protects minority denial classes) alongside weighted F1.

### 9.3 Appeal Success Prediction (Feature 15)
Only train a supervised model if your synthetic label-generation process is documented and consistent. Otherwise ship a transparent, documented heuristic/ranking score (e.g., weighted combination of documentation completeness, denial-reason appealability, payer historical appeal-overturn rate) explicitly labeled "heuristic baseline, not a validated predictive model." Never silently generate appeal outcomes and represent them as ground truth — say so in the UI copy itself.

### 9.4 Anomaly/Duplicate Detection (Feature 11)
Isolation Forest for MVP (fast, no distance-metric tuning, robust to mixed feature types after encoding) over Local Outlier Factor (sensitive to density parameter, slower at scale) or an autoencoder (overkill for tabular claims data at this scale, harder to explain to a reviewer). Label this "anomaly detection," never "fraud detection," in every surface — fraud is a legal determination.

### 9.5 Expected Recovery Value & Ranking (Feature 4/5)
```
ExpectedRecoveryValue = ClaimAmount × P(AppealSuccess) × P(PaymentRecovery) − ProcessingCost
PriorityScore = (ExpectedRecoveryValue × UrgencyFactor × RecoverabilityFactor) / EstimatedEffort
```
This is a ranking/optimization problem, not a classifier — evaluate with Precision@K / NDCG@K / "expected recovery captured @ K" rather than accuracy.

---

## 10. Agentic AI Architecture (Feature 18/19)

```
                         Recovery Agent (LLM-orchestrated)
                                     │
        ┌────────────────┬──────────┼──────────┬────────────────┐
        │  Claim Tools     │ Retrieval  │ Analytics  │  Explanation    │
        │ (read claim,     │ Tools (RAG │ Tools      │  Tools (SHAP,   │
        │  read history)   │  search)   │ (recovery  │  denial reason) │
        │                  │            │  value)    │                 │
        └────────────────┴──────────┴──────────┴────────────────┘
                                     │
                              Reasoning / planning
                                     │
                        ┌────────────────────────┐
                        │   Guardrail Engine       │  ← hard allow/deny table (Section 19 below)
                        └────────────────────────┘
                                     │
                        ┌────────────────────────┐
                        │   Human Approval Gate    │  ← every write/consequential action
                        └────────────────────────┘
                                     │
                          Approved action executes
                                     │
                            Append-only Audit Log
```

Design principle straight from the current HITL literature: guardrails should be an *explicit allow-list*, not a prompt-only instruction, because prompt-level restrictions are brittle at scale — the enforcement has to live in code the agent cannot argue its way around, with tool-level permission checks independent of the LLM's own output. UCHealth's public description of their approach ("agents can start in shadow mode, then *earn* autonomy only when proven accurate," with full audit logs, strict permissions, and an immediate kill switch) is a good real-world reference point for how to narrate this in an interview.

### 10.1 Guardrail Table (Feature 19)

| Action | AI can execute | Requires human |
|---|---|---|
| Analyze claim / retrieve history | ✅ | |
| Predict denial risk / reason | ✅ | |
| Generate SHAP explanation | ✅ | |
| Recommend correction / next-best-action | ✅ | |
| Retrieve RAG sources, draft appeal | ✅ | |
| Modify claim data | ❌ | ✅ |
| Submit claim | ❌ | ✅ |
| Submit appeal | ❌ | ✅ |
| Delete claim / record | ❌ | ✅ |
| Mark workflow "recovery attempt stopped" | ❌ | ✅ |

Enforce this as a **middleware layer** the agent's tool-calls pass through (a decorator or dependency-injected policy check on every tool function), not as agent-prompt text — this is the detail that makes the "guardrail engine" a real engineering artifact rather than a slide.

---

## 11. RAG Architecture (Feature 16/17)

**Vector store:** PostgreSQL + `pgvector` for MVP (co-located with your relational data, one fewer service to run, good enough recall at portfolio scale) over Chroma/Qdrant/Weaviate (better at massive scale, unnecessary operational surface for a solo dev demo).

```
Payer policy docs / denial-code refs / synthetic templates
   → Parser (pdf/html/txt) → Cleaner → Chunker (semantic, ~300–500 tokens, overlap)
   → Embedding model → pgvector store (metadata: payer, doc type, version, effective_date)
   → Retriever (top-k similarity) → Optional reranker
   → LLM (claim facts + retrieved chunks + evidence-completeness check as input)
   → Structured appeal draft with inline citations back to document_chunks.id
   → Citation validator (does every claim in the draft trace to a retrieved chunk?)
   → Human review queue
```

**Non-negotiables, backed by the current RAG-in-healthcare literature:** faithfulness (the draft is grounded *only* in retrieved context, not the model's parametric memory) and traceability (every claim in the appeal draft must be attributable to a specific retrieved chunk) are the two properties healthcare RAG evaluations converge on as necessary but still imperfectly solved — RAG reduces hallucination, it does not eliminate it, and systematic reviews of RAG in medicine explicitly flag that retrieval quality and evaluation methodology remain open problems. So: (1) never let the LLM answer without at least one retrieved chunk; (2) run a cheap post-hoc "does this sentence appear supported by any retrieved chunk" check (even a simple NLI/entailment check or a second LLM pass) before showing a draft to a human; (3) block appeal-draft generation entirely when the Evidence Completeness Checker (Feature 17) is below a threshold, and say why.

---

## 12. API Design (FastAPI) — Summary Table

| Endpoint | Purpose | Auth |
|---|---|---|
| `POST /claims` | Ingest a claim | JWT, role: biller+ |
| `GET /claims`, `GET /claims/{id}` | List/read | JWT |
| `POST /claims/{id}/score` | Run denial + reason models | JWT |
| `GET /claims/{id}/explanation` | SHAP breakdown | JWT |
| `POST /claims/{id}/validate` | Pre-submission validator | JWT |
| `POST /claims/{id}/simulate` | What-if simulator | JWT |
| `GET /recovery-queue`, `GET /recovery-queue/{id}` | Priority queue | JWT |
| `POST /appeals/predict` | Appeal success score | JWT |
| `POST /appeals/draft` | RAG appeal draft (agent) | JWT, triggers guardrail |
| `GET /appeals/{id}` | Read draft + citations | JWT |
| `POST /workflow-actions` | Agent proposes an action | System/agent |
| `POST /workflow-actions/{id}/approve` / `/reject` | Human decision | JWT, role: reviewer+ |
| `GET /dashboard/metrics`, `/payer-intelligence`, `/provider-intelligence`, `/denial-analytics` | Analytics reads | JWT |
| `GET /model-monitoring`, `/model-versions` | MLOps reads | JWT, role: admin |
| `GET /audit-logs` | Full trail | JWT, role: admin |

Every mutating endpoint validates input with Pydantic schemas, checks RBAC role via a dependency, and writes an `audit_logs` row inside the same DB transaction as the mutation (never as an afterthought/best-effort call) so audit and action can't drift apart.

---

## 13. Frontend Design (React/Next.js)

11 pages exactly as specified in the brief (Executive Dashboard, Claim Workbench, Claim Validator, Recovery Priority Queue, Appeal Copilot, Recovery Simulator, What-If Simulator, Payer Intelligence, Provider Intelligence, Model Monitoring, Audit Trail). Build order should mirror the roadmap (Section 20) — dashboard and workbench first (they're the demo backbone), Appeal Copilot and Model Monitoring last (they depend on RAG and on having models to compare).

---

## 14. Recovery Workflow (End-to-End)

```
Claim created (synthetic ingest)
   → Pre-submission validation (Feature 8) → readiness score
   → Denial risk scored (Feature 1) + explained (Feature 3)
   → [if submitted and denied] Denial event recorded → reason classified (Feature 2)
   → Expected recovery value computed (Feature 4) → queued by priority (Feature 5)
   → Next-best-action recommended (Feature 6)
   → Agent investigates: retrieves policy (RAG), checks evidence completeness (Feature 17)
   → Appeal draft generated (Feature 16) with citations
   → Human reviews draft + recommendation → approves or rejects (Feature 19 gate)
   → [only on approval] workflow_action marked approved → downstream system-of-record notified (out of scope: no real submission)
   → Outcome recorded when available → feeds appeal-success and denial models' next training run
   → Every step appended to audit_logs
```

---

## 15. Evaluation Strategy

| System | Metrics |
|---|---|
| Denial prediction | PR-AUC, ROC-AUC, Precision/Recall @ fixed threshold, Recall@K, calibration curve, Brier score |
| Denial reason (multiclass) | Macro-F1, weighted F1, per-class precision/recall, confusion matrix |
| Recovery ranking | Precision@K, NDCG@K, expected-recovery-captured@K |
| Anomaly detection | Manual-review agreement rate, flagged-rate stability over time (no ground truth fraud labels — say so) |
| RAG | Retrieval Recall@K, citation correctness (does cited chunk support the sentence?), faithfulness/groundedness rate, unsupported-claim rate |
| Agentic workflow | Correct tool-selection rate, correct next-best-action rate (vs. rule-based reference), guardrail-violation rate (should be 0), human-approval-compliance rate, workflow completion rate |

Never lead with accuracy alone on an imbalanced denial-prediction task — say this explicitly in your README and in interviews.

---

## 16. Business KPI Framework

Revenue at Risk · Preventable Revenue · Recoverable Revenue · Expected Recovery · Recovery Yield (Actual/Expected) · Clean-Claim Rate · Manual-Review Workload · Avg Resolution Time · Recovery per Staff Hour · Appeal Success Rate.

Rule: every KPI card in the UI carries a small "SIMULATED — synthetic demo data" badge, and the README has one paragraph explicitly stating these are not validated real-world savings figures.

---

## 17. Security Threat Model

| Threat | Scenario | Mitigation | Test |
|---|---|---|---|
| Prompt injection via RAG documents | A malicious/crafted "payer policy" doc instructs the LLM to ignore guardrails or fabricate approval | Treat retrieved content as data, never as instructions; guardrail checks live in code, not prompt; sanitize/strip instruction-like text from ingested docs | Adversarial doc injection test suite |
| Data poisoning | Corrupted synthetic training data skews denial model | Schema + range validation on ingest; drift monitoring catches distributional shifts | Unit tests on generator output bounds |
| Unauthorized claim/appeal modification | Compromised token attempts a write | RBAC on every mutating endpoint; guardrail table blocks AI-initiated writes entirely | Negative-path API tests per role |
| Broken access control | User escalates role via IDOR | Row-level checks + role checks server-side, never trust client-supplied role | Automated authz test matrix |
| Sensitive-data exposure | Logs/errors leak claim PII-shaped synthetic fields | Structured logging with field redaction; no PII fields exist since data is synthetic, but treat it as if real | Log-scrubbing tests |
| RAG data leakage across tenants (if multi-tenant later) | Retrieval crosses tenant boundary | Metadata filter on tenant_id enforced at query layer, not just app layer | Cross-tenant retrieval test |
| Audit-log tampering | Insider edits audit history | Append-only table, hash-chain, DB role with no UPDATE/DELETE grant on `audit_logs` | Attempt-to-mutate test expecting DB error |
| API abuse / scraping | Bulk extraction of claim data | Rate limiting, pagination caps, auth required everywhere | Load test with rate-limit assertions |
| Model manipulation (adversarial inputs) | Crafted claim features to force a low-risk score | Feature validation, monitoring for out-of-distribution inputs | Adversarial input battery |

---

## 18. Responsible AI

- Explicit "decision support, not decision" disclaimer on every AI-generated surface.
- SHAP explanations labeled as **associational**, not causal.
- LLM outputs never allowed to state a fact absent from retrieved context; evidence-completeness gate blocks generation below threshold.
- No claim that the system is HIPAA compliant anywhere in the repo, README, or UI — "HIPAA-aware architecture" only.
- RBAC + audit logging + encryption-in-transit (TLS) everywhere; encryption-at-rest documented as a deployment-time requirement (managed Postgres with disk encryption) rather than something a laptop demo can prove.
- Bias/fairness check: compare denial-risk score distributions across payer and provider segments in your synthetic data to confirm the model isn't systematically penalizing a segment purely due to synthetic generator artifacts — document what you checked and what you found either way.

---

## 19. DevOps Architecture

```
GitHub → PR → Lint (ruff/eslint) → Unit tests (pytest/vitest) → Integration tests (API)
      → ML tests (schema + prediction-shape smoke tests, not full retraining)
      → Security scan (pip-audit / npm audit, basic SAST)
      → Docker build → docker-compose smoke test → (manual) deploy
```
GitHub Actions workflow file per stage; fail fast on lint before burning CI minutes on tests.

---

## 20. 4–6 Week Implementation Roadmap

| Phase | Weeks | Deliverable | Demo |
|---|---|---|---|
| 1 — Foundation | 1 | Postgres + FastAPI + Next.js skeleton, DE-SynPUF ingestion, schema, basic dashboard | Load claims → dashboard → claim detail |
| 2 — Baseline ML | 1–1.5 | Feature engineering, LogReg/RF/XGBoost, PR-AUC/ROC-AUC/calibration, model persistence | Submit claim → denial probability |
| 3 — Explainability + Intelligence | 1 | SHAP, denial-reason classifier, root-cause analytics, payer/provider intelligence, validator, anomaly detection | Claim → risk → reasons → recommendations |
| 4 — Recovery Intelligence | 1 | Appeal-success heuristic/model, expected recovery value, priority queue, next-best-action, strategy + what-if simulators | 10K claims → AI prioritizes top recovery opportunities |
| 5 — Agentic AI + RAG | 1 | Document ingestion, pgvector, appeal copilot, evidence checker, orchestrator, guardrails, human approval, audit log | Denied claim → agent investigates → drafts appeal → human approves |
| 6 — Production Engineering | 0.5–1 | Docker/Compose, GitHub Actions, auth/RBAC, rate limiting, model monitoring, champion/challenger, docs | Full walkthrough demo |

If time is tight, MVP = Phases 1–4 + a thin slice of Phase 6 (Docker + one CI workflow). Phase 5 is the differentiator but is explicitly "Advanced" scope — don't let it block a working, demoable MVP.

---

## 21. Repository Structure

Use the structure exactly as specified in the brief (`frontend/`, `backend/app/{api,models,schemas,services,agents,workflows,rag,guardrails,core}`, `backend/tests/`, `ml/{data,preprocessing,features,models,training,evaluation,explainability,monitoring,notebooks}`, `rag/{ingestion,chunking,embeddings,retrieval,evaluation,prompts}`, `data/{raw,processed,synthetic,schemas}`, `docs/{architecture,research,api,security,ml,decisions}`, `scripts/`, `tests/`, `docker/`, `.github/workflows/`, plus `docker-compose.yml`, `README.md`, `requirements.txt`, `.env.example`). This maps 1:1 onto Section 4's layer table, which is worth stating explicitly in your README so reviewers see the architecture-to-code correspondence immediately.

---

## 22. MVP vs Advanced Scope

**MVP (weeks 1–4, ship this even if nothing else lands):** Denial prediction, denial reason prediction, SHAP, claim validator, expected recovery value, recovery-priority queue, next-best-action, revenue-at-risk dashboard, human approval, audit trail, Postgres, FastAPI, React/Next.js, XGBoost/CatBoost, Docker, one GitHub Actions workflow.

**Advanced (week 5–6, differentiators):** Appeal success prediction, RAG appeal copilot, evidence completeness, recovery strategy simulator, what-if simulator, payer/provider intelligence, anomaly detection, agentic orchestrator, guardrail engine, model monitoring, champion/challenger, synthetic data generator (can actually move earlier if you build it first to bootstrap everything else — recommend building the generator in Phase 1, not Phase 6, since every later phase depends on data).

**Do not** let RAG/agent polish come at the cost of the core denial-prediction pipeline reliably working end to end — the base model + dashboard is what a recruiter clicks through in the first 90 seconds.

---

## 23. Research References

*Labeled by source type. All links verified via web search on 2026-08-23; none invented.*

**Peer-reviewed / academic**
- Vatambeti et al., "Assessment of healthcare claims rejection risk using machine learning" — Random Forest + real-time Power BI denial risk on Medicare/CMS data. researchgate.net/publication/321897211
- "Responsible Artificial Intelligence in Healthcare: Predicting and Preventing Insurance Claim Denials for Economic and Social Wellbeing," *Information Systems Frontiers* — Design Science Research framework, 6 white/black-box algorithms, cites ~1-in-7 claims denied and ~$262B annual hospital losses. dl.acm.org/doi/10.1007/s10796-021-10137-5
- "Leveraging Predictive Analytics to Minimize Claim Denials in Healthcare RCM," *Journal of Technological Innovations* — logistic regression, decision trees, neural nets on real-world claims data. jtipublishing.com/jti/article/view/37
- "Deep Claim: Payer Response Prediction from Claims Data with Deep Learning" (arXiv) — pre-submission payer-response prediction system. arxiv.org/pdf/2007.06229
- "Exploiting Machine Learning Bias: Predicting Medical Denials" (AAAI Symposium) — survey/table of denial-prediction studies and bias considerations in high-dimensional, imbalanced claims data. ojs.aaai.org/index.php/AAAI-SS/article/download/31181/33341
- Survey of Explainable AI Techniques in Healthcare, PMC/NIH — categorization of SHAP/LIME/CAM methods for clinical ML. pmc.ncbi.nlm.nih.gov/articles/PMC9862413
- "Retrieval augmented generation for large language models in healthcare: A systematic review," *PLOS Digital Health* — methods/datasets/evaluation gaps for medical RAG. journals.plos.org/digitalhealth/article?id=10.1371/journal.pdig.0000877
- "Contradictions in Context: Challenges for RAG in Healthcare" (arXiv) — RAG faithfulness/contradiction risks specific to healthcare documents. arxiv.org/html/2511.06668v2

**Preprints**
- "Correctness is not Faithfulness in RAG Attributions" — distinguishes citation correctness from answer correctness, directly informs Feature 16/17 citation-validation design. staff.fnwi.uva.nl/m.derijke/.../wallat-2025-correctness.pdf
- "Attribution Techniques for Mitigating Hallucinated Information in RAG Systems: A Survey" (arXiv). arxiv.org/html/2601.19927v1
- "Medical Hallucinations in Foundation Models and Their Impact on Healthcare" (arXiv) — RAG's role and limits in clinical interpretability/control. arxiv.org/pdf/2503.05777
- International AI Safety Report 2026 (arXiv) — human-in-the-loop limits, automation bias, sandboxing for autonomous agents in high-stakes domains including healthcare. arxiv.org/pdf/2602.21012

**Government source**
- CMS, "Medicare Claims Synthetic Public Use Files (DE-SynPUF)" — official data, purpose statement, and disclosure-treatment caveats. cms.gov/data-research/statistics-trends-and-reports/medicare-claims-synthetic-public-use-files
- CMS DE-SynPUF User Manual / Codebook (PDF). cms.gov/Research-Statistics-Data-and-Systems/Downloadable-Public-Use-Files/SynPUFs/Downloads/SynPUF_DUG.pdf

**Industry / vendor / trade press** *(directional only — not validated research findings)*
- "AI-Driven Predictive Analytics for Reducing Healthcare Claim Denials and Administrative Waste in U.S. Health Systems" — applied XGBoost + Random Forest + Isolation Forest on synthetic RCM data, closest analog to RecoverAI's exact model stack. researchgate.net/publication/406312197
- Beckers Hospital Review, "Kill switches, guardrails: The raging debate over healthcare AI agents" — UCHealth's shadow-mode/earned-autonomy approach, useful real-world analog for Feature 19. beckershospitalreview.com/.../kill-switches-guardrails-the-raging-debate-over-healthcare-ai-agents
- Galileo AI, "How to Build Human-in-the-Loop Oversight for AI Agents" — HITL architecture patterns and regulatory context (EU AI Act Aug 2026 deadline). galileo.ai/blog/human-in-the-loop-agent-oversight
- Healthcare IT Today, "Human in the Loop Does Not Mean Safe" — critique that HITL labeling alone doesn't guarantee safety, useful counterpoint for your Responsible AI section. healthcareittoday.com/2026/07/29/human-in-the-loop-does-not-mean-safe-the-hidden-risks-of-agentic-ai-in-healthcare

Use the government and peer-reviewed sources for factual claims in your README; treat vendor blog posts as narrative color only, and never state their numbers as validated fact.

---

## 24. Resume Bullet Points

- Architected and built **RecoverAI**, a full-stack agentic healthcare revenue-recovery platform (React/Next.js, FastAPI, PostgreSQL+pgvector) predicting claim denials pre-submission and prioritizing post-denial recovery by expected value.
- Trained and calibrated gradient-boosted (XGBoost/CatBoost) denial-risk models with SHAP explainability and Brier-score/PR-AUC evaluation; built a hybrid rule+ML denial-reason classifier.
- Designed an expected-recovery-value ranking engine and priority queue, evaluated with Precision@K/NDCG@K, to sequence revenue-recovery work by dollar impact.
- Built a citation-grounded RAG appeal-drafting copilot (pgvector retrieval + evidence-completeness gating) with a hard-guardrail human-approval workflow — zero autonomous submission of claims or appeals.
- Implemented an append-only, hash-chained audit trail and RBAC across a 15-table PostgreSQL schema supporting full traceability of every AI prediction, agent action, and human decision.
- Shipped model monitoring with data/prediction drift detection and a champion-vs-challenger model comparison framework; containerized with Docker and CI'd with GitHub Actions.

## 25. Interview Explanation (60-Second Version)

"Denials cost U.S. providers a huge amount every year, and most of it is preventable — missing auth, bad coding, incomplete docs. RecoverAI predicts denial risk before submission, explains *why* with SHAP, and for claims that do get denied, it ranks them by expected recovery value — claim amount times appeal-success probability times recovery probability, minus processing cost — so staff work the highest-value claims first. There's an agent that can investigate a denial, pull the relevant payer policy with RAG, and draft an appeal with citations, but it can't submit anything — every consequential action needs a human click, and every prediction, retrieval, and approval is logged in an append-only audit trail. It's all built on CMS's own synthetic Medicare data plus a documented synthetic generator, so there's zero real PHI risk, and I was explicit everywhere about what's a validated prediction versus a labeled simulation."

## 26. Demo Script (5 Minutes)

1. **Dashboard (30s):** Revenue-at-risk funnel, labeled synthetic.
2. **Claim Workbench (60s):** Pick a high-risk claim, show SHAP breakdown, show validator warnings.
3. **What-If Simulator (45s):** Toggle "missing auth" → "present," watch risk drop live.
4. **Recovery Queue (45s):** Sort by expected recovery value, show priority tiers.
5. **Appeal Copilot (90s):** Denied claim → agent retrieves policy → evidence-completeness gate → draft with citations → human approve/reject buttons front and center.
6. **Audit Trail (30s):** Scroll the full event history for the claim just processed.
7. **Model Monitoring (30s):** Champion vs. challenger table, drift indicator.

## 27. Future Scalability Plan

- Extract `ml/` inference into a dedicated service once request volume justifies independent scaling (module boundaries already support this — Section 6).
- Move the internal Postgres-outbox event log to Kafka/Redpanda if you need true multi-consumer event fan-out.
- Swap pgvector for Qdrant/Weaviate if the RAG corpus grows past what a single Postgres instance comfortably indexes.
- Add real payer EDI (837/835) ingestion adapters behind the same `claims` schema, so the ingestion layer — not the ML/agent layer — absorbs real-world integration complexity.
- Formal fairness/bias audit and clinical-informatics review would be required before any real-data pilot; today's scope is explicitly synthetic-only.

---

*This document is a design artifact for a portfolio project. All financial/statistical figures cited in Section 2 and Section 23 come from third-party research/industry sources describing real-world RCM trends generally — they are not validated outcomes of this project, which uses only synthetic and public CMS data.*
