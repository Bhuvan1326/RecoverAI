"""
ORM models for RecoverAI.

Every synthetic-capable table carries is_synthetic + data_source for
provenance (Section 6/7 of the spec). audit_logs is intentionally
append-only at the application layer (see services/audit.py) and hash-chained.
"""
from datetime import datetime, timezone
from enum import Enum as PyEnum

from sqlalchemy import (
    Boolean,
    DateTime,
    Enum,
    ForeignKey,
    Integer,
    JSON,
    Numeric,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base
from app.models.types import GUID, Vector, new_uuid


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


# --------------------------------------------------------------------------
# Auth / RBAC
# --------------------------------------------------------------------------
class UserRole(str, PyEnum):
    ADMIN = "ADMIN"
    REVIEWER = "REVIEWER"
    BILLER = "BILLER"
    ANALYST = "ANALYST"


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(GUID, primary_key=True, default=new_uuid)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    full_name: Mapped[str] = mapped_column(String(255), nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[UserRole] = mapped_column(Enum(UserRole), default=UserRole.ANALYST, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


# --------------------------------------------------------------------------
# Reference entities
# --------------------------------------------------------------------------
class Payer(Base):
    __tablename__ = "payers"

    id: Mapped[str] = mapped_column(GUID, primary_key=True, default=new_uuid)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    payer_type: Mapped[str] = mapped_column(String(50), default="commercial")
    is_synthetic: Mapped[bool] = mapped_column(Boolean, default=True)
    data_source: Mapped[str] = mapped_column(String(100), default="recoverai_synthetic_generator")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Provider(Base):
    __tablename__ = "providers"

    id: Mapped[str] = mapped_column(GUID, primary_key=True, default=new_uuid)
    npi: Mapped[str] = mapped_column(String(20), unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    specialty: Mapped[str] = mapped_column(String(100), default="general")
    is_synthetic: Mapped[bool] = mapped_column(Boolean, default=True)
    data_source: Mapped[str] = mapped_column(String(100), default="recoverai_synthetic_generator")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


# --------------------------------------------------------------------------
# Claims
# --------------------------------------------------------------------------
class ClaimStatus(str, PyEnum):
    DRAFT = "DRAFT"
    SUBMITTED = "SUBMITTED"
    PAID = "PAID"
    DENIED = "DENIED"
    APPEALED = "APPEALED"
    RECOVERED = "RECOVERED"
    CLOSED = "CLOSED"


class Claim(Base):
    __tablename__ = "claims"

    id: Mapped[str] = mapped_column(GUID, primary_key=True, default=new_uuid)
    claim_number: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)
    provider_id: Mapped[str] = mapped_column(GUID, ForeignKey("providers.id"), nullable=False)
    payer_id: Mapped[str] = mapped_column(GUID, ForeignKey("payers.id"), nullable=False)
    patient_ref: Mapped[str] = mapped_column(String(50), nullable=False)  # synthetic pseudonymous ref, never real PHI

    claim_amount: Mapped[float] = mapped_column(Numeric(12, 2), nullable=False)
    claim_type: Mapped[str] = mapped_column(String(30), default="professional")
    place_of_service: Mapped[str] = mapped_column(String(10), default="11")

    status: Mapped[ClaimStatus] = mapped_column(Enum(ClaimStatus), default=ClaimStatus.DRAFT)

    eligibility_status: Mapped[str] = mapped_column(String(20), default="VERIFIED")
    authorization_status: Mapped[str] = mapped_column(String(20), default="PRESENT")
    documentation_completeness: Mapped[float] = mapped_column(Numeric(5, 2), default=100.0)

    service_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    submission_date: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    timely_filing_deadline: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    is_synthetic: Mapped[bool] = mapped_column(Boolean, default=True)
    data_source: Mapped[str] = mapped_column(String(100), default="recoverai_synthetic_generator")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    lines: Mapped[list["ClaimLine"]] = relationship(back_populates="claim", cascade="all, delete-orphan")


class ClaimLine(Base):
    __tablename__ = "claim_lines"

    id: Mapped[str] = mapped_column(GUID, primary_key=True, default=new_uuid)
    claim_id: Mapped[str] = mapped_column(GUID, ForeignKey("claims.id"), nullable=False)
    procedure_code: Mapped[str] = mapped_column(String(10), nullable=False)
    diagnosis_code: Mapped[str] = mapped_column(String(10), nullable=False)
    modifiers: Mapped[str] = mapped_column(String(50), default="")
    line_amount: Mapped[float] = mapped_column(Numeric(12, 2), nullable=False)
    units: Mapped[int] = mapped_column(Integer, default=1)

    claim: Mapped["Claim"] = relationship(back_populates="lines")


# --------------------------------------------------------------------------
# Outcomes
# --------------------------------------------------------------------------
class DenialEvent(Base):
    __tablename__ = "denial_events"

    id: Mapped[str] = mapped_column(GUID, primary_key=True, default=new_uuid)
    claim_id: Mapped[str] = mapped_column(GUID, ForeignKey("claims.id"), nullable=False)
    denial_reason_code: Mapped[str] = mapped_column(String(50), nullable=False)
    denial_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    raw_reason_text: Mapped[str] = mapped_column(Text, default="")
    is_synthetic: Mapped[bool] = mapped_column(Boolean, default=True)


class AppealEvent(Base):
    __tablename__ = "appeal_events"

    id: Mapped[str] = mapped_column(GUID, primary_key=True, default=new_uuid)
    denial_event_id: Mapped[str] = mapped_column(GUID, ForeignKey("denial_events.id"), nullable=False)
    appeal_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    outcome: Mapped[str] = mapped_column(String(20), default="PENDING")  # PENDING/WON/LOST/PARTIAL
    recovered_amount: Mapped[float] = mapped_column(Numeric(12, 2), default=0)
    decision_date: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    is_synthetic: Mapped[bool] = mapped_column(Boolean, default=True)


class RecoveryOutcome(Base):
    __tablename__ = "recovery_outcomes"

    id: Mapped[str] = mapped_column(GUID, primary_key=True, default=new_uuid)
    claim_id: Mapped[str] = mapped_column(GUID, ForeignKey("claims.id"), nullable=False)
    expected_recovery: Mapped[float] = mapped_column(Numeric(12, 2), nullable=False)
    actual_recovery: Mapped[float | None] = mapped_column(Numeric(12, 2), nullable=True)
    closed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


# --------------------------------------------------------------------------
# ML
# --------------------------------------------------------------------------
class ModelVersion(Base):
    __tablename__ = "model_versions"

    id: Mapped[str] = mapped_column(GUID, primary_key=True, default=new_uuid)
    model_name: Mapped[str] = mapped_column(String(100), nullable=False)
    model_type: Mapped[str] = mapped_column(String(50), nullable=False)  # denial_risk / denial_reason / appeal_success / anomaly_detection
    version_tag: Mapped[str] = mapped_column(String(50), nullable=False)
    artifact_path: Mapped[str] = mapped_column(String(255), nullable=False)
    metrics: Mapped[dict] = mapped_column(JSON, default=dict)
    feature_names: Mapped[list] = mapped_column(JSON, default=list)
    params: Mapped[dict] = mapped_column(JSON, default=dict)
    # Model registry / experiment-tracking fields (Phase 2, Section 12/14):
    dataset_version: Mapped[str | None] = mapped_column(String(50), nullable=True)
    feature_schema_version: Mapped[str] = mapped_column(String(20), default="1.0")
    calibration_status: Mapped[str] = mapped_column(String(20), default="none")  # none / isotonic / platt
    is_champion: Mapped[bool] = mapped_column(Boolean, default=False)
    trained_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    __table_args__ = (UniqueConstraint("model_type", "version_tag", name="uq_model_type_version"),)


class ModelPrediction(Base):
    __tablename__ = "model_predictions"

    id: Mapped[str] = mapped_column(GUID, primary_key=True, default=new_uuid)
    claim_id: Mapped[str] = mapped_column(GUID, ForeignKey("claims.id"), nullable=False)
    model_version_id: Mapped[str] = mapped_column(GUID, ForeignKey("model_versions.id"), nullable=False)
    prediction_type: Mapped[str] = mapped_column(String(50), nullable=False)
    score: Mapped[float] = mapped_column(Numeric(6, 5), nullable=False)
    label: Mapped[str | None] = mapped_column(String(50), nullable=True)
    explanation: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class ModelMetric(Base):
    __tablename__ = "model_metrics"

    id: Mapped[str] = mapped_column(GUID, primary_key=True, default=new_uuid)
    model_version_id: Mapped[str] = mapped_column(GUID, ForeignKey("model_versions.id"), nullable=False)
    metric_name: Mapped[str] = mapped_column(String(50), nullable=False)
    metric_value: Mapped[float] = mapped_column(Numeric(10, 6), nullable=False)
    recorded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class DriftMetric(Base):
    __tablename__ = "drift_metrics"

    id: Mapped[str] = mapped_column(GUID, primary_key=True, default=new_uuid)
    model_version_id: Mapped[str | None] = mapped_column(GUID, ForeignKey("model_versions.id"), nullable=True)
    metric_type: Mapped[str] = mapped_column(String(30), default="data_drift")  # data_drift / prediction_drift / missing_value
    metric_name: Mapped[str] = mapped_column(String(100), nullable=False)  # e.g. PSI, KS, avg_denial_probability
    feature_name: Mapped[str | None] = mapped_column(String(100), nullable=True)
    baseline_value: Mapped[float] = mapped_column(Numeric(12, 6), nullable=False)
    current_value: Mapped[float] = mapped_column(Numeric(12, 6), nullable=False)
    drift_score: Mapped[float] = mapped_column(Numeric(12, 6), nullable=False)
    status: Mapped[str] = mapped_column(String(20), default="NORMAL")  # NORMAL / WARNING / CRITICAL
    is_drift_detected: Mapped[bool] = mapped_column(Boolean, default=False)
    reference_period: Mapped[str] = mapped_column(String(100), default="")
    current_period: Mapped[str] = mapped_column(String(100), default="")
    recorded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


# --------------------------------------------------------------------------
# RAG
# --------------------------------------------------------------------------
class Document(Base):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(GUID, primary_key=True, default=new_uuid)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    source_type: Mapped[str] = mapped_column(String(50), default="payer_policy")
    payer_id: Mapped[str | None] = mapped_column(GUID, ForeignKey("payers.id"), nullable=True)
    version: Mapped[str] = mapped_column(String(20), default="1.0")
    effective_date: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    is_synthetic: Mapped[bool] = mapped_column(Boolean, default=True)
    uploaded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id: Mapped[str] = mapped_column(GUID, primary_key=True, default=new_uuid)
    document_id: Mapped[str] = mapped_column(GUID, ForeignKey("documents.id"), nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    chunk_text: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(384), nullable=True)


class RagRetrieval(Base):
    __tablename__ = "rag_retrievals"

    id: Mapped[str] = mapped_column(GUID, primary_key=True, default=new_uuid)
    appeal_draft_id: Mapped[str] = mapped_column(GUID, ForeignKey("workflow_actions.id"), nullable=False)
    document_chunk_id: Mapped[str] = mapped_column(GUID, ForeignKey("document_chunks.id"), nullable=False)
    similarity_score: Mapped[float] = mapped_column(Numeric(6, 5), nullable=False)
    rank: Mapped[int] = mapped_column(Integer, nullable=False)


# --------------------------------------------------------------------------
# Workflow / governance
# --------------------------------------------------------------------------
class WorkflowActionType(str, PyEnum):
    VERIFY_ELIGIBILITY = "VERIFY_ELIGIBILITY"
    OBTAIN_AUTHORIZATION = "OBTAIN_AUTHORIZATION"
    REQUEST_DOCUMENTATION = "REQUEST_DOCUMENTATION"
    CORRECT_CODING = "CORRECT_CODING"
    RESUBMIT = "RESUBMIT"
    APPEAL = "APPEAL"
    CONTACT_PAYER = "CONTACT_PAYER"
    ESCALATE = "ESCALATE"
    STOP_RECOVERY = "STOP_RECOVERY"
    DRAFT_APPEAL = "DRAFT_APPEAL"


class WorkflowActionStatus(str, PyEnum):
    PENDING_APPROVAL = "PENDING_APPROVAL"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    EXECUTED = "EXECUTED"


class WorkflowAction(Base):
    __tablename__ = "workflow_actions"

    id: Mapped[str] = mapped_column(GUID, primary_key=True, default=new_uuid)
    claim_id: Mapped[str] = mapped_column(GUID, ForeignKey("claims.id"), nullable=False)
    action_type: Mapped[WorkflowActionType] = mapped_column(Enum(WorkflowActionType), nullable=False)
    recommended_by: Mapped[str] = mapped_column(String(50), default="agent")
    payload: Mapped[dict] = mapped_column(JSON, default=dict)  # e.g. appeal draft text + citations
    status: Mapped[WorkflowActionStatus] = mapped_column(
        Enum(WorkflowActionStatus), default=WorkflowActionStatus.PENDING_APPROVAL
    )
    created_by: Mapped[str | None] = mapped_column(GUID, ForeignKey("users.id"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class HumanReview(Base):
    __tablename__ = "human_reviews"

    id: Mapped[str] = mapped_column(GUID, primary_key=True, default=new_uuid)
    workflow_action_id: Mapped[str] = mapped_column(GUID, ForeignKey("workflow_actions.id"), nullable=False)
    reviewer_id: Mapped[str] = mapped_column(GUID, ForeignKey("users.id"), nullable=False)
    decision: Mapped[str] = mapped_column(String(20), nullable=False)  # APPROVED / REJECTED
    notes: Mapped[str] = mapped_column(Text, default="")
    decided_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


# --------------------------------------------------------------------------
# Audit (append-only, hash-chained)
# --------------------------------------------------------------------------
class AuditLog(Base):
    __tablename__ = "audit_logs"

    id: Mapped[str] = mapped_column(GUID, primary_key=True, default=new_uuid)
    actor_type: Mapped[str] = mapped_column(String(20), nullable=False)  # user / agent / system
    actor_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
    event_type: Mapped[str] = mapped_column(String(100), nullable=False)
    claim_id: Mapped[str | None] = mapped_column(GUID, ForeignKey("claims.id"), nullable=True)
    payload: Mapped[dict] = mapped_column(JSON, default=dict)
    prev_hash: Mapped[str] = mapped_column(String(64), default="")
    hash: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class SyntheticDataRun(Base):
    __tablename__ = "synthetic_data_runs"

    id: Mapped[str] = mapped_column(GUID, primary_key=True, default=new_uuid)
    dataset_version: Mapped[str] = mapped_column(String(50), default="v1")
    generator_version: Mapped[str] = mapped_column(String(20), default="1.0")
    generation_seed: Mapped[int] = mapped_column(Integer, default=42)
    run_params: Mapped[dict] = mapped_column(JSON, default=dict)
    records_created: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class BackgroundJob(Base):
    """
    Tracks Celery task status (Feature E). Written by the task itself on
    start/finish so /jobs/{id} can report status without querying Celery's
    result backend directly (keeps the API simple and DB-transactional with
    the rest of the app's audit trail).
    """

    __tablename__ = "background_jobs"

    id: Mapped[str] = mapped_column(GUID, primary_key=True, default=new_uuid)
    task_name: Mapped[str] = mapped_column(String(100), nullable=False)
    status: Mapped[str] = mapped_column(String(20), default="QUEUED")  # QUEUED / RUNNING / SUCCESS / FAILURE
    params: Mapped[dict] = mapped_column(JSON, default=dict)
    result: Mapped[dict] = mapped_column(JSON, default=dict)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
