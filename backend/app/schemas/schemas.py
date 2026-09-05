from datetime import datetime

from pydantic import BaseModel, ConfigDict, EmailStr, Field

from app.models.domain import UserRole, WorkflowActionStatus, WorkflowActionType


# --- Auth ---
class UserCreate(BaseModel):
    email: EmailStr
    full_name: str
    password: str = Field(min_length=8)
    role: UserRole = UserRole.ANALYST


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    email: str
    full_name: str
    role: UserRole
    is_active: bool = True
    created_at: datetime


class ChangePasswordIn(BaseModel):
    current_password: str
    new_password: str = Field(min_length=8)


class UserRoleUpdateIn(BaseModel):
    role: UserRole


class UserActiveUpdateIn(BaseModel):
    is_active: bool


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserOut


# --- Claims ---
class ClaimLineIn(BaseModel):
    procedure_code: str
    diagnosis_code: str
    modifiers: str = ""
    line_amount: float
    units: int = 1


class ClaimCreate(BaseModel):
    claim_number: str
    provider_id: str
    payer_id: str
    patient_ref: str
    claim_amount: float
    claim_type: str = "professional"
    place_of_service: str = "11"
    eligibility_status: str = "VERIFIED"
    authorization_status: str = "PRESENT"
    documentation_completeness: float = 100.0
    service_date: datetime
    submission_date: datetime | None = None
    lines: list[ClaimLineIn] = []


class ClaimOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    claim_number: str
    provider_id: str
    payer_id: str
    claim_amount: float
    status: str
    eligibility_status: str
    authorization_status: str
    documentation_completeness: float
    service_date: datetime
    submission_date: datetime | None
    is_synthetic: bool


class DenialScoreOut(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    claim_id: str
    denial_probability: float
    risk_category: str
    model_version: str


class WorkflowActionApprovalIn(BaseModel):
    notes: str = ""


class WorkflowActionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    claim_id: str
    action_type: WorkflowActionType
    recommended_by: str
    status: WorkflowActionStatus
    payload: dict
    created_at: datetime
