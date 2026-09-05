from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.deps import get_current_user, get_optional_current_user
from app.core.security import create_access_token, hash_password, verify_password
from app.models.domain import User, UserRole
from app.schemas.schemas import ChangePasswordIn, Token, UserCreate, UserOut
from app.services.audit import record_event

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/login", response_model=Token)
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.execute(select(User).where(User.email == form_data.username)).scalar_one_or_none()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect email or password")
    if not user.is_active:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="User is inactive")

    token = create_access_token(subject=user.id, role=user.role.value)
    record_event(db, actor_type="user", actor_id=user.id, event_type="user.login", payload={"email": user.email})
    return Token(access_token=token, user=UserOut.model_validate(user))


@router.post("/register", response_model=UserOut)
def register(
    payload: UserCreate,
    db: Session = Depends(get_db),
    current_user: User | None = Depends(get_optional_current_user),
):
    """
    Registration is locked down to two cases, and only two:

    1. Bootstrap: the database has NO users yet. Anyone may call this
       unauthenticated, and the resulting account is ALWAYS forced to
       ADMIN regardless of what `role` was sent in the payload -- there is
       no other way to get an admin into a fresh install.
    2. Post-bootstrap: once at least one user exists, this endpoint
       requires a valid ADMIN bearer token. Any other caller -- no token,
       an expired/invalid token, or a valid token for a non-admin role --
       is rejected. The role in the payload is honored ONLY because an
       authenticated admin is the one submitting it; it is never trusted
       from an unauthenticated request, and a non-admin authenticated user
       cannot use this endpoint at all (not even to create another
       non-admin account) -- see app/api/users.py for the equivalent
       admin-only user-management surface if self-service signup for
       lower-privileged roles is ever wanted, which would still need its
       own explicit role allow-list rather than trusting the payload.
    """
    any_user_exists = db.execute(select(User.id).limit(1)).first() is not None

    if not any_user_exists:
        role = UserRole.ADMIN  # bootstrap: first user is always admin, payload.role is ignored
    else:
        if current_user is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Registration requires administrator authentication once an account already exists.",
            )
        if current_user.role != UserRole.ADMIN:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{current_user.role.value}' is not permitted to register new users.",
            )
        role = payload.role  # trusted here ONLY because the caller is a verified admin

    existing = db.execute(select(User).where(User.email == payload.email)).scalar_one_or_none()
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")

    user = User(email=payload.email, full_name=payload.full_name, hashed_password=hash_password(payload.password), role=role)
    db.add(user)
    db.commit()
    db.refresh(user)
    record_event(
        db, actor_type="user", actor_id=(current_user.id if current_user else user.id),
        event_type="user.registered", payload={"email": user.email, "role": role.value, "bootstrap": not any_user_exists},
    )
    return UserOut.model_validate(user)


@router.get("/me", response_model=UserOut)
def me(current_user: User = Depends(get_current_user)):
    return UserOut.model_validate(current_user)


@router.post("/change-password")
def change_password(
    payload: ChangePasswordIn,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Any authenticated user may change their own password. Requires the
    current password (never trusts that holding a valid JWT alone is
    sufficient to change credentials -- protects against a stolen/leaked
    short-lived token being used to lock the real owner out).
    """
    if not verify_password(payload.current_password, current_user.hashed_password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Current password is incorrect")

    current_user.hashed_password = hash_password(payload.new_password)
    db.commit()
    record_event(db, actor_type="user", actor_id=current_user.id, event_type="user.password_changed", payload={})
    return {"status": "password updated"}
