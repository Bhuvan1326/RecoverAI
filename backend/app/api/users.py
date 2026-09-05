"""
Admin-only user management (Settings page). Every mutating endpoint here is
gated to ADMIN via require_roles -- role is read from the DB-backed user
record via the JWT, never trusted from client input, same as everywhere
else in the app.
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select, func
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.deps import get_current_user, require_roles
from app.models.domain import User, UserRole
from app.schemas.schemas import UserActiveUpdateIn, UserOut, UserRoleUpdateIn
from app.services.audit import record_event

router = APIRouter(prefix="/users", tags=["users"])


@router.get("", response_model=list[UserOut])
def list_users(db: Session = Depends(get_db), _user: User = Depends(require_roles(UserRole.ADMIN))):
    return db.execute(select(User).order_by(User.created_at.asc())).scalars().all()


def _count_active_admins(db: Session, excluding_user_id: str | None = None) -> int:
    q = select(func.count()).select_from(User).where(User.role == UserRole.ADMIN, User.is_active.is_(True))
    if excluding_user_id:
        q = q.where(User.id != excluding_user_id)
    return db.execute(q).scalar_one()


@router.patch("/{user_id}/role", response_model=UserOut)
def update_user_role(
    user_id: str,
    payload: UserRoleUpdateIn,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(UserRole.ADMIN)),
):
    target = db.get(User, user_id)
    if not target:
        raise HTTPException(status_code=404, detail="User not found")

    if target.id == current_user.id and target.role == UserRole.ADMIN and payload.role != UserRole.ADMIN:
        # Prevent an admin from demoting themselves out of the only admin
        # seat and locking the whole app out of admin-only actions (user
        # management, workflow approval, drift/training triggers).
        if _count_active_admins(db, excluding_user_id=current_user.id) == 0:
            raise HTTPException(status_code=400, detail="Cannot remove the last active admin's admin role")

    old_role = target.role
    target.role = payload.role
    db.commit()
    db.refresh(target)
    record_event(
        db, actor_type="user", actor_id=current_user.id, event_type="user.role_changed",
        payload={"target_user_id": user_id, "old_role": old_role.value, "new_role": payload.role.value},
    )
    return target


@router.patch("/{user_id}/active", response_model=UserOut)
def update_user_active(
    user_id: str,
    payload: UserActiveUpdateIn,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(UserRole.ADMIN)),
):
    target = db.get(User, user_id)
    if not target:
        raise HTTPException(status_code=404, detail="User not found")

    if not payload.is_active and target.role == UserRole.ADMIN:
        if _count_active_admins(db, excluding_user_id=target.id) == 0:
            raise HTTPException(status_code=400, detail="Cannot deactivate the last active admin")

    target.is_active = payload.is_active
    db.commit()
    db.refresh(target)
    record_event(
        db, actor_type="user", actor_id=current_user.id, event_type="user.active_changed",
        payload={"target_user_id": user_id, "is_active": payload.is_active},
    )
    return target
