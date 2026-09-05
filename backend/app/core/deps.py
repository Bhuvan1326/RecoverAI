from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.security import decode_access_token
from app.models.domain import User, UserRole

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login", auto_error=False)


def get_current_user(token: str | None = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    if not token:
        raise credentials_exception
    payload = decode_access_token(token)
    if payload is None:
        raise credentials_exception
    user_id = payload.get("sub")
    user = db.get(User, user_id)
    if user is None or not user.is_active:
        raise credentials_exception
    return user


def get_optional_current_user(token: str | None = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> User | None:
    """
    Like get_current_user, but returns None instead of raising 401 when no
    (or an invalid) token is present. Used ONLY for the bootstrap-registration
    endpoint, which must distinguish "no users yet, allow unauthenticated
    bootstrap" from "users already exist, an admin token is required" --
    every other endpoint should use get_current_user / require_roles, which
    fail closed.
    """
    if not token:
        return None
    payload = decode_access_token(token)
    if payload is None:
        return None
    user_id = payload.get("sub")
    user = db.get(User, user_id)
    if user is None or not user.is_active:
        return None
    return user


def require_roles(*allowed_roles: UserRole):
    """
    RBAC dependency factory. Role is ALWAYS read from the server-validated
    JWT / DB user record -- never trusted from client-supplied data.
    """

    def _check(user: User = Depends(get_current_user)) -> User:
        if user.role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{user.role.value}' is not permitted to perform this action.",
            )
        return user

    return _check
