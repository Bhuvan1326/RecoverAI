from datetime import datetime, timedelta, timezone

import bcrypt
from jose import JWTError, jwt

from app.core.config import get_settings

settings = get_settings()

# Using the `bcrypt` library directly rather than passlib's CryptContext:
# passlib 1.7.4's bcrypt backend-detection probe is incompatible with
# bcrypt>=4.1 (known upstream issue), so we call bcrypt directly for a
# robust, dependency-light password hash.
_BCRYPT_MAX_BYTES = 72  # bcrypt's own input limit


def hash_password(password: str) -> str:
    pw_bytes = password.encode("utf-8")[:_BCRYPT_MAX_BYTES]
    return bcrypt.hashpw(pw_bytes, bcrypt.gensalt()).decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    pw_bytes = plain_password.encode("utf-8")[:_BCRYPT_MAX_BYTES]
    try:
        return bcrypt.checkpw(pw_bytes, hashed_password.encode("utf-8"))
    except ValueError:
        return False


def create_access_token(subject: str, role: str, expires_minutes: int | None = None) -> str:
    expire = datetime.now(timezone.utc) + timedelta(
        minutes=expires_minutes or settings.ACCESS_TOKEN_EXPIRE_MINUTES
    )
    payload = {"sub": subject, "role": role, "exp": expire}
    return jwt.encode(payload, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


def decode_access_token(token: str) -> dict | None:
    try:
        return jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
    except JWTError:
        return None
