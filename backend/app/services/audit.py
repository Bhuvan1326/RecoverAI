"""
Append-only, hash-chained audit log.

Every important AI/human event is written here (Feature 20). This is a
cheap hash chain -- NOT a blockchain -- but it gives us tamper-evidence:
if any row's payload is edited after the fact, its hash no longer matches
prev_hash + payload, and the chain visibly breaks from that point forward.
The DB user the API runs as should be granted INSERT/SELECT only on this
table (no UPDATE/DELETE) in the Postgres deployment -- see docker/init.sql.
"""
import hashlib
import json

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.domain import AuditLog


def _hash(prev_hash: str, actor_type: str, event_type: str, payload: dict) -> str:
    material = json.dumps(
        {"prev_hash": prev_hash, "actor_type": actor_type, "event_type": event_type, "payload": payload},
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def record_event(
    db: Session,
    *,
    actor_type: str,
    actor_id: str | None,
    event_type: str,
    claim_id: str | None = None,
    payload: dict | None = None,
) -> AuditLog:
    payload = payload or {}
    last = db.execute(select(AuditLog).order_by(AuditLog.created_at.desc()).limit(1)).scalar_one_or_none()
    prev_hash = last.hash if last else ""
    entry_hash = _hash(prev_hash, actor_type, event_type, payload)

    entry = AuditLog(
        actor_type=actor_type,
        actor_id=actor_id,
        event_type=event_type,
        claim_id=claim_id,
        payload=payload,
        prev_hash=prev_hash,
        hash=entry_hash,
    )
    db.add(entry)
    db.commit()
    db.refresh(entry)
    return entry


def verify_chain(db: Session) -> tuple[bool, str | None]:
    """Walk the whole log in order and confirm every hash is consistent."""
    rows = db.execute(select(AuditLog).order_by(AuditLog.created_at.asc())).scalars().all()
    prev_hash = ""
    for row in rows:
        expected = _hash(prev_hash, row.actor_type, row.event_type, row.payload)
        if expected != row.hash or row.prev_hash != prev_hash:
            return False, row.id
        prev_hash = row.hash
    return True, None
