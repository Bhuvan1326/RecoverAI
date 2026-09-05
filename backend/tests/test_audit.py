from sqlalchemy import select

from app.models.domain import AuditLog
from app.services.audit import record_event, verify_chain


def test_hash_chain_valid_after_multiple_events(db_engine):
    from sqlalchemy.orm import sessionmaker

    Session = sessionmaker(bind=db_engine)
    db = Session()

    record_event(db, actor_type="user", actor_id="u1", event_type="claim.created", payload={"a": 1})
    record_event(db, actor_type="agent", actor_id="model", event_type="claim.scored", payload={"score": 0.5})
    record_event(db, actor_type="user", actor_id="u1", event_type="workflow_action.approved", payload={"note": "ok"})

    ok, broken_at = verify_chain(db)
    assert ok is True
    assert broken_at is None


def test_hash_chain_detects_tampering(db_engine):
    from sqlalchemy.orm import sessionmaker

    Session = sessionmaker(bind=db_engine)
    db = Session()

    record_event(db, actor_type="user", actor_id="u1", event_type="claim.created", payload={"a": 1})
    record_event(db, actor_type="agent", actor_id="model", event_type="claim.scored", payload={"score": 0.5})

    # Simulate tampering: directly mutate a stored payload after the fact.
    row = db.execute(select(AuditLog).order_by(AuditLog.created_at.asc())).scalars().first()
    row.payload = {"a": 99999}
    db.commit()

    ok, broken_at = verify_chain(db)
    assert ok is False
    assert broken_at == row.id
