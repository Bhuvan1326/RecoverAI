"""
Regression tests for Section 6/7 of the hardening spec: single-claim
inference for both the denial-risk and appeal-success models used to
hardcode payer/provider (and payer/denial-reason) historical rate features
to a constant, ignoring whatever real history actually existed. Fixed in
ml/features/build_features.py:compute_asof_denial_rate and
ml/features/build_appeal_features.py:compute_asof_appeal_rate.
"""
from datetime import datetime, timedelta, timezone

import pytest


@pytest.fixture
def db_session(client):
    override_gen_fn = next(iter(client.app.dependency_overrides.values()))
    return next(override_gen_fn())


def _make_payer_provider(db, suffix):
    from app.models.domain import Payer, Provider

    payer = Payer(name=f"Historical Test Payer {suffix}")
    provider = Provider(npi=f"9{suffix:0>9}"[:10], name=f"Historical Test Provider {suffix}")
    db.add(payer)
    db.add(provider)
    db.commit()
    db.refresh(payer)
    db.refresh(provider)
    return payer, provider


def _make_claim(db, payer, provider, submission_date, denied: bool, claim_number: str):
    from app.models.domain import Claim, DenialEvent

    claim = Claim(
        claim_number=claim_number,
        provider_id=provider.id,
        payer_id=payer.id,
        patient_ref="SYN-PT",
        claim_amount=1000.0,
        service_date=submission_date - timedelta(days=5),
        submission_date=submission_date,
        is_synthetic=False,
        data_source="test",
    )
    db.add(claim)
    db.flush()
    if denied:
        db.add(DenialEvent(claim_id=claim.id, denial_reason_code="OTHER", denial_date=submission_date + timedelta(days=5)))
    db.commit()
    return claim


def test_asof_denial_rate_returns_neutral_prior_with_insufficient_history(db_session):
    from ml.features.build_features import compute_asof_denial_rate

    payer, _ = _make_payer_provider(db_session, 1)
    cutoff = datetime.now(timezone.utc)
    rate = compute_asof_denial_rate(db_session, "payer_id", payer.id, cutoff)
    assert rate == 0.15  # documented cold-start fallback, zero prior claims


def test_asof_denial_rate_reflects_actual_history_not_a_constant(db_session):
    """
    The core regression: a payer with a genuinely high historical denial
    rate must produce a HIGH historical_denial_rate feature, and a payer
    with a genuinely low rate must produce a LOW one -- proving the value
    is actually computed from data, not hardcoded to 0.15 for everyone.
    """
    from ml.features.build_features import compute_asof_denial_rate

    high_payer, provider = _make_payer_provider(db_session, 2)
    low_payer, _ = _make_payer_provider(db_session, 3)
    base = datetime.now(timezone.utc) - timedelta(days=100)

    # High-denial payer: 8 of 10 prior claims denied.
    for i in range(10):
        _make_claim(db_session, high_payer, provider, base + timedelta(days=i), denied=(i < 8), claim_number=f"HIST-HIGH-{i}")
    # Low-denial payer: 1 of 10 prior claims denied.
    for i in range(10):
        _make_claim(db_session, low_payer, provider, base + timedelta(days=i), denied=(i < 1), claim_number=f"HIST-LOW-{i}")

    cutoff = base + timedelta(days=50)
    high_rate = compute_asof_denial_rate(db_session, "payer_id", high_payer.id, cutoff)
    low_rate = compute_asof_denial_rate(db_session, "payer_id", low_payer.id, cutoff)

    assert high_rate == pytest.approx(0.8)
    assert low_rate == pytest.approx(0.1)
    assert high_rate != low_rate  # the actual regression: these used to both be 0.15


def test_asof_denial_rate_never_sees_claims_at_or_after_cutoff(db_session):
    """A claim submitted exactly at or after the cutoff must not count --
    this is the leakage guard, re-verified for the live-query path (not
    just the training dataframe path already covered elsewhere)."""
    from ml.features.build_features import compute_asof_denial_rate

    payer, provider = _make_payer_provider(db_session, 4)
    base = datetime.now(timezone.utc) - timedelta(days=50)

    for i in range(6):
        _make_claim(db_session, payer, provider, base + timedelta(days=i), denied=True, claim_number=f"HIST-CUT-{i}")
    # A claim submitted AFTER the cutoff, denied=False -- must not dilute the rate.
    _make_claim(db_session, payer, provider, base + timedelta(days=100), denied=False, claim_number="HIST-CUT-FUTURE")

    cutoff = base + timedelta(days=10)
    rate = compute_asof_denial_rate(db_session, "payer_id", payer.id, cutoff)
    assert rate == 1.0  # only the 6 prior all-denied claims count


def test_ml_inference_uses_real_historical_rate_not_hardcoded_constant(db_session, client, admin_token):
    """
    Full integration: score a real claim for a payer with a strong denial
    history and confirm the feature actually fed to the model reflects
    that history, via the same code path /claims/{id}/score uses.
    """
    from app.models.domain import Claim
    from app.services.ml_inference import _claim_to_feature_row

    payer, provider = _make_payer_provider(db_session, 5)
    base = datetime.now(timezone.utc) - timedelta(days=100)
    for i in range(10):
        _make_claim(db_session, payer, provider, base + timedelta(days=i), denied=True, claim_number=f"HIST-INF-{i}")

    new_claim = _make_claim(db_session, payer, provider, base + timedelta(days=50), denied=False, claim_number="HIST-INF-NEW")
    row = _claim_to_feature_row(db_session, new_claim)
    assert row["payer_historical_denial_rate"].iloc[0] == pytest.approx(1.0)


def test_asof_appeal_rate_reflects_actual_history_not_a_constant(db_session):
    from app.models.domain import AppealEvent, DenialEvent
    from ml.features.build_appeal_features import compute_asof_appeal_rate

    payer, provider = _make_payer_provider(db_session, 6)
    base = datetime.now(timezone.utc) - timedelta(days=100)

    for i in range(8):
        claim = _make_claim(db_session, payer, provider, base + timedelta(days=i), denied=True, claim_number=f"HIST-APPEAL-{i}")
        denial = db_session.query(DenialEvent).filter(DenialEvent.claim_id == claim.id).first()
        # 6 of 8 appeals WON for this payer.
        db_session.add(
            AppealEvent(
                denial_event_id=denial.id,
                appeal_date=base + timedelta(days=i, hours=1),
                outcome="WON" if i < 6 else "LOST",
                recovered_amount=1000.0 if i < 6 else 0.0,
            )
        )
    db_session.commit()

    cutoff = base + timedelta(days=50)
    rate = compute_asof_appeal_rate(db_session, "payer_id", payer.id, cutoff)
    assert rate == pytest.approx(0.75)
    assert rate != 0.40  # the actual regression: this used to always be 0.40
