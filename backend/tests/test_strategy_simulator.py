from datetime import datetime, timedelta, timezone


def _create_denied_claim(client, headers, claim_number, amount, doc_completeness=95.0, auth="PRESENT", denial_reason="MISSING_AUTHORIZATION"):
    from app.models.domain import Claim, ClaimStatus, DenialEvent, Payer, Provider

    override_gen_fn = next(iter(client.app.dependency_overrides.values()))
    db = next(override_gen_fn())
    payer = Payer(name="Strategy Test Payer")
    npi = "3" + str(abs(hash(claim_number)) % 10_000_000_000).zfill(10)[:9]
    provider = Provider(npi=npi, name="Strategy Test Provider")
    db.add(payer)
    db.add(provider)
    db.commit()
    db.refresh(payer)
    db.refresh(provider)

    payload = {
        "claim_number": claim_number,
        "provider_id": provider.id,
        "payer_id": payer.id,
        "patient_ref": "SYN-PT-STRAT",
        "claim_amount": amount,
        "authorization_status": auth,
        "documentation_completeness": doc_completeness,
        "service_date": "2026-01-01T00:00:00Z",
        "lines": [{"procedure_code": "99213", "diagnosis_code": "I10", "line_amount": amount, "units": 1}],
    }
    r = client.post("/claims", headers=headers, json=payload)
    assert r.status_code == 200, r.text
    claim_out = r.json()

    claim = db.get(Claim, claim_out["id"])
    claim.status = ClaimStatus.DENIED
    denial = DenialEvent(claim_id=claim.id, denial_reason_code=denial_reason, denial_date=datetime.now(timezone.utc) - timedelta(days=3))
    db.add(denial)
    db.commit()
    return claim_out


def test_simulate_strategy_requires_auth(client):
    r = client.post("/recovery-queue/simulate-strategy")
    assert r.status_code == 401


def test_simulate_strategy_rejects_non_positive_staff_hours(client, admin_token):
    headers = {"Authorization": f"Bearer {admin_token}"}
    r = client.post("/recovery-queue/simulate-strategy?staff_hours=0", headers=headers)
    assert r.status_code == 400
    r = client.post("/recovery-queue/simulate-strategy?staff_hours=-5", headers=headers)
    assert r.status_code == 400


def test_simulate_strategy_rejects_implausibly_large_staff_hours(client, admin_token):
    headers = {"Authorization": f"Bearer {admin_token}"}
    r = client.post("/recovery-queue/simulate-strategy?staff_hours=999999", headers=headers)
    assert r.status_code == 400


def test_simulate_strategy_handles_no_denied_claims_gracefully(client, admin_token):
    headers = {"Authorization": f"Bearer {admin_token}"}
    r = client.post("/recovery-queue/simulate-strategy?staff_hours=40", headers=headers)
    assert r.status_code == 200
    body = r.json()
    assert body["claims_available"] == 0
    assert body["strategies"] == []
    assert body["best_strategy"] is None


def test_simulate_strategy_returns_all_three_strategies(client, admin_token):
    headers = {"Authorization": f"Bearer {admin_token}"}
    _create_denied_claim(client, headers, "CLM-STRAT-1", 15000.0)
    _create_denied_claim(client, headers, "CLM-STRAT-2", 500.0)

    r = client.post("/recovery-queue/simulate-strategy?staff_hours=40", headers=headers)
    assert r.status_code == 200
    body = r.json()
    strategy_names = {s["strategy"] for s in body["strategies"]}
    assert strategy_names == {"highest_claim_amount", "highest_appeal_probability", "highest_expected_recovery"}
    assert body["claims_available"] == 2
    assert body["best_strategy"] in strategy_names


def test_simulate_strategy_highest_claim_amount_processes_biggest_claim_first():
    """
    Direct unit test of the ordering logic (not via HTTP) with a tiny staff
    budget that can only fit ONE claim -- proves the 'highest_claim_amount'
    strategy genuinely works the bigger-dollar claim first, not just
    reports a label.
    """
    from app.services.strategy_simulator import _order_for_strategy, _run_strategy

    claims = [
        {"claim_id": "a", "claim_amount": 500.0, "appeal_success_probability": 0.9, "expected_recovery_value": 400.0, "estimated_effort_minutes": 20},
        {"claim_id": "b", "claim_amount": 20000.0, "appeal_success_probability": 0.3, "expected_recovery_value": 5000.0, "estimated_effort_minutes": 20},
    ]
    ordered = _order_for_strategy(claims, "highest_claim_amount")
    assert ordered[0]["claim_id"] == "b"  # the $20,000 claim, despite lower probability

    # A budget that fits exactly one claim's effort -- only the highest-amount claim gets worked.
    result = _run_strategy(claims, "highest_claim_amount", staff_minutes_budget=20)
    assert result["claims_processed"] == 1
    assert result["worked_claim_ids"] == ["b"]


def test_simulate_strategy_capacity_constraint_is_real_not_decorative():
    """A staff budget too small to fit ANY claim's effort must process zero
    claims -- the capacity constraint has to actually bind."""
    from app.services.strategy_simulator import _run_strategy

    claims = [
        {"claim_id": "a", "claim_amount": 500.0, "appeal_success_probability": 0.9, "expected_recovery_value": 400.0, "estimated_effort_minutes": 45},
    ]
    result = _run_strategy(claims, "highest_expected_recovery", staff_minutes_budget=10)
    assert result["claims_processed"] == 0
    assert result["expected_recovery_captured"] == 0.0


def test_simulate_strategy_unlimited_capacity_captures_everything():
    from app.services.strategy_simulator import _run_strategy

    claims = [
        {"claim_id": "a", "claim_amount": 1000.0, "appeal_success_probability": 0.5, "expected_recovery_value": 300.0, "estimated_effort_minutes": 20},
        {"claim_id": "b", "claim_amount": 2000.0, "appeal_success_probability": 0.5, "expected_recovery_value": 600.0, "estimated_effort_minutes": 20},
    ]
    result = _run_strategy(claims, "highest_expected_recovery", staff_minutes_budget=1000)
    assert result["claims_processed"] == 2
    assert result["recovery_yield"] == 1.0


def test_simulate_strategy_live_end_to_end_with_real_claims(client, admin_token):
    """
    Creates several denied claims with genuinely different amount/appeal-
    likelihood profiles, then confirms different strategies actually
    produce DIFFERENT orderings under a constrained budget -- not just
    three copies of the same result.
    """
    headers = {"Authorization": f"Bearer {admin_token}"}
    # High amount, low documentation (lower appeal likelihood via the heuristic).
    _create_denied_claim(client, headers, "CLM-STRAT-BIG", 25000.0, doc_completeness=40.0, denial_reason="OTHER")
    # Low amount, high documentation + strong denial reason (higher appeal likelihood).
    _create_denied_claim(client, headers, "CLM-STRAT-SMALL", 800.0, doc_completeness=98.0, denial_reason="MISSING_AUTHORIZATION")

    # Tight budget: only enough effort minutes for ONE claim (each next-best-action
    # estimate is <=45 min in this codebase, so 45 min fits at most one).
    r = client.post("/recovery-queue/simulate-strategy?staff_hours=0.75", headers=headers)
    assert r.status_code == 200
    body = r.json()

    by_strategy = {s["strategy"]: s["worked_claim_ids"] for s in body["strategies"]}
    amount_first = by_strategy["highest_claim_amount"]
    probability_first = by_strategy["highest_appeal_probability"]

    # Under a tight budget that can only fit one claim, the two strategies
    # should pick DIFFERENT claims given how differently the two claims were
    # constructed above -- this is the real behavioral proof the strategies
    # aren't decorative labels on identical output.
    assert amount_first != probability_first
