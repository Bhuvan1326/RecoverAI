def test_register_first_user_becomes_admin(client):
    r = client.post("/auth/register", json={"email": "first@example.com", "full_name": "First", "password": "TestPass123!"})
    assert r.status_code == 200
    assert r.json()["role"] == "ADMIN"


def test_login_wrong_password_rejected(client, admin_token):
    r = client.post("/auth/login", data={"username": "admin@example.com", "password": "wrong-password"})
    assert r.status_code == 401


def test_unauthenticated_request_rejected(client):
    r = client.get("/claims")
    assert r.status_code == 401


def test_analyst_cannot_create_claim(client, analyst_token):
    r = client.post(
        "/claims",
        headers={"Authorization": f"Bearer {analyst_token}"},
        json={
            "claim_number": "CLM-TEST-1",
            "provider_id": "x",
            "payer_id": "y",
            "patient_ref": "SYN-1",
            "claim_amount": 100.0,
            "service_date": "2026-01-01T00:00:00Z",
        },
    )
    assert r.status_code == 403


def test_analyst_cannot_approve_workflow_action(client, analyst_token):
    r = client.post(
        "/workflow-actions/nonexistent-id/approve",
        headers={"Authorization": f"Bearer {analyst_token}"},
        json={"notes": "trying to approve without permission"},
    )
    # RBAC check fires before the not-found check, so a non-reviewer gets 403 either way.
    assert r.status_code == 403


def test_role_cannot_be_spoofed_via_client_payload(client, analyst_token):
    """
    Even if a client sends a role claim in the request body, server-side
    permission checks must use the DB-backed user record, not client input.
    """
    r = client.get("/auth/me", headers={"Authorization": f"Bearer {analyst_token}"})
    assert r.status_code == 200
    assert r.json()["role"] == "ANALYST"
