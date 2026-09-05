"""
Regression tests for the P0 admin-registration vulnerability: prior to this
fix, /auth/register accepted an arbitrary `role` in the request body for
ANY caller once the bootstrap admin already existed -- an unauthenticated
attacker could register as ADMIN or REVIEWER simply by sending
{"role": "ADMIN"}. Fixed in app/api/auth.py:register.
"""


def test_first_registration_creates_admin_regardless_of_requested_role(client):
    """Bootstrap: the very first user is always ADMIN, even if a different
    role was requested in the payload."""
    r = client.post(
        "/auth/register",
        json={"email": "first@example.com", "full_name": "First User", "password": "TestPass123!", "role": "ANALYST"},
    )
    assert r.status_code == 200
    assert r.json()["role"] == "ADMIN"


def test_second_unauthenticated_registration_is_rejected(client, admin_token):
    """Once a user exists, an unauthenticated registration attempt -- for
    ANY role, including the lowest-privilege one -- must be rejected."""
    r = client.post(
        "/auth/register",
        json={"email": "nobody@example.com", "full_name": "Nobody", "password": "TestPass123!", "role": "ANALYST"},
    )
    assert r.status_code == 401


def test_unauthenticated_admin_role_creation_is_rejected(client, admin_token):
    """The exact attack this vulnerability enabled: an unauthenticated
    request explicitly asking for role=ADMIN after bootstrap."""
    r = client.post(
        "/auth/register",
        json={"email": "attacker@example.com", "full_name": "Attacker", "password": "Password123!", "role": "ADMIN"},
    )
    assert r.status_code == 401

    # And confirm no such account was actually created.
    login = client.post("/auth/login", data={"username": "attacker@example.com", "password": "Password123!"})
    assert login.status_code == 401


def test_unauthenticated_reviewer_role_creation_is_rejected(client, admin_token):
    r = client.post(
        "/auth/register",
        json={"email": "attacker2@example.com", "full_name": "Attacker", "password": "Password123!", "role": "REVIEWER"},
    )
    assert r.status_code == 401


def test_biller_cannot_create_admin_user(client, admin_token):
    # Create a BILLER via the admin (legitimate path).
    client.post(
        "/auth/register",
        headers={"Authorization": f"Bearer {admin_token}"},
        json={"email": "biller@example.com", "full_name": "Biller", "password": "TestPass123!", "role": "BILLER"},
    )
    biller_login = client.post("/auth/login", data={"username": "biller@example.com", "password": "TestPass123!"})
    biller_token = biller_login.json()["access_token"]

    r = client.post(
        "/auth/register",
        headers={"Authorization": f"Bearer {biller_token}"},
        json={"email": "escalated@example.com", "full_name": "Escalated", "password": "TestPass123!", "role": "ADMIN"},
    )
    assert r.status_code == 403


def test_analyst_cannot_create_admin_user(client, admin_token, analyst_token):
    r = client.post(
        "/auth/register",
        headers={"Authorization": f"Bearer {analyst_token}"},
        json={"email": "escalated2@example.com", "full_name": "Escalated", "password": "TestPass123!", "role": "ADMIN"},
    )
    assert r.status_code == 403

    # Confirm no account was created despite the 403.
    login = client.post("/auth/login", data={"username": "escalated2@example.com", "password": "TestPass123!"})
    assert login.status_code == 401


def test_analyst_cannot_create_any_user_not_just_admin(client, analyst_token):
    """The fix is not merely 'non-admins can't grant ADMIN' -- non-admins
    cannot use /auth/register at all, matching the admin-only user-creation
    model the rest of the app (app/api/users.py) already uses."""
    r = client.post(
        "/auth/register",
        headers={"Authorization": f"Bearer {analyst_token}"},
        json={"email": "another-analyst@example.com", "full_name": "Another", "password": "TestPass123!", "role": "ANALYST"},
    )
    assert r.status_code == 403


def test_admin_can_create_users_with_allowed_roles(client, admin_token):
    headers = {"Authorization": f"Bearer {admin_token}"}
    for role in ["ADMIN", "REVIEWER", "BILLER", "ANALYST"]:
        r = client.post(
            "/auth/register",
            headers=headers,
            json={"email": f"user-{role.lower()}@example.com", "full_name": role, "password": "TestPass123!", "role": role},
        )
        assert r.status_code == 200, r.text
        assert r.json()["role"] == role


def test_role_cannot_be_escalated_through_request_payload_by_a_valid_lower_role_token(client, admin_token):
    """
    Even with a syntactically valid, currently-active JWT for a real
    non-admin user, the payload's requested role must never be honored.
    This directly re-tests the original vulnerability's exact shape
    (a real, authenticated-but-unprivileged caller supplying role=ADMIN)
    rather than only the fully-unauthenticated case above.
    """
    headers = {"Authorization": f"Bearer {admin_token}"}
    client.post(
        "/auth/register",
        headers=headers,
        json={"email": "lowpriv@example.com", "full_name": "Low Priv", "password": "TestPass123!", "role": "ANALYST"},
    )
    low_login = client.post("/auth/login", data={"username": "lowpriv@example.com", "password": "TestPass123!"})
    low_token = low_login.json()["access_token"]

    r = client.post(
        "/auth/register",
        headers={"Authorization": f"Bearer {low_token}"},
        json={"email": "shouldnotexist@example.com", "full_name": "X", "password": "TestPass123!", "role": "ADMIN"},
    )
    assert r.status_code == 403

    login = client.post("/auth/login", data={"username": "shouldnotexist@example.com", "password": "TestPass123!"})
    assert login.status_code == 401


def test_expired_or_garbage_token_is_treated_as_unauthenticated_for_registration(client, admin_token):
    """A malformed/garbage bearer token must fall through to the
    unauthenticated rejection path, not crash or bypass the check."""
    r = client.post(
        "/auth/register",
        headers={"Authorization": "Bearer not-a-real-jwt-at-all"},
        json={"email": "garbage@example.com", "full_name": "G", "password": "TestPass123!", "role": "ADMIN"},
    )
    assert r.status_code == 401
