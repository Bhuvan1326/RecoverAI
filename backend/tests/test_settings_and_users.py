def test_change_password_requires_auth(client):
    r = client.post("/auth/change-password", json={"current_password": "x", "new_password": "newpass123"})
    assert r.status_code == 401


def test_change_password_wrong_current_password_rejected(client, admin_token):
    headers = {"Authorization": f"Bearer {admin_token}"}
    r = client.post("/auth/change-password", headers=headers, json={"current_password": "WrongPassword!", "new_password": "NewPassword123!"})
    assert r.status_code == 401


def test_change_password_succeeds_and_new_password_works(client, admin_token):
    headers = {"Authorization": f"Bearer {admin_token}"}
    r = client.post("/auth/change-password", headers=headers, json={"current_password": "TestPass123!", "new_password": "BrandNewPass456!"})
    assert r.status_code == 200

    # Old password no longer works.
    r = client.post("/auth/login", data={"username": "admin@example.com", "password": "TestPass123!"})
    assert r.status_code == 401

    # New password works.
    r = client.post("/auth/login", data={"username": "admin@example.com", "password": "BrandNewPass456!"})
    assert r.status_code == 200


def test_change_password_rejects_too_short_new_password(client, admin_token):
    headers = {"Authorization": f"Bearer {admin_token}"}
    r = client.post("/auth/change-password", headers=headers, json={"current_password": "TestPass123!", "new_password": "short"})
    assert r.status_code == 422


def test_list_users_requires_admin(client, analyst_token):
    headers = {"Authorization": f"Bearer {analyst_token}"}
    r = client.get("/users", headers=headers)
    assert r.status_code == 403


def test_list_users_returns_all_users_for_admin(client, admin_token, analyst_token):
    headers = {"Authorization": f"Bearer {admin_token}"}
    r = client.get("/users", headers=headers)
    assert r.status_code == 200
    emails = {u["email"] for u in r.json()}
    assert "admin@example.com" in emails
    assert "analyst@example.com" in emails


def test_update_user_role_requires_admin(client, analyst_token):
    headers = {"Authorization": f"Bearer {analyst_token}"}
    r = client.patch("/users/some-id/role", headers=headers, json={"role": "ADMIN"})
    assert r.status_code == 403


def test_update_user_role_changes_role(client, admin_token, analyst_token):
    admin_headers = {"Authorization": f"Bearer {admin_token}"}
    r = client.get("/auth/me", headers={"Authorization": f"Bearer {analyst_token}"})
    analyst_id = r.json()["id"]

    r = client.patch(f"/users/{analyst_id}/role", headers=admin_headers, json={"role": "REVIEWER"})
    assert r.status_code == 200
    assert r.json()["role"] == "REVIEWER"


def test_cannot_demote_the_only_admin(client, admin_token):
    headers = {"Authorization": f"Bearer {admin_token}"}
    r = client.get("/auth/me", headers=headers)
    admin_id = r.json()["id"]

    r = client.patch(f"/users/{admin_id}/role", headers=headers, json={"role": "ANALYST"})
    assert r.status_code == 400


def test_can_demote_admin_when_another_admin_exists(client, admin_token):
    headers = {"Authorization": f"Bearer {admin_token}"}
    # Create a second admin.
    client.post("/auth/register", headers=headers, json={"email": "admin2@example.com", "full_name": "Admin Two", "password": "TestPass123!", "role": "ADMIN"})

    r = client.get("/auth/me", headers=headers)
    admin_id = r.json()["id"]
    r = client.patch(f"/users/{admin_id}/role", headers=headers, json={"role": "ANALYST"})
    assert r.status_code == 200


def test_update_user_active_requires_admin(client, analyst_token):
    headers = {"Authorization": f"Bearer {analyst_token}"}
    r = client.patch("/users/some-id/active", headers=headers, json={"is_active": False})
    assert r.status_code == 403


def test_deactivate_user_then_they_cannot_log_in(client, admin_token, analyst_token):
    admin_headers = {"Authorization": f"Bearer {admin_token}"}
    r = client.get("/auth/me", headers={"Authorization": f"Bearer {analyst_token}"})
    analyst_id = r.json()["id"]

    r = client.patch(f"/users/{analyst_id}/active", headers=admin_headers, json={"is_active": False})
    assert r.status_code == 200
    assert r.json()["is_active"] is False

    r = client.post("/auth/login", data={"username": "analyst@example.com", "password": "TestPass123!"})
    assert r.status_code == 403


def test_cannot_deactivate_the_only_admin(client, admin_token):
    headers = {"Authorization": f"Bearer {admin_token}"}
    r = client.get("/auth/me", headers=headers)
    admin_id = r.json()["id"]

    r = client.patch(f"/users/{admin_id}/active", headers=headers, json={"is_active": False})
    assert r.status_code == 400


def test_update_user_role_404_for_unknown_user(client, admin_token):
    headers = {"Authorization": f"Bearer {admin_token}"}
    r = client.patch("/users/nonexistent-id/role", headers=headers, json={"role": "ADMIN"})
    assert r.status_code == 404


def test_system_settings_requires_admin(client, analyst_token):
    headers = {"Authorization": f"Bearer {analyst_token}"}
    r = client.get("/settings/system", headers=headers)
    assert r.status_code == 403


def test_system_settings_never_exposes_secrets(client, admin_token):
    headers = {"Authorization": f"Bearer {admin_token}"}
    r = client.get("/settings/system", headers=headers)
    assert r.status_code == 200
    body_text = str(r.json())
    # No secret material should ever appear in the response, regardless of
    # what's actually configured in this test environment.
    assert "JWT_SECRET_KEY" not in body_text
    assert "postgresql" not in body_text.lower()
    assert "password" not in body_text.lower()
    # API keys are surfaced as booleans, never as key material.
    assert r.json()["rag"]["anthropic_api_key_configured"] in (True, False)
    assert r.json()["rag"]["openai_api_key_configured"] in (True, False)


def test_system_settings_returns_real_configured_values(client, admin_token):
    headers = {"Authorization": f"Bearer {admin_token}"}
    r = client.get("/settings/system", headers=headers)
    body = r.json()
    assert body["drift_detection"]["psi_warning_threshold"] == 0.10
    assert body["anomaly_detection"]["contamination"] == 0.05
    assert "daily_drift_check_utc" in body["celery"]
