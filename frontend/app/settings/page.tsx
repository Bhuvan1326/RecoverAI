"use client";

import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import { useAuthGuard } from "@/lib/useAuthGuard";

const ROLES = ["ADMIN", "REVIEWER", "BILLER", "ANALYST"];

function ProfileCard({ user }: { user: any }) {
  const [current, setCurrent] = useState("");
  const [next, setNext] = useState("");
  const [confirm, setConfirm] = useState("");
  const [status, setStatus] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setStatus(null);
    setError(null);
    if (next !== confirm) {
      setError("New passwords do not match.");
      return;
    }
    setBusy(true);
    try {
      await api.changePassword(current, next);
      setStatus("Password updated.");
      setCurrent("");
      setNext("");
      setConfirm("");
    } catch (e: any) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="card mb-6">
      <h2 className="font-semibold mb-3">Profile</h2>
      <div className="text-sm text-white/70 mb-4">
        <div>{user.full_name}</div>
        <div className="text-white/50">{user.email}</div>
        <span className="badge badge-medium mt-1 inline-block">{user.role}</span>
      </div>

      <h3 className="text-sm font-medium mb-2">Change password</h3>
      <form onSubmit={submit} className="space-y-2 max-w-sm">
        <input
          type="password"
          placeholder="Current password"
          className="w-full bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-sm"
          value={current}
          onChange={(e) => setCurrent(e.target.value)}
        />
        <input
          type="password"
          placeholder="New password"
          className="w-full bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-sm"
          value={next}
          onChange={(e) => setNext(e.target.value)}
        />
        <input
          type="password"
          placeholder="Confirm new password"
          className="w-full bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-sm"
          value={confirm}
          onChange={(e) => setConfirm(e.target.value)}
        />
        {error && <div className="text-bad text-sm">{error}</div>}
        {status && <div className="text-good text-sm">{status}</div>}
        <button className="btn btn-primary text-sm" disabled={busy}>
          {busy ? "Updating..." : "Update password"}
        </button>
      </form>
    </div>
  );
}

function UserManagementCard({ currentUserId }: { currentUserId: string }) {
  const [users, setUsers] = useState<any[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [busyId, setBusyId] = useState<string | null>(null);

  function load() {
    api.listUsers().then(setUsers).catch((e) => setError(e.message));
  }

  useEffect(load, []);

  async function changeRole(userId: string, role: string) {
    setBusyId(userId);
    setError(null);
    try {
      await api.updateUserRole(userId, role);
      load();
    } catch (e: any) {
      setError(e.message);
    } finally {
      setBusyId(null);
    }
  }

  async function toggleActive(userId: string, isActive: boolean) {
    setBusyId(userId);
    setError(null);
    try {
      await api.updateUserActive(userId, !isActive);
      load();
    } catch (e: any) {
      setError(e.message);
    } finally {
      setBusyId(null);
    }
  }

  return (
    <div className="card mb-6">
      <h2 className="font-semibold mb-3">User Management</h2>
      {error && <div className="text-bad text-sm mb-3">{error}</div>}
      <table className="w-full text-sm">
        <thead className="text-white/50 text-xs uppercase">
          <tr>
            <th className="text-left py-2">Name</th>
            <th className="text-left py-2">Email</th>
            <th className="text-left py-2">Role</th>
            <th className="text-left py-2">Status</th>
            <th className="text-left py-2"></th>
          </tr>
        </thead>
        <tbody>
          {users.map((u) => (
            <tr key={u.id} className="border-t border-white/5">
              <td className="py-2">{u.full_name}</td>
              <td className="py-2 text-white/60">{u.email}</td>
              <td className="py-2">
                <select
                  className="bg-white/5 border border-white/10 rounded-lg px-2 py-1 text-xs"
                  value={u.role}
                  disabled={busyId === u.id}
                  onChange={(e) => changeRole(u.id, e.target.value)}
                >
                  {ROLES.map((r) => (
                    <option key={r} value={r}>
                      {r}
                    </option>
                  ))}
                </select>
              </td>
              <td className="py-2">
                <span className={`badge ${u.is_active ? "badge-low" : "badge-critical"}`}>
                  {u.is_active ? "Active" : "Inactive"}
                </span>
              </td>
              <td className="py-2">
                <button
                  className="btn btn-secondary text-xs"
                  disabled={busyId === u.id || u.id === currentUserId}
                  onClick={() => toggleActive(u.id, u.is_active)}
                >
                  {u.is_active ? "Deactivate" : "Activate"}
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function SystemConfigCard() {
  const [config, setConfig] = useState<any>(null);
  const [auditStatus, setAuditStatus] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  const [checking, setChecking] = useState(false);

  useEffect(() => {
    api.systemSettings().then(setConfig).catch((e) => setError(e.message));
  }, []);

  async function runAuditCheck() {
    setChecking(true);
    setError(null);
    try {
      setAuditStatus(await api.verifyAuditChain());
    } catch (e: any) {
      setError(e.message);
    } finally {
      setChecking(false);
    }
  }

  if (error) return <div className="card text-bad text-sm">{error}</div>;
  if (!config) return null;

  return (
    <div className="card">
      <h2 className="font-semibold mb-3">System Configuration</h2>
      <div className="grid grid-cols-2 gap-x-8 gap-y-2 text-sm text-white/70 mb-4">
        <div className="flex justify-between"><span>Environment</span><span>{config.environment}</span></div>
        <div className="flex justify-between"><span>Rate limit</span><span>{config.rate_limiting.requests_per_minute}/min</span></div>
        <div className="flex justify-between"><span>LLM provider</span><span>{config.rag.llm_provider}</span></div>
        <div className="flex justify-between"><span>Embedding provider</span><span>{config.rag.embedding_provider}</span></div>
        <div className="flex justify-between"><span>Anthropic key configured</span><span>{config.rag.anthropic_api_key_configured ? "Yes" : "No"}</span></div>
        <div className="flex justify-between"><span>OpenAI key configured</span><span>{config.rag.openai_api_key_configured ? "Yes" : "No"}</span></div>
        <div className="flex justify-between"><span>Anomaly contamination</span><span>{config.anomaly_detection.contamination}</span></div>
        <div className="flex justify-between"><span>Drift WARNING / CRITICAL PSI</span><span>{config.drift_detection.psi_warning_threshold} / {config.drift_detection.psi_critical_threshold}</span></div>
        <div className="flex justify-between"><span>Daily drift check (UTC)</span><span>{config.celery.daily_drift_check_utc}</span></div>
        <div className="flex justify-between"><span>Weekly model retrain (UTC)</span><span>{config.celery.weekly_model_retrain_utc}</span></div>
      </div>

      <div className="pt-3 border-t border-white/10">
        <div className="flex items-center justify-between">
          <span className="text-sm text-white/60">Audit trail integrity</span>
          <button className="btn btn-secondary text-xs" onClick={runAuditCheck} disabled={checking}>
            {checking ? "Checking..." : "Verify hash chain"}
          </button>
        </div>
        {auditStatus && (
          <div className={`mt-2 text-sm ${auditStatus.chain_valid ? "text-good" : "text-bad"}`}>
            {auditStatus.chain_valid ? "Chain valid — no tampering detected." : `Chain broken at record ${auditStatus.broken_at_id}`}
          </div>
        )}
      </div>
    </div>
  );
}

export default function SettingsPage() {
  const { ready, user } = useAuthGuard();

  if (!ready || !user) return null;

  return (
    <div>
      <h1 className="text-2xl font-bold mb-6">Settings</h1>
      <ProfileCard user={user} />
      {user.role === "ADMIN" && <UserManagementCard currentUserId={user.id} />}
      {user.role === "ADMIN" && <SystemConfigCard />}
    </div>
  );
}
