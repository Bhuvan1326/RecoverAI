"use client";

import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import { useAuthGuard } from "@/lib/useAuthGuard";

export default function AuditLogPage() {
  const { ready } = useAuthGuard();
  const [logs, setLogs] = useState<any[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!ready) return;
    api
      .auditLogs()
      .then(setLogs)
      .catch((e) => setError(e.message + " (audit logs require ADMIN or REVIEWER role)"));
  }, [ready]);

  if (!ready) return null;

  return (
    <div>
      <h1 className="text-2xl font-bold mb-6">Audit Trail</h1>
      {error && <div className="text-bad mb-4">{error}</div>}
      <div className="space-y-2">
        {logs.map((log) => (
          <div key={log.id} className="card py-3">
            <div className="flex justify-between text-xs text-white/40 mb-1">
              <span>{new Date(log.created_at).toLocaleString()}</span>
              <span className="font-mono">{log.hash?.slice(0, 12)}...</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="badge badge-medium">{log.actor_type}</span>
              <span className="font-medium">{log.event_type}</span>
              {log.actor_id && <span className="text-white/40 text-xs">by {log.actor_id.slice(0, 8)}</span>}
            </div>
            {Object.keys(log.payload || {}).length > 0 && (
              <pre className="text-xs text-white/50 mt-2 overflow-x-auto">{JSON.stringify(log.payload, null, 2)}</pre>
            )}
          </div>
        ))}
        {logs.length === 0 && !error && <div className="text-white/40">No audit events yet.</div>}
      </div>
    </div>
  );
}
