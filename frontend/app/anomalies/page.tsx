"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { api } from "@/lib/api";
import { useAuthGuard } from "@/lib/useAuthGuard";

function severityClass(s: string) {
  return (
    { HIGH: "badge-critical", MEDIUM: "badge-high", LOW: "badge-medium", NORMAL: "badge-low" }[s] || "badge-low"
  );
}

export default function AnomaliesPage() {
  const { ready } = useAuthGuard();
  const [data, setData] = useState<any>(null);
  const [severity, setSeverity] = useState("");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!ready) return;
    const params = severity ? { severity } : {};
    api.anomalyAnalytics(params).then(setData).catch((e) => setError(e.message));
  }, [ready, severity]);

  if (!ready) return null;

  return (
    <div>
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-2xl font-bold">Anomaly Detection</h1>
        <select
          className="bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-sm"
          value={severity}
          onChange={(e) => setSeverity(e.target.value)}
        >
          <option value="">All severities</option>
          <option value="HIGH">High</option>
          <option value="MEDIUM">Medium</option>
          <option value="LOW">Low</option>
          <option value="NORMAL">Normal</option>
        </select>
      </div>

      {error && <div className="text-bad mb-4">{error}</div>}
      {data?.error && (
        <div className="card text-warn mb-4">
          {data.error}
          <div className="text-xs text-white/40 mt-1">Run: python -m ml.training.train_anomaly_model</div>
        </div>
      )}

      {data && !data.error && (
        <>
          <div className="grid grid-cols-3 gap-4 mb-6">
            <div className="card">
              <div className="text-xs text-white/50 mb-1">Claims Scored</div>
              <div className="text-2xl font-semibold">{data.total_claims_scored}</div>
            </div>
            <div className="card">
              <div className="text-xs text-white/50 mb-1">Anomalies Flagged</div>
              <div className="text-2xl font-semibold">
                {data.anomaly_count} <span className="text-sm text-white/40">({data.anomaly_percentage}%)</span>
              </div>
            </div>
            <div className="card">
              <div className="text-xs text-white/50 mb-1">Severity Distribution</div>
              <div className="text-sm">
                {Object.entries(data.severity_distribution || {}).map(([k, v]) => (
                  <span key={k} className="mr-3">
                    {k}: {v as any}
                  </span>
                ))}
              </div>
            </div>
          </div>

          <div className="card p-0 overflow-hidden">
            <table className="w-full text-sm">
              <thead className="bg-white/5 text-white/50 text-xs uppercase">
                <tr>
                  <th className="text-left p-3">Claim #</th>
                  <th className="text-left p-3">Score</th>
                  <th className="text-left p-3">Severity</th>
                  <th className="text-left p-3">Unusual Features</th>
                  <th className="p-3"></th>
                </tr>
              </thead>
              <tbody>
                {data.items?.map((r: any) => (
                  <tr key={r.claim_id} className="border-t border-white/5">
                    <td className="p-3">{r.claim_number}</td>
                    <td className="p-3">{r.anomaly_score}</td>
                    <td className="p-3">
                      <span className={`badge ${severityClass(r.severity)}`}>{r.severity}</span>
                    </td>
                    <td className="p-3 text-white/60">{r.contributing_features?.map((f: any) => f.feature).join(", ") || "—"}</td>
                    <td className="p-3">
                      <Link href={`/claims/${r.claim_id}`} className="text-accent hover:underline">
                        Open →
                      </Link>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  );
}
