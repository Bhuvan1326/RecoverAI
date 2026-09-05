"use client";

import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import { useAuthGuard } from "@/lib/useAuthGuard";

function Metric({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="card">
      <div className="text-xs text-white/50 mb-1">{label}</div>
      <div className="text-2xl font-semibold">{value}</div>
    </div>
  );
}

function fmtMoney(n: number) {
  return `$${n.toLocaleString(undefined, { maximumFractionDigits: 0 })}`;
}

export default function DashboardPage() {
  const { ready } = useAuthGuard();
  const [metrics, setMetrics] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!ready) return;
    api.dashboardMetrics().then(setMetrics).catch((e) => setError(e.message));
  }, [ready]);

  if (!ready) return null;
  if (error) return <div className="text-bad">{error}</div>;
  if (!metrics) return <div className="text-white/50">Loading...</div>;

  return (
    <div>
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-2xl font-bold">Revenue at Risk</h1>
        <span className="simulated-badge">{metrics.label}</span>
      </div>

      <div className="grid grid-cols-4 gap-4 mb-6">
        <Metric label="Total Claims" value={metrics.total_claims} />
        <Metric label="Revenue at Risk" value={fmtMoney(metrics.revenue_at_risk)} />
        <Metric label="Preventable Revenue" value={fmtMoney(metrics.preventable_revenue)} />
        <Metric label="Recoverable Revenue" value={fmtMoney(metrics.recoverable_revenue)} />
      </div>
      <div className="grid grid-cols-4 gap-4 mb-6">
        <Metric label="Expected Recovery" value={fmtMoney(metrics.expected_recovery)} />
        <Metric label="Denial Rate" value={`${(metrics.denial_rate * 100).toFixed(1)}%`} />
        <Metric label="Clean Claim Rate" value={`${(metrics.clean_claim_rate * 100).toFixed(1)}%`} />
        <Metric label="High Priority Claims" value={metrics.high_priority_claims} />
      </div>

      <div className="card text-sm text-white/60">
        This is a decision-support demonstration built entirely on synthetic and public data. No
        real PHI is used, and RecoverAI never submits claims or appeals autonomously — every
        consequential action requires human approval.
      </div>
    </div>
  );
}
