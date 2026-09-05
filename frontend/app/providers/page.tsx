"use client";

import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import { useAuthGuard } from "@/lib/useAuthGuard";

export default function ProviderIntelligencePage() {
  const { ready } = useAuthGuard();
  const [data, setData] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!ready) return;
    api.providerIntelligence().then(setData).catch((e) => setError(e.message));
  }, [ready]);

  if (!ready) return null;

  return (
    <div>
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-2xl font-bold">Provider Intelligence</h1>
        {data && <span className="simulated-badge">{data.label}</span>}
      </div>
      {error && <div className="text-bad">{error}</div>}
      <div className="card p-0 overflow-hidden">
        <table className="w-full text-sm">
          <thead className="bg-white/5 text-white/50 text-xs uppercase">
            <tr>
              <th className="text-left p-3">Provider</th>
              <th className="text-left p-3">Claim Volume</th>
              <th className="text-left p-3">Denial Rate</th>
              <th className="text-left p-3">Top Denial Reasons</th>
              <th className="text-left p-3">Revenue at Risk</th>
            </tr>
          </thead>
          <tbody>
            {data?.providers?.map((p: any) => (
              <tr key={p.provider_id} className="border-t border-white/5">
                <td className="p-3">{p.provider_name}</td>
                <td className="p-3">{p.claim_volume}</td>
                <td className="p-3">{(p.denial_rate * 100).toFixed(1)}%</td>
                <td className="p-3 text-white/60">
                  {p.top_denial_reasons.map((r: any) => r[0]).join(", ")}
                </td>
                <td className="p-3">${p.revenue_at_risk.toLocaleString()}</td>
              </tr>
            ))}
          </tbody>
        </table>
        {(!data?.providers || data.providers.length === 0) && (
          <div className="p-6 text-center text-white/40">No provider data available.</div>
        )}
      </div>
    </div>
  );
}
