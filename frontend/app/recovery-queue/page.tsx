"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { api } from "@/lib/api";
import { useAuthGuard } from "@/lib/useAuthGuard";

function tierClass(tier: string) {
  return (
    { CRITICAL: "badge-critical", HIGH: "badge-high", MEDIUM: "badge-medium", LOW: "badge-low" }[tier] || "badge-low"
  );
}

export default function RecoveryQueuePage() {
  const { ready } = useAuthGuard();
  const [items, setItems] = useState<any[]>([]);
  const [tier, setTier] = useState("");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!ready) return;
    const params = tier ? { tier } : {};
    api
      .recoveryQueue(params)
      .then((r) => setItems(r.items))
      .catch((e) => setError(e.message));
  }, [ready, tier]);

  if (!ready) return null;

  return (
    <div>
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-2xl font-bold">Recovery Priority Queue</h1>
        <select
          className="bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-sm"
          value={tier}
          onChange={(e) => setTier(e.target.value)}
        >
          <option value="">All tiers</option>
          <option value="CRITICAL">Critical</option>
          <option value="HIGH">High</option>
          <option value="MEDIUM">Medium</option>
          <option value="LOW">Low</option>
        </select>
      </div>

      {error && <div className="text-bad mb-4">{error}</div>}

      <div className="card p-0 overflow-hidden">
        <table className="w-full text-sm">
          <thead className="bg-white/5 text-white/50 text-xs uppercase">
            <tr>
              <th className="text-left p-3">Claim #</th>
              <th className="text-left p-3">Amount</th>
              <th className="text-left p-3">Denial Reason</th>
              <th className="text-left p-3">Appeal %</th>
              <th className="text-left p-3">Expected Recovery</th>
              <th className="text-left p-3">Tier</th>
              <th className="text-left p-3">Next Best Action</th>
              <th className="p-3"></th>
            </tr>
          </thead>
          <tbody>
            {items.map((c) => (
              <tr key={c.claim_id} className="border-t border-white/5 hover:bg-white/5">
                <td className="p-3">{c.claim_number}</td>
                <td className="p-3">${c.claim_amount.toLocaleString()}</td>
                <td className="p-3">{c.denial_reason?.replace(/_/g, " ")}</td>
                <td className="p-3">{(c.appeal_success_probability * 100).toFixed(0)}%</td>
                <td className="p-3">${c.expected_recovery.toLocaleString()}</td>
                <td className="p-3">
                  <span className={`badge ${tierClass(c.priority_tier)}`}>{c.priority_tier}</span>
                </td>
                <td className="p-3">{c.recommended_action?.replace(/_/g, " ")}</td>
                <td className="p-3">
                  <Link href={`/claims/${c.claim_id}`} className="text-accent hover:underline">
                    Open →
                  </Link>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        {items.length === 0 && <div className="p-6 text-center text-white/40">No denied claims in queue.</div>}
      </div>
    </div>
  );
}
