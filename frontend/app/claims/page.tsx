"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { api } from "@/lib/api";
import { useAuthGuard } from "@/lib/useAuthGuard";

export default function ClaimsPage() {
  const { ready } = useAuthGuard();
  const [claims, setClaims] = useState<any[]>([]);
  const [status, setStatus] = useState("");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!ready) return;
    const params = status ? { status } : {};
    api.listClaims(params).then(setClaims).catch((e) => setError(e.message));
  }, [ready, status]);

  if (!ready) return null;

  return (
    <div>
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-2xl font-bold">Claims</h1>
        <select
          className="bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-sm"
          value={status}
          onChange={(e) => setStatus(e.target.value)}
        >
          <option value="">All statuses</option>
          <option value="SUBMITTED">Submitted</option>
          <option value="PAID">Paid</option>
          <option value="DENIED">Denied</option>
          <option value="RECOVERED">Recovered</option>
        </select>
      </div>

      {error && <div className="text-bad mb-4">{error}</div>}

      <div className="card p-0 overflow-hidden">
        <table className="w-full text-sm">
          <thead className="bg-white/5 text-white/50 text-xs uppercase">
            <tr>
              <th className="text-left p-3">Claim #</th>
              <th className="text-left p-3">Amount</th>
              <th className="text-left p-3">Status</th>
              <th className="text-left p-3">Auth</th>
              <th className="text-left p-3">Documentation</th>
              <th className="p-3"></th>
            </tr>
          </thead>
          <tbody>
            {claims.map((c) => (
              <tr key={c.id} className="border-t border-white/5 hover:bg-white/5">
                <td className="p-3">{c.claim_number}</td>
                <td className="p-3">${Number(c.claim_amount).toLocaleString()}</td>
                <td className="p-3">{c.status}</td>
                <td className="p-3">{c.authorization_status}</td>
                <td className="p-3">{c.documentation_completeness}%</td>
                <td className="p-3">
                  <Link href={`/claims/${c.id}`} className="text-accent hover:underline">
                    Open →
                  </Link>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        {claims.length === 0 && <div className="p-6 text-center text-white/40">No claims found.</div>}
      </div>
    </div>
  );
}
