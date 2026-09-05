"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { api } from "@/lib/api";
import { useAuthGuard } from "@/lib/useAuthGuard";

function ClaimWhatIfSection() {
  const [claims, setClaims] = useState<any[]>([]);
  const [selectedId, setSelectedId] = useState<string>("");
  const [selectedClaim, setSelectedClaim] = useState<any>(null);
  const [whatIfAuth, setWhatIfAuth] = useState("PRESENT");
  const [whatIfDoc, setWhatIfDoc] = useState(95);
  const [simResult, setSimResult] = useState<any>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.listClaims({ limit: "100" }).then(setClaims).catch((e) => setError(e.message));
  }, []);

  async function selectClaim(id: string) {
    setSelectedId(id);
    setSimResult(null);
    setError(null);
    if (!id) {
      setSelectedClaim(null);
      return;
    }
    try {
      const claim = await api.getClaim(id);
      setSelectedClaim(claim);
      setWhatIfAuth(claim.authorization_status);
      setWhatIfDoc(Number(claim.documentation_completeness));
    } catch (e: any) {
      setError(e.message);
    }
  }

  async function runWhatIf() {
    if (!selectedId) return;
    setBusy(true);
    setError(null);
    try {
      setSimResult(
        await api.simulateClaim(selectedId, {
          authorization_status: whatIfAuth,
          documentation_completeness: whatIfDoc,
        })
      );
    } catch (e: any) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="card mb-8">
      <h2 className="font-semibold mb-1">Claim What-If Simulator</h2>
      <p className="text-xs text-white/40 mb-4">
        Pick any claim, change authorization/documentation, and see how the denial-risk
        model&apos;s prediction would have changed. Reflects a learned association, not a
        proven causal effect.
      </p>

      <div className="flex gap-3 mb-4 items-end flex-wrap">
        <div>
          <label className="text-xs text-white/50 block mb-1">Claim</label>
          <select
            className="bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-sm min-w-[220px]"
            value={selectedId}
            onChange={(e) => selectClaim(e.target.value)}
          >
            <option value="">Select a claim...</option>
            {claims.map((c) => (
              <option key={c.id} value={c.id}>
                {c.claim_number} — ${Number(c.claim_amount).toLocaleString()} ({c.status})
              </option>
            ))}
          </select>
        </div>

        {selectedClaim && (
          <>
            <div>
              <label className="text-xs text-white/50 block mb-1">Authorization</label>
              <select
                className="bg-white/5 border border-white/10 rounded-lg px-2 py-2 text-sm"
                value={whatIfAuth}
                onChange={(e) => setWhatIfAuth(e.target.value)}
              >
                <option value="PRESENT">Present</option>
                <option value="MISSING">Missing</option>
              </select>
            </div>
            <div>
              <label className="text-xs text-white/50 block mb-1">Documentation %</label>
              <input
                type="number"
                min={0}
                max={100}
                className="bg-white/5 border border-white/10 rounded-lg px-2 py-2 text-sm w-24"
                value={whatIfDoc}
                onChange={(e) => setWhatIfDoc(Number(e.target.value))}
              />
            </div>
            <button className="btn btn-primary text-sm" onClick={runWhatIf} disabled={busy}>
              {busy ? "Simulating..." : "Recalculate risk"}
            </button>
            <Link href={`/claims/${selectedId}`} className="text-accent text-sm hover:underline">
              Open full workbench →
            </Link>
          </>
        )}
      </div>

      {error && <div className="text-bad text-sm mb-3">{error}</div>}

      {selectedClaim && !simResult && (
        <div className="text-sm text-white/50">
          Current: auth {selectedClaim.authorization_status}, documentation{" "}
          {selectedClaim.documentation_completeness}%. Adjust above and recalculate.
        </div>
      )}

      {simResult && (
        <div className="text-sm max-w-md">
          <div className="flex justify-between">
            <span>Original risk</span>
            <span>{(simResult.original_risk * 100).toFixed(1)}%</span>
          </div>
          <div className="flex justify-between">
            <span>Simulated risk</span>
            <span>{(simResult.simulated_risk * 100).toFixed(1)}%</span>
          </div>
          <div className="flex justify-between font-semibold">
            <span>Difference</span>
            <span className={simResult.risk_difference > 0 ? "text-good" : "text-bad"}>
              {(simResult.risk_difference * 100).toFixed(1)} pts
            </span>
          </div>
          <div className="text-xs text-white/40 italic mt-2">{simResult.disclaimer}</div>
        </div>
      )}
    </div>
  );
}

function tierColor(name: string, isBest: boolean) {
  if (isBest) return "border-good/50 bg-good/5";
  return "border-white/10";
}

function RecoveryStrategySimulatorSection() {
  const [staffHours, setStaffHours] = useState(40);
  const [result, setResult] = useState<any>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function run() {
    setBusy(true);
    setError(null);
    try {
      setResult(await api.simulateRecoveryStrategy(staffHours));
    } catch (e: any) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  }

  const strategyLabels: Record<string, string> = {
    highest_claim_amount: "Highest Claim Amount First",
    highest_appeal_probability: "Highest Appeal Probability First",
    highest_expected_recovery: "Highest Expected Recovery First (live queue default)",
  };

  return (
    <div className="card">
      <h2 className="font-semibold mb-1">Recovery Strategy Simulator</h2>
      <p className="text-xs text-white/40 mb-4">
        Compares prioritization strategies against a real staff-capacity budget, using
        the actual current set of denied claims and the same effort estimates the
        Next-Best-Action engine produces. Values are expected, not guaranteed, outcomes.
      </p>

      <div className="flex gap-3 mb-5 items-end">
        <div>
          <label className="text-xs text-white/50 block mb-1">Available staff hours</label>
          <input
            type="number"
            min={1}
            className="bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-sm w-28"
            value={staffHours}
            onChange={(e) => setStaffHours(Number(e.target.value))}
          />
        </div>
        <button className="btn btn-primary text-sm" onClick={run} disabled={busy}>
          {busy ? "Simulating..." : "Run simulation"}
        </button>
      </div>

      {error && <div className="text-bad text-sm mb-3">{error}</div>}

      {result && result.claims_available === 0 && (
        <div className="text-white/40 text-sm">No denied claims available to simulate against.</div>
      )}

      {result && result.claims_available > 0 && (
        <div>
          <div className="text-xs text-white/40 mb-3">
            {result.claims_available} denied claims available · {result.label}
          </div>
          <div className="grid grid-cols-3 gap-4">
            {result.strategies.map((s: any) => {
              const isBest = s.strategy === result.best_strategy;
              return (
                <div key={s.strategy} className={`rounded-xl border p-4 ${tierColor(s.strategy, isBest)}`}>
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm font-medium">{strategyLabels[s.strategy] || s.strategy}</span>
                    {isBest && <span className="badge badge-low">BEST</span>}
                  </div>
                  <div className="text-2xl font-semibold mb-1">
                    ${s.expected_recovery_captured.toLocaleString()}
                  </div>
                  <div className="text-xs text-white/50 space-y-1">
                    <div className="flex justify-between">
                      <span>Claims worked</span>
                      <span>
                        {s.claims_processed} / {s.claims_available}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span>Staff hours used</span>
                      <span>
                        {s.staff_hours_used} / {s.staff_hours_budget}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span>Recovery yield</span>
                      <span>{(s.recovery_yield * 100).toFixed(1)}%</span>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}

export default function WhatIfSimulatorPage() {
  const { ready } = useAuthGuard();
  if (!ready) return null;

  return (
    <div>
      <h1 className="text-2xl font-bold mb-6">What-If &amp; Recovery Simulator</h1>
      <ClaimWhatIfSection />
      <RecoveryStrategySimulatorSection />
    </div>
  );
}
