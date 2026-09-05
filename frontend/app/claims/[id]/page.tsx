"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import { api } from "@/lib/api";
import { useAuthGuard } from "@/lib/useAuthGuard";

function riskBadgeClass(cat: string) {
  return (
    {
      CRITICAL: "badge-critical",
      HIGH: "badge-high",
      MEDIUM: "badge-medium",
      LOW: "badge-low",
    }[cat] || "badge-low"
  );
}

export default function ClaimDetailPage() {
  const { ready } = useAuthGuard();
  const params = useParams();
  const claimId = params.id as string;

  const [claim, setClaim] = useState<any>(null);
  const [score, setScore] = useState<any>(null);
  const [explanation, setExplanation] = useState<any>(null);
  const [validation, setValidation] = useState<any>(null);
  const [simResult, setSimResult] = useState<any>(null);
  const [appealDraft, setAppealDraft] = useState<any>(null);
  const [anomaly, setAnomaly] = useState<any>(null);
  const [appealSuccess, setAppealSuccess] = useState<any>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  // what-if form state
  const [whatIfAuth, setWhatIfAuth] = useState("PRESENT");
  const [whatIfDoc, setWhatIfDoc] = useState(95);

  useEffect(() => {
    if (!ready || !claimId) return;
    api.getClaim(claimId).then((c) => {
      setClaim(c);
      setWhatIfAuth(c.authorization_status);
      setWhatIfDoc(Number(c.documentation_completeness));
    }).catch((e) => setError(e.message));
  }, [ready, claimId]);

  async function runScore() {
    setBusy("score");
    setError(null);
    try {
      setScore(await api.scoreClaim(claimId));
    } catch (e: any) {
      setError(e.message);
    } finally {
      setBusy(null);
    }
  }

  async function runExplain() {
    setBusy("explain");
    setError(null);
    try {
      setExplanation(await api.explainClaim(claimId));
    } catch (e: any) {
      setError(e.message);
    } finally {
      setBusy(null);
    }
  }

  async function runValidate() {
    setBusy("validate");
    setError(null);
    try {
      setValidation(await api.validateClaim(claimId));
    } catch (e: any) {
      setError(e.message);
    } finally {
      setBusy(null);
    }
  }

  async function runWhatIf() {
    setBusy("whatif");
    setError(null);
    try {
      setSimResult(
        await api.simulateClaim(claimId, {
          authorization_status: whatIfAuth,
          documentation_completeness: whatIfDoc,
        })
      );
    } catch (e: any) {
      setError(e.message);
    } finally {
      setBusy(null);
    }
  }

  async function runAnomaly() {
    setBusy("anomaly");
    setError(null);
    try {
      setAnomaly(await api.scoreAnomaly(claimId));
    } catch (e: any) {
      setError(e.message);
    } finally {
      setBusy(null);
    }
  }

  async function runAppealSuccess() {
    setBusy("appealSuccess");
    setError(null);
    try {
      setAppealSuccess(await api.scoreAppealSuccess(claimId));
    } catch (e: any) {
      setError(e.message);
    } finally {
      setBusy(null);
    }
  }

  async function runAppealDraft() {
    setBusy("appeal");
    setError(null);
    try {
      setAppealDraft(await api.draftAppeal(claimId));
    } catch (e: any) {
      setError(e.message);
    } finally {
      setBusy(null);
    }
  }

  async function decide(decision: "approve" | "reject") {
    if (!appealDraft?.workflow_action_id) return;
    setBusy("decide");
    try {
      const fn = decision === "approve" ? api.approveAction : api.rejectAction;
      const result = await fn(appealDraft.workflow_action_id, `${decision} via claim workbench`);
      setAppealDraft({ ...appealDraft, status: result.status });
    } catch (e: any) {
      setError(e.message);
    } finally {
      setBusy(null);
    }
  }

  if (!ready || !claim) return <div className="text-white/50">Loading...</div>;

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold">{claim.claim_number}</h1>
        <p className="text-white/50 text-sm">
          ${Number(claim.claim_amount).toLocaleString()} · {claim.status} · Auth: {claim.authorization_status} ·
          Elig: {claim.eligibility_status} · Docs: {claim.documentation_completeness}%
        </p>
      </div>

      {error && <div className="text-bad">{error}</div>}

      <div className="grid grid-cols-2 gap-6">
        {/* Denial risk + SHAP */}
        <div className="card">
          <div className="flex items-center justify-between mb-3">
            <h2 className="font-semibold">Denial Risk</h2>
            <div className="flex gap-2">
              <button className="btn btn-secondary text-xs" onClick={runScore} disabled={busy === "score"}>
                {busy === "score" ? "Scoring..." : "Score claim"}
              </button>
              <button className="btn btn-secondary text-xs" onClick={runExplain} disabled={busy === "explain"}>
                {busy === "explain" ? "Explaining..." : "Explain (SHAP)"}
              </button>
            </div>
          </div>
          {score && (
            <div className="mb-3">
              <span className={`badge ${riskBadgeClass(score.risk_category)}`}>{score.risk_category}</span>
              <span className="ml-2 text-lg font-semibold">{(score.denial_probability * 100).toFixed(1)}%</span>
              <div className="text-xs text-white/40 mt-1">model: {score.model_version}</div>
            </div>
          )}
          {explanation && (
            <div className="text-sm">
              <div className="text-white/60 mb-1">Top risk-increasing factors:</div>
              <ul className="space-y-1 mb-2">
                {explanation.top_positive_factors?.map((f: any) => (
                  <li key={f.feature} className="flex justify-between">
                    <span className="text-white/70">{f.feature.replace(/^(num__|cat__)/, "")}</span>
                    <span className="text-bad">+{f.contribution}</span>
                  </li>
                ))}
              </ul>
              <div className="text-white/60 mb-1">Top risk-reducing factors:</div>
              <ul className="space-y-1 mb-3">
                {explanation.top_negative_factors?.map((f: any) => (
                  <li key={f.feature} className="flex justify-between">
                    <span className="text-white/70">{f.feature.replace(/^(num__|cat__)/, "")}</span>
                    <span className="text-good">{f.contribution}</span>
                  </li>
                ))}
              </ul>
              {explanation.denial_reason && (
                <div className="mb-3 pt-2 border-t border-white/10">
                  <div className="text-white/60 mb-1">Predicted denial reason:</div>
                  <div className="flex items-center gap-2">
                    <span className="font-medium">{explanation.denial_reason.predicted_reason?.replace(/_/g, " ")}</span>
                    <span className="badge badge-medium">{(explanation.denial_reason.confidence * 100).toFixed(0)}%</span>
                    <span className="text-xs text-white/40">
                      {explanation.denial_reason.source === "rule"
                        ? "rule-based"
                        : explanation.denial_reason.source === "ml"
                        ? `ML (${explanation.denial_reason.model_version})`
                        : "heuristic"}
                    </span>
                  </div>
                  {explanation.denial_reason.alternatives?.length > 0 && (
                    <div className="text-xs text-white/40 mt-1">
                      Alternatives: {explanation.denial_reason.alternatives.map((a: any) => `${a.reason.replace(/_/g, " ")} (${(a.confidence * 100).toFixed(0)}%)`).join(", ")}
                    </div>
                  )}
                </div>
              )}
              <div className="text-xs text-white/40 italic">{explanation.disclaimer}</div>
            </div>
          )}
        </div>

        {/* Validator */}
        <div className="card">
          <div className="flex items-center justify-between mb-3">
            <h2 className="font-semibold">Pre-Submission Validator</h2>
            <button className="btn btn-secondary text-xs" onClick={runValidate} disabled={busy === "validate"}>
              {busy === "validate" ? "Validating..." : "Run validator"}
            </button>
          </div>
          {validation && (
            <div>
              <div className="text-2xl font-semibold mb-2">{validation.readiness_score}/100</div>
              <ul className="text-sm space-y-1">
                {validation.checks.map((c: any) => (
                  <li key={c.name} className="flex justify-between">
                    <span className="text-white/70">{c.name.replace(/_/g, " ")}</span>
                    <span
                      className={
                        c.status === "PASS" ? "text-good" : c.status === "WARNING" ? "text-warn" : "text-bad"
                      }
                    >
                      {c.status}
                    </span>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>

        {/* What-if simulator */}
        <div className="card">
          <h2 className="font-semibold mb-3">What-If Simulator</h2>
          <div className="flex gap-3 mb-3 items-end">
            <div>
              <label className="text-xs text-white/50 block mb-1">Authorization</label>
              <select
                className="bg-white/5 border border-white/10 rounded-lg px-2 py-1 text-sm"
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
                className="bg-white/5 border border-white/10 rounded-lg px-2 py-1 text-sm w-20"
                value={whatIfDoc}
                onChange={(e) => setWhatIfDoc(Number(e.target.value))}
              />
            </div>
            <button className="btn btn-primary text-xs" onClick={runWhatIf} disabled={busy === "whatif"}>
              {busy === "whatif" ? "Simulating..." : "Recalculate"}
            </button>
          </div>
          {simResult && (
            <div className="text-sm">
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

        {/* Anomaly detection */}
        <div className="card">
          <div className="flex items-center justify-between mb-3">
            <h2 className="font-semibold">Anomaly Detection</h2>
            <button className="btn btn-secondary text-xs" onClick={runAnomaly} disabled={busy === "anomaly"}>
              {busy === "anomaly" ? "Scoring..." : "Run anomaly check"}
            </button>
          </div>
          {anomaly && (
            <div className="text-sm">
              <div className="flex items-center gap-2 mb-2">
                <span className={`badge ${riskBadgeClass(anomaly.severity === "HIGH" ? "CRITICAL" : anomaly.severity === "MEDIUM" ? "HIGH" : anomaly.severity === "LOW" ? "MEDIUM" : "LOW")}`}>
                  {anomaly.severity}
                </span>
                <span className="text-lg font-semibold">{anomaly.anomaly_score}/100</span>
              </div>
              {anomaly.contributing_features?.length > 0 ? (
                <div>
                  <div className="text-white/60 mb-1">Potentially unusual:</div>
                  <ul className="space-y-1">
                    {anomaly.contributing_features.map((f: any) => (
                      <li key={f.feature} className="text-white/70">
                        • {f.feature.replace(/_/g, " ")} (z={f.z_score})
                      </li>
                    ))}
                  </ul>
                </div>
              ) : (
                <div className="text-white/40">No strongly unusual features detected.</div>
              )}
              <div className="text-xs text-white/40 italic mt-2">{anomaly.disclaimer}</div>
            </div>
          )}
        </div>

        {/* Appeal success / recovery */}
        <div className="card">
          <div className="flex items-center justify-between mb-3">
            <h2 className="font-semibold">Appeal Success & Recovery</h2>
            <button className="btn btn-secondary text-xs" onClick={runAppealSuccess} disabled={busy === "appealSuccess"}>
              {busy === "appealSuccess" ? "Scoring..." : "Score appeal success"}
            </button>
          </div>
          {appealSuccess && (
            <div className="text-sm">
              <div className="flex items-center gap-2 mb-2">
                <span className="text-lg font-semibold">{(appealSuccess.appeal_success_probability * 100).toFixed(0)}%</span>
                <span className="badge badge-medium">{appealSuccess.risk_category?.replace(/_/g, " ")}</span>
              </div>
              <div className="flex justify-between text-white/70">
                <span>Source</span>
                <span>{appealSuccess.source === "ml" ? `ML model (${appealSuccess.model_version})` : "Heuristic baseline"}</span>
              </div>
              <div className="flex justify-between text-white/70">
                <span>Calibrated</span>
                <span>{appealSuccess.calibrated ? "Yes (isotonic)" : "No"}</span>
              </div>
              {appealSuccess.disclaimer && (
                <div className="text-xs text-white/40 italic mt-2">{appealSuccess.disclaimer}</div>
              )}
            </div>
          )}
        </div>

        {/* Appeal copilot */}
        <div className="card">
          <div className="flex items-center justify-between mb-3">
            <h2 className="font-semibold">Appeal Copilot (Agentic)</h2>
            <button className="btn btn-secondary text-xs" onClick={runAppealDraft} disabled={busy === "appeal"}>
              {busy === "appeal" ? "Investigating..." : "Investigate & draft"}
            </button>
          </div>
          {appealDraft?.blocked && (
            <div className="text-warn text-sm">
              Blocked: {appealDraft.reason}
              <div className="text-xs text-white/40 mt-1">
                Missing: {appealDraft.evidence_completeness?.missing_evidence?.join(", ")}
              </div>
            </div>
          )}
          {appealDraft && !appealDraft.blocked && (
            <div className="text-sm">
              <pre className="whitespace-pre-wrap text-xs bg-black/30 p-3 rounded-lg mb-3 max-h-64 overflow-y-auto">
                {appealDraft.draft_text}
              </pre>
              <div className="text-xs text-white/50 mb-2">{appealDraft.citations?.length || 0} citation(s) retrieved from payer policy corpus</div>
              <div className="flex items-center gap-2">
                <span className="badge badge-medium">{appealDraft.status || "PENDING_APPROVAL"}</span>
                {(!appealDraft.status || appealDraft.status === "PENDING_APPROVAL") && (
                  <>
                    <button className="btn btn-primary text-xs" onClick={() => decide("approve")} disabled={busy === "decide"}>
                      Approve
                    </button>
                    <button className="btn btn-danger text-xs" onClick={() => decide("reject")} disabled={busy === "decide"}>
                      Reject
                    </button>
                  </>
                )}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
