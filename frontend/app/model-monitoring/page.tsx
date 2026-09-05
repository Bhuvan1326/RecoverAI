"use client";

import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import { useAuthGuard } from "@/lib/useAuthGuard";

function statusClass(status: string) {
  return { CRITICAL: "badge-critical", WARNING: "badge-high", NORMAL: "badge-low" }[status] || "badge-medium";
}

function overallStatus(metrics: any[], metricType: string): string {
  const relevant = metrics.filter((m) => m.metric_type === metricType);
  if (relevant.some((m) => m.status === "CRITICAL")) return "CRITICAL";
  if (relevant.some((m) => m.status === "WARNING")) return "WARNING";
  if (relevant.length === 0) return "NO DATA";
  return "NORMAL";
}

export default function ModelMonitoringPage() {
  const { ready } = useAuthGuard();
  const [data, setData] = useState<any>(null);
  const [drift, setDrift] = useState<any[]>([]);
  const [ranking, setRanking] = useState<any>(null);
  const [ragEval, setRagEval] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  const [computing, setComputing] = useState(false);
  const [computingRanking, setComputingRanking] = useState(false);
  const [computingRag, setComputingRag] = useState(false);

  function loadAll() {
    api.modelMonitoring().then(setData).catch((e) => setError(e.message));
    api.drift().then(setDrift).catch((e) => setError(e.message));
    api.recoveryRankingEval().then(setRanking).catch((e) => setError(e.message));
    api.ragEval().then(setRagEval).catch((e) => setError(e.message));
  }

  useEffect(() => {
    if (!ready) return;
    loadAll();
  }, [ready]);

  async function runDriftCheck() {
    setComputing(true);
    setError(null);
    try {
      await api.computeDrift(500);
      loadAll();
    } catch (e: any) {
      setError(e.message);
    } finally {
      setComputing(false);
    }
  }

  async function runRagEval() {
    setComputingRag(true);
    setError(null);
    try {
      const result = await api.computeRagEval();
      setRagEval({ computed: true, ...result });
    } catch (e: any) {
      setError(e.message);
    } finally {
      setComputingRag(false);
    }
  }

  async function runRankingEval() {
    setComputingRanking(true);
    setError(null);
    try {
      const result = await api.computeRecoveryRankingEval();
      setRanking({ computed: !result.skipped, ...result });
    } catch (e: any) {
      setError(e.message);
    } finally {
      setComputingRanking(false);
    }
  }

  if (!ready) return null;

  const dataStatus = overallStatus(drift, "data_drift");
  const predictionStatus = overallStatus(drift, "prediction_drift");
  const missingStatus = overallStatus(drift, "missing_value");

  return (
    <div>
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-2xl font-bold">Model Monitoring</h1>
        <button className="btn btn-primary text-sm" onClick={runDriftCheck} disabled={computing}>
          {computing ? "Computing..." : "Run drift check now"}
        </button>
      </div>
      {error && <div className="text-bad mb-4">{error}</div>}

      <div className="grid grid-cols-4 gap-4 mb-6">
        <div className="card">
          <div className="text-xs text-white/50 mb-1">Data Drift</div>
          <span className={`badge ${statusClass(dataStatus)}`}>{dataStatus}</span>
        </div>
        <div className="card">
          <div className="text-xs text-white/50 mb-1">Prediction Drift</div>
          <span className={`badge ${statusClass(predictionStatus)}`}>{predictionStatus}</span>
        </div>
        <div className="card">
          <div className="text-xs text-white/50 mb-1">Missing Values</div>
          <span className={`badge ${statusClass(missingStatus)}`}>{missingStatus}</span>
        </div>
        <div className="card">
          <div className="text-xs text-white/50 mb-1">Metrics Tracked</div>
          <div className="text-2xl font-semibold">{drift.length}</div>
        </div>
      </div>

      <h2 className="font-semibold mb-3">Champion vs Challenger</h2>
      <div className="card p-0 overflow-hidden mb-8">
        <table className="w-full text-sm">
          <thead className="bg-white/5 text-white/50 text-xs uppercase">
            <tr>
              <th className="text-left p-3">Model</th>
              <th className="text-left p-3">Type</th>
              <th className="text-left p-3">PR-AUC</th>
              <th className="text-left p-3">ROC-AUC</th>
              <th className="text-left p-3">Precision</th>
              <th className="text-left p-3">Recall</th>
              <th className="text-left p-3">Brier Score</th>
              <th className="text-left p-3">Champion</th>
            </tr>
          </thead>
          <tbody>
            {data?.champion_vs_challenger?.map((m: any, i: number) => (
              <tr key={i} className="border-t border-white/5">
                <td className="p-3">{m.model_name}</td>
                <td className="p-3 text-white/50">{m.model_type}</td>
                <td className="p-3">{m.metrics.pr_auc?.toFixed(3) ?? m.metrics.macro_f1?.toFixed(3) ?? "—"}</td>
                <td className="p-3">{m.metrics.roc_auc?.toFixed(3) ?? "—"}</td>
                <td className="p-3">{m.metrics.precision?.toFixed(3) ?? "—"}</td>
                <td className="p-3">{m.metrics.recall?.toFixed(3) ?? "—"}</td>
                <td className="p-3">{m.metrics.brier_score?.toFixed(3) ?? "—"}</td>
                <td className="p-3">{m.is_champion ? <span className="badge badge-low">CHAMPION</span> : "—"}</td>
              </tr>
            ))}
          </tbody>
        </table>
        {(!data?.champion_vs_challenger || data.champion_vs_challenger.length === 0) && (
          <div className="p-6 text-center text-white/40">
            No models trained yet. Run: python -m ml.training.train_denial_model
          </div>
        )}
      </div>

      <h2 className="font-semibold mb-3">Drift Metrics</h2>
      <div className="card p-0 overflow-hidden">
        <table className="w-full text-sm">
          <thead className="bg-white/5 text-white/50 text-xs uppercase">
            <tr>
              <th className="text-left p-3">Type</th>
              <th className="text-left p-3">Feature</th>
              <th className="text-left p-3">Metric</th>
              <th className="text-left p-3">Value</th>
              <th className="text-left p-3">Baseline → Current</th>
              <th className="text-left p-3">Status</th>
              <th className="text-left p-3">Recorded</th>
            </tr>
          </thead>
          <tbody>
            {drift.map((d: any) => (
              <tr key={d.id} className="border-t border-white/5">
                <td className="p-3 text-white/50">{d.metric_type?.replace(/_/g, " ")}</td>
                <td className="p-3">{d.feature || "—"}</td>
                <td className="p-3">{d.metric}</td>
                <td className="p-3 font-mono">{Number(d.value).toFixed(4)}</td>
                <td className="p-3 text-white/50">
                  {Number(d.baseline_value).toFixed(3)} → {Number(d.current_value).toFixed(3)}
                </td>
                <td className="p-3">
                  <span className={`badge ${statusClass(d.status)}`}>{d.status}</span>
                </td>
                <td className="p-3 text-white/40 text-xs">{new Date(d.recorded_at).toLocaleString()}</td>
              </tr>
            ))}
          </tbody>
        </table>
        {drift.length === 0 && (
          <div className="p-6 text-center text-white/40">
            No drift metrics computed yet — click &ldquo;Run drift check now&rdquo; above.
          </div>
        )}
      </div>

      <div className="flex items-center justify-between mt-8 mb-3">
        <h2 className="font-semibold">Recovery Ranking Evaluation</h2>
        <button className="btn btn-primary text-sm" onClick={runRankingEval} disabled={computingRanking}>
          {computingRanking ? "Evaluating..." : "Run ranking eval now"}
        </button>
      </div>
      <p className="text-xs text-white/40 mb-3">
        Precision@K / NDCG@K / recovery-captured@K, comparing the model-driven priority
        queue against naive baselines on real held-out appeal outcomes — not accuracy on
        training data.
      </p>
      {ranking?.computed && (
        <div className="card p-0 overflow-hidden">
          <table className="w-full text-sm">
            <thead className="bg-white/5 text-white/50 text-xs uppercase">
              <tr>
                <th className="text-left p-3">Strategy</th>
                {ranking.k_values.map((k: number) => (
                  <th key={`p${k}`} className="text-left p-3">P@{k}</th>
                ))}
                {ranking.k_values.map((k: number) => (
                  <th key={`n${k}`} className="text-left p-3">NDCG@{k}</th>
                ))}
                {ranking.k_values.map((k: number) => (
                  <th key={`r${k}`} className="text-left p-3">Recov@{k}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {Object.entries(ranking.strategies || {}).map(([name, m]: [string, any]) => (
                <tr key={name} className={`border-t border-white/5 ${name === ranking.best_strategy ? "bg-accent/10" : ""}`}>
                  <td className="p-3">
                    {name.replace(/_/g, " ")}
                    {name === ranking.best_strategy && <span className="badge badge-low ml-2">BEST</span>}
                  </td>
                  {ranking.k_values.map((k: number) => (
                    <td key={`p${k}`} className="p-3">{m[`precision_at_${k}`]}</td>
                  ))}
                  {ranking.k_values.map((k: number) => (
                    <td key={`n${k}`} className="p-3">{m[`ndcg_at_${k}`]}</td>
                  ))}
                  {ranking.k_values.map((k: number) => (
                    <td key={`r${k}`} className="p-3">{m[`recovery_captured_at_${k}`]}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
          <div className="p-3 text-xs text-white/40 border-t border-white/5">
            Evaluated on {ranking.n_eval_claims} held-out resolved appeals · model version: {ranking.model_version || "heuristic fallback"}
          </div>
        </div>
      )}
      {!ranking?.computed && (
        <div className="card p-6 text-center text-white/40">
          No ranking evaluation computed yet — click &ldquo;Run ranking eval now&rdquo; above.
        </div>
      )}

      <div className="flex items-center justify-between mt-8 mb-3">
        <h2 className="font-semibold">RAG Evaluation</h2>
        <button className="btn btn-primary text-sm" onClick={runRagEval} disabled={computingRag}>
          {computingRag ? "Evaluating..." : "Run RAG eval now"}
        </button>
      </div>
      <p className="text-xs text-white/40 mb-3">
        Retrieval recall@K against the ingested policy corpus, plus citation
        referential-integrity and excerpt-fidelity checks against real generated appeal
        drafts — catches fabricated or paraphrased citations regardless of LLM provider.
      </p>
      {ragEval?.computed && (
        <div className="grid grid-cols-2 gap-4">
          <div className="card">
            <h3 className="text-sm font-medium mb-3">Retrieval Recall</h3>
            {ragEval.retrieval_recall?.skipped ? (
              <div className="text-white/40 text-sm">{ragEval.retrieval_recall.reason}</div>
            ) : (
              <div className="text-sm space-y-2">
                <div className="flex gap-4">
                  {[1, 3, 5].map((k) => (
                    <div key={k}>
                      <div className="text-xs text-white/50">Recall@{k}</div>
                      <div className="text-lg font-semibold">
                        {((ragEval.retrieval_recall[`recall_at_${k}`] ?? 0) * 100).toFixed(0)}%
                      </div>
                    </div>
                  ))}
                </div>
                <div className="text-xs text-white/40 pt-2 border-t border-white/10">
                  {ragEval.retrieval_recall.n_denial_reasons_evaluated} denial reasons evaluated against the ingested corpus
                </div>
              </div>
            )}
          </div>

          <div className="card">
            <h3 className="text-sm font-medium mb-3">Citation Correctness</h3>
            {ragEval.citation_correctness?.skipped ? (
              <div className="text-white/40 text-sm">{ragEval.citation_correctness.reason}</div>
            ) : (
              <div className="text-sm space-y-1">
                <div className="flex justify-between">
                  <span className="text-white/60">Referential integrity</span>
                  <span>{((ragEval.citation_correctness.referential_integrity_rate ?? 0) * 100).toFixed(0)}%</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-white/60">Excerpt fidelity</span>
                  <span>{((ragEval.citation_correctness.excerpt_fidelity_rate ?? 0) * 100).toFixed(0)}%</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-white/60">Unsupported citation rate</span>
                  <span>{((ragEval.citation_correctness.unsupported_citation_rate ?? 0) * 100).toFixed(0)}%</span>
                </div>
                <div className="text-xs text-white/40 pt-2 border-t border-white/10">
                  {ragEval.citation_correctness.total_citations} citations checked across{" "}
                  {ragEval.citation_correctness.n_drafts_generated} generated drafts
                  {ragEval.citation_correctness.n_blocked_by_evidence_gate > 0 &&
                    ` (${ragEval.citation_correctness.n_blocked_by_evidence_gate} blocked by evidence gate)`}
                </div>
              </div>
            )}
          </div>
        </div>
      )}
      {!ragEval?.computed && (
        <div className="card p-6 text-center text-white/40">
          No RAG evaluation computed yet — click &ldquo;Run RAG eval now&rdquo; above.
        </div>
      )}
    </div>
  );
}
