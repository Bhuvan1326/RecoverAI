#!/usr/bin/env python3
"""
Evaluates the recovery-priority-queue's ranking quality against REAL
outcomes (Section 11/15 of the design doc: "Precision@K, NDCG@K, expected
recovery captured@K"). Answers the question the priority queue exists to
answer: "if staff work claims in this order, how much of the actual
recoverable money do they capture early, and how often is the top of the
queue actually a claim that gets successfully appealed?"

Methodology
-----------
Uses the SAME held-out chronological test split the appeal-success model
was evaluated on (ml/training/train_appeal_success_model.py) -- this is a
genuine out-of-sample ranking evaluation, not an evaluation on training
data. Ground truth relevance:
  - Binary relevance (for Precision@K): 1 if the appeal was actually WON,
    0 otherwise.
  - Graded/dollar relevance (for NDCG@K and expected-recovery-captured@K):
    the actual amount recovered -- which in this synthetic dataset equals
    claim_amount if won, 0 otherwise (see scripts/generate_data.py).

Compares three ranking strategies:
  - "model_expected_value": rank by the trained model's predicted
    P(appeal success) x claim_amount -- what the live recovery queue
    actually does (see app/services/recovery.py:expected_recovery_value).
  - "claim_amount_only": naive baseline -- work the biggest claims first,
    ignoring appeal-success likelihood entirely (mirrors "Strategy A" in
    the design doc's Recovery Strategy Simulator).
  - "appeal_probability_only": rank purely by predicted success
    probability, ignoring dollar amount.
  - "random": average over repeated random shuffles, as a floor baseline.

If the model-driven ranking doesn't clearly beat both naive baselines,
that's a real, useful, honest finding -- this script reports it either way
rather than only ever printing a flattering number.

Usage:
    python -m ml.evaluation.eval_recovery_ranking
"""
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "backend"))

from app.core import database as database_module
from app.models.domain import ModelMetric, ModelVersion
from app.services import appeal_success as appeal_success_service
from ml.evaluation.ranking_metrics import evaluate_ranking
from ml.features.build_appeal_features import ALL_FEATURES, build_feature_matrix

ARTIFACT_DIR = Path(__file__).resolve().parent.parent / "models" / "artifacts"
K_VALUES = [5, 10, 20]
N_RANDOM_SHUFFLES = 50  # averaged for a stable random-baseline estimate
MIN_EVAL_ROWS = 20


def _random_baseline(binary_relevance, dollar_relevance, k_values, seed=42) -> dict:
    rng = random.Random(seed)
    n = len(binary_relevance)
    accum = {f"precision_at_{k}": [] for k in k_values}
    accum |= {f"ndcg_at_{k}": [] for k in k_values}
    accum |= {f"recovery_captured_at_{k}": [] for k in k_values}

    for _ in range(N_RANDOM_SHUFFLES):
        random_scores = [rng.random() for _ in range(n)]
        result = evaluate_ranking(random_scores, binary_relevance, dollar_relevance, k_values)
        for key in accum:
            accum[key].append(result[key])

    return {key: round(float(np.mean(vals)), 4) for key, vals in accum.items()} | {"n_items": n}


def run_ranking_evaluation(db, k_values: list[int] | None = None, persist: bool = True) -> dict:
    """
    Reusable entry point (Section 11 eval, callable from the CLI script's
    __main__ block AND the API's on-demand endpoint -- one code path,
    exactly like ml/monitoring/drift.py:run_full_drift_check). Returns a
    dict describing the evaluation, or {"skipped": True, "reason": ...} if
    there isn't enough resolved-appeal data yet to evaluate meaningfully.
    """
    k_values = k_values or K_VALUES
    df = build_feature_matrix(db)
    if len(df) < MIN_EVAL_ROWS:
        return {
            "skipped": True,
            "reason": f"Only {len(df)} resolved appeals available (need >= {MIN_EVAL_ROWS}).",
        }

    df = df.sort_values("appeal_date").reset_index(drop=True)
    split_point = int(len(df) * 0.8)
    test_df = df.iloc[split_point:].reset_index(drop=True)
    if len(test_df) < MIN_EVAL_ROWS:
        return {
            "skipped": True,
            "reason": f"Held-out test split only has {len(test_df)} rows (need >= {MIN_EVAL_ROWS}).",
        }

    try:
        mv = appeal_success_service.get_champion_model_version(db)
        pipeline = appeal_success_service._load_artifact(mv.artifact_path)
        predicted_probabilities = pipeline.predict_proba(test_df[ALL_FEATURES])[:, 1]
        model_version_label = f"{mv.model_name}-{mv.version_tag}"
        model_version_id = mv.id
    except appeal_success_service.AppealModelNotTrainedError:
        from app.services.recovery import APPEALABILITY_PRIOR

        predicted_probabilities = np.array(
            [
                max(0.02, min(0.97, APPEALABILITY_PRIOR.get(reason, 0.30) * (0.5 + 0.5 * doc / 100.0)))
                for reason, doc in zip(test_df["denial_reason"], test_df["documentation_completeness"])
            ]
        )
        model_version_label = None
        model_version_id = None

    claim_amounts = test_df["claim_amount"].values
    binary_relevance = test_df["appeal_success"].astype(int).tolist()
    dollar_relevance = [amt if won else 0.0 for amt, won in zip(claim_amounts, binary_relevance)]

    strategies = {
        "model_expected_value": (predicted_probabilities * claim_amounts).tolist(),
        "claim_amount_only": claim_amounts.tolist(),
        "appeal_probability_only": predicted_probabilities.tolist(),
    }

    results = {name: evaluate_ranking(scores, binary_relevance, dollar_relevance, k_values) for name, scores in strategies.items()}
    results["random"] = _random_baseline(binary_relevance, dollar_relevance, k_values)

    if persist and model_version_id:
        for k in k_values:
            for metric_name in (f"precision_at_{k}", f"ndcg_at_{k}", f"recovery_captured_at_{k}"):
                db.add(
                    ModelMetric(
                        model_version_id=model_version_id,
                        metric_name=f"ranking_{metric_name}",
                        metric_value=results["model_expected_value"][metric_name],
                    )
                )
        db.commit()

    best_at_first_k = max(results, key=lambda k: results[k][f"recovery_captured_at_{k_values[0]}"])

    summary = {
        "skipped": False,
        "model_version": model_version_label,
        "n_eval_claims": len(test_df),
        "k_values": k_values,
        "strategies": results,
        "best_strategy": best_at_first_k,
    }

    if persist:
        # Written here (not only from the CLI's __main__) so the API's
        # on-demand /model-monitoring/recovery-ranking/compute endpoint and
        # the CLI script leave the SAME artifact behind -- otherwise
        # GET /model-monitoring/recovery-ranking (which reads this file)
        # would go stale relative to API-triggered computations.
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        (ARTIFACT_DIR / "recovery_ranking_eval_summary.json").write_text(json.dumps(summary, indent=2))

    return summary


def main():
    db = database_module.SessionLocal()
    try:
        result = run_ranking_evaluation(db)
        if result["skipped"]:
            print(result["reason"])
            if "Generate more" not in result["reason"]:
                print("Generate more synthetic data first: python scripts/generate_data.py --claims 5000")
            return

        print(f"Evaluating ranking quality on {result['n_eval_claims']} held-out (chronologically most recent) resolved appeals.")
        results = result["strategies"]

        print(f"\n{'Strategy':<26}" + "".join(f"P@{k:<7}" for k in K_VALUES) + "".join(f"NDCG@{k:<5}" for k in K_VALUES) + "".join(f"Recov@{k:<5}" for k in K_VALUES))
        for name, r in results.items():
            row = f"{name:<26}"
            row += "".join(f"{r[f'precision_at_{k}']:<9}" for k in K_VALUES)
            row += "".join(f"{r[f'ndcg_at_{k}']:<10}" for k in K_VALUES)
            row += "".join(f"{r[f'recovery_captured_at_{k}']:<11}" for k in K_VALUES)
            print(row)

        if result["model_version"]:
            print(f"\nPersisted ranking metrics to ModelMetric for {result['model_version']}.")

        summary_path = ARTIFACT_DIR / "recovery_ranking_eval_summary.json"
        print(f"Summary written to {summary_path}")
        print(f"\nBest strategy by recovery-captured@{K_VALUES[0]}: {result['best_strategy']}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
