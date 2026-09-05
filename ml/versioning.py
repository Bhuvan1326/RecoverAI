"""
Shared model-versioning helpers (Section 9 of the hardening spec).

Bug fixed here: every training script used a STATIC version_tag
(e.g. "1.0-logistic_regression"), so re-running training a second time --
which a scheduled weekly retrain (Celery Beat) or simply re-running the
seed script does routinely -- crashed with a UNIQUE constraint violation
on (model_type, version_tag), and historical model versions could never
accumulate. Every training script now calls new_version_tag() to get a
timestamp-based tag that is unique per training RUN (not per model type),
so repeated retraining always succeeds and past versions are preserved
rather than overwritten or deleted.
"""
import subprocess
from datetime import datetime, timezone


def new_version_tag(model_name: str, run_timestamp: str | None = None) -> str:
    """
    `run_timestamp` lets every candidate trained in the SAME training run
    (e.g. logistic_regression/random_forest/xgboost/catboost, all trained
    together by one `python -m ml.training.train_*` invocation) share one
    timestamp -- so a single run's models are recognizable as a cohort --
    while remaining unique across DIFFERENT runs (down to the second).
    """
    ts = run_timestamp or datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{ts}-{model_name}"


def new_run_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def get_git_commit() -> str | None:
    """Best-effort short git commit hash for reproducibility tracking
    (Section 9). Returns None (not a fake value) if git metadata isn't
    available -- e.g. the repo was extracted from a zip rather than cloned."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def promote_if_better(
    db,
    model_type: str,
    results: dict,
    primary_metric: str = "pr_auc",
    higher_is_better: bool = True,
) -> str | None:
    """
    Champion promotion gate (Section 10 of the hardening spec).

    Bug fixed here: training scripts used to unconditionally promote
    whichever candidate from the CURRENT run scored highest, without ever
    comparing against the champion from a PREVIOUS run -- so retraining on
    a worse data sample, or a run where every candidate underperformed,
    could silently demote a genuinely better existing champion. It also
    left multiple ModelVersion rows with is_champion=True simultaneously
    (one per run), since only the current run's own rows were touched.

    `results` is {model_name: {"metrics": {...}, "model_version_id": ...}}
    for the candidates just trained in ONE run. This function:
      1. Refuses to promote anything if no candidate has a valid
         (non-null, numeric) `primary_metric` -- a validation failure,
         not a promotion decision.
      2. Compares the best NEW candidate against the CURRENT champion
         (from any prior run) for this model_type, if one exists.
      3. Only promotes if the new candidate is at least as good; the
         previous champion is left in place otherwise.
      4. When promoting, un-champions every OTHER ModelVersion row for
         this model_type (across all runs) so exactly one champion ever
         exists at a time, then commits.

    Returns the promoted model's name, or None if no promotion happened
    (either because nothing beat the incumbent, or every candidate failed
    validation).
    """
    from sqlalchemy import select, update

    from app.models.domain import ModelVersion

    valid_results = {
        name: r for name, r in results.items() if isinstance(r["metrics"].get(primary_metric), (int, float))
    }
    if not valid_results:
        print(f"  [promotion gate] No candidate has a valid '{primary_metric}' metric -- refusing to promote anything.")
        db.commit()
        return None

    best_name = max(valid_results, key=lambda n: valid_results[n]["metrics"][primary_metric]) if higher_is_better \
        else min(valid_results, key=lambda n: valid_results[n]["metrics"][primary_metric])
    best_value = valid_results[best_name]["metrics"][primary_metric]

    current_champion = db.execute(
        select(ModelVersion).where(ModelVersion.model_type == model_type, ModelVersion.is_champion.is_(True))
    ).scalar_one_or_none()

    should_promote = True
    if current_champion is not None:
        champion_value = current_champion.metrics.get(primary_metric)
        if isinstance(champion_value, (int, float)):
            should_promote = (best_value >= champion_value) if higher_is_better else (best_value <= champion_value)
            if not should_promote:
                print(
                    f"  [promotion gate] New best '{best_name}' ({primary_metric}={best_value:.4f}) did not beat "
                    f"current champion ({primary_metric}={champion_value:.4f}) -- keeping existing champion."
                )

    if should_promote:
        db.execute(update(ModelVersion).where(ModelVersion.model_type == model_type).values(is_champion=False))
        winner = db.get(ModelVersion, valid_results[best_name]["model_version_id"])
        winner.is_champion = True
        db.commit()
        return best_name

    db.commit()
    return None
