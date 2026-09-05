"""
Feature engineering for anomaly/duplicate detection (Phase 2, Feature A).

Feature rationale (documented per the task spec's A1 requirement):

- claim_amount / claim_amount_percentile: unusually large or small claims
  relative to the overall distribution are the most direct anomaly signal.
- procedure_count / diagnosis_count / modifier_count: unusual combinations
  (e.g. many procedures on one line, excessive modifiers) can indicate
  data-entry errors or upcoding patterns worth a human look.
- days_to_submission: extremely fast or extremely delayed submission
  relative to typical provider/payer behavior is itself unusual.
- provider_claim_frequency / payer_claim_frequency / procedure_frequency:
  rare provider/payer/procedure combinations are inherently less "seen
  before" by any downstream process, which is what anomaly detection means
  here -- statistically unusual, not fraudulent.
- provider_average_claim_amount / payer_average_claim_amount: lets the
  model learn whether *this* claim looks unusual relative to *this*
  provider's/payer's own typical billing pattern, not just the global one.
- claim_line_amount_variance: high variance across line amounts on one
  claim is a distinct pattern from a claim with uniform line pricing.
- duplicate_similarity_score: a simple same-day/same-provider/same-
  procedure/same-amount match count, used as an explicit duplicate-claim
  signal distinct from the general outlier signal above.

All features are computed from data that exists at prediction time (claim +
its lines + reference aggregates) -- there is no leakage concern here in
the denial-prediction sense, since anomaly detection is unsupervised and
runs on claims regardless of outcome.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "backend"))

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.domain import Claim, ClaimLine, Payer, Provider

ANOMALY_FEATURES = [
    "claim_amount",
    "claim_amount_percentile",
    "procedure_count",
    "diagnosis_count",
    "modifier_count",
    "days_to_submission",
    "provider_claim_frequency",
    "payer_claim_frequency",
    "procedure_frequency",
    "provider_average_claim_amount",
    "payer_average_claim_amount",
    "claim_line_amount_variance",
    "duplicate_similarity_score",
]


def load_anomaly_frame(db: Session) -> pd.DataFrame:
    # Only claim_lines/claim fields are needed here -- payer/provider rows
    # themselves aren't referenced, only their IDs (used below purely for
    # frequency counting via df["payer_id"]/df["provider_id"]).
    claims = db.execute(select(Claim)).scalars().all()

    rows = []
    for c in claims:
        lines = c.lines
        line_amounts = [float(l.line_amount) for l in lines] or [float(c.claim_amount)]
        rows.append(
            {
                "claim_id": c.id,
                "payer_id": c.payer_id,
                "provider_id": c.provider_id,
                "claim_amount": float(c.claim_amount),
                "procedure_count": len({l.procedure_code for l in lines}) or 1,
                "diagnosis_count": len({l.diagnosis_code for l in lines}) or 1,
                "modifier_count": sum(1 for l in lines if l.modifiers),
                "days_to_submission": (c.submission_date - c.service_date).days if c.submission_date else 0,
                "line_amounts": line_amounts,
                "service_date": c.service_date,
                "primary_procedure": lines[0].procedure_code if lines else "unknown",
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Percentile of claim_amount within the overall distribution.
    df["claim_amount_percentile"] = df["claim_amount"].rank(pct=True)

    # Frequency-based features: how common is this provider/payer/procedure
    # in the dataset (rarer = more anomalous by construction).
    provider_counts = df["provider_id"].value_counts()
    payer_counts = df["payer_id"].value_counts()
    procedure_counts = df["primary_procedure"].value_counts()
    df["provider_claim_frequency"] = df["provider_id"].map(provider_counts) / len(df)
    df["payer_claim_frequency"] = df["payer_id"].map(payer_counts) / len(df)
    df["procedure_frequency"] = df["primary_procedure"].map(procedure_counts) / len(df)

    provider_avg = df.groupby("provider_id")["claim_amount"].transform("mean")
    payer_avg = df.groupby("payer_id")["claim_amount"].transform("mean")
    df["provider_average_claim_amount"] = provider_avg
    df["payer_average_claim_amount"] = payer_avg

    df["claim_line_amount_variance"] = df["line_amounts"].apply(lambda x: float(np.var(x)) if len(x) > 1 else 0.0)

    # Duplicate-similarity: count of other claims from the same provider,
    # same primary procedure, same service date, similar amount (+/-5%).
    def _dup_score(row):
        same_day = df[
            (df["provider_id"] == row["provider_id"])
            & (df["primary_procedure"] == row["primary_procedure"])
            & (df["service_date"] == row["service_date"])
        ]
        if len(same_day) <= 1:
            return 0.0
        amt = row["claim_amount"]
        close_amount = same_day[(same_day["claim_amount"] - amt).abs() <= 0.05 * max(amt, 1)]
        return min(1.0, (len(close_amount) - 1) / 3.0)  # 3+ near-identical same-day claims -> max score

    df["duplicate_similarity_score"] = df.apply(_dup_score, axis=1)

    return df[["claim_id"] + ANOMALY_FEATURES]
