"""
Feature engineering for the denial-prediction model.

LEAKAGE RULE (Section 7 of spec): every feature here must be knowable
*before* claim submission / adjudication. denial_reason_code, appeal
outcome, and paid amount are never used as inputs -- only as labels or
for downstream (post-hoc) models. Payer/provider historical denial rates
are computed as *expanding* (as-of-submission-date) rates, not global
rates, to avoid leaking future denial information backward in time.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "backend"))

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.domain import Claim, DenialEvent, Payer, Provider

CATEGORICAL_FEATURES = ["claim_type", "place_of_service", "payer_type", "provider_specialty", "procedure_code"]
NUMERIC_FEATURES = [
    "claim_amount",
    "documentation_completeness",
    "days_to_submission",
    "modifier_count",
    "procedure_count",
    "line_count",
    "payer_historical_denial_rate",
    "provider_historical_denial_rate",
    "auth_missing",
    "eligibility_fail",
]
ALL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def load_raw_frame(db: Session) -> pd.DataFrame:
    claims = db.execute(select(Claim)).scalars().all()
    payers = {p.id: p for p in db.execute(select(Payer)).scalars().all()}
    providers = {p.id: p for p in db.execute(select(Provider)).scalars().all()}
    denials = {d.claim_id: d for d in db.execute(select(DenialEvent)).scalars().all()}

    rows = []
    for c in claims:
        if c.submission_date is None:
            continue
        lines = c.lines
        payer = payers.get(c.payer_id)
        provider = providers.get(c.provider_id)
        denial = denials.get(c.id)
        rows.append(
            {
                "claim_id": c.id,
                "payer_id": c.payer_id,
                "provider_id": c.provider_id,
                "claim_type": c.claim_type,
                "place_of_service": c.place_of_service,
                "payer_type": payer.payer_type if payer else "unknown",
                "provider_specialty": provider.specialty if provider else "unknown",
                "procedure_code": lines[0].procedure_code if lines else "unknown",
                "claim_amount": float(c.claim_amount),
                "documentation_completeness": float(c.documentation_completeness),
                "days_to_submission": (c.submission_date - c.service_date).days,
                "modifier_count": sum(1 for l in lines if l.modifiers),
                "procedure_count": len({l.procedure_code for l in lines}),
                "line_count": len(lines),
                "auth_missing": int(c.authorization_status == "MISSING"),
                "eligibility_fail": int(c.eligibility_status == "FAIL"),
                "submission_date": c.submission_date,
                "is_denied": int(denial is not None),
                "denial_reason_code": denial.denial_reason_code if denial else None,
            }
        )
    return pd.DataFrame(rows)


def add_expanding_historical_rates(df: pd.DataFrame) -> pd.DataFrame:
    """
    As-of-submission-date expanding denial rate per payer/provider.
    Sorted chronologically so a claim only "sees" denial history from
    claims submitted strictly before it -- this is the leakage guard.
    """
    df = df.sort_values("submission_date").reset_index(drop=True)

    for key, out_col in [("payer_id", "payer_historical_denial_rate"), ("provider_id", "provider_historical_denial_rate")]:
        df[out_col] = 0.15  # neutral prior for the first claim(s) seen for a given key
        cum_count: dict = {}
        cum_denied: dict = {}
        rates = []
        for _, row in df.iterrows():
            k = row[key]
            n = cum_count.get(k, 0)
            d = cum_denied.get(k, 0)
            rates.append(d / n if n >= 5 else 0.15)  # need >=5 prior claims before trusting the empirical rate
            cum_count[k] = n + 1
            cum_denied[k] = d + row["is_denied"]
        df[out_col] = rates
    return df


def build_feature_matrix(db: Session) -> pd.DataFrame:
    df = load_raw_frame(db)
    if df.empty:
        return df
    df = add_expanding_historical_rates(df)
    return df


# --------------------------------------------------------------------------
# As-of-date historical rate lookups for SINGLE-CLAIM inference.
#
# Bug fixed here (Section 6/7 of the hardening spec): single-claim scoring
# in app/services/ml_inference.py and app/services/appeal_success.py used
# to hardcode payer/provider_historical_*_rate to a constant (0.15 / 0.40)
# regardless of what's actually known about that payer/provider -- meaning
# every claim for a given payer got the same historical-rate feature value
# no matter how much real history existed. This queries the ACTUAL history
# available strictly before the cutoff timestamp, using the same
# "expanding, as-of-date, minimum-sample-size-before-trusting-it" policy
# training uses (add_expanding_historical_rates above), just evaluated as
# a live query instead of a batch dataframe walk. Both call sites (denial
# risk and appeal success) now go through this one function so the policy
# can't silently diverge between them.
# --------------------------------------------------------------------------
def compute_asof_denial_rate(
    db: Session,
    key_column: str,
    key_value: str,
    cutoff_date,
    min_observations: int = 5,
    neutral_prior: float = 0.15,
) -> float:
    """
    Real as-of-`cutoff_date` denial rate for a given payer_id or
    provider_id, computed from claims actually submitted strictly before
    that date. Returns `neutral_prior` if fewer than `min_observations`
    qualifying prior claims exist -- this is a documented cold-start
    fallback, not an attempt to fabricate a rate from insufficient data.
    """
    from sqlalchemy import func

    from app.models.domain import Claim, DenialEvent

    if cutoff_date is None or key_value is None:
        return neutral_prior

    prior_claim_ids = [
        row[0]
        for row in db.execute(
            select(Claim.id).where(
                getattr(Claim, key_column) == key_value,
                Claim.submission_date.is_not(None),
                Claim.submission_date < cutoff_date,
            )
        ).all()
    ]
    if len(prior_claim_ids) < min_observations:
        return neutral_prior

    denied_count = db.execute(
        select(func.count()).select_from(DenialEvent).where(DenialEvent.claim_id.in_(prior_claim_ids))
    ).scalar_one()
    return denied_count / len(prior_claim_ids)
