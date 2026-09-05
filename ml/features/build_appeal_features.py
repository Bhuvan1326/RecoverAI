"""
Feature engineering for the appeal-success model (Phase 2, Feature C).

LEAKAGE RULE (C2, critical): only information knowable AT THE TIME AN APPEAL
IS FILED may be used as a feature. `recovered_amount`, `outcome`, and
`decision_date` are appeal-RESOLUTION fields -- they are only known after
the fact and are used exclusively to construct the training LABEL, never as
a feature. `appeal_timing` (days between denial and appeal filing) IS
allowed, because that's a decision the biller makes before the outcome is
known.

Labels are synthetic by construction: outcome in scripts/generate_data.py is
generated from a documented appealability-prior function, not observed real
appeal results. Every row is implicitly is_synthetic=True (inherited from
the underlying AppealEvent/Claim rows) -- see the model card written by
train_appeal_success_model.py for the explicit disclaimer surfaced to the UI.
"""
import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.domain import AppealEvent, Claim, ClaimLine, DenialEvent, Payer, Provider

CATEGORICAL_FEATURES = ["denial_reason", "payer_type", "provider_specialty", "procedure_code"]
NUMERIC_FEATURES = [
    "claim_amount",
    "documentation_completeness",
    "authorization_missing",
    "eligibility_fail",
    "timely_filing_breach",
    "appeal_timing_days",
    "historical_payer_appeal_rate",
    "historical_reason_appeal_rate",
]
ALL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES

# Fields that must NEVER appear as a feature -- they're only knowable after
# the appeal is resolved. Enforced by test_appeal_leakage.py.
FORBIDDEN_POST_RESOLUTION_FIELDS = {"outcome", "recovered_amount", "decision_date", "appeal_success"}


def load_raw_frame(db: Session) -> pd.DataFrame:
    appeals = db.execute(select(AppealEvent).where(AppealEvent.outcome.in_(["WON", "LOST"]))).scalars().all()
    denials = {d.id: d for d in db.execute(select(DenialEvent)).scalars().all()}
    claims = {c.id: c for c in db.execute(select(Claim)).scalars().all()}
    payers = {p.id: p for p in db.execute(select(Payer)).scalars().all()}
    providers = {p.id: p for p in db.execute(select(Provider)).scalars().all()}

    lines_by_claim: dict[str, list[ClaimLine]] = {}
    for line in db.execute(select(ClaimLine)).scalars().all():
        lines_by_claim.setdefault(line.claim_id, []).append(line)

    rows = []
    for appeal in appeals:
        denial = denials.get(appeal.denial_event_id)
        if not denial:
            continue
        claim = claims.get(denial.claim_id)
        if not claim:
            continue
        payer = payers.get(claim.payer_id)
        provider = providers.get(claim.provider_id)
        lines = lines_by_claim.get(claim.id, [])

        timely_filing_breach = bool(
            claim.timely_filing_deadline and claim.submission_date and claim.submission_date > claim.timely_filing_deadline
        )
        appeal_timing_days = max((appeal.appeal_date - denial.denial_date).days, 0)

        rows.append(
            {
                "appeal_id": appeal.id,
                "claim_id": claim.id,
                "payer_id": claim.payer_id,
                "denial_reason": denial.denial_reason_code,
                "payer_type": payer.payer_type if payer else "unknown",
                "provider_specialty": provider.specialty if provider else "unknown",
                "procedure_code": lines[0].procedure_code if lines else "unknown",
                "claim_amount": float(claim.claim_amount),
                "documentation_completeness": float(claim.documentation_completeness),
                "authorization_missing": int(claim.authorization_status == "MISSING"),
                "eligibility_fail": int(claim.eligibility_status == "FAIL"),
                "timely_filing_breach": int(timely_filing_breach),
                "appeal_timing_days": appeal_timing_days,
                "appeal_date": appeal.appeal_date,
                # Label -- constructed from resolution fields HERE ONLY; these
                # raw fields are dropped before the frame reaches ALL_FEATURES.
                "appeal_success": 1 if appeal.outcome == "WON" else 0,
            }
        )
    return pd.DataFrame(rows)


def add_expanding_historical_appeal_rates(df: pd.DataFrame) -> pd.DataFrame:
    """
    As-of-appeal-date expanding success rate per payer and per denial reason.
    Sorted chronologically by appeal_date so a given appeal only "sees"
    outcomes from appeals filed strictly before it -- the same leakage guard
    pattern used for the denial-risk model's historical rate features.
    """
    df = df.sort_values("appeal_date").reset_index(drop=True)

    for key, out_col in [("payer_id", "historical_payer_appeal_rate"), ("denial_reason", "historical_reason_appeal_rate")]:
        cum_count: dict = {}
        cum_won: dict = {}
        rates = []
        for _, row in df.iterrows():
            k = row[key]
            n = cum_count.get(k, 0)
            w = cum_won.get(k, 0)
            rates.append(w / n if n >= 5 else 0.40)  # neutral prior until >=5 prior observations
            cum_count[k] = n + 1
            cum_won[k] = w + row["appeal_success"]
        df[out_col] = rates
    return df


def build_feature_matrix(db: Session) -> pd.DataFrame:
    df = load_raw_frame(db)
    if df.empty:
        return df
    df = add_expanding_historical_appeal_rates(df)
    return df


# --------------------------------------------------------------------------
# As-of-date historical appeal-rate lookup for SINGLE-CLAIM inference.
#
# Bug fixed here (Section 7 of the hardening spec): single-claim scoring in
# app/services/appeal_success.py used to hardcode
# historical_payer_appeal_rate / historical_reason_appeal_rate to a
# constant 0.40 for every claim, rather than the actual appeal-success
# history for that specific payer or denial reason. Mirrors
# ml/features/build_features.py:compute_asof_denial_rate's policy exactly
# (same minimum-sample-size-before-trusting-it cold-start fallback), just
# over resolved AppealEvents instead of DenialEvents.
# --------------------------------------------------------------------------
def compute_asof_appeal_rate(
    db: Session,
    key: str,
    key_value: str,
    cutoff_date,
    min_observations: int = 5,
    neutral_prior: float = 0.40,
) -> float:
    """
    Real as-of-`cutoff_date` appeal-success rate for a given payer_id
    (`key="payer_id"`) or denial_reason_code (`key="denial_reason"`),
    computed from appeals actually filed strictly before that date whose
    outcome is already resolved (WON/LOST). Returns `neutral_prior` if
    fewer than `min_observations` qualifying prior appeals exist.
    """
    from app.models.domain import AppealEvent, Claim, DenialEvent

    if cutoff_date is None or key_value is None:
        return neutral_prior

    q = (
        select(AppealEvent.outcome)
        .join(DenialEvent, DenialEvent.id == AppealEvent.denial_event_id)
        .where(AppealEvent.outcome.in_(["WON", "LOST"]), AppealEvent.appeal_date < cutoff_date)
    )
    if key == "payer_id":
        q = q.join(Claim, Claim.id == DenialEvent.claim_id).where(Claim.payer_id == key_value)
    elif key == "denial_reason":
        q = q.where(DenialEvent.denial_reason_code == key_value)
    else:
        raise ValueError(f"Unsupported key: {key}")

    outcomes = [row[0] for row in db.execute(q).all()]
    if len(outcomes) < min_observations:
        return neutral_prior
    return sum(1 for o in outcomes if o == "WON") / len(outcomes)
