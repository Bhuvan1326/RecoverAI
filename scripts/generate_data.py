#!/usr/bin/env python3
"""
Synthetic data generator for RecoverAI (Section 6 of spec).

Generates payers, providers, claims, claim lines, denial events, and appeal
events with DOCUMENTED, DELIBERATELY SIMPLE assumptions (below). This is a
development/demo generator, not a validated model of real-world RCM
behavior -- every record is tagged is_synthetic=True, data_source=
"recoverai_synthetic_generator".

Documented label-generation assumptions
----------------------------------------
1. Missing authorization on a claim that requires one is the single
   strongest deterministic driver of denial (mirrors real RCM literature's
   ranking of auth issues as a top denial cause).
2. Lower documentation completeness, longer days-to-submission, and higher
   claim amount each independently increase denial probability.
3. Each (payer, procedure) pair has a persistent latent "strictness" drawn
   once per generator run, so payer-level and procedure-level denial-rate
   patterns are internally consistent across the run (this is what makes
   "payer intelligence" / "provider intelligence" analytics meaningful on
   synthetic data instead of pure noise).
4. Appeal outcomes are generated ONLY for a subset of denied claims that
   are actually appealed, and success probability is a function of
   documentation completeness and denial-reason "appealability" -- this is
   a synthetic label, explicitly not a real appeal-outcome dataset.

Usage:
    python scripts/generate_data.py --claims 10000 --seed 42
"""
import argparse
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from faker import Faker

from app.core.database import Base, SessionLocal, engine
from app.models.domain import (
    AppealEvent,
    Claim,
    ClaimLine,
    ClaimStatus,
    DenialEvent,
    Payer,
    Provider,
    SyntheticDataRun,
)

DENIAL_REASONS = [
    "MISSING_AUTHORIZATION",
    "ELIGIBILITY_ISSUE",
    "CODING_MISMATCH",
    "DUPLICATE_CLAIM",
    "MISSING_DOCUMENTATION",
    "TIMELY_FILING",
    "MEDICAL_NECESSITY",
    "OTHER",
]

# Reasons a human/agent could plausibly overturn on appeal vs. not.
APPEALABLE_REASONS = {
    "MISSING_AUTHORIZATION": 0.75,
    "ELIGIBILITY_ISSUE": 0.55,
    "CODING_MISMATCH": 0.65,
    "DUPLICATE_CLAIM": 0.10,
    "MISSING_DOCUMENTATION": 0.70,
    "TIMELY_FILING": 0.15,
    "MEDICAL_NECESSITY": 0.45,
    "OTHER": 0.30,
}

PROCEDURE_CODES = ["99213", "99214", "99215", "93000", "36415", "80053", "71046", "97110", "20610", "45378"]
DIAGNOSIS_CODES = ["I10", "E11.9", "M54.5", "J06.9", "K21.9", "R51", "F41.1", "M25.50", "N39.0", "Z00.00"]


def build_reference_data(db, n_payers: int, n_providers: int, seed: int):
    fake = Faker()
    Faker.seed(seed)
    random.seed(seed)

    payers = []
    for _ in range(n_payers):
        p = Payer(name=f"{fake.company()} Health Plan", payer_type=random.choice(["commercial", "medicare_advantage", "medicaid"]))
        db.add(p)
        payers.append(p)
    providers = []
    for _ in range(n_providers):
        pr = Provider(
            npi=str(random.randint(1_000_000_000, 1_999_999_999)),
            name=f"Dr. {fake.last_name()} {random.choice(['Clinic', 'Medical Group', 'Family Practice'])}",
            specialty=random.choice(["internal_medicine", "cardiology", "orthopedics", "family_practice", "radiology"]),
        )
        db.add(pr)
        providers.append(pr)
    db.commit()

    # Latent per-(payer) and per-(procedure) "strictness" -- documented assumption #3.
    payer_strictness = {p.id: random.uniform(0.05, 0.35) for p in payers}
    procedure_strictness = {c: random.uniform(0.0, 0.25) for c in PROCEDURE_CODES}
    provider_quality = {p.id: random.uniform(-0.05, 0.15) for p in providers}  # higher = fewer clean-claim issues

    return payers, providers, payer_strictness, procedure_strictness, provider_quality


def denial_probability(auth_missing, elig_issue, doc_completeness, days_to_submit, amount, payer_strict, proc_strict, provider_q):
    """Deterministic synthetic risk function -- see documented assumptions above."""
    base = 0.04
    base += 0.35 if auth_missing else 0
    base += 0.22 if elig_issue else 0
    base += (100 - doc_completeness) / 100 * 0.20
    base += min(days_to_submit, 60) / 60 * 0.10
    base += min(amount, 20000) / 20000 * 0.05
    base += payer_strict * 0.5
    base += proc_strict * 0.4
    base -= provider_q
    return max(0.01, min(0.97, base))


def generate(n_claims: int, n_payers: int, n_providers: int, seed: int):
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        payers, providers, payer_strict, proc_strict, provider_q = build_reference_data(db, n_payers, n_providers, seed)
        rng = random.Random(seed)

        counts = {"claims": 0, "claim_lines": 0, "denial_events": 0, "appeal_events": 0}
        now = datetime.now(timezone.utc)

        for i in range(n_claims):
            payer = rng.choice(payers)
            provider = rng.choice(providers)
            procedure = rng.choice(PROCEDURE_CODES)
            diagnosis = rng.choice(DIAGNOSIS_CODES)

            service_date = now - timedelta(days=rng.randint(1, 365))
            days_to_submit = rng.choice([1, 2, 3, 5, 7, 10, 14, 21, 30, 45, 60])
            submission_date = service_date + timedelta(days=days_to_submit)

            auth_missing = rng.random() < 0.18
            elig_issue = rng.random() < 0.10
            doc_completeness = round(rng.uniform(40, 100), 2)
            amount = round(rng.uniform(75, 25000), 2)

            claim = Claim(
                claim_number=f"CLM-{100000 + i}",
                provider_id=provider.id,
                payer_id=payer.id,
                patient_ref=f"SYN-PT-{rng.randint(100000, 999999)}",
                claim_amount=amount,
                claim_type=rng.choice(["professional", "institutional"]),
                place_of_service=rng.choice(["11", "21", "22", "23"]),
                status=ClaimStatus.SUBMITTED,
                eligibility_status="FAIL" if elig_issue else "VERIFIED",
                authorization_status="MISSING" if auth_missing else "PRESENT",
                documentation_completeness=doc_completeness,
                service_date=service_date,
                submission_date=submission_date,
                timely_filing_deadline=service_date + timedelta(days=90),
            )
            db.add(claim)
            db.flush()
            counts["claims"] += 1

            for _ in range(rng.randint(1, 3)):
                db.add(
                    ClaimLine(
                        claim_id=claim.id,
                        procedure_code=procedure,
                        diagnosis_code=diagnosis,
                        modifiers=rng.choice(["", "25", "59", "76"]),
                        line_amount=round(amount / rng.randint(1, 3), 2),
                        units=rng.randint(1, 2),
                    )
                )
                counts["claim_lines"] += 1

            p_denied = denial_probability(
                auth_missing, elig_issue, doc_completeness, days_to_submit, amount,
                payer_strict[payer.id], proc_strict[procedure], provider_q[provider.id],
            )
            is_denied = rng.random() < p_denied

            if is_denied:
                claim.status = ClaimStatus.DENIED
                if auth_missing:
                    reason = "MISSING_AUTHORIZATION"
                elif elig_issue:
                    reason = "ELIGIBILITY_ISSUE"
                elif doc_completeness < 70:
                    reason = "MISSING_DOCUMENTATION"
                elif days_to_submit > 45:
                    reason = "TIMELY_FILING"
                else:
                    reason = rng.choices(DENIAL_REASONS, k=1)[0]

                denial = DenialEvent(
                    claim_id=claim.id,
                    denial_reason_code=reason,
                    denial_date=submission_date + timedelta(days=rng.randint(5, 20)),
                    raw_reason_text=f"Synthetic denial: {reason.replace('_', ' ').title()}",
                )
                db.add(denial)
                db.flush()
                counts["denial_events"] += 1

                # Only ~55% of denials are actually appealed -- documented assumption #4.
                if rng.random() < 0.55:
                    appeal_p = APPEALABLE_REASONS[reason] * (doc_completeness / 100) * 0.9 + 0.05
                    won = rng.random() < appeal_p
                    outcome_amount = amount if won else 0
                    db.add(
                        AppealEvent(
                            denial_event_id=denial.id,
                            appeal_date=denial.denial_date + timedelta(days=rng.randint(3, 15)),
                            outcome="WON" if won else "LOST",
                            recovered_amount=outcome_amount,
                            decision_date=denial.denial_date + timedelta(days=rng.randint(20, 60)),
                        )
                    )
                    counts["appeal_events"] += 1
                    if won:
                        claim.status = ClaimStatus.RECOVERED
            else:
                claim.status = ClaimStatus.PAID

            if (i + 1) % 2000 == 0:
                db.commit()
                print(f"  ...{i + 1}/{n_claims} claims generated")

        db.commit()

        run = SyntheticDataRun(
            run_params={"n_claims": n_claims, "n_payers": n_payers, "n_providers": n_providers, "seed": seed},
            records_created=counts,
        )
        db.add(run)
        db.commit()

        print("Synthetic data generation complete:")
        for k, v in counts.items():
            print(f"  {k}: {v}")
        return counts
    finally:
        db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic RecoverAI claims data")
    parser.add_argument("--claims", type=int, default=2000)
    parser.add_argument("--payers", type=int, default=15)
    parser.add_argument("--providers", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    generate(args.claims, args.payers, args.providers, args.seed)
