from collections import Counter, defaultdict

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.deps import get_current_user
from app.models.domain import Claim, DenialEvent, Payer, Provider, User
from app.services import anomaly as anomaly_service
from app.services.appeal_success import predict_appeal_success_batch
from app.services.recovery import expected_recovery_value

router = APIRouter(tags=["analytics"])

SIMULATED_LABEL = "SIMULATED — SYNTHETIC DEMO DATA"


@router.get("/dashboard/metrics")
def dashboard_metrics(db: Session = Depends(get_db), _user: User = Depends(get_current_user)):
    claims = db.execute(select(Claim)).scalars().all()
    denials = {d.claim_id: d for d in db.execute(select(DenialEvent)).scalars().all()}

    total_claims = len(claims)
    denied = [c for c in claims if c.id in denials]
    total_amount = sum(float(c.claim_amount) for c in claims)
    revenue_at_risk = sum(float(c.claim_amount) for c in denied)

    preventable_reasons = {"MISSING_AUTHORIZATION", "ELIGIBILITY_ISSUE", "MISSING_DOCUMENTATION", "CODING_MISMATCH"}
    preventable = sum(float(c.claim_amount) for c in denied if denials[c.id].denial_reason_code in preventable_reasons)

    expected_recovery_total = 0.0
    appeal_predictions = predict_appeal_success_batch(db, [(c, denials[c.id]) for c in denied])
    for c in denied:
        appeal = appeal_predictions[c.id]
        expected_recovery_total += expected_recovery_value(float(c.claim_amount), appeal["appeal_success_probability"])["expected_recovery_value"]

    denial_rate = (len(denied) / total_claims) if total_claims else 0
    clean_claim_rate = 1 - denial_rate

    return {
        "label": SIMULATED_LABEL,
        "total_claims": total_claims,
        "total_billed": round(total_amount, 2),
        "claims_at_risk": len(denied),
        "revenue_at_risk": round(revenue_at_risk, 2),
        "preventable_revenue": round(preventable, 2),
        "recoverable_revenue": round(revenue_at_risk - preventable, 2),
        "expected_recovery": round(expected_recovery_total, 2),
        "denial_rate": round(denial_rate, 4),
        "clean_claim_rate": round(clean_claim_rate, 4),
        "high_priority_claims": sum(1 for c in denied if float(c.claim_amount) > 5000),
        "pending_human_review": 0,  # filled from workflow_actions count in a live deployment
    }


@router.get("/payer-intelligence")
def payer_intelligence(db: Session = Depends(get_db), _user: User = Depends(get_current_user)):
    payers = {p.id: p for p in db.execute(select(Payer)).scalars().all()}
    claims = db.execute(select(Claim)).scalars().all()
    denials = {d.claim_id: d for d in db.execute(select(DenialEvent)).scalars().all()}

    by_payer = defaultdict(list)
    for c in claims:
        by_payer[c.payer_id].append(c)

    results = []
    for payer_id, payer_claims in by_payer.items():
        payer = payers.get(payer_id)
        denied = [c for c in payer_claims if c.id in denials]
        reason_counts = Counter(denials[c.id].denial_reason_code for c in denied)
        results.append(
            {
                "payer_id": payer_id,
                "payer_name": payer.name if payer else "Unknown",
                "claim_volume": len(payer_claims),
                "denial_rate": round(len(denied) / len(payer_claims), 4) if payer_claims else 0,
                "top_denial_reasons": reason_counts.most_common(3),
                "revenue_at_risk": round(sum(float(c.claim_amount) for c in denied), 2),
            }
        )
    results.sort(key=lambda r: r["denial_rate"], reverse=True)
    return {"label": SIMULATED_LABEL, "payers": results}


@router.get("/provider-intelligence")
def provider_intelligence(db: Session = Depends(get_db), _user: User = Depends(get_current_user)):
    providers = {p.id: p for p in db.execute(select(Provider)).scalars().all()}
    claims = db.execute(select(Claim)).scalars().all()
    denials = {d.claim_id: d for d in db.execute(select(DenialEvent)).scalars().all()}

    by_provider = defaultdict(list)
    for c in claims:
        by_provider[c.provider_id].append(c)

    results = []
    for provider_id, provider_claims in by_provider.items():
        provider = providers.get(provider_id)
        denied = [c for c in provider_claims if c.id in denials]
        reason_counts = Counter(denials[c.id].denial_reason_code for c in denied)
        results.append(
            {
                "provider_id": provider_id,
                "provider_name": provider.name if provider else "Unknown",
                "claim_volume": len(provider_claims),
                "denial_rate": round(len(denied) / len(provider_claims), 4) if provider_claims else 0,
                "top_denial_reasons": reason_counts.most_common(3),
                "revenue_at_risk": round(sum(float(c.claim_amount) for c in denied), 2),
            }
        )
    results.sort(key=lambda r: r["denial_rate"], reverse=True)
    return {"label": SIMULATED_LABEL, "providers": results}


@router.get("/denial-analytics")
def denial_analytics(db: Session = Depends(get_db), _user: User = Depends(get_current_user)):
    denials = db.execute(select(DenialEvent)).scalars().all()
    total = len(denials) or 1
    counts = Counter(d.denial_reason_code for d in denials)
    return {
        "label": SIMULATED_LABEL,
        "total_denials": len(denials),
        "root_causes": [{"reason": r, "count": c, "percent": round(100 * c / total, 1)} for r, c in counts.most_common()],
    }


@router.get("/analytics/anomalies")
def anomaly_analytics(
    severity: str | None = None,
    payer_id: str | None = None,
    provider_id: str | None = None,
    limit: int = 50,
    offset: int = 0,
    db: Session = Depends(get_db),
    _user: User = Depends(get_current_user),
):
    """
    Aggregate anomaly analytics (Feature A6). Scores every claim through the
    trained Isolation Forest model (real computation -- not a fixed/fake
    aggregate) and supports severity/payer/provider filtering + pagination.
    """
    try:
        mv, all_results = anomaly_service.score_all_claims(db)
    except anomaly_service.AnomalyModelNotTrainedError as e:
        return {"label": SIMULATED_LABEL, "error": str(e), "items": [], "total": 0}

    claims_by_id = {c.id: c for c in db.execute(select(Claim)).scalars().all()}
    results = []
    for r in all_results:
        c = claims_by_id.get(r["claim_id"])
        if not c:
            continue
        if payer_id and c.payer_id != payer_id:
            continue
        if provider_id and c.provider_id != provider_id:
            continue
        r["claim_number"] = c.claim_number
        r["payer_id"] = c.payer_id
        r["provider_id"] = c.provider_id
        results.append(r)

    if severity:
        results = [r for r in results if r["severity"] == severity]

    severity_counts = Counter(r["severity"] for r in results)
    anomalous = [r for r in results if r["is_anomaly"]]

    results.sort(key=lambda r: r["anomaly_score"], reverse=True)

    return {
        "label": SIMULATED_LABEL,
        "model_version": f"{mv.model_name}-{mv.version_tag}",
        "total_claims_scored": len(results),
        "anomaly_count": len(anomalous),
        "anomaly_percentage": round(100 * len(anomalous) / len(results), 2) if results else 0,
        "severity_distribution": dict(severity_counts),
        "total": len(results),
        "items": results[offset : offset + min(limit, 200)],
    }
