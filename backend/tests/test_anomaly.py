import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pytest

from app.services import anomaly as anomaly_service


def test_score_claim_raises_when_no_model_trained(db_engine):
    from sqlalchemy.orm import sessionmaker

    Session = sessionmaker(bind=db_engine)
    db = Session()
    with pytest.raises(anomaly_service.AnomalyModelNotTrainedError):
        anomaly_service.score_claim(db, "nonexistent-claim-id")


def test_severity_thresholds_are_monotonic():
    assert anomaly_service.severity_from_score(95) == "HIGH"
    assert anomaly_service.severity_from_score(60) == "MEDIUM"
    assert anomaly_service.severity_from_score(35) == "LOW"
    assert anomaly_service.severity_from_score(5) == "NORMAL"


def test_score_claim_missing_from_feature_set_returns_normal_not_error(db_engine):
    """A claim with no lines yet (not in the anomaly feature frame) should
    return a safe NORMAL/zero result, never raise or return a fake score."""
    from sqlalchemy.orm import sessionmaker

    from app.models.domain import ModelVersion

    Session = sessionmaker(bind=db_engine)
    db = Session()
    # Register a fake champion model version to get past the "not trained" check
    # without needing to actually train one for this narrow unit test.
    db.add(
        ModelVersion(
            model_name="isolation_forest",
            model_type="anomaly_detection",
            version_tag="test",
            artifact_path="/nonexistent/path.joblib",
            is_champion=True,
        )
    )
    db.commit()

    # score_claim will try to load the artifact only if the claim IS in the
    # feature frame; since load_anomaly_frame() finds no claims at all here,
    # it should short-circuit before touching the (nonexistent) artifact.
    result = anomaly_service.score_claim(db, "some-claim-id")
    assert result["is_anomaly"] is False
    assert result["severity"] == "NORMAL"
    assert result["anomaly_score"] == 0
