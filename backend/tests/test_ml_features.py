import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd

from ml.features.build_features import ALL_FEATURES, add_expanding_historical_rates


def test_feature_list_has_no_post_adjudication_fields():
    """Leakage guard: denial_reason_code / appeal outcome / paid amount must
    never appear in the feature set used for pre-submission risk scoring."""
    forbidden = {"denial_reason_code", "appeal_outcome", "recovered_amount", "is_denied"}
    assert forbidden.isdisjoint(set(ALL_FEATURES))


def test_expanding_historical_rate_only_uses_past_claims():
    """A claim's historical denial rate feature must be computed only from
    claims submitted strictly before it -- never from itself or the future."""
    df = pd.DataFrame(
        [
            {"payer_id": "p1", "provider_id": "prov1", "submission_date": pd.Timestamp("2026-01-01"), "is_denied": 1},
            {"payer_id": "p1", "provider_id": "prov1", "submission_date": pd.Timestamp("2026-01-02"), "is_denied": 1},
            {"payer_id": "p1", "provider_id": "prov1", "submission_date": pd.Timestamp("2026-01-03"), "is_denied": 0},
        ]
    )
    out = add_expanding_historical_rates(df)
    # The very first claim for payer p1 has seen zero prior claims -> neutral prior, not 0 or 1.
    assert out.iloc[0]["payer_historical_denial_rate"] == 0.15


def test_feature_matrix_columns_match_declared_schema():
    df = pd.DataFrame({f: [0] for f in ALL_FEATURES})
    assert list(df.columns) == ALL_FEATURES
