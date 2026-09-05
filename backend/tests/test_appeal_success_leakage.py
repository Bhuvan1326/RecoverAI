import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd

from ml.features.build_appeal_features import (
    ALL_FEATURES,
    FORBIDDEN_POST_RESOLUTION_FIELDS,
    add_expanding_historical_appeal_rates,
)


def test_no_post_resolution_field_in_feature_set():
    """
    Critical leakage guard (C2): outcome, recovered_amount, and decision_date
    are only known AFTER an appeal is resolved. If any of these ever leak
    into ALL_FEATURES, the model would be trained on information it
    wouldn't have at prediction time -- this test exists specifically to
    catch that regression.
    """
    assert FORBIDDEN_POST_RESOLUTION_FIELDS.isdisjoint(set(ALL_FEATURES))


def test_forbidden_fields_are_the_expected_ones():
    assert FORBIDDEN_POST_RESOLUTION_FIELDS == {"outcome", "recovered_amount", "decision_date", "appeal_success"}


def test_appeal_timing_is_allowed_pre_resolution_feature():
    """appeal_timing_days (when the biller chose to file) is knowable BEFORE
    the outcome and is legitimately allowed as a feature -- distinguishing
    this from the forbidden post-resolution fields is the point of this test."""
    assert "appeal_timing_days" in ALL_FEATURES


def test_expanding_appeal_rate_uses_neutral_prior_for_first_observations():
    df = pd.DataFrame(
        [
            {"payer_id": "p1", "denial_reason": "MISSING_AUTHORIZATION", "appeal_date": pd.Timestamp("2026-01-01"), "appeal_success": 1},
            {"payer_id": "p1", "denial_reason": "MISSING_AUTHORIZATION", "appeal_date": pd.Timestamp("2026-01-02"), "appeal_success": 1},
        ]
    )
    out = add_expanding_historical_appeal_rates(df)
    # First appeal for payer p1 has seen zero prior appeals -> neutral prior, not 0 or 1.
    assert out.iloc[0]["historical_payer_appeal_rate"] == 0.40


def test_expanding_appeal_rate_only_uses_past_appeals():
    rows = []
    for i in range(6):
        rows.append(
            {
                "payer_id": "p1",
                "denial_reason": "OTHER",
                "appeal_date": pd.Timestamp("2026-01-01") + pd.Timedelta(days=i),
                "appeal_success": 1,  # every prior appeal won
            }
        )
    # A 7th appeal, this one lost -- shouldn't affect the rate computed for it.
    rows.append({"payer_id": "p1", "denial_reason": "OTHER", "appeal_date": pd.Timestamp("2026-01-10"), "appeal_success": 0})
    df = pd.DataFrame(rows)
    out = add_expanding_historical_appeal_rates(df)
    # By the 7th row, 6 prior appeals were seen (all won) -> rate should be 1.0,
    # NOT influenced by this row's own (losing) outcome.
    assert out.iloc[6]["historical_payer_appeal_rate"] == 1.0
