import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ml.evaluation.ranking_metrics import (
    dcg_at_k,
    evaluate_ranking,
    expected_recovery_captured_at_k,
    ndcg_at_k,
    precision_at_k,
)


def test_precision_at_k_perfect_ranking():
    # All relevant items already in the top 3.
    assert precision_at_k([1, 1, 1, 0, 0], 3) == 1.0


def test_precision_at_k_worst_ranking():
    assert precision_at_k([0, 0, 0, 1, 1], 3) == 0.0


def test_precision_at_k_partial():
    assert precision_at_k([1, 0, 1, 0], 2) == 0.5


def test_precision_at_k_k_larger_than_list():
    assert precision_at_k([1, 0], 10) == 0.5


def test_precision_at_k_empty_list():
    assert precision_at_k([], 5) == 0.0


def test_ndcg_at_k_perfect_ranking_is_one():
    # Descending order already matches the ideal ranking exactly.
    ranked = [10.0, 8.0, 5.0, 1.0]
    assert ndcg_at_k(ranked, 4) == 1.0


def test_ndcg_at_k_worst_ranking_is_less_than_one():
    ranked = [1.0, 5.0, 8.0, 10.0]  # ascending -- exact inverse of ideal
    score = ndcg_at_k(ranked, 4)
    assert 0.0 < score < 1.0


def test_ndcg_at_k_all_zero_relevance_returns_zero_not_nan():
    assert ndcg_at_k([0.0, 0.0, 0.0], 3) == 0.0


def test_ndcg_at_k_matches_hand_computed_value():
    # DCG@2 for [3, 2] = 3/log2(2) + 2/log2(3) = 3.0 + 1.2618... = 4.2618...
    # Ideal is the same ranking here, so NDCG@2 == 1.0.
    ranked = [3.0, 2.0]
    assert abs(dcg_at_k(ranked, 2) - (3.0 / 1.0 + 2.0 / (1.5849625))) < 1e-6
    assert ndcg_at_k(ranked, 2) == 1.0


def test_expected_recovery_captured_at_k_top_heavy_ranking():
    # Top-1 alone captures the biggest dollar amount.
    amounts = [1000.0, 100.0, 50.0]
    assert expected_recovery_captured_at_k(amounts, 1) == 1000.0 / 1150.0


def test_expected_recovery_captured_at_k_full_k_captures_everything():
    amounts = [1000.0, 100.0, 50.0]
    assert expected_recovery_captured_at_k(amounts, 3) == 1.0


def test_expected_recovery_captured_at_k_zero_total_returns_zero():
    assert expected_recovery_captured_at_k([0.0, 0.0], 1) == 0.0


def test_evaluate_ranking_good_score_beats_bad_score_on_same_data():
    """
    The core sanity check this whole eval exists for: a score that's
    perfectly correlated with actual relevance must outperform a score
    that's anti-correlated, on identical underlying data.
    """
    binary_relevance = [1, 0, 1, 0, 1]
    dollar_relevance = [500.0, 0.0, 800.0, 0.0, 300.0]

    good_scores = dollar_relevance  # ranks exactly by actual dollar value
    bad_scores = [-r for r in dollar_relevance]  # inverse ranking

    good = evaluate_ranking(good_scores, binary_relevance, dollar_relevance, k_values=[2])
    bad = evaluate_ranking(bad_scores, binary_relevance, dollar_relevance, k_values=[2])

    assert good["precision_at_2"] >= bad["precision_at_2"]
    assert good["ndcg_at_2"] > bad["ndcg_at_2"]
    assert good["recovery_captured_at_2"] > bad["recovery_captured_at_2"]


def test_evaluate_ranking_reports_n_items():
    result = evaluate_ranking([1, 2, 3], [1, 0, 1], [10.0, 0.0, 20.0], k_values=[1, 2])
    assert result["n_items"] == 3
    assert "precision_at_1" in result
    assert "ndcg_at_2" in result
    assert "recovery_captured_at_1" in result
