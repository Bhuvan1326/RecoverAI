"""
Ranking metrics for the recovery-priority-queue evaluation (Section 11 of
the design doc: "Precision@K, NDCG@K, expected recovery captured@K").

These are pure functions over an already-ranked list of ground-truth
relevance values -- callers are responsible for producing the ranking
(sorted by whatever scoring strategy is being evaluated) and passing in
the relevance values in that ranked order.
"""
import math


def precision_at_k(ranked_relevance: list[float], k: int) -> float:
    """Fraction of the top-k ranked items that are relevant (relevance > 0).
    Binary treatment of relevance even if graded values are passed in."""
    if k <= 0 or not ranked_relevance:
        return 0.0
    top_k = ranked_relevance[:k]
    return sum(1 for r in top_k if r > 0) / len(top_k)


def dcg_at_k(ranked_relevance: list[float], k: int) -> float:
    """Discounted Cumulative Gain: sum(relevance_i / log2(i + 2)) for the
    top-k ranked items (i is 0-indexed position)."""
    top_k = ranked_relevance[:k]
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(top_k))


def ndcg_at_k(ranked_relevance: list[float], k: int) -> float:
    """Normalized DCG: DCG of the given ranking divided by the DCG of the
    IDEAL ranking (relevance values sorted descending) -- a value of 1.0
    means the ranking is perfect within the top k; 0.0 means no relevant
    items were surfaced at all. Returns 0.0 (not NaN) when there is no
    relevance signal to normalize against (e.g. every item has relevance 0)."""
    ideal = sorted(ranked_relevance, reverse=True)
    ideal_dcg = dcg_at_k(ideal, k)
    if ideal_dcg == 0:
        return 0.0
    return dcg_at_k(ranked_relevance, k) / ideal_dcg


def expected_recovery_captured_at_k(ranked_dollar_relevance: list[float], k: int) -> float:
    """
    Business-facing companion to NDCG: what fraction of the TOTAL actual
    recoverable dollars across the whole evaluation set would have been
    captured by working only the top-k ranked claims first. This is the
    "if we only had time to work K claims today, how much of the money did
    we get" metric -- directly answers whether the priority queue's
    ordering is actually worth following.
    """
    total = sum(ranked_dollar_relevance)
    if total <= 0:
        return 0.0
    return sum(ranked_dollar_relevance[:k]) / total


def evaluate_ranking(
    scores: list[float],
    binary_relevance: list[float],
    dollar_relevance: list[float],
    k_values: list[int],
) -> dict:
    """
    Sorts (binary_relevance, dollar_relevance) by `scores` descending (i.e.
    "if we ranked claims by this score, highest first"), then computes
    Precision@K / NDCG@K / expected-recovery-captured@K at each K.
    `scores`, `binary_relevance`, and `dollar_relevance` must be
    parallel lists (same order, same length) -- one entry per claim.
    """
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    ranked_binary = [binary_relevance[i] for i in order]
    ranked_dollar = [dollar_relevance[i] for i in order]

    return {
        f"precision_at_{k}": round(precision_at_k(ranked_binary, k), 4)
        for k in k_values
    } | {
        f"ndcg_at_{k}": round(ndcg_at_k(ranked_dollar, k), 4)
        for k in k_values
    } | {
        f"recovery_captured_at_{k}": round(expected_recovery_captured_at_k(ranked_dollar, k), 4)
        for k in k_values
    } | {
        "n_items": len(scores),
    }
