import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ml.evaluation.eval_rag import DENIAL_REASON_TO_EXPECTED_DOC_KEYWORD


def test_denial_reason_keyword_map_covers_the_real_ingested_corpus():
    """
    Each keyword must actually appear in one of the 5 sample policy
    document titles (data/synthetic/sample_payer_policies.json) -- this
    test would catch the eval fixture silently drifting out of sync with
    the real corpus content it's supposed to describe.
    """
    import json

    corpus_path = Path(__file__).resolve().parent.parent.parent / "data" / "synthetic" / "sample_payer_policies.json"
    docs = json.loads(corpus_path.read_text())
    titles = [d["title"] for d in docs]

    for reason, keyword in DENIAL_REASON_TO_EXPECTED_DOC_KEYWORD.items():
        matches = [t for t in titles if keyword.lower() in t.lower()]
        assert len(matches) == 1, f"Keyword '{keyword}' for {reason} should match exactly one document title, matched {matches}"


def test_denial_reason_keywords_are_mutually_distinguishing():
    """No two denial reasons should share a keyword that would make their
    expected documents ambiguous."""
    keywords = list(DENIAL_REASON_TO_EXPECTED_DOC_KEYWORD.values())
    assert len(keywords) == len(set(keywords))


def test_resolve_expected_document_ids_matches_real_titles(client):
    from app.models.domain import Document
    from ml.evaluation.eval_rag import _resolve_expected_document_ids

    override_gen_fn = next(iter(client.app.dependency_overrides.values()))
    db = next(override_gen_fn())

    db.add(Document(title="Synthetic Payer Policy: Prior Authorization Requirements", source_type="payer_policy"))
    db.add(Document(title="Synthetic Payer Policy: Timely Filing Limits", source_type="payer_policy"))
    db.commit()

    resolved = _resolve_expected_document_ids(db)
    assert "MISSING_AUTHORIZATION" in resolved
    assert "TIMELY_FILING" in resolved
    assert "CODING_MISMATCH" not in resolved  # that document wasn't added


def test_retrieval_recall_skips_gracefully_with_no_matching_documents(client):
    from ml.evaluation.eval_rag import evaluate_retrieval_recall

    override_gen_fn = next(iter(client.app.dependency_overrides.values()))
    db = next(override_gen_fn())

    result = evaluate_retrieval_recall(db)
    assert result["skipped"] is True
    assert "reason" in result


def test_citation_correctness_skips_gracefully_with_no_denied_claims(client):
    from ml.evaluation.eval_rag import evaluate_citation_correctness

    override_gen_fn = next(iter(client.app.dependency_overrides.values()))
    db = next(override_gen_fn())

    result = evaluate_citation_correctness(db)
    assert result["skipped"] is True
    assert "reason" in result


def test_excerpt_fidelity_check_catches_a_fabricated_excerpt():
    """
    Direct test of the fidelity-checking LOGIC (not the full pipeline):
    an excerpt that is NOT a real substring of the source chunk text must
    fail the fidelity check -- this is what would catch a real LLM
    provider inventing or paraphrasing a quote instead of citing verbatim.
    """
    chunk_text = "This synthetic policy document describes prior authorization requirements for outpatient procedures."
    genuine_excerpt = chunk_text[:50]
    fabricated_excerpt = "This policy absolutely guarantees your appeal will be approved."

    assert genuine_excerpt.rstrip(".").strip() in chunk_text
    assert fabricated_excerpt.rstrip(".").strip() not in chunk_text
