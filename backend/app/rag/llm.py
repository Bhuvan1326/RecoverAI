"""
LLM provider abstraction (Section 39). Mirrors embeddings.py: the app must
work with zero external API keys. MockLLMProvider builds a fully templated,
non-hallucinating appeal draft using ONLY the structured facts and retrieved
chunks it's given -- it never invents anything, which is a strictly stronger
faithfulness guarantee than a real LLM and a fine default for a portfolio
demo. Set LLM_PROVIDER=anthropic + ANTHROPIC_API_KEY to use a real model.
"""
from abc import ABC, abstractmethod

from app.core.config import get_settings


class LLMProvider(ABC):
    @abstractmethod
    def draft_appeal(self, claim_facts: dict, denial_reason: str, retrieved_chunks: list[dict], missing_evidence: list[str]) -> dict: ...


class MockLLMProvider(LLMProvider):
    def draft_appeal(self, claim_facts: dict, denial_reason: str, retrieved_chunks: list[dict], missing_evidence: list[str]) -> dict:
        citations = [
            {"chunk_id": c["chunk_id"], "document_title": c["document_title"], "excerpt": c["text"][:220]}
            for c in retrieved_chunks
        ]
        policy_lines = "\n".join(f"- {c['document_title']}: {c['text'][:200]}..." for c in retrieved_chunks) or "- No policy documents retrieved."

        draft = (
            f"APPEAL DRAFT (synthetic demo — human review required before any submission)\n\n"
            f"Claim: {claim_facts.get('claim_number')}\n"
            f"Denial reason: {denial_reason}\n\n"
            f"Claim facts:\n"
            f"- Billed amount: ${claim_facts.get('claim_amount')}\n"
            f"- Service date: {claim_facts.get('service_date')}\n"
            f"- Documentation completeness: {claim_facts.get('documentation_completeness')}%\n"
            f"- Authorization status: {claim_facts.get('authorization_status')}\n\n"
            f"Relevant policy references:\n{policy_lines}\n\n"
            f"Argument: Based on the claim facts above and the referenced payer policy, "
            f"this claim should be reconsidered for the reason on file ({denial_reason}). "
            f"All factual statements above are drawn directly from claim records and the "
            f"retrieved policy excerpts cited; no facts have been added beyond what is in the record.\n\n"
        )
        if missing_evidence:
            draft += f"Missing evidence still required: {', '.join(missing_evidence)}\n"
        draft += "\nThis draft requires human review and approval before submission. RecoverAI does not submit appeals autonomously."

        return {"draft_text": draft, "citations": citations, "missing_evidence": missing_evidence}


class AnthropicLLMProvider(LLMProvider):
    def __init__(self, api_key: str):
        self.api_key = api_key

    def draft_appeal(self, claim_facts: dict, denial_reason: str, retrieved_chunks: list[dict], missing_evidence: list[str]) -> dict:
        import anthropic  # imported lazily so it's not a hard dependency

        client = anthropic.Anthropic(api_key=self.api_key)
        context = "\n\n".join(f"[{c['chunk_id']}] {c['document_title']}: {c['text']}" for c in retrieved_chunks)
        prompt = (
            "You are drafting a healthcare claim appeal. Use ONLY the claim facts and "
            "retrieved policy excerpts below. Never invent facts not present here. "
            "Cite chunk IDs inline like [chunk_id] for every factual claim.\n\n"
            f"Claim facts: {claim_facts}\nDenial reason: {denial_reason}\n\n"
            f"Retrieved policy excerpts:\n{context}\n\n"
            f"Missing evidence (flag these as gaps, do not fabricate): {missing_evidence}"
        )
        resp = client.messages.create(
            model="claude-sonnet-4-6", max_tokens=1200, messages=[{"role": "user", "content": prompt}]
        )
        text = "".join(block.text for block in resp.content if hasattr(block, "text"))
        citations = [{"chunk_id": c["chunk_id"], "document_title": c["document_title"], "excerpt": c["text"][:220]} for c in retrieved_chunks]
        return {"draft_text": text, "citations": citations, "missing_evidence": missing_evidence}


def get_llm_provider() -> LLMProvider:
    settings = get_settings()
    if settings.LLM_PROVIDER == "anthropic" and settings.ANTHROPIC_API_KEY:
        return AnthropicLLMProvider(settings.ANTHROPIC_API_KEY)
    return MockLLMProvider()
