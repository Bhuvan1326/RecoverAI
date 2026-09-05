"""
Guardrail engine (Feature 19).

This is the enforcement point every agent tool call and every AI-triggered
API path MUST pass through. It is deliberately NOT implemented as prompt
instructions -- prompts are advisory to the LLM and can be argued around;
this is a plain Python allow-list checked in code, independent of anything
the LLM decides to output. Attempting a blocked action raises
GuardrailViolation regardless of what the caller claims its intent is.
"""


class GuardrailViolation(Exception):
    def __init__(self, action: str):
        self.action = action
        super().__init__(f"Action '{action}' is not permitted for AI/agent execution.")


# Actions the AI/agent may execute autonomously (read/analyze/draft/recommend only).
ALLOWED_AI_ACTIONS: set[str] = {
    "analyze_claim",
    "get_claim",
    "get_claim_history",
    "get_denial",
    "retrieve_history",
    "predict_denial",
    "predict_denial_reason",
    "generate_explanation",
    "get_shap_explanation",
    "calculate_anomaly",
    "recommend_action",
    "retrieve_policy",
    "search_documents",
    "get_payer_policy",
    "calculate_recovery",
    "validate_claim",
    "predict_appeal_success",
    "draft_appeal",
    "create_appeal_draft",
    "create_workflow_action",  # creates a PENDING_APPROVAL row only -- never executes
}

# Actions that ALWAYS require a human decision, regardless of caller/context.
BLOCKED_AI_ACTIONS: set[str] = {
    "modify_claim",
    "submit_claim",
    "submit_appeal",
    "delete_claim",
    "approve_financial_action",
    "approve_workflow_action",
    "reject_workflow_action",
    "execute_workflow_action",
}


def check_action(action: str) -> None:
    """Raise GuardrailViolation unless `action` is explicitly allow-listed."""
    if action in BLOCKED_AI_ACTIONS:
        raise GuardrailViolation(action)
    if action not in ALLOWED_AI_ACTIONS:
        # Fail closed: unknown actions are blocked, not silently allowed.
        raise GuardrailViolation(action)


def guarded(action_name: str):
    """Decorator: wraps an agent tool function with a guardrail check."""

    def decorator(func):
        def wrapper(*args, **kwargs):
            check_action(action_name)
            return func(*args, **kwargs)

        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper

    return decorator
