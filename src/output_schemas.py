"""
Expected output schemas for the simulated model layer (src/model_simulator.py).

Each schema is a plain ordered dict of field_name -> human-readable type/
description, used for two things:
  1. Rendered into the prompt text (src/prompts.py) so the "expected JSON
     output schema" section of every prompt is generated from a single
     source of truth instead of being hand-typed per template.
  2. Validating that a generated output actually contains every required
     field (`validate_output`), so a bug in the deterministic generator
     logic fails loudly instead of silently shipping a partial record.

Valid label sets (risk/opportunity levels, routes) are also centralized
here so every prompt/generator agrees on the same vocabulary.
"""

from __future__ import annotations

VALID_RISK_OPPORTUNITY_LEVELS = ["low", "medium", "high"]

VALID_ROUTES = [
    "no_action",
    "resolve_now",
    "schedule_follow_up",
    "csm_review",
    "manager_escalation",
    "specialist_escalation",
    "revise_and_resubmit",
    "human_approval_required",
]

VALID_URGENCY_LEVELS = ["low", "medium", "high", "immediate"]

VALID_QUALITY_ROUTES = ["approve", "revise_and_resubmit", "human_approval_required", "escalate"]


ACCOUNT_REVIEW_SCHEMA = {
    "account_id": "string",
    "risk_level": f"one of: {' | '.join(VALID_RISK_OPPORTUNITY_LEVELS)}",
    "opportunity_level": f"one of: {' | '.join(VALID_RISK_OPPORTUNITY_LEVELS)}",
    "key_signals": "array of strings (specific, evidence-based signals, not generic labels)",
    "summary": "string (2-4 sentences)",
    "recommended_action": "string (specific, concrete next action for the CSM)",
    "confidence": "number 0.0-1.0",
    "rationale": "string citing specific account evidence (numbers, dates, quotes)",
}

PRIORITIZATION_SCHEMA = {
    "item_id": "string",
    "item_type": "string (e.g. account | ticket | checkin | intervention)",
    "priority_rank": "integer, 1 = highest priority within this batch",
    "priority_score": "number",
    "reason": "string citing the specific signals driving this rank",
    "owner_or_next_step": "string",
    "confidence": "number 0.0-1.0",
}

INBOUND_ISSUE_SCHEMA = {
    "ticket_id": "string",
    "account_id": "string",
    "issue_type": "string",
    "severity_assessment": f"one of: {' | '.join(VALID_RISK_OPPORTUNITY_LEVELS)}",
    "customer_sentiment": "string (e.g. positive | neutral | concerned | negative | frustrated)",
    "summary": "string",
    "recommended_response": "string (specific, not generic)",
    "route": f"one of: {' | '.join(VALID_ROUTES)}",
    "confidence": "number 0.0-1.0",
    "rationale": "string citing specific ticket/account evidence",
}

CHECKIN_SUPPORT_SCHEMA = {
    "checkin_id": "string",
    "account_id": "string",
    "agenda": "array of strings",
    "talking_points": "array of strings",
    "risks_to_discuss": "array of strings",
    "opportunities_to_discuss": "array of strings",
    "suggested_guidance": "string",
    "follow_up_items": "array of strings",
    "confidence": "number 0.0-1.0",
}

QUALITY_REVIEW_SCHEMA = {
    "output_id": "string",
    "account_id": "string",
    "passed": "boolean",
    "failed_standards": "array of strings (standard_id values)",
    "quality_score": "number 0-100",
    "issues_found": "array of strings",
    "correction_guidance": "string (specific, actionable revision guidance)",
    "route": f"one of: {' | '.join(VALID_QUALITY_ROUTES)}",
    "confidence": "number 0.0-1.0",
}

INTERVENTION_PLANNING_SCHEMA = {
    "account_or_segment_id": "string",
    "problem_pattern": "string",
    "likely_causes": "array of strings",
    "intervention_actions": "array of strings (specific, owned, time-boxed)",
    "owner": "string",
    "timeline": "string",
    "success_measures": "array of strings",
    "risks": "array of strings",
    "confidence": "number 0.0-1.0",
}

ROUTING_SCHEMA = {
    "item_id": "string",
    "item_type": "string",
    "route": f"one of: {' | '.join(VALID_ROUTES)}",
    "owner": "string",
    "urgency": f"one of: {' | '.join(VALID_URGENCY_LEVELS)}",
    "reason": "string citing specific evidence",
    "confidence": "number 0.0-1.0",
}

# Complex escalation review (TM_037) is a premium-model synthesis step for
# the highest-risk/most-ambiguous cases only. Not given an explicit schema
# in the spec's section 3 list; this extends the routing schema with the
# extra judgement fields that a genuinely ambiguous escalation needs.
COMPLEX_ESCALATION_SCHEMA = {
    "item_id": "string",
    "account_id": "string",
    "escalation_summary": "string (why this case is ambiguous/high-risk enough for premium review)",
    "root_cause_assessment": "string",
    "recommended_resolution": "string (specific, not generic)",
    "requires_manager_review": "boolean",
    "risk_if_unresolved": "string",
    "confidence": "number 0.0-1.0",
}

DEPLOYMENT_TRACKING_SCHEMA = {
    "item_id": "string",
    "current_status": "string",
    "next_check_date_or_cycle": "string",
    "unresolved_risk": "string",
    "reenter_prioritization": "boolean",
    "confidence": "number 0.0-1.0",
}

# Signal-monitoring / context-assembly stages (embedding model, zero planned
# output tokens) don't produce a JSON verdict -- just a compact index record.
CONTEXT_INDEXING_SCHEMA = {
    "item_id": "string",
    "indexed_fields": "array of strings",
    "context_chars": "integer",
}

SCHEMAS = {
    "account_review_prompt": ACCOUNT_REVIEW_SCHEMA,
    "prioritization_prompt": PRIORITIZATION_SCHEMA,
    "inbound_issue_prompt": INBOUND_ISSUE_SCHEMA,
    "checkin_support_prompt": CHECKIN_SUPPORT_SCHEMA,
    "quality_review_prompt": QUALITY_REVIEW_SCHEMA,
    "intervention_planning_prompt": INTERVENTION_PLANNING_SCHEMA,
    "routing_prompt": ROUTING_SCHEMA,
    "complex_escalation_prompt": COMPLEX_ESCALATION_SCHEMA,
    "deployment_tracking_prompt": DEPLOYMENT_TRACKING_SCHEMA,
    "context_indexing": CONTEXT_INDEXING_SCHEMA,
}


def schema_block(template_name: str) -> str:
    """Renders a schema as an indented `field: description` text block for
    embedding directly into a prompt's 'expected output' section."""
    schema = SCHEMAS[template_name]
    return "\n".join(f'  "{field}": {desc}' for field, desc in schema.items())


class OutputValidationError(Exception):
    pass


def validate_output(template_name: str, output: dict) -> None:
    """Raises OutputValidationError if `output` is missing any field
    required by `template_name`'s schema."""
    schema = SCHEMAS[template_name]
    missing = [field for field in schema if field not in output]
    if missing:
        raise OutputValidationError(
            f"Generated output for template '{template_name}' is missing required "
            f"field(s): {missing}"
        )
