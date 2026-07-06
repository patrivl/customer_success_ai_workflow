"""
Prompt templates.

These are the prompts that WOULD be sent to a real LLM in a production
version of this workflow. They are fully rendered with real joined data and
passed through llm_simulator.SimulatedLLMClient for token/cost logging, even
though the "response" is generated deterministically rather than by an
actual model call.
"""

from __future__ import annotations

ACCOUNT_BRIEFING_PROMPT = """You are a Customer Success strategist preparing a briefing for a CSM.

ACCOUNT
- Name: {account_name} ({segment})
- Contract value: ${contract_value:,.0f}
- Renewal date: {renewal_date}
- CSM owner: {csm_owner}
- Current health score: {current_health_score} (previous: {previous_health_score})
- Product usage trend: {product_usage_trend}
- Support tickets (last 30d): {support_ticket_count_30d}
- NPS: {nps_score}
- Expansion signal: {expansion_signal}
- Notes: {notes}

RECENT USAGE EVENTS
{usage_events_block}

OPEN SUPPORT TICKETS
{open_tickets_block}

MOST RECENT CALL NOTE
{call_note_block}

NEXT SCHEDULED CHECK-IN
{checkin_block}

TASK
Summarize this account's current situation in 2-4 sentences, call out the
material risks and/or opportunities, and recommend concrete next actions
for the CSM ahead of the next check-in. Be specific and reference the data
above rather than giving generic advice.
"""


QUALITY_REVIEW_PROMPT = """You are a Customer Success quality reviewer. Review a junior CSM's draft
output against a defined set of quality standards, using the underlying
account context as ground truth.

ACCOUNT CONTEXT
{account_context_block}

DRAFT OUTPUT TO REVIEW
- Output type: {output_type}
- Intended customer action: {intended_customer_action}
- Draft text: "{draft_text}"

QUALITY STANDARDS TO APPLY
{standards_block}

TASK
For each quality standard listed above, give a verdict of PASS, PARTIAL, or
FAIL with a one-sentence rationale grounded in the account context. Then
give an overall recommendation (Approved / Needs revision / Rejected) and,
if not Approved, a specific suggested revision.
"""


def render_account_briefing_prompt(**kwargs) -> str:
    return ACCOUNT_BRIEFING_PROMPT.format(**kwargs)


def render_quality_review_prompt(**kwargs) -> str:
    return QUALITY_REVIEW_PROMPT.format(**kwargs)


# ---------------------------------------------------------------------------
# Token-math-plan-driven prompt templates.
#
# These 9 templates back the model-routing/simulated-model layer described
# in config/token_math_plan.csv (src/token_math_config.py,
# src/model_simulator.py). Each one is rendered from a `StagePlan` (the
# looked-up config/token_math_plan.csv row for a stage_id) plus a compact
# `context` dict built from the deterministic preprocessing layer
# (src/preprocessing.py) or a directly-joined ticket/check-in/output record.
#
# Every template includes: the stage identity (stage_id/workflow_component/
# operating_area/trigger_schedule), a task objective, a compact input
# context block, the data fields the model is expected to draw on, the
# expected JSON output schema, a confidence-score requirement, any valid
# labels/routes, escalation/quality criteria where relevant, and explicit
# instructions against generic advice and in favor of citing evidence.
# ---------------------------------------------------------------------------

from src import output_schemas


def _stage_header(stage) -> str:
    return (
        f"stage_id: {stage.stage_id}\n"
        f"workflow_component: {stage.workflow_component}\n"
        f"operating_area: {stage.operating_area}\n"
        f"trigger_schedule: {stage.trigger_schedule}\n"
        f"model: {stage.model}"
    )


def _footer(schema_name: str, valid_labels: str = "", escalation_criteria: str = "",
            quality_criteria: str = "") -> str:
    parts = [
        "EXPECTED OUTPUT (JSON)",
        "{",
        output_schemas.schema_block(schema_name),
        "}",
        "",
        "CONFIDENCE SCORE REQUIREMENT",
        "Include a `confidence` field (0.0-1.0) that reflects how well the "
        "evidence above actually supports this output. Do not default to a "
        "fixed value -- ambiguous or thin evidence must produce a lower score.",
    ]
    if valid_labels:
        parts += ["", "VALID LABELS / ROUTES", valid_labels]
    if escalation_criteria:
        parts += ["", "ESCALATION CRITERIA", escalation_criteria]
    if quality_criteria:
        parts += ["", "QUALITY / EVALUATION CRITERIA", quality_criteria]
    parts += [
        "",
        "INSTRUCTIONS",
        "- Do not give generic recommendations; every recommendation/action must be "
        "specific to the evidence presented above.",
        "- Cite or account for the relevant account evidence (numbers, dates, "
        "quotes, ticket/segment ids) directly in the rationale.",
    ]
    return "\n".join(parts)


def _fmt_list(items, empty="none") -> str:
    items = list(items or [])
    return ", ".join(str(i) for i in items) if items else empty


def account_review_prompt(stage, context: dict) -> str:
    """TM_001-family: per-account risk/opportunity review, second-pass
    validation, flagged summaries, and urgent account-level intake."""
    flags = context.get("flags", {})
    scores = context.get("scores", {})
    open_tickets = context.get("open_tickets", [])
    upcoming_checkins = context.get("upcoming_checkins", [])

    body = f"""{_stage_header(stage)}

TASK OBJECTIVE
Review this account's current risk and opportunity posture and recommend a
concrete next action for its CSM. Reason from the signals below, not from
the account's segment or name.

INPUT CONTEXT
- account_id: {context.get('account_id', 'n/a')}
- account_name: {context.get('account_name', 'n/a')} ({context.get('segment', 'n/a')})
- csm_owner: {context.get('csm_owner', 'n/a')}
- contract_value: {context.get('contract_value', 'n/a')}
- current_health_score: {context.get('current_health_score', 'n/a')} (previous: {context.get('previous_health_score', 'n/a')}, delta: {context.get('health_score_delta', 'n/a')})
- product_usage_trend: {context.get('product_usage_trend', 'n/a')}
- nps_score: {context.get('nps_score', 'n/a')}
- expansion_signal: {context.get('expansion_signal', 'n/a')}
- renewal_days_remaining: {context.get('renewal_days_remaining', 'n/a')}
- open_tickets ({len(open_tickets)}): {_fmt_list(f"{t.get('ticket_id')}[{t.get('severity')}/{t.get('customer_sentiment')}]" for t in open_tickets)}
- upcoming_checkins ({len(upcoming_checkins)}): {_fmt_list(c.get('checkin_id') for c in upcoming_checkins)}
- notes: {context.get('notes', 'n/a')}
- deterministic flags: {_fmt_list(k for k, v in flags.items() if v)}
- risk_score={scores.get('risk_score', 'n/a')} opportunity_score={scores.get('opportunity_score', 'n/a')} escalation_score={scores.get('escalation_score', 'n/a')} priority_score={scores.get('priority_score', 'n/a')}
- selection reason: {context.get('selector_reason', 'n/a')}

DATA FIELDS USED
account_id, account_name, segment, csm_owner, contract_value,
current_health_score, previous_health_score, product_usage_trend,
nps_score, expansion_signal, renewal_days_remaining, open_tickets,
upcoming_checkins, notes, deterministic flags, risk_score, opportunity_score,
escalation_score, priority_score.

{_footer(
    "account_review_prompt",
    valid_labels=f"risk_level / opportunity_level: {', '.join(output_schemas.VALID_RISK_OPPORTUNITY_LEVELS)}",
    escalation_criteria="Escalate (recommend manager review / CSM follow-up) when a severe health "
    "decline coincides with a near-term renewal, or when negative sentiment appears on an open ticket "
    "close to renewal.",
)}
"""
    return body


def prioritization_prompt(stage, context: dict) -> str:
    """TM_003-family: cross-work-item priority ranking and rationale."""
    body = f"""{_stage_header(stage)}

TASK OBJECTIVE
Assign this item a priority rank and score within its batch, and explain
the ranking using the specific signals below (not a generic "this is
important" statement).

INPUT CONTEXT
- item_id: {context.get('item_id', context.get('account_id', 'n/a'))}
- item_type: {context.get('item_type', 'account')}
- account_id: {context.get('account_id', 'n/a')}
- priority_score: {context.get('priority_score', 'n/a')}
- risk_score: {context.get('risk_score', 'n/a')}
- opportunity_score: {context.get('opportunity_score', 'n/a')}
- escalation_score: {context.get('escalation_score', 'n/a')}
- selection reason: {context.get('selector_reason', 'n/a')}

DATA FIELDS USED
item_id, item_type, account_id, priority_score, risk_score,
opportunity_score, escalation_score, selection reason.

{_footer(
    "prioritization_prompt",
    escalation_criteria="If escalation_score materially exceeds risk_score, note that this item "
    "may need routing to a specialist/manager in owner_or_next_step rather than standard follow-up.",
)}
"""
    return body


def inbound_issue_prompt(stage, context: dict) -> str:
    """TM_012-family: inbound support ticket classification, triage, and
    response recommendation."""
    body = f"""{_stage_header(stage)}

TASK OBJECTIVE
Classify this inbound support ticket and recommend a specific customer
response, grounded in the ticket text and the owning account's context.

INPUT CONTEXT
- ticket_id: {context.get('ticket_id', 'n/a')}
- account_id: {context.get('account_id', 'n/a')} ({context.get('account_name', 'n/a')})
- severity (frontline): {context.get('severity', 'n/a')}
- customer_sentiment (frontline): {context.get('customer_sentiment', 'n/a')}
- issue_summary: {context.get('issue_summary', 'n/a')}
- frontline_notes: {context.get('frontline_notes', 'n/a')}
- current_status: {context.get('current_status', 'n/a')}
- date_received: {context.get('date_received', 'n/a')}
- account risk_score: {context.get('risk_score', 'n/a')} | renewal_days_remaining: {context.get('renewal_days_remaining', 'n/a')}

DATA FIELDS USED
ticket_id, account_id, severity, customer_sentiment, issue_summary,
frontline_notes, current_status, date_received, account risk_score,
renewal_days_remaining.

{_footer(
    "inbound_issue_prompt",
    valid_labels=f"route: {', '.join(output_schemas.VALID_ROUTES)}",
    escalation_criteria="Route to specialist_escalation/manager_escalation when severity is high AND "
    "customer_sentiment is negative or frustrated, or when the account's renewal_days_remaining is "
    "inside the urgent window.",
)}
"""
    return body


def checkin_support_prompt(stage, context: dict) -> str:
    """TM_018-family: pre/post check-in agenda, guidance, and follow-up."""
    open_tickets = context.get("open_tickets", [])
    body = f"""{_stage_header(stage)}

TASK OBJECTIVE
Prepare the CSM for this scheduled customer check-in: a concrete agenda,
talking points, and guidance grounded in the account's current signals.

INPUT CONTEXT
- checkin_id: {context.get('checkin_id', 'n/a')}
- account_id: {context.get('account_id', 'n/a')} ({context.get('account_name', 'n/a')})
- scheduled_date: {context.get('scheduled_date', 'n/a')}
- checkin_type: {context.get('checkin_type', 'n/a')}
- priority: {context.get('priority', 'n/a')}
- topics_to_cover: {context.get('topics_to_cover', 'n/a')}
- open_tickets ({len(open_tickets)}): {_fmt_list(f"{t.get('ticket_id')}[{t.get('severity')}]" for t in open_tickets)}
- risk_score: {context.get('risk_score', 'n/a')} | opportunity_score: {context.get('opportunity_score', 'n/a')}
- renewal_days_remaining: {context.get('renewal_days_remaining', 'n/a')}
- notes: {context.get('notes', 'n/a')}

DATA FIELDS USED
checkin_id, account_id, scheduled_date, checkin_type, priority,
topics_to_cover, open_tickets, risk_score, opportunity_score,
renewal_days_remaining, notes.

{_footer("checkin_support_prompt")}
"""
    return body


def quality_review_prompt(stage, context: dict) -> str:
    """TM_023-family: junior-output quality evaluation against standards,
    and intervention-plan validation (TM_031) using the same rubric shape."""
    standards = context.get("standards", [])
    standards_block = "\n".join(
        f"  - [{s.get('standard_id')}] {s.get('standard_name')}: {s.get('description')}"
        for s in standards
    ) or "  - (no standards attached -- use general CS quality judgement)"

    body = f"""{_stage_header(stage)}

TASK OBJECTIVE
Evaluate the draft/plan below against the listed quality standards, using
the account context as ground truth, and give a pass/fail verdict per
standard plus an overall quality score and routing decision.

INPUT CONTEXT
- output_id: {context.get('output_id', context.get('account_id', 'n/a'))}
- account_id: {context.get('account_id', 'n/a')}
- output_type: {context.get('output_type', 'n/a')}
- intended_customer_action: {context.get('intended_customer_action', 'n/a')}
- draft_text: "{context.get('draft_text', 'n/a')}"
- account risk_score: {context.get('risk_score', 'n/a')} | escalation_score: {context.get('escalation_score', 'n/a')}

QUALITY STANDARDS TO APPLY
{standards_block}

DATA FIELDS USED
output_id, account_id, output_type, intended_customer_action, draft_text,
quality standards (standard_id/name/description), account risk_score,
escalation_score.

{_footer(
    "quality_review_prompt",
    valid_labels=f"route: {', '.join(output_schemas.VALID_QUALITY_ROUTES)}",
    quality_criteria="passed=true only if every listed standard is met; any failed standard forces "
    "passed=false and route in {revise_and_resubmit, human_approval_required, escalate}.",
)}
"""
    return body


def intervention_planning_prompt(stage, context: dict) -> str:
    """TM_016/TM_028-family: segment- or account-level corrective action
    planning for accounts/issue clusters showing sustained decline."""
    flags = context.get("flags", {})
    body = f"""{_stage_header(stage)}

TASK OBJECTIVE
Design a specific, owned, time-boxed intervention for this declining
account/segment/issue-cluster -- not a generic "monitor and follow up".

INPUT CONTEXT
- account_or_segment_id: {context.get('account_or_segment_id', context.get('account_id', 'n/a'))}
- theme / cluster: {context.get('theme', 'n/a')}
- related_ticket_ids: {_fmt_list(context.get('ticket_ids', []))}
- declining_usage_flag: {flags.get('declining_usage_flag', 'n/a')}
- low_nps_flag: {flags.get('low_nps_flag', 'n/a')} (nps_score={context.get('nps_score', 'n/a')})
- negative_sentiment_flag: {flags.get('negative_sentiment_flag', 'n/a')}
- severe_decline_flag: {flags.get('severe_decline_flag', 'n/a')} (health_score_delta={context.get('health_score_delta', 'n/a')})
- risk_score: {context.get('risk_score', context.get('scores', {}).get('risk_score', 'n/a'))}
- selection reason: {context.get('selector_reason', 'n/a')}

DATA FIELDS USED
account_or_segment_id, theme/cluster, related_ticket_ids, declining_usage_flag,
low_nps_flag, negative_sentiment_flag, severe_decline_flag, risk_score,
selection reason.

{_footer(
    "intervention_planning_prompt",
    quality_criteria="intervention_actions must each name an owner and a timeline; success_measures "
    "must be observable (a metric or event), not a feeling.",
)}
"""
    return body


def routing_prompt(stage, context: dict) -> str:
    """TM_006/TM_011/TM_015/TM_025/TM_034-family: route an item to its
    correct owner/handling path."""
    body = f"""{_stage_header(stage)}

TASK OBJECTIVE
Decide the correct handling route and owner for this item, and state the
urgency, grounded in the evidence below.

INPUT CONTEXT
- item_id: {context.get('item_id', context.get('account_id', context.get('ticket_id', 'n/a')))}
- item_type: {context.get('item_type', 'n/a')}
- account_id: {context.get('account_id', 'n/a')}
- escalation_score: {context.get('escalation_score', 'n/a')}
- risk_score: {context.get('risk_score', 'n/a')}
- severity (if ticket): {context.get('severity', 'n/a')}
- customer_sentiment (if ticket): {context.get('customer_sentiment', 'n/a')}
- selection reason: {context.get('selector_reason', 'n/a')}

DATA FIELDS USED
item_id, item_type, account_id, escalation_score, risk_score, severity,
customer_sentiment, selection reason.

{_footer(
    "routing_prompt",
    valid_labels=f"route: {', '.join(output_schemas.VALID_ROUTES)}; "
                 f"urgency: {', '.join(output_schemas.VALID_URGENCY_LEVELS)}",
    escalation_criteria="urgency=immediate and route in {manager_escalation, specialist_escalation} "
    "when escalation_score crosses the alert threshold or severity is high with negative sentiment.",
)}
"""
    return body


def complex_escalation_prompt(stage, context: dict) -> str:
    """TM_037: premium-model review reserved for the highest-risk/most-
    ambiguous escalation cases (capped at 3/day upstream)."""
    flags = context.get("flags", {})
    open_tickets = context.get("open_tickets", [])
    body = f"""{_stage_header(stage)}

TASK OBJECTIVE
This case cleared the daily complex-escalation cap -- it is one of the
highest-risk or most ambiguous cases in the portfolio today. Give a full
root-cause assessment and a specific resolution recommendation; do not
default to "escalate to manager" without justification.

INPUT CONTEXT
- account_id: {context.get('account_id', 'n/a')} ({context.get('account_name', 'n/a')})
- escalation_score: {context.get('scores', {}).get('escalation_score', context.get('escalation_score', 'n/a'))}
- risk_score: {context.get('scores', {}).get('risk_score', context.get('risk_score', 'n/a'))}
- open_tickets ({len(open_tickets)}): {_fmt_list(f"{t.get('ticket_id')}[{t.get('severity')}/{t.get('customer_sentiment')}]" for t in open_tickets)}
- notes: {context.get('notes', 'n/a')}
- deterministic flags: {_fmt_list(k for k, v in flags.items() if v)}
- selection reason: {context.get('selector_reason', 'n/a')}

DATA FIELDS USED
account_id, escalation_score, risk_score, open_tickets, notes,
deterministic flags, selection reason.

{_footer(
    "complex_escalation_prompt",
    escalation_criteria="requires_manager_review=true only when the risk is account-level "
    "(renewal, contract, executive relationship) rather than a purely technical issue a specialist "
    "can resolve.",
)}
"""
    return body


def deployment_tracking_prompt(stage, context: dict) -> str:
    """TM_007/TM_021/TM_026/TM_032/TM_036-family: track whether a prior
    action/output/intervention resolved, and whether it should re-enter
    prioritization."""
    body = f"""{_stage_header(stage)}

TASK OBJECTIVE
Assess the current status of this previously-actioned item and decide
whether it needs to re-enter the prioritization queue.

INPUT CONTEXT
- item_id: {context.get('item_id', context.get('account_id', 'n/a'))}
- item_type: {context.get('item_type', 'n/a')}
- account_id: {context.get('account_id', 'n/a')}
- unresolved_follow_up_items: {context.get('unresolved_follow_up_items', 'n/a')}
- open_ticket_ids: {_fmt_list(context.get('open_ticket_ids', []))}
- days_since_last_contact: {context.get('days_since_last_contact', 'n/a')}
- selection reason: {context.get('selector_reason', 'n/a')}

DATA FIELDS USED
item_id, item_type, account_id, unresolved_follow_up_items, open_ticket_ids,
days_since_last_contact, selection reason.

{_footer(
    "deployment_tracking_prompt",
    escalation_criteria="reenter_prioritization=true when the item is still unresolved and "
    "days_since_last_contact exceeds the account's normal contact cadence.",
)}
"""
    return body


# name -> template function, used by src/model_simulator.py for dispatch.
PROMPT_TEMPLATES = {
    "account_review_prompt": account_review_prompt,
    "prioritization_prompt": prioritization_prompt,
    "inbound_issue_prompt": inbound_issue_prompt,
    "checkin_support_prompt": checkin_support_prompt,
    "quality_review_prompt": quality_review_prompt,
    "intervention_planning_prompt": intervention_planning_prompt,
    "routing_prompt": routing_prompt,
    "complex_escalation_prompt": complex_escalation_prompt,
    "deployment_tracking_prompt": deployment_tracking_prompt,
}


def render_prompt(template_name: str, stage, context: dict) -> str:
    """Dispatches to the right template function by name."""
    if template_name == "context_indexing":
        item_id = context.get("item_id", context.get("account_id", "n/a"))
        indexed_fields = sorted(k for k in context.keys() if not k.startswith("_"))
        char_count = len(str(context))
        return (
            f"{_stage_header(stage)}\n\n"
            f"TASK OBJECTIVE\nIndex compact context for item_id={item_id} so later stages retrieve "
            f"only what they need instead of the full record.\n\n"
            f"INDEXED FIELDS ({len(indexed_fields)}): {_fmt_list(indexed_fields)}\n"
            f"context_chars: {char_count}\n"
        )
    if template_name not in PROMPT_TEMPLATES:
        raise ValueError(f"Unknown prompt template '{template_name}'")
    return PROMPT_TEMPLATES[template_name](stage, context)
