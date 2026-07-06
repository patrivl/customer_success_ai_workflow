"""
Simulated model layer driven by config/token_math_plan.csv.

IMPORTANT: no external AI API is ever called here. `simulate_model_call()`
looks up a stage's planned model/token/pricing/retry/QA assumptions from
config/token_math_plan.csv (via src/token_math_config.py), renders the
appropriate prompt template (src/prompts.py) from real context, and
produces a *deterministic* structured output grounded in that context
(src/preprocessing.py's flags/scores/reasons) -- never random filler.

STAGE_RUNTIME_MAP wires each of the 37 stage_ids in token_math_plan.csv to:
  - which deterministic-preprocessing population feeds it (a selector name
    from src/preprocessing.py's `selected_workflow_items` dict),
  - what kind of item that population contains (account/ticket/checkin/
    output/issue_theme), used by src/token_measurement.py to build the
    right per-item context, and
  - which of the 9 prompt templates renders that stage's calls.
"""

from __future__ import annotations

import json
from typing import Any, Dict

from src import config, output_schemas, prompts, token_costs
from src.token_math_config import StagePlan, get_stage, load_token_math_plan

# ---------------------------------------------------------------------------
# stage_id -> {population selector, item kind, prompt template}
# ---------------------------------------------------------------------------
STAGE_RUNTIME_MAP: Dict[str, Dict[str, str]] = {
    # Account review
    "TM_001": {"population": "daily_account_review", "item_kind": "account", "template": "context_indexing"},
    "TM_002": {"population": "daily_account_review", "item_kind": "account", "template": "account_review_prompt"},
    "TM_003": {"population": "daily_account_review", "item_kind": "account", "template": "prioritization_prompt"},
    "TM_004": {"population": "second_pass_validation", "item_kind": "account", "template": "account_review_prompt"},
    "TM_005": {"population": "flagged_account_summary", "item_kind": "account", "template": "account_review_prompt"},
    "TM_006": {"population": "csm_alerts", "item_kind": "account", "template": "routing_prompt"},
    "TM_007": {"population": "unresolved_items", "item_kind": "account", "template": "deployment_tracking_prompt"},
    # Prioritization
    "TM_008": {"population": "unresolved_items", "item_kind": "account", "template": "context_indexing"},
    "TM_009": {"population": "unresolved_items", "item_kind": "account", "template": "prioritization_prompt"},
    "TM_010": {"population": "flagged_account_summary", "item_kind": "account", "template": "prioritization_prompt"},
    "TM_011": {"population": "csm_alerts", "item_kind": "account", "template": "routing_prompt"},
    # Inbound issue handling
    "TM_012": {"population": "inbound_issues", "item_kind": "ticket", "template": "inbound_issue_prompt"},
    "TM_013": {"population": "inbound_issues", "item_kind": "ticket", "template": "context_indexing"},
    "TM_014": {"population": "inbound_issues", "item_kind": "ticket", "template": "inbound_issue_prompt"},
    "TM_015": {"population": "inbound_issues", "item_kind": "ticket", "template": "routing_prompt"},
    "TM_016": {"population": "issue_pattern_review", "item_kind": "issue_theme", "template": "intervention_planning_prompt"},
    # Customer check-in support
    "TM_017": {"population": "scheduled_checkins", "item_kind": "checkin", "template": "context_indexing"},
    "TM_018": {"population": "scheduled_checkins", "item_kind": "checkin", "template": "checkin_support_prompt"},
    "TM_019": {"population": "scheduled_checkins", "item_kind": "checkin", "template": "checkin_support_prompt"},
    "TM_020": {"population": "scheduled_checkins", "item_kind": "checkin", "template": "prioritization_prompt"},
    "TM_021": {"population": "scheduled_checkins", "item_kind": "checkin", "template": "deployment_tracking_prompt"},
    # Quality review
    "TM_022": {"population": "quality_review_outputs", "item_kind": "output", "template": "context_indexing"},
    "TM_023": {"population": "quality_review_outputs", "item_kind": "output", "template": "quality_review_prompt"},
    "TM_024": {"population": "failed_or_weak_outputs", "item_kind": "output", "template": "quality_review_prompt"},
    "TM_025": {"population": "failed_or_weak_outputs", "item_kind": "output", "template": "routing_prompt"},
    "TM_026": {"population": "failed_or_weak_outputs", "item_kind": "output", "template": "deployment_tracking_prompt"},
    # Targeted intervention planning
    "TM_027": {"population": "intervention_candidates", "item_kind": "account", "template": "context_indexing"},
    "TM_028": {"population": "intervention_candidates", "item_kind": "account", "template": "intervention_planning_prompt"},
    "TM_029": {"population": "intervention_candidates", "item_kind": "account", "template": "intervention_planning_prompt"},
    "TM_030": {"population": "intervention_candidates", "item_kind": "account", "template": "prioritization_prompt"},
    "TM_031": {"population": "intervention_candidates", "item_kind": "account", "template": "quality_review_prompt"},
    "TM_032": {"population": "intervention_candidates", "item_kind": "account", "template": "deployment_tracking_prompt"},
    # Routing for resolution, follow-up, or escalation
    "TM_033": {"population": "csm_alerts", "item_kind": "account", "template": "account_review_prompt"},
    "TM_034": {"population": "csm_alerts", "item_kind": "account", "template": "routing_prompt"},
    "TM_035": {"population": "csm_alerts", "item_kind": "account", "template": "prioritization_prompt"},
    "TM_036": {"population": "csm_alerts", "item_kind": "account", "template": "deployment_tracking_prompt"},
    "TM_037": {"population": "complex_escalation_candidates", "item_kind": "account", "template": "complex_escalation_prompt"},
}


def _score(context: dict, name: str, default=0.0):
    """Reads a score that may be flattened at the top level (ticket/checkin/
    output/issue_theme contexts) or nested under 'scores' (account contexts,
    which come straight from src/preprocessing.py)."""
    if name in context:
        return context[name]
    return context.get("scores", {}).get(name, default)


def _weak_draft(draft_text: str) -> bool:
    text = str(draft_text or "").lower()
    word_count = len(text.split())
    vague_hit = any(marker in text for marker in config.WEAK_OUTPUT_VAGUE_MARKERS)
    return word_count < config.WEAK_OUTPUT_MIN_WORD_COUNT or vague_hit


# ---------------------------------------------------------------------------
# Deterministic output generators, one per prompt template. Every generator
# reasons from context signals produced by src/preprocessing.py (flags,
# scores, score_reasons) or from directly-joined ticket/check-in/output
# fields -- never randomly.
# ---------------------------------------------------------------------------

def _generate_account_review_output(stage: StagePlan, context: dict) -> dict:
    flags = context.get("flags", {})
    reasons = context.get("score_reasons", {})
    risk_score = _score(context, "risk_score")
    opportunity_score = _score(context, "opportunity_score")

    risk_level = (
        "high" if risk_score >= config.HIGH_RISK_THRESHOLD
        else "medium" if risk_score >= config.MEDIUM_RISK_THRESHOLD
        else "low"
    )
    opp_high = config.OPPORTUNITY_ALERT_THRESHOLD
    opportunity_level = (
        "high" if opportunity_score >= opp_high
        else "medium" if opportunity_score >= opp_high / 2
        else "low"
    )

    key_signals = list(reasons.get("risk", []))[:3] + list(reasons.get("opportunity", []))[:2]
    if not key_signals:
        key_signals = [k for k, v in flags.items() if v][:5]

    if flags.get("severe_decline_flag") and flags.get("renewal_soon_flag"):
        recommended_action = (
            f"Escalate to manager for a renewal-risk review and schedule an urgent CSM "
            f"follow-up before the {context.get('renewal_days_remaining', 'n/a')}-day renewal window closes."
        )
    elif flags.get("renewal_soon_flag") and flags.get("health_decline_flag"):
        recommended_action = (
            f"Schedule a CSM follow-up ahead of the renewal (in {context.get('renewal_days_remaining', 'n/a')} "
            f"days) to directly address the {context.get('health_score_delta', 'n/a')}-point health decline."
        )
    elif flags.get("expansion_opportunity_flag") and str(context.get("product_usage_trend")) == "growing":
        recommended_action = (
            f"Prepare an expansion proposal: usage is growing and the expansion signal is "
            f"'{context.get('expansion_signal', 'n/a')}'."
        )
    elif flags.get("unresolved_issue_flag"):
        follow_up = context.get("unresolved_follow_up_items")
        ticket_ids = context.get("open_ticket_ids", [])
        detail = follow_up or (", ".join(ticket_ids) if ticket_ids else "an outstanding item")
        recommended_action = f"Resolve the outstanding item ({detail}) before the next customer touchpoint."
    elif flags.get("negative_sentiment_flag"):
        recommended_action = "Proactively reach out to address negative sentiment on the open ticket before it affects renewal."
    else:
        recommended_action = "Maintain standard cadence; no immediate action required beyond the next scheduled check-in."

    confidence = round(min(0.95, 0.45 + 0.12 * len(key_signals)), 2)

    rationale = (
        f"health_score_delta={context.get('health_score_delta', 'n/a')}, risk_score={risk_score}, "
        f"opportunity_score={opportunity_score}, renewal_days_remaining={context.get('renewal_days_remaining', 'n/a')}. "
        + ("; ".join(key_signals[:3]) if key_signals else "no material risk/opportunity signals present.")
    )

    return {
        "account_id": context.get("account_id"),
        "risk_level": risk_level,
        "opportunity_level": opportunity_level,
        "key_signals": key_signals,
        "summary": (
            f"{context.get('account_name', context.get('account_id'))} has a risk_score of {risk_score} "
            f"and an opportunity_score of {opportunity_score}. {rationale}"
        ),
        "recommended_action": recommended_action,
        "confidence": confidence,
        "rationale": rationale,
    }


def _generate_prioritization_output(stage: StagePlan, context: dict) -> dict:
    priority_score = _score(context, "priority_score")
    risk_score = _score(context, "risk_score")
    escalation_score = _score(context, "escalation_score")
    item_id = context.get("item_id") or context.get("account_id") or context.get("checkin_id")
    item_type = context.get("item_type", "account")

    if escalation_score >= config.ESCALATION_ALERT_THRESHOLD:
        owner_or_next_step = "Route to CSM manager for escalation review before ranking is finalized."
    elif risk_score >= config.MEDIUM_RISK_THRESHOLD:
        owner_or_next_step = "Assign to the primary CSM for a proactive check-in this cadence."
    else:
        owner_or_next_step = "Monitor at standard cadence; no owner change needed."

    reason = (
        f"priority_score={priority_score} (risk_score={risk_score}, escalation_score={escalation_score}); "
        f"{context.get('selector_reason', 'selected for this cadence')}."
    )
    confidence = round(min(0.95, 0.5 + priority_score / 200), 2)

    return {
        "item_id": item_id,
        "item_type": item_type,
        "priority_rank": context.get("priority_rank", 1),
        "priority_score": priority_score,
        "reason": reason,
        "owner_or_next_step": owner_or_next_step,
        "confidence": confidence,
    }


def _generate_inbound_issue_output(stage: StagePlan, context: dict) -> dict:
    severity = str(context.get("severity", "low")).lower()
    sentiment = str(context.get("customer_sentiment", "neutral")).lower()
    issue_summary = context.get("issue_summary", "")
    combined = f"{issue_summary} {context.get('frontline_notes', '')}"

    issue_type = "general_inquiry"
    for theme, keywords in config.ISSUE_THEME_KEYWORDS.items():
        if any(kw in combined.lower() for kw in keywords):
            issue_type = theme
            break

    severity_assessment = severity if severity in output_schemas.VALID_RISK_OPPORTUNITY_LEVELS else "medium"
    renewal_soon = (context.get("renewal_days_remaining") is not None
                    and context.get("renewal_days_remaining") <= config.RENEWAL_URGENT_WINDOW_DAYS)

    if severity == "high" and sentiment in ("frustrated", "negative"):
        recommended_response = (
            f"Immediate specialist response acknowledging the impact of '{issue_summary}', with a "
            f"same-day resolution ETA and a scheduled follow-up call."
        )
        route = "specialist_escalation"
    elif severity == "high":
        recommended_response = f"Prioritize a same-day technical response to '{issue_summary}'; confirm root cause before closing."
        route = "resolve_now"
    elif sentiment in ("frustrated", "negative", "concerned"):
        recommended_response = f"Respond within 24 hours with a specific update addressing '{issue_summary}', acknowledging the {sentiment} sentiment."
        route = "csm_review" if renewal_soon else "schedule_follow_up"
    else:
        recommended_response = f"Standard queue response addressing '{issue_summary}' within SLA."
        route = "schedule_follow_up"

    confidence = round(0.6 + (0.2 if severity == "high" else 0.0) + (0.1 if sentiment in ("frustrated", "negative") else 0.0), 2)
    rationale = (
        f"severity={severity}, customer_sentiment={sentiment}, renewal_days_remaining="
        f"{context.get('renewal_days_remaining', 'n/a')}. issue_summary=\"{issue_summary}\"."
    )

    return {
        "ticket_id": context.get("ticket_id"),
        "account_id": context.get("account_id"),
        "issue_type": issue_type,
        "severity_assessment": severity_assessment,
        "customer_sentiment": sentiment,
        "summary": f"Ticket {context.get('ticket_id')} ({context.get('account_id')}): {issue_summary}",
        "recommended_response": recommended_response,
        "route": route,
        "confidence": confidence,
        "rationale": rationale,
    }


def _generate_checkin_support_output(stage: StagePlan, context: dict) -> dict:
    open_tickets = context.get("open_tickets", [])
    topics = [t.strip() for t in str(context.get("topics_to_cover", "")).split(",") if t.strip()]

    agenda = list(topics) or ["Review account health and usage since last contact"]
    if open_tickets:
        agenda.append(f"Status update on {len(open_tickets)} open ticket(s)")
    if context.get("renewal_days_remaining") is not None and context.get("renewal_days_remaining") <= config.RENEWAL_URGENT_WINDOW_DAYS:
        agenda.append(f"Renewal conversation ({context['renewal_days_remaining']} days out)")

    talking_points = []
    risks_to_discuss = []
    opportunities_to_discuss = []

    if context.get("health_score_delta", 0) is not None and context.get("health_score_delta", 0) < 0:
        talking_points.append(f"Health score declined by {abs(context['health_score_delta'])} points -- ask what changed")
        risks_to_discuss.append("Health score decline")
    for t in open_tickets:
        risks_to_discuss.append(f"Open ticket {t.get('ticket_id')} ({t.get('severity')} severity)")
    if str(context.get("product_usage_trend")) == "growing":
        opportunities_to_discuss.append("Growing product usage -- explore expansion")
    if context.get("expansion_signal") in ("high", "medium"):
        opportunities_to_discuss.append(f"Expansion signal is {context.get('expansion_signal')}")

    if risks_to_discuss:
        suggested_guidance = f"Lead with the open risk(s) ({', '.join(risks_to_discuss[:2])}) before discussing growth."
    elif opportunities_to_discuss:
        suggested_guidance = f"Lead with the expansion opportunity ({opportunities_to_discuss[0]})."
    else:
        suggested_guidance = "Standard relationship check-in; no material risk or opportunity to lead with."

    follow_up_items = [f"Send recap and next steps for {context.get('checkin_id', 'this check-in')} within 2 business days"]

    confidence = round(0.55 + 0.1 * len(risks_to_discuss) + 0.1 * len(opportunities_to_discuss), 2)
    confidence = min(confidence, 0.95)

    return {
        "checkin_id": context.get("checkin_id"),
        "account_id": context.get("account_id"),
        "agenda": agenda,
        "talking_points": talking_points or ["No material change since last contact -- confirm goals are still on track"],
        "risks_to_discuss": risks_to_discuss,
        "opportunities_to_discuss": opportunities_to_discuss,
        "suggested_guidance": suggested_guidance,
        "follow_up_items": follow_up_items,
        "confidence": confidence,
    }


def _generate_quality_review_output(stage: StagePlan, context: dict) -> dict:
    draft_text = context.get("draft_text")
    standards = context.get("standards", [])

    if draft_text is not None:
        weak = _weak_draft(draft_text)
        failed_standards = [s.get("standard_id") for s in standards] if weak else []
        issues_found = ["draft is short and/or uses vague filler phrasing"] if weak else []
        quality_score = 35.0 if weak else 92.0
        correction_guidance = (
            "Rewrite with specific account evidence (numbers, dates, named next steps) instead of "
            "generic filler language." if weak else "No correction needed."
        )
    else:
        # TM_031: intervention-plan validation -- no draft_text, validate the
        # plan's evidentiary basis using the account's own decline signals.
        flags = context.get("flags", {})
        coherent = bool(flags.get("declining_usage_flag") or flags.get("severe_decline_flag"))
        failed_standards = [] if coherent else ["evidence_alignment"]
        issues_found = [] if coherent else ["intervention plan lacks a grounded decline signal to justify action"]
        quality_score = 88.0 if coherent else 45.0
        correction_guidance = (
            "Plan is grounded in observed decline signals." if coherent
            else "Tie every intervention action back to a specific flagged signal before approval."
        )

    passed = len(failed_standards) == 0
    route = "approve" if passed else ("human_approval_required" if quality_score < 50 else "revise_and_resubmit")
    confidence = 0.9 if passed else 0.65

    return {
        "output_id": context.get("output_id", context.get("account_id")),
        "account_id": context.get("account_id"),
        "passed": passed,
        "failed_standards": failed_standards,
        "quality_score": quality_score,
        "issues_found": issues_found,
        "correction_guidance": correction_guidance,
        "route": route,
        "confidence": confidence,
    }


def _generate_intervention_planning_output(stage: StagePlan, context: dict) -> dict:
    theme = context.get("theme")
    if theme:
        ticket_ids = context.get("ticket_ids", [])
        problem_pattern = f"Recurring '{theme}' issues across {len(ticket_ids)} ticket(s) suggest a systemic product/process gap."
        likely_causes = [f"Unresolved '{theme}' root cause affecting multiple accounts"]
        intervention_actions = [
            f"Support engineering lead to audit the '{theme}' issue cluster within 10 business days",
            "Product team to review whether a fix or documentation update is needed",
        ]
        owner = "Support engineering lead"
        timeline = "10 business days"
        success_measures = [f"New '{theme}' ticket volume drops below 1/week within 30 days"]
        risks = ["Theme recurs across additional accounts if root cause isn't addressed"]
        account_or_segment_id = theme
    else:
        flags = context.get("flags", {})
        causes = []
        if flags.get("declining_usage_flag"):
            causes.append("declining product usage")
        if flags.get("low_nps_flag"):
            causes.append(f"low NPS ({context.get('nps_score', 'n/a')})")
        if flags.get("negative_sentiment_flag"):
            causes.append("negative/concerned sentiment on an open ticket")
        if flags.get("severe_decline_flag"):
            causes.append(f"severe health decline ({context.get('health_score_delta', 'n/a')} points)")
        problem_pattern = f"Account shows sustained disengagement risk: {', '.join(causes) or 'declining usage'}."
        likely_causes = causes or ["insufficient recent engagement"]
        owner = context.get("csm_owner", "CSM team")
        intervention_actions = [
            f"{owner} to schedule an executive health-check call within 5 business days",
            "Solutions engineer to review recent product usage and blockers within 10 business days",
        ]
        timeline = "30 days"
        success_measures = [
            "current_health_score improves by at least 10 points within 60 days",
            "product_usage_trend shifts from declining to stable or growing",
        ]
        risks = ["Customer disengages further if outreach is delayed"]
        account_or_segment_id = context.get("account_id")

    risk_score = _score(context, "risk_score")
    confidence = round(min(0.95, 0.5 + risk_score / 150), 2)

    return {
        "account_or_segment_id": account_or_segment_id,
        "problem_pattern": problem_pattern,
        "likely_causes": likely_causes,
        "intervention_actions": intervention_actions,
        "owner": owner,
        "timeline": timeline,
        "success_measures": success_measures,
        "risks": risks,
        "confidence": confidence,
    }


def _generate_routing_output(stage: StagePlan, context: dict) -> dict:
    escalation_score = _score(context, "escalation_score")
    risk_score = _score(context, "risk_score")
    severity = str(context.get("severity", "")).lower()
    sentiment = str(context.get("customer_sentiment", "")).lower()
    item_id = context.get("item_id") or context.get("account_id") or context.get("ticket_id")
    item_type = context.get("item_type", "account")

    if escalation_score >= config.ESCALATION_ALERT_THRESHOLD or (severity == "high" and sentiment in ("negative", "frustrated")):
        route = "manager_escalation" if escalation_score >= config.ESCALATION_ALERT_THRESHOLD * 1.5 else "specialist_escalation"
        urgency = "immediate"
        owner = "CSM manager" if route == "manager_escalation" else "Support specialist"
    elif risk_score >= config.MEDIUM_RISK_THRESHOLD:
        route = "csm_review"
        urgency = "high"
        owner = context.get("csm_owner", "Primary CSM")
    else:
        route = "schedule_follow_up"
        urgency = "medium" if risk_score > 0 else "low"
        owner = context.get("csm_owner", "Primary CSM")

    reason = (
        f"escalation_score={escalation_score}, risk_score={risk_score}"
        + (f", severity={severity}, customer_sentiment={sentiment}" if severity else "")
        + f". {context.get('selector_reason', '')}"
    )
    confidence = round(min(0.95, 0.55 + escalation_score / 100), 2)

    return {
        "item_id": item_id,
        "item_type": item_type,
        "route": route,
        "owner": owner,
        "urgency": urgency,
        "reason": reason,
        "confidence": confidence,
    }


def _generate_complex_escalation_output(stage: StagePlan, context: dict) -> dict:
    flags = context.get("flags", {})
    reasons = context.get("score_reasons", {}).get("escalation", [])
    escalation_score = _score(context, "escalation_score")
    risk_score = _score(context, "risk_score")

    notes = context.get("notes", "")
    exec_language = any(kw in notes.lower() for kw in config.EXEC_RENEWAL_KEYWORDS)
    technical_language = any(kw in notes.lower() for kw in config.TECHNICAL_BLOCKER_KEYWORDS)

    if exec_language or flags.get("severe_decline_flag"):
        root_cause_assessment = "Account-level relationship/renewal risk, not a purely technical issue."
        requires_manager_review = True
        recommended_resolution = (
            "Manager-led account review: align on renewal terms and assign an executive sponsor "
            "touchpoint before the next customer contact."
        )
    elif technical_language:
        root_cause_assessment = "Technical blocker driving the escalation; likely resolvable by a specialist without manager involvement."
        requires_manager_review = False
        recommended_resolution = "Assign to a senior support specialist to resolve the underlying technical blocker within 48 hours."
    else:
        root_cause_assessment = "Escalation driven by accumulated ticket/sentiment signals rather than a single root cause."
        requires_manager_review = escalation_score >= config.ESCALATION_ALERT_THRESHOLD * 1.5
        recommended_resolution = "CSM to personally reach out and consolidate open items into one resolution plan within 3 business days."

    risk_if_unresolved = (
        f"contract_value=${context.get('contract_value', 'n/a'):,.0f}, "
        f"renewal_days_remaining={context.get('renewal_days_remaining', 'n/a')}" if isinstance(context.get("contract_value"), (int, float))
        else f"renewal_days_remaining={context.get('renewal_days_remaining', 'n/a')}"
    )

    escalation_summary = (
        f"escalation_score={escalation_score} cleared the daily complex-escalation cap. "
        + ("; ".join(reasons[:3]) if reasons else "Multiple compounding risk signals present.")
    )
    confidence = round(min(0.95, 0.6 + escalation_score / 150), 2)

    return {
        "item_id": context.get("account_id"),
        "account_id": context.get("account_id"),
        "escalation_summary": escalation_summary,
        "root_cause_assessment": root_cause_assessment,
        "recommended_resolution": recommended_resolution,
        "requires_manager_review": requires_manager_review,
        "risk_if_unresolved": risk_if_unresolved,
        "confidence": confidence,
    }


def _generate_deployment_tracking_output(stage: StagePlan, context: dict) -> dict:
    unresolved_follow_up = context.get("unresolved_follow_up_items")
    open_ticket_ids = context.get("open_ticket_ids", [])
    days_since_last_contact = context.get("days_since_last_contact")
    item_id = context.get("item_id") or context.get("account_id") or context.get("output_id") or context.get("checkin_id")
    item_type = context.get("item_type", "account")

    unresolved = bool(unresolved_follow_up) or bool(open_ticket_ids)
    current_status = "unresolved" if unresolved else "resolved"
    unresolved_risk = (
        f"Open item(s) still pending: {unresolved_follow_up or ', '.join(open_ticket_ids)}"
        if unresolved else "No unresolved items outstanding."
    )
    reenter_prioritization = bool(unresolved and days_since_last_contact is not None and days_since_last_contact > 14)

    return {
        "item_id": item_id,
        "current_status": current_status,
        "next_check_date_or_cycle": stage.cadence,
        "unresolved_risk": unresolved_risk,
        "reenter_prioritization": reenter_prioritization,
        "confidence": 0.85 if unresolved else 0.7,
    }


def _generate_context_indexing_output(stage: StagePlan, context: dict) -> dict:
    item_id = context.get("item_id") or context.get("account_id")
    indexed_fields = sorted(k for k in context.keys() if not k.startswith("_"))
    return {
        "item_id": item_id,
        "indexed_fields": indexed_fields,
        "context_chars": len(str(context)),
    }


GENERATORS = {
    "account_review_prompt": _generate_account_review_output,
    "prioritization_prompt": _generate_prioritization_output,
    "inbound_issue_prompt": _generate_inbound_issue_output,
    "checkin_support_prompt": _generate_checkin_support_output,
    "quality_review_prompt": _generate_quality_review_output,
    "intervention_planning_prompt": _generate_intervention_planning_output,
    "routing_prompt": _generate_routing_output,
    "complex_escalation_prompt": _generate_complex_escalation_output,
    "deployment_tracking_prompt": _generate_deployment_tracking_output,
    "context_indexing": _generate_context_indexing_output,
}


def simulate_model_call(stage_id: str, context: dict, prompt_template_name: str, run_id: str) -> Dict[str, Any]:
    """
    Simulates one "model call" for a planned workflow stage. No external API
    is ever called -- the "response" is produced by a deterministic
    generator grounded in `context` (see GENERATORS above).

    Looks up workflow_component/operating_area/trigger_schedule/model/
    planned tokens/prices/retry_rate/qa_eval_multiplier from
    config/token_math_plan.csv, renders the prompt, generates the output,
    measures actual token usage from the rendered text, and returns a full
    trace object with planned vs. measured cost/variance/review_flag.
    """
    stage = get_stage(stage_id)

    prompt_text = prompts.render_prompt(prompt_template_name, stage, context)

    generator = GENERATORS.get(prompt_template_name)
    if generator is None:
        raise ValueError(f"No output generator registered for template '{prompt_template_name}'")
    result = generator(stage, context)
    output_schemas.validate_output(prompt_template_name, result)

    measured_input_tokens = token_costs.estimate_tokens(prompt_text)
    measured_output_tokens = token_costs.estimate_tokens(json.dumps(result, default=str))

    planned_cost = token_costs.calculate_cost(
        stage.planned_input_tokens_per_run, stage.planned_output_tokens_per_run,
        stage.input_price_per_1m, stage.output_price_per_1m,
    )
    measured_cost = token_costs.calculate_cost(
        measured_input_tokens, measured_output_tokens,
        stage.input_price_per_1m, stage.output_price_per_1m,
    )
    adjusted_measured_cost = token_costs.calculate_adjusted_cost(
        measured_cost, stage.retry_rate, stage.qa_eval_multiplier,
    )
    variance_pct = token_costs.calculate_variance(planned_cost, measured_cost)
    review_flag = token_costs.assign_review_flag(variance_pct)

    return {
        "run_id": run_id,
        "stage_id": stage.stage_id,
        "workflow_component": stage.workflow_component,
        "operating_area": stage.operating_area,
        "trigger_schedule": stage.trigger_schedule,
        "model": stage.model,
        # Bookkeeping field (not part of the spec's original trace shape) so
        # downstream reporting (src/final_report.py) can group calls by
        # account without re-deriving it from `result`, whose id field name
        # varies by template (account_id / ticket_id / output_id / ...).
        "account_id": context.get("account_id"),
        "prompt_text": prompt_text,
        "result": result,
        "planned_input_tokens": stage.planned_input_tokens_per_run,
        "planned_output_tokens": stage.planned_output_tokens_per_run,
        "measured_input_tokens": measured_input_tokens,
        "measured_output_tokens": measured_output_tokens,
        "planned_cost": planned_cost,
        "measured_cost": measured_cost,
        "retry_rate": stage.retry_rate,
        "qa_eval_multiplier": stage.qa_eval_multiplier,
        "adjusted_measured_cost": adjusted_measured_cost,
        "variance_pct": None if variance_pct == float("inf") else round(variance_pct, 2),
        "review_flag": review_flag,
        "confidence": result.get("confidence"),
    }
