"""
Stage 1: Account prioritization + AI-style briefing generation.

For every account, we:
  1. Compute a deterministic risk_score and opportunity_score from the
     joined data (see config.py for the weightings).
  2. Render the account_briefing prompt template with real data.
  3. Generate a deterministic "simulated LLM" narrative + recommended
     actions from that same data (via llm_simulator.SimulatedLLMClient so
     the call is metered like a real one).
  4. Rank all accounts by risk_score so the highest-priority accounts
     surface first.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import List

from src import config, prompts
from src.data_loader import AccountContext
from src.llm_simulator import SimulatedLLMClient


# ---------------------------------------------------------------------------
# Deterministic scoring
# ---------------------------------------------------------------------------

def _days_until(date_str: str, reference: datetime) -> int:
    try:
        target = datetime.strptime(str(date_str), "%Y-%m-%d")
    except (ValueError, TypeError):
        return 9999
    return (target - reference).days


def compute_risk_score(ctx: AccountContext, reference_date: datetime) -> float:
    acc = ctx.account
    score = 0.0

    score += (100 - float(acc["current_health_score"])) * config.HEALTH_SCORE_WEIGHT

    decline = float(acc["previous_health_score"]) - float(acc["current_health_score"])
    score += max(0.0, decline) * config.HEALTH_DECLINE_WEIGHT

    score += float(acc["support_ticket_count_30d"]) * config.TICKET_COUNT_WEIGHT

    score += (10 - float(acc["nps_score"])) * config.NPS_GAP_WEIGHT

    for ticket in ctx.open_tickets:
        severity = str(ticket.get("severity", "")).lower()
        sentiment = str(ticket.get("customer_sentiment", "")).lower()
        score += config.SEVERITY_WEIGHTS.get(severity, 0)
        score += config.SENTIMENT_WEIGHTS.get(sentiment, 0)

    expansion = str(acc.get("expansion_signal", "")).lower()
    score += config.EXPANSION_RISK_OFFSET.get(expansion, 0)

    days_to_renewal = _days_until(acc.get("renewal_date"), reference_date)
    if 0 <= days_to_renewal <= config.RENEWAL_URGENT_WINDOW_DAYS:
        # Renewal urgency amplifies whatever risk already exists.
        score *= 1.25

    return round(score, 2)


def compute_opportunity_score(ctx: AccountContext) -> float:
    acc = ctx.account
    score = 0.0
    expansion = str(acc.get("expansion_signal", "")).lower()
    score += {"high": 20, "medium": 10, "low": 0}.get(expansion, 0)
    if str(acc.get("product_usage_trend", "")).lower() == "growing":
        score += 10
    score += max(0.0, float(acc["nps_score"]) - 7) * 5
    score += max(0.0, float(acc["current_health_score"]) - 80) * 0.5
    return round(score, 2)


def priority_tier(risk_score: float) -> str:
    if risk_score >= config.HIGH_RISK_THRESHOLD:
        return "High"
    if risk_score >= config.MEDIUM_RISK_THRESHOLD:
        return "Medium"
    return "Low"


def days_to_renewal_str(acc: dict, reference_date: datetime) -> str:
    days = _days_until(acc.get("renewal_date"), reference_date)
    if days == 9999:
        return "unknown"
    if days < 0:
        return f"{abs(days)} days overdue"
    return f"{days} days"


# ---------------------------------------------------------------------------
# Prompt block builders
# ---------------------------------------------------------------------------

def _usage_events_block(ctx: AccountContext) -> str:
    if not ctx.usage_events:
        return "No recent usage events recorded."
    lines = []
    for ev in ctx.usage_events:
        lines.append(
            f"- {ev['event_date']}: {ev['active_users']} active users, "
            f"{ev['key_feature_users']} key-feature users, trend={ev['usage_trend']} "
            f"({ev['notable_change']})"
        )
    first, last = ctx.usage_events[0], ctx.usage_events[-1]
    delta = int(last["active_users"]) - int(first["active_users"])
    pct = (delta / int(first["active_users"]) * 100) if int(first["active_users"]) else 0.0
    lines.append(f"- Net change over period: {delta:+d} active users ({pct:+.1f}%)")
    return "\n".join(lines)


def _open_tickets_block(ctx: AccountContext) -> str:
    if not ctx.open_tickets:
        return "No open support tickets."
    lines = []
    for t in ctx.open_tickets:
        lines.append(
            f"- [{t['ticket_id']}] ({t['severity']}, sentiment={t['customer_sentiment']}, "
            f"status={t['current_status']}): {t['issue_summary']} "
            f"— frontline note: {t['frontline_notes']}"
        )
    return "\n".join(lines)


def _call_note_block(ctx: AccountContext) -> str:
    if not ctx.call_notes:
        return "No call notes on file."
    note = sorted(ctx.call_notes, key=lambda r: r.get("call_date", ""))[-1]
    return (
        f"{note['call_date']} with {note['participants']} — {note['summary']}. "
        f"Customer goal: {note['customer_goal']}. "
        f"Risk/blocker: {note['risk_or_blocker']}. "
        f"Follow-up items: {note['follow_up_items']}."
    )


def _checkin_block(ctx: AccountContext) -> str:
    if not ctx.upcoming_checkins:
        return "No check-in currently scheduled."
    checkin = sorted(ctx.upcoming_checkins, key=lambda r: r.get("scheduled_date", ""))[0]
    return (
        f"{checkin['scheduled_date']} — {checkin['checkin_type']} "
        f"(priority: {checkin['priority']}). Topics: {checkin['topics_to_cover']}."
    )


# ---------------------------------------------------------------------------
# Deterministic "simulated LLM" narrative generation
# ---------------------------------------------------------------------------

def _generate_summary_sentence(ctx: AccountContext, risk_score: float,
                                opportunity_score: float, reference_date: datetime) -> str:
    acc = ctx.account
    delta = float(acc["previous_health_score"]) - float(acc["current_health_score"])
    trend_phrase = (
        f"has declined {delta:.0f} points to {acc['current_health_score']}"
        if delta > 0
        else f"has improved {abs(delta):.0f} points to {acc['current_health_score']}"
        if delta < 0
        else f"is stable at {acc['current_health_score']}"
    )
    renewal_phrase = days_to_renewal_str(acc, reference_date)

    sentence = (
        f"{acc['account_name']} ({acc['segment']}, ${float(acc['contract_value']):,.0f} ARR) "
        f"health score {trend_phrase}, with {acc['support_ticket_count_30d']} support "
        f"ticket(s) in the last 30 days and an NPS of {acc['nps_score']}. "
        f"Renewal is in {renewal_phrase} ({acc['renewal_date']})."
    )
    return sentence


def _generate_risk_and_opportunity_notes(ctx: AccountContext) -> List[str]:
    notes = []
    acc = ctx.account

    if str(acc["product_usage_trend"]).lower() == "declining":
        notes.append(
            f"Usage is trending downward across the tracked window "
            f"({acc.get('notes', 'see account notes')})."
        )

    high_sev_open = [t for t in ctx.open_tickets if str(t["severity"]).lower() == "high"]
    if high_sev_open:
        summaries = "; ".join(t["issue_summary"] for t in high_sev_open)
        notes.append(f"High-severity issue(s) open: {summaries}.")

    negative_sentiment = [
        t for t in ctx.open_tickets
        if str(t["customer_sentiment"]).lower() in ("frustrated", "negative", "concerned")
    ]
    if negative_sentiment:
        notes.append(
            f"{len(negative_sentiment)} open ticket(s) carry negative/concerned sentiment, "
            f"indicating relationship risk beyond the technical issue itself."
        )

    if str(acc["expansion_signal"]).lower() in ("high", "medium") and not high_sev_open:
        notes.append(
            f"Expansion signal is {acc['expansion_signal']} — this is a candidate for a "
            f"growth conversation rather than pure risk mitigation."
        )

    if ctx.call_notes:
        latest = sorted(ctx.call_notes, key=lambda r: r.get("call_date", ""))[-1]
        if latest.get("follow_up_items"):
            notes.append(
                f"Outstanding follow-up from the last call ({latest['call_date']}): "
                f"{latest['follow_up_items']}."
            )

    if not notes:
        notes.append("No material risk signals beyond routine account health.")

    return notes


def _generate_recommended_actions(ctx: AccountContext, risk_score: float,
                                   reference_date: datetime) -> List[str]:
    acc = ctx.account
    actions = []

    days_to_renewal = _days_until(acc.get("renewal_date"), reference_date)
    renewal_urgent = 0 <= days_to_renewal <= config.RENEWAL_URGENT_WINDOW_DAYS

    high_sev_open = [t for t in ctx.open_tickets if str(t["severity"]).lower() == "high"]
    if high_sev_open:
        actions.append(
            "Escalate open high-severity ticket(s) to technical/support leadership "
            f"({', '.join(t['ticket_id'] for t in high_sev_open)}) before the next check-in."
        )

    if renewal_urgent and risk_score >= config.MEDIUM_RISK_THRESHOLD:
        actions.append(
            f"Flag to CSM leadership: renewal in {days_to_renewal} day(s) with elevated risk "
            f"(score {risk_score}) — prepare a save/renewal plan, not just a status update."
        )
    elif renewal_urgent:
        actions.append(
            f"Renewal is in {days_to_renewal} day(s); prepare a value/ROI summary ahead of "
            f"the renewal conversation."
        )

    if ctx.call_notes:
        latest = sorted(ctx.call_notes, key=lambda r: r.get("call_date", ""))[-1]
        if latest.get("follow_up_items"):
            actions.append(f"Close the loop on the outstanding commitment: {latest['follow_up_items']}.")

    if ctx.upcoming_checkins:
        checkin = sorted(ctx.upcoming_checkins, key=lambda r: r.get("scheduled_date", ""))[0]
        actions.append(
            f"Prepare talking points for the {checkin['checkin_type']} check-in on "
            f"{checkin['scheduled_date']}, covering: {checkin['topics_to_cover']}."
        )

    if str(acc["expansion_signal"]).lower() in ("high", "medium") and not high_sev_open:
        actions.append(
            "Draft an expansion proposal leveraging current usage strength — good timing "
            "independent of renewal-driven urgency."
        )

    if not actions:
        actions.append("Maintain standard cadence; no urgent action required this cycle.")

    return actions


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

@dataclass
class AccountBriefing:
    account_id: str
    account_name: str
    csm_owner: str
    risk_score: float
    opportunity_score: float
    priority_tier: str
    summary: str
    risk_and_opportunity_notes: List[str]
    recommended_actions: List[str]
    prompt_used: str = field(repr=False, default="")


def generate_account_briefing(ctx: AccountContext, client: SimulatedLLMClient,
                               reference_date: datetime) -> AccountBriefing:
    acc = ctx.account
    risk_score = compute_risk_score(ctx, reference_date)
    opportunity_score = compute_opportunity_score(ctx)

    prompt = prompts.render_account_briefing_prompt(
        account_name=acc["account_name"],
        segment=acc["segment"],
        contract_value=float(acc["contract_value"]),
        renewal_date=acc["renewal_date"],
        csm_owner=acc["csm_owner"],
        current_health_score=acc["current_health_score"],
        previous_health_score=acc["previous_health_score"],
        product_usage_trend=acc["product_usage_trend"],
        support_ticket_count_30d=acc["support_ticket_count_30d"],
        nps_score=acc["nps_score"],
        expansion_signal=acc["expansion_signal"],
        notes=acc["notes"],
        usage_events_block=_usage_events_block(ctx),
        open_tickets_block=_open_tickets_block(ctx),
        call_note_block=_call_note_block(ctx),
        checkin_block=_checkin_block(ctx),
    )

    def _response_fn() -> str:
        summary = _generate_summary_sentence(ctx, risk_score, opportunity_score, reference_date)
        notes = _generate_risk_and_opportunity_notes(ctx)
        actions = _generate_recommended_actions(ctx, risk_score, reference_date)
        return (
            summary + "\n"
            + "Risk/opportunity notes:\n- " + "\n- ".join(notes) + "\n"
            + "Recommended actions:\n- " + "\n- ".join(actions)
        )

    response_text = client.call(
        task="account_briefing",
        reference_id=ctx.account_id,
        prompt=prompt,
        response_fn=_response_fn,
    )

    summary = _generate_summary_sentence(ctx, risk_score, opportunity_score, reference_date)
    notes = _generate_risk_and_opportunity_notes(ctx)
    actions = _generate_recommended_actions(ctx, risk_score, reference_date)

    return AccountBriefing(
        account_id=ctx.account_id,
        account_name=acc["account_name"],
        csm_owner=acc["csm_owner"],
        risk_score=risk_score,
        opportunity_score=opportunity_score,
        priority_tier=priority_tier(risk_score),
        summary=summary,
        risk_and_opportunity_notes=notes,
        recommended_actions=actions,
        prompt_used=prompt,
    )


def generate_all_briefings(contexts: dict, client: SimulatedLLMClient,
                            reference_date: datetime) -> List[AccountBriefing]:
    briefings = [
        generate_account_briefing(ctx, client, reference_date)
        for ctx in contexts.values()
    ]
    briefings.sort(key=lambda b: b.risk_score, reverse=True)
    return briefings
