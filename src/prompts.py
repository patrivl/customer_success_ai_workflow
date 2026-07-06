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
