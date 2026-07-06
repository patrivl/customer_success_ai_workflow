"""
Stage 2: Quality review of junior_outputs.csv against quality_standards.csv.

For every (output, standard) pair produced by
data_loader.explode_quality_standards(), we:
  1. Render the quality_review prompt template with the draft + account
     context + standard description.
  2. Run a deterministic, rule-based "LLM-as-judge" heuristic (per standard)
     that produces a PASS / PARTIAL / FAIL verdict with a rationale grounded
     in the account data.
  3. Roll the per-standard verdicts up into an overall score and
     recommendation (Approved / Needs revision / Rejected), plus a
     suggested revision when the output isn't a clean approval.

Every heuristic below is generic (keyword/pattern based against whatever
account context is supplied) rather than hard-coded to specific accounts or
outputs, so it generalizes to new rows added to the same CSVs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import pandas as pd

from src import config, prompts
from src.data_loader import AccountContext
from src.llm_simulator import SimulatedLLMClient


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"[a-zA-Z']+")


def _keywords(text: str) -> set:
    words = _WORD_RE.findall(str(text).lower())
    return {w for w in words if len(w) >= 4 and w not in config.STOPWORDS}


VAGUE_PHRASES = [
    "let us know if you have questions", "reach out if", "sometime",
    "try again", "will update you soon", "we think", "checking the",
    "do you want to meet", "a lot this year",
]

ACTIONABILITY_SIGNALS = [
    r"\bby \w+ \d", r"\bnext \w+", r"\bwe will\b", r"\bi will\b", r"\bi'll\b",
    r"\bwe'll\b", r"\bphase\b", r"\bstep\b", r"\bschedule\b", r"\bescalat\w*",
    r"\bplan\b", r"\bprovide\b", r"\bsend\b", r"\bshare\b", r"\bcall\b",
    r"\bmetrics\b", r"\btraining\b", r"\bsetup\b",
]

RISK_ACK_WORDS = [
    "issue", "risk", "delay", "block", "problem", "urgent", "churn",
    "concern", "escalat", "critical", "unresolved", "fail",
]

ALARMIST_WORDS = ["probably going to churn", "immediately or", "critical failure"]

ESCALATION_WORDS = ["escalat", "priority", "urgent", "immediately", "leadership", "technical team"]


def _contains_any(text: str, phrases: List[str]) -> List[str]:
    text_lower = str(text).lower()
    hits = []
    for p in phrases:
        if re.search(p, text_lower):
            hits.append(p)
    return hits


# ---------------------------------------------------------------------------
# Shared "actual state" derivation from account context
# ---------------------------------------------------------------------------

def _actual_risk_level(ctx: AccountContext) -> str:
    acc = ctx.account
    decline = float(acc["previous_health_score"]) - float(acc["current_health_score"])
    high_sev_open = any(str(t["severity"]).lower() == "high" for t in ctx.open_tickets)
    negative_sentiment_open = any(
        str(t["customer_sentiment"]).lower() in ("frustrated", "negative")
        for t in ctx.open_tickets
    )
    if high_sev_open or decline >= 15 or negative_sentiment_open:
        return "high"
    medium_sev_open = any(str(t["severity"]).lower() == "medium" for t in ctx.open_tickets)
    if medium_sev_open or decline >= 5:
        return "medium"
    return "low"


def _context_keywords(ctx: AccountContext) -> set:
    acc = ctx.account
    parts = [acc.get("account_name", ""), acc.get("notes", "")]
    if ctx.call_notes:
        latest = sorted(ctx.call_notes, key=lambda r: r.get("call_date", ""))[-1]
        parts += [latest.get("customer_goal", ""), latest.get("risk_or_blocker", ""),
                  latest.get("follow_up_items", ""), latest.get("summary", "")]
    if ctx.upcoming_checkins:
        checkin = sorted(ctx.upcoming_checkins, key=lambda r: r.get("scheduled_date", ""))[0]
        parts.append(checkin.get("topics_to_cover", ""))
    for t in ctx.open_tickets:
        parts.append(t.get("issue_summary", ""))
    kw = set()
    for p in parts:
        kw |= _keywords(p)
    return kw


def _follow_up_items(ctx: AccountContext) -> str:
    if not ctx.call_notes:
        return ""
    latest = sorted(ctx.call_notes, key=lambda r: r.get("call_date", ""))[-1]
    return latest.get("follow_up_items", "") or ""


# ---------------------------------------------------------------------------
# Per-standard evaluators. Each returns (verdict, rationale).
# ---------------------------------------------------------------------------

def _eval_customer_specific_context(draft_text: str, ctx: AccountContext, output_row: dict) -> Tuple[str, str]:
    ctx_kw = _context_keywords(ctx)
    draft_kw = _keywords(draft_text)
    overlap = ctx_kw & draft_kw
    if len(overlap) >= 2:
        return "PASS", f"Draft references specific account context ({', '.join(sorted(overlap))})."
    if len(overlap) == 1:
        return "PARTIAL", f"Draft references only one specific account detail ({next(iter(overlap))}); mostly generic otherwise."
    return "FAIL", "Draft does not reference any specific account context (situation, history, or stated customer goal)."


def _eval_actionability(draft_text: str, ctx: AccountContext, output_row: dict) -> Tuple[str, str]:
    signals = _contains_any(draft_text, ACTIONABILITY_SIGNALS)
    vague = _contains_any(draft_text, VAGUE_PHRASES)
    if signals and not vague:
        return "PASS", f"Draft includes concrete next-step language ({len(signals)} actionability signal(s) found)."
    if signals and vague:
        return "PARTIAL", "Draft has some concrete language but also relies on vague phrasing without a clear owner/timing."
    if vague and not signals:
        return "FAIL", f"Draft is vague with no concrete owner, timing, or next step (e.g. '{vague[0]}')."
    return "FAIL", "Draft gives no concrete next steps, owner, or timing."


def _eval_risk_accuracy(draft_text: str, ctx: AccountContext, output_row: dict) -> Tuple[str, str]:
    actual = _actual_risk_level(ctx)
    ack_hits = _contains_any(draft_text, RISK_ACK_WORDS)
    alarmist_hits = _contains_any(draft_text, ALARMIST_WORDS)
    is_internal = "internal" in str(output_row.get("output_type", "")).lower()

    if actual == "high" and not ack_hits:
        return "FAIL", "Account shows material risk signals (high-severity ticket, sharp health decline, or negative sentiment) that the draft does not acknowledge."
    if actual == "low" and alarmist_hits and not is_internal:
        return "FAIL", "Account shows low material risk, but the draft uses alarmist language inappropriate for a customer-facing message."
    if actual == "high" and ack_hits:
        return "PASS", "Draft appropriately acknowledges the account's material risk signals."
    if actual in ("medium", "low"):
        return "PASS", f"Draft's risk framing is consistent with the account's {actual} current risk level."
    return "PARTIAL", "Risk framing is only partially consistent with the account's current risk signals."


def _eval_tone_and_clarity(draft_text: str, ctx: AccountContext, output_row: dict) -> Tuple[str, str]:
    word_count = len(str(draft_text).split())
    vague_hits = _contains_any(draft_text, VAGUE_PHRASES)
    is_internal = "internal" in str(output_row.get("output_type", "")).lower()

    if word_count < 6:
        return "FAIL", "Draft is too short to be useful/professional in context."
    if len(vague_hits) >= 2 and not is_internal:
        return "FAIL", "Draft relies on multiple generic/filler phrases, undermining professional clarity."
    if len(vague_hits) == 1 and not is_internal:
        return "PARTIAL", f"Draft is understandable but includes a generic phrase ('{vague_hits[0]}') that reduces polish."
    return "PASS", "Draft is concise and professionally toned for its intended audience."


def _eval_escalation_judgment(draft_text: str, ctx: AccountContext, output_row: dict) -> Tuple[str, str]:
    actual = _actual_risk_level(ctx)
    esc_hits = _contains_any(draft_text, ESCALATION_WORDS)
    if actual == "high" and not esc_hits:
        return "FAIL", "Account has a high-severity/urgent issue that the draft does not route toward escalation or specialist follow-up."
    if actual == "low" and esc_hits:
        return "FAIL", "Account risk is low, but the draft escalates language/urgency beyond what the situation warrants."
    if actual == "high" and esc_hits:
        return "PASS", "Draft appropriately signals escalation for a high-urgency issue."
    return "PASS", f"Escalation framing matches the account's {actual} urgency level."


def _eval_follow_up_continuity(draft_text: str, ctx: AccountContext, output_row: dict) -> Tuple[str, str]:
    follow_up = _follow_up_items(ctx)
    if not follow_up:
        return "PASS", "No prior follow-up commitments were on file to carry forward."
    follow_kw = _keywords(follow_up)
    draft_kw = _keywords(draft_text)
    overlap = follow_kw & draft_kw
    if overlap:
        return "PASS", f"Draft carries forward the prior commitment ({follow_up})."
    return "FAIL", f"Draft does not reference the outstanding follow-up commitment from the last call: '{follow_up}'."


STANDARD_EVALUATORS = {
    "QS001": _eval_customer_specific_context,
    "QS002": _eval_actionability,
    "QS003": _eval_risk_accuracy,
    "QS004": _eval_tone_and_clarity,
    "QS005": _eval_escalation_judgment,
    "QS006": _eval_follow_up_continuity,
}


# ---------------------------------------------------------------------------
# Suggested revision generation
# ---------------------------------------------------------------------------

def _suggest_revision(failed_or_partial: List[Tuple[str, str, str]], ctx: AccountContext,
                       output_row: dict) -> str:
    """failed_or_partial: list of (standard_id, standard_name, rationale)."""
    if not failed_or_partial:
        return ""
    acc = ctx.account
    fixes = []
    ids = {s_id for s_id, _, _ in failed_or_partial}

    if "QS001" in ids:
        fixes.append(
            f"reference the account's actual situation (e.g. '{acc.get('notes')}')"
        )
    if "QS002" in ids:
        fixes.append("add a concrete next step with a named owner and a specific date")
    if "QS003" in ids:
        fixes.append("adjust the risk framing to match the account's true current risk level")
    if "QS004" in ids:
        fixes.append("tighten the language and remove generic filler phrasing")
    if "QS005" in ids:
        fixes.append("align escalation language with the issue's actual severity")
    if "QS006" in ids:
        follow_up = _follow_up_items(ctx)
        fixes.append(f"explicitly reference the outstanding commitment: '{follow_up}'")

    return "Suggested revision: " + "; ".join(fixes) + "."


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

@dataclass
class StandardVerdict:
    standard_id: str
    standard_name: str
    verdict: str
    rationale: str


@dataclass
class QualityReview:
    output_id: str
    account_id: str
    output_type: str
    draft_text: str
    intended_customer_action: str
    standard_verdicts: List[StandardVerdict]
    overall_score: float
    recommendation: str
    suggested_revision: str
    prompt_used: str = field(repr=False, default="")


def _standards_block(rows: pd.DataFrame) -> str:
    lines = []
    for _, r in rows.iterrows():
        lines.append(f"- [{r['standard_id']}] {r['standard_name']}: {r['description']}")
    return "\n".join(lines)


def _account_context_block(ctx: AccountContext) -> str:
    acc = ctx.account
    lines = [
        f"Account: {acc['account_name']} ({acc['segment']}), health {acc['current_health_score']} "
        f"(prev {acc['previous_health_score']}), NPS {acc['nps_score']}, "
        f"renewal {acc['renewal_date']}, notes: {acc['notes']}",
    ]
    if ctx.open_tickets:
        lines.append("Open tickets: " + "; ".join(
            f"[{t['severity']}/{t['customer_sentiment']}] {t['issue_summary']}" for t in ctx.open_tickets
        ))
    if ctx.call_notes:
        latest = sorted(ctx.call_notes, key=lambda r: r.get("call_date", ""))[-1]
        lines.append(
            f"Last call ({latest['call_date']}): goal={latest['customer_goal']}, "
            f"risk/blocker={latest['risk_or_blocker']}, follow-up={latest['follow_up_items']}"
        )
    return "\n".join(lines)


def review_output(output_id: str, output_group: pd.DataFrame, ctx: AccountContext,
                   client: SimulatedLLMClient) -> QualityReview:
    """output_group: the exploded+joined rows (one per standard) for this output_id."""
    first = output_group.iloc[0]
    draft_text = first["draft_text"]
    output_row = first.to_dict()

    prompt = prompts.render_quality_review_prompt(
        account_context_block=_account_context_block(ctx),
        output_type=first["output_type"],
        intended_customer_action=first["intended_customer_action"],
        draft_text=draft_text,
        standards_block=_standards_block(output_group),
    )

    def _response_fn() -> str:
        lines = []
        for _, r in output_group.iterrows():
            verdict, rationale = STANDARD_EVALUATORS[r["standard_id"]](draft_text, ctx, output_row)
            lines.append(f"[{r['standard_id']}] {verdict}: {rationale}")
        return "\n".join(lines)

    client.call(task="quality_review", reference_id=output_id, prompt=prompt, response_fn=_response_fn)

    verdicts: List[StandardVerdict] = []
    failed_or_partial = []
    for _, r in output_group.iterrows():
        std_id = r["standard_id"]
        evaluator = STANDARD_EVALUATORS.get(std_id)
        if evaluator is None:
            verdict, rationale = "PARTIAL", "No automated evaluator defined for this standard."
        else:
            verdict, rationale = evaluator(draft_text, ctx, output_row)
        verdicts.append(StandardVerdict(std_id, r["standard_name"], verdict, rationale))
        if verdict != "PASS":
            failed_or_partial.append((std_id, r["standard_name"], rationale))

    overall_score = round(
        sum(config.VERDICT_SCORES[v.verdict] for v in verdicts) / len(verdicts) * 100, 1
    )

    if overall_score >= config.PASS_THRESHOLD:
        recommendation = "Approved"
    elif overall_score >= config.PARTIAL_THRESHOLD:
        recommendation = "Needs revision"
    else:
        recommendation = "Rejected"

    suggested_revision = (
        "" if recommendation == "Approved"
        else _suggest_revision(failed_or_partial, ctx, output_row)
    )

    return QualityReview(
        output_id=output_id,
        account_id=first["account_id"],
        output_type=first["output_type"],
        draft_text=draft_text,
        intended_customer_action=first["intended_customer_action"],
        standard_verdicts=verdicts,
        overall_score=overall_score,
        recommendation=recommendation,
        suggested_revision=suggested_revision,
        prompt_used=prompt,
    )


def review_all_outputs(exploded_standards: pd.DataFrame, contexts: Dict[str, AccountContext],
                        client: SimulatedLLMClient) -> List[QualityReview]:
    reviews = []
    for output_id, group in exploded_standards.groupby("output_id", sort=False):
        account_id = group.iloc[0]["account_id"]
        ctx = contexts.get(account_id)
        if ctx is None:
            continue  # orphaned account_id; already warned about during load
        reviews.append(review_output(output_id, group, ctx, client))
    return reviews
