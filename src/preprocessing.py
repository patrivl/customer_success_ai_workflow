"""
Deterministic preprocessing layer.

This module runs BEFORE any simulated-LLM stage (briefing_generator.py /
quality_reviewer.py). It is responsible for everything that does not
require judgement or synthesis: validating and normalizing the raw CSVs,
joining everything onto account_id, computing transparent flags/scores,
and selecting which items should move into the (simulated) model-facing
stages. Nothing in this file is AI-generated or calls an external API --
it's plain pandas/stdlib, on purpose, so it's fast, free, and fully
auditable.

Public entry point: run_preprocessing(tables) -> dict of all artifacts,
also written to outputs/preprocessing/.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from src import config
from src.data_loader import DataValidationError, RawTables


# ---------------------------------------------------------------------------
# 1. Validate and normalize
# ---------------------------------------------------------------------------

def _revalidate_required_columns(tables: RawTables) -> None:
    """Defensive re-check: confirm required columns exist in every table.
    (The primary check already happened in data_loader.load_all_tables();
    this guards preprocessing against being called on hand-built tables.)"""
    name_to_df = {
        "accounts.csv": tables.accounts,
        "usage_events.csv": tables.usage_events,
        "support_tickets.csv": tables.support_tickets,
        "call_notes.csv": tables.call_notes,
        "scheduled_checkins.csv": tables.scheduled_checkins,
        "junior_outputs.csv": tables.junior_outputs,
        "quality_standards.csv": tables.quality_standards,
    }
    errors = []
    for filename, required in config.REQUIRED_COLUMNS.items():
        df = name_to_df[filename]
        missing = [c for c in required if c not in df.columns]
        if missing:
            errors.append(f"{filename} is missing required column(s): {missing}")
    if errors:
        raise DataValidationError(
            "Preprocessing column validation failed:\n  - " + "\n  - ".join(errors)
        )


def _norm_lower(series: pd.Series, default: str) -> pd.Series:
    """Lowercase/strip a string field, filling missing values with a
    sensible default so downstream comparisons never see NaN."""
    return series.fillna(default).astype(str).str.strip().str.lower()


def _norm_numeric(series: pd.Series, default: Optional[float] = None) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if default is None:
        default = numeric.median()
        if pd.isna(default):
            default = 0.0
    return numeric.fillna(default)


def validate_and_normalize_data(tables: RawTables) -> RawTables:
    """
    Confirm required columns exist (defensive re-check), then return a NEW
    RawTables whose string fields are normalized (lowercase/stripped with
    safe defaults for missing values), whose dates are parsed into
    `*_parsed` datetime columns, and whose key numeric fields are coerced
    and defaulted. `account_id` is preserved as the join key throughout;
    `quality_standard_ids` is left untouched here (splitting/joining it
    against quality_standards.csv happens in build_outputs_precheck /
    the account-context builder below, mirroring data_loader's approach).

    This returns a *separate* normalized copy -- it does not mutate the
    tables used by the rest of the workflow, so Stage 1/2 report text
    (which preserves the original casing from the source CSVs) is
    unaffected by this internal, preprocessing-only normalization.
    """
    _revalidate_required_columns(tables)

    accounts = tables.accounts.copy()
    usage_events = tables.usage_events.copy()
    support_tickets = tables.support_tickets.copy()
    call_notes = tables.call_notes.copy()
    scheduled_checkins = tables.scheduled_checkins.copy()
    junior_outputs = tables.junior_outputs.copy()
    quality_standards = tables.quality_standards.copy()

    # --- accounts ---
    accounts["segment"] = accounts["segment"].fillna("Unknown").astype(str).str.strip()
    accounts["product_usage_trend"] = _norm_lower(accounts["product_usage_trend"], "flat")
    accounts["expansion_signal"] = _norm_lower(accounts["expansion_signal"], "low")
    accounts["current_health_score"] = _norm_numeric(accounts["current_health_score"])
    accounts["previous_health_score"] = _norm_numeric(accounts["previous_health_score"])
    accounts["support_ticket_count_30d"] = _norm_numeric(accounts["support_ticket_count_30d"], default=0.0)
    accounts["nps_score"] = _norm_numeric(accounts["nps_score"])
    accounts["contract_value"] = _norm_numeric(accounts["contract_value"], default=0.0)
    accounts["notes"] = accounts["notes"].fillna("").astype(str).str.strip()
    accounts["renewal_date_parsed"] = pd.to_datetime(accounts["renewal_date"], errors="coerce")
    accounts["last_contact_date_parsed"] = pd.to_datetime(accounts["last_contact_date"], errors="coerce")

    # --- usage_events ---
    usage_events["usage_trend"] = _norm_lower(usage_events["usage_trend"], "flat")
    usage_events["active_users"] = _norm_numeric(usage_events["active_users"], default=0.0)
    usage_events["key_feature_users"] = _norm_numeric(usage_events["key_feature_users"], default=0.0)
    usage_events["event_date_parsed"] = pd.to_datetime(usage_events["event_date"], errors="coerce")

    # --- support_tickets ---
    support_tickets["severity"] = _norm_lower(support_tickets["severity"], "low")
    support_tickets["customer_sentiment"] = _norm_lower(support_tickets["customer_sentiment"], "neutral")
    support_tickets["current_status"] = _norm_lower(support_tickets["current_status"], "open")
    support_tickets["issue_summary"] = support_tickets["issue_summary"].fillna("").astype(str)
    support_tickets["frontline_notes"] = support_tickets["frontline_notes"].fillna("").astype(str)
    support_tickets["date_received_parsed"] = pd.to_datetime(support_tickets["date_received"], errors="coerce")

    # --- call_notes ---
    call_notes["follow_up_items"] = call_notes["follow_up_items"].fillna("").astype(str).str.strip()
    call_notes["risk_or_blocker"] = call_notes["risk_or_blocker"].fillna("").astype(str).str.strip()
    call_notes["customer_goal"] = call_notes["customer_goal"].fillna("").astype(str).str.strip()
    call_notes["summary"] = call_notes["summary"].fillna("").astype(str).str.strip()
    call_notes["call_date_parsed"] = pd.to_datetime(call_notes["call_date"], errors="coerce")

    # --- scheduled_checkins ---
    scheduled_checkins["priority"] = _norm_lower(scheduled_checkins["priority"], "medium")
    scheduled_checkins["checkin_type"] = scheduled_checkins["checkin_type"].fillna("").astype(str).str.strip()
    scheduled_checkins["topics_to_cover"] = scheduled_checkins["topics_to_cover"].fillna("").astype(str).str.strip()
    scheduled_checkins["scheduled_date_parsed"] = pd.to_datetime(scheduled_checkins["scheduled_date"], errors="coerce")

    # --- junior_outputs ---
    junior_outputs["quality_standard_ids"] = junior_outputs["quality_standard_ids"].fillna("").astype(str)
    junior_outputs["draft_text"] = junior_outputs["draft_text"].fillna("").astype(str)
    junior_outputs["output_type"] = junior_outputs["output_type"].fillna("Unspecified").astype(str)

    # --- quality_standards --- (no normalization needed; just ensure no NaNs)
    quality_standards["standard_name"] = quality_standards["standard_name"].fillna("").astype(str)
    quality_standards["description"] = quality_standards["description"].fillna("").astype(str)

    return RawTables(
        accounts=accounts,
        usage_events=usage_events,
        support_tickets=support_tickets,
        call_notes=call_notes,
        scheduled_checkins=scheduled_checkins,
        junior_outputs=junior_outputs,
        quality_standards=quality_standards,
    )


# ---------------------------------------------------------------------------
# quality_standard_ids split + join (same operation as data_loader's, kept
# self-contained here so preprocessing.py has no hidden dependency on
# Stage 2 having already run).
# ---------------------------------------------------------------------------

def explode_quality_standards(normalized: RawTables) -> pd.DataFrame:
    outputs = normalized.junior_outputs.copy()
    outputs["quality_standard_ids"] = outputs["quality_standard_ids"].str.split(";")
    exploded = outputs.explode("quality_standard_ids")
    exploded["standard_id"] = exploded["quality_standard_ids"].str.strip()
    exploded = exploded.drop(columns=["quality_standard_ids"])
    merged = exploded.merge(
        normalized.quality_standards, on="standard_id", how="left", validate="many_to_one"
    )
    return merged.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Cheap deterministic output pre-check (feeds escalation_score and
# select_failed_or_weak_outputs). NOT the authoritative quality review --
# that's Stage 2's job.
# ---------------------------------------------------------------------------

def _weak_output_precheck(draft_text: str) -> bool:
    text = str(draft_text).lower()
    word_count = len(text.split())
    vague_hit = any(marker in text for marker in config.WEAK_OUTPUT_VAGUE_MARKERS)
    return word_count < config.WEAK_OUTPUT_MIN_WORD_COUNT or vague_hit


def build_outputs_precheck(normalized: RawTables) -> Dict[str, dict]:
    """Returns output_id -> {account_id, output_type, weak_flag, reason,
    standard_ids: [...]} using only cheap deterministic heuristics."""
    precheck: Dict[str, dict] = {}
    exploded = explode_quality_standards(normalized)
    for output_id, group in exploded.groupby("output_id", sort=False):
        first = group.iloc[0]
        weak = _weak_output_precheck(first["draft_text"])
        reason = (
            "draft is short and/or uses vague filler phrasing (preliminary check)"
            if weak else "draft passes preliminary length/specificity check"
        )
        precheck[output_id] = {
            "output_id": output_id,
            "account_id": first["account_id"],
            "output_type": first["output_type"],
            "weak_output_flag": weak,
            "reason": reason,
            "standard_ids": [s for s in group["standard_id"].tolist() if s],
            "standard_count": int(group["standard_id"].notna().sum()),
        }
    return precheck


# ---------------------------------------------------------------------------
# Keyword-theme helpers (deterministic pattern detection, no AI).
# ---------------------------------------------------------------------------

def _match_themes(text: str, theme_keywords: Dict[str, List[str]]) -> List[str]:
    text_lower = str(text).lower()
    matches = []
    for theme, keywords in theme_keywords.items():
        if any(kw in text_lower for kw in keywords):
            matches.append(theme)
    return matches


def _contains_any_keyword(text: str, keywords: List[str]) -> bool:
    text_lower = str(text).lower()
    return any(kw in text_lower for kw in keywords)


# ---------------------------------------------------------------------------
# 2 & 3. Account-level context + deterministic features
# ---------------------------------------------------------------------------

def _days_between(later: Optional[pd.Timestamp], earlier: Optional[pd.Timestamp]) -> Optional[int]:
    if later is None or earlier is None or pd.isna(later) or pd.isna(earlier):
        return None
    return int((later - earlier).days)


def build_account_contexts(normalized: RawTables, outputs_precheck: Dict[str, dict],
                            reference_date: datetime) -> Dict[str, Dict[str, Any]]:
    """
    Build one compact, JSON-serializable context dict per account, joined
    on account_id across every child table, plus deterministic flags and
    scores. account_id is the join key throughout.
    """
    ref_ts = pd.Timestamp(reference_date)
    contexts: Dict[str, Dict[str, Any]] = {}

    accounts = normalized.accounts
    usage_events = normalized.usage_events
    support_tickets = normalized.support_tickets
    call_notes = normalized.call_notes
    scheduled_checkins = normalized.scheduled_checkins
    junior_outputs = normalized.junior_outputs

    # High-value threshold computed once across the portfolio.
    high_value_cutoff = accounts["contract_value"].quantile(config.HIGH_VALUE_PERCENTILE)

    for _, acc in accounts.iterrows():
        aid = acc["account_id"]

        acc_usage = usage_events[usage_events["account_id"] == aid].sort_values("event_date_parsed")
        acc_tickets = support_tickets[support_tickets["account_id"] == aid]
        acc_open_tickets = acc_tickets[acc_tickets["current_status"].isin(["open", "new"])]
        acc_calls = call_notes[call_notes["account_id"] == aid].sort_values("call_date_parsed")
        acc_checkins = scheduled_checkins[scheduled_checkins["account_id"] == aid].sort_values("scheduled_date_parsed")
        acc_outputs = junior_outputs[junior_outputs["account_id"] == aid]

        latest_call = acc_calls.iloc[-1] if len(acc_calls) else None
        next_checkin = acc_checkins.iloc[0] if len(acc_checkins) else None

        # ---- deterministic feature calculations ----
        current_health = float(acc["current_health_score"])
        previous_health = float(acc["previous_health_score"])
        health_score_delta = current_health - previous_health  # spec: current - previous

        renewal_days_remaining = _days_between(acc["renewal_date_parsed"], ref_ts)
        days_since_last_contact = _days_between(ref_ts, acc["last_contact_date_parsed"])

        follow_up_items = latest_call["follow_up_items"] if latest_call is not None else ""
        unresolved_follow_up = follow_up_items if follow_up_items else None

        open_ticket_ids = acc_open_tickets["ticket_id"].tolist()
        high_sev_open = acc_open_tickets[acc_open_tickets["severity"] == "high"]
        negative_sentiment_open = acc_open_tickets[
            acc_open_tickets["customer_sentiment"].isin(["frustrated", "negative", "concerned"])
        ]

        declining_usage = str(acc["product_usage_trend"]) == "declining"
        if not declining_usage and len(acc_usage) >= 2:
            first_u, last_u = acc_usage.iloc[0], acc_usage.iloc[-1]
            declining_usage = float(last_u["active_users"]) < float(first_u["active_users"])

        expansion_signal = str(acc["expansion_signal"])

        # ---- flags (booleans, per spec section 3) ----
        health_decline_flag = health_score_delta < 0
        severe_decline_flag = health_score_delta <= config.SEVERE_HEALTH_DECLINE_THRESHOLD
        renewal_soon_flag = (
            renewal_days_remaining is not None
            and 0 <= renewal_days_remaining <= config.RENEWAL_URGENT_WINDOW_DAYS
        )
        high_ticket_volume_flag = float(acc["support_ticket_count_30d"]) >= config.HIGH_TICKET_VOLUME_THRESHOLD
        negative_sentiment_flag = len(negative_sentiment_open) > 0
        low_nps_flag = float(acc["nps_score"]) <= config.LOW_NPS_THRESHOLD
        declining_usage_flag = declining_usage
        expansion_opportunity_flag = expansion_signal in ("high", "medium")
        unresolved_issue_flag = len(open_ticket_ids) > 0 or unresolved_follow_up is not None
        checkin_due_flag = next_checkin is not None
        quality_review_needed_flag = len(acc_outputs) > 0
        intervention_candidate_flag = declining_usage_flag and (
            low_nps_flag or negative_sentiment_flag or severe_decline_flag
        )

        flags = {
            "health_decline_flag": bool(health_decline_flag),
            "severe_decline_flag": bool(severe_decline_flag),
            "renewal_soon_flag": bool(renewal_soon_flag),
            "high_ticket_volume_flag": bool(high_ticket_volume_flag),
            "negative_sentiment_flag": bool(negative_sentiment_flag),
            "low_nps_flag": bool(low_nps_flag),
            "declining_usage_flag": bool(declining_usage_flag),
            "expansion_opportunity_flag": bool(expansion_opportunity_flag),
            "unresolved_issue_flag": bool(unresolved_issue_flag),
            "checkin_due_flag": bool(checkin_due_flag),
            "quality_review_needed_flag": bool(quality_review_needed_flag),
            "intervention_candidate_flag": bool(intervention_candidate_flag),
            "high_value_account_flag": bool(float(acc["contract_value"]) >= high_value_cutoff),
        }

        # ---- related junior outputs (for this account), with precheck ----
        related_outputs = []
        for _, o in acc_outputs.iterrows():
            pc = outputs_precheck.get(o["output_id"], {})
            related_outputs.append({
                "output_id": o["output_id"],
                "output_type": o["output_type"],
                "weak_output_flag": pc.get("weak_output_flag", False),
                "standard_ids": pc.get("standard_ids", []),
            })
        any_weak_output = any(o["weak_output_flag"] for o in related_outputs)

        context: Dict[str, Any] = {
            "account_id": aid,
            "account_name": acc["account_name"],
            "segment": acc["segment"],
            "csm_owner": acc["csm_owner"],
            "contract_value": float(acc["contract_value"]),
            "current_health_score": current_health,
            "previous_health_score": previous_health,
            "health_score_delta": health_score_delta,
            "product_usage_trend": acc["product_usage_trend"],
            "support_ticket_count_30d": float(acc["support_ticket_count_30d"]),
            "nps_score": float(acc["nps_score"]),
            "expansion_signal": expansion_signal,
            "renewal_date": str(acc["renewal_date"]),
            "renewal_days_remaining": renewal_days_remaining,
            "days_since_last_contact": days_since_last_contact,
            "notes": acc["notes"],
            "recent_usage_events": [
                {
                    "event_date": str(r["event_date"]),
                    "active_users": float(r["active_users"]),
                    "key_feature_users": float(r["key_feature_users"]),
                    "usage_trend": r["usage_trend"],
                }
                for _, r in acc_usage.iterrows()
            ],
            "open_tickets": [
                {
                    "ticket_id": r["ticket_id"],
                    "severity": r["severity"],
                    "customer_sentiment": r["customer_sentiment"],
                    "issue_summary": r["issue_summary"],
                    "current_status": r["current_status"],
                }
                for _, r in acc_open_tickets.iterrows()
            ],
            "all_tickets_count": int(len(acc_tickets)),
            "call_notes": [
                {
                    "call_date": str(r["call_date"]),
                    "customer_goal": r["customer_goal"],
                    "risk_or_blocker": r["risk_or_blocker"],
                    "follow_up_items": r["follow_up_items"],
                }
                for _, r in acc_calls.iterrows()
            ],
            "upcoming_checkins": [
                {
                    "checkin_id": r["checkin_id"],
                    "scheduled_date": str(r["scheduled_date"]),
                    "checkin_type": r["checkin_type"],
                    "priority": r["priority"],
                    "topics_to_cover": r["topics_to_cover"],
                }
                for _, r in acc_checkins.iterrows()
            ],
            "related_junior_outputs": related_outputs,
            "unresolved_follow_up_items": unresolved_follow_up,
            "open_ticket_ids": open_ticket_ids,
            "flags": flags,
            "_internal": {
                # kept out of the public-facing keys above but used by
                # scoring functions below; harmless to serialize to JSON.
                "high_sev_open_count": int(len(high_sev_open)),
                "negative_sentiment_open_count": int(len(negative_sentiment_open)),
                "any_weak_output": bool(any_weak_output),
                "checkin_priority": str(next_checkin["priority"]) if next_checkin is not None else None,
                "customer_goal_text": " ".join(
                    str(r["customer_goal"]) for _, r in acc_calls.iterrows()
                ),
                "risk_blocker_text": " ".join(
                    str(r["risk_or_blocker"]) for _, r in acc_calls.iterrows()
                ),
            },
        }
        contexts[aid] = context

    return contexts


# ---------------------------------------------------------------------------
# 4. Deterministic scoring rules
# ---------------------------------------------------------------------------

def compute_risk_score(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Higher = more at-risk. Returns {'score': float, 'reasons': [str, ...]}."""
    flags = ctx["flags"]
    reasons = []
    score = 0.0

    decline_points = max(0.0, -ctx["health_score_delta"])
    if decline_points > 0:
        score += decline_points * config.HEALTH_DECLINE_WEIGHT
        reasons.append(f"health score declined by {decline_points:.0f} points")

    score += (100 - ctx["current_health_score"]) * config.HEALTH_SCORE_WEIGHT

    if flags["declining_usage_flag"]:
        score += 15
        reasons.append("usage trend is declining")

    if flags["high_ticket_volume_flag"]:
        score += ctx["support_ticket_count_30d"] * config.TICKET_COUNT_WEIGHT
        reasons.append(f"high ticket volume ({ctx['support_ticket_count_30d']:.0f} in 30d)")
    else:
        score += ctx["support_ticket_count_30d"] * (config.TICKET_COUNT_WEIGHT / 2)

    if flags["negative_sentiment_flag"]:
        score += 15
        reasons.append("open ticket(s) carry negative/concerned sentiment")

    if flags["low_nps_flag"]:
        score += (10 - ctx["nps_score"]) * config.NPS_GAP_WEIGHT
        reasons.append(f"low NPS ({ctx['nps_score']:.0f})")

    if flags["renewal_soon_flag"]:
        score *= 1.25
        reasons.append(f"renewal within {config.RENEWAL_URGENT_WINDOW_DAYS} days ({ctx['renewal_days_remaining']}d)")

    if flags["unresolved_issue_flag"]:
        score += 10
        if ctx["unresolved_follow_up_items"]:
            reasons.append(f"unresolved follow-up: {ctx['unresolved_follow_up_items']}")
        elif ctx["open_ticket_ids"]:
            reasons.append(f"open ticket(s): {', '.join(ctx['open_ticket_ids'])}")

    return {"score": round(score, 2), "reasons": reasons}


def compute_opportunity_score(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Higher = better expansion/growth opportunity."""
    flags = ctx["flags"]
    reasons = []
    score = 0.0

    if str(ctx["product_usage_trend"]) == "growing":
        score += 15
        reasons.append("usage trend is growing")

    expansion_bonus = {"high": 25, "medium": 12, "low": 0}.get(ctx["expansion_signal"], 0)
    if expansion_bonus:
        score += expansion_bonus
        reasons.append(f"expansion signal is {ctx['expansion_signal']}")

    if not flags["negative_sentiment_flag"] and ctx["open_tickets"]:
        positive_tickets = [t for t in ctx["open_tickets"] if t["customer_sentiment"] == "positive"]
        if positive_tickets:
            score += 8
            reasons.append("open ticket(s) carry positive sentiment")

    if flags["checkin_due_flag"]:
        score += 5
        reasons.append("upcoming check-in scheduled")

    if flags["high_value_account_flag"]:
        score += 10
        reasons.append("high-value account (top quartile contract value)")

    goal_text = ctx["_internal"]["customer_goal_text"]
    if _contains_any_keyword(goal_text, ["expand", "grow", "advanced", "additional", "premium", "rollout", "phase", "scale"]):
        score += 10
        reasons.append("stated customer goal aligns with growth/expansion language")

    return {"score": round(score, 2), "reasons": reasons}


def compute_escalation_score(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Higher = more urgently needs human/specialist escalation."""
    flags = ctx["flags"]
    reasons = []
    score = 0.0

    if ctx["_internal"]["high_sev_open_count"] > 0:
        score += 25 * ctx["_internal"]["high_sev_open_count"]
        reasons.append(f"{ctx['_internal']['high_sev_open_count']} high-severity open ticket(s)")

    if flags["negative_sentiment_flag"]:
        score += 15
        reasons.append("negative/frustrated customer sentiment on an open ticket")

    combined_text = " ".join([
        ctx["notes"], ctx["_internal"]["risk_blocker_text"],
        " ".join(t["issue_summary"] for t in ctx["open_tickets"]),
    ])

    if _contains_any_keyword(combined_text, config.EXEC_RENEWAL_KEYWORDS):
        score += 10
        reasons.append("executive/renewal risk language present")

    if _contains_any_keyword(combined_text, config.TECHNICAL_BLOCKER_KEYWORDS):
        score += 10
        reasons.append("technical blocker language present")

    if _contains_any_keyword(combined_text, config.MANAGER_ESCALATION_KEYWORDS):
        score += 10
        reasons.append("manager/product escalation language present")

    if ctx["_internal"]["any_weak_output"]:
        score += 10
        reasons.append("a related junior output failed the preliminary quality pre-check")

    return {"score": round(score, 2), "reasons": reasons}


def compute_priority_score(ctx: Dict[str, Any], risk: Dict[str, Any],
                            opportunity: Dict[str, Any]) -> Dict[str, Any]:
    """Combines risk + opportunity + renewal proximity + account value +
    ticket severity + check-in priority + unresolved-item ageing into a
    single 'how much attention does this need right now' score."""
    reasons = []
    score = risk["score"] * 0.6 + opportunity["score"] * 0.4

    if ctx["flags"]["renewal_soon_flag"] and ctx["renewal_days_remaining"] is not None:
        proximity_bonus = max(0, config.RENEWAL_URGENT_WINDOW_DAYS - ctx["renewal_days_remaining"])
        score += proximity_bonus
        reasons.append(f"renewal proximity bonus (+{proximity_bonus:.0f}, {ctx['renewal_days_remaining']}d out)")

    if ctx["flags"]["high_value_account_flag"]:
        score += 10
        reasons.append("high-value account weighting")

    severity_bonus = sum(config.SEVERITY_WEIGHTS.get(t["severity"], 0) for t in ctx["open_tickets"])
    if severity_bonus:
        score += severity_bonus
        reasons.append(f"open-ticket severity weighting (+{severity_bonus})")

    checkin_priority = ctx["_internal"]["checkin_priority"]
    if checkin_priority == "high":
        score += 8
        reasons.append("high-priority check-in scheduled")
    elif checkin_priority == "medium":
        score += 3

    if ctx["days_since_last_contact"] is not None and ctx["days_since_last_contact"] > 14:
        ageing_bonus = min(15, (ctx["days_since_last_contact"] - 14) * 0.5)
        score += ageing_bonus
        reasons.append(f"unresolved-item ageing ({ctx['days_since_last_contact']}d since last contact)")

    return {"score": round(score, 2), "reasons": reasons}


def attach_scores(contexts: Dict[str, Dict[str, Any]]) -> None:
    """Mutates each context dict in place, adding a 'scores' key."""
    for ctx in contexts.values():
        risk = compute_risk_score(ctx)
        opportunity = compute_opportunity_score(ctx)
        escalation = compute_escalation_score(ctx)
        priority = compute_priority_score(ctx, risk, opportunity)
        ctx["scores"] = {
            "risk_score": risk["score"],
            "opportunity_score": opportunity["score"],
            "priority_score": priority["score"],
            "escalation_score": escalation["score"],
        }
        ctx["score_reasons"] = {
            "risk": risk["reasons"],
            "opportunity": opportunity["reasons"],
            "priority": priority["reasons"],
            "escalation": escalation["reasons"],
        }


# ---------------------------------------------------------------------------
# 5. Logical workflow population selectors.
# Each returns a list of small dicts with an "id" (or ids) and a short
# deterministic "reason" string, per the traceability requirement.
# ---------------------------------------------------------------------------

def select_daily_account_review_accounts(contexts: Dict[str, Dict[str, Any]]) -> List[dict]:
    """All accounts -- the baseline daily portfolio review population."""
    return [
        {"account_id": aid, "reason": "part of daily portfolio review cadence"}
        for aid in contexts
    ]


def select_second_pass_validation_accounts(contexts: Dict[str, Dict[str, Any]],
                                            max_n: int = config.SECOND_PASS_MAX_ACCOUNTS) -> List[dict]:
    """Accounts whose risk or opportunity signal warrants a second look,
    capped at max_n (200 by default; this synthetic portfolio has far
    fewer accounts, so in practice everything that clears the bar is kept)."""
    candidates = [
        c for c in contexts.values()
        if c["scores"]["risk_score"] >= config.MEDIUM_RISK_THRESHOLD
        or c["scores"]["opportunity_score"] >= config.OPPORTUNITY_ALERT_THRESHOLD
    ]
    candidates.sort(key=lambda c: max(c["scores"]["risk_score"], c["scores"]["opportunity_score"]), reverse=True)
    candidates = candidates[:max_n]
    return [
        {
            "account_id": c["account_id"],
            "reason": f"risk_score={c['scores']['risk_score']} / opportunity_score={c['scores']['opportunity_score']} clears second-pass threshold",
        }
        for c in candidates
    ]


def select_flagged_account_summary_accounts(contexts: Dict[str, Dict[str, Any]],
                                             min_n: int = config.FLAGGED_SUMMARY_MIN,
                                             max_n: int = config.FLAGGED_SUMMARY_MAX) -> List[dict]:
    """Top 25-40 highest-priority accounts. Gracefully scales down to the
    full portfolio when fewer than min_n accounts exist (as in this
    synthetic dataset)."""
    ranked = sorted(contexts.values(), key=lambda c: c["scores"]["priority_score"], reverse=True)
    n = min(max_n, len(ranked))
    selected = ranked[:n]
    return [
        {
            "account_id": c["account_id"],
            "reason": f"ranked #{i + 1} of {len(ranked)} by priority_score ({c['scores']['priority_score']})",
        }
        for i, c in enumerate(selected)
    ]


def select_csm_alert_accounts(contexts: Dict[str, Dict[str, Any]]) -> List[dict]:
    """Accounts crossing an explicit routing/escalation threshold."""
    alerts = []
    for c in contexts.values():
        reasons = []
        f = c["flags"]
        if f["severe_decline_flag"]:
            reasons.append(f"health score declined by {abs(c['health_score_delta']):.0f} points")
        if f["renewal_soon_flag"] and f["negative_sentiment_flag"]:
            reasons.append(f"renewal within {config.RENEWAL_URGENT_WINDOW_DAYS} days and negative sentiment")
        if c["scores"]["escalation_score"] >= config.ESCALATION_ALERT_THRESHOLD:
            reasons.append(f"escalation_score={c['scores']['escalation_score']} crosses alert threshold ({config.ESCALATION_ALERT_THRESHOLD})")
        if reasons:
            alerts.append({"account_id": c["account_id"], "reason": "; ".join(reasons)})
    return alerts


def select_unresolved_items(contexts: Dict[str, Dict[str, Any]]) -> List[dict]:
    """Accounts/tickets/follow-ups that remain open or stale."""
    items = []
    for c in contexts.values():
        if not c["flags"]["unresolved_issue_flag"]:
            continue
        reason_parts = []
        if c["unresolved_follow_up_items"]:
            reason_parts.append(f"open follow-up commitment: {c['unresolved_follow_up_items']}")
        if c["open_ticket_ids"]:
            reason_parts.append(f"open ticket(s): {', '.join(c['open_ticket_ids'])}")
        items.append({
            "account_id": c["account_id"],
            "reason": "; ".join(reason_parts) or "unresolved issue present",
        })
    return items


def select_inbound_issues(normalized: RawTables) -> List[dict]:
    """All support tickets -- the full inbound issue queue."""
    return [
        {
            "ticket_id": r["ticket_id"],
            "account_id": r["account_id"],
            "reason": f"{r['severity']} severity inbound ticket, sentiment={r['customer_sentiment']}",
        }
        for _, r in normalized.support_tickets.iterrows()
    ]


def select_issue_pattern_review_items(normalized: RawTables) -> List[dict]:
    """Grouped ticket themes that may suggest broader, cross-account
    deterioration (deterministic keyword clustering, no AI)."""
    theme_to_tickets: Dict[str, List[str]] = defaultdict(list)
    for _, r in normalized.support_tickets.iterrows():
        combined = f"{r['issue_summary']} {r['frontline_notes']}"
        for theme in _match_themes(combined, config.ISSUE_THEME_KEYWORDS):
            theme_to_tickets[theme].append(r["ticket_id"])

    items = []
    for theme, ticket_ids in theme_to_tickets.items():
        if len(ticket_ids) >= 2:
            items.append({
                "theme": theme,
                "ticket_ids": ticket_ids,
                "reason": f"{len(ticket_ids)} ticket(s) share the '{theme}' theme, indicating a possible broader pattern",
            })
    return items


def select_scheduled_checkins(normalized: RawTables) -> List[dict]:
    """All scheduled check-ins."""
    return [
        {
            "checkin_id": r["checkin_id"],
            "account_id": r["account_id"],
            "reason": f"{r['priority']} priority '{r['checkin_type']}' check-in scheduled for {r['scheduled_date']}",
        }
        for _, r in normalized.scheduled_checkins.iterrows()
    ]


def select_quality_review_outputs(outputs_precheck: Dict[str, dict]) -> List[dict]:
    """All junior outputs -- the full quality-review queue."""
    return [
        {
            "output_id": oid,
            "account_id": pc["account_id"],
            "reason": f"assigned {pc['standard_count']} quality standard(s) for review",
        }
        for oid, pc in outputs_precheck.items()
    ]


def select_failed_or_weak_outputs(outputs_precheck: Dict[str, dict]) -> List[dict]:
    """Outputs that fail the cheap preliminary quality pre-check (short
    and/or vague drafts) -- candidates for prioritized Stage 2 review."""
    return [
        {"output_id": oid, "account_id": pc["account_id"], "reason": pc["reason"]}
        for oid, pc in outputs_precheck.items()
        if pc["weak_output_flag"]
    ]


def select_intervention_candidates(contexts: Dict[str, Dict[str, Any]]) -> List[dict]:
    """Declining accounts/segments that need a deliberate save-play, not
    just a routine check-in."""
    items = []
    for c in contexts.values():
        if not c["flags"]["intervention_candidate_flag"]:
            continue
        reasons = []
        if c["flags"]["declining_usage_flag"]:
            reasons.append("usage is declining")
        if c["flags"]["low_nps_flag"]:
            reasons.append(f"NPS is low ({c['nps_score']:.0f})")
        if c["flags"]["negative_sentiment_flag"]:
            reasons.append("negative/concerned sentiment on an open ticket")
        if c["flags"]["severe_decline_flag"]:
            reasons.append(f"severe health decline ({c['health_score_delta']:.0f} points)")
        items.append({"account_id": c["account_id"], "reason": "; ".join(reasons)})
    return items


def select_complex_escalation_candidates(contexts: Dict[str, Dict[str, Any]],
                                          max_per_day: int = config.MAX_ESCALATIONS_PER_DAY) -> List[dict]:
    """Highest escalation_score cases, capped at max_per_day to simulate a
    deliberate daily token/attention budget for the most expensive
    (highest-judgement) simulated-LLM escalation-reasoning calls."""
    ranked = sorted(contexts.values(), key=lambda c: c["scores"]["escalation_score"], reverse=True)
    ranked = [c for c in ranked if c["scores"]["escalation_score"] > 0][:max_per_day]
    return [
        {
            "account_id": c["account_id"],
            "reason": f"escalation_score={c['scores']['escalation_score']} (top {max_per_day} under daily cap)",
        }
        for c in ranked
    ]


def build_selected_workflow_items(contexts: Dict[str, Dict[str, Any]], normalized: RawTables,
                                   outputs_precheck: Dict[str, dict]) -> Dict[str, List[dict]]:
    return {
        "daily_account_review": select_daily_account_review_accounts(contexts),
        "second_pass_validation": select_second_pass_validation_accounts(contexts),
        "flagged_account_summary": select_flagged_account_summary_accounts(contexts),
        "csm_alerts": select_csm_alert_accounts(contexts),
        "unresolved_items": select_unresolved_items(contexts),
        "inbound_issues": select_inbound_issues(normalized),
        "issue_pattern_review": select_issue_pattern_review_items(normalized),
        "scheduled_checkins": select_scheduled_checkins(normalized),
        "quality_review_outputs": select_quality_review_outputs(outputs_precheck),
        "failed_or_weak_outputs": select_failed_or_weak_outputs(outputs_precheck),
        "intervention_candidates": select_intervention_candidates(contexts),
        "complex_escalation_candidates": select_complex_escalation_candidates(contexts),
    }


# ---------------------------------------------------------------------------
# 6. Segment and pattern detection (deterministic grouping + keyword themes)
# ---------------------------------------------------------------------------

def build_portfolio_patterns(contexts: Dict[str, Dict[str, Any]], normalized: RawTables) -> Dict[str, Any]:
    accounts = normalized.accounts

    by_segment = (
        accounts.groupby("segment")
        .agg(account_count=("account_id", "count"), avg_current_health_score=("current_health_score", "mean"))
        .round(1)
        .reset_index()
        .to_dict(orient="records")
    )

    declining_by_segment = Counter(
        c["segment"] for c in contexts.values() if c["flags"]["declining_usage_flag"]
    )
    intervention_by_segment = Counter(
        c["segment"] for c in contexts.values() if c["flags"]["intervention_candidate_flag"]
    )
    expansion_by_segment = Counter(
        c["segment"] for c in contexts.values() if c["flags"]["expansion_opportunity_flag"]
    )

    # Common issue themes from support tickets.
    issue_theme_counts: Counter = Counter()
    for _, r in normalized.support_tickets.iterrows():
        combined = f"{r['issue_summary']} {r['frontline_notes']}"
        for theme in _match_themes(combined, config.ISSUE_THEME_KEYWORDS):
            issue_theme_counts[theme] += 1

    # Common blockers from call notes.
    blocker_theme_counts: Counter = Counter()
    for _, r in normalized.call_notes.iterrows():
        for theme in _match_themes(r["risk_or_blocker"], config.ISSUE_THEME_KEYWORDS):
            blocker_theme_counts[theme] += 1

    return {
        "accounts_by_segment": by_segment,
        "declining_accounts_by_segment": dict(declining_by_segment),
        "intervention_candidate_segments": dict(intervention_by_segment),
        "expansion_opportunity_segments": dict(expansion_by_segment),
        "common_issue_themes": dict(issue_theme_counts.most_common()),
        "common_blocker_themes": dict(blocker_theme_counts.most_common()),
    }


# ---------------------------------------------------------------------------
# 7. Representative run selection
# ---------------------------------------------------------------------------

def _representative_entry(ctx: Dict[str, Any], case_type: str, reason: str) -> dict:
    return {
        "account_id": ctx["account_id"],
        "account_name": ctx["account_name"],
        "case_type": case_type,
        "reason": reason,
        "risk_score": ctx["scores"]["risk_score"],
        "opportunity_score": ctx["scores"]["opportunity_score"],
        "priority_score": ctx["scores"]["priority_score"],
        "escalation_score": ctx["scores"]["escalation_score"],
    }


def select_representative_runs(contexts: Dict[str, Dict[str, Any]],
                                selected_items: Optional[Dict[str, List[dict]]] = None,
                                min_n: int = config.MIN_REPRESENTATIVE_RUNS,
                                max_n: int = config.MAX_REPRESENTATIVE_RUNS) -> List[dict]:
    """
    Returns between min_n and max_n representative account runs.

    Preferring config.PREFERRED_REPRESENTATIVE_ACCOUNT_IDS (each covering a
    distinct case type) when present. If `selected_items` (the
    preprocessing selectors' own populations) is supplied, additional
    case types beyond the preferred 5 (config.ADDITIONAL_REPRESENTATIVE_CASE_TYPES
    -- quality review, intervention planning, complex escalation) are
    filled in deterministically from those same selectors, never from a
    hardcoded account id, up to max_n total. Any substitution (a preferred
    id missing, or an additional case type filled from a selector) carries
    an explicit `reason` string for traceability.
    """
    result = []
    chosen_ids = set()
    covered_case_types = set()

    for aid in config.PREFERRED_REPRESENTATIVE_ACCOUNT_IDS:
        ctx = contexts.get(aid)
        if ctx is None:
            continue
        case_type = config.REPRESENTATIVE_CASE_TYPES[aid]
        result.append(_representative_entry(
            ctx, case_type, f"preferred representative account for case type '{case_type}'",
        ))
        chosen_ids.add(aid)
        covered_case_types.add(case_type)

    if selected_items is not None:
        for case_type, selector_key in config.ADDITIONAL_REPRESENTATIVE_CASE_TYPES.items():
            if len(result) >= max_n or case_type in covered_case_types:
                continue
            for entry in selected_items.get(selector_key, []):
                aid = entry.get("account_id")
                if not aid or aid in chosen_ids or aid not in contexts:
                    continue
                ctx = contexts[aid]
                result.append(_representative_entry(
                    ctx, case_type,
                    f"deterministic selector '{selector_key}' match for case type "
                    f"'{case_type}' (no preferred id available/uncovered): {entry.get('reason', '')}",
                ))
                chosen_ids.add(aid)
                covered_case_types.add(case_type)
                break

    if len(result) < min_n:
        remaining = sorted(
            (c for aid, c in contexts.items() if aid not in chosen_ids),
            key=lambda c: c["scores"]["priority_score"],
            reverse=True,
        )
        for ctx in remaining:
            if len(result) >= min_n:
                break
            result.append(_representative_entry(
                ctx, "closest_match_by_priority_score",
                "a preferred representative account was unavailable; selected as the closest match by priority_score",
            ))
            chosen_ids.add(ctx["account_id"])

    return result[:max_n]


# ---------------------------------------------------------------------------
# 8. Save preprocessing outputs
# ---------------------------------------------------------------------------

def _account_scores_dataframe(contexts: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for c in contexts.values():
        rows.append({
            "account_id": c["account_id"],
            "account_name": c["account_name"],
            "segment": c["segment"],
            "csm_owner": c["csm_owner"],
            "risk_score": c["scores"]["risk_score"],
            "opportunity_score": c["scores"]["opportunity_score"],
            "priority_score": c["scores"]["priority_score"],
            "escalation_score": c["scores"]["escalation_score"],
            "health_score_delta": c["health_score_delta"],
            "renewal_days_remaining": c["renewal_days_remaining"],
            "health_decline_flag": c["flags"]["health_decline_flag"],
            "severe_decline_flag": c["flags"]["severe_decline_flag"],
            "renewal_soon_flag": c["flags"]["renewal_soon_flag"],
            "high_ticket_volume_flag": c["flags"]["high_ticket_volume_flag"],
            "negative_sentiment_flag": c["flags"]["negative_sentiment_flag"],
            "low_nps_flag": c["flags"]["low_nps_flag"],
            "declining_usage_flag": c["flags"]["declining_usage_flag"],
            "expansion_opportunity_flag": c["flags"]["expansion_opportunity_flag"],
            "unresolved_issue_flag": c["flags"]["unresolved_issue_flag"],
            "checkin_due_flag": c["flags"]["checkin_due_flag"],
            "quality_review_needed_flag": c["flags"]["quality_review_needed_flag"],
            "intervention_candidate_flag": c["flags"]["intervention_candidate_flag"],
            "high_value_account_flag": c["flags"]["high_value_account_flag"],
        })
    df = pd.DataFrame(rows).sort_values("priority_score", ascending=False).reset_index(drop=True)
    return df


def save_outputs(contexts: Dict[str, Dict[str, Any]], account_scores_df: pd.DataFrame,
                  portfolio_patterns: Dict[str, Any], selected_items: Dict[str, List[dict]],
                  representative_runs: List[dict]) -> None:
    out_dir = config.PREPROCESSING_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # account_contexts.json needs its "_internal" scratch keys stripped;
    # everything else is already JSON-safe (plain str/float/bool/list/dict).
    public_contexts = {}
    for aid, ctx in contexts.items():
        clean = {k: v for k, v in ctx.items() if k != "_internal"}
        public_contexts[aid] = clean

    (out_dir / "account_contexts.json").write_text(json.dumps(public_contexts, indent=2, default=str))
    account_scores_df.to_csv(out_dir / "account_scores.csv", index=False)
    (out_dir / "portfolio_patterns.json").write_text(json.dumps(portfolio_patterns, indent=2, default=str))
    (out_dir / "selected_workflow_items.json").write_text(json.dumps(selected_items, indent=2, default=str))
    (out_dir / "representative_accounts.json").write_text(json.dumps(representative_runs, indent=2, default=str))


# ---------------------------------------------------------------------------
# Public orchestrator
# ---------------------------------------------------------------------------

def run_preprocessing(tables: RawTables, reference_date: Optional[datetime] = None) -> Dict[str, Any]:
    """
    Runs the full deterministic preprocessing layer:
      normalize -> account contexts + flags -> scores -> selectors ->
      portfolio patterns -> representative runs -> write outputs.

    Returns a dict with every artifact in memory (useful if a later stage
    wants to consume it directly instead of re-reading the written files).
    """
    reference_date = reference_date or config.REFERENCE_DATE

    normalized = validate_and_normalize_data(tables)
    outputs_precheck = build_outputs_precheck(normalized)
    contexts = build_account_contexts(normalized, outputs_precheck, reference_date)
    attach_scores(contexts)

    account_scores_df = _account_scores_dataframe(contexts)
    portfolio_patterns = build_portfolio_patterns(contexts, normalized)
    selected_items = build_selected_workflow_items(contexts, normalized, outputs_precheck)
    representative_runs = select_representative_runs(contexts, selected_items)

    save_outputs(contexts, account_scores_df, portfolio_patterns, selected_items, representative_runs)

    return {
        "normalized_tables": normalized,
        "outputs_precheck": outputs_precheck,
        "account_contexts": contexts,
        "account_scores_df": account_scores_df,
        "portfolio_patterns": portfolio_patterns,
        "selected_workflow_items": selected_items,
        "representative_runs": representative_runs,
    }






