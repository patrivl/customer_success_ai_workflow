"""
Orchestration + aggregation for the token-math-plan-driven simulated model
layer.

`run_token_math_layer(preproc)` walks every stage_id in
config/token_math_plan.csv (via src/model_simulator.STAGE_RUNTIME_MAP),
pulls its item population from the deterministic preprocessing layer's
`selected_workflow_items` selectors, builds a grounded per-item context,
and calls `src.model_simulator.simulate_model_call()` for each item. No
external AI API is called anywhere in this module.

The resulting call traces are aggregated and written to:
  - outputs/stage_token_counts.csv          (one row per simulated call)
  - outputs/cost_summary.csv                (one row per simulated call)
  - outputs/token_math_measurement_summary.csv
        (one row per stage_id -- copy-paste-ready for the spreadsheet's
        measurement columns: num_calls, avg measured tokens/cost, planned
        tokens/cost, variance_pct, review_flag)
"""

from __future__ import annotations

import csv
from typing import Any, Dict, List

import pandas as pd

from src import config, token_costs
from src.data_loader import RawTables
from src.model_simulator import STAGE_RUNTIME_MAP, _score, simulate_model_call
from src.preprocessing import explode_quality_standards
from src.token_math_config import get_stage, load_token_math_plan

ITEM_ID_KEYS = (
    "account_id", "ticket_id", "checkin_id", "output_id",
    "account_or_segment_id", "item_id",
)


# ---------------------------------------------------------------------------
# Per-item context builders. Account-kind items pass the preprocessing
# layer's own account context straight through; ticket/checkin/output/
# issue-theme items get a purpose-built context merging the raw record with
# the owning account's scores/flags for grounding.
# ---------------------------------------------------------------------------

def _account_common_fields(account_ctx: dict) -> dict:
    scores = account_ctx.get("scores", {})
    return {
        "risk_score": scores.get("risk_score"),
        "opportunity_score": scores.get("opportunity_score"),
        "escalation_score": scores.get("escalation_score"),
        "priority_score": scores.get("priority_score"),
    }


def _build_account_item_context(account_ctx: dict, reason: str) -> dict:
    return {
        **account_ctx,
        "item_id": account_ctx.get("account_id"),
        "item_type": "account",
        "selector_reason": reason,
    }


def _build_ticket_context(ticket_row: dict, account_ctx: dict, reason: str) -> dict:
    return {
        **_account_common_fields(account_ctx),
        "item_id": ticket_row["ticket_id"],
        "item_type": "ticket",
        "ticket_id": ticket_row["ticket_id"],
        "account_id": ticket_row["account_id"],
        "account_name": account_ctx.get("account_name"),
        "csm_owner": account_ctx.get("csm_owner"),
        "severity": ticket_row["severity"],
        "customer_sentiment": ticket_row["customer_sentiment"],
        "issue_summary": ticket_row["issue_summary"],
        "frontline_notes": ticket_row["frontline_notes"],
        "current_status": ticket_row["current_status"],
        "date_received": str(ticket_row["date_received"]),
        "renewal_days_remaining": account_ctx.get("renewal_days_remaining"),
        "selector_reason": reason,
    }


def _build_checkin_context(checkin_row: dict, account_ctx: dict, reason: str) -> dict:
    return {
        **_account_common_fields(account_ctx),
        "item_id": checkin_row["checkin_id"],
        "item_type": "checkin",
        "checkin_id": checkin_row["checkin_id"],
        "account_id": checkin_row["account_id"],
        "account_name": account_ctx.get("account_name"),
        "scheduled_date": str(checkin_row["scheduled_date"]),
        "checkin_type": checkin_row["checkin_type"],
        "priority": checkin_row["priority"],
        "topics_to_cover": checkin_row["topics_to_cover"],
        "open_tickets": account_ctx.get("open_tickets", []),
        "health_score_delta": account_ctx.get("health_score_delta"),
        "product_usage_trend": account_ctx.get("product_usage_trend"),
        "expansion_signal": account_ctx.get("expansion_signal"),
        "renewal_days_remaining": account_ctx.get("renewal_days_remaining"),
        "notes": account_ctx.get("notes"),
        "selector_reason": reason,
    }


def _build_output_context(output_id: str, exploded_group: pd.DataFrame, account_ctx: dict, reason: str) -> dict:
    first = exploded_group.iloc[0]
    standards = [
        {"standard_id": r["standard_id"], "standard_name": r["standard_name"], "description": r["description"]}
        for _, r in exploded_group.iterrows()
    ]
    return {
        **_account_common_fields(account_ctx),
        "item_id": output_id,
        "item_type": "output",
        "output_id": output_id,
        "account_id": first["account_id"],
        "output_type": first["output_type"],
        "draft_text": first["draft_text"],
        "intended_customer_action": first["intended_customer_action"],
        "standards": standards,
        "selector_reason": reason,
    }


def _build_issue_theme_context(theme_item: dict) -> dict:
    return {
        "item_id": theme_item["theme"],
        "item_type": "issue_theme",
        "theme": theme_item["theme"],
        "ticket_ids": theme_item.get("ticket_ids", []),
        "risk_score": 0,
        "selector_reason": theme_item.get("reason", ""),
    }


def _build_stage_items(stage_id: str, meta: dict, normalized: RawTables,
                        account_contexts: Dict[str, dict], selected_items: Dict[str, list],
                        exploded_standards: pd.DataFrame,
                        tickets_by_id: Dict[str, dict], checkins_by_id: Dict[str, dict]) -> List[tuple]:
    population = selected_items.get(meta["population"], [])
    item_kind = meta["item_kind"]
    items: List[tuple] = []

    if item_kind == "account":
        for entry in population:
            aid = entry["account_id"]
            account_ctx = account_contexts.get(aid)
            if account_ctx is None:
                continue
            items.append((aid, _build_account_item_context(account_ctx, entry.get("reason", ""))))

    elif item_kind == "ticket":
        for entry in population:
            ticket_row = tickets_by_id.get(entry["ticket_id"])
            if ticket_row is None:
                continue
            account_ctx = account_contexts.get(entry["account_id"], {})
            items.append((entry["ticket_id"], _build_ticket_context(ticket_row, account_ctx, entry.get("reason", ""))))

    elif item_kind == "checkin":
        for entry in population:
            checkin_row = checkins_by_id.get(entry["checkin_id"])
            if checkin_row is None:
                continue
            account_ctx = account_contexts.get(entry["account_id"], {})
            items.append((entry["checkin_id"], _build_checkin_context(checkin_row, account_ctx, entry.get("reason", ""))))

    elif item_kind == "output":
        for entry in population:
            group = exploded_standards[exploded_standards["output_id"] == entry["output_id"]]
            if group.empty:
                continue
            account_id = entry.get("account_id") or group.iloc[0]["account_id"]
            account_ctx = account_contexts.get(account_id, {})
            items.append((entry["output_id"], _build_output_context(entry["output_id"], group, account_ctx, entry.get("reason", ""))))

    elif item_kind == "issue_theme":
        for entry in population:
            items.append((entry["theme"], _build_issue_theme_context(entry)))

    else:
        raise ValueError(f"Unknown item_kind '{item_kind}' for stage {stage_id}")

    return items


def run_all_stages(preproc: Dict[str, Any]) -> List[dict]:
    """Runs every stage_id in STAGE_RUNTIME_MAP against its deterministic-
    preprocessing population and returns the full list of call traces."""
    normalized: RawTables = preproc["normalized_tables"]
    account_contexts: Dict[str, dict] = preproc["account_contexts"]
    selected_items: Dict[str, list] = preproc["selected_workflow_items"]
    exploded_standards = explode_quality_standards(normalized)

    tickets_by_id = normalized.support_tickets.set_index("ticket_id", drop=False).to_dict(orient="index")
    checkins_by_id = normalized.scheduled_checkins.set_index("checkin_id", drop=False).to_dict(orient="index")

    traces: List[dict] = []
    run_counter = 0

    for stage_id, meta in STAGE_RUNTIME_MAP.items():
        items = _build_stage_items(
            stage_id, meta, normalized, account_contexts, selected_items,
            exploded_standards, tickets_by_id, checkins_by_id,
        )

        template = meta["template"]
        if template == "prioritization_prompt":
            # Rank this stage's own batch by priority_score, highest first.
            items.sort(key=lambda pair: _score(pair[1], "priority_score", 0.0), reverse=True)
            for rank, (_, ctx) in enumerate(items, start=1):
                ctx["priority_rank"] = rank

        for item_id, ctx in items:
            run_counter += 1
            run_id = f"{stage_id}-{item_id}-{run_counter:05d}"
            traces.append(simulate_model_call(stage_id, ctx, template, run_id))

    return traces


def _extract_item_id(trace: dict) -> str:
    result = trace.get("result", {})
    for key in ITEM_ID_KEYS:
        if result.get(key):
            return str(result[key])
    # run_id is "{stage_id}-{item_id}-{counter}" -- fall back to that middle segment.
    parts = trace["run_id"].split("-")
    return "-".join(parts[1:-1]) if len(parts) > 2 else trace["run_id"]


# ---------------------------------------------------------------------------
# Aggregation + CSV writers
# ---------------------------------------------------------------------------

def write_stage_token_counts(traces: List[dict], out_dir=config.OUTPUT_DIR):
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "stage_token_counts.csv"
    fieldnames = [
        "run_id", "stage_id", "workflow_component", "operating_area", "item_id", "model",
        "planned_input_tokens", "planned_output_tokens",
        "measured_input_tokens", "measured_output_tokens",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for t in traces:
            writer.writerow({
                "run_id": t["run_id"],
                "stage_id": t["stage_id"],
                "workflow_component": t["workflow_component"],
                "operating_area": t["operating_area"],
                "item_id": _extract_item_id(t),
                "model": t["model"],
                "planned_input_tokens": t["planned_input_tokens"],
                "planned_output_tokens": t["planned_output_tokens"],
                "measured_input_tokens": t["measured_input_tokens"],
                "measured_output_tokens": t["measured_output_tokens"],
            })
    return path


def write_cost_summary(traces: List[dict], out_dir=config.OUTPUT_DIR):
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "cost_summary.csv"
    fieldnames = [
        "run_id", "stage_id", "planned_cost", "measured_cost", "adjusted_measured_cost",
        "retry_rate", "qa_eval_multiplier", "variance_pct", "review_flag",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for t in traces:
            writer.writerow({
                "run_id": t["run_id"],
                "stage_id": t["stage_id"],
                "planned_cost": round(t["planned_cost"], 6),
                "measured_cost": round(t["measured_cost"], 6),
                "adjusted_measured_cost": round(t["adjusted_measured_cost"], 6),
                "retry_rate": t["retry_rate"],
                "qa_eval_multiplier": t["qa_eval_multiplier"],
                "variance_pct": t["variance_pct"],
                "review_flag": t["review_flag"],
            })
    return path


def write_token_math_measurement_summary(traces: List[dict], out_dir=config.OUTPUT_DIR):
    """One row per stage_id, formatted for direct copy into the token math
    spreadsheet's measurement columns."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "token_math_measurement_summary.csv"
    fieldnames = [
        "stage_id", "workflow_component", "operating_area", "trigger_schedule", "model",
        "num_calls", "avg_measured_input_tokens", "avg_measured_output_tokens",
        "avg_measured_cost_per_run", "planned_input_tokens_per_run",
        "planned_output_tokens_per_run", "planned_cost_per_run",
        "variance_pct", "review_flag", "source_output_file",
    ]

    by_stage: Dict[str, List[dict]] = {}
    for t in traces:
        by_stage.setdefault(t["stage_id"], []).append(t)

    plan = load_token_math_plan()
    rows = []
    for stage_id, stage in plan.items():
        stage_traces = by_stage.get(stage_id, [])
        planned_cost_per_run = token_costs.calculate_cost(
            stage.planned_input_tokens_per_run, stage.planned_output_tokens_per_run,
            stage.input_price_per_1m, stage.output_price_per_1m,
        )
        if stage_traces:
            n = len(stage_traces)
            avg_input = sum(t["measured_input_tokens"] for t in stage_traces) / n
            avg_output = sum(t["measured_output_tokens"] for t in stage_traces) / n
            avg_cost = sum(t["measured_cost"] for t in stage_traces) / n
            variance_pct = token_costs.calculate_variance(planned_cost_per_run, avg_cost)
            review_flag = token_costs.assign_review_flag(variance_pct)
        else:
            avg_input = avg_output = avg_cost = None
            variance_pct = None
            review_flag = token_costs.assign_review_flag(None)

        rows.append({
            "stage_id": stage.stage_id,
            "workflow_component": stage.workflow_component,
            "operating_area": stage.operating_area,
            "trigger_schedule": stage.trigger_schedule,
            "model": stage.model,
            "num_calls": len(stage_traces),
            "avg_measured_input_tokens": round(avg_input, 1) if avg_input is not None else "",
            "avg_measured_output_tokens": round(avg_output, 1) if avg_output is not None else "",
            "avg_measured_cost_per_run": round(avg_cost, 6) if avg_cost is not None else "",
            "planned_input_tokens_per_run": stage.planned_input_tokens_per_run,
            "planned_output_tokens_per_run": stage.planned_output_tokens_per_run,
            "planned_cost_per_run": round(planned_cost_per_run, 6),
            "variance_pct": round(variance_pct, 2) if variance_pct is not None else "",
            "review_flag": review_flag,
            "source_output_file": "outputs/stage_token_counts.csv",
        })

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def run_token_math_layer(preproc: Dict[str, Any]) -> Dict[str, Any]:
    """Public entry point: runs every stage, writes the three aggregation
    CSVs, and returns a small summary dict for console reporting."""
    traces = run_all_stages(preproc)

    stage_counts_path = write_stage_token_counts(traces)
    cost_summary_path = write_cost_summary(traces)
    measurement_summary_path = write_token_math_measurement_summary(traces)

    total_planned_cost = sum(t["planned_cost"] for t in traces)
    total_measured_cost = sum(t["measured_cost"] for t in traces)
    total_adjusted_cost = sum(t["adjusted_measured_cost"] for t in traces)
    review_flag_counts: Dict[str, int] = {}
    for t in traces:
        review_flag_counts[t["review_flag"]] = review_flag_counts.get(t["review_flag"], 0) + 1

    return {
        "traces": traces,
        "total_calls": len(traces),
        "total_planned_cost": total_planned_cost,
        "total_measured_cost": total_measured_cost,
        "total_adjusted_measured_cost": total_adjusted_cost,
        "review_flag_counts": review_flag_counts,
        "stage_token_counts_path": stage_counts_path,
        "cost_summary_path": cost_summary_path,
        "measurement_summary_path": measurement_summary_path,
    }
