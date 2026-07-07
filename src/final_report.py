"""
Final end-to-end orchestration / Submission-B reporting.

This module does NOT re-run or duplicate the token/model simulation layer
(src/model_simulator.py, src/token_measurement.py) -- it consumes the call
traces those modules already produced (one full prompt/response/token/cost
trace per simulated stage call, across every stage_id in
config/token_math_plan.csv) and:

  1. Groups those traces by account and assembles one JSON file per
     representative end-to-end account run under
     outputs/representative_runs/.
  2. Extracts portfolio-wide quality/routing/intervention rollups
     (outputs/quality_review_results.csv, outputs/routing_decisions.csv,
     outputs/intervention_plans.csv) by filtering the same trace list by
     prompt template -- no new simulated calls are made.
  3. Writes the top-level outputs/workflow_summary.json and
     outputs/workflow_summary.md.

No external AI API is called anywhere in this module.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from typing import Any, Dict, List

from src import config, token_measurement
from src.model_simulator import STAGE_RUNTIME_MAP
from src.token_math_config import load_token_math_plan

ROUTE_LIKE_TEMPLATES = {"routing_prompt", "complex_escalation_prompt"}
ESCALATION_ROUTES = {"manager_escalation", "specialist_escalation"}

CATEGORY_BY_TEMPLATE = {
    "account_review_prompt": "account_review_outputs",
    "prioritization_prompt": "prioritization_outputs",
    "inbound_issue_prompt": "inbound_issue_outputs",
    "checkin_support_prompt": "checkin_outputs",
    "quality_review_prompt": "quality_review_outputs",
    "intervention_planning_prompt": "intervention_outputs",
    "routing_prompt": "routing_outputs",
    "complex_escalation_prompt": "routing_outputs",
}


def _template_of(trace: dict) -> str:
    return STAGE_RUNTIME_MAP[trace["stage_id"]]["template"]


def _ensure_output_dirs() -> None:
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config.REPRESENTATIVE_RUNS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Representative end-to-end run assembly
# ---------------------------------------------------------------------------

def group_traces_by_account(traces: List[dict]) -> Dict[str, List[dict]]:
    by_account: Dict[str, List[dict]] = {}
    for t in traces:
        aid = t.get("account_id")
        if not aid:
            continue  # issue_theme-kind traces aren't scoped to one account
        by_account.setdefault(aid, []).append(t)
    return by_account


def build_representative_run(account_ctx: dict, case_type: str, reason: str,
                              account_traces: List[dict]) -> dict:
    account_id = account_ctx["account_id"]
    buckets: Dict[str, List[dict]] = {v: [] for v in set(CATEGORY_BY_TEMPLATE.values())}
    evaluation_flags: List[dict] = []
    stages_completed: List[str] = []
    total_input_tokens = total_output_tokens = 0
    total_measured_cost = total_adjusted_cost = 0.0

    for t in account_traces:
        template = _template_of(t)
        stages_completed.append(t["stage_id"])
        total_input_tokens += t["measured_input_tokens"]
        total_output_tokens += t["measured_output_tokens"]
        total_measured_cost += t["measured_cost"]
        total_adjusted_cost += t["adjusted_measured_cost"]

        category = CATEGORY_BY_TEMPLATE.get(template)
        if category:
            buckets[category].append({"stage_id": t["stage_id"], "run_id": t["run_id"], **t["result"]})

        if template == "quality_review_prompt":
            evaluation_flags.append({
                "stage_id": t["stage_id"],
                "passed": t["result"].get("passed"),
                "quality_score": t["result"].get("quality_score"),
                "issues_found": t["result"].get("issues_found", []),
            })

    route_traces = [t for t in account_traces if _template_of(t) in ROUTE_LIKE_TEMPLATES]
    if route_traces:
        last = route_traces[-1]
        r = last["result"]
        if _template_of(last) == "complex_escalation_prompt":
            final_route = {
                "stage_id": last["stage_id"],
                "route": "manager_escalation" if r.get("requires_manager_review") else "specialist_escalation",
                "owner": "CSM manager" if r.get("requires_manager_review") else "Support specialist",
                "urgency": "immediate",
            }
        else:
            final_route = {
                "stage_id": last["stage_id"],
                "route": r.get("route"),
                "owner": r.get("owner"),
                "urgency": r.get("urgency"),
            }
    else:
        final_route = {"stage_id": None, "route": "not_routed_this_cycle", "owner": None, "urgency": None}

    return {
        "run_id": f"rep-{account_id}",
        "account_id": account_id,
        "account_name": account_ctx.get("account_name"),
        "representative_case_type": case_type,
        "selection_reason": reason,
        "stages_completed": stages_completed,
        "account_review_outputs": buckets["account_review_outputs"],
        "prioritization_outputs": buckets["prioritization_outputs"],
        "inbound_issue_outputs": buckets["inbound_issue_outputs"],
        "checkin_outputs": buckets["checkin_outputs"],
        "quality_review_outputs": buckets["quality_review_outputs"],
        "intervention_outputs": buckets["intervention_outputs"],
        "routing_outputs": buckets["routing_outputs"],
        "evaluation_flags": evaluation_flags,
        "final_route": final_route,
        "total_measured_input_tokens": total_input_tokens,
        "total_measured_output_tokens": total_output_tokens,
        "total_measured_cost": round(total_measured_cost, 6),
        "total_adjusted_measured_cost": round(total_adjusted_cost, 6),
        "stage_traces": account_traces,
    }


def write_representative_runs(representative_selection: List[dict], account_contexts: Dict[str, dict],
                               traces_by_account: Dict[str, List[dict]]) -> List[dict]:
    """Writes one JSON file per representative run to
    outputs/representative_runs/ and returns the assembled run dicts."""
    _ensure_output_dirs()
    runs = []
    for entry in representative_selection:
        account_id = entry["account_id"]
        account_ctx = account_contexts.get(account_id)
        if account_ctx is None:
            continue
        run = build_representative_run(
            account_ctx, entry["case_type"], entry["reason"],
            traces_by_account.get(account_id, []),
        )
        runs.append(run)
        path = config.REPRESENTATIVE_RUNS_DIR / f"{account_id}.json"
        path.write_text(json.dumps(run, indent=2, default=str), encoding="utf-8")
    return runs


# ---------------------------------------------------------------------------
# 2. Portfolio-wide rollup CSVs (filtered straight from the trace list)
# ---------------------------------------------------------------------------

def write_quality_review_results_csv(traces: List[dict], out_dir=None):
    out_dir = out_dir or config.OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "quality_review_results.csv"
    fieldnames = [
        "run_id", "stage_id", "output_id", "account_id", "passed",
        "quality_score", "route", "failed_standards", "confidence",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for t in traces:
            if _template_of(t) != "quality_review_prompt":
                continue
            r = t["result"]
            writer.writerow({
                "run_id": t["run_id"],
                "stage_id": t["stage_id"],
                "output_id": r.get("output_id"),
                "account_id": r.get("account_id"),
                "passed": r.get("passed"),
                "quality_score": r.get("quality_score"),
                "route": r.get("route"),
                "failed_standards": ";".join(r.get("failed_standards") or []),
                "confidence": r.get("confidence"),
            })
    return path


def write_routing_decisions_csv(traces: List[dict], out_dir=None):
    out_dir = out_dir or config.OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "routing_decisions.csv"
    fieldnames = [
        "run_id", "stage_id", "item_id", "item_type", "account_id",
        "route", "owner", "urgency", "confidence", "notes",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for t in traces:
            template = _template_of(t)
            if template not in ROUTE_LIKE_TEMPLATES:
                continue
            r = t["result"]
            if template == "complex_escalation_prompt":
                row = {
                    "run_id": t["run_id"], "stage_id": t["stage_id"],
                    "item_id": r.get("item_id"), "item_type": "account",
                    "account_id": r.get("account_id"),
                    "route": "manager_escalation" if r.get("requires_manager_review") else "specialist_escalation",
                    "owner": "CSM manager" if r.get("requires_manager_review") else "Support specialist",
                    "urgency": "immediate",
                    "confidence": r.get("confidence"),
                    "notes": r.get("recommended_resolution"),
                }
            else:
                row = {
                    "run_id": t["run_id"], "stage_id": t["stage_id"],
                    "item_id": r.get("item_id"), "item_type": r.get("item_type"),
                    "account_id": t.get("account_id"),
                    "route": r.get("route"), "owner": r.get("owner"), "urgency": r.get("urgency"),
                    "confidence": r.get("confidence"), "notes": r.get("reason"),
                }
            writer.writerow(row)
    return path


def write_intervention_plans_csv(traces: List[dict], out_dir=None):
    out_dir = out_dir or config.OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "intervention_plans.csv"
    fieldnames = [
        "run_id", "stage_id", "account_or_segment_id", "problem_pattern",
        "owner", "timeline", "num_actions", "num_success_measures", "confidence",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for t in traces:
            if _template_of(t) != "intervention_planning_prompt":
                continue
            r = t["result"]
            writer.writerow({
                "run_id": t["run_id"], "stage_id": t["stage_id"],
                "account_or_segment_id": r.get("account_or_segment_id"),
                "problem_pattern": r.get("problem_pattern"),
                "owner": r.get("owner"), "timeline": r.get("timeline"),
                "num_actions": len(r.get("intervention_actions") or []),
                "num_success_measures": len(r.get("success_measures") or []),
                "confidence": r.get("confidence"),
            })
    return path


# ---------------------------------------------------------------------------
# 3. Top-level workflow summary (JSON + Markdown)
# ---------------------------------------------------------------------------

def build_workflow_summary(traces: List[dict], representative_runs: List[dict],
                            dataset_files_used: List[str], measurement_files_created: List[str],
                            generated_at: str) -> Dict[str, Any]:
    plan = load_token_math_plan()
    workflow_components_covered = sorted({s.workflow_component for s in plan.values()})
    operating_areas_covered = sorted({s.operating_area for s in plan.values()})

    # Reuses src.token_measurement's own row-building (no duplicated math)
    # to roll each stage's planned_annual_cost (Adjusted cost/run x Runs
    # per cadence x Annualization factor) up to the portfolio total -- this
    # should match the Token Math Template's own budget total, allowing for
    # minor rounding.
    total_planned_annual_cost = sum(
        r["planned_annual_cost"] for r in token_measurement._build_measurement_rows(traces)
    )

    total_calls = len(traces)
    total_input = sum(t["measured_input_tokens"] for t in traces)
    total_output = sum(t["measured_output_tokens"] for t in traces)
    total_cost = sum(t["measured_cost"] for t in traces)
    total_adjusted_cost = sum(t["adjusted_measured_cost"] for t in traces)

    quality_flags_total = sum(
        1 for t in traces if _template_of(t) == "quality_review_prompt" and t["result"].get("passed") is False
    )
    escalation_cases_total = sum(1 for t in traces if _template_of(t) == "complex_escalation_prompt") + sum(
        1 for t in traces if _template_of(t) == "routing_prompt" and t["result"].get("route") in ESCALATION_ROUTES
    )
    intervention_plans_total = sum(1 for t in traces if _template_of(t) == "intervention_planning_prompt")

    n_runs = len(representative_runs) or 1
    measured_average_cost_per_end_to_end_run = round(
        sum(r["total_measured_cost"] for r in representative_runs) / n_runs, 6
    )
    average_input_tokens_per_run = round(
        sum(r["total_measured_input_tokens"] for r in representative_runs) / n_runs, 1
    )
    average_output_tokens_per_run = round(
        sum(r["total_measured_output_tokens"] for r in representative_runs) / n_runs, 1
    )

    final_routes_by_account = {
        r["account_id"]: r["final_route"].get("route") for r in representative_runs
    }

    return {
        "generated_at": generated_at,
        "dataset_files_used": dataset_files_used,
        "total_representative_runs": len(representative_runs),
        "representative_accounts": [
            {
                "account_id": r["account_id"],
                "account_name": r["account_name"],
                "representative_case_type": r["representative_case_type"],
                "selection_reason": r["selection_reason"],
            }
            for r in representative_runs
        ],
        "workflow_components_covered": workflow_components_covered,
        "operating_areas_covered": operating_areas_covered,
        "total_simulated_model_calls": total_calls,
        "total_measured_input_tokens": total_input,
        "total_measured_output_tokens": total_output,
        "total_measured_cost": round(total_cost, 6),
        "total_adjusted_measured_cost": round(total_adjusted_cost, 6),
        # Planned annual cost across all 37 stage_ids, using each stage's
        # ADJUSTED cost/run (not base cost/run) x runs_per_cadence x
        # annualization_factor -- comparable to the Token Math Template's
        # own budget total (~$12,517.52, allowing for minor rounding).
        "total_planned_annual_cost": round(total_planned_annual_cost, 2),
        "measured_average_cost_per_end_to_end_run": measured_average_cost_per_end_to_end_run,
        "average_input_tokens_per_run": average_input_tokens_per_run,
        "average_output_tokens_per_run": average_output_tokens_per_run,
        "quality_flags_total": quality_flags_total,
        "escalation_cases_total": escalation_cases_total,
        "intervention_plans_total": intervention_plans_total,
        "final_routes_by_account": final_routes_by_account,
        "measurement_files_created": measurement_files_created,
        "no_external_ai_apis_called": True,
    }


def write_workflow_summary_json(summary: Dict[str, Any], out_dir=None):
    out_dir = out_dir or config.OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "workflow_summary.json"
    path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return path


def write_workflow_summary_md(summary: Dict[str, Any], out_dir=None):
    out_dir = out_dir or config.OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "workflow_summary.md"

    lines = ["# Customer Success AI Workflow — Final Run Summary\n"]
    lines.append(f"_Generated: {summary['generated_at']}_\n")
    lines.append(
        "**No external AI APIs are called anywhere in this workflow.** Every "
        "stage below is executed by the deterministic simulated model layer "
        "(`src/model_simulator.py`), grounded in real joined account/ticket/"
        "check-in/output data.\n"
    )

    lines.append("## Dataset used")
    for f in summary["dataset_files_used"]:
        lines.append(f"- `{f}`")

    lines.append("\n## Representative end-to-end runs")
    lines.append(f"Selected {summary['total_representative_runs']} representative account run(s), "
                  f"covering {len(summary['representative_accounts'])} distinct case(s):\n")
    for acc in summary["representative_accounts"]:
        lines.append(
            f"- **{acc['account_name']}** (`{acc['account_id']}`) — "
            f"*{acc['representative_case_type']}* -- {acc['selection_reason']}"
        )
    lines.append("\nOne JSON file per representative run is written to "
                  "`outputs/representative_runs/<account_id>.json`.")

    lines.append("\n## Workflow components & operating areas covered")
    lines.append("Components: " + ", ".join(summary["workflow_components_covered"]))
    lines.append("\nOperating areas: " + ", ".join(summary["operating_areas_covered"]))

    lines.append("\n## Token / cost measurement")
    lines.append(f"- Total simulated model calls: {summary['total_simulated_model_calls']}")
    lines.append(
        f"- Total measured tokens: {summary['total_measured_input_tokens']} input + "
        f"{summary['total_measured_output_tokens']} output"
    )
    lines.append(f"- Total measured cost: ${summary['total_measured_cost']:.6f}")
    lines.append(f"- Total adjusted measured cost (incl. retry/QA overhead): ${summary['total_adjusted_measured_cost']:.6f}")
    lines.append(
        f"- Total planned annual cost (all 37 stages, using each stage's Adjusted cost/run x "
        f"Runs per cadence x Annualization factor): ${summary['total_planned_annual_cost']:,.2f} -- "
        f"should match the Token Math Template's own budget total (~$12,517.52), allowing for "
        f"minor rounding."
    )
    lines.append(
        f"- Average per representative end-to-end run: "
        f"${summary['measured_average_cost_per_end_to_end_run']:.6f} "
        f"({summary['average_input_tokens_per_run']} input tokens + "
        f"{summary['average_output_tokens_per_run']} output tokens)"
    )
    lines.append("\nDetailed per-call and per-stage figures: `outputs/stage_token_counts.csv`, "
                  "`outputs/cost_summary.csv`, `outputs/token_math_measurement_summary.csv`.")

    lines.append("\n## Measurement methodology & Token Math Template export")
    lines.append(
        "`outputs/token_math_measurement_summary.csv` has one row for every one of the 37 "
        "`stage_id`s in `config/token_math_plan.csv` -- including any that had zero calls in a "
        "given run (marked `Not measured`). Its `planned_cost_per_run` column is each stage's "
        "**Adjusted cost/run** (base cost x (1 + retry_rate) x qa_eval_multiplier) -- the plan's "
        "actual final per-run estimate, matching the Token Math Template, not the unadjusted base "
        "token cost. Its `notebook_measured_avg_cost_per_run` column is the average cost per call "
        "**including** each stage's planned retry rate and QA/eval multiplier (the realistic "
        "per-run cost), and `estimate_vs_measured_variance` is the percent difference between "
        "that figure and `planned_cost_per_run` (adjusted vs. adjusted, an apples-to-apples "
        "comparison), using the same bands as `review_flag` (±20% = Measured cost on par with "
        "original estimate, +20% to +50% = Measured cost above original estimate, more than +50% "
        "= Measured cost materially above original estimate, -20% to -50% = Measured cost below "
        "original estimate, less than -50% = Measured cost materially below original estimate, no "
        "measurement = Not measured). `outputs/token_math_spreadsheet_export.csv` is the same data "
        "narrowed to exactly the "
        "Token Math Template's measurement columns (`stage_id`, `notebook_measured_avg_cost_per_run`, "
        "`estimate_vs_measured_variance`, `source_measurement_link`, `review_flag`, plus `exercised`) "
        "-- one row per stage, ready to paste back into the spreadsheet."
    )
    lines.append(
        "\nMeasured costs are based on the runnable synthetic dataset and therefore reflect short "
        "sample prompts, not full production context. The Token Math Template remains the "
        "conservative production budget. The measured columns verify that the workflow logs "
        "tokens, costs, and variance correctly; they are not intended to replace the "
        "production-scale budget estimate."
    )

    lines.append("\n## Quality review summary")
    lines.append(f"- Quality flags (failed review) across the portfolio: {summary['quality_flags_total']}")
    lines.append("- Full detail: `outputs/quality_review_results.csv`")

    lines.append("\n## Routing / escalation summary")
    lines.append(f"- Escalation-routed cases across the portfolio: {summary['escalation_cases_total']}")
    lines.append("- Final route per representative account:")
    for account_id, route in summary["final_routes_by_account"].items():
        lines.append(f"  - {account_id}: {route}")
    lines.append("- Full detail: `outputs/routing_decisions.csv`")

    lines.append("\n## Intervention planning summary")
    lines.append(f"- Intervention plans generated across the portfolio: {summary['intervention_plans_total']}")
    lines.append("- Full detail: `outputs/intervention_plans.csv`")

    lines.append("\n## Files written")
    for f in summary["measurement_files_created"]:
        lines.append(f"- `{f}`")

    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_final_reports(preproc: Dict[str, Any], token_math_summary: Dict[str, Any],
                       dataset_files_used: List[str]) -> Dict[str, Any]:
    """Assembles representative runs + portfolio rollups + the top-level
    workflow summary from the already-computed simulation traces. Called
    once by run_workflow.py after src.token_measurement.run_token_math_layer()."""
    _ensure_output_dirs()

    traces = token_math_summary["traces"]
    account_contexts = preproc["account_contexts"]
    representative_selection = preproc["representative_runs"]

    traces_by_account = group_traces_by_account(traces)
    representative_runs = write_representative_runs(representative_selection, account_contexts, traces_by_account)

    quality_path = write_quality_review_results_csv(traces)
    routing_path = write_routing_decisions_csv(traces)
    intervention_path = write_intervention_plans_csv(traces)

    measurement_files_created = [
        str(p.relative_to(config.PROJECT_ROOT)) for p in (
            token_math_summary["stage_token_counts_path"],
            token_math_summary["cost_summary_path"],
            token_math_summary["measurement_summary_path"],
            token_math_summary["spreadsheet_export_path"],
            quality_path,
            routing_path,
            intervention_path,
        )
    ]

    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    summary = build_workflow_summary(
        traces, representative_runs, dataset_files_used, measurement_files_created, generated_at,
    )
    summary_json_path = write_workflow_summary_json(summary)
    summary_md_path = write_workflow_summary_md(summary)

    return {
        "representative_runs": representative_runs,
        "quality_review_results_path": quality_path,
        "routing_decisions_path": routing_path,
        "intervention_plans_path": intervention_path,
        "workflow_summary_json_path": summary_json_path,
        "workflow_summary_md_path": summary_md_path,
        "summary": summary,
    }
