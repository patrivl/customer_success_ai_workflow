#!/usr/bin/env python3
"""
Customer Success AI Workflow — single-command runner.

Usage:
    python run_workflow.py

This script:
  1. Loads and validates the 7 source CSVs in data/.
  2. Runs the deterministic preprocessing layer (src/preprocessing.py):
     normalization, account_id joins, quality_standard_ids split/join,
     flags, risk/opportunity/priority/escalation scores, workflow-population
     selectors, portfolio pattern detection, and representative-run
     selection. Nothing here calls an external AI API.
  3. Joins account-scoped tables on account_id (Stage 1/2 input shape).
  4. Splits junior_outputs.csv's quality_standard_ids and joins against
     quality_standards.csv (Stage 1/2 input shape).
  5. Generates a prioritized, risk-ranked briefing for every SELECTED
     account (Stage 1) using deterministic logic behind a simulated-LLM
     interface. The selected population comes from the preprocessing
     layer's `daily_account_review` selector.
  6. Runs a quality review of every SELECTED junior output against its
     assigned quality standards (Stage 2), same simulated-LLM interface.
     The selected population comes from the preprocessing layer's
     `quality_review_outputs` selector.
  7. Runs the token-math-plan-driven simulated model layer
     (src/model_simulator.py, src/token_measurement.py): every one of the
     37 stage_ids in config/token_math_plan.csv is executed against its
     preprocessing-selected population, producing a full prompt/response/
     token/cost trace per call, measured against that stage's planned
     tokens and pricing.
  8. Assembles the final end-to-end deliverable (src/final_report.py) from
     those same traces -- no re-simulation: one JSON file per
     representative account run, portfolio-wide quality/routing/
     intervention rollup CSVs, and the top-level workflow summary
     (JSON + Markdown).

outputs/ is the ONE final output directory for this project -- every file
listed above lands directly under it (representative_runs/, preprocessing/,
legacy_stage_reports/, and the top-level CSVs/JSON/Markdown). No external
network calls or AI APIs are used anywhere in this workflow.
"""

from __future__ import annotations

import sys

from src import config, final_report, model_simulator, preprocessing, report_writer, token_measurement
from src.briefing_generator import generate_all_briefings
from src.data_loader import (
    DataValidationError,
    build_account_contexts,
    explode_quality_standards,
    load_all_tables,
)
from src.llm_simulator import SimulatedLLMClient
from src.quality_reviewer import review_all_outputs

# Shared reference "today" for renewal-window / ageing calculations, used by
# both this script and src/preprocessing.py, so results stay reproducible
# regardless of when the script is actually run.
REFERENCE_DATE = config.REFERENCE_DATE


def _hr(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def main() -> int:
    _hr("STEP 1/8 — Loading & validating source data")
    try:
        tables = load_all_tables()
    except DataValidationError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 1

    for name, count in tables.row_counts().items():
        print(f"  loaded {name:<18} {count:>3} row(s)")
    print("  all required columns present in all 7 files.")

    _hr("STEP 2/8 — Deterministic preprocessing layer (no AI, no API calls)")
    try:
        preproc = preprocessing.run_preprocessing(tables, reference_date=REFERENCE_DATE)
    except DataValidationError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 1

    selected_items = preproc["selected_workflow_items"]
    representative_runs = preproc["representative_runs"]
    patterns = preproc["portfolio_patterns"]

    print(f"  normalized 7 tables; built context for {len(preproc['account_contexts'])} account(s)")
    print("  selected workflow populations:")
    for name, items in selected_items.items():
        print(f"    {name:<28} {len(items):>3} item(s)")
    print(f"  representative end-to-end runs selected: {len(representative_runs)}")
    for r in representative_runs:
        print(f"    {r['account_id']}  {r['case_type']:<45} priority_score={r['priority_score']}")
    print(
        "  common issue themes: "
        + (", ".join(f"{k}({v})" for k, v in patterns["common_issue_themes"].items()) or "none")
    )
    print("  wrote outputs/preprocessing/{account_contexts.json, account_scores.csv,")
    print("                              portfolio_patterns.json, selected_workflow_items.json,")
    print("                              representative_accounts.json}")

    _hr("STEP 3/8 — Joining account-scoped tables on account_id (Stage 1/2 input shape)")
    contexts = build_account_contexts(tables)
    print(f"  built consolidated context for {len(contexts)} account(s)")

    _hr("STEP 4/8 — Splitting quality_standard_ids and joining quality_standards.csv")
    try:
        exploded_standards = explode_quality_standards(tables)
    except DataValidationError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 1
    n_outputs = exploded_standards["output_id"].nunique()
    n_pairs = len(exploded_standards)
    print(f"  {n_outputs} junior output(s) exploded into {n_pairs} (output, standard) pair(s)")

    # Restrict what actually reaches the simulated-LLM stages to the
    # populations the deterministic preprocessing layer selected. On this
    # small synthetic dataset the selectors return everything (there's
    # nothing to filter out yet), but wiring it this way is what makes the
    # later stages "cost-controlled" on a larger, real portfolio.
    daily_review_ids = {item["account_id"] for item in selected_items["daily_account_review"]}
    quality_review_ids = {item["output_id"] for item in selected_items["quality_review_outputs"]}
    contexts = {aid: ctx for aid, ctx in contexts.items() if aid in daily_review_ids}
    exploded_standards = exploded_standards[exploded_standards["output_id"].isin(quality_review_ids)]

    client = SimulatedLLMClient()

    _hr("STEP 5/8 — Generating account briefings (Stage 1, simulated LLM)")
    briefings = generate_all_briefings(contexts, client, REFERENCE_DATE)
    print(f"  generated {len(briefings)} briefing(s), ranked by risk_score (highest first)")
    for b in briefings[:5]:
        print(f"    [{b.priority_tier:<6}] {b.account_id}  {b.account_name:<24} risk={b.risk_score}")
    if len(briefings) > 5:
        print(f"    ... and {len(briefings) - 5} more (see outputs/legacy_stage_reports/account_briefings.md)")

    _hr("STEP 6/8 — Reviewing junior outputs against quality standards (Stage 2, simulated LLM)")
    reviews = review_all_outputs(exploded_standards, contexts, client)
    print(f"  reviewed {len(reviews)} output(s)")
    for r in reviews:
        print(f"    {r.output_id} ({r.account_id}): {r.overall_score:>5.1f}/100 -> {r.recommendation}")

    _hr("Writing legacy Stage 1/2 reports to outputs/legacy_stage_reports/ (debug only, not the final deliverable)")
    report_writer.write_account_briefings(briefings)
    report_writer.write_quality_review_report(reviews)
    report_writer.write_token_usage_log(client)
    report_writer.write_workflow_summary(briefings, reviews, client, validation_notes=[])
    for f in sorted(config.LEGACY_STAGE_REPORTS_DIR.glob("*")):
        print(f"  wrote {f.relative_to(config.PROJECT_ROOT)}")

    _hr("STEP 7/8 — Token-math-plan-driven simulated model layer (no AI, no API calls)")
    token_math_summary = token_measurement.run_token_math_layer(preproc)
    print(f"  simulated {token_math_summary['total_calls']} model call(s) across "
          f"{len(model_simulator.STAGE_RUNTIME_MAP)} planned stage_id(s)")
    print(f"  planned cost:  ${token_math_summary['total_planned_cost']:.6f}")
    print(f"  measured cost: ${token_math_summary['total_measured_cost']:.6f}")
    print(f"  adjusted measured cost (incl. retry/QA overhead): ${token_math_summary['total_adjusted_measured_cost']:.6f}")
    print("  review flag breakdown: " + ", ".join(
        f"{flag}={count}" for flag, count in sorted(token_math_summary["review_flag_counts"].items())
    ))
    for path in (
        token_math_summary["stage_token_counts_path"],
        token_math_summary["cost_summary_path"],
        token_math_summary["measurement_summary_path"],
    ):
        print(f"  wrote {path.relative_to(config.PROJECT_ROOT)}")

    _hr("STEP 8/8 — Final end-to-end orchestration (representative runs + summary, no AI, no API calls)")
    dataset_files_used = [f"data/{name}" for name in (
        "accounts.csv", "usage_events.csv", "support_tickets.csv", "call_notes.csv",
        "scheduled_checkins.csv", "junior_outputs.csv", "quality_standards.csv",
    )]
    final = final_report.run_final_reports(preproc, token_math_summary, dataset_files_used)
    print(f"  wrote {len(final['representative_runs'])} representative run file(s) to "
          f"{config.REPRESENTATIVE_RUNS_DIR.relative_to(config.PROJECT_ROOT)}/")
    for r in final["representative_runs"]:
        print(f"    {r['account_id']}  {r['representative_case_type']:<45} "
              f"final_route={r['final_route'].get('route')}")
    for path in (
        final["quality_review_results_path"],
        final["routing_decisions_path"],
        final["intervention_plans_path"],
        final["workflow_summary_json_path"],
        final["workflow_summary_md_path"],
    ):
        print(f"  wrote {path.relative_to(config.PROJECT_ROOT)}")

    cost_summary = client.summary()
    _hr("Run complete")
    print(f"  simulated LLM calls (legacy Stage 1/2): {cost_summary['total_calls']}")
    print(f"  simulated model calls (token-math layer): {token_math_summary['total_calls']}")
    print(f"  total measured cost (token-math layer):   ${final['summary']['total_measured_cost']:.6f}")
    print("\nSee outputs/workflow_summary.md and outputs/workflow_summary.json for the full final summary.\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
