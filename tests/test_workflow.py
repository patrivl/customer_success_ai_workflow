"""
Lightweight sanity tests. Run with:  python -m pytest tests/ -v
(or simply `python tests/test_workflow.py` to run without pytest)
"""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import (
    build_account_contexts,
    explode_quality_standards,
    load_all_tables,
)
from src.llm_simulator import SimulatedLLMClient, estimate_tokens


def test_load_all_tables_succeeds():
    tables = load_all_tables()
    assert len(tables.accounts) > 0
    assert len(tables.quality_standards) > 0


def test_account_join_covers_all_accounts():
    tables = load_all_tables()
    contexts = build_account_contexts(tables)
    assert set(contexts.keys()) == set(tables.accounts["account_id"])


def test_account_with_tickets_has_open_tickets_populated():
    tables = load_all_tables()
    contexts = build_account_contexts(tables)
    # A008 (Harbor Insurance) has an open High severity ticket in the fixture data.
    ctx = contexts["A008"]
    assert len(ctx.open_tickets) >= 1
    assert any(t["severity"] == "High" for t in ctx.open_tickets)


def test_quality_standard_ids_split_and_join():
    tables = load_all_tables()
    exploded = explode_quality_standards(tables)
    # O001 has 4 semicolon-separated standard ids -> 4 rows after explode.
    o001_rows = exploded[exploded["output_id"] == "O001"]
    assert len(o001_rows) == 4
    assert set(o001_rows["standard_id"]) == {"QS001", "QS002", "QS003", "QS005"}
    # Every exploded row must have a non-null standard_name from the join.
    assert exploded["standard_name"].isna().sum() == 0


def test_token_estimation_is_positive():
    assert estimate_tokens("hello world") > 0
    assert estimate_tokens("") == 1  # floor of at least 1 token


def test_simulated_llm_client_logs_calls():
    client = SimulatedLLMClient()
    result = client.call(
        task="unit_test", reference_id="X1", prompt="some prompt text",
        response_fn=lambda: "some response text",
    )
    assert result == "some response text"
    assert len(client.call_log) == 1
    assert client.call_log[0].total_tokens > 0
    assert client.total_cost() > 0


def test_preprocessing_health_score_delta_is_current_minus_previous():
    from src import preprocessing
    tables = load_all_tables()
    result = preprocessing.run_preprocessing(tables)
    ctx = result["account_contexts"]["A008"]
    # Harbor Insurance: current=48, previous=66 -> delta = 48 - 66 = -18
    assert ctx["health_score_delta"] == ctx["current_health_score"] - ctx["previous_health_score"]
    assert ctx["health_score_delta"] < 0
    assert ctx["flags"]["severe_decline_flag"] is True


def test_preprocessing_representative_runs_match_preferred_ids():
    from src import config as cfg
    from src import preprocessing
    tables = load_all_tables()
    result = preprocessing.run_preprocessing(tables)
    reps = result["representative_runs"]
    assert cfg.MIN_REPRESENTATIVE_RUNS <= len(reps) <= cfg.MAX_REPRESENTATIVE_RUNS
    rep_ids = {r["account_id"] for r in reps}
    # All 5 preferred IDs are present in this dataset, so they must be used.
    assert rep_ids.issuperset({"A008", "A005", "A014", "A003", "A017"})
    # Additional case types should be filled in from the deterministic
    # selectors (never hardcoded), covering as many distinct cases as
    # this dataset supports.
    case_types = {r["case_type"] for r in reps}
    assert case_types.issuperset({
        "severe_support_escalation_case", "renewal_value_review_case",
        "negative_sentiment_declining_adoption_case", "healthy_expansion_opportunity_case",
        "low_usage_reactivation_intervention_case",
    })


def test_preprocessing_selected_workflow_items_cover_all_selectors():
    from src import preprocessing
    tables = load_all_tables()
    result = preprocessing.run_preprocessing(tables)
    items = result["selected_workflow_items"]
    expected_keys = {
        "daily_account_review", "second_pass_validation", "flagged_account_summary",
        "csm_alerts", "unresolved_items", "inbound_issues", "issue_pattern_review",
        "scheduled_checkins", "quality_review_outputs", "failed_or_weak_outputs",
        "intervention_candidates", "complex_escalation_candidates",
    }
    assert expected_keys.issubset(items.keys())
    # Every selected item must carry a non-empty deterministic reason.
    for name, entries in items.items():
        for entry in entries:
            assert entry.get("reason"), f"{name} entry missing a reason: {entry}"
    # Daily review covers the whole portfolio; quality review covers every output.
    assert len(items["daily_account_review"]) == len(tables.accounts)
    assert len(items["quality_review_outputs"]) == tables.junior_outputs["output_id"].nunique()


def test_preprocessing_outputs_written_to_disk():
    from src import preprocessing, config as cfg
    tables = load_all_tables()
    preprocessing.run_preprocessing(tables)
    out_dir = cfg.PREPROCESSING_OUTPUT_DIR
    for filename in (
        "account_contexts.json", "account_scores.csv", "portfolio_patterns.json",
        "selected_workflow_items.json", "representative_accounts.json",
    ):
        assert (out_dir / filename).exists(), f"missing {filename}"


def test_token_math_plan_loads_and_validates():
    from src.token_math_config import load_token_math_plan
    plan = load_token_math_plan()
    assert len(plan) == 37
    assert "TM_001" in plan and "TM_037" in plan
    tm002 = plan["TM_002"]
    assert tm002.model == "Claude Haiku 4.5"
    assert tm002.planned_input_tokens_per_run == 3500
    assert 0 < tm002.retry_rate < 1  # retry_rate_percent (5) -> 0.05 fraction


def test_token_math_plan_unknown_stage_raises():
    from src.token_math_config import TokenMathConfigError, get_stage
    try:
        get_stage("TM_999")
        raised = False
    except TokenMathConfigError:
        raised = True
    assert raised, "get_stage() should raise TokenMathConfigError for an unknown stage_id"


def test_token_cost_math():
    from src import token_costs
    assert token_costs.estimate_tokens("") == 0
    assert token_costs.estimate_tokens("abcd") == 1
    assert token_costs.estimate_tokens("abcde") == 2
    cost = token_costs.calculate_cost(1_000_000, 1_000_000, 1.0, 5.0)
    assert cost == 6.0
    adjusted = token_costs.calculate_adjusted_cost(1.0, retry_rate=0.10, qa_eval_multiplier=1.20)
    assert round(adjusted, 4) == round(1.0 * 1.10 * 1.20, 4)
    assert token_costs.calculate_variance(1.0, 1.0) == 0.0
    assert round(token_costs.calculate_variance(1.0, 1.3), 6) == 30.0
    assert token_costs.assign_review_flag(10) == token_costs.REVIEW_FLAG_OK
    assert token_costs.assign_review_flag(30) == token_costs.REVIEW_FLAG_ABOVE_ESTIMATE
    assert token_costs.assign_review_flag(60) == token_costs.REVIEW_FLAG_HIGH_ABOVE
    assert token_costs.assign_review_flag(-30) == token_costs.REVIEW_FLAG_OVERESTIMATED
    assert token_costs.assign_review_flag(-60) == token_costs.REVIEW_FLAG_HIGH_BELOW
    assert token_costs.assign_review_flag(None) == token_costs.REVIEW_FLAG_PENDING


def test_simulate_model_call_produces_full_trace():
    from src import preprocessing
    from src.model_simulator import simulate_model_call

    tables = load_all_tables()
    result = preprocessing.run_preprocessing(tables)
    account_ctx = result["account_contexts"]["A008"]
    context = {**account_ctx, "item_id": "A008", "item_type": "account", "selector_reason": "unit test"}

    trace = simulate_model_call("TM_002", context, "account_review_prompt", run_id="unit-test-1")

    for field in (
        "run_id", "stage_id", "workflow_component", "operating_area", "trigger_schedule",
        "model", "prompt_text", "result", "planned_input_tokens", "planned_output_tokens",
        "measured_input_tokens", "measured_output_tokens", "planned_cost", "measured_cost",
        "retry_rate", "qa_eval_multiplier", "adjusted_measured_cost", "variance_pct",
        "review_flag", "confidence",
    ):
        assert field in trace, f"trace missing field {field}"

    assert trace["stage_id"] == "TM_002"
    assert trace["result"]["account_id"] == "A008"
    assert trace["result"]["risk_level"] in ("low", "medium", "high")
    assert 0.0 <= trace["confidence"] <= 1.0
    assert trace["measured_input_tokens"] > 0


def test_token_measurement_runs_all_stages_and_writes_outputs(tmp_path=None):
    from src import config as cfg
    from src import preprocessing, token_measurement

    tables = load_all_tables()
    preproc = preprocessing.run_preprocessing(tables)
    summary = token_measurement.run_token_math_layer(preproc)

    assert summary["total_calls"] > 0
    # Every trace's item population size should match its preprocessing selector.
    from src.model_simulator import STAGE_RUNTIME_MAP
    calls_by_stage = {}
    for t in summary["traces"]:
        calls_by_stage.setdefault(t["stage_id"], 0)
        calls_by_stage[t["stage_id"]] += 1
    assert set(calls_by_stage.keys()) == set(STAGE_RUNTIME_MAP.keys())

    for path in (
        summary["stage_token_counts_path"],
        summary["cost_summary_path"],
        summary["measurement_summary_path"],
    ):
        assert path.exists()

    with open(cfg.OUTPUT_DIR / "token_math_measurement_summary.csv") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 37
    assert {r["stage_id"] for r in rows} == set(STAGE_RUNTIME_MAP.keys())


def test_final_report_writes_representative_runs_and_rollup_csvs():
    from src import config as cfg
    from src import final_report, preprocessing, token_measurement

    tables = load_all_tables()
    preproc = preprocessing.run_preprocessing(tables)
    token_math_summary = token_measurement.run_token_math_layer(preproc)
    dataset_files_used = [f"data/{name}" for name in (
        "accounts.csv", "usage_events.csv", "support_tickets.csv", "call_notes.csv",
        "scheduled_checkins.csv", "junior_outputs.csv", "quality_standards.csv",
    )]

    final = final_report.run_final_reports(preproc, token_math_summary, dataset_files_used)

    assert len(final["representative_runs"]) >= 5
    for run in final["representative_runs"]:
        for field in (
            "run_id", "account_id", "account_name", "representative_case_type",
            "stages_completed", "account_review_outputs", "prioritization_outputs",
            "inbound_issue_outputs", "checkin_outputs", "quality_review_outputs",
            "intervention_outputs", "routing_outputs", "evaluation_flags", "final_route",
            "total_measured_input_tokens", "total_measured_output_tokens",
            "total_measured_cost", "total_adjusted_measured_cost", "stage_traces",
        ):
            assert field in run, f"representative run missing field {field}"

    rep_dir = cfg.REPRESENTATIVE_RUNS_DIR
    rep_files = list(rep_dir.glob("*.json"))
    assert len(rep_files) >= 5

    for path in (
        final["quality_review_results_path"], final["routing_decisions_path"],
        final["intervention_plans_path"], final["workflow_summary_json_path"],
        final["workflow_summary_md_path"],
    ):
        assert path.exists()

    summary = final["summary"]
    for field in (
        "generated_at", "dataset_files_used", "total_representative_runs",
        "representative_accounts", "workflow_components_covered", "operating_areas_covered",
        "total_simulated_model_calls", "total_measured_input_tokens", "total_measured_output_tokens",
        "total_measured_cost", "total_adjusted_measured_cost",
        "measured_average_cost_per_end_to_end_run", "average_input_tokens_per_run",
        "average_output_tokens_per_run", "quality_flags_total", "escalation_cases_total",
        "intervention_plans_total", "final_routes_by_account", "measurement_files_created",
    ):
        assert field in summary, f"workflow summary missing field {field}"


def test_run_workflow_end_to_end_creates_all_final_outputs():
    """The real acceptance test: `python3 run_workflow.py` end-to-end,
    verifying every Submission B final output lands under outputs/."""
    import subprocess
    from src import config as cfg

    proj_root = cfg.PROJECT_ROOT
    result = subprocess.run(
        [sys.executable, "run_workflow.py"],
        cwd=str(proj_root), capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, f"run_workflow.py failed:\n{result.stdout}\n{result.stderr}"

    out = cfg.OUTPUT_DIR
    assert (out / "workflow_summary.json").exists()
    assert (out / "workflow_summary.md").exists()
    assert (out / "representative_runs").is_dir()
    rep_files = list((out / "representative_runs").glob("*.json"))
    assert len(rep_files) >= 5, f"expected >=5 representative run files, found {len(rep_files)}"

    for filename in (
        "stage_token_counts.csv", "cost_summary.csv", "token_math_measurement_summary.csv",
        "quality_review_results.csv", "routing_decisions.csv", "intervention_plans.csv",
    ):
        assert (out / filename).exists(), f"missing outputs/{filename}"

    assert not (proj_root / "output").exists(), "stale top-level output/ directory should no longer be written"


if __name__ == "__main__":
    tests = [obj for name, obj in list(globals().items()) if name.startswith("test_")]
    failures = 0
    for t in tests:
        try:
            t()
            print(f"PASS: {t.__name__}")
        except AssertionError as e:
            failures += 1
            print(f"FAIL: {t.__name__}: {e}")
    if failures:
        raise SystemExit(f"{failures} test(s) failed")
    print(f"\nAll {len(tests)} tests passed.")
