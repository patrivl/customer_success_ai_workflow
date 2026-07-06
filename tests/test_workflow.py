"""
Lightweight sanity tests. Run with:  python -m pytest tests/ -v
(or simply `python tests/test_workflow.py` to run without pytest)
"""

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
    from src import preprocessing
    tables = load_all_tables()
    result = preprocessing.run_preprocessing(tables)
    reps = result["representative_runs"]
    assert len(reps) >= 5
    rep_ids = {r["account_id"] for r in reps}
    # All 5 preferred IDs are present in this dataset, so they must be used.
    assert rep_ids.issuperset({"A008", "A005", "A014", "A003", "A017"})


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
