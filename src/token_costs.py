"""
Token estimation and cost math for the simulated model layer.

Prices always come from config/token_math_plan.csv (via
src/token_math_config.py), never from a hardcoded pricing table -- these
functions just take the priced-per-1M-token rates as arguments.
"""

from __future__ import annotations

import math

# ---------------------------------------------------------------------------
# Review-flag variance bands.
# ---------------------------------------------------------------------------
REVIEW_FLAG_OK = "OK"
REVIEW_FLAG_ABOVE_ESTIMATE = "Review: above estimate"
REVIEW_FLAG_HIGH_ABOVE = "High variance: revise assumptions"
REVIEW_FLAG_OVERESTIMATED = "Review: overestimated"
REVIEW_FLAG_HIGH_BELOW = "High variance: estimate too conservative"
REVIEW_FLAG_PENDING = "Pending measurement"
# Used specifically by the stage-level token_math_measurement_summary.csv
# aggregation (src/token_measurement.py) when a stage_id had zero calls in
# this run -- distinct from REVIEW_FLAG_PENDING, which covers a per-call
# variance that couldn't be computed (e.g. zero planned cost).
REVIEW_FLAG_NOT_EXERCISED = "Not exercised in representative runs"


def estimate_tokens(text: str) -> int:
    """Approximate token count as ceil(characters / 4), the standard rough
    heuristic for English prose. Minimum of 1 token for any non-empty text,
    0 for empty text."""
    if not text:
        return 0
    return math.ceil(len(text) / 4)


def calculate_cost(input_tokens: int, output_tokens: int,
                    input_price_per_1m: float, output_price_per_1m: float) -> float:
    """Cost in USD for a single call, given per-1M-token prices."""
    return (
        (input_tokens / 1_000_000) * input_price_per_1m
        + (output_tokens / 1_000_000) * output_price_per_1m
    )


def calculate_adjusted_cost(base_cost: float, retry_rate: float, qa_eval_multiplier: float) -> float:
    """Adjusts a base cost for the expected retry overhead and QA/eval
    review overhead planned for this stage.

    `retry_rate` is a fraction (e.g. 0.05 for 5%): on average each run costs
    base_cost * (1 + retry_rate) once retries are accounted for.
    `qa_eval_multiplier` is applied on top of that (e.g. 1.15 means the
    stage's QA/eval overhead adds 15% on top of the retry-adjusted cost).
    """
    return base_cost * (1.0 + retry_rate) * qa_eval_multiplier


def calculate_variance(planned_cost: float, measured_cost: float) -> float:
    """Percent variance of measured vs. planned cost. Positive means the
    measured cost ran over the plan; negative means it ran under."""
    if planned_cost == 0:
        return 0.0 if measured_cost == 0 else float("inf")
    return ((measured_cost - planned_cost) / planned_cost) * 100.0


def assign_review_flag(variance_pct: float | None) -> str:
    """Maps a variance percentage to a review-flag label.

        within +/-20%:      OK
        +20% to +50%:       Review: above estimate
        above +50%:         High variance: revise assumptions
        -20% to -50%:       Review: overestimated
        below -50%:         High variance: estimate too conservative
        missing measurement: Pending measurement
    """
    if variance_pct is None:
        return REVIEW_FLAG_PENDING
    if variance_pct > 50:
        return REVIEW_FLAG_HIGH_ABOVE
    if variance_pct > 20:
        return REVIEW_FLAG_ABOVE_ESTIMATE
    if variance_pct >= -20:
        return REVIEW_FLAG_OK
    if variance_pct >= -50:
        return REVIEW_FLAG_OVERESTIMATED
    return REVIEW_FLAG_HIGH_BELOW


def format_variance_pct(variance_pct: float | None) -> str:
    """Formats a variance percentage as a spreadsheet-ready signed string,
    e.g. "+18.7%" or "-42.3%". Returns "N/A" for missing/non-finite values
    (callers with a genuinely "not exercised" stage should use the literal
    "Not exercised" string instead, not this function)."""
    if variance_pct is None or variance_pct in (float("inf"), float("-inf")):
        return "N/A"
    sign = "+" if variance_pct >= 0 else ""
    return f"{sign}{variance_pct:.1f}%"
