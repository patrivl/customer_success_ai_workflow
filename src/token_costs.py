"""
Token estimation and cost math for the simulated model layer.

Prices always come from config/token_math_plan.csv (via
src/token_math_config.py), never from a hardcoded pricing table -- these
functions just take the priced-per-1M-token rates as arguments.
"""

from __future__ import annotations

import math

# ---------------------------------------------------------------------------
# Review-flag variance bands. Wording describes measured cost relative to
# the original (planned) estimate.
# ---------------------------------------------------------------------------
REVIEW_FLAG_ON_PAR = "Measured cost on par with original estimate"
REVIEW_FLAG_ABOVE_ESTIMATE = "Measured cost above original estimate"
REVIEW_FLAG_MATERIALLY_ABOVE = "Measured cost materially above original estimate"
REVIEW_FLAG_BELOW_ESTIMATE = "Measured cost below original estimate"
REVIEW_FLAG_MATERIALLY_BELOW = "Measured cost materially below original estimate"
# Covers both a per-call variance that couldn't be computed (e.g. zero
# planned cost) and a stage_id with zero calls in this run -- one flag for
# "there is no measured cost to compare against the estimate."
REVIEW_FLAG_NOT_MEASURED = "Not measured"


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
    """Maps a variance percentage to a review-flag label describing
    measured cost relative to the original estimate.

        within +/-20%:        Measured cost on par with original estimate
        +20% to +50%:         Measured cost above original estimate
        more than +50%:       Measured cost materially above original estimate
        -20% to -50%:         Measured cost below original estimate
        less than -50%:       Measured cost materially below original estimate
        no measurement:       Not measured
    """
    if variance_pct is None:
        return REVIEW_FLAG_NOT_MEASURED
    if variance_pct > 50:
        return REVIEW_FLAG_MATERIALLY_ABOVE
    if variance_pct > 20:
        return REVIEW_FLAG_ABOVE_ESTIMATE
    if variance_pct >= -20:
        return REVIEW_FLAG_ON_PAR
    if variance_pct >= -50:
        return REVIEW_FLAG_BELOW_ESTIMATE
    return REVIEW_FLAG_MATERIALLY_BELOW


def format_variance_pct(variance_pct: float | None) -> str:
    """Formats a variance percentage as a spreadsheet-ready signed string,
    e.g. "+18.7%" or "-42.3%". Returns "N/A" for missing/non-finite values
    (callers with a genuinely "not exercised" stage should use the literal
    "Not exercised" string instead, not this function)."""
    if variance_pct is None or variance_pct in (float("inf"), float("-inf")):
        return "N/A"
    sign = "+" if variance_pct >= 0 else ""
    return f"{sign}{variance_pct:.1f}%"
