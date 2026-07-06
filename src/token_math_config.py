"""
Loader/validator for config/token_math_plan.csv.

This is the single source of truth for planned workflow stages, model
routing, planned token estimates, pricing, retry/QA multipliers, and
cadence/annualization assumptions. Nothing in this module calls an
external API or depends on any Excel workbook -- it only reads the CSV
that already exists in this repo.

`stage_id` is the unique identifier for a planned workflow stage.
`workflow_component` + `operating_area` is intentionally NOT unique --
several stages share the same component/area but differ by
`trigger_schedule` (e.g. TM_004 and TM_005 are both "Account review" /
"Synthesis & recommendation" but one is a second-pass validation run and
the other is a flagged-account summary run).

Column-name note: the CSV's retry-rate column is named
`retry_rate_percent` (a plain percentage, e.g. `5` for 5%) rather than
`retry_rate`. This loader validates against the actual header and exposes
the value on `StagePlan.retry_rate` as a 0-1 fraction (`retry_rate_percent
/ 100`) for direct use in cost math.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict

from src import config

TOKEN_MATH_PLAN_CSV = config.PROJECT_ROOT / "config" / "token_math_plan.csv"

# Column names as they actually appear in config/token_math_plan.csv.
REQUIRED_COLUMNS = [
    "stage_id",
    "workflow_component",
    "operating_area",
    "trigger_schedule",
    "runs_per_cadence",
    "cadence",
    "annualization_factor",
    "model",
    "planned_input_tokens_per_run",
    "planned_output_tokens_per_run",
    "input_price_per_1m",
    "output_price_per_1m",
    "retry_rate_percent",
    "qa_eval_multiplier",
    "candidate_notes_assumptions",
]


class TokenMathConfigError(Exception):
    """Raised when config/token_math_plan.csv is missing, malformed, or
    fails validation."""


@dataclass(frozen=True)
class StagePlan:
    """One row of config/token_math_plan.csv, coerced to the right types."""

    stage_id: str
    workflow_component: str
    operating_area: str
    trigger_schedule: str
    runs_per_cadence: float
    cadence: str
    annualization_factor: float
    model: str
    planned_input_tokens_per_run: int
    planned_output_tokens_per_run: int
    input_price_per_1m: float
    output_price_per_1m: float
    retry_rate: float  # fraction, e.g. 0.05 for a 5% retry rate
    qa_eval_multiplier: float
    candidate_notes_assumptions: str

    @property
    def is_embedding_stage(self) -> bool:
        """Signal monitoring / context-assembly stages call an embedding
        model with zero planned output tokens -- there is no JSON-schema
        response to generate for these, only a context index."""
        return self.planned_output_tokens_per_run == 0


def _read_rows(csv_path: Path) -> list[dict]:
    if not csv_path.exists():
        raise TokenMathConfigError(f"Token math plan not found at {csv_path}")
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        missing = [c for c in REQUIRED_COLUMNS if c not in fieldnames]
        if missing:
            raise TokenMathConfigError(
                f"'{csv_path.name}' is missing required column(s): {missing}. "
                f"Found columns: {fieldnames}"
            )
        return list(reader)


def _parse_row(row: dict, row_num: int) -> StagePlan:
    stage_id = (row.get("stage_id") or "").strip()
    if not stage_id:
        raise TokenMathConfigError(f"Row {row_num}: missing stage_id.")
    try:
        return StagePlan(
            stage_id=stage_id,
            workflow_component=row["workflow_component"].strip(),
            operating_area=row["operating_area"].strip(),
            trigger_schedule=row["trigger_schedule"].strip(),
            runs_per_cadence=float(row["runs_per_cadence"]),
            cadence=row["cadence"].strip(),
            annualization_factor=float(row["annualization_factor"]),
            model=row["model"].strip(),
            planned_input_tokens_per_run=int(float(row["planned_input_tokens_per_run"])),
            planned_output_tokens_per_run=int(float(row["planned_output_tokens_per_run"])),
            input_price_per_1m=float(row["input_price_per_1m"]),
            output_price_per_1m=float(row["output_price_per_1m"]),
            retry_rate=float(row["retry_rate_percent"]) / 100.0,
            qa_eval_multiplier=float(row["qa_eval_multiplier"]),
            candidate_notes_assumptions=(row.get("candidate_notes_assumptions") or "").strip(),
        )
    except (KeyError, ValueError) as exc:
        raise TokenMathConfigError(f"Row {row_num} (stage_id={stage_id}): {exc}") from exc


@lru_cache(maxsize=None)
def load_token_math_plan(csv_path: Path = TOKEN_MATH_PLAN_CSV) -> Dict[str, StagePlan]:
    """Load, validate, and parse config/token_math_plan.csv into a dict of
    stage_id -> StagePlan. Cached (the file is read once per process)."""
    rows = _read_rows(csv_path)
    stages: Dict[str, StagePlan] = {}
    for i, row in enumerate(rows, start=2):  # header occupies line 1
        plan = _parse_row(row, i)
        if plan.stage_id in stages:
            raise TokenMathConfigError(
                f"Row {i}: duplicate stage_id '{plan.stage_id}' -- stage_id must be unique."
            )
        stages[plan.stage_id] = plan
    return stages


def get_stage(stage_id: str, plan: Dict[str, StagePlan] = None) -> StagePlan:
    """Look up a single stage by stage_id, raising a clear error if unknown."""
    plan = plan if plan is not None else load_token_math_plan()
    if stage_id not in plan:
        raise TokenMathConfigError(
            f"Unknown stage_id '{stage_id}'. Not present in {TOKEN_MATH_PLAN_CSV.name}."
        )
    return plan[stage_id]
