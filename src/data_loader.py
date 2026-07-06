"""
Data loading and validation for the Customer Success AI Workflow.

Responsibilities:
  1. Load each CSV and validate that all required columns are present.
  2. Join account-level context across all account-scoped tables
     (accounts, usage_events, support_tickets, call_notes, scheduled_checkins)
     on `account_id`.
  3. Split the semicolon-delimited `quality_standard_ids` column in
     junior_outputs.csv into individual rows, then join against
     quality_standards.csv on `standard_id` to attach the full standard
     name/description to each output being reviewed.

Nothing here calls an external service -- it is pure pandas/stdlib.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import pandas as pd

from src import config


class DataValidationError(Exception):
    """Raised when a required input file is missing or malformed."""


# ---------------------------------------------------------------------------
# Low-level load + validate
# ---------------------------------------------------------------------------

def load_csv(path: Path) -> pd.DataFrame:
    """Load a single CSV and validate it against config.REQUIRED_COLUMNS."""
    filename = path.name

    if not path.exists():
        raise DataValidationError(
            f"Missing required data file: '{filename}' was not found at {path}."
        )

    try:
        df = pd.read_csv(path)
    except Exception as exc:  # noqa: BLE001 - surface any parse error clearly
        raise DataValidationError(f"Could not parse '{filename}': {exc}") from exc

    required = config.REQUIRED_COLUMNS.get(filename, [])
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise DataValidationError(
            f"'{filename}' is missing required column(s): {missing}. "
            f"Found columns: {list(df.columns)}"
        )

    # Normalize account_id / standard_id whitespace where present, so joins
    # don't silently fail on stray spaces from manual CSV edits.
    for id_col in ("account_id", "standard_id", "output_id", "ticket_id", "checkin_id"):
        if id_col in df.columns:
            df[id_col] = df[id_col].astype(str).str.strip()

    return df


@dataclass
class RawTables:
    """Container for the seven raw, validated DataFrames."""
    accounts: pd.DataFrame
    usage_events: pd.DataFrame
    support_tickets: pd.DataFrame
    call_notes: pd.DataFrame
    scheduled_checkins: pd.DataFrame
    junior_outputs: pd.DataFrame
    quality_standards: pd.DataFrame

    def row_counts(self) -> Dict[str, int]:
        return {
            "accounts": len(self.accounts),
            "usage_events": len(self.usage_events),
            "support_tickets": len(self.support_tickets),
            "call_notes": len(self.call_notes),
            "scheduled_checkins": len(self.scheduled_checkins),
            "junior_outputs": len(self.junior_outputs),
            "quality_standards": len(self.quality_standards),
        }


def load_all_tables() -> RawTables:
    """Load and validate all seven source CSVs. Raises DataValidationError
    (with a message identifying every problem found) if anything is wrong."""
    errors: List[str] = []
    loaded = {}

    file_map = {
        "accounts": config.ACCOUNTS_CSV,
        "usage_events": config.USAGE_EVENTS_CSV,
        "support_tickets": config.SUPPORT_TICKETS_CSV,
        "call_notes": config.CALL_NOTES_CSV,
        "scheduled_checkins": config.SCHEDULED_CHECKINS_CSV,
        "junior_outputs": config.JUNIOR_OUTPUTS_CSV,
        "quality_standards": config.QUALITY_STANDARDS_CSV,
    }

    for key, path in file_map.items():
        try:
            loaded[key] = load_csv(path)
        except DataValidationError as exc:
            errors.append(str(exc))

    if errors:
        raise DataValidationError(
            "Data validation failed with the following issue(s):\n  - "
            + "\n  - ".join(errors)
        )

    # Referential integrity check: warn (not fail) on account_ids that show
    # up in child tables but not in accounts.csv, since that's recoverable
    # (we just can't build a full briefing for that account).
    known_accounts = set(loaded["accounts"]["account_id"])
    orphan_report = []
    for key in ("usage_events", "support_tickets", "call_notes", "scheduled_checkins", "junior_outputs"):
        orphans = set(loaded[key]["account_id"]) - known_accounts
        if orphans:
            orphan_report.append(f"{key} references unknown account_id(s): {sorted(orphans)}")
    if orphan_report:
        print(
            "WARNING: referential integrity issues found (continuing anyway):\n  - "
            + "\n  - ".join(orphan_report),
            file=sys.stderr,
        )

    return RawTables(**loaded)


# ---------------------------------------------------------------------------
# Account-level join
# ---------------------------------------------------------------------------

@dataclass
class AccountContext:
    """Everything known about a single account, consolidated for prompting."""
    account_id: str
    account: dict
    usage_events: List[dict] = field(default_factory=list)
    open_tickets: List[dict] = field(default_factory=list)
    all_tickets: List[dict] = field(default_factory=list)
    call_notes: List[dict] = field(default_factory=list)
    upcoming_checkins: List[dict] = field(default_factory=list)


def build_account_contexts(tables: RawTables) -> Dict[str, AccountContext]:
    """Join accounts.csv with all child tables on account_id, producing one
    AccountContext per account. This is the 'account_id join' step."""
    contexts: Dict[str, AccountContext] = {}

    for _, row in tables.accounts.iterrows():
        acc_id = row["account_id"]
        contexts[acc_id] = AccountContext(account_id=acc_id, account=row.to_dict())

    def _attach(df: pd.DataFrame, target_attr: str, filter_fn=None):
        for _, row in df.iterrows():
            acc_id = row["account_id"]
            if acc_id not in contexts:
                continue  # orphan already warned about above
            record = row.to_dict()
            if filter_fn is None or filter_fn(record):
                getattr(contexts[acc_id], target_attr).append(record)

    _attach(tables.usage_events, "usage_events")
    _attach(tables.call_notes, "call_notes")
    _attach(tables.scheduled_checkins, "upcoming_checkins")
    _attach(tables.support_tickets, "all_tickets")
    _attach(
        tables.support_tickets,
        "open_tickets",
        filter_fn=lambda r: str(r.get("current_status", "")).lower() in ("open", "new"),
    )

    # Sort usage_events by date so trend deltas are chronological.
    for ctx in contexts.values():
        ctx.usage_events.sort(key=lambda r: r.get("event_date", ""))

    return contexts


# ---------------------------------------------------------------------------
# quality_standard_ids split + join
# ---------------------------------------------------------------------------

def explode_quality_standards(tables: RawTables) -> pd.DataFrame:
    """Split the semicolon-delimited `quality_standard_ids` column in
    junior_outputs.csv into one row per (output_id, standard_id), then join
    against quality_standards.csv to attach standard_name/description.

    Returns a long-format DataFrame with columns:
        output_id, account_id, output_type, draft_text,
        intended_customer_action, standard_id, standard_name, description
    """
    outputs = tables.junior_outputs.copy()
    outputs["quality_standard_ids"] = (
        outputs["quality_standard_ids"].astype(str).str.split(";")
    )
    exploded = outputs.explode("quality_standard_ids")
    exploded["standard_id"] = exploded["quality_standard_ids"].str.strip()
    exploded = exploded.drop(columns=["quality_standard_ids"])

    merged = exploded.merge(
        tables.quality_standards, on="standard_id", how="left", validate="many_to_one"
    )

    unknown = merged[merged["standard_name"].isna()]
    if not unknown.empty:
        bad_ids = sorted(unknown["standard_id"].unique())
        raise DataValidationError(
            f"junior_outputs.csv references unknown quality_standard_id(s): {bad_ids}. "
            f"These do not exist in quality_standards.csv."
        )

    return merged.reset_index(drop=True)
