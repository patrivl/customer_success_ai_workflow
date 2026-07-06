"""
Central configuration for the Customer Success AI Workflow.

Keeping every tunable constant in one place makes the deterministic logic
in briefing_generator.py and quality_reviewer.py easy to audit and adjust
without hunting through the codebase.
"""

from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# outputs/ is the ONE final output directory for this project. Every report
# that matters for the final deliverable (token/cost measurement,
# representative end-to-end runs, quality/routing/intervention rollups, and
# the top-level workflow summary) is written directly under here.
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# Deterministic preprocessing layer's own artifacts (account contexts,
# scores, portfolio patterns, selectors) -- a subfolder of outputs/, not a
# separate top-level directory.
PREPROCESSING_OUTPUT_DIR = OUTPUT_DIR / "preprocessing"

# One JSON file per representative end-to-end account run.
REPRESENTATIVE_RUNS_DIR = OUTPUT_DIR / "representative_runs"

# The original Stage 1/2 (briefing_generator.py / quality_reviewer.py)
# reports are superseded by the token-math-plan-driven layer's per-stage
# outputs, but are still handy for local debugging -- kept out of the way
# in a legacy subfolder instead of a separate top-level output/ directory.
LEGACY_STAGE_REPORTS_DIR = OUTPUT_DIR / "legacy_stage_reports"

# Reference "today" used for renewal-window / ageing calculations across the
# whole workflow (run_workflow.py and src/preprocessing.py both use this, so
# results stay reproducible regardless of when the script is actually run).
# The dataset's dates cluster around late April / May 2026.
REFERENCE_DATE = datetime(2026, 5, 1)

ACCOUNTS_CSV = DATA_DIR / "accounts.csv"
USAGE_EVENTS_CSV = DATA_DIR / "usage_events.csv"
SUPPORT_TICKETS_CSV = DATA_DIR / "support_tickets.csv"
CALL_NOTES_CSV = DATA_DIR / "call_notes.csv"
SCHEDULED_CHECKINS_CSV = DATA_DIR / "scheduled_checkins.csv"
JUNIOR_OUTPUTS_CSV = DATA_DIR / "junior_outputs.csv"
QUALITY_STANDARDS_CSV = DATA_DIR / "quality_standards.csv"

# ---------------------------------------------------------------------------
# Required columns per input file.
# The loader fails fast (with a clear message) if any of these are missing.
# ---------------------------------------------------------------------------
REQUIRED_COLUMNS = {
    "accounts.csv": [
        "account_id", "account_name", "segment", "contract_value",
        "renewal_date", "csm_owner", "current_health_score",
        "previous_health_score", "product_usage_trend",
        "support_ticket_count_30d", "nps_score", "expansion_signal",
        "last_contact_date", "notes",
    ],
    "usage_events.csv": [
        "account_id", "event_date", "active_users", "key_feature_users",
        "login_frequency", "usage_trend", "notable_change",
    ],
    "support_tickets.csv": [
        "ticket_id", "account_id", "date_received", "issue_summary",
        "severity", "customer_sentiment", "frontline_notes", "current_status",
    ],
    "call_notes.csv": [
        "account_id", "call_date", "participants", "summary",
        "customer_goal", "risk_or_blocker", "follow_up_items",
    ],
    "scheduled_checkins.csv": [
        "checkin_id", "account_id", "scheduled_date", "checkin_type",
        "priority", "topics_to_cover",
    ],
    "junior_outputs.csv": [
        "output_id", "account_id", "output_type", "draft_text",
        "intended_customer_action", "quality_standard_ids",
    ],
    "quality_standards.csv": [
        "standard_id", "standard_name", "description",
    ],
}

# ---------------------------------------------------------------------------
# Simulated LLM pricing.
# No real API is ever called -- these rates only exist so the workflow can
# demonstrate realistic token/cost logging. Numbers are illustrative and
# loosely modeled on public per-token pricing tiers; they are NOT a quote
# for any real Anthropic product.
# ---------------------------------------------------------------------------
SIMULATED_MODEL_NAME = "simulated-cs-assistant-v1"
SIMULATED_INPUT_RATE_PER_1K_TOKENS = 0.003   # USD per 1K prompt tokens
SIMULATED_OUTPUT_RATE_PER_1K_TOKENS = 0.015  # USD per 1K completion tokens

# Rough words-to-tokens heuristic (English prose is ~0.75 words/token, i.e.
# ~1.33 tokens/word). This is only used for the simulated cost log.
TOKENS_PER_WORD = 1.33

# ---------------------------------------------------------------------------
# Risk-scoring weights (all deterministic, documented here so the scoring
# logic in briefing_generator.py is transparent and tunable in one spot).
# ---------------------------------------------------------------------------
HEALTH_SCORE_WEIGHT = 1.0          # per point below 100
HEALTH_DECLINE_WEIGHT = 2.0        # per point of quarter-over-quarter decline
TICKET_COUNT_WEIGHT = 4.0          # per open/new ticket in the last 30 days
NPS_GAP_WEIGHT = 2.0               # per point NPS is below 10
SEVERITY_WEIGHTS = {"high": 15, "medium": 7, "low": 2}
SENTIMENT_WEIGHTS = {
    "frustrated": 10, "negative": 10, "concerned": 5,
    "neutral": 0, "positive": -5,
}
EXPANSION_RISK_OFFSET = {"high": -10, "medium": -5, "low": 0}

RENEWAL_URGENT_WINDOW_DAYS = 45   # renewals inside this window get flagged
HIGH_RISK_THRESHOLD = 60          # risk_score >= this -> "high" priority tier
MEDIUM_RISK_THRESHOLD = 30        # risk_score >= this -> "medium" priority tier

# ---------------------------------------------------------------------------
# Quality review scoring
# ---------------------------------------------------------------------------
VERDICT_SCORES = {"PASS": 1.0, "PARTIAL": 0.5, "FAIL": 0.0}
PASS_THRESHOLD = 80    # overall_score >= this -> "Approved"
PARTIAL_THRESHOLD = 50  # overall_score >= this -> "Needs revision"
                         # below PARTIAL_THRESHOLD -> "Rejected"

STOPWORDS = {
    "the", "and", "for", "that", "this", "with", "from", "will", "have",
    "has", "are", "was", "were", "been", "your", "team", "please", "about",
    "into", "onto", "over", "than", "then", "them", "they", "their", "our",
    "you", "can", "not", "but", "all", "any", "who", "what", "when", "how",
    "why", "her", "his", "its", "it's", "sometime", "soon", "much", "very",
}

# ---------------------------------------------------------------------------
# Deterministic preprocessing layer (src/preprocessing.py).
# This layer runs BEFORE any simulated-LLM stage. It normalizes data, builds
# per-account context, computes flags/scores, and selects which items move
# into the (simulated) model-facing stages -- the point being cost control
# and reproducibility on a real system, demonstrated here deterministically.
# ---------------------------------------------------------------------------
SEVERE_HEALTH_DECLINE_THRESHOLD = -15   # health_score_delta <= this -> severe decline
MODERATE_HEALTH_DECLINE_THRESHOLD = -5  # health_score_delta <= this -> any decline flag territory
HIGH_TICKET_VOLUME_THRESHOLD = 5        # support_ticket_count_30d >= this -> high volume
LOW_NPS_THRESHOLD = 6                   # nps_score <= this -> detractor / low NPS
HIGH_VALUE_PERCENTILE = 0.75            # top quartile of contract_value -> "high-value" account
ESCALATION_ALERT_THRESHOLD = 40         # escalation_score >= this -> CSM alert
OPPORTUNITY_ALERT_THRESHOLD = 25        # opportunity_score >= this -> second-pass candidate

SECOND_PASS_MAX_ACCOUNTS = 200          # cap for select_second_pass_validation_accounts
FLAGGED_SUMMARY_MIN = 25                # target lower bound for select_flagged_account_summary_accounts
FLAGGED_SUMMARY_MAX = 40                # target upper bound for select_flagged_account_summary_accounts
MAX_ESCALATIONS_PER_DAY = 3             # daily token-budget cap for complex escalation candidates
MIN_REPRESENTATIVE_RUNS = 5             # minimum end-to-end representative runs to select
MAX_REPRESENTATIVE_RUNS = 8             # cap -- broaden case-type coverage without an unbounded list

# Preferred representative accounts (used if present in the dataset), each
# covering a distinct case type for a representative end-to-end demo.
PREFERRED_REPRESENTATIVE_ACCOUNT_IDS = ["A008", "A005", "A014", "A003", "A017"]
REPRESENTATIVE_CASE_TYPES = {
    "A008": "severe_support_escalation_case",
    "A005": "renewal_value_review_case",
    "A014": "negative_sentiment_declining_adoption_case",
    "A003": "healthy_expansion_opportunity_case",
    "A017": "low_usage_reactivation_intervention_case",
}

# Beyond the 5 preferred accounts above, these additional case types are
# filled in (up to MAX_REPRESENTATIVE_RUNS) from the deterministic
# preprocessing selectors' own populations -- never hardcoded account ids.
# Maps case_type -> the `selected_workflow_items` selector key to draw from.
ADDITIONAL_REPRESENTATIVE_CASE_TYPES = {
    "quality_review_case": "failed_or_weak_outputs",
    "intervention_planning_case": "intervention_candidates",
    "complex_escalation_case": "complex_escalation_candidates",
}

# Deterministic keyword themes used for support-ticket / blocker pattern
# detection (no AI involved -- simple substring matching).
ISSUE_THEME_KEYWORDS = {
    "sso_authentication": ["sso", "login", "authentication", "admin users"],
    "integration_sync": ["integration", "sync", "api"],
    "onboarding_mobile": ["onboarding", "mobile"],
    "reporting_dashboard": ["dashboard", "reporting", "export", "exports"],
    "permissions_access": ["permission", "permissions", "access"],
    "automation_features": ["automation", "premium", "feature"],
    "roadmap_strategic": ["roadmap", "strategic"],
    "usage_decline_reset": ["reset", "stopped", "churn"],
}

EXEC_RENEWAL_KEYWORDS = ["executive", "exec", "sponsor", "renewal", "churn"]
TECHNICAL_BLOCKER_KEYWORDS = ["sso", "integration", "technical", "bug", "block", "api", "sync"]
MANAGER_ESCALATION_KEYWORDS = ["escalate", "escalation", "manager", "leadership", "product team"]

# Cheap, deterministic pre-check used only to feed escalation_score / the
# select_failed_or_weak_outputs selector. This is NOT the authoritative
# quality determination -- that happens in the fuller Stage 2 simulated-LLM
# review in src/quality_reviewer.py.
WEAK_OUTPUT_MIN_WORD_COUNT = 12
WEAK_OUTPUT_VAGUE_MARKERS = [
    "let us know", "sometime", "try again", "we think", "will update you soon",
]
