# Customer Success AI Workflow (Simulated)

A runnable, deterministic simulation of an end-to-end Customer Success AI
workflow. It ingests the seven synthetic CSVs in `data/`, joins them,
generates a risk-ranked briefing for every account, runs a quality review
of junior-drafted customer communications against a defined set of quality
standards, and then executes all 37 planned workflow stages defined in
`config/token_math_plan.csv` (model routing, prompt templates, token
measurement, and cost logging) against those same accounts/tickets/
check-ins/outputs.

**No external AI API is called anywhere in this project.** Every "LLM
response" is produced by deterministic, rule-based logic that is fed
through a `SimulatedLLMClient` (see `src/llm_simulator.py`), which mimics
the call shape, prompt handling, and token/cost metering of a real LLM
integration. This keeps the workflow fully offline, free, and 100%
reproducible, while still exercising every part of the pipeline a real
integration would need (prompt templates, a call interface, token counting,
cost logging).

## Quick start

```bash
pip install -r requirements.txt
python run_workflow.py
```

That's it — one command, no API keys, no network access required. `outputs/`
is the single final output directory; a run summary is also printed to the
console.

## What it does

1. **Load & validate** — every CSV in `data/` is loaded and checked against
   a required-column schema (`src/config.py::REQUIRED_COLUMNS`). Missing
   files or columns cause a clear, fail-fast error listing every problem
   found (see "Data validation" below).
2. **Deterministic preprocessing layer** (`src/preprocessing.py`, runs
   before any simulated-LLM stage) — normalizes every table, joins
   everything on `account_id`, splits/joins `quality_standard_ids`, builds
   a compact per-account context, computes transparent flags and four
   scores (`risk_score`, `opportunity_score`, `priority_score`,
   `escalation_score`), selects which accounts/tickets/outputs move into
   the later stages, detects portfolio-level patterns, and picks 5
   representative end-to-end runs. See "The preprocessing layer" below.
3. **Join on `account_id`** — `accounts.csv` is joined with
   `usage_events.csv`, `support_tickets.csv`, `call_notes.csv`, and
   `scheduled_checkins.csv` into one consolidated `AccountContext` per
   account (this is the Stage 1/2 input shape; separate from, but
   consistent with, the join done inside the preprocessing layer).
4. **Split + join `quality_standard_ids`** — `junior_outputs.csv` has a
   semicolon-delimited `quality_standard_ids` column (e.g.
   `QS001;QS002;QS003`). This is exploded into one row per
   `(output_id, standard_id)` pair, then joined against
   `quality_standards.csv` on `standard_id` to attach the full standard
   name and description to every row that needs reviewing.
5. **Stage 1 — Account briefings** (`src/briefing_generator.py`):
   for every SELECTED account (from the preprocessing layer's
   `daily_account_review` selector), compute a deterministic `risk_score`
   and `opportunity_score`, render the `account_briefing` prompt template
   with real data, and generate a narrative summary + risk/opportunity
   notes + recommended next actions. Accounts are ranked highest-risk-first.
6. **Stage 2 — Quality review** (`src/quality_reviewer.py`): for every
   SELECTED junior output (from the `quality_review_outputs` selector),
   evaluate each assigned quality standard with a dedicated heuristic and
   produce a PASS / PARTIAL / FAIL verdict with a grounded rationale.
   Verdicts roll up into an overall score and a recommendation of
   *Approved*, *Needs revision*, or *Rejected*, plus a suggested revision
   when it isn't a clean approval.
7. **Reports** — Stage 1/2 output is kept for local debugging only, under
   `outputs/legacy_stage_reports/` (JSON + Markdown + a per-call token/cost
   log) — it is **not** part of the final deliverable. The preprocessing
   layer's own output lands in `outputs/preprocessing/` (see below).
8. **Token-math-plan-driven simulated model layer** (`src/model_simulator.py`,
   `src/token_measurement.py`, runs after Stage 1/2) — every one of the 37
   planned workflow stages in `config/token_math_plan.csv` is executed
   against the preprocessing layer's real selected populations (accounts,
   tickets, check-ins, junior outputs, intervention candidates, issue
   themes), producing a full prompt/response/token/cost trace per call,
   measured against that stage's planned model, tokens, and pricing. See
   "The token-math-plan-driven simulated model layer" below.
9. **Final end-to-end orchestration** (`src/final_report.py`) — reuses those
   same traces (no re-simulation) to assemble one JSON file per
   representative account run (`outputs/representative_runs/`),
   portfolio-wide quality/routing/intervention rollup CSVs, and the
   top-level `outputs/workflow_summary.json` / `outputs/workflow_summary.md`.
   See "Final output (outputs/)" below.

## Project structure

```
run_workflow.py            # single entry point — orchestrates everything
requirements.txt
config/
  token_math_plan.csv       # planned workflow stages, model routing, token/price/retry/QA assumptions
data/                       # the 7 source CSVs
  accounts.csv
  usage_events.csv
  support_tickets.csv
  call_notes.csv
  scheduled_checkins.csv
  junior_outputs.csv
  quality_standards.csv
src/
  config.py                 # required columns, scoring weights, thresholds, pricing constants
  data_loader.py             # load, validate, account_id joins, standard_ids split/join (Stage 1/2 shape)
  preprocessing.py            # deterministic preprocessing layer: normalize, flags, scores, selectors, patterns
  llm_simulator.py           # simulated LLM call wrapper + token/cost metering (Stage 1/2)
  prompts.py                 # prompt templates for Stage 1/2 + the 9 token-math-plan templates
  briefing_generator.py       # Stage 1: risk/opportunity scoring + briefings
  quality_reviewer.py         # Stage 2: per-standard heuristic evaluators
  report_writer.py            # writes Stage 1/2 JSON/Markdown reports + logs
  token_math_config.py        # loads/validates config/token_math_plan.csv into StagePlan objects
  output_schemas.py           # expected JSON output schema per prompt template + validation
  token_costs.py              # estimate_tokens / calculate_cost / variance / review_flag math
  model_simulator.py          # STAGE_RUNTIME_MAP + simulate_model_call() (deterministic, no API calls)
  token_measurement.py        # runs all 37 stages against preprocessing populations + writes aggregation CSVs
  final_report.py             # assembles representative runs + rollup CSVs + workflow_summary from those traces
tests/
  test_workflow.py            # sanity tests, incl. a full `python3 run_workflow.py` acceptance test
outputs/                      # THE single final output directory (everything below lives here)
  workflow_summary.json       # top-level machine-readable run summary
  workflow_summary.md         # top-level human-readable run summary
  stage_token_counts.csv      # one row per simulated call: planned vs. measured tokens
  cost_summary.csv            # one row per simulated call: planned/measured/adjusted cost + variance
  token_math_measurement_summary.csv  # one row per stage_id (all 37), spreadsheet-ready
  token_math_spreadsheet_export.csv   # 5 copy-back columns + optional 'exercised' flag
  quality_review_results.csv  # every quality_review_prompt call, portfolio-wide
  routing_decisions.csv       # every routing_prompt / complex_escalation_prompt call, portfolio-wide
  intervention_plans.csv      # every intervention_planning_prompt call, portfolio-wide
  representative_runs/        # one JSON file per representative end-to-end account run
    A008.json, A005.json, ...
  preprocessing/               # deterministic preprocessing layer's own artifacts
    account_contexts.json, account_scores.csv, portfolio_patterns.json,
    selected_workflow_items.json, representative_accounts.json
  legacy_stage_reports/        # OLD Stage 1/2 reports -- debug only, not the final deliverable
    account_briefings.json / .md, quality_review_report.json / .md,
    token_usage_log.csv, workflow_summary.md
```

## The preprocessing layer (`src/preprocessing.py`)

This layer does everything that doesn't require judgement or synthesis, so
the (simulated) model-facing stages only see clean, compact, decision-ready
context -- and, on a real/larger portfolio, only see the accounts/outputs
that actually warrant a model call.

**Normalization** (`validate_and_normalize_data`): re-confirms required
columns, lowercases/strips categorical fields (`severity`, `sentiment`,
`status`, `priority`, `usage_trend`) with safe defaults for missing values,
parses `renewal_date` / `call_date` / `event_date` / `scheduled_date` into
real datetimes, and coerces key numeric fields. This produces a *separate*
normalized copy — it never mutates the tables Stage 1/2 use, so existing
report text/casing is unaffected.

**Account context** (`build_account_contexts`): one compact,
JSON-serializable dict per account — identity fields, health/usage/ticket/
NPS/expansion signals, `renewal_days_remaining`, recent usage events, open
tickets, prior call notes, upcoming check-ins, related junior outputs, and
any unresolved follow-up item.

**Deterministic flags** (12, per account): `health_decline_flag`,
`severe_decline_flag`, `renewal_soon_flag`, `high_ticket_volume_flag`,
`negative_sentiment_flag`, `low_nps_flag`, `declining_usage_flag`,
`expansion_opportunity_flag`, `unresolved_issue_flag`, `checkin_due_flag`,
`quality_review_needed_flag`, `intervention_candidate_flag` (plus a bonus
`high_value_account_flag`).

**Scores** (all transparent, weights documented in `config.py`):
- `risk_score` — health decline, low current health, declining usage,
  ticket volume/severity, negative sentiment, low NPS, renewal proximity,
  unresolved blockers.
- `opportunity_score` — growing usage, expansion signal, positive
  sentiment, upcoming check-in, high account value, customer-goal language
  aligned with growth/expansion.
- `priority_score` — combines risk + opportunity with renewal proximity,
  account value, ticket severity, check-in priority, and unresolved-item
  ageing (days since last contact).
- `escalation_score` — high-severity tickets, negative/frustrated
  sentiment, executive/renewal risk language, technical-blocker language,
  manager/escalation language, and a failed preliminary output quality
  pre-check.

**Selectors** (each returns IDs + a short deterministic reason, e.g.
*"health score declined by 18 points"* or *"renewal within 45 days and
negative sentiment"*): `select_daily_account_review_accounts` (all),
`select_second_pass_validation_accounts`, `select_flagged_account_summary_accounts`
(top 25-40, gracefully scaled down on this small dataset),
`select_csm_alert_accounts`, `select_unresolved_items`,
`select_inbound_issues`, `select_issue_pattern_review_items`,
`select_scheduled_checkins`, `select_quality_review_outputs`,
`select_failed_or_weak_outputs`, `select_intervention_candidates`,
`select_complex_escalation_candidates` (capped at 3/day, simulating a daily
token budget for the most expensive escalation-reasoning calls).

**Portfolio patterns** (`build_portfolio_patterns`): accounts/avg health by
segment, declining-account counts by segment, common support-ticket issue
themes and call-note blocker themes (deterministic keyword clustering),
and intervention/expansion segment breakdowns.

**Representative runs** (`select_representative_runs`): returns >= 5
accounts covering distinct cases (severe escalation, renewal/value review,
negative-sentiment/declining adoption, healthy expansion, low-usage
reactivation). On this dataset the preferred IDs (`A008, A005, A014, A003,
A017`) are all present and used directly; if any were missing, the
function falls back to the closest match by `priority_score`.

`run_workflow.py` calls `preprocessing.run_preprocessing()` right after
loading the raw CSVs (Step 2 of 6), writes its five output files, and then
uses the `daily_account_review` / `quality_review_outputs` selectors to
decide which accounts/outputs actually reach Stage 1/2. On this small
synthetic dataset those selectors return everything (nothing to filter out
yet) — but that's exactly the wiring that makes the later stages
cost-controlled on a real, larger portfolio.



## Data validation

`src/data_loader.load_csv()` checks every input file against the required
columns defined in `src/config.py::REQUIRED_COLUMNS`. If a file is missing
or a required column isn't present, the workflow stops before doing any
processing and prints every problem it found, e.g.:

```
FATAL: Data validation failed with the following issue(s):
  - 'accounts.csv' is missing required column(s): ['nps_score']. Found columns: [...]
```

The loader also runs a referential-integrity check: any `account_id` that
appears in a child table (tickets, usage, calls, check-ins, outputs) but
not in `accounts.csv` is reported as a warning (not a hard failure), since
the rest of the workflow can still run for every account that *does*
resolve.

## The `quality_standard_ids` split/join, concretely

`junior_outputs.csv` stores multiple standard IDs per row:

| output_id | quality_standard_ids          |
|-----------|--------------------------------|
| O001      | `QS001;QS002;QS003;QS005`     |

`data_loader.explode_quality_standards()` turns this into one row per
standard, then merges in the standard's name/description:

| output_id | standard_id | standard_name              | description                 |
|-----------|-------------|-----------------------------|------------------------------|
| O001      | QS001       | Customer-specific context   | Output references ...        |
| O001      | QS002       | Actionability                | Output gives concrete ...    |
| O001      | QS003       | Risk accuracy                 | Output identifies material...|
| O001      | QS005       | Escalation judgment            | Output escalates urgent ...  |

This long-format table is what `quality_reviewer.review_all_outputs()`
iterates over (grouped back by `output_id`) to produce one verdict per
standard and one overall recommendation per output.

## Simulated LLM calls & cost logging (legacy Stage 1/2)

Every "LLM call" (one per account for briefings, one per junior output for
quality review) goes through `SimulatedLLMClient.call()`:

- The full prompt is rendered from `src/prompts.py` templates with real
  joined data (nothing is hard-coded).
- A deterministic Python function produces the "response" text (the
  narrative/verdicts described above).
- Both prompt and response are token-counted (`~1.33 tokens/word`
  heuristic) and priced against illustrative per-1K-token rates in
  `config.py`, then logged to `outputs/legacy_stage_reports/token_usage_log.csv`
  with a timestamp, task name, reference id, and estimated cost. This is the
  older, simpler cost log kept for local debugging -- the final deliverable's
  token/cost measurement is `outputs/stage_token_counts.csv` /
  `outputs/cost_summary.csv` (see below).

This means the pipeline's *shape* — prompt in, metered response out — is a
faithful stand-in for a real LLM integration; swapping `response_fn` for
an actual API call in `llm_simulator.py` is the only change needed to make
this a live system.

## Extending this

- **New/changed data**: as long as the required columns in
  `src/config.py::REQUIRED_COLUMNS` are present, the workflow will run on
  any number of accounts/tickets/outputs — nothing is hard-coded to the
  specific rows in the provided sample data.
- **New quality standards**: add a row to `quality_standards.csv` and a
  matching evaluator function to `src/quality_reviewer.py::STANDARD_EVALUATORS`.
- **Real LLM integration**: replace the `response_fn` callbacks in
  `briefing_generator.py` / `quality_reviewer.py` with an actual API call,
  keeping the same prompt templates and `SimulatedLLMClient` interface (or
  swap in a real client with the same `.call()` signature).

## The token-math-plan-driven simulated model layer

`config/token_math_plan.csv` is the single source of truth for 37 planned
workflow stages (`stage_id` TM_001-TM_037): which model each stage would
call, planned input/output tokens per run, per-1M-token pricing,
retry/QA-eval overhead, and cadence/annualization assumptions.
`stage_id` is the unique key -- `workflow_component` + `operating_area` is
**not** unique by design (e.g. TM_004 and TM_005 are both "Account
review" / "Synthesis & recommendation" but cover different trigger
schedules: second-pass validation vs. flagged-account summaries).

Note on column naming: the CSV's retry column is `retry_rate_percent`
(e.g. `5` for 5%), not `retry_rate`. `src/token_math_config.py` validates
against the actual header and exposes it as `StagePlan.retry_rate`, a 0-1
fraction, for direct use in cost math.

**How a stage becomes a simulated call** (`src/model_simulator.py`,
`src/token_measurement.py`):

1. `STAGE_RUNTIME_MAP` wires each `stage_id` to the deterministic
   preprocessing layer's population that feeds it (e.g. TM_002 pulls from
   `daily_account_review`, TM_012 from `inbound_issues`, TM_037 from
   `complex_escalation_candidates`), the *kind* of item in that population
   (account / ticket / checkin / output / issue_theme), and which of the 9
   prompt templates renders it.
2. For each item, `src/token_measurement.py` builds a compact, grounded
   context (the account's own preprocessing context for account-kind
   stages; a joined ticket/check-in/output record plus its owning
   account's scores/flags otherwise).
3. `model_simulator.simulate_model_call(stage_id, context, prompt_template_name, run_id)`
   looks up the stage's config row, renders the prompt
   (`src/prompts.py`), generates a deterministic structured output grounded
   in that context (`src/output_schemas.py` defines and validates the
   expected shape), measures actual prompt/response token counts
   (`ceil(chars / 4)`, `src/token_costs.py`), and returns a full trace:
   planned vs. measured tokens, planned vs. measured cost, a retry/QA-
   adjusted cost, percent variance, and a `review_flag`.

**The 9 prompt templates** (`src/prompts.py`): `account_review_prompt`,
`prioritization_prompt`, `inbound_issue_prompt`, `checkin_support_prompt`,
`quality_review_prompt`, `intervention_planning_prompt`, `routing_prompt`,
`complex_escalation_prompt`, `deployment_tracking_prompt`. Every one
includes the stage identity, task objective, compact input context, the
data fields it draws on, the expected JSON output schema, a confidence-
score requirement, valid labels/routes, escalation/quality criteria where
relevant, and explicit instructions against generic recommendations and in
favor of citing account evidence in the rationale. Signal-monitoring /
context-assembly stages (embedding models, zero planned output tokens) use
a lightweight `context_indexing` pseudo-template instead of a JSON verdict.

**Review-flag bands** (`src/token_costs.assign_review_flag`): within
±20% variance = `OK`; +20% to +50% = `Review: above estimate`; above +50%
= `High variance: revise assumptions`; -20% to -50% =
`Review: overestimated`; below -50% =
`High variance: estimate too conservative`; no measurement yet =
`Pending measurement`.

**Output** (written by `src/token_measurement.py` to `outputs/`):
- `stage_token_counts.csv` -- one row per simulated call: planned vs.
  measured input/output tokens.
- `cost_summary.csv` -- one row per simulated call: planned/measured/
  adjusted cost, retry_rate, qa_eval_multiplier, variance_pct, review_flag.
- `token_math_measurement_summary.csv` -- one row for **every** `stage_id`
  in `config/token_math_plan.csv` (all 37, whether or not that stage had a
  call in this run), formatted for direct copy into the Token Math
  Template's measurement columns. See "Spreadsheet export" below for the
  exact column meanings.
- `token_math_spreadsheet_export.csv` -- the same one-row-per-`stage_id`
  data narrowed to just the 5 spreadsheet copy-back columns (plus an
  optional `exercised` flag). See "Spreadsheet export" below.

### Spreadsheet export: populating the Token Math Template's measurement columns

`token_math_measurement_summary.csv` and `token_math_spreadsheet_export.csv`
are both built by `src/token_measurement.py::_build_measurement_rows()` (one
function, so the two files can never drift apart) and map onto the Token
Math Template's measurement columns as follows:

| Spreadsheet column | CSV column | How it's computed |
|---|---|---|
| Notebook measured avg cost/run | `notebook_measured_avg_cost_per_run` | Average **adjusted** measured cost per call for that `stage_id` (`adjusted_measured_cost` = measured cost × (1 + retry_rate) × qa_eval_multiplier) -- the realistic per-run cost, not just the raw token cost. |
| Estimate vs measured variance | `estimate_vs_measured_variance` | `(notebook_measured_avg_cost_per_run − planned_cost_per_run) / planned_cost_per_run`, formatted as a signed percentage string (e.g. `-59.6%`, `+18.7%`). |
| Source / measurement link | `source_measurement_link` | `outputs/stage_token_counts.csv` for any stage with at least one call in this run (aggregated rows point at the per-call detail file); `No representative call in sample` if the stage had zero calls. |
| Review flag | `review_flag` | The variance bands below, applied to `estimate_vs_measured_variance`'s underlying number; `Not exercised in representative runs` if the stage had zero calls. |

Supporting (non-spreadsheet) columns on `token_math_measurement_summary.csv`
give the unadjusted view for context: `avg_measured_cost_per_run` (no
retry/QA overhead), `avg_adjusted_measured_cost_per_run` (same value as
`notebook_measured_avg_cost_per_run`, named literally), `variance_pct`
(numeric, unadjusted-basis variance), plus `avg_measured_input_tokens`,
`avg_measured_output_tokens`, the stage's planned tokens/cost, and
`source_output_file`.

**Review-flag bands** (`src/token_costs.assign_review_flag`): within
±20% variance = `OK`; +20% to +50% = `Review: above estimate`; above +50%
= `High variance: revise assumptions`; -20% to -50% =
`Review: overestimated`; below -50% =
`High variance: estimate too conservative`; a `stage_id` with zero calls in
this run = `Not exercised in representative runs`.

**`outputs/token_math_spreadsheet_export.csv`** narrows
`token_math_measurement_summary.csv` down to exactly the 5 copy-ready
columns above, plus one optional `exercised` (`Yes`/`No`) flag for whether
this run measured at least one call for that `stage_id` -- one row per
`stage_id`, ready to paste straight back into the spreadsheet. It
intentionally does **not** include any production-scaled annual
projection (no `planned_annual_cost`, no annualized variance): projecting
this small sample up to production scale didn't improve its
interpretation, so the export stays focused on what was actually measured.

Measured costs are based on the runnable synthetic dataset and therefore
reflect short sample prompts, not full production context. The Token Math
Template remains the conservative production budget. The measured columns
verify that the workflow logs tokens, costs, and variance correctly; they
are not intended to replace the production-scale budget estimate.

## Final output (`outputs/`) and `src/final_report.py`

`outputs/` is the single final output directory. `src/final_report.py` runs
last (Step 8/8 of `run_workflow.py`) and assembles the final deliverable
from the token-math layer's own call traces -- it does **not** re-run or
duplicate any simulated model call.

**Representative account selection** (`src/preprocessing.select_representative_runs`):
picks between `MIN_REPRESENTATIVE_RUNS` (5) and `MAX_REPRESENTATIVE_RUNS`
(8) accounts. It starts from the 5 preferred accounts in
`config.PREFERRED_REPRESENTATIVE_ACCOUNT_IDS` (each a distinct case type --
severe support escalation, renewal/value review, negative sentiment/
declining adoption, healthy expansion, low usage reactivation) when present,
then -- **never by hardcoding additional account ids** -- fills in up to 3
more case types (quality review, intervention planning, complex escalation)
from `config.ADDITIONAL_REPRESENTATIVE_CASE_TYPES`, each drawn from the
matching deterministic preprocessing selector (`failed_or_weak_outputs`,
`intervention_candidates`, `complex_escalation_candidates`). Any
substitution or additional pick carries an explicit `reason` string, and
that reasoning is echoed into both `workflow_summary.json` and
`workflow_summary.md`.

**Representative run JSON** (`outputs/representative_runs/<account_id>.json`):
one file per representative account, grouping every simulated-stage trace
that account went through (across all 37 stage_ids) into
`account_review_outputs`, `prioritization_outputs`, `inbound_issue_outputs`,
`checkin_outputs`, `quality_review_outputs`, `intervention_outputs`, and
`routing_outputs` (routing_prompt + complex_escalation_prompt calls share
this bucket), plus `evaluation_flags`, a `final_route` (the account's last
routing/escalation decision), token/cost totals, and the full
`stage_traces` list for auditability.

**Portfolio-wide rollup CSVs** (filtered straight from the same trace list,
one row per matching call): `quality_review_results.csv`,
`routing_decisions.csv`, `intervention_plans.csv`.

**Top-level summary**: `workflow_summary.json` (machine-readable) and
`workflow_summary.md` (human-readable) cover the dataset used, the
representative runs and any substitutions, workflow components/operating
areas covered, portfolio-wide token/cost totals and per-run averages,
quality/escalation/intervention counts, and each representative account's
final route.

## Tests

```bash
python -m pytest tests/ -v
# or, without pytest installed:
python tests/test_workflow.py
```
