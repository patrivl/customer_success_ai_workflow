# Customer Success AI Workflow — Final Run Summary

_Generated: 2026-07-07T06:22:30+00:00_

**No external AI APIs are called anywhere in this workflow.** Every stage below is executed by the deterministic simulated model layer (`src/model_simulator.py`), grounded in real joined account/ticket/check-in/output data.

## Dataset used
- `data/accounts.csv`
- `data/usage_events.csv`
- `data/support_tickets.csv`
- `data/call_notes.csv`
- `data/scheduled_checkins.csv`
- `data/junior_outputs.csv`
- `data/quality_standards.csv`

## Representative end-to-end runs
Selected 8 representative account run(s), covering 8 distinct case(s):

- **Harbor Insurance** (`A008`) — *severe_support_escalation_case* -- preferred representative account for case type 'severe_support_escalation_case'
- **Evergreen Finance** (`A005`) — *renewal_value_review_case* -- preferred representative account for case type 'renewal_value_review_case'
- **Northstar Travel** (`A014`) — *negative_sentiment_declining_adoption_case* -- preferred representative account for case type 'negative_sentiment_declining_adoption_case'
- **Cobalt Health** (`A003`) — *healthy_expansion_opportunity_case* -- preferred representative account for case type 'healthy_expansion_opportunity_case'
- **Quartz Media** (`A017`) — *low_usage_reactivation_intervention_case* -- preferred representative account for case type 'low_usage_reactivation_intervention_case'
- **Lumen Energy** (`A012`) — *quality_review_case* -- deterministic selector 'failed_or_weak_outputs' match for case type 'quality_review_case' (no preferred id available/uncovered): draft is short and/or uses vague filler phrasing (preliminary check)
- **Acme Retail** (`A001`) — *intervention_planning_case* -- deterministic selector 'intervention_candidates' match for case type 'intervention_planning_case' (no preferred id available/uncovered): usage is declining; NPS is low (6); negative/concerned sentiment on an open ticket
- **BrightPath Logistics** (`A002`) — *complex_escalation_case* -- deterministic selector 'complex_escalation_candidates' match for case type 'complex_escalation_case' (no preferred id available/uncovered): escalation_score=50.0 (top 3 under daily cap)

One JSON file per representative run is written to `outputs/representative_runs/<account_id>.json`.

## Workflow components & operating areas covered
Components: Account review, Customer check-in support, Inbound issue handling, Prioritization, Quality review, Routing for resolution, follow-up, or escalation, Targeted intervention planning

Operating areas: Customer check-in & communication support, Deployment tracking & outcome measurement, Intake, classification & triage, Intervention design, Prioritization & queue ranking, Quality evaluation & validation, Routing, handoff & escalation, Signal monitoring, context assembly & portfolio pattern detection, Synthesis & recommendation

## Token / cost measurement
- Total simulated model calls: 378
- Total measured tokens: 168725 input + 45641 output
- Total measured cost: $0.701310
- Total adjusted measured cost (incl. retry/QA overhead): $1.105500
- Total planned annual cost (all 37 stages, using each stage's Adjusted cost/run x Runs per cadence x Annualization factor): $12,517.51 -- should match the Token Math Template's own budget total (~$12,517.52), allowing for minor rounding.
- Average per representative end-to-end run: $0.055217 (13530.1 input tokens + 3619.1 output tokens)

Detailed per-call and per-stage figures: `outputs/stage_token_counts.csv`, `outputs/cost_summary.csv`, `outputs/token_math_measurement_summary.csv`.

## Measurement methodology & Token Math Template export
`outputs/token_math_measurement_summary.csv` has one row for every one of the 37 `stage_id`s in `config/token_math_plan.csv` -- including any that had zero calls in a given run (marked `Not measured`). Its `planned_cost_per_run` column is each stage's **Adjusted cost/run** (base cost x (1 + retry_rate) x qa_eval_multiplier) -- the plan's actual final per-run estimate, matching the Token Math Template, not the unadjusted base token cost. Its `notebook_measured_avg_cost_per_run` column is the average cost per call **including** each stage's planned retry rate and QA/eval multiplier (the realistic per-run cost), and `estimate_vs_measured_variance` is the percent difference between that figure and `planned_cost_per_run` (adjusted vs. adjusted, an apples-to-apples comparison), using the same bands as `review_flag` (±20% = Measured cost on par with original estimate, +20% to +50% = Measured cost above original estimate, more than +50% = Measured cost materially above original estimate, -20% to -50% = Measured cost below original estimate, less than -50% = Measured cost materially below original estimate, no measurement = Not measured). `outputs/token_math_spreadsheet_export.csv` is the same data narrowed to exactly the Token Math Template's measurement columns (`stage_id`, `notebook_measured_avg_cost_per_run`, `estimate_vs_measured_variance`, `source_measurement_link`, `review_flag`, plus `exercised`) -- one row per stage, ready to paste back into the spreadsheet.

Measured costs are based on the runnable synthetic dataset and therefore reflect short sample prompts, not full production context. The Token Math Template remains the conservative production budget. The measured columns verify that the workflow logs tokens, costs, and variance correctly; they are not intended to replace the production-scale budget estimate.

## Quality review summary
- Quality flags (failed review) across the portfolio: 8
- Full detail: `outputs/quality_review_results.csv`

## Routing / escalation summary
- Escalation-routed cases across the portfolio: 22
- Final route per representative account:
  - A008: manager_escalation
  - A005: manager_escalation
  - A014: csm_review
  - A003: schedule_follow_up
  - A017: specialist_escalation
  - A012: csm_review
  - A001: csm_review
  - A002: specialist_escalation
- Full detail: `outputs/routing_decisions.csv`

## Intervention planning summary
- Intervention plans generated across the portfolio: 21
- Full detail: `outputs/intervention_plans.csv`

## Files written
- `outputs/stage_token_counts.csv`
- `outputs/cost_summary.csv`
- `outputs/token_math_measurement_summary.csv`
- `outputs/token_math_spreadsheet_export.csv`
- `outputs/quality_review_results.csv`
- `outputs/routing_decisions.csv`
- `outputs/intervention_plans.csv`