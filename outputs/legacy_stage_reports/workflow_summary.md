# Customer Success AI Workflow — Run Summary

## Data validation
- All 7 input files loaded and validated successfully; all required columns present.

## Stage 1 — Account Briefings
- Accounts processed: 18
- Priority tiers: High=9, Medium=3, Low=6
- Top-priority account: **Harbor Insurance** (A008), risk score 206.25

## Stage 2 — Quality Review
- Outputs reviewed: 8
- Recommendations: Approved=3, Needs revision=2, Rejected=3

## Simulated LLM usage & cost (no external API calls made)
- Model label: simulated-cs-assistant-v1
- Total simulated calls: 26
- Tokens: 7388 prompt + 2980 completion = 10368 total
- Estimated cost: $0.066864

See `token_usage_log.csv` for the per-call breakdown.