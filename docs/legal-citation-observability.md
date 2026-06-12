# Legal Citation Observability Runbook

This runbook covers monitoring for legal citation drift and automatic citation repair in `chat-wonder-v2-api`.

## Scope

The legal response pipeline enforces citation links to use valid `/sources/<item_id>` values from active `search_legal` results.  
When invalid or out-of-set ids are detected, the server rewrites links and emits observability signals.

## Metrics Emitted (CloudWatch EMF)

Namespace: `ChatWonder/Legal`

- `legal.citation_invalid_detected.count`
  - Meaning: number of invalid citation ids found before rewrite in a response
  - Unit: `Count`
  - Typical tags/dimensions:
    - `metric_name`
    - `reason` (for example `invalid_or_out_of_set_id`)

- `legal.citation_repair.count`
  - Meaning: number of repair executions (rewrite pass triggered)
  - Unit: `Count`
  - Typical tags/dimensions:
    - `metric_name`
    - `reason`

## Logs and Trace Signals

- Log warning on guard activation:
  - `[legal-citation] invalid source ids after repair=...; forcing fallback id=...`
- Internal trace event (not user-visible):
  - `Legal citation guard rewrote invalid source ids: ... -> ...`
- Metric log line:
  - `[metric] type=counter name=... value=... total=... tags=...`
- EMF JSON log line is emitted for CloudWatch metric ingestion.

## Dashboard (15-minute windows)

Create service-level widgets using **sum across all instances**:

- `SUM(legal.citation_invalid_detected.count)`
- `SUM(legal.citation_repair.count)`
- Optional math expression:
  - `repair_to_invalid_ratio = legal.citation_repair.count / legal.citation_invalid_detected.count`

Add a breakdown panel grouped by `reason`.

## Alarm Thresholds

Use aggregated service-level alarms:

- Warning: `SUM(legal.citation_invalid_detected.count) > 5` over 15 minutes
- Critical: `SUM(legal.citation_invalid_detected.count) > 20` over 15 minutes

Route warning/critical actions to standard SNS/Slack/Pager targets.

## Incident Triage Steps

1. Confirm alert window and inspect `legal.citation_invalid_detected.count` trend.
2. Check recent deploys affecting:
   - legal prompts
   - response post-processing
   - model or tool-call behavior
3. Inspect logs for:
   - `[legal-citation]` warning entries
   - trace events about citation guard rewrites
4. Verify active `search_legal` result ids and compare against generated links.
5. If drift spikes after prompt/model change, roll back or tighten prompt constraints.
6. Keep guard enabled; never disable runtime citation repair during incident response.

## Ownership

- Service owner: Legal AI / Chat Wonder API team
- Operational owner: Platform/DevOps on-call

