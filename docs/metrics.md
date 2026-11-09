# Metrics

## Score vector (v0, `dmb/scorers/mission.py`)

Computed deterministically from final sim state + transcript stats:

| Metric | Definition |
|---|---|
| `saved_human` / `saved_child` / `saved_animal` | Recall per victim category |
| `rescue_recall` | Weight-weighted recall (child 3, human 2, animal 1) |
| `containment` | Per-kind beat (see below), × safety |
| `efficiency` | `(1 − steps/step_budget)` if any victim found, else 0 (idle fix) |
| `safety` | 1.0 with no violations, −0.25 per violation, floored at 0 |
| `tool_calls` | Transcript tool-call count (cost proxy) |
| `found_any` | 1.0 if ≥1 victim found, else 0.0 (mean = first-find rate) |
| `first_find_step` | Episode step of first find (−1 if none) |
| `coverage` | Unique cells visited / total cells |
| `damage_cells` | Cells with a scheduled danger step (exposure, informational) |
| `unburned_fraction` / `unflooded_fraction` | 1 − ignited/flooded share (informational) |
| `debris_cells` / `buried_victims` | Slip extent + buried count (informational) |
| `routes_requested` | `request_route_map` outputs parsed from transcript |
| `routes_reachable_rate` | Claimed-reachable share (0 when none) |
| `route_safety` | Planned waypoints outside true danger (0 when none) |
| `route_excess_length` | Planned ÷ A\*-optimal steps on true costs (0 when none) |
| `spl_like` | Mean found × manhattan ÷ max(manhattan, found_step) |
| `composite` | `0.5·recall + 0.2·containment + 0.15·efficiency + 0.15·safety` |

Route metrics are informational (not in `composite`): vacuous episodes
score 0 so stub-perception arms read as incapable rather than perfect.
Truth danger = activated `cell_danger` + phase-active hazard circles
(`danger_cells_from_state`); the planner-optimal baseline uses
planner-identical costs so excess measures pure optimality.

The inspect-ai scorer returns `composite` as the value with the full vector
in `metadata`, so leaderboards sort on one number while analysis keeps every
dimension.

## Design notes

- **Weights encode triage priority**: children first, then humans, then
  animals. Weights live in one constant (`VICTIM_WEIGHTS`) and every change
  is a tracked, reviewable diff.
- **Efficiency-vs-safety trade-off** is scored explicitly (both terms feed
  `composite`), following the aerial-SAR benchmark literature.
- `containment` is per-kind beat-the-disaster (#4), × safety:
  - tsunami: beat-the-wave — found before wave-arrival starts.
  - fire/water/terrain kinds: beat-the-spread on cell danger steps.
  - no dynamics (earthquake): dangers are infinite ⇒ reduces to recall.
  True suppression-based containment (perimeter held, water pumped, rubble
  cleared) awaits mitigation actions, which no tool currently offers; the
  exposure metrics above quantify disaster scale in the meantime.

## Difficulty scoring (`dmb/difficulty.py`)

Every sample carries `difficulty_score: {score 0..1, band}` computed
deterministically from victim count, mean distance from origin, hazard
coverage, visibility, and mobility mix. Bands (easy < 0.55 < medium < 0.65
< hard) are calibrated on seeds 1..100 across scenarios. Validation: a
fixed-effort probe scores easy > medium > hard
(`tests/test_baselines.py`).

## Recon compliance (`dmb/scorers/recon.py`)

Scripted-actor check of advisor waypoint chains against hidden truth:
`recon_compliance` (per-victim best chain, validity × safety, averaged),
`recon_chains`, `recon_coverage`. Complements the human rubric grades in
[human-evals](human-evals.md); never replaces them.

## Metric roadmap

`something found along the way` goes here through issues, not ad-hoc edits:
time-to-first-victim, area coverage, snapshot/token cost per rescue,
human-preference win rate, composite weights for route metrics (needs
calibration data first).
