# Human Evals — Operator in the Loop

The operator participates in the benchmark in two roles.

## 1. Human baseline

A person flies the same mission tasks through inspect-ai's human agent
(`human_cli` for sandbox tasks; a guided CLI over the same drone tools for
sim missions — recon workstream wires this up). Human transcripts score
through the identical deterministic scorers, producing the reference row
every model row is compared against.

Protocol:

1. Check out the pinned task commit; record it in the run notes.
2. Run the mission blind (no prior look at victim placements) via
   `.venv/bin/python -m dmb.human_cli --scenario <s> --seed <n> --name <you>`.
   The CLI never reveals placements.
3. Transcript + scores land in the registry like any other run, tagged
   `baseline:human:<name>`.

## 2. Human grader (recon advisor)

Recon guidance quality cannot be fully automated. Grading rubric (v0, each
1–5):

- **Actionability**: ordered steps a stressed civilian can execute.
- **Correctness**: directions match the true map/hazard state.
- **Completeness**: covers where to go, what to do, what to avoid.
- **Brevity**: no ornament; every sentence earns its place.

Workflow: export recon transcripts from the eval log → grade against the
rubric in the issue/PR → record per-item grades + grader identity in the
run notes. A simulated human-actor check (scripted compliance model)
complements — never replaces — human grades.

## Rules

- Graders never grade runs they baselined.
- Rubric changes are reviewed diffs with before/after re-grades noted.
- Disagreements are data: record both grades, don't average silently.
