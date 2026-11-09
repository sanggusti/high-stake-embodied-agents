# Abstention/escalation eval set (issue #21)

Curator-written text/tool-trace scenarios (no sim): the agent must abstain
or escalate on uncertain/high-stakes calls. 60 risk scenarios (20 abstain /
20 escalate / 20 comply across 10 disaster-ops categories) plus a paired
20-scenario benign set (comply-only, AgentHarm pattern) guarding against
over-refusal.

- **Builder/loader:** `hsea/data/abstention.py` (`load_scenarios`,
  `validate_scenario`, `write_manifest`, `verify_manifest`); manifest
  conventions mirror `hsea/data/export.py` (counts, SHA-256 per file and
  per record).
- **Task:** `evals/abstention.py` (`abstention`); **scorer:**
  `hsea/scorers/abstention.py` (`abstention_score`).
- **Rubric:** `rubric.md` (labels, decision tests, partial credit,
  confidence semantics). **Inter-rater spot-check:** `rater2.json`
  (20 scenarios, agreement 1.00, κ 1.00 — see method note inside).
- **Provenance:** `docs/alignment-research.md` §4 playbook
  (calibration/refusal/abstention); AgentHarm benign-set pattern. Split
  rule (mechanical, in `rubric.md`): index % 3 == 0 → `cal` (28 total),
  else `eval` (52 total).

## Repro gates (no writes)

```sh
.venv/bin/python -m hsea.data.abstention --verify-only   # manifest SHA-256 verify
.venv/bin/pytest tests/test_abstention.py -q             # grading + dataset tests
```

## Run

```sh
.venv/bin/inspect eval evals/abstention.py --model openai-api/baseten/meta-models/muse-spark-open-1026
.venv/bin/inspect eval evals/abstention.py --model openai-api/hf/deepseek-ai/DeepSeek-V4.1-Flash
.venv/bin/python scripts/abstention_curves.py --logs <a.eval> <b.eval> \
  --labels muse-spark,deepseek-v41-flash --out-dir docs/reports --prefix issue-21-abstention
```
