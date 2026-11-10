# High-stakes Embodied Agents (HSEA)

Safety-graded finetuning loop for embodied disaster-response agents —
Research project. Seeded from the disaster-mitigation
bench (Inspect-ai harness + Modal serving + HF datasets); the sim tracks
(Gazebo/SITL, Colosseum/Unreal) stayed behind.

## Stack

| Layer | Tool |
|---|---|
| Evals / agents | Inspect-ai (`evals/`, `hsea/`) |
| Serving | Modal (`infra/modal/`) |
| Datasets / models | Hugging Face |
| Curation | CuratorKIT (planned) |
| Finetuning | AlignTune (planned) |
| Safety drift | SafeTune (planned) |
| Post-training | AgentTune (planned) |

Only the `mock` backend ships; SAM 3.1 survey work needs a render
corpus via `HSEA_RENDER_FRAMES_DIR`.

## Quickstart

```bash
uv venv .venv
VIRTUAL_ENV=$PWD/.venv uv pip install -e ".[dev]"
cp .env.example .env   # fill in provider keys (HSEA_ prefix)
.venv/bin/python -m pytest -q
.venv/bin/inspect eval evals/forest_fire.py --model mockllm/model --limit 1
```

## Docs

- [docs/](docs/) — bench technical docs (lineage-noted)
- [docs/training-runbook.md](docs/training-runbook.md) — curate → tune → eval → publish runbook (+ [cards](docs/cards/))
- [WORKFLOW.md](WORKFLOW.md) — Issue → PR → Merge workflow
- [SEEDING.md](SEEDING.md) — what was seeded, adapted, left behind
- Demo: [High-stakes Embodied Agents](https://huggingface.co/spaces/sanggusti/high-stakes-embodied-agents) (HF Space, private)
