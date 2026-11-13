# High-stakes Embodied Agents (HSEA)

Safety-graded finetuning loop for embodied disaster-response agents —
Research project. Seeded from the disaster-mitigation
bench (Inspect-ai harness + Modal serving + HF datasets); the sim tracks
(Gazebo/SITL, Colosseum/Unreal) stayed behind.

## Showcase and writeup

- Demo: [High-stakes Embodied Agents](https://huggingface.co/spaces/sanggusti/high-stakes-embodied-agents) (HF Space, public)
- Writeup: [docs/articles/high-stakes-embodied-agents.md](docs/articles/high-stakes-embodied-agents.md): submission-ready source for the HF community blog post (methods, both loops, plots, tables); publish via https://huggingface.co/new-blog
- Loop reports: [issue-73-loop.md](docs/reports/issue-73-loop.md) (Tiny Aya), [issue-74-aya-expanse.md](docs/reports/issue-74-aya-expanse.md) (Aya Expanse), [issue-10-safetune-harness.md](docs/reports/issue-10-safetune-harness.md) (shared SafeTune harness)

## Stack

| Layer | Tool |
|---|---|
| Evals / agents | Inspect-ai (`evals/`, `hsea/`) |
| Serving | Modal vLLM (`infra/modal/`) + HF Inference routing |
| Datasets / models | Hugging Face |
| Curation | CuratorKIT ([flywheel-v2 report](docs/reports/issue-7-flywheel-v2-curation.md)) |
| Finetuning | AlignTune (loops [#73](https://github.com/sanggusti/expert-spork/issues/73), [#74](https://github.com/sanggusti/expert-spork/issues/74)) |
| Safety drift | SafeTune shared harness ([issue #10](https://github.com/sanggusti/expert-spork/issues/10), `hsea/safety/harness.py`) |
| Post-training | AgentTune (RFT + GRPO legs in the loop reports) |

SAM 3.1 survey work needs a render corpus via `HSEA_RENDER_FRAMES_DIR`.

## Quickstart

```bash
uv venv .venv
VIRTUAL_ENV=$PWD/.venv uv pip install -e ".[dev]"
cp .env.example .env   # fill in provider keys (HSEA_ prefix)
.venv/bin/python -m pytest -q
.venv/bin/inspect eval evals/forest_fire.py --model mockllm/model --limit 1
```

## Docs

- [docs/research-track.md](docs/research-track.md): HSEA big picture (mission, tracks, model lanes, loop, retired docs)
- [docs/process.md](docs/process.md): Lexsi-loop operating procedure (curate, align, harden, agent, eval)
- [docs/](docs/) — bench technical docs (lineage-noted)
- [WORKFLOW.md](WORKFLOW.md) — Issue → PR → Merge workflow
- [SEEDING.md](SEEDING.md) — what was seeded, adapted, left behind
- [docs/articles/high-stakes-embodied-agents.md](docs/articles/high-stakes-embodied-agents.md): HF community blog post source
