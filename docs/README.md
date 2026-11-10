# HSEA Technical Docs (seeded from the Disaster Mitigation Bench)

> Lineage note: these docs describe the bench the project was seeded
> from. The sim tracks (Gazebo/SITL, Colosseum/Unreal) and the
> survey-render corpus stayed behind; HSEA runs dataset-composed
> evals with the `mock` backend; mount a corpus via
> `HSEA_RENDER_FRAMES_DIR` for SAM survey work.

Safety-graded finetuning loop for embodied disaster-response agents,
evaluated on dataset-composed footage, action, and safety tracks.

| Doc | Contents |
|---|---|
| [research-track](research-track.md) | HSEA big picture: mission, tracks, model lanes, loop, retired docs |
| [process](process.md) | Lexsi-loop operating procedure: curate, align, harden, agent, eval |
| [specs](specs.md) | Pinned versions, endpoints, budgets, status |
| [architecture](architecture.md) | System layout, eval loop, package map |
| [edge-perception](edge-perception.md) | Edge-first vision design and budgets |
| [sam-segmap](sam-segmap.md) | SAM segmentation maps → routes + severity |
| [sam-routes-report](sam-routes-report.md) | Progress report: SAM routes + baseline figures |
| [watcher](watcher.md) | Scenario watcher: observability & replay |
| [metrics](metrics.md) | Score vector, weights, and metric roadmap |
| [inference](inference.md) | Baseten + Modal + HF model routing through inspect-ai |
| [models](models.md) | Newest open models to benchmark/finetune/evaluate |
| [datasets](datasets.md) | Disaster footage + action datasets to align on |
| [alignment-research](alignment-research.md) | DORA-inspired alignment research + playbook |
| [alignment-methods-vlm](alignment-methods-vlm.md) | 2024-2026 VLM alignment survey for footage agents |
| [tracking](tracking.md) | Run registry, eval logs, reproducibility contract |
| [footage-evals](footage-evals.md) | Footage VLM eval results and repro |
| [abstention-evals](abstention-evals.md) | Abstention and escalation threshold procedure |
| [redteam](redteam.md) | Red-teaming harness and scoring |
| [safety-gates](safety-gates.md) | Safety gates for finetune PRs and SafeTune |
| [license-clearance-24](license-clearance-24.md) | Dataset license clearance sweep |
| [human-evals](human-evals.md) | Operator-in-the-loop baselines and grading |
| [runbooks](runbooks.md) | Setup, daily commands, troubleshooting |

Conventions: `hsea/` is the Python package, `evals/` holds inspect-ai tasks,
`scenarios/` the disaster definitions, `assets/drones/` the vehicle specs,
`infra/modal/` the serving definitions, `runs/` the tracked outputs.
