# HSEA Technical Docs (seeded from the Disaster Mitigation Bench)

> Lineage note: these docs describe the bench the project was seeded
> from. The sim tracks (Gazebo/SITL, Colosseum/Unreal) and the
> survey-render corpus stayed behind — HSEA runs the `mock` backend;
> mount a corpus via `HSEA_RENDER_FRAMES_DIR` for SAM survey work.

Benchmarking language models as **drone navigators** and **reconnaissance
advisors** in simulated 3D disaster environments.

| Doc | Contents |
|---|---|
| [specs](specs.md) | Pinned versions, endpoints, budgets, status |
| [architecture](architecture.md) | System layout, eval loop, package map |
| [action-surface](action-surface.md) | ArduPilot-derived drone tools and skills |
| [edge-perception](edge-perception.md) | Edge-first vision design and budgets |
| [sam-segmap](sam-segmap.md) | SAM segmentation maps → routes + severity |
| [sam-routes-report](sam-routes-report.md) | Progress report: SAM routes + baseline figures |
| [scenarios](scenarios.md) | Disaster scenario catalog and generator |
| [watcher](watcher.md) | Scenario watcher: observability & replay |
| [metrics](metrics.md) | Score vector, weights, and metric roadmap |
| [inference](inference.md) | Baseten + Modal model routing through inspect-ai |
| [models](models.md) | Newest open models to benchmark/finetune/evaluate |
| [datasets](datasets.md) | Disaster footage + action datasets to align on |
| [alignment-research](alignment-research.md) | DORA-inspired alignment research + playbook |
| [tracking](tracking.md) | Run registry, eval logs, reproducibility contract |
| [flywheel](flywheel.md) | Phase D data loop: export, finetune, delta, artifacts |
| [human-evals](human-evals.md) | Operator-in-the-loop baselines and grading |
| [runbooks](runbooks.md) | Setup, daily commands, troubleshooting |
| [training-runbook](training-runbook.md) | Curate → tune → eval → publish loop (Lexsi stack) |
| [cards](cards/) | Dataset/model card templates + per-release cards |
| [roadmap](roadmap.md) | Workstreams and their GitHub issues |

Conventions: `hsea/` is the Python package, `evals/` holds inspect-ai tasks,
`scenarios/` the disaster definitions, `assets/drones/` the vehicle specs,
`infra/modal/` the serving definitions, `runs/` the tracked outputs.
