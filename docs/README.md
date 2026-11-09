# Disaster Mitigation Bench — Technical Docs

Benchmarking language models as **drone navigators** and **reconnaissance
advisors** in simulated 3D disaster environments.

| Doc | Contents |
|---|---|
| [specs](specs.md) | Pinned versions, endpoints, budgets, status |
| [frontier](frontier.md) | Feasibility verdict + improvement roadmap |
| [architecture](architecture.md) | System layout, eval loop, package map |
| [compute-split](compute-split.md) | Where each workload runs + trade-off analysis |
| [action-surface](action-surface.md) | ArduPilot-derived drone tools and skills |
| [edge-perception](edge-perception.md) | Edge-first vision design and budgets |
| [sam-segmap](sam-segmap.md) | SAM segmentation maps → routes + severity |
| [sam-routes-report](sam-routes-report.md) | Progress report: SAM routes + baseline figures |
| [scenarios](scenarios.md) | Disaster scenario catalog and generator |
| [watcher](watcher.md) | Scenario watcher: observability & replay |
| [physics-assets](physics-assets.md) | Physics asset catalog plan |
| [metrics](metrics.md) | Score vector, weights, and metric roadmap |
| [inference](inference.md) | Baseten + Modal model routing through inspect-ai |
| [tracking](tracking.md) | Run registry, eval logs, reproducibility contract |
| [track2-runbook](track2-runbook.md) | Colosseum workstation setup + verdict repro |
| [flywheel](flywheel.md) | Phase D data loop: export, finetune, delta, artifacts |
| [human-evals](human-evals.md) | Operator-in-the-loop baselines and grading |
| [assets](assets.md) | Drone specification assets pipeline |
| [runbooks](runbooks.md) | Setup, daily commands, troubleshooting |
| [roadmap](roadmap.md) | Workstreams and their GitHub issues |

Conventions: `dmb/` is the Python package, `evals/` holds inspect-ai tasks,
`scenarios/` the disaster definitions, `assets/drones/` the vehicle specs,
`infra/` the sim + serving definitions, `runs/` the tracked outputs.
