# Seeding Manifest

Seeded from `automatic-octo-meme` @ `308027c` (Inspect-ai + Modal + HF subset).
Left behind: Unreal Engine (`unreal/`), Colosseum bridges/tests, Gazebo world
builders + model/texture bulk (`assets/` except drone specs), sim Docker image
(`infra/sim/`), footage/renders bulk, bulk logs, UE/Colosseum docs and skill.

## COPIED as-is

- `hsea/` harness core: scenarios, solvers (mission/baselines), scorers
  (mission/firms_spread/recon), tools/drone, models, config, registry,
  aggregate, difficulty, human_cli, verdicts; data/{export,splits,firms};
  physics/{fire,terrain,water}; perception/{edge,vlm,oracle,segmap,grounding};
  watch/{recorder,render,replay,compose}.
- `evals/` (17 inspect-ai Tasks incl. flywheel_train/test) + `scenarios/`
  (17 YAMLs).
- `infra/modal/` (6 files: vllm servers, finetune, reshard).
- `data/flywheel-v1/` (SFT/RLVR + manifest) + `data/firm/` (zip + readme;
  30M CSV regenerates).
- `runs/` subset (trajectories/ + scores/config/events only, 108 runs) +
  `logs/*.eval`.
- `docs/` (17 bench .md files incl. peatforest-evals, flywheel, inference).
- `tests/` (24 files) + fixtures; root `.env.example`, `pyproject.toml`,
  `.gitignore`; `assets/drones/` specs.

## ADAPTED post-seed (2026-11-09; package `dmb` -> `hsea` first)

- `hsea/sim/bridge.py`: MockBridge/SimBridge only (SITL + Colosseum cut).
- `hsea/sim/assets.py` + drone specs: spec loader kept, gazebo_model cut.
- `hsea/perception/frames.py`: map-render only (Gazebo capture cut).
- `hsea/perception/sam.py` + `render_frames.py`: Meta SAM API layer kept;
  survey corpus NOT shipped, mount via `HSEA_RENDER_FRAMES_DIR`.
- `hsea/solvers/mission.py`: `mock` is the only backend.
- Tests: `test_backends.py` (renamed, mock-only), assets/flood/quake/slide
  keep Mock/concept asserts; SDF builder tests cut, survey tests skip
  without a mounted corpus.
- `pyproject.toml` (sim extra dropped), `README.md` (HSEA rewrite),
  `.env.example` (mock-only), docs lineage-noted + runbooks retired.
- Live external names stay `dmb-` (Modal apps/volumes/endpoints, HF repos).

## Stack mapping (planned)

- CuratorKIT <- hsea/data/export.py + runs trajectories + data/flywheel-v1.
- AlignTune <- infra/modal/finetune.py + reshard.py + vllm_* eval endpoints.
- SafeTune <- NEW around AlignTune/AgentTune outputs (Harden/Recover/Steer).
- AgentTune <- hsea/solvers/mission.py ReAct loop + tools + watch L2 logs.
- Modal <- infra/modal/* + hsea/models.py routing. HF: new gated rev-pinned
  dataset + model collections (do NOT reuse peatforest pointers).
