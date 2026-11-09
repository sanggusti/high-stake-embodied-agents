# Seeding Manifest

Seeded from `automatic-octo-meme` @ `308027c` (Inspect-ai + Modal + HF subset).
Left behind: Unreal Engine (`unreal/`), Colosseum bridges/tests, Gazebo world
builders + model/texture bulk (`assets/` except drone specs), sim Docker image
(`infra/sim/`), footage/renders bulk, bulk logs, UE/Colosseum docs and skill.

## COPIED as-is

- `dmb/` harness core: scenarios, solvers (mission/baselines), scorers
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

## COPIED pending ADAPT (follow-up issues)

- `dmb/sim/bridge.py`: keep MockBridge/SimBridge; strip GazeboSITLBridge.
- `dmb/sim/assets.py`: keep spec loader; drop gazebo_model refs.
- `dmb/perception/{frames,sam,render_frames}.py`: keep map-render + Meta
  SAM API layer; drop Gazebo capture + survey-corpus paths (or re-point).
- `dmb/watch/l3.py`: keep runs/<id> contract; drop Gazebo record hooks.
- Docs: compute-split, physics-assets, assets, frontier, peatforest-evals
  (bench+sim mix; re-point slugs/app names).
- Tests: test_sitl_units, test_assets, test_flood, test_quake_slide (split
  Mock/concept vs Gazebo/SDF asserts).
- Root `pyproject.toml` (drop sim extras), `README.md` (rewrite for the
  Lexsi stack; see scaffold plan).

## Stack mapping (planned)

- CuratorKIT <- dmb/data/export.py + runs trajectories + data/flywheel-v1.
- AlignTune <- infra/modal/finetune.py + reshard.py + vllm_* eval endpoints.
- SafeTune <- NEW around AlignTune/AgentTune outputs (Harden/Recover/Steer).
- AgentTune <- dmb/solvers/mission.py ReAct loop + tools + watch L2 logs.
- Modal <- infra/modal/* + dmb/models.py routing. HF: new gated rev-pinned
  dataset + model collections (do NOT reuse peatforest pointers).
