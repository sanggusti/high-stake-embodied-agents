# Architecture

## Eval loop

One benchmark sample is `(scenario, seed)`. Execution per sample:

```
Sample (mission brief + metadata)
  -> Solver builds isolated episode: SimBridge + EdgePerception + Tools
  -> ReAct agent loop (later: Deep Agent for long-horizon missions)
       LM --tool call--> drone tool --> SimBridge.act() --> Observation
       SimBridge state --> EdgePerception --> event-driven reports
  -> Solver stores final sim state in state.store["sim_state"]
  -> mission_score: deterministic metric vector from sim state + transcript
  -> RunRegistry records config, scores, eval-log pointers
```

## Package map

- `hsea/config.py` — env-driven `Settings` (providers, backend, budgets).
- `hsea/models.py` — inspect-ai model ids for Baseten/Modal
  (`openai-api/<service>/<model>`).
- `hsea/registry.py` — `RunRegistry`: `runs/<run_id>/{config,scores,events}`.
- `hsea/scenarios.py` — YAML loading, seeded `Sample` generation, world hashes.
- `hsea/sim/bridge.py` — `SimBridge` ABC; `MockBridge` (runnable grid world,
  the only backend — sim tracks stayed in the bench repo).
- `hsea/sim/assets.py` — versioned drone-spec loader.
- `hsea/perception/edge.py` — `EdgePerception` interface, trigger config,
  snapshot budget, rule-based stub.
- `hsea/tools/drone.py` — `@tool` factories bound to one episode's instances.
- `hsea/scorers/mission.py` — pure `compute_mission_scores` + scorer wrapper.
- `evals/` — `smoke_inference` (provider routing), `forest_fire` (sample
  mission), `recon_advisor` (human-guidance stub).
- `infra/modal/` — vLLM OpenAI-compatible server for custom models.

## Backend isolation rule

Every sample gets its own bridge instance. The mock backend satisfies this
trivially. Parallelism never shares mutable sim state.

## Role composition

- **Drone navigator** (this scaffold): flight tools + edge reports.
- **Recon advisor** (`evals/recon_advisor.py` stub): produces human guidance;
  graded by rubric + simulated human actor (see [human-evals](human-evals.md)).
- Combined missions compose both agents; the advisor consumes the navigator's
  edge reports rather than raw sim state.
