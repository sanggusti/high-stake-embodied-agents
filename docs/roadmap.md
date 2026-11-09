# Roadmap

Ordered workstreams; each owns a GitHub issue. Dependencies flow top-down.
Status as of 2026-10-27 in brackets.

1. **Inference spike** (#1) [done 2026-10-29: Baseten muse-spark-open-1026
   verified (smoke 1.0, forest_fire 0.608); Modal half done 2026-10-27] —
   first verified Baseten + Modal runs. Unblocks all real-model work.
2. **Sim spike (Track 1)** (#2) [done 2026-10-27: image, bridge, flight
   proofs, 2-parallel isolation] — pinned image builds; SITL+Gazebo boots;
   scripted GUIDED takeoff/hover/land; MAVLink message set documented;
   `GazeboSITLBridge` implemented against the `SimBridge` contract.
3. **Drone-spec assets** (#3) [done 2026-10-29: both specs sourced with
   per-value citations, solver + reset enforcement] — replace placeholder
   specs with pulls from ArduPilot frame params + `ardupilot_gazebo`
   models; enforce
   capability gating in the solver.
4. **Scenario pack** (#4) [done 2026-10-29: bands documented, weather/time
   axes wired, per-kind containment + efficiency idle fix, determinism
   holds] — difficulty stratification, weather/time dynamics, per-disaster
   containment models replacing the v0 proxy.
5. **Recon advisor + human evals** (#5) — human-actor check, guided human-agent
   baseline tooling, first graded + baselined runs.
6. **Fidelity track (Track 2)** (#6) [done 2026-10-29: Blocks live on PC,
   hello flight green, mock-vs-Colosseum verdict 3/3 non-vacuous] —
   Colosseum/Unreal on the GPU workstation, `ColosseumBridge`, one
   parported scenario.
7. **Baseline sweep** (#7) [done 2026-10-29: 4-column matrix tabled —
   spark 0.755, heuristic 0.704, random 0.638, qwen-7b 0.535] — fixed
   Baseten model + Modal customs + scripted baselines over the full
   scenario pack; first published results table.
8. **Results aggregation** (#8) [done 2026-10-29: record + table CLIs,
   3 runs backfilled, reproduces posted numbers] — `runs/*/scores.json` →
   tables/plots script.
9. **Tsunami scenario** (#15) — task wiring + phase dynamics + golden scores.
10. **Scenario watcher** (#16) [done 2026-10-29: L2+T1, all T2 legs,
    T3 HTML, registry evidence links, L2 audit] — L2 event log, T1 replay
    CLI, T2 backend capture, T3 composite.
11. **Asset pack: vegetation + fire** (#17) [done 2026-10-29: fire weather
    coupling + tolerance bands, Fuel tree pins + placement script, T2 plan].
12. **Asset pack: terrain + water** (#18) [done 2026-10-29: slip weather
    coupling + rule doc, buoyancy spike (stock plugins), loud reset].
13. **Asset pack: actors** (#19) [done 2026-10-29: mobility schema +
    detectability profile, figure pins + scaling, ethics rule].
14. **Frontier A: eval rigor** (#25) [done 2026-10-27: difficulty bands,
    intermediate metrics, 3 human baselines, 100-seed CI] — difficulty,
    intermediate metrics, human baselines, 100+ seeds, heuristic agents.
15. **Frontier B: physics accuracy** (#26) [done 2026-10-28: T0 fire/water/
    slip models, beat-spread containment, snapshot suite] — validated
    fire/water/terrain dynamics, per-disaster containment.
16. **Frontier C: perception + fidelity** (#27) [done 2026-10-28: native
    SITL frames, vision tools, edge-VLM baseline, Colosseum bridge +
    verdict harness] — camera frames, edge VLM, UE GIS terrain.
17. **Frontier D: data flywheel** (#28) [done 2026-10-28: L2 + export,
    224-pair SFT, LoRA finetune, held-out delta −0.049 (null, mechanism
    diagnosed), dataset + weights on HF] — trajectory export, finetune,
    held-out delta.
18. **Frontier E: release bar** (#29) — 600+ tasks, full baselines,
    one-command reproduction.
