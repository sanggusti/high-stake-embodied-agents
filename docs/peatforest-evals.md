# Peatforest eval results (2026-11-04)

Task: `evals/forest_fire_peat.py` (peat mission, seeds 1-3, mock
backend) + SAM 3.1 probes (mock edge + native frames).

## Mission scores (mock backend, stub edge unless noted)

| model | visual | edge | seeds | mission_score (mean) | tokens | log |
|---|---|---|---|---|---|---|
| baseten `meta-models/muse-spark-open-1026` | yes | stub | 1 | 0.780 | 102k | `logs/2026-11-04T08-18-38*_GrezgBnGE497G8vtJWYPCR.eval` |
| baseten muse-spark (same slug) | yes | stub | 1-3 | **0.722** | 341k | `logs/2026-11-04T11-18-00*_axK9Tqay7yGMgcBCmYtjtC.eval` |
| modal `Qwen/Qwen2.5-7B-Instruct` (fresh `modal deploy`) | text (`DMB_VISION=text`) | stub | 1 | 0.633 | 2.7k | `logs/2026-11-04T08-27-32*_GnNwPE8B7ptre3DdviuBit.eval` |
| modal Qwen2.5-7B | text | stub | 1-3 | **0.586** | 26k | `logs/2026-11-04T11-19-06*_7PDusFNuSY9LRHNiVpDFaj.eval` |
| baseten muse-spark | yes | SAM 3.1 (Meta API `sam-3.1`) | 1 | 0.075 | 108k | `logs/2026-11-04T08-47-57*_7knBm5hQzsHzUDScCSfLRz.eval` |

Scope lock (permanent, per user): Baseten serves ONLY
`meta-models/muse-spark-open-1026` — never other slugs. Modal
column = the project vLLM deployment
(`infra/modal/vllm_server.py`, redeployed 2026-11-04, same URL as
recorded in `docs/specs.md`). The 0.075 SAM-on-mock run is the
documented degradation (SAM returns zero matches on symbolic
mock renders, agent flies blind) — kept as the wiring proof.

## SAM 3.1 native frames (Meta API `sam-3.1` via `META_AI_API_KEY`)

Frames: `runs/sam_frames/*.png` (1280x720 native, scored at
640px). Recorder split for transition flakiness:
`runs/sam_capture.py` (game) + `runs/sam_score.py` (offline API).

| frame | concept | matches | frame frac |
|---|---|---|---|
| fire (level, hotspot4) | fire | 16 | 0.0086 |
| fire | smoke | 4 | 0.0399 |
| fire | forest | 0 | 0 |
| fire_high (pitched) | fire | 14 | 0.0030 |
| fire_high | smoke | 0 | 0 |
| fire_high | forest | 0 | 0 |
| canopy | forest | 0 | 0 |
| canopy | tree | 19 | 0.0266 |
| canopy | vegetation | 2 | 0.0054 |
| canopy | fire | 0 (correct reject) | 0 |
| path | road/trail/dirt-road/path/footpath | 0 each | 0 |

Strict score 3/6 (fire+smoke on level frames hit; smoke from
above, forest-as-word, and the dirt track miss). Counting
tree~=forest: 4/6. Takeaways: SAM covers flame/smoke/tree on
native sim frames; "forest" is a vocabulary miss (use "tree");
the flat synthetic dirt ribbon is invisible to SAM under all
five track concepts -- a fidelity gap (no edge/shoulder
texture), also noted for the hidden-path eval design.

Latency note: native 640px calls take ~6-10 s; 720p exceeded
the OpenAI 60 s default, so `openai_segmenter` now defaults to
`timeout=600`. No local SAM runs anywhere: all SAM traffic goes
to `https://api.meta.ai/v1` (Mac+Gazebo evals use the same key
and work because their frames are small).

## Cost notes

- Baseten 3-seed: 341k tokens total (~88 min include reasoning).
- Modal: A10G spot while deployed; deployment left UP for
  follow-up evals (stop with `modal app stop` when done).
- SAM: ~20 calls total (mock limit-1 + native), single-digit
  minutes of API time excluding transitions.
