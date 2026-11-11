---
license: cc-by-nc-4.0
tags: [replay, eval, disaster-response, hsea]
---
# Issue #74 replay bundle (fixed seed 74)

Before/after example pairs: inputs, per-stage generations
(base/sft/dpo), and every judge's reasoning + verdicts.
`aya74_gpu_replay.jsonl` extends the bundle with the GPU-leg
post-repair T5 duels (base/repair generations + all three judges).
Reproduced by `scripts/replay_loop.py --issue 74 --check`
in sanggusti/expert-spork. Research-only (NC stack).
