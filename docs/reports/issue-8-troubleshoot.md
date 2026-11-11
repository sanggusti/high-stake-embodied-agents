# Issue #8 troubleshooting: why the overnight SFT-only run was ditched

Date: 2026-11-11. Sources: Modal app logs (`dmb-qwen38-finetune`
apps 07:52 and 07:55 WIB), app-list trail (~15 stopped apps 21:32-07:55),
`infra/modal/finetune_qwen38.py` at ditch time. No guessing: each finding
below traces to one of those.

## What happened

The agent iterated all night and finally got QLoRA SFT training at 07:55
(steps 0-35/3255, loss 1.574 at step 25). The run was stopped per user
order (SFT-only is not wanted) and the R2a worker cancelled. The SFT was
healthy but ~1% done on a 5-8 hour trajectory.

## Root causes

1. Bleeding-edge API drift. `transformers` + `trl` installed from git
   main. The 07:52 attempt died with
   `TypeError: SFTConfig.__init__() got an unexpected keyword argument
   'group_by_length'` (TRL main removed the arg). Each such failure cost
   a full setup cycle (image build + 27B weight load).
2. Fused-kernel saga. The qwen3_5 hybrid linear-attention tower wants
   `causal-conv1d` / `flash-linear-attention`. Source builds proved
   brittle (see the image-build comments); packing was abandoned; the
   final run fell back to the reference PyTorch implementation
   (`chunk_gated_delta_rule ... much slower`). Throughput was 5-9 s/it.
3. Estimate math was wrong. The pre-run estimate said ~1,630 steps;
   the true count is 3,255 (8,673 rows / eff-batch 8 x 3 epochs).
   Combined with (2), true SFT cost was ~$11-20, not <=$6.25.
4. GPU queue wars. The script requested a strict single class
   (`gpu="A100-80GB"`, no fallback) while #74's finetune wanted the
   same class. Overlap windows produced Modal "waiting for GPU" stalls.
   Spawning more instances cannot help: every new `modal run` joins
   the same queue.
5. Setup-cycle compounding. ~15 stopped apps = ~15 image-build +
   weight-load cycles before the first real training step.

## Rules for the redo (binding)

- No fused-kernel source builds. Prebuilt wheels only, or accept the
  reference implementation with honest time math.
- Steps = rows / effective-batch x epochs. Measure s/it over the first
  50 steps, extrapolate wall time and cost, and re-estimate BEFORE
  committing to a multi-hour run. Abort and re-plan if 2x over estimate.
- GPU fallback chains on every Modal function (e.g.
  `["A100-80GB", "H100"]`). Never a strict single class.
- Pin `transformers`/`trl` to a tested release or a known-good commit,
  not floating main. Record the pins.
- Sequence 80GB-class jobs against #73/#74; do not queue-war them.
