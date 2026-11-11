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

## 2026-11-11: LoRA-serve cold-start crash (ap-d4obCWpn1REz469ExJ4dly)

First `dmb-vllm-qwen38-ft` deploy (bf16 base + DPO LoRA): task 1
crashed 41 min in during BASE weight load —
`qwen3_5.py load_weights -> load_merged_column_weight`, shape assert
on a fused-qkv shard. Task 2 (identical code/weights, ~25 min later
after crash-loop backoff) loaded fine from the warm HF cache (52.09
GiB in 39.8 s), compiled, and was mid-warmup when stopped.
Zero completions served; both tasks burned GPU: ledgered 41 + 8 min
at h100 rate (+$3.19, conservative — per-task GPU class is not
recoverable post-hoc via the CLI).

Verdict: transient cold-cache materialization race, not a config
error (the config is proven loadable by task 2). Fix in
`vllm_qwen38_ft.py`: `serve()` pre-flights `snapshot_download` +
a safetensors inventory gate before exec'ing vLLM, so the server
never starts on a half-materialized cache. Same edit adds the
eval-critical flags the crashed revision lacked vs the working base
serve (`vllm_qwen38.py`): `--enable-auto-tool-choice`,
`--tool-call-parser qwen3_xml`, `--limit-mm-per-prompt
'{"image": 8, "video": 2}'`. Only `--max-model-len` still differs
(32768 vs 262144: bf16 weights leave no room for 262K KV).

Side finding: vLLM logs `no matching PunicaWrapper ... visual.blocks.*
will be ignored` at LoRA load. Harmless — SFT/DPO target the text
tower only (`lora_regex_for` excludes vision), so there is no visual
LoRA to drop; the warnings are vLLM scanning ineligible modules.

## 2026-11-11: LoRA-serve Mamba-block hang (ap-SO2bmLSbHZVhKFX0S8ZFFN)

Redeploy with the cold-start gate + base-parity flags: containers
reached `Application startup complete` yet `/v1/models` hung with 0
bytes, repeatedly, across Modal restarts. Full-log forensics found
the deterministic root cause in EngineCore init:

    ValueError: max_num_seqs (1024) exceeds available Mamba cache
    blocks (280). Each decode sequence requires one Mamba cache
    block, so CUDA graph capture cannot proceed. Please lower
    max_num_seqs to at most 280 or increase gpu_memory_utilization.

The hybrid qwen3_5 tower needs one Mamba state block per decode
sequence; bf16 weights (52 GiB) at 0.90 util leave room for only
280, while vLLM defaults to 1024. (The FP8 base serve never hit
this: 31 GB weights leave ample block headroom.) The engine died
in `_initialize_kv_caches`, the API server started anyway without
a working engine (hence "startup complete" + hung requests), and
Modal health-checks eventually recycled the containers — ~110 min
over 3 containers, zero completions. Ledgered 110 min at h100 rate
(+$7.15, conservative: per-task GPU class is unverifiable post-hoc
via the CLI; the chain prefers a100-80gb, so true spend is likely
~$2.60 lower). Fix: `--max-num-seqs 256` (error's own prescription,
with margin). Side benefit: profiling warmup drops from the 465 s
seen with 1024 seqs.

Budget note: issue-8 slice now ~$29.37/$35
(19.03 + 3.19 crash + 7.15 hang). The trim order in the plan WILL
trigger for the remaining stages (dpo-eval + SafeTune + GRPO + RFT +
final + judges on ~$5.60).
