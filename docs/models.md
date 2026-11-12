# HSEA Model Bench — Newest Open-Weight LLMs/VLMs (Nov 2026)

Models to benchmark, finetune, inspect, and evaluate for the High-stakes
Embodied Agents (HSEA) project: disaster-response drone navigator evals
(Inspect-ai, Modal + Baseten serving). Researched 2026-11-09; key picks
independently verified. Every spec below is tied to a source in §5;
anything unverified is marked **[UNVERIFIED]** — do not treat it as fact.

Current bench (to replace/augment): Qwen2.5-7B, Qwen2.5-14B-AWQ,
Mistral-7B-v0.3 on Modal vLLM; `meta-models/muse-spark-open-1026` on Baseten.
Serving constraint: **the Baseten key is scope-locked to muse-spark-open-1026
ONLY** — no new Baseten slugs. Model breadth must come from Modal-hosted
vLLM, scripted baselines, or a *new* provider segment (e.g. HF Inference).

Project direction: build on **Cohere
open-weights models + Hugging Face datasets/models**. Cohere coverage is
mandatory, and every shortlist pick below has an HF repo.

## 1. Comparison table

Legend — Modality: T = text, I = image, V = video, A = audio.
Serving path: which HSEA route can serve it (Modal = self-host vLLM on Modal;
HFInf = new `openai-api/hf/<model>` segment via HF Inference Providers or
vendor API; edge = laptop/on-device class).

| Model | Org | Params | Modality | Context | License | Serving path for HSEA | Role |
|---|---|---|---|---|---|---|---|
| Command A+ (05-2026) | Cohere | 218B MoE / 25B active | T+I | 128K in / 64K out | Apache 2.0 | HFInf; Modal 2×H100 via W4A4 (Cohere-stated) | ★ benchmark frontier + Lexsi anchor |
| Aya Vision 8B / 32B | Cohere (C4AI) | 8B / 32B | T+I | 16K | CC-BY-NC-4.0 (+AUP) | Modal vLLM (8B) | inspect-eval Cohere VLM baseline |
| Tiny Aya | Cohere Labs | 3.35B | T (70+ langs) | 8K | CC-BY-NC-4.0 | Modal / edge | small Cohere baseline (Lexsi breadth) |
| Qwen3.8-27B | Alibaba Qwen | 27B dense | T+I+V | 262K (→1M YaRN) | Apache 2.0 | Modal vLLM (FP8 ~31GB) | ★ main workhorse + finetune candidate |
| Qwen3-VL-8B / 32B | Alibaba Qwen | 8B / 32B dense | T+I+V | 256K (→1M) | Apache 2.0 | Modal vLLM | VLM baseline / finetune candidate |
| Qwen3.5 (0.8B–397B-A17B) | Alibaba Qwen | 0.8B–397B | T+I (all sizes) | 256K–262K | Apache 2.0 (open sizes) | Modal (small sizes) | alt baseline family |
| Molmo2-8B / 4B | Ai2 | 8B (Qwen3) / 4B | T+I+V, pointing/grounding | 37K | Apache 2.0 (weights; some datasets NC) | Modal vLLM (nightly build) | ★ grounding VLM + finetune candidate |
| OLMo 3.1 32B | Ai2 | 32B (Think/Instruct) | T | 64K | Apache 2.0 | Modal vLLM | text reasoning baseline |
| Gemma 4 12B | Google | 12B unified | T+I+V+A | 256K | Apache 2.0 | Modal vLLM (single GPU) | ★ multimodal baseline + finetune cand. |
| Gemma 4 E2B | Google | ~2B edge | T+I+V+A | 128K | Apache 2.0 | edge / Modal | edge-floor baseline |
| Mistral Large 3 | Mistral | 675B MoE / 41B active | T+I | 256K | Apache 2.0 | HFInf only (too big for Modal) | frontier baseline (API evals) |
| Mistral Small 4 | Mistral | MoE [counts UNVERIFIED] | T+I (multimodal) | 256K [UNVERIFIED] | Apache 2.0 | HFInf / Modal [UNVERIFIED] | watch |
| Ministral 3 3B/8B/14B | Mistral | 3B / 8B / 14B | T | 262K [UNVERIFIED] | Apache 2.0 | Modal / edge | small text baseline |
| DeepSeek-V4.1-Flash | DeepSeek | MoE [counts UNVERIFIED] | T+I | 1M [UNVERIFIED] | MIT | HFInf / DeepSeek API | frontier VLM baseline (API evals) |
| DeepSeek-V4-Pro-0813 | DeepSeek | 1.6T MoE / 49B active | T only | 1M | MIT | HFInf / API only | frontier text baseline (API evals) |
| gpt-oss-120b / 20b | OpenAI | 117B/5.1B, 21B/3.6B MoE | T only (reasoning) | 128K | Apache 2.0 | Modal (20b MXFP4); 120b 1×80GB | reasoning/text baseline |
| Kimi K2.6 | Moonshot | 1T MoE / 32B active | T+I (MoonViT) | 256K | modified MIT | HFInf / API only | frontier agent baseline (watch: K2 API retires Dec 2026) |
| Phi-4-reasoning-vision-15B | Microsoft | 15B | T+I (reasoning) | 16K | MIT | Modal vLLM | reasoning-VLM baseline |
| SmolVLM-256M / 500M | HF (HuggingFaceTB) | 256M / 500M | T+I | 8K (256M verified; 500M [UNVERIFIED]) | Apache 2.0 (256M verified on card) | edge | extreme-edge VLM floor |
| SmolLM3-3B | HF (HuggingFaceTB) | 3B | T (hybrid reasoning) | 64K trained (→128K YARN) | Apache 2.0 (verified on card) | Modal / edge | small text baseline |
| Llama 4 Scout / Maverick | Meta | 109B / ~400B MoE | T+I | 10M / 1M | Llama Community License | HFInf only; **HF-gated (401)** — same blocker as before | baseline only if gating resolved |
| InternVL3.5 | OpenGVLab | 1B–38B dense, to 241B-A28B | T+I+V | [UNVERIFIED] | mixed — verify PER CHECKPOINT | Modal (small dense) | alt VLM (license diligence needed) |

## 2. Per-org notes

### Cohere (MANDATORY — Lexsi requirement)
- **Command A+** (May 20–24, 2026) is the prize: first Cohere model under
  **full Apache 2.0**, 218B sparse MoE (25B active), text+image input,
  reasoning + tool use + structured outputs + citations, 48 languages,
  128K in / 64K out. Replaces four earlier Cohere models in one. Managed
  API + open weights released simultaneously. Weights on HF as
  `CohereLabs/command-a-plus-05-2026` in BF16 / FP8 / W4A4; Cohere states
  W4A4 runs on 2×H100 (or 1×B200), so Modal self-host is plausible —
  still verify vLLM + tool-parser support before planning the serve.
- **Command A** (Mar 13, 2025, 111B dense, 256K, CC-BY-NC) and **Command A
  Reasoning** (Aug 21, 2025) are superseded by A+ for frontier purposes;
  HF repo `CohereLabs/c4ai-command-a-03-2025` (gated behind access request).
- **Aya Vision 8B/32B** (CohereLabs, CC-BY-NC, 16K ctx, 23 languages) is
  Cohere's open VLM: image captioning, VQA. Non-commercial only — fine for
  a research bench, not for product. 8B fits Modal vLLM easily.
  No newer Aya-Vision generation found (2025 vintage).
- **Tiny Aya** (announced Feb 17, 2026, India AI Summit; on HF + Cohere
  Platform): 3.35B family, 70+ languages, 8K ctx, CC-BY-NC, runs on a
  laptop offline. **Text-only** (translation/multilingual) — no vision
  found. Good "small Cohere" for Lexsi breadth, not for camera-frame evals.
  Newer sibling **Tiny Aya L2 Thinker** (Oct 6, 2026, 3.35B, 32K ctx,
  CC-BY-NC) adds multilingual reasoning — prefer it where transcripts
  exceed 8K.
- License split to respect: Command A+ = Apache 2.0 (commercial OK);
  everything else Cohere open-weight = CC-BY-NC + Acceptable Use Policy.

### Allen Institute for AI (Ai2)
- **Molmo2** (Dec 2025 / Jan 9, 2026 on OpenRouter; `allenai/Molmo2-8B`):
  still the newest Molmo line. 4B + 8B on Qwen3 + SigLIP 2, plus 7B-O on
  Ai2's own OLMo. Multi-image + **video understanding, pointing, object
  tracking across frames** — the single most drone-relevant capability in
  this survey (ground a described object in a camera frame, track across
  frames). Weights Apache 2.0; 9M+ training examples published, **some
  datasets NC-only** (matters if finetuning on their data mix). 37K ctx
  (OpenRouter). vLLM support via nightly build; HF Transformers needs
  remote code. Full training code released Mar 2026. Also **MolmoWeb**
  (Mar 24, 2026, 4B/8B Apache 2.0, screenshot-agent) — adjacent agentic
  prior, not directly needed.
- **OLMo 3 / 3.1** (Nov 2025; 3.1 upgrades 32B Think/Instruct): 7B + 32B,
  Base/Instruct/Think/RL-Zero variants, 64K ctx, Apache 2.0 with full
  weights+data+code+logs. Text-only. Best "fully open science" text
  reasoning baseline. Next-gen sparse-MoE Olmo in training on OlmoCore 3
  (Oct 2026 infra release) — no weights, watch item.

### Google — Gemma
- **Gemma 4** (Apr 2, 2026; 12B added Jun 3, 2026) is the current
  generation, and the first under **Apache 2.0** (Gemma ToU now covers 3
  and earlier only). Five sizes, verified against HF model-card READMEs:
  E2B, E4B (128K ctx), 12B, 26B-A4B (MoE), 31B (256K ctx). **All five take
  text+image**; E2B/E4B/12B also take audio (12B "Unified" is encoder-free,
  140+ languages). The 12B fits a single desktop GPU; E2B/E4B are
  phone/edge class. Cleanest license story in its weight class.

### Alibaba — Qwen
- **Qwen3.8-27B** (Aug 14, 2026, Apache 2.0): 27B dense, **text+image+video
  in**, 262K native (→1M YaRN), official FP8 (~31GB) — fits one 80GB GPU
  on Modal. Strong agent/computer-use scores (family flagship scored 86.1
  OSWorld-Verified). The natural Qwen2.5-7B/14B successor for HSEA.
- **Qwen3-VL** (Sep–Oct 2025, Apache 2.0): dedicated VL line, 2B/4B/8B/32B
  dense + 30B-A3B/235B-A22B MoE, 256K→1M, OCR/GUI-agent/visual-coding
  features, Instruct + Thinking variants. 8B is the cheap VLM workhorse.
- **Qwen3.5** (Feb 2026, Apache 2.0 for open sizes): 0.8B–397B-A17B, vision
  in every size per one source. **Qwen3.8-Max** (2.4T, custom license w/
  revenue thresholds, weights text-only) and **Qwen3.8-Flash-Next**
  (Aug 26, 2026, 125B/6B-active experimental Qwen4-arch preview,
  intentionally under-trained) are NOT recommended targets. **Qwen4**:
  in training per Sep 22, 2026 statement — no weights, no repo, no date.

### Mistral
- **Mistral Large 3** (Dec 2, 2025): 675B/41B MoE, 256K, image input, base
  + instruct under **Apache 2.0**. Largest Apache-2.0 Mistral; API
  deprecated Oct 31, 2026 (retires Apr 2027) but weights persist.
- **Mistral Large 4** (preview Oct 6, 2026; **open weights Oct 29, 2026**):
  52B-active/1.05T MoE, 1M ctx, multimodal — but under a NEW **Mistral
  Ecosystem Open-Weight License** (single-source claim — verify text
  before use). Too big for Modal regardless; HFInf/API evals only.
- **Mistral Small 4** (Mar 2026, Apache 2.0, multimodal) and **Ministral 3**
  3B/8B/14B (Dec 2025, Apache 2.0) cover small; exact Small 4 param counts
  conflict across sources — verify on HF.

### DeepSeek
- **V4 family** (preview Apr 24, 2026; GA Aug 2026; MIT): V4-Pro
  (1.6T/49B, 1M ctx, **text-only**) + V4-Flash → **V4.1-Flash**
  (Sep 10 / Oct 17, 2026, **has vision**). No V5, no R2 (R2 never shipped
  — widely expected reasoning successor, still vapor as of Oct 2026), no
  V4.1-Pro. Too big to self-host; API/HFInf frontier baselines only.

### Moonshot (Kimi)
- **K2.6** (Apr 20, 2026): 1T/32B MoE, native multimodal (400M MoonViT),
  256K, modified MIT, 300-agent swarm. **K3** (Jul 2026, claimed 2.8T,
  "largest open weights ever") — weights slated Jul 27 but details thin
  **[mostly UNVERIFIED]**. Caution: Moonshot closes K2 APIs to new users
  Oct 31, 2026 and retires all K2 models Dec 2026 — evals must use
  self-hosted/API-alternative weights, not the K2 API.

### Microsoft — Phi (small specialist)
- No Phi-5 (multiple sources debunk as of Oct 2026). Current: **Phi-4
  family**, MIT: Phi-4 14B (16K), Phi-4-mini 3.8B (128K), Phi-4-multimodal
  5.6B, Phi-4-reasoning/plus 14B, and newest **Phi-4-reasoning-vision-15B**
  (Mar 4, 2026, vision+reasoning, 16K ctx — tight for long tool-calling
  transcripts). All Modal-friendly.

### Hugging Face Smol stack (edge specialist)
- **SmolLM3-3B** (Jul 2025, hybrid reasoning, multilingual, long ctx),
  **SmolVLM2-2.2B** (Feb 2025 — still current, no SmolVLM3), plus brand-new
  **SmolVLM-256M/500M-Instruct** (Nov 2026, "world's smallest VLM").
  License + ctx verified on card 2026-11-10 (issue #4): SmolLM3-3B and
  SmolVLM-256M-Instruct are Apache-2.0 (64K trained / 8K ctx); 500M
  still unverified. Extreme-edge floor for the bench.

### OpenAI / Meta / others
- **gpt-oss-120b/20b** (Aug 5, 2025, Apache 2.0, 128K, text-only MoE
  reasoning; native MXFP4 — 120b runs on 1×80GB): no successor as of Nov
  3, 2026. Solid reasoning/text baseline; no vision.
- **Meta**: newest open Llama generation is still **Llama 4**
  (Apr 2025: Scout 109B/10M ctx, Maverick ~400B/1M ctx; Llama Community
  License). No Llama 5. 2026 frontier (Muse Spark/Wonder/Glimmer) is a
  separate line; Glimmer 30B (Apache 2.0, Aug 2026) is a watch item
  **[serving path UNVERIFIED]**. Note: Llama HF repos are gated (401) —
  same Modal blocker documented in `docs/inference.md`.
- **InternVL3.5** (OpenGVLab, ~Mar 2026): 1B–38B dense + to 241B-A28B
  sparse, strong VLM quality, but license is **checkpoint-dependent**
  (MIT/Apache 2.0/inherited) — diligence per checkpoint before adopting.
- Not surveyed in depth (watch): Zhipu GLM-5.x, Liquid LFM2, IBM Granite
  Vision, StepFun, MiniMax M2.x.

## 3. Recommended shortlist (top 6)

Tied to: (a) Lexsi Cohere+HF mandate, (b) drone camera-frame evals need
vision + tool calling + long tool transcripts, (c) Modal vLLM reality
(single/few GPUs, ungated HF repos only), (d) finetune candidacy (open
data/recipes help).

1. **Cohere Command A+** — *frontier VLM + Lexsi anchor (mandatory Cohere
   pick).* Only frontier-class open model that is BOTH Cohere (Lexsi
   requirement) and Apache 2.0 (no commercial/licensing friction), with
   native vision + reasoning + tool use + structured outputs — exactly the
   drone-navigator skill stack. Serve via HF Inference/new API segment
   first (verify gating + vLLM/tool-parser support); W4A4 Modal self-host
   on 2×H100 is Cohere-stated. Role: benchmark frontier ceiling.
2. **Qwen3.8-27B** — *main Modal workhorse + top finetune candidate.*
   Drop-in successor to Qwen2.5-7B/14B: Apache 2.0, dense 27B, official
   FP8 (~31GB, one 80GB GPU), text+image+video, 262K→1M ctx for long
   mission transcripts, strongest agent scores in its class. Role:
   primary inspect-eval model + SFT/RLVR base.
3. **Ai2 Molmo2-8B** — *grounding VLM + second finetune candidate.*
   Only shortlisted VLM with pointing/grounding + multi-frame object
   tracking — maps directly onto "find the survivor in the drone feed"
   evals. Apache 2.0 weights, 8B Modal-friendly, full training code + 9M
   examples published (finetune-friendly; mind NC-only subsets). Caveat:
   vLLM needs nightly build; verify tool-calling behavior. Role:
   vision-grounding specialist + finetune base.
4. **Google Gemma 4 12B** — *permissive single-GPU multimodal baseline.*
   Apache 2.0, encoder-free text+image+video+audio, 256K ctx, 140+
   languages (multilingual disaster zones), runs on one desktop GPU.
   Cleanest license in class; good "Gemma lane" to complement Qwen lane.
   Role: benchmark baseline + alt finetune base.
5. **Cohere Tiny Aya 3.35B** — *small/edge Cohere pick (Lexsi breadth).*
   Second Cohere model satisfying "Cohere + small" simultaneously:
   laptop-class, 70+ languages (multilingual first-responder scenarios),
   HF-hosted. Caveats: CC-BY-NC (research-only — fine for bench, flag for
   product), text-only (no camera frames), 8K ctx. Role: small-model
   floor + Cohere multilingual baseline.
6. **SmolVLM-256M-Instruct** — *extreme-edge VLM floor.* Nov 2026, fits
   anywhere (edge NPUs, cheap GPUs), establishes the minimum viable
   vision-language score for drone evals. License (Apache-2.0) + ctx (8K)
   verified on card 2026-11-10 (issue #4).
   Role: edge baseline (pairs with Tiny Aya: text floor vs vision floor).
   - Runner-up swap: **Phi-4-reasoning-vision-15B** (MIT, reasoning+vision,
     Modal-friendly) if the bench wants a mid-size reasoning VLM instead
     of an extreme-edge point; or **Qwen3-VL-8B** for a cheap Apache-2.0
     VLM workhorse.

Coverage check: ≥1 Cohere ✓ (×2: A+, Tiny Aya), ≥1 small/edge ✓ (×2: Tiny
Aya, SmolVLM-256M), ≥1 frontier VLM ✓ (Command A+, + API-only DeepSeek
V4.1-Flash / Mistral Large 3 as optional frontier reference points).

## 4. Serving-path implications for HSEA

- Baseten: no change possible (scope lock). New frontier models (A+,
  V4.1-Flash, Large 3/4, K2.6) need a **new inspect-ai service segment**
  (e.g. `openai-api/hf/...` via HF Inference Providers, or vendor APIs)
  following the existing `hsea.models` convention.
- Modal vLLM (fits today): Qwen3.8-27B-FP8, Qwen3-VL-8B/32B, Molmo2-8B
  (nightly), Gemma 4 12B/E2B, Aya Vision 8B, Tiny Aya, Phi-4RV-15B,
  gpt-oss-20b, SmolLM3/SmolVLM — one server file per model per current
  convention. Prefer **ungated** repos (Llama gating burned us before;
  Command A non-plus is access-request gated — verify A+ gating).
- Tool calling + vision must be re-verified per slug (per
  `docs/inference.md` gotchas: `--enable-auto-tool-choice` + matching
  `--tool-call-parser`, vLLM pin 0.21.0/CUDA 12.9, temp=0 for some).

## 5. Key sources

- Cohere Command A+: cohere.com/blog/command-a-plus (via worldngayon.com
  roundup, Sep 2026); yfarmx.com/ai/llms/command-a-plus (May 20, 2026,
  `command-a-plus-05-2026`); systems-analysis.info (218B/25B, 128K/64K);
  aiforsocialgood.ca (Apache 2.0 vs CC-BY-NC split); aiinsightlab.cloud
  memo (May 24, 2026 Apache 2.0 release).
- Command A: huggingface.co/CohereLabs/c4ai-command-a-03-2025 model card
  (111B, 256K, CC-BY-NC); spheron.network deploy guide.
- Aya Vision: huggingface.co/CohereLabs/aya-vision-8b model card (CC-BY-NC,
  16K); aibase.com/news/16021.
- Tiny Aya: techcrunch.com 2026-02-17 (3.35B, 70+ langs); cohere.com
  models-overview (CC-BY-NC-4.0); turingpost.com SLM survey (8K ctx);
  huggingface.co/CohereLabs/tiny-aya-earth; arxiv 2603.11510; bytewatchr
  (L2 Thinker Oct 6, 2026, 32K, CC-BY-NC).
- Qwen3.8: thinkfacility.com (Aug 12–26 2026 releases, 27B Apache 2.0);
  reviewramble.com qwen-3-8-27b (262K→1M, FP8); openllmstack.com
  (Max custom license); llmcheck.net (Qwen4 in training, no weights).
- Qwen3-VL: aifoss.dev VLM guide Jun 2026; localaimaster.com (Nov 2, 2026);
  vibecoderjournal.com (Sep–Oct 2025, Apache 2.0);
  aichina.news (32B-Thinking Apr 2, 2026).
- Molmo2: reviewramble.com molmo2-8b (Dec 11, 2025, Apache 2.0 weights,
  NC datasets); openrouter.ai/allenai/molmo-2-8b (37K ctx);
  huggingface.co/docs/transformers molmo2; therundown.ai (license layers).
- OLMo 3/3.1: aillistingjungle.com/llms/olmo (Nov 2025, 7B/32B, 64K);
  hpcwire.com Jun 2026; unite.ai Olmo-core 3 (Oct 1, 2026).
- Gemma 4: huggingface.co/google/gemma-4-* READMEs (5 sizes, multimodal,
  128K/256K); lilachbullock.com (Apr 2, 2026, Apache 2.0); cognativ.com
  (Gemma ToU → Apache 2.0 transition); tech-insider.org (Oct 31, 2026).
- Mistral: mungomash.com/ai/mistral/versions (Large 4 Oct 29, 2026 open
  weights, new license); mixroute.ai; ai-solutions.wiki (Large 3 Apache
  2.0); llmconfigurator.com (Nov 3, 2026).
- DeepSeek: geotoolbox.ai/blog/deepseek-v4 (V4.1-Flash Sep 10/Oct 17,
  MIT); deepseekai.guide roadmap (Oct 24, 2026); layer3labs.io (R2
  unreleased); theplanettools.ai (V4-Pro 1.6T/49B, 1M).
- Kimi: guidetoartificialintelligence.com (K2→K2.5→K2.6 timeline, mod
  MIT); evermx.com (K2.6 Apr 20, 2026); layer3labs.io (K3 Jul 2026, 2.8T
  claimed); topfive.ai (K2 API retirement Dec 2026).
- Phi: frankx.ai (Phi-4 family, reasoning-vision-15B Mar 4, 2026, no
  Phi-5); innfactory.ai (Oct 28, 2026); llmcheck.net (no Phi-5/Llama-5/
  Qwen-4.1 weights, Oct 24, 2026).
- Smol: singularitybyte.com (SmolLM3/SmolVLM2 currency, Apr 2026);
  hackster.io (SmolVLM-256M/500M, Nov 7, 2026); gadgets360 (Nov 9, 2026).
- gpt-oss: axiomi.si (no successor as of Nov 3, 2026); localaimaster.com
  (128K, MXFP4); innfactory.ai (117B/5.1B, no vision).
- Llama/Meta: llama.online (Llama 4 newest, Oct 2026); aicomparison.ai
  (Scout 109B/10M, Maverick ~400B/1M); codersera.com (Glimmer 30B
  Apache 2.0, Aug 10, 2026); coursiv.io (no Llama 5).
- InternVL: turingpost.com 10-open-mllms (3.5, checkpoint zoo, Aug 2026);
  mixpeek.com (Nov 6, 2026); forasoft.com (MIT-style).

## 6. Could not verify / omissions

- Command A+ **gating + vLLM/tool-parser support** — verify on HF
  before planning Modal serve. (Repo id
  `CohereLabs/command-a-plus-05-2026` and W4A4-on-2×H100 confirmed via
  hokai.io review + spheron.network citing cohere.com/blog.)
- Mistral Large 4 **license text** ("Mistral Ecosystem Open-Weight
  License" — one source) and Mistral Small 4 **exact param counts** and
  context (sources conflict).
- Kimi **K3** weights/license/specs (thin sourcing); K2.6 256K + mod-MIT
  are multi-sourced.
- DeepSeek V4.1-Flash param counts + context (V4-Pro 1.6T/49B/1M/MIT is
  solid; Flash specifics thin).
- SmolVLM-500M license + context (256M + SmolLM3 resolved 2026-11-10,
  issue #4: both Apache-2.0, 8K / 64K); InternVL3.5 per-checkpoint
  licenses; Gemma 4 QAT quant availability on Modal (GGUF/QAT quants
  exist via Unsloth/Ollama — vLLM path needs a check).
- Llama 4.5 (one source claims Jun 2026; contradicted by several "no new
  Llama generation" sources) — treated as nonexistent.
- GLM-5.x, Liquid LFM2, Granite Vision, StepFun, MiniMax, Muse Glimmer
  serving: not researched — watch items only.
