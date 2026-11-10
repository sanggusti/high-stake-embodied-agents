# Modal + HF spend ledger (shared $50 cap)

Full mirror of `/tmp/modal-spend.log` (which is never committed).
Each row: `<issue> <run> <class> <minutes> <$>`; merge conflicts on
this file resolve by UNION of rows. HF Inference spend counts against
the same $50 cap as Modal GPU spend.

```
6 hf-smoke-v41flash api 0.03 $0.00
6 hf-ff1-v41flash api 2.0 $0.08
7 curate-v2 cpu 45 $0.00
21 abs-smoke-mock cpu 0.0 $0.00
21 abs-smoke-v41flash api 0.2 $0.00
21 abs-full-v41flash api 1.0 $0.07
21 abs-smoke-musespark api 0.2 $0.00
21 abs-full-musespark api 0.5 $0.00
14 safety14-probe-v41flash api 0.2 $0.01
14 safety14-probe-musespark api 0.2 $0.00
14 safety14-full-v41flash api 1.5 $0.18
14 safety14-full-musespark api 1.5 $0.00
22 rt-mock-seed0 cpu 0.1 $0.00
22 rt-v41flash-probe1 api 0.4 $0.04
22 rt-v41flash-probe2 api 0.3 $0.04
22 rt-v41flash-probe3 api 0.6 $0.04
22 rt-v41flash-seed0 api 0.6 $0.04
5 cmdaplus-probes api 0.1 $0.00
23 dora-access-probe api 0.1 $0.00
16 space-refresh cpu 0.2 $0.00
17 runbook-docs cpu 0.1 $0.00
25 partial-writeup cpu 0.2 $0.00
10 st-synth-fixtures cpu 0.0 $0.00
10 st-mock-mockllm cpu 0.1 $0.00
1 qwen38-probes a100-80gb 12.1 $0.50
1 qwen38-smoke a100-80gb 8.4 $0.35
1 qwen38-ff1-hermes a100-80gb 12.0 $0.50
1 qwen38-smoke2 a100-80gb 5.4 $0.23
1 qwen38-ff1 a100-80gb 12.7 $0.53
1 qwen38-video a100-80gb 4.1 $0.17
41 showcase-B cpu 0.5 $0.00
42 docs-realign cpu 0.2 $0.00
41 showcase-A cpu 0.5 $0.00
41 feedback cpu 0.5 $0.00
46 align-safety-survey cpu 0.5 $0.00
2 molmo2-a10g-probes a10g 27.0 $0.50
2 molmo2-coldstart1 a100-80gb 5.4 $0.22
2 molmo2-ground-tool-probes a100-80gb 9.0 $0.37
2 molmo2-ff1-nocall a100-80gb 0.3 $0.01
2 molmo2-coldstarts2-5 a100-80gb 14.0 $0.58
2 molmo2-ff1-stall a100-80gb 0.5 $0.02
2 molmo2-ff1-refusal a100-80gb 3.9 $0.16
2 molmo2-ff1-explore a100-80gb 0.6 $0.03
2 molmo2-ff1-full a100-80gb 0.4 $0.02
2 molmo2-smoke a100-80gb 0.2 $0.01
41 results-table cpu 0.5 $0.00
48 align-pt-survey cpu 0.5 $0.00
3 gemma4-coldstart a100-80gb 6.9 $0.29
3 gemma4-probes a100-80gb 5.2 $0.22
3 gemma4-smoke a100-80gb 4.8 $0.20
3 gemma4-ff1 a100-80gb 4.4 $0.18
3 qwen3vl-coldstart a100-80gb 4.5 $0.19
3 qwen3vl-probes a100-80gb 3.5 $0.15
3 qwen3vl-smoke a100-80gb 4.2 $0.17
3 qwen3vl-ff1-xml a100-80gb 0.4 $0.02
3 qwen3vl-hermes-coldstart a100-80gb 3.2 $0.13
3 qwen3vl-ff1 a100-80gb 0.9 $0.04
50 align-vlm-survey cpu 0.5 $0.00
51 align-index-synth cpu 0.5 $0.00
57 findings-narrative cpu 0.5 $0.00
59 tech-details cpu 0.5 $0.00
4 smollm3-coldstart l4 16.4 $0.22
4 smollm3-probes l4 26.9 $0.36
4 smollm3-smoke l4 23.5 $0.31
4 smollm3-ff1 l4 40.1 $0.53
4 tinyaya-coldstart l4 11.7 $0.16
4 tinyaya-smoke l4 0.1 $0.00
4 tinyaya-ff1-noparser l4 0.1 $0.00
4 tinyaya-rewarm l4 5.9 $0.08
4 tinyaya-smoke2 l4 29.3 $0.39
4 tinyaya-ff1 l4 7.4 $0.10
4 smolvlm-coldstart l4 8.1 $0.11
4 smolvlm-warm l4 20.7 $0.28
4 smolvlm-smoke l4 0.1 $0.00
4 smolvlm-smoke2 l4 0.1 $0.00
4 smolvlm-ff1 l4 0.4 $0.01
5 cmdaplus-coldstart1 h100-2x 21.4 $2.82
5 cmdaplus-probe-smoke-ff1a h100-2x 3.6 $0.47
5 cmdaplus-coldstart2 h100-2x 14.2 $1.87
5 cmdaplus-ff1 h100-2x 1.6 $0.21
62 judge-probes api 0.5 $0.00
62 judge-pilot-62 api 6.3 $0.02
61 metrics-playbook cpu 0.5 $0.00
TOTAL 14.23
```
