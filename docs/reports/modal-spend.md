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
TOTAL 6.29
```
