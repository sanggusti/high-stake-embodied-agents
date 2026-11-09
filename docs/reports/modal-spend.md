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
TOTAL 0.50
```
