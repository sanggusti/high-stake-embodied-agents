# Modal + HF spend ledger (shared $50 cap)

Full mirror of `/tmp/modal-spend.log` (which is never committed).
Each row: `<issue> <run> <class> <minutes> <$>`; merge conflicts on
this file resolve by UNION of rows. HF Inference spend counts against
the same $50 cap as Modal GPU spend.

```
6 hf-smoke-v41flash api 0.03 $0.00
6 hf-ff1-v41flash api 2.0 $0.08
TOTAL 0.08
```
