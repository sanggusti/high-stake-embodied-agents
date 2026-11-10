# TODO: first-finetune model card (OUT OF SCOPE for #17 partial)

No v2 finetune exists yet, so there is no model card to publish. Do
not upload this stub to the Hub.

When the finetune lands, fill a copy of
[model-template.md](model-template.md) with:

- [ ] Base id + rev (`Qwen/Qwen2.5-7B-Instruct` or successor, pinned)
- [ ] Data rev (`sanggusti/dmb-flywheel-v2` @ `b44237e…` or successor)
- [ ] LoRA/data config (r/alpha/dropout, targets, epochs, batch,
      lr/sched/warmup, precision, max_len, seed, hardware, steps, loss)
- [ ] Exact train commands + code commit
- [ ] Held-out delta table (`evals/flywheel_test.py`, seeds 101..110)
- [ ] Safety-gate table (HarmBench/AIR-Bench, AgentHarm+redteam,
      SafeAgentBench+abstention)
- [ ] Serving notes (tokenizer pin, `--enforce-eager`, parser, URL)
- [ ] HF model id + pinned revision; git tag `model-vX.Y`
- [ ] Publish weights + card to HF; update this stub into the real card
