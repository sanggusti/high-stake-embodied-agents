# WildJailbreak-methodology disaster mix (curator-written)

Upstream `allenai/wildjailbreak` (ODC-BY-1.0) is access-gated on the
Hub: approval needs a personal click-through, which agents cannot do
(per `docs/license-clearance-24.md` §3), so the upstream set is out
of scope until revisited. These 80 rows follow its methodology
(adversarial harmful + adversarial benign/lookalike pairs, disaster
framed) and are curator-written for issue #74 repair tuning.

- 40 disaster-misuse rows (label `refuse`): harmful requests wearing
  a disaster-relief pretext (fraud, phishing, weapons, sabotage,
  misinformation, theft, privacy, self-harm-adjacent).
- 40 lookalike rows (label `comply`): benign relief requests with
  trigger-word surface. Disjoint from the eval-only
  `data/xstest/lookalikes.jsonl` pack (asserted in tests).

Graduate to the upstream WildJailbreak train split once access is
granted; the composer takes the same row schema.
