# Figure provenance for `docs/alignment-methods-posttraining.md`

## Original figures (vendored, HSEA-authored)

Rendered by [render_figs.py](render_figs.py) (matplotlib, deterministic,
no network). Original redraws for HSEA data (flywheel-v2 / dispatch /
abstention); not copies of any published figure.

| File | Explains | Note section |
|---|---|---|
| `fig-rlvr-safety.png` | E-RLVR pilot: prompts → group sampling → deterministic verifier (a–d) → GRPO → gates | §3.1 |
| `fig-preference-map.png` | Preference-stage decision map: data regime → DPO/SimPO/KTO/ORPO (+IPO arm) | §8.5 |
| `fig-testtime-verify.png` | Compute-optimal test-time routing: greedy → verifier gate → best-of-8 / abstain-escalate | §6.1 |

Re-render: `MPLCONFIGDIR=/tmp .venv/bin/python
docs/alignment-methods-posttraining/render_figs.py`.

## Published figures (LINKED only, never vendored)

Copyrighted paper figures are not vendored. Open each link and look at
the named figure/table:

| Paper | Figure | What to look at | License |
|---|---|---|---|
| [DPO](https://ar5iv.labs.arxiv.org/html/2305.18290) | Fig 1 (teaser) | Two-stage RM+RL pipeline vs DPO's single classification loss | arXiv perpetual — link only |
| [SimPO](https://arxiv.org/html/2405.14734v3) | Fig 1 | Shaded-box reward difference (ref-ratio vs avg logprob) + AlpacaEval/Arena-Hard bars | arXiv perpetual — link only |
| [Tülu 3](https://arxiv.org/html/2411.15124v3) | §6 + pipeline fig | SFT→DPO→RLVR stage diagram; RLVR GSM8K/MATH/IFEval gains | arXiv perpetual — link only |
| [DeepSeek-R1](https://arxiv.org/html/2501.12948v1) | Fig 1, §2.2.4 | Benchmark bars vs o1; R1-Zero aha-moment trace | arXiv perpetual — link only |
| [DeepSeekMath/GRPO](https://ar5iv.labs.arxiv.org/html/2402.03300) | §4 + Fig 1 | Group-sampling objective (no critic); MATH bars | arXiv perpetual — link only |
| [Snell et al.](https://arxiv.org/html/2408.03314v1) | Scaling figs | Compute-optimal curves; 14×-model parity point | arXiv perpetual — link only |
| [s1](https://arxiv.org/html/2501.19393v3) | Budget-forcing figs | "Wait"-extension scaling slope on AIME24 | arXiv perpetual — link only |
| [Search-R1](https://arxiv.org/html/2503.09516v1) | Framework fig | Multi-turn rollout with retrieved-token masking | arXiv perpetual — link only |
| [SWE-RL](https://arxiv.org/html/2502.18449v1) | §2 + results | Rule-reward RL on evolution data; 41.0% Verified | arXiv perpetual — link only |
| [WildTeaming](https://arxiv.org/html/2406.18510v1) | §3–§4 | Tactic-cluster mining → 262K 4-type schema | arXiv perpetual — link only |

## CC-BY papers (eligible to vendor in future; currently linked)

These two papers carry an explicit CC BY 4.0 license on their arXiv
HTML page, so their figures MAY be vendored with an attribution line +
source link. This survey vends none (the originals above cover what
HSEA needs); a follow-up may vendor:

| Paper | Candidate figure | License |
|---|---|---|
| [XSTest](https://arxiv.org/html/2308.01263v3) | Table 1: ten safe prompt types + unsafe contrasts | CC BY 4.0 (per arXiv page) |
| [ToRL](https://arxiv.org/html/2503.23383v1) | Framework + code/pass-ratio training curves | CC BY 4.0 (per arXiv page) |

If vendored later, add to this file: filename, source link, and the
line `© <authors>, <year>, CC BY 4.0, via <source link>`.
