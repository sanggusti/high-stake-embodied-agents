"""Phase A benchmark: 100 seeded episodes, baselines, difficulty validation.

Provider-free and deterministic: greedy must beat random, and a
fixed-effort probe must score easy > medium > hard.
"""

import statistics
from collections import defaultdict

from dmb.scenarios import build_samples
from dmb.scorers.mission import compute_mission_scores
from dmb.solvers.baselines import run_baseline

SEEDS_100 = list(range(1, 101))


def _composites(policy: str, **kwargs) -> list[float]:
    out = []
    for sample in build_samples("forest_fire", seeds=SEEDS_100):
        state = run_baseline(dict(sample.metadata), policy, **kwargs)
        out.append(compute_mission_scores(state, 0)["composite"])
    return out


def test_greedy_beats_random_over_100_seeds():
    greedy = statistics.mean(_composites("greedy"))
    random = statistics.mean(_composites("random"))
    assert greedy > random + 0.1, f"greedy={greedy:.3f} random={random:.3f}"


def test_difficulty_bands_stratify_fixed_effort_probe():
    bands: dict[str, list[float]] = defaultdict(list)
    for sample in build_samples("forest_fire", seeds=SEEDS_100):
        md = dict(sample.metadata)
        recall = compute_mission_scores(
            run_baseline(md, "greedy", max_moves=10), 0
        )["rescue_recall"]
        bands[md["difficulty_score"]["band"]].append(recall)
    assert set(bands) == {"easy", "medium", "hard"}, dict(bands)
    means = {b: statistics.mean(v) for b, v in bands.items()}
    assert means["easy"] > means["medium"] > means["hard"], means
    assert means["easy"] - means["hard"] > 0.05, means


def test_difficulty_scores_are_deterministic():
    first = [s.metadata["difficulty_score"] for s in build_samples("tsunami", seeds=[1, 2, 3])]
    second = [s.metadata["difficulty_score"] for s in build_samples("tsunami", seeds=[1, 2, 3])]
    assert first == second
    assert all(set(d) == {"score", "band"} for d in first)
