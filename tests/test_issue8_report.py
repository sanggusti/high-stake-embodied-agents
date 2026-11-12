"""Tests for issue-8 report rendering (fabricated data, tmp outputs)."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import issue8_report as rep


def _summary() -> dict:
    def run(metrics):
        return {"log": "logs/x.eval", "metrics": metrics}

    return {"runs": {
        "base/flywheel_test": run({"n": 20, "mean_composite": 0.5}),
        "ft/flywheel_test": run({"n": 20, "mean_composite": 0.6}),
        "base/floodnet_vqa": run({"n": 25, "mean_composite": 0.7,
                                  "by_kind": {"vqa": {"n": 25, "mean": 0.7}}}),
        "ft/floodnet_vqa": run({"n": 25, "mean_composite": 0.8,
                                "by_kind": {"vqa": {"n": 25, "mean": 0.8}}}),
        "base/ladi_v2": run({"n": 60, "mean_composite": 0.6,
                             "by_kind": {"severity": {"n": 60, "mean": 0.6}}}),
        "ft/ladi_v2": run({"n": 60, "mean_composite": 0.6,
                           "by_kind": {"severity": {"n": 60, "mean": 0.6}}}),
        "base/safety": run({"n": 4, "match_rate": 0.75,
                            "by_bench": {"harmbench": {"n": 2, "match": 1.0},
                                         "benign": {"n": 2, "match": 0.5}},
                            "refusal": {"harmbench": {"n": 2, "rate": 1.0},
                                        "airbench": {"n": 0, "rate": 0.0}}}),
        "ft/safety": run({"n": 4, "match_rate": 1.0,
                          "by_bench": {"harmbench": {"n": 2, "match": 1.0},
                                       "benign": {"n": 2, "match": 1.0}},
                          "refusal": {"harmbench": {"n": 2, "rate": 1.0},
                                      "airbench": {"n": 0, "rate": 0.0}}}),
        "base/redteam": run({"n": 10, "asr": 0.2, "needs_review": 1,
                             "by_family": {}}),
        "ft/redteam": run({"n": 10, "asr": 0.1, "needs_review": 0,
                           "by_family": {}}),
    }}


def _judges() -> dict:
    return {"T1": {"n": 4, "agreement": 1.0, "kappa": 1.0, "parse_fail": 0,
                   "by_model": {
                       "base": {"n": 2, "verdict_counts": {"correct": 2},
                                "mean_score": 5.0},
                       "ft": {"n": 2, "verdict_counts": {"correct": 1,
                                                        "wrong": 1},
                              "mean_score": 3.0}}},
            "T2": {"n": 1, "agreement": 1.0, "kappa": None, "parse_fail": 0,
                   "by_model": {"base": {"n": 1,
                                         "verdict_counts": {"refuse": 1},
                                         "mean_score": 4.0}}}}


def test_render_report_tables(tmp_path, monkeypatch):
    monkeypatch.setattr(rep, "REPORT_DIR", tmp_path)
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path))
    summary, judges = _summary(), _judges()
    rep.plot_before_after(summary)
    rep.plot_footage_kinds(summary, judges)
    rep.plot_safety_bench(summary, judges)
    rep.plot_judge_agreement(judges)
    for name in ("issue-8-before-after.png", "issue-8-footage-kinds.png",
                 "issue-8-safety-bench.png", "issue-8-judge-agreement.png"):
        assert (tmp_path / name).exists()
    md = rep.render_report(
        summary, judges,
        {"base": "Qwen/Qwen3.8-27B", "pairs": 8673, "tokens": 1980483,
         "truncated": 43, "train_loss": 1.23, "steps": 345, "minutes": 60.0,
         "hf_repo": "sanggusti/x", "hf_rev": "abc123"},
        {"estimated_cost": 0.3, "calls": 10, "prompt_tokens": 100,
         "completion_tokens": 50},
        {"runs": {"base/safety": {"log": "logs/a.eval"}}})
    assert "0.500 | 0.600 | +0.100 (up)" in md
    assert "Closes" not in md  # body text lives in the PR, not the report
    assert "issue-8-before-after.png" in md


def test_plot_loss_curves_mixed_legacy_and_reward(tmp_path, monkeypatch):
    monkeypatch.setattr(rep, "REPORT_DIR", tmp_path)
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path))
    trains = {
        "dpo": {"train_loss": 0.02, "steps": 875},  # legacy: marker only
        "safetune": {"train_loss": 0.3, "steps": 174,
                     "loss_curve": [{"step": 50, "loss": 0.9},
                                    {"step": 100, "loss": 0.5},
                                    {"step": 174, "loss": 0.3}]},
        "grpo": {"mean_reward": 0.6, "steps": 30,
                 "loss_curve": [{"step": 10, "loss": 0.4, "reward": 0.5},
                                {"step": 30, "loss": 0.2, "reward": 0.6}]},
    }
    path = rep.plot_loss_curves(trains)
    assert path.exists()
    assert path.stat().st_size > 0
