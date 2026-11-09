"""Aggregation: registry rows -> markdown + CSV."""

import json

from dmb.aggregate import _load_rows, render_csv, render_markdown, uplift_markdown


def _write_run(root, run_id, model, samples):
    path = root / run_id
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(
        json.dumps({"run_id": run_id, "task": "t", "model": model, "scenario": "pack"})
    )
    (path / "scores.json").write_text(json.dumps({"samples": samples}))


def _sample(sid, scenario, composite):
    return {
        "sample_id": sid,
        "scenario": scenario,
        "seed": 1,
        "scores": {
            "rescue_recall": composite,
            "containment": composite,
            "efficiency": composite,
            "safety": 1.0,
            "composite": composite,
        },
        "error": None,
    }


def test_table_groups_and_averages(tmp_path):
    _write_run(
        tmp_path, "r1", "mA",
        [_sample("a-1", "forest_fire", 0.5), _sample("a-2", "forest_fire", 0.7)],
    )
    _write_run(tmp_path, "r2", "mA", [_sample("b-1", "tsunami", 1.0)])
    (tmp_path / "empty-dir").mkdir()  # no config/scores: skipped
    rows = _load_rows(tmp_path)
    assert len(rows) == 3
    md = render_markdown(rows)
    assert "| mA | forest_fire | 2 | 0.600±0.141 |" in md
    assert "| mA | tsunami | 1 | 1.000 |" in md


def test_csv_has_full_vectors(tmp_path):
    _write_run(tmp_path, "r1", "mA", [_sample("a-1", "forest_fire", 0.5)])
    csv_text = render_csv(_load_rows(tmp_path))
    header, body = csv_text.strip().splitlines()
    assert header.startswith("run_id,task,model,scenario,seed,sample_id,")
    assert body.startswith("r1,t,mA,forest_fire,1,a-1,")
    assert body.count("0.5") >= 4


def test_empty_registry_renders_header(tmp_path):
    assert render_markdown(_load_rows(tmp_path)).startswith("| model |")


def test_scores_from_l2_reproduces_breakdown(tmp_path):
    import json

    from dmb.aggregate import scores_from_l2

    path = tmp_path / "s.jsonl"
    path.write_text(
        '{"t": 0.0, "step": 0, "kind": "note", "event": "episode_start"}\n'
        '{"t": -1.0, "step": -1, "kind": "score", "composite": 0.75,'
        ' "rescue_recall": 1.0}\n'
    )
    assert scores_from_l2(path) == {"composite": 0.75, "rescue_recall": 1.0}


def _uplift_sample(sid, composite, reachable):
    sample = _sample(sid, "forest_fire", composite)
    sample["scores"]["routes_reachable_rate"] = reachable
    return sample


def test_uplift_paired_deltas(tmp_path):
    _write_run(tmp_path, "base", "m", [_uplift_sample("a-1", 0.5, 0.0)])
    _write_run(tmp_path, "sam", "m",
               [_uplift_sample("a-1", 0.8, 1.0), _uplift_sample("a-9", 1.0, 1.0)])
    text = uplift_markdown("base", "sam", tmp_path)
    assert "| composite | 0.500 | 0.800 | +0.300 | 1 |" in text
    assert "| routes_reachable_rate | 0.000 | 1.000 | +1.000 | 1 |" in text
    assert "spl_like" in text  # present but n/a without values


def test_uplift_no_pairs_fails_loudly(tmp_path):
    _write_run(tmp_path, "base", "m", [_sample("a-1", "forest_fire", 0.5)])
    _write_run(tmp_path, "sam", "m", [_sample("b-1", "forest_fire", 0.5)])
    try:
        uplift_markdown("base", "sam", tmp_path)
    except ValueError:
        pass
    else:
        raise AssertionError("unpaired runs must fail loudly")


def test_table_run_filter(tmp_path, monkeypatch):
    import sys

    from dmb.aggregate import main

    _write_run(
        tmp_path, "r1", "mA", [_sample("a-1", "forest_fire", 0.5)]
    )
    _write_run(tmp_path, "r2", "mB", [_sample("b-1", "forest_fire", 1.0)])
    out = tmp_path / "t.md"
    monkeypatch.setattr(
        sys, "argv",
        ["aggregate", "table", "--runs", str(tmp_path), "--out", str(out),
         "--run", "r2"],
    )
    main()
    text = out.read_text()
    assert "mB" in text and "mA" not in text
