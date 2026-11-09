import json

from dmb.registry import RunRegistry


def test_run_lifecycle(tmp_path):
    registry = RunRegistry(tmp_path)
    registry.start_run(
        "run-1", task="forest_fire", model="mockllm/model",
        scenario="forest_fire", seeds=[1, 2], limits={"steps": 60},
    )
    registry.log_event("run-1", {"kind": "note", "text": "hello"})
    registry.write_scores("run-1", {"composite": 0.5})
    record = registry.finish_run("run-1", eval_logs=["logs/a.eval"])

    assert registry.list_runs() == ["run-1"]
    assert record.finished_at is not None
    config = json.loads((tmp_path / "run-1" / "config.json").read_text())
    assert config["eval_logs"] == ["logs/a.eval"]
    assert config["seeds"] == [1, 2]
    assert (tmp_path / "run-1" / "events.jsonl").read_text().strip()
