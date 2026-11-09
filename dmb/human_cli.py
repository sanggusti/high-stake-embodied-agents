"""Interactive human-baseline driver: a person flies the mission blind.

Run: .venv/bin/python -m dmb.human_cli --scenario forest_fire --seed 1 --name <you>

Victim placements are never shown; the operator only sees telemetry and
edge reports — the same information the LM agents get. The session is
scored with the identical deterministic scorers and recorded in the run
registry tagged baseline:human:<name>.
"""

from __future__ import annotations

import argparse
import sys

from dmb.config import Settings
from dmb.perception.edge import FrameBudget, StubEdgePerception, TriggerConfig
from dmb.registry import RunRegistry
from dmb.scenarios import build_samples
from dmb.scorers.mission import compute_mission_scores
from dmb.sim.bridge import MockBridge

HELP = """commands:
  arm | takeoff <alt_m> | goto <x> <y> | loiter [sec] | land | rtl
  tele   show position/battery/steps/violations (no victim info)
  scan   request edge perception report (uses snapshot budget)
  submit end mission and score | quit abandon (still scored)
"""


def _edge_line(bridge: MockBridge, edge: StubEdgePerception) -> str | None:
    report = edge.on_step(int(bridge.get_state()["steps"]))
    if report is None:
        return None
    return f"[edge:{report.trigger}] {report.summary}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fly a blind human-baseline mission.")
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--name", required=True, help="operator name for baseline tag")
    args = parser.parse_args(argv)

    settings = Settings.from_env()
    sample = build_samples(args.scenario, seeds=[args.seed])[0]
    print(f"Mission: {args.scenario} seed {args.seed}")
    print(sample.input)
    print("Victim placements are hidden. Fly blind, then submit.")
    print(HELP)

    bridge = MockBridge.from_metadata(dict(sample.metadata))
    edge = StubEdgePerception(
        bridge,
        TriggerConfig(heartbeat_every_steps=settings.edge_report_interval_steps),
        FrameBudget(settings.edge_max_snapshots_per_episode),
    )
    transcript: list[str] = []
    tool_calls = 0

    def act(action: str, params: dict) -> None:
        nonlocal tool_calls
        obs = bridge.act(action, params)
        tool_calls += 1
        line = f"> {action} {params}\n{obs.text}"
        extra = _edge_line(bridge, edge)
        if extra:
            line += f"\n{extra}"
        print(line)
        transcript.append(line)

    while True:
        try:
            raw = input("dmb> ").strip()
        except EOFError:
            print("\n(quit)")
            break
        if not raw:
            continue
        parts = raw.split()
        cmd, rest = parts[0].lower(), parts[1:]
        transcript.append(f"dmb> {raw}")
        try:
            if cmd == "help":
                print(HELP)
            elif cmd == "arm":
                act("arm", {})
            elif cmd == "takeoff":
                act("takeoff", {"altitude_m": float(rest[0])})
            elif cmd == "goto":
                act("goto", {"x": int(rest[0]), "y": int(rest[1])})
            elif cmd == "loiter":
                act("loiter", {"seconds": int(rest[0]) if rest else 5})
            elif cmd == "land":
                act("land", {})
            elif cmd == "rtl":
                act("rtl", {})
            elif cmd == "tele":
                print(bridge.get_state()["drone"], bridge.get_state()["steps"])
            elif cmd == "scan":
                report = edge.on_demand(int(bridge.get_state()["steps"]))
                tool_calls += 1
                print(f"[edge:{report.trigger}] {report.summary}")
            elif cmd in ("submit", "quit"):
                break
            else:
                print(f"unknown command: {cmd} (try help)")
        except (IndexError, ValueError):
            print("bad arguments (try help)")
        except KeyboardInterrupt:
            print("\n(quit)")
            break

    final = bridge.get_state()
    scores = compute_mission_scores(final, tool_calls)
    print(f"scores: composite={scores['composite']:.3f} recall={scores['rescue_recall']:.3f} "
          f"safety={scores['safety']:.3f} coverage={scores['coverage']:.3f}")

    registry = RunRegistry(settings.runs_dir)
    run_id = f"baseline-human-{args.name}-{args.scenario}-seed{args.seed}"
    registry.start_run(
        run_id, task=f"{args.scenario}:human",
        model=f"human:{args.name}", scenario=args.scenario, seeds=[args.seed],
    )
    registry.log_event(run_id, {"kind": "transcript", "lines": transcript})
    registry.write_scores(run_id, scores)
    registry.finish_run(run_id)
    print(f"recorded {run_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
