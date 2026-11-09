"""Non-LM baseline agents: greedy coverage + random walk.

Pure policies over (bridge, edge) — no model calls — so they run fast,
deterministically, and provider-free. Used by the CI benchmark
(tests/test_baselines.py) and by thin inspect-ai solver wrappers for
task-level comparisons.
"""

from __future__ import annotations

import random

from inspect_ai.model import ChatMessageAssistant, ChatMessageTool
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.tool import ToolCall

from dmb.perception.edge import FrameBudget, StubEdgePerception, TriggerConfig
from dmb.perception.grounding import (
    grid_from_state,
    pose_from_state,
    project_pixel,
    survey_camera,
)
from dmb.perception.oracle import frame_size, oracle_seg_provider
from dmb.perception.sam import SAMEdgePerception
from dmb.perception.segmap import ASSUMED_CELL_M, render_route_map
from dmb.sim.bridge import MockBridge


def _step(bridge: MockBridge, edge: StubEdgePerception, action: str, params: dict | None = None):
    obs = bridge.act(action, params or {})
    edge.on_step(bridge.get_state()["steps"])
    return obs


def greedy_policy(
    bridge: MockBridge, edge: StubEdgePerception, *, stride: int = 3, max_moves: int | None = None
) -> None:
    """Lawnmower sweep: arm, takeoff, visit every stride-th cell, land."""
    _step(bridge, edge, "arm")
    _step(bridge, edge, "takeoff", {"altitude_m": 10})
    moves = 0
    for y in range(0, bridge.height, stride):
        xs = range(0, bridge.width, stride)
        if (y // stride) % 2 == 1:
            xs = reversed(list(xs))
        for x in xs:
            if bridge.get_state()["steps"] >= bridge.step_budget - 2:
                break
            if max_moves is not None and moves >= max_moves:
                break
            _step(bridge, edge, "goto", {"x": x, "y": y})
            moves += 1
        else:
            continue
        break
    _step(bridge, edge, "land")


def random_policy(
    bridge: MockBridge, edge: StubEdgePerception, *, rng: random.Random, max_moves: int = 20
) -> None:
    """Random legal moves, then land."""
    _step(bridge, edge, "arm")
    _step(bridge, edge, "takeoff", {"altitude_m": 10})
    for _ in range(max_moves):
        if bridge.get_state()["steps"] >= bridge.step_budget - 2:
            break
        _step(
            bridge, edge, "goto",
            {"x": rng.randrange(bridge.width), "y": rng.randrange(bridge.height)},
        )
    _step(bridge, edge, "land")


def greedy_fleet_policy(
    bridge: MockBridge, edge: StubEdgePerception, *, stride: int = 3, max_moves: int | None = None
) -> None:
    """Partitioned lawnmower: cells go to the nearest launch pad (Voronoi),
    each drone snakes its own cells, moves interleave round-robin so the
    step clock models parallel wall time."""
    n = len(bridge.drones)
    spots = [(d.x, d.y) for d in bridge.drones]
    for i in range(n):
        _step(bridge, edge, "arm", {"drone_idx": i})
        _step(bridge, edge, "takeoff", {"altitude_m": 10, "drone_idx": i})
    order: list[tuple[int, int]] = []
    for y in range(0, bridge.height, stride):
        xs = range(0, bridge.width, stride)
        if (y // stride) % 2 == 1:
            xs = reversed(list(xs))
        order.extend((x, y) for x in xs)

    def owner(cell: tuple[int, int]) -> int:
        best, best_d = 0, None
        for i, (sx, sy) in enumerate(spots):
            d = abs(cell[0] - sx) + abs(cell[1] - sy)
            if best_d is None or d < best_d:
                best, best_d = i, d
        return best

    queues: list[list[tuple[int, int]]] = [[] for _ in range(n)]
    for cell in order:
        queues[owner(cell)].append(cell)
    moves = 0
    while any(queues):
        if bridge.get_state()["steps"] >= bridge.step_budget - 2:
            break
        if max_moves is not None and moves >= max_moves:
            break
        for i in range(n):
            if not queues[i]:
                continue
            if bridge.get_state()["steps"] >= bridge.step_budget - 2:
                break
            if max_moves is not None and moves >= max_moves:
                break
            x, y = queues[i].pop(0)
            _step(bridge, edge, "goto", {"x": x, "y": y, "drone_idx": i})
            moves += 1
    for i in range(n):
        if bridge.get_state()["steps"] >= bridge.step_budget:
            break
        _step(bridge, edge, "land", {"drone_idx": i})


def _episode(metadata: dict, policy: str, *, stride: int = 3, max_moves: int | None = None) -> dict:
    bridge = MockBridge.from_metadata(dict(metadata))
    edge = StubEdgePerception(bridge, TriggerConfig(), FrameBudget(max_snapshots=8))
    if policy == "greedy":
        greedy_policy(bridge, edge, stride=stride, max_moves=max_moves)
    elif policy == "greedy_fleet":
        greedy_fleet_policy(bridge, edge, stride=stride, max_moves=max_moves)
    elif policy == "random":
        random_policy(bridge, edge, rng=random.Random(int(metadata.get("seed", 0))))
    else:
        raise ValueError(f"unknown policy: {policy}")
    return bridge.get_state()


def run_baseline(metadata: dict, policy: str, **kwargs) -> dict:
    """Run one policy episode; return final sim state for scorers."""
    return _episode(metadata, policy, **kwargs)


ORACLE_CELL_M = ASSUMED_CELL_M
ORACLE_ALTITUDE_M = 40.0


def _boom_segmenter(png: bytes, concept: str) -> str:
    raise AssertionError("oracle path must use seg_provider, never the segmenter")


def _oracle_grounding(bridge: MockBridge):
    """Pixel box center -> grid cell through the live survey stance."""

    def ground(record) -> tuple[int, int] | None:
        state = bridge.get_state()
        png = bridge.capture_frame("oracle-ground")
        if png is None:
            return None
        cam = survey_camera(*frame_size(png))
        box = record.box
        return project_pixel(
            (box.x1 + box.x2) / 2,
            (box.y1 + box.y2) / 2,
            cam,
            pose_from_state(state),
            grid_from_state(state, ORACLE_CELL_M),
        )

    return ground


def _safe_survey_point(bridge: MockBridge) -> tuple[int, int]:
    """First safe in-grid cell on axis rings around center (deterministic)."""
    from dmb.sim.bridge import danger_cells_from_state  # noqa: PLC0415

    danger = danger_cells_from_state(bridge.get_state())
    cx, cy = bridge.width // 2, bridge.height // 2
    ring = [(0, 0), (1, 0), (0, 1), (-1, 0), (0, -1)]
    for radius in range(0, max(bridge.width, bridge.height) + 1):
        for dx, dy in ring:
            x, y = cx + dx * radius, cy + dy * radius
            if 0 <= x < bridge.width and 0 <= y < bridge.height:
                if (x, y) not in danger:
                    return (x, y)
    return (0, 0)


def oracle_route_policy(bridge: MockBridge, edge: SAMEdgePerception) -> str:
    """One high survey scan, then route text; returns the route-map text."""
    _step(bridge, edge, "arm")
    _step(bridge, edge, "takeoff", {"altitude_m": ORACLE_ALTITUDE_M})
    survey = _safe_survey_point(bridge)
    _step(bridge, edge, "goto", {"x": survey[0], "y": survey[1]})
    report = edge.on_demand(bridge.get_state()["steps"])
    seg = edge.last_seg_map
    if seg is None or seg.frame_size is None:
        text = f"{report.summary} Route map: empty seg-map, no routes."
    else:
        text = render_route_map(seg, bridge.get_state(), ORACLE_CELL_M, True)
    _step(bridge, edge, "land")
    return text


def run_oracle_route(metadata: dict) -> tuple[dict, str]:
    """Run the oracle route episode; return (final state, route text)."""
    bridge = MockBridge.from_metadata(dict(metadata))
    edge = SAMEdgePerception(
        bridge,
        TriggerConfig(),
        FrameBudget(max_snapshots=8),
        _boom_segmenter,
        grounding=_oracle_grounding(bridge),
        seg_provider=oracle_seg_provider(bridge, ORACLE_CELL_M),
    )
    text = oracle_route_policy(bridge, edge)
    return bridge.get_state(), text


@solver
def heuristic_mission() -> Solver:
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        state.store.set("sim_state", _episode(dict(state.metadata), "greedy"))
        return state

    return solve


@solver
def random_mission() -> Solver:
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        state.store.set("sim_state", _episode(dict(state.metadata), "random"))
        return state

    return solve


def run_live_sam_route(metadata: dict) -> tuple[dict, str]:
    """Live SAM route episode; return (final state, route text).

    Same shape as the oracle run but segments real frames through the
    Meta API (costs ~$0.05: 3 scans x 7 concepts). On mock renders the
    seg-map is honestly empty; route quality needs native backends.
    """
    from dmb.perception.sam import max_scans_from_env, sam_from_env  # noqa: PLC0415

    bridge = MockBridge.from_metadata(dict(metadata))
    edge = SAMEdgePerception(
        bridge,
        TriggerConfig(),
        FrameBudget(max_snapshots=8),
        sam_from_env(),
        grounding=_oracle_grounding(bridge),
        max_scans=max_scans_from_env(),
    )
    text = oracle_route_policy(bridge, edge)
    return bridge.get_state(), text


@solver
def live_sam_route_mission() -> Solver:
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        final, text = run_live_sam_route(dict(state.metadata))
        call = ToolCall(
            id="live-sam-route-1", function="request_route_map", arguments={}
        )
        state.messages.append(
            ChatMessageAssistant(
                content="Requesting route map.", tool_calls=[call]
            )
        )
        state.messages.append(
            ChatMessageTool(
                content=text, tool_call_id=call.id, function="request_route_map"
            )
        )
        state.store.set("sim_state", final)
        return state

    return solve


@solver
def oracle_route_mission() -> Solver:
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        final, text = run_oracle_route(dict(state.metadata))
        call = ToolCall(
            id="oracle-route-1", function="request_route_map", arguments={}
        )
        state.messages.append(
            ChatMessageAssistant(
                content="Requesting route map.", tool_calls=[call]
            )
        )
        state.messages.append(
            ChatMessageTool(
                content=text, tool_call_id=call.id, function="request_route_map"
            )
        )
        state.store.set("sim_state", final)
        return state

    return solve
