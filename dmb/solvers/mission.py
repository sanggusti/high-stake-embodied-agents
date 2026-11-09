"""Shared mission solver: every drone-navigation task uses this.

Builds an isolated episode (bridge + edge perception + tools) from sample
metadata, runs a ReAct agent with a hard message cap, and stores final sim
state for the deterministic mission scorer.
"""

from inspect_ai.agent import react, run
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.tool import Tool
from inspect_ai.util import message_limit

from dmb.config import Settings
from dmb.perception.edge import FrameBudget, StubEdgePerception, TriggerConfig
from dmb.sim.assets import assert_mission_capable, load_drone_spec
from dmb.sim.bridge import GazeboSITLBridge, MockBridge
from dmb.tools.drone import make_drone_tools
from dmb.watch import watcher_from_env


def _bridge_for_backend(backend: str):
    if backend == "gazebo-sitl":
        return GazeboSITLBridge
    if backend == "mock":
        return MockBridge
    raise ValueError(f"Unknown DMB_SIM_BACKEND: {backend!r} (want mock|gazebo-sitl)")


@solver
def drone_mission() -> Solver:
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        settings = Settings.from_env()
        backend = settings.sim_backend
        spec_name = str(dict(state.metadata).get("drone_spec", "quad_iris_class"))
        assert_mission_capable(load_drone_spec(spec_name))
        md = dict(state.metadata)
        md["sample_id"] = str(state.sample_id)
        bridge = _bridge_for_backend(backend).from_metadata(md)
        tod = dict(state.metadata).get("time_of_day_active") or "day"
        perception = settings.edge_perception
        if perception == "stub":
            edge = StubEdgePerception(
                bridge,
                TriggerConfig(time_of_day=str(tod)),
                FrameBudget(max_snapshots=8),
            )
        elif perception == "sam":
            from dmb.perception.sam import (  # noqa: PLC0415
                SAMEdgePerception,
                hazard_point_proposer,
                ignition_point_proposer,
                max_scans_from_env,
                sam_from_env,
            )

            scenario = str(md.get("scenario", ""))
            if scenario == "earthquake_hard":
                proposer = hazard_point_proposer(bridge, "rubble")
            elif scenario == "landslide_hard":
                proposer = hazard_point_proposer(bridge, "debris")
            elif scenario in (
                "flood_hard",
                "flash_flood_hard",
                "tsunami_hard",
            ):
                proposer = hazard_point_proposer(bridge, "water")
            else:
                proposer = ignition_point_proposer(bridge)
            edge = SAMEdgePerception(
                bridge,
                TriggerConfig(time_of_day=str(tod)),
                FrameBudget(max_snapshots=8),
                sam_from_env(),
                max_scans=max_scans_from_env(),
                point_proposer=proposer,
                fuse_stub=settings.sam_fuse_stub,
            )
        else:
            raise ValueError(
                f"Unknown DMB_EDGE_PERCEPTION: {perception!r} (want stub|sam)"
            )
        watcher = watcher_from_env(str(state.sample_id))
        watcher.on_start({**dict(state.metadata), "backend": backend, "perception": perception})
        tools: list[Tool] = make_drone_tools(bridge, edge, watcher)
        fleet_size = int(dict(state.metadata).get("fleet_size", 1) or 1)
        prompt = (
            "You are a disaster-response drone navigator piloting a "
            "simulated drone through FUNCTION CALLS. Rules: act ONLY "
            "through function calls — plain text changes nothing in the "
            "simulator, so never narrate an action without emitting its "
            "matching function call in the same turn. Use the flight "
            "tools to search the area, rely on edge perception reports, "
            "minimise steps, stay out of hazard zones, then land and "
            "submit a summary of victim positions."
        )
        if fleet_size > 1:
            prompt += (
                f" You command a fleet of {fleet_size} drones from spread "
                "launch pads; every flight tool takes drone_idx (0-based). "
                "Split the search: give each drone its own area, interleave "
                "their moves, and land every drone you launched."
            )
        agent = react(prompt=prompt, tools=tools)
        # Hard cap: a model that never submits must still terminate.
        agent_state, _limit_error = await run(
            agent, state.messages, [message_limit(50)]
        )
        state.messages = agent_state.messages
        state.output = agent_state.output
        final = bridge.get_state()
        state.store.set("sim_state", final)
        watcher.on_end({"state": final})
        if perception == "sam" and getattr(edge, "last_seg_map", None) is not None:
            try:
                from dmb.perception.segmap import save_seg_artifacts  # noqa: PLC0415

                frame = bridge.capture_frame("final-segmap")
                if frame is not None:
                    save_seg_artifacts(
                        str(state.sample_id), frame, edge.last_seg_map
                    )
            except Exception:
                pass  # footage must never fail the episode
        bridge.close()
        return state

    return solve
