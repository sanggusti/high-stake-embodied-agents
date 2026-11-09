"""inspect-ai tools: the LM-facing drone action surface.

Tools are factories bound to one episode's bridge + edge perception instance,
so every sample gets isolated simulator state. Docstrings and type
annotations are required by inspect-ai so the model knows how to call them.
"""

from __future__ import annotations

import os

from inspect_ai.tool import ContentImage, ContentText, Tool, tool

from dmb.perception.edge import EdgePerception
from dmb.perception.frames import to_data_url
from dmb.perception.sam import SAMEdgePerception
from dmb.perception.segmap import ASSUMED_CELL_M, render_route_map
from dmb.sim.bridge import SimBridge
from dmb.watch.recorder import NullWatcher, Watcher


def edge_report_message(report, image: bytes | None) -> str | list:
    """Text-only or multimodal (text + frame) tool result.

    ``DMB_VISION=text`` forces text-only results so text models (e.g. the
    flywheel teacher) can run the bench; the LM then navigates purely on
    edge reports, which is the intended edge-first architecture.
    """
    detections = ", ".join(f"{d.label}@({d.x},{d.y})" for d in report.detections)
    text = f"{report.summary} Detections: [{detections}]"
    if image is None:
        return text + (" [frame unavailable]" if report.frame_ref else "")
    if os.environ.get("DMB_VISION", "").lower() == "text":
        return text + " [frame withheld: text-only run]"
    return [
        ContentText(text=text),
        ContentImage(image=to_data_url(image), detail="low"),
    ]


def make_drone_tools(
    bridge: SimBridge, edge: EdgePerception, watcher: Watcher | None = None
) -> list[Tool]:
    tools: list[Tool] = []
    watch: Watcher = watcher or NullWatcher()

    def _clock(idx: int = 0) -> tuple[float, int, int, int]:
        state = bridge.get_state()
        step = int(state.get("steps", 0))
        drones = state.get("drones") or [state.get("drone", {})]
        drone = drones[idx] if 0 <= idx < len(drones) else state.get("drone", {})
        return (
            float(state.get("sim_time", step * 5.0)),
            step,
            int(drone.get("x", 0)),
            int(drone.get("y", 0)),
        )

    def _edge_event(report) -> None:
        t, step, _x, _y = _clock()
        watch.on_event(
            "edge",
            t,
            step,
            trigger=report.trigger,
            summary=report.summary,
            frame_ref=report.frame_ref,
            detections=[
                {"label": d.label, "x": d.x, "y": d.y} for d in report.detections
            ],
        )

    def _with_edge_report(obs_text: str) -> str:
        state = bridge.get_state()
        report = edge.on_step(int(state.get("steps", 0)))
        if report is not None:
            _edge_event(report)
            return f"{obs_text}\n[edge:{report.trigger}] {report.summary}"
        return obs_text

    def _action(tool: str, args: dict, obs_text: str, idx: int = 0) -> None:
        t, step, x, y = _clock(idx)
        watch.on_event(
            "action", t, step, tool=tool, args=args, obs=obs_text[:500],
            pos=[x, y],
        )

    @tool
    def arm():
        async def execute(drone_idx: int) -> str:
            """Arm the drone motors.

            Args:
                drone_idx: Fleet drone to command (0-based; 0 when solo).

            Returns:
                Arm result message.
            """
            obs = bridge.act("arm", {"drone_idx": drone_idx}).text
            _action("arm", {"drone_idx": drone_idx}, obs, drone_idx)
            return obs

        return execute

    @tool
    def takeoff():
        async def execute(altitude_m: float, drone_idx: int) -> str:
            """Take off to an altitude and hover.

            Args:
                altitude_m: Target altitude in metres (must be positive).
                drone_idx: Fleet drone to command (0-based; 0 when solo).

            Returns:
                Takeoff result plus any edge perception report.
            """
            obs = bridge.act("takeoff", {"altitude_m": altitude_m, "drone_idx": drone_idx})
            _action("takeoff", {"altitude_m": altitude_m, "drone_idx": drone_idx}, obs.text, drone_idx)
            return _with_edge_report(obs.text)

        return execute

    @tool
    def goto_location():
        async def execute(x: int, y: int, drone_idx: int) -> str:
            """Fly to grid coordinates (x, y).

            Args:
                x: Target x coordinate (0-based).
                y: Target y coordinate (0-based).
                drone_idx: Fleet drone to command (0-based; 0 when solo).

            Returns:
                Movement result plus any edge perception report.
            """
            obs = bridge.act("goto", {"x": x, "y": y, "drone_idx": drone_idx})
            _action("goto_location", {"x": x, "y": y, "drone_idx": drone_idx}, obs.text, drone_idx)
            return _with_edge_report(obs.text)

        return execute

    @tool
    def loiter():
        async def execute(seconds: int, drone_idx: int) -> str:
            """Hold position to let edge perception scan the area.

            Args:
                seconds: How long to loiter.
                drone_idx: Fleet drone to command (0-based; 0 when solo).

            Returns:
                Loiter result plus any edge perception report.
            """
            obs = bridge.act("loiter", {"seconds": seconds, "drone_idx": drone_idx})
            _action("loiter", {"seconds": seconds, "drone_idx": drone_idx}, obs.text, drone_idx)
            return _with_edge_report(obs.text)

        return execute

    @tool
    def land():
        async def execute(drone_idx: int) -> str:
            """Land at the current position.

            Args:
                drone_idx: Fleet drone to command (0-based; 0 when solo).

            Returns:
                Landing result message.
            """
            obs = bridge.act("land", {"drone_idx": drone_idx}).text
            _action("land", {"drone_idx": drone_idx}, obs, drone_idx)
            return obs

        return execute

    @tool
    def return_to_launch():
        async def execute(drone_idx: int) -> str:
            """Abort the mission: fly home and land.

            Args:
                drone_idx: Fleet drone to command (0-based; 0 when solo).

            Returns:
                RTL result message.
            """
            obs = bridge.act("rtl", {"drone_idx": drone_idx}).text
            _action("return_to_launch", {"drone_idx": drone_idx}, obs, drone_idx)
            return obs

        return execute

    @tool
    def get_telemetry():
        async def execute(drone_idx: int) -> str:
            """Read position, altitude, battery, steps, and violations.

            Args:
                drone_idx: Fleet drone to command (0-based; 0 when solo).

            Returns:
                Telemetry summary string.
            """
            state = bridge.get_state()
            drones = state.get("drones") or [state["drone"]]
            drone = drones[drone_idx] if 0 <= drone_idx < len(drones) else state["drone"]
            text = (
                f"drone={drone_idx} pos=({drone['x']},{drone['y']}) alt={drone['altitude_m']}m "
                f"airborne={drone['airborne']} battery={drone['battery_pct']}% "
                f"steps={state['steps']}/{state['step_budget']} "
                f"violations={state['safety_violations']}"
            )
            if len(drones) > 1:
                others = " ".join(
                    f"#{i}@({d['x']},{d['y']}){'^' if d['airborne'] else 'v'} batt={d['battery_pct']}%"
                    for i, d in enumerate(drones) if i != drone_idx
                )
                text += f" fleet: {others}"
            _action("get_telemetry", {"drone_idx": drone_idx}, text, drone_idx)
            return text

        return execute

    @tool
    def request_edge_report():
        async def execute():
            """Request an on-demand edge perception scan (uses snapshot budget).

            Returns:
                Edge report with detections, snapshot status, and — while
                budget remains — the observation frame image.
            """
            report = edge.on_demand(int(bridge.get_state().get("steps", 0)))
            image = bridge.capture_frame(report.frame_ref) if report.frame_ref else None
            _edge_event(report)
            _action("request_edge_report", {}, report.summary)
            return edge_report_message(report, image)

        return execute

    @tool
    def request_route_map():
        async def execute():
            """Plan ground routes from the drone to each located victim.

            Captures a fresh segmentation map (consumes one snapshot of
            the episode budget), grounds road/hazard masks to grid cells,
            and routes around blockages. Only present with SAM
            perception; routes to unlocated victims are never given.

            Returns:
                Rescuer-facing route text: severity, blockages, and one
                waypoint path per located victim.
            """
            state = bridge.get_state()
            steps = int(state.get("steps", 0))
            report = edge.on_demand(steps)
            _edge_event(report)
            seg = edge.last_seg_map
            if seg is None or seg.frame_size is None:
                text = f"{report.summary} Route map: empty seg-map, no routes."
                _action("request_route_map", {}, text)
                return text
            cell_m = getattr(
                bridge, "grid_cell_m", getattr(bridge, "cell_m", None)
            )
            assumed = cell_m is None
            if assumed:
                cell_m = ASSUMED_CELL_M
            survey = getattr(bridge, "survey_camera", None)
            if survey is not None:
                text = render_route_map(
                    seg, bridge.get_state(), cell_m, False,
                    cam=survey.cam, pose=survey.pose,
                )
            else:
                text = render_route_map(seg, bridge.get_state(), cell_m, assumed)
            _action("request_route_map", {}, text)
            return text

        return execute

    factories = [
        arm,
        takeoff,
        goto_location,
        loiter,
        land,
        return_to_launch,
        get_telemetry,
        request_edge_report,
    ]
    if isinstance(edge, SAMEdgePerception):
        factories.append(request_route_map)
    for factory in factories:
        tools.append(factory())
    return tools
