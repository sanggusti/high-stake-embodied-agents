"""Seg-map routing: costs, A*, blockages, rescuer text (no network)."""

from dmb.perception.grounding import (
    CameraModel,
    DronePose,
    GridSpec,
    project_mask,
)
from dmb.perception.sam import (
    DerivedBlockage,
    SAMBox,
    SAMMask,
    SAMRecord,
    SegMap,
    SeverityScore,
)
from dmb.perception.segmap import (
    COST_BLOCKED,
    COST_HAZARD,
    COST_OPEN,
    COST_ROAD,
    cost_grid,
    format_route_map,
    ground_concepts,
    plan_route,
    plan_routes,
)


def test_cost_precedence():
    grid = cost_grid(3, 3, {(0, 0), (1, 1)}, {(0, 0), (2, 2)}, {(0, 0)})
    assert grid[0][0] == COST_BLOCKED
    assert grid[2][2] == COST_HAZARD
    assert grid[1][1] == COST_ROAD
    assert grid[0][1] == COST_OPEN


def test_plan_route_open_grid():
    grid = cost_grid(3, 3, set(), set(), set())
    route = plan_route(grid, (0, 0), (2, 2))
    assert route.reachable and route.steps == 4
    assert route.waypoints[0] == (0, 0) and route.waypoints[-1] == (2, 2)
    for a, b in zip(route.waypoints, route.waypoints[1:]):
        assert abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1


def test_plan_route_detours_around_hazard_wall_gap():
    grid = cost_grid(3, 3, set(), {(1, 0), (1, 2)}, set())
    route = plan_route(grid, (0, 0), (2, 0))
    assert route.reachable
    assert (1, 1) in route.waypoints
    assert (1, 0) not in route.waypoints and (1, 2) not in route.waypoints


def test_plan_route_start_is_goal_and_out_of_bounds():
    grid = cost_grid(3, 3, set(), set(), set())
    same = plan_route(grid, (1, 1), (1, 1))
    assert same.reachable and same.waypoints == [(1, 1)] and same.cost == 0.0
    bad = plan_route(grid, (0, 0), (9, 9))
    assert not bad.reachable and bad.waypoints == []


def test_ground_concepts_blocked_is_road_meets_hazard():
    cells = {"road": {(0, 0), (1, 0)}, "water": {(1, 0)}, "smoke": set()}
    road, hazard, blocked = ground_concepts(cells.get)
    assert road == {(0, 0), (1, 0)}
    assert hazard == {(1, 0)}
    assert blocked == {(1, 0)}


def _seg():
    return SegMap(
        results={},
        blockages=[DerivedBlockage("flooded_road", "road", "water", 0.41, 12)],
        severity=SeverityScore(0.36, "severe"),
        frame_size=(100, 100),
    )


def test_format_route_map():
    grid = cost_grid(10, 10, set(), set(), set())
    routes = plan_routes(grid, (5, 5), [(7, 2), (9, 9)])
    text = format_route_map(_seg(), routes, (5, 5), 5.0, True)
    assert "assumed survey geometry" in text
    assert "severe(0.36)" in text and "flooded_road:0.41" in text
    assert "Victim (7,2): 5 steps, 25.0 m" in text
    assert "(5,5) -> " in text and " -> (7,2)." in text


def test_format_route_map_unreachable_and_no_victims():
    grid = cost_grid(10, 10, set(), set(), set())
    [lost] = plan_routes(grid, (5, 5), [(50, 50)])
    text = format_route_map(_seg(), [lost], (5, 5), None, False)
    assert "UNREACHABLE" in text and "calibrated geometry" in text
    assert "No located victims yet" in format_route_map(_seg(), [], (5, 5), 5.0, True)


def _rec(x1, y1, x2, y2, H=2, W=2):  # noqa: N803
    return SAMRecord(0, SAMBox(x1, y1, x2, y2, 100, 100, 0.9), SAMMask(H, W, "~", "~x"))


def test_grounding_to_costs_to_route_integration():
    """Real composition: synthetic road+water masks -> grounded cells -> route."""
    cam = CameraModel(100, 100, 90.0, 90.0, 0.0)
    pose = DronePose(5, 5, 25.0)
    spec = GridSpec(10, 10, 5.0)
    full = bytes([1, 1, 1, 1])
    road = project_mask(_rec(0, 40, 99, 59), full, cam, pose, spec)
    assert road == {(2, 4), (7, 4), (2, 5), (7, 5)}
    water = project_mask(_rec(60, 40, 79, 59), full, cam, pose, spec)
    assert water == {(6, 4), (7, 4), (6, 5), (7, 5)}
    _, _, blocked = ground_concepts({"road": road, "water": water}.get)
    assert blocked == {(7, 4), (7, 5)}
    grid = cost_grid(10, 10, road, water, blocked)
    assert grid[4][7] == COST_BLOCKED and grid[4][2] == COST_ROAD
    route = plan_route(grid, (0, 0), (9, 9))
    assert route.reachable
    assert not (set(route.waypoints) & blocked)


def _overlay_seg():
    from dmb.perception.sam import SegConceptResult

    rec = SAMRecord(
        object_id=0,
        box=SAMBox(10, 10, 20, 20, 100, 100, 0.9),
        mask=SAMMask(2, 2, "lossless", "~fake"),
    )
    seg = SegMap()
    seg.results["fire"] = SegConceptResult("fire", [rec], [bytes([1, 1, 1, 1])])
    seg.frame_size = (100, 100)
    seg.severity = SeverityScore(0.03, "minor")
    return seg


def test_render_seg_overlay_png_with_legend():
    import io

    from PIL import Image

    from dmb.perception.segmap import render_seg_overlay

    buf = io.BytesIO()
    Image.new("RGB", (100, 100), (10, 10, 10)).save(buf, format="PNG")
    out = render_seg_overlay(buf.getvalue(), _overlay_seg(), grid=20)
    img = Image.open(io.BytesIO(out))
    assert img.size == (100, 122)  # frame + legend strip
    assert out[:8] == b"\x89PNG\r\n\x1a\n"


def test_render_seg_overlay_registers_nonsquare():
    """Mask tint must land on the record box (no axis squash)."""
    import io

    from PIL import Image

    from dmb.perception.sam import SegConceptResult
    from dmb.perception.segmap import render_seg_overlay

    rec = SAMRecord(
        object_id=0,
        box=SAMBox(60, 30, 79, 49, 100, 60, 0.9),
        mask=SAMMask(2, 2, "lossless", "~fake"),
    )
    seg = SegMap()
    seg.results["fire"] = SegConceptResult("fire", [rec], [bytes([1, 1, 1, 1])])
    seg.frame_size = (100, 60)
    buf = io.BytesIO()
    Image.new("RGB", (100, 60), (10, 10, 10)).save(buf, format="PNG")
    img = Image.open(io.BytesIO(render_seg_overlay(buf.getvalue(), seg, grid=20)))
    r, g, b = img.getpixel((70, 40))  # inside the box: fire tint
    assert r > 100 and r > g + 30
    r0, g0, b0 = img.getpixel((10, 10))  # far outside: untouched
    assert (r0, g0, b0) == (10, 10, 10)


def test_render_route_map_calibrated_override():
    from dmb.perception.segmap import render_route_map

    seg = _overlay_seg()
    state = {"drone": {"x": 0, "y": 0}, "grid": {"width": 10, "height": 10},
             "victims": []}
    cam = CameraModel(100, 100, 60.0, 90.0, 0.0)
    pose = DronePose(0, 0, 50.0)
    text = render_route_map(seg, state, 5.0, False, cam=cam, pose=pose)
    assert "calibrated geometry" in text
    text2 = render_route_map(seg, state, 5.0, True)
    assert "assumed survey geometry" in text2
