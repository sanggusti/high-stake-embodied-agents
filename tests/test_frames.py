"""Frame pipeline: rendering, honesty (no-leak), overlay, messages."""

import io

from PIL import Image

from dmb.perception.edge import EdgeReport
from dmb.perception.frames import (
    FOUND_COLOR,
    overlay_telemetry,
    render_map_frame,
    render_sensor_frame,
    to_data_url,
)
from dmb.scenarios import build_samples
from dmb.sim.bridge import MockBridge
from dmb.tools.drone import edge_report_message
from inspect_ai.tool import ContentImage, ContentText


def _state():
    bridge = MockBridge.from_metadata(dict(build_samples("forest_fire", seeds=[1])[0].metadata))
    bridge.act("arm", {})
    bridge.act("takeoff", {"altitude_m": 10})
    return bridge


def _has_color(png: bytes, color: tuple[int, int, int]) -> bool:
    img = Image.open(io.BytesIO(png)).convert("RGB")
    colors = img.getcolors(maxcolors=1 << 20)
    assert colors is not None
    return any(px == color for _, px in colors)


def test_map_frame_is_valid_png_with_caption():
    bridge = _state()
    png = bridge.capture_frame("t1")
    assert png is not None
    img = Image.open(io.BytesIO(png))
    assert img.size == (10 * 48, 10 * 48 + 28)


def test_map_frame_leaks_nothing_unfound():
    bridge = _state()
    png = bridge.capture_frame("t2")
    assert png is not None
    assert not _has_color(png, FOUND_COLOR)


def test_map_frame_shows_found_victims():
    bridge = _state()
    victim = bridge.victims[0]
    bridge.act("goto", {"x": victim.x, "y": victim.y})
    bridge.mark_found(victim)
    png = bridge.capture_frame("t3")
    assert png is not None
    assert _has_color(png, FOUND_COLOR)


def test_sensor_frame_draws_only_in_range_contacts():
    bridge = _state()
    grid = {"width": bridge.width, "height": bridge.height}
    near = render_sensor_frame(bridge.get_state(), grid, radius=3.0)
    far = render_sensor_frame(bridge.get_state(), grid, radius=300.0)
    assert near != far  # horizon actually filters
    Image.open(io.BytesIO(near)).verify()


def test_overlay_keeps_valid_png():
    bridge = _state()
    png = bridge.capture_frame("t4")
    assert png is not None
    stamped = overlay_telemetry(png, ["step=1 phase=nominal"])
    assert stamped != png
    Image.open(io.BytesIO(stamped)).verify()


def test_edge_report_message_multimodal():
    report = EdgeReport(trigger="requested", step=1, summary="scan", frame_ref="f1")
    text_only = edge_report_message(report, None)
    assert isinstance(text_only, str) and "frame unavailable" in text_only
    bridge = _state()
    png = bridge.capture_frame("t5")
    assert png is not None
    multi = edge_report_message(report, png)
    assert isinstance(multi, list)
    assert isinstance(multi[0], ContentText)
    assert isinstance(multi[1], ContentImage)
    assert multi[1].image.startswith("data:image/png;base64,")
    assert multi[1].detail == "low"


def test_edge_report_message_text_mode(monkeypatch):
    from dmb.tools.drone import edge_report_message

    monkeypatch.setenv("DMB_VISION", "text")
    report = EdgeReport(trigger="requested", step=1, summary="scan", frame_ref="f1")
    bridge = _state()
    png = bridge.capture_frame("t6")
    assert png is not None
    out = edge_report_message(report, png)
    assert isinstance(out, str) and "text-only run" in out
