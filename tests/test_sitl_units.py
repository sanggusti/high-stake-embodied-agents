"""Pure unit tests for SITL bridge helpers (no Docker needed)."""

import pytest

from dmb.sim.bridge import _offset_latlon, _offset_meters
from dmb.solvers.mission import _bridge_for_backend
from dmb.sim.bridge import GazeboSITLBridge, MockBridge


def test_latlon_offset_roundtrip():
    lat0, lon0 = -35.3632621, 149.1652374
    for east, north in [(0, 0), (25, 0), (0, 45), (-12.5, 33.3), (50, 50)]:
        lat, lon = _offset_latlon(lat0, lon0, east, north)
        east2, north2 = _offset_meters(lat0, lon0, lat, lon)
        assert abs(east2 - east) < 1e-6
        assert abs(north2 - north) < 1e-6


def test_latlon_offset_scale():
    # ~111 m per 0.001 deg latitude
    lat, _ = _offset_latlon(0.0, 0.0, 0.0, 111.195)
    assert abs(lat - 0.001) < 1e-5


def test_backend_selection():
    assert _bridge_for_backend("mock") is MockBridge
    assert _bridge_for_backend("gazebo-sitl") is GazeboSITLBridge
    with pytest.raises(ValueError):
        _bridge_for_backend("colosseum")  # not via this switch yet


def test_docker_args_mount_record_by_default(tmp_path, monkeypatch):
    monkeypatch.setenv("DMB_L3_DIR", str(tmp_path / "sim"))
    monkeypatch.delenv("DMB_L3", raising=False)
    bridge = GazeboSITLBridge.__new__(GazeboSITLBridge)
    bridge._partition = "dmb-test"
    bridge._sample_id = "s-1"
    bridge.image = "dmb/sim:track1"
    args = GazeboSITLBridge._docker_run_args(bridge)
    assert "-v" in args and "DMB_GZ_RECORD_PATH=/record" in args
    assert (tmp_path / "sim" / "s-1").exists()


def test_docker_args_skip_record_when_l3_off(tmp_path, monkeypatch):
    monkeypatch.setenv("DMB_L3", "off")
    bridge = GazeboSITLBridge.__new__(GazeboSITLBridge)
    bridge._partition = "dmb-test"
    bridge.image = "dmb/sim:track1"
    args = GazeboSITLBridge._docker_run_args(bridge)
    assert "-v" not in args
    assert "DMB_GZ_RECORD_PATH=/record" not in args
