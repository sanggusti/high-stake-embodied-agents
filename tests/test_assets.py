import pytest

from dmb.sim.assets import assert_mission_capable, list_drone_specs, load_drone_spec


def test_specs_are_sourced_not_placeholders():
    names = list_drone_specs()
    assert {"quad_iris_class", "fixedwing_scout"} <= set(names)
    for name in ("quad_iris_class", "fixedwing_scout"):
        spec = load_drone_spec(name)
        assert spec.sourced, f"{name} still a placeholder"
        assert isinstance(spec.provenance, dict)
        assert {"params", "model", "docs"} <= set(spec.provenance)


def test_quad_matches_track1_reality():
    quad = load_drone_spec("quad_iris_class")
    assert quad.vehicle_class == "multicopter"
    assert quad.supports("guided_takeoff")
    assert quad.supports("guided_goto")
    assert not quad.supports("non_gps_nav")  # no flow sensor in the parm file
    assert "rangefinder" in quad.sensors
    assert "gimbal-camera" in quad.sensors
    assert quad.sitl_params["FRAME_CLASS"] == 1
    assert quad.sitl_params["FRAME_TYPE"] == 1
    assert quad.gazebo_model == "iris_with_gimbal"


def test_fixedwing_guided_limits():
    wing = load_drone_spec("fixedwing_scout")
    assert wing.vehicle_class == "fixed-wing"
    assert not wing.supports("guided_takeoff")  # Plane GUIDED is fly-to-point
    assert wing.supports("guided_goto")
    assert wing.supports("auto_mission")
    assert "pitot" not in wing.sensors  # ARSPD_TYPE 0 in the parm file


def test_mission_capability_enforcement():
    assert_mission_capable(load_drone_spec("quad_iris_class"))
    with pytest.raises(ValueError, match="guided_takeoff"):
        assert_mission_capable(load_drone_spec("fixedwing_scout"))


def test_backend_rejects_unsupported_vehicle_class():
    from dmb.scenarios import build_samples
    from dmb.sim.bridge import GazeboSITLBridge, MockBridge

    meta = dict(build_samples("forest_fire", seeds=[1])[0].metadata)
    MockBridge().reset(meta)  # quad on mock: fine
    meta["drone_spec"] = "fixedwing_scout"
    MockBridge().reset(meta)  # mock simulates both classes: fine
    # Uninitialised SITL instance: the check must fire before any worker
    # starts (super().reset runs first in GazeboSITLBridge.reset).
    bridge = GazeboSITLBridge.__new__(GazeboSITLBridge)
    with pytest.raises(ValueError, match="cannot honour.*fixed-wing"):
        MockBridge.reset(bridge, meta)


def test_missing_spec_raises():
    with pytest.raises(OSError):
        load_drone_spec("does-not-exist")
