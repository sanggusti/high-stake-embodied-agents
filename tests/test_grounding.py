"""Camera grounding: hand-checked pixel-to-grid projections."""

from dmb.perception.grounding import (
    CameraModel,
    DronePose,
    GridSpec,
    project_mask,
    project_pixel,
    survey_camera,
)
from dmb.perception.sam import SAMBox, SAMMask, SAMRecord


def _nadir():
    cam = CameraModel(100, 100, 90.0, 90.0, 0.0)
    pose = DronePose(5, 5, 50.0)
    grid = GridSpec(10, 10, 5.0)
    return cam, pose, grid


def test_nadir_center_pixel_is_drone_cell():
    cam, pose, grid = _nadir()
    assert project_pixel(50, 50, cam, pose, grid) == (5, 5)


def test_nadir_known_offsets():
    cam, pose, grid = _nadir()
    assert project_pixel(70, 50, cam, pose, grid) == (9, 5)  # +20 m east
    assert project_pixel(50, 70, cam, pose, grid) == (5, 1)  # +20 m south


def test_off_grid_grounded_and_sky_rays_are_none():
    cam, pose, grid = _nadir()
    assert project_pixel(100, 50, cam, pose, grid) is None  # 75 m east, off-grid
    assert project_pixel(50, 50, cam, DronePose(5, 5, 0.0), grid) is None


def test_level_camera_horizon_and_below():
    cam = CameraModel(100, 100, 90.0, 0.0, 0.0)
    pose = DronePose(5, 5, 10.0)
    grid = GridSpec(10, 10, 5.0)
    assert project_pixel(50, 50, cam, pose, grid) is None  # level ray never lands
    assert project_pixel(50, 75, cam, pose, grid) == (5, 9)  # 20 m north


def test_yaw_rotates_image_frame():
    cam = CameraModel(100, 100, 90.0, 90.0, 90.0)  # yaw east
    pose = DronePose(5, 5, 50.0)
    grid = GridSpec(10, 10, 5.0)
    assert project_pixel(50, 70, cam, pose, grid) == (1, 5)  # image-down is west


def test_invalid_models_fail_loudly():
    for kwargs in ({"hfov_deg": 0.0}, {"hfov_deg": 180.0}, {"pitch_deg": 91.0}):
        try:
            CameraModel(100, 100, kwargs.get("hfov_deg", 90.0),
                        kwargs.get("pitch_deg", 90.0))
        except ValueError:
            pass
        else:
            raise AssertionError(f"must reject {kwargs}")


def _rec(x1, y1, x2, y2, H=2, W=2):  # noqa: N803
    return SAMRecord(0, SAMBox(x1, y1, x2, y2, 100, 100, 0.9), SAMMask(H, W, "~", "~x"))


def test_project_mask_maps_set_pixels():
    cam, pose, grid = _nadir()
    cells = project_mask(_rec(40, 40, 59, 59), bytes([1, 1, 1, 1]), cam, pose, grid)
    assert cells == {(4, 6), (6, 6), (4, 4), (6, 4)}


def test_project_mask_stride_and_bad_raster():
    cam, pose, grid = _nadir()
    cells = project_mask(
        _rec(40, 40, 59, 59), bytes([1, 1, 1, 1]), cam, pose, grid, stride=2)
    assert cells == {(4, 6)}
    assert project_mask(_rec(40, 40, 59, 59), bytes([1]), cam, pose, grid) == set()
    try:
        project_mask(_rec(40, 40, 59, 59), bytes([1] * 4), cam, pose, grid, stride=0)
    except ValueError:
        pass
    else:
        raise AssertionError("stride 0 must fail loudly")


def test_survey_camera_is_nadir():
    cam = survey_camera(640, 480)
    assert (cam.pitch_deg, cam.yaw_deg) == (90.0, 0.0)
    assert (cam.width, cam.height) == (640, 480)


def test_grid_to_pixel_yaw_roundtrip():
    from dmb.perception.grounding import (
        CameraModel, DronePose, GridSpec, grid_to_pixel, project_pixel,
    )

    grid = GridSpec(120, 120, 5.0)
    pose = DronePose(59, 63, 150.0)
    for yaw in (0.0, 90.0, 180.0, 270.0):
        cam = CameraModel(1280, 720, 60.0, 90.0, yaw)
        for cell in ((58, 60), (50, 54), (59, 63)):
            px = grid_to_pixel(cell, cam, pose, grid)
            if px is None:
                continue
            assert project_pixel(px[0], px[1], cam, pose, grid) == cell


def test_grid_to_pixel_firms_nadir_hits_beds():
    """Yaw-90 survey model lands on the rendered firebeds (<30 px)."""
    from dmb.perception.grounding import (
        CameraModel, DronePose, GridSpec, grid_to_pixel,
    )

    cam = CameraModel(1280, 720, 60.0, 90.0, 90.0)
    pose = DronePose(59, 63, 150.0)
    grid = GridSpec(120, 120, 5.0)
    # Measured orange centroids in survey-...-nadir.png (2026-11-04).
    for cell, want in (((58, 60), (748, 386)), ((50, 54), (978, 678))):
        u, v = grid_to_pixel(cell, cam, pose, grid)
        assert abs(u - want[0]) < 30 and abs(v - want[1]) < 30
