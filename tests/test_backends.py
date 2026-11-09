"""Backend selection: mock is the only backend (no Docker needed)."""

import pytest

from hsea.sim.bridge import MockBridge
from hsea.solvers.mission import _bridge_for_backend


def test_backend_selection():
    assert _bridge_for_backend("mock") is MockBridge
    with pytest.raises(ValueError):
        _bridge_for_backend("gazebo-sitl")  # sim tracks stayed in the bench repo
    with pytest.raises(ValueError):
        _bridge_for_backend("colosseum")
