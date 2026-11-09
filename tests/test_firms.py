"""FIRMS parsing/projection/event-split (synthetic fixture, no network)."""

from dmb.data.firms import FirmsEvent, load_hotspots, project, split_event

FIX = "tests/fixtures/firms_sample.csv"
EVENT = FirmsEvent(lat=0.0, lon=0.0, t0="2019-09-20", grid=120)


def test_load_skips_malformed():
    hotspots = load_hotspots(FIX)
    assert len(hotspots) == 5  # bad-lat row dropped


def test_project_anchor_and_known_cell():
    assert project(EVENT, 0.0, 0.0) == (60, 60)
    assert project(EVENT, 0.0001, 0.0001) == (62, 62)
    assert project(EVENT, 5.0, 5.0) is None  # outside window


def test_split_ignitions_and_new_blocks():
    ignitions, ref_blocks, stats = split_event(load_hotspots(FIX), EVENT)
    assert ignitions == [(62, 62), (64, 62)]
    assert ref_blocks == [(1, 0)]  # same-block day-1 repeat excluded
    assert stats == {
        "n_ignition_cells": 2,
        "n_ref_blocks": 1,
        "frp_mean": 15.0,
        "frp_max": 20.0,
    }
