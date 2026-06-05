"""Off-road / near-stationary candidate path behaviour."""

from __future__ import annotations

import numpy as np
import pytest

from src.model import FEATURE_DIM, Path, StateV1
from src.model.features import path_features
from src.api.interpolation import position_in_transition
from src.network.routing import (
    _add_offroad_candidates, _offroad_path, candidate_paths,
)


def _state(network, link_id, offset, t):
    return StateV1(link_id=link_id, offset=offset, entry_time=t, perp_m=5.0)


# --------------------------------------------------------- Path / construction


def test_is_off_road_defaults_false():
    p = Path(
        edges=(1, 2), start_offset=0.0, end_offset=0.0,
        expected_travel_time=10.0, length_meters=50.0,
        feature_vector=np.zeros(FEATURE_DIM), time_budget=120.0,
    )
    assert p.is_off_road is False


def test_offroad_path_construction(grid_network, t0):
    e1 = grid_network.edge_index_for_link(1)
    e9 = grid_network.edge_index_for_link(9)
    src = _state(grid_network, 1, 10.0, t0)
    dst = _state(grid_network, 9, 5.0, t0)
    off = _offroad_path(grid_network, src, dst, straight_m=80.0, actual_budget=120.0)
    assert off.is_off_road is True
    assert off.edges == (1, 9)
    assert off.length_meters == pytest.approx(80.0)
    assert off.start_offset == pytest.approx(10.0)
    assert off.end_offset == pytest.approx(5.0)
    # Travel time uses a slow maneuver speed → large inferred dwell.
    assert off.expected_travel_time > 0
    assert off.inferred_dwell > 60.0     # most of the 120s budget is dwell
    # min_traversal_time tiny → always admissible.
    assert off.min_traversal_time < off.expected_travel_time


# --------------------------------------------------------- features


def test_offroad_features_zero_adjacency_slots(grid_network, t0):
    # Build an off-road path whose two edges, if treated as adjacent, would
    # register turns/intersections. Off-road must zero those.
    src = _state(grid_network, 1, 10.0, t0)
    dst = _state(grid_network, 9, 5.0, t0)
    off = _offroad_path(grid_network, src, dst, straight_m=80.0, actual_budget=120.0)
    feats = path_features(off, grid_network)
    assert feats.shape == (FEATURE_DIM,)
    # Geometry / time / dwell / anchor slots populated.
    assert feats[0] == pytest.approx(80.0 / 1000.0)        # length_km
    assert feats[12] > 0                                    # travel_min
    assert feats[13] > 0                                    # dwell_min
    assert feats[15] == pytest.approx(0.5)                  # start_perp 5m/10
    # Adjacency-dependent slots all zero (no traversed sequence).
    for slot in (1, 2, 3, 4, 17):
        assert feats[slot] == 0.0, f"slot {slot} should be zero for off-road"
    # Road-class fractions ARE populated from the two endpoint edges
    # (0.5 each) so the off-road path isn't structurally handicapped
    # against routed alternatives on the class-fraction reward. Edge 1 is
    # primary (slot 7), edge 9 is residential (slot 10).
    assert feats[7] == pytest.approx(0.5)    # primary (edge 1)
    assert feats[10] == pytest.approx(0.5)   # residential (edge 9)
    assert feats[5:12].sum() == pytest.approx(1.0)


# --------------------------------------------------- position_in_transition


def _offroad(grid_network, t0, travel=20.0, budget=120.0):
    src = _state(grid_network, 1, 0.0, t0)
    dst = _state(grid_network, 9, 0.0, t0)
    off = _offroad_path(grid_network, src, dst, straight_m=80.0, actual_budget=budget)
    # Force a known travel time for deterministic dwell-window math.
    from dataclasses import replace
    return replace(off, expected_travel_time=travel)


def test_offroad_front_rule_snaps_to_endpoints(grid_network, t0):
    off = _offroad(grid_network, t0, travel=20.0, budget=120.0)   # dwell=100s
    # During dwell window (tau <= 100): at source endpoint.
    link, offset = position_in_transition(off, grid_network, 50.0, rule="front")
    assert link == off.edges[0] and offset == pytest.approx(off.start_offset)
    # After dwell: at destination endpoint.
    link, offset = position_in_transition(off, grid_network, 110.0, rule="front")
    assert link == off.edges[-1] and offset == pytest.approx(off.end_offset)


def test_offroad_back_rule_snaps_to_destination_late(grid_network, t0):
    off = _offroad(grid_network, t0, travel=20.0, budget=120.0)
    # Back: once tau >= travel(20), dwelling at destination.
    link, offset = position_in_transition(off, grid_network, 90.0, rule="back")
    assert link == off.edges[-1] and offset == pytest.approx(off.end_offset)


def test_offroad_spread_rule_snaps_by_midpoint(grid_network, t0):
    off = _offroad(grid_network, t0, travel=20.0, budget=120.0)
    # Spread: frac = tau/budget. tau=30 → 0.25 < 0.5 → source.
    link, _ = position_in_transition(off, grid_network, 30.0, rule="spread")
    assert link == off.edges[0]
    # tau=90 → 0.75 ≥ 0.5 → destination.
    link, _ = position_in_transition(off, grid_network, 90.0, rule="spread")
    assert link == off.edges[-1]


# --------------------------------------------------------- trigger logic


# best_routed is keyed pair → (routed_len_m, routed_ett_s). Budget below
# is 120s, so ett > 120 means overslacked (detour can't be driven in time).


def test_trigger_fires_on_short_straight_big_detour_overslacked(grid_network, t0):
    # Different edges near shared node 1 → small straight-line; supply a
    # routed detour that's both long (big detour ratio) AND overslacked
    # (ett 200 > 120 budget). All three gates hold → off-road fires.
    e1_len = float(grid_network.lengths_m[grid_network.edge_index_for_link(1)])
    src = _state(grid_network, 1, e1_len - 5.0, t0)    # near node 1
    dst = _state(grid_network, 9, 5.0, t0)             # near node 1
    by_edges: dict = {}
    best_routed = {(1, 9): (2000.0, 200.0)}
    _add_offroad_candidates(
        [src], [dst], grid_network, 120.0, by_edges, best_routed,
        max_straight_m=300.0, min_detour_ratio=3.0, min_overslack=1.0,
    )
    off_paths = [p for p in by_edges.values() if p.is_off_road]
    assert len(off_paths) == 1
    assert off_paths[0].edges == (1, 9)


def test_trigger_skips_when_detour_fits_budget(grid_network, t0):
    # Big detour ratio + short straight, BUT the detour fits the budget
    # (ett 80 < 120) → plausibly a real short drive → off-road must NOT
    # fire. This is the over-allow guard.
    e1_len = float(grid_network.lengths_m[grid_network.edge_index_for_link(1)])
    src = _state(grid_network, 1, e1_len - 5.0, t0)
    dst = _state(grid_network, 9, 5.0, t0)
    by_edges: dict = {}
    best_routed = {(1, 9): (2000.0, 80.0)}     # not overslacked
    _add_offroad_candidates(
        [src], [dst], grid_network, 120.0, by_edges, best_routed,
        max_straight_m=300.0, min_detour_ratio=3.0, min_overslack=1.0,
    )
    assert not any(p.is_off_road for p in by_edges.values())


def test_trigger_skips_when_detour_small(grid_network, t0):
    src = _state(grid_network, 1, 10.0, t0)
    dst = _state(grid_network, 9, 5.0, t0)
    by_edges: dict = {}
    # Routed length only slightly above straight-line → not a detour.
    best_routed = {(1, 9): (30.0, 200.0)}
    _add_offroad_candidates(
        [src], [dst], grid_network, 120.0, by_edges, best_routed,
        max_straight_m=300.0, min_detour_ratio=3.0, min_overslack=1.0,
    )
    assert not any(p.is_off_road for p in by_edges.values())


def test_trigger_skips_when_straight_too_long(grid_network, t0):
    src = _state(grid_network, 1, 10.0, t0)
    dst = _state(grid_network, 9, 5.0, t0)
    by_edges: dict = {}
    best_routed = {(1, 9): (5000.0, 400.0)}
    _add_offroad_candidates(
        [src], [dst], grid_network, 120.0, by_edges, best_routed,
        max_straight_m=10.0,    # tighter than the real straight-line → skip
        min_detour_ratio=3.0, min_overslack=1.0,
    )
    assert not any(p.is_off_road for p in by_edges.values())


def test_trigger_skips_same_edge(grid_network, t0):
    src = _state(grid_network, 1, 10.0, t0)
    dst = _state(grid_network, 1, 200.0, t0)    # same edge
    by_edges: dict = {}
    best_routed = {(1, 1): (2000.0, 200.0)}
    _add_offroad_candidates(
        [src], [dst], grid_network, 120.0, by_edges, best_routed,
        max_straight_m=300.0, min_detour_ratio=3.0, min_overslack=1.0,
    )
    assert not any(p.is_off_road for p in by_edges.values())


def test_candidate_paths_offroad_disabled_by_default(grid_network, t0):
    # With enable_offroad unset, no off-road candidate ever appears.
    src = _state(grid_network, 1, 0.0, t0)
    dst = _state(grid_network, 9, 0.0, t0)
    paths = candidate_paths([src], [dst], grid_network, 120.0)
    assert not any(getattr(p, "is_off_road", False) for p in paths)
