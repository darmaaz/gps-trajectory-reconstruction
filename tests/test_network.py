"""RoadNetwork construction, projection, and routing."""

from __future__ import annotations

import math

import pytest
from shapely.geometry import LineString

from src.network import (
    RoadNetwork, build_network_from_records, candidate_paths,
    shortest_travel_time,
)
from src.model import StateV1


def test_build_network_arrays_match_records():
    records = [
        (10, "primary", 16.7,
         LineString([(-99.0, 19.0), (-99.001, 19.0)]), 1, 2, 105.0),
    ]
    net = build_network_from_records(records)
    assert len(net) == 1
    assert int(net.edge_ids[0]) == 10
    assert net.road_classes[0] == "primary"
    assert net.adjacency == {1: [0]}
    assert net.node_positions[1] == (19.0, -99.0)
    assert net.node_positions[2] == (19.0, -99.001)
    assert net.turn_restrictions == set()


def test_edge_index_for_link_is_cached(grid_network: RoadNetwork):
    # First call builds the cache; second call must return the same value.
    idx1 = grid_network.edge_index_for_link(3)
    cached = grid_network._link_to_idx_cache    # type: ignore[attr-defined]
    assert isinstance(cached, dict)
    idx2 = grid_network.edge_index_for_link(3)
    assert idx1 == idx2
    with pytest.raises(KeyError):
        grid_network.edge_index_for_link(999_999)


# ----------------------------------------------------------- project_point


def test_project_point_returns_topk_within_radius(grid_network: RoadNetwork):
    # N0 at (19.430, -99.130) — multiple incident edges
    hits = grid_network.project_point(19.430, -99.130,
                                      radius_meters=50.0, max_candidates=5)
    assert len(hits) >= 1
    # all hits must be within radius and sorted by perp distance
    perps = [h[2] for h in hits]
    assert perps == sorted(perps)
    assert all(p <= 50.0 for p in perps)


def test_project_point_off_network_returns_empty(grid_network: RoadNetwork):
    # 5 km from the network
    hits = grid_network.project_point(19.500, -99.000,
                                      radius_meters=200.0, max_candidates=5)
    assert hits == []


def test_project_point_offset_in_meters_sensible(grid_network: RoadNetwork):
    # Project a point at the midpoint of edge 1 (N0->N1, 0.005° lon at 19.43°)
    # → midpoint is at lon=-99.1325; expected offset ≈ half edge length ≈ 262m
    hits = grid_network.project_point(19.430, -99.1325,
                                      radius_meters=20.0, max_candidates=5)
    # all candidates should have offset close to half the ~525m edge length
    for _, offset_m, perp_m in hits:
        assert abs(offset_m - 262.0) < 10.0
        assert perp_m < 1.0    # essentially on the line


# ------------------------------------------------------- routing primitives


def test_shortest_travel_time_unreachable_returns_inf(grid_network: RoadNetwork):
    assert shortest_travel_time(grid_network, 19.5, -99.0, 19.43, -99.13) == float("inf")


def test_shortest_travel_time_within_network_finite(grid_network: RoadNetwork):
    t = shortest_travel_time(grid_network, 19.430, -99.1349, 19.430, -99.1251)
    # ~1.1 km at 60 km/h → ~66 s
    assert 50.0 < t < 90.0


def test_shortest_travel_time_factor_speeds_up(grid_network: RoadNetwork):
    base = shortest_travel_time(grid_network, 19.430, -99.1349,
                                19.430, -99.1251, max_speed_factor=1.0)
    fast = shortest_travel_time(grid_network, 19.430, -99.1349,
                                19.430, -99.1251, max_speed_factor=2.0)
    # higher speed factor → strictly less travel time
    assert fast < base
    assert fast == pytest.approx(base / 2.0, rel=0.01)


def test_typical_speeds_populated_from_defaults(grid_network: RoadNetwork):
    """`build_network_from_records` initialises `typical_speeds_ms` from the
    `V_TYPICAL_MS` defaults per road class."""
    from src.config import v_typical_for
    for i, rc in enumerate(grid_network.road_classes):
        assert grid_network.typical_speeds_ms[i] == pytest.approx(
            v_typical_for(str(rc))
        )
    # Sanity: typical < max for at least some edges (whole point of the
    # split — typical speed is a routing-cost prior, max is feasibility).
    assert (
        grid_network.typical_speeds_ms < grid_network.max_speeds_ms
    ).any()


def test_set_typical_speeds_by_class_overrides_and_clears_cache(
    grid_network: RoadNetwork,
):
    """Calling `set_typical_speeds_by_class` updates the per-edge array and
    invalidates `_nx_graph_cache` so subsequent routing rebuilds against
    the new costs."""
    from src.network.routing import _get_nx_graph
    # Snapshot session-fixture state and restore at end — grid_network is
    # session-scoped, so mutation here would leak into later tests.
    saved_speeds = grid_network.typical_speeds_ms.copy()
    saved_cache = grid_network._nx_graph_cache
    try:
        # Warm the cache.
        _ = _get_nx_graph(grid_network)
        assert grid_network._nx_graph_cache is not None
        # Override: bump primary to 30 m/s.
        grid_network.set_typical_speeds_by_class({"primary": 30.0})
        # Cache invalidated.
        assert grid_network._nx_graph_cache is None
        for i, rc in enumerate(grid_network.road_classes):
            if str(rc) == "primary":
                assert grid_network.typical_speeds_ms[i] == pytest.approx(30.0)
    finally:
        grid_network.typical_speeds_ms = saved_speeds
        grid_network._nx_graph_cache = saved_cache


def test_shortest_travel_time_still_uses_max_speeds(grid_network: RoadNetwork):
    """Stale-jump detection depends on `shortest_travel_time` being a
    feasibility lower bound (uses `max_speeds_ms`, not typical). Confirm
    the bound stays unaffected when `typical_speeds_ms` is overridden."""
    saved_speeds = grid_network.typical_speeds_ms.copy()
    saved_cache = grid_network._nx_graph_cache
    try:
        t_before = shortest_travel_time(
            grid_network, 19.430, -99.130, 19.430, -99.135, max_speed_factor=1.0,
        )
        grid_network.set_typical_speeds_by_class({"primary": 1.0})    # absurd
        t_after = shortest_travel_time(
            grid_network, 19.430, -99.130, 19.430, -99.135, max_speed_factor=1.0,
        )
        assert t_after == pytest.approx(t_before, rel=1e-6)
    finally:
        grid_network.typical_speeds_ms = saved_speeds
        grid_network._nx_graph_cache = saved_cache


def test_candidate_paths_same_edge_forward(grid_network: RoadNetwork, t0):
    src = StateV1(link_id=3, offset=100.0, entry_time=t0)
    dst = StateV1(link_id=3, offset=400.0, entry_time=t0)
    paths = candidate_paths([src], [dst], grid_network, time_budget_seconds=60)
    assert len(paths) == 1
    p = paths[0]
    assert p.edges == (3,)
    # Routing cost uses typical speeds (`length / typical_speed`); link 3 is
    # primary → V_TYPICAL_MS["primary"] = 13.5 m/s.
    expected_t = 300.0 / 13.5
    assert abs(p.expected_travel_time - expected_t) < 0.1
    assert p.length_meters == pytest.approx(300.0, abs=0.5)


def test_candidate_paths_budget_pruning(grid_network: RoadNetwork, t0):
    src = StateV1(link_id=2, offset=550.0, entry_time=t0)
    dst = StateV1(link_id=7, offset=275.0, entry_time=t0)
    # very tight budget → no paths
    assert candidate_paths([src], [dst], grid_network, 5.0) == []
    # generous budget → at least one
    assert len(candidate_paths([src], [dst], grid_network, 200.0)) >= 1


def test_subgraph_for_bbox_filters_to_intersecting_edges(grid_network: RoadNetwork):
    """Subgraph contains only edges intersecting the padded bbox; nx cache
    on the new subgraph starts fresh."""
    full_count = len(grid_network)
    # Tight bbox around N4 (the south end) — only edges incident to N4
    # should make it; the north/east/west edges around N0/N1/N2/N3 are
    # too far to clip even at 10 m buffer.
    sub = grid_network.subgraph_for_bbox(
        lat_min=19.4249, lat_max=19.4251,
        lon_min=-99.1301, lon_max=-99.1299,
        buffer_m=10.0,
    )
    kept = {int(eid) for eid in sub.edge_ids}
    # Edges 1, 2 (around N1 westside) and 3, 4 (around N2 eastside) and
    # 5, 6 (N0↔N3 northbound) are well clear of the south-end bbox.
    for excluded in (1, 2, 3, 4, 5, 6):
        assert excluded not in kept, f"edge {excluded} should not be in subgraph"
    assert len(sub) < full_count
    assert sub._nx_graph_cache is None    # type: ignore[attr-defined]


def test_subgraph_for_bbox_buffer_includes_nearby_edges(grid_network: RoadNetwork):
    """A wide-enough buffer should pull in adjacent edges that don't
    intersect the raw bbox but matter for routing."""
    sub = grid_network.subgraph_for_bbox(
        lat_min=19.4299, lat_max=19.4301,
        lon_min=-99.135, lon_max=-99.130,
        buffer_m=2000.0,    # 2km buffer pulls in everything in this synthetic grid
    )
    kept = {int(eid) for eid in sub.edge_ids}
    assert kept == {int(e) for e in grid_network.edge_ids}


def test_candidate_paths_dedup_by_edge_tuple(grid_network: RoadNetwork, t0):
    # Two source candidates that produce the same routed path; result should
    # collapse to a single Path keyed on edge tuple.
    src_a = StateV1(link_id=2, offset=550.0, entry_time=t0)
    src_b = StateV1(link_id=2, offset=540.0, entry_time=t0)
    dst = StateV1(link_id=7, offset=275.0, entry_time=t0)
    paths = candidate_paths([src_a, src_b], [dst], grid_network,
                            time_budget_seconds=200, max_paths=20)
    seen = {p.edges for p in paths}
    assert len(seen) == len(paths)    # no duplicate edge tuples
