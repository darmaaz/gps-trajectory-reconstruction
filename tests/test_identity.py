"""Physical-road identity (loader twin/segment keys, identity.canonical_route,
truncate_with_route_diversity) and offset-honouring path_polyline.

Grid fixture (tests/conftest.py): links 1/2, 3/4, 5/6, 7/8 are two-way twin
pairs; link 9 is one-way. Node pairs: 1↔2 = (0,1), 3↔4 = (0,2), 5↔6 = (0,3),
7↔8 = (0,4), 9 = (1,4).
"""

from __future__ import annotations

import numpy as np
import pytest

from src.model import Path, StateV1
from src.network import (
    RoadNetwork, candidate_paths, canonical_route, path_polyline,
    truncate_with_route_diversity,
)


def _path(edges, ett, start=0.0, end=0.0, off_road=False, budget=120.0):
    return Path(
        edges=tuple(edges), start_offset=start, end_offset=end,
        expected_travel_time=ett, length_meters=0.0,
        feature_vector=np.zeros(0, dtype=float), time_budget=budget,
        is_off_road=off_road,
    )


# ───────────────────────────────── F1: loader identity layer

def test_twin_indices_pair_two_way_edges(grid_network: RoadNetwork):
    twins = grid_network.twin_indices()
    i1 = grid_network.edge_index_for_link(1)
    i2 = grid_network.edge_index_for_link(2)
    assert twins[i1] == i2
    assert twins[i2] == i1


def test_twin_indices_one_way_has_no_twin(grid_network: RoadNetwork):
    twins = grid_network.twin_indices()
    i9 = grid_network.edge_index_for_link(9)
    assert twins[i9] == -1


def test_twin_indices_symmetric_for_all_two_way_pairs(grid_network: RoadNetwork):
    twins = grid_network.twin_indices()
    for a, b in ((1, 2), (3, 4), (5, 6), (7, 8)):
        ia = grid_network.edge_index_for_link(a)
        ib = grid_network.edge_index_for_link(b)
        assert twins[ia] == ib and twins[ib] == ia


def test_segment_keys_unify_twins(grid_network: RoadNetwork):
    assert (grid_network.segment_key_for_link(1)
            == grid_network.segment_key_for_link(2) == (0, 1))
    assert grid_network.segment_key_for_link(9) == (1, 4)


def test_segment_key_unknown_link_raises(grid_network: RoadNetwork):
    with pytest.raises(KeyError):
        grid_network.segment_key_for_link(424242)


def test_identity_caches_are_lazy_and_stable(grid_network: RoadNetwork):
    assert grid_network.twin_indices() is grid_network.twin_indices()
    assert (grid_network.undirected_segment_keys()
            is grid_network.undirected_segment_keys())


# ───────────────────────────────── F2: canonical_route

def test_canonical_route_unifies_twin_spellings(grid_network: RoadNetwork):
    # Same physical route N0→N1→N4 anchored on either twin of street (0,1).
    assert (canonical_route(_path((1, 9), 10.0), grid_network)
            == canonical_route(_path((2, 9), 11.0), grid_network)
            == ((0, 1), (1, 4)))


def test_canonical_route_collapses_consecutive_same_segment(grid_network):
    # Doubling back onto the twin of the same street is one segment, not two.
    assert canonical_route(_path((1, 2), 5.0), grid_network) == ((0, 1),)


def test_canonical_route_off_road_is_none(grid_network: RoadNetwork):
    assert canonical_route(_path((1, 3), 5.0, off_road=True), grid_network) is None


def test_canonical_route_skips_unknown_links(grid_network: RoadNetwork):
    assert canonical_route(_path((1, 424242, 9), 5.0), grid_network) \
        == ((0, 1), (1, 4))


# ───────────────────────────────── F2: diversity-aware truncation

def test_truncation_prefers_new_route_over_twin_spelling(grid_network):
    p_best = _path((1,), 10.0)     # route (0,1)
    p_twin = _path((2,), 11.0)     # same physical route, other spelling
    p_new = _path((3,), 12.0)      # different physical route (0,2)
    kept = truncate_with_route_diversity(
        [p_best, p_twin, p_new], grid_network, max_paths=2)
    assert kept == [p_best, p_new]   # legacy cut would keep p_best, p_twin


def test_truncation_backfills_with_spellings_when_routes_exhausted(grid_network):
    p1 = _path((1,), 10.0)
    p2 = _path((2,), 11.0)
    p3 = _path((2,), 12.0, start=1.0)
    kept = truncate_with_route_diversity([p1, p2, p3], grid_network, max_paths=2)
    assert kept == [p1, p2]          # never smaller than the legacy cut


def test_truncation_output_sorted_by_travel_time(grid_network):
    paths = [_path((1,), 10.0), _path((2,), 11.0), _path((3,), 12.0),
             _path((5,), 13.0), _path((7,), 14.0)]
    kept = truncate_with_route_diversity(paths, grid_network, max_paths=3)
    etts = [p.expected_travel_time for p in kept]
    assert etts == sorted(etts) and len(kept) == 3


def test_truncation_noop_when_under_cap(grid_network):
    paths = [_path((1,), 10.0), _path((2,), 11.0)]
    assert truncate_with_route_diversity(paths, grid_network, 20) is paths


def test_truncation_off_road_always_admitted(grid_network):
    p1 = _path((1,), 10.0)
    p_off = _path((1, 3), 11.0, off_road=True)
    p_twin = _path((2,), 11.5)
    kept = truncate_with_route_diversity([p1, p_off, p_twin], grid_network, 2)
    assert kept == [p1, p_off]


def test_candidate_paths_diversify_flag_respects_contract(grid_network, t0):
    # Integration: flag on/off both return ≤ max_paths, ett-sorted lists.
    src = [StateV1(link_id=1, offset=10.0, entry_time=t0)]
    dst = [StateV1(link_id=9, offset=200.0, entry_time=t0)]
    for flag in (True, False):
        out = candidate_paths(src, dst, grid_network, 600.0, max_paths=3,
                              diversify_truncation=flag)
        assert out and len(out) <= 3
        etts = [p.expected_travel_time for p in out]
        assert etts == sorted(etts)


# ───────────────────────────────── F3: path_polyline

def _interp_lonlat(net: RoadNetwork, link: int, offset_m: float):
    idx = net.edge_index_for_link(link)
    geom = net.geoms[idx]
    pt = geom.interpolate((offset_m / float(net.lengths_m[idx])) * geom.length)
    return float(pt.x), float(pt.y)


def test_polyline_single_edge_trims_to_offsets(grid_network: RoadNetwork):
    idx = grid_network.edge_index_for_link(1)
    length = float(grid_network.lengths_m[idx])
    p = _path((1,), 5.0, start=0.25 * length, end=0.75 * length)
    pl = path_polyline(p, grid_network)
    assert pl.shape[0] >= 2
    np.testing.assert_allclose(pl[0], _interp_lonlat(grid_network, 1, 0.25 * length), atol=1e-7)
    np.testing.assert_allclose(pl[-1], _interp_lonlat(grid_network, 1, 0.75 * length), atol=1e-7)


def test_polyline_multi_edge_trims_terminals_only(grid_network: RoadNetwork):
    # N0→N1 on link 1 (entered at 200 m), then N1→N4 on link 9 (left at 50 m).
    p = _path((1, 9), 5.0, start=200.0, end=50.0)
    pl = path_polyline(p, grid_network)
    np.testing.assert_allclose(pl[0], _interp_lonlat(grid_network, 1, 200.0), atol=1e-7)
    np.testing.assert_allclose(pl[-1], _interp_lonlat(grid_network, 9, 50.0), atol=1e-7)
    # The shared junction N1 (-99.135, 19.430) must be on the polyline.
    assert any(np.allclose(v, (-99.135, 19.430), atol=1e-7) for v in pl)


def test_polyline_shorter_than_full_edge_concatenation(grid_network):
    p = _path((1, 9), 5.0, start=400.0, end=50.0)
    pl = path_polyline(p, grid_network)
    full = 0.0
    for link in (1, 9):
        full += float(grid_network.lengths_m[grid_network.edge_index_for_link(link)])
    deg = np.linalg.norm(np.diff(pl, axis=0), axis=1).sum()
    # crude deg→m upper bound: 1 deg ≤ 111.5 km
    assert deg * 111_500.0 < full


def test_polyline_stay_path_is_local(grid_network: RoadNetwork):
    # Backward-offset stay path: a short jitter span, NOT the whole edge.
    p = _path((1,), 0.0, start=100.0, end=90.0)
    pl = path_polyline(p, grid_network)
    assert pl.shape[0] >= 2
    span_deg = np.linalg.norm(pl[-1] - pl[0])
    edge_deg = grid_network.geoms[grid_network.edge_index_for_link(1)].length
    assert span_deg < 0.05 * edge_deg


def test_polyline_off_road_is_straight_segment(grid_network: RoadNetwork):
    p = _path((1, 3), 5.0, start=50.0, end=50.0, off_road=True)
    pl = path_polyline(p, grid_network)
    assert pl.shape == (2, 2)
    np.testing.assert_allclose(pl[0], _interp_lonlat(grid_network, 1, 50.0), atol=1e-7)
    np.testing.assert_allclose(pl[1], _interp_lonlat(grid_network, 3, 50.0), atol=1e-7)
