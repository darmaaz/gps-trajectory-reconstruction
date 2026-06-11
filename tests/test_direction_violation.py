"""F5 — direction-violation candidates (`Config.enable_direction_violation`).

Grid fixture: links 1/2, 3/4, 5/6, 7/8 are two-way twin pairs; link 9
(N1 → N4, residential) is the only one-way. Wrong-way candidates should
appear ONLY for one-way edges, only when the flag is on, and always carry
`reversed_mask` + a positive `n_direction_violations` feature.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.model import FEATURE_DIM, Path, StateV1
from src.model.features import path_features
from src.network import RoadNetwork, candidate_paths, path_polyline
from src.network.routing import _get_nx_graph, _get_permissive_graph


def _vpath(edges, mask, start=0.0, end=0.0, length=100.0, ett=10.0):
    return Path(
        edges=tuple(edges), start_offset=start, end_offset=end,
        expected_travel_time=ett, length_meters=length,
        feature_vector=np.zeros(0), time_budget=120.0,
        reversed_mask=mask,
    )


# ───────────────────────────────── permissive graph

def test_permissive_graph_adds_reverse_arc_for_oneway_only(grid_network):
    G = _get_nx_graph(grid_network)
    P = _get_permissive_graph(grid_network, 3.0)
    i9 = grid_network.edge_index_for_link(9)
    u, v = int(grid_network.from_node[i9]), int(grid_network.to_node[i9])  # 1 → 4
    assert not G.has_edge(v, u)                  # legal graph: no wrong-way
    assert P.has_edge(v, u)                      # permissive: penalized arc
    assert P[v][u]["illegal"] is True
    assert P[v][u]["edge_idx"] == i9
    assert P[v][u]["weight"] == pytest.approx(G[u][v]["weight"] * 3.0)
    # Two-way street (0,1): twin already covers the reverse — no new arcs.
    assert P.number_of_edges() == G.number_of_edges() + 1


def test_permissive_graph_cached_per_factor(grid_network):
    assert _get_permissive_graph(grid_network, 3.0) is _get_permissive_graph(grid_network, 3.0)
    assert _get_permissive_graph(grid_network, 2.0) is not None  # rebuild ok


# ───────────────────────────────── candidate enumeration

def test_flag_off_no_reversed_paths(grid_network, t0):
    # N4 → N1 against one-way link 9: legally must detour via N0.
    src = [StateV1(link_id=9, offset=700.0, entry_time=t0)]
    dst = [StateV1(link_id=9, offset=100.0, entry_time=t0)]
    out = candidate_paths(src, dst, grid_network, 600.0)
    assert all(p.reversed_mask is None for p in out)


def test_same_edge_backroll_on_oneway(grid_network, t0):
    # Backward offsets on the one-way: flag off → only zero-motion stay;
    # flag on → ALSO a real reverse-roll with honest length.
    src = [StateV1(link_id=9, offset=700.0, entry_time=t0)]
    dst = [StateV1(link_id=9, offset=100.0, entry_time=t0)]
    legal = candidate_paths(src, dst, grid_network, 600.0)
    stays = [p for p in legal if len(p.edges) == 1]
    assert stays and all(p.length_meters == 0.0 for p in stays)

    out = candidate_paths(src, dst, grid_network, 600.0,
                          enable_direction_violation=True)
    rolls = [p for p in out if p.reversed_mask == (True,)]
    assert len(rolls) == 1
    assert rolls[0].length_meters == pytest.approx(600.0)
    assert rolls[0].n_direction_violations == 1
    # The stay path must survive alongside (different physical story).
    assert any(len(p.edges) == 1 and p.reversed_mask is None for p in out)


def test_same_edge_backward_on_twoway_gets_no_roll(grid_network, t0):
    # Two-way link 1: backward offsets stay zero-motion even with flag on
    # (the reverse twin is a separate projection candidate).
    src = [StateV1(link_id=1, offset=300.0, entry_time=t0)]
    dst = [StateV1(link_id=1, offset=100.0, entry_time=t0)]
    out = candidate_paths(src, dst, grid_network, 600.0,
                          enable_direction_violation=True)
    assert all(p.reversed_mask is None for p in out)


def test_violation_path_enumerated_when_legal_route_absent(grid_network, t0):
    # src deep on one-way 9, dst on its upstream edge 1 near N1: legally
    # the vehicle must finish 9 at N4 and loop back 8 → 0 → 2; with the
    # flag, a backward exit of 9 via N1 becomes available and shorter.
    src = [StateV1(link_id=9, offset=600.0, entry_time=t0)]
    dst = [StateV1(link_id=2, offset=50.0, entry_time=t0)]
    legal = candidate_paths(src, dst, grid_network, 600.0)
    assert all(p.reversed_mask is None for p in legal)
    out = candidate_paths(src, dst, grid_network, 600.0,
                          enable_direction_violation=True)
    viol = [p for p in out if p.n_direction_violations > 0]
    assert viol, "permissive enumeration should add a wrong-way candidate"
    # Wrong-way exit of edge 9 backward to N1: 600 m vs ~1.9 km legal loop.
    best = min(viol, key=lambda p: p.length_meters)
    assert best.edge_reversed(0)
    assert best.length_meters < min(
        (p.length_meters for p in legal), default=float("inf"))


def test_violation_paths_have_higher_search_cost_than_legal(grid_network, t0):
    # Legal candidates must still be enumerated and sorted first when they
    # are genuinely better.
    src = [StateV1(link_id=1, offset=10.0, entry_time=t0)]
    dst = [StateV1(link_id=9, offset=200.0, entry_time=t0)]
    out = candidate_paths(src, dst, grid_network, 600.0,
                          enable_direction_violation=True)
    assert out[0].reversed_mask is None


# ───────────────────────────────── features

def test_n_direction_violations_feature_slot(grid_network):
    p = _vpath((1, 9), (False, True), start=100.0, end=50.0)
    feats = path_features(p, grid_network)
    assert feats.shape == (FEATURE_DIM,)
    assert feats[18] == 1.0


def test_feature_slot_counts_runs_not_edges(grid_network):
    # One continuous wrong-way maneuver OSM split into two edges = ONE run.
    # Slot [18] must read 1.0, NOT the per-edge count of 2 (the bug being fixed:
    # otherwise μ[18] is paid once per OSM split, not once per maneuver).
    p_run = _vpath((1, 9), (True, True))
    assert path_features(p_run, grid_network)[18] == 1.0
    assert p_run.n_direction_violation_runs == 1
    assert p_run.n_direction_violations == 2          # raw edge count still 2

    # Two separate maneuvers (a legal edge between them) = TWO runs.
    p_two = _vpath((1, 2, 9), (True, False, True))
    assert path_features(p_two, grid_network)[18] == 2.0
    assert p_two.n_direction_violation_runs == 2
    legal = _vpath((1, 9), None, start=100.0, end=50.0)
    assert path_features(legal, grid_network)[18] == 0.0


def test_default_mu_pads_pre_violation_file(tmp_path, monkeypatch):
    import src.data as data_mod
    from src.model.features import DEFAULT_DIRECTION_VIOLATION_WEIGHT
    old = np.arange(18, dtype=float)
    f = tmp_path / "mu_default.npy"
    np.save(f, old)
    monkeypatch.setattr(data_mod, "_MU_PATH", f)
    mu = data_mod.default_mu()
    assert mu.shape == (FEATURE_DIM,)
    np.testing.assert_allclose(mu[:18], old)
    assert mu[18] == DEFAULT_DIRECTION_VIOLATION_WEIGHT


# ───────────────────────────────── geometry + interpolation

def test_polyline_reversed_terminal_runs_backward(grid_network):
    # Reversed exit of one-way 9 from offset 600 toward N1, then onto 2.
    p = _vpath((9, 2), (True, False), start=600.0, end=50.0)
    pl = path_polyline(p, grid_network)
    i9 = grid_network.edge_index_for_link(9)
    geom = grid_network.geoms[i9]
    m_len = float(grid_network.lengths_m[i9])
    start_pt = geom.interpolate((600.0 / m_len) * geom.length)
    np.testing.assert_allclose(pl[0], (start_pt.x, start_pt.y), atol=1e-7)
    # First vertex after walking backward must be N1's position
    # (-99.135, 19.430) — edge 9's from_node end.
    assert any(np.allclose(v, (-99.135, 19.430), atol=1e-7) for v in pl)


def test_interpolation_walks_backroll_downward(grid_network):
    from src.api.interpolation import link_offset_at_fraction
    p = _vpath((9,), (True,), start=600.0, end=100.0, length=500.0)
    link, off = link_offset_at_fraction(p, grid_network, 0.5)
    assert link == 9
    assert off == pytest.approx(350.0)   # 600 − 0.5×500, walking DOWN


def test_interpolation_reversed_multi_edge(grid_network):
    from src.api.interpolation import link_offset_at_fraction
    # Reversed first edge 9 (600 → 0 at N1), then legal edge 2 to 50 m.
    p = _vpath((9, 2), (True, False), start=600.0, end=50.0, length=650.0)
    link, off = link_offset_at_fraction(p, grid_network, 0.0)
    assert (link, off) == (9, pytest.approx(600.0))
    link, off = link_offset_at_fraction(p, grid_network, 300.0 / 650.0)
    assert link == 9 and off == pytest.approx(300.0)   # walked downward
    link, off = link_offset_at_fraction(p, grid_network, 1.0)
    assert (link, off) == (2, pytest.approx(50.0))
