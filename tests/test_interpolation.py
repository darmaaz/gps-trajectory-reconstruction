"""Tests for src/api/interpolation.py — MLE position-at-time."""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pytest

from src.api.interpolation import (
    _position_on_edge,
    interpolate_along_path,
    position_at_time,
    position_in_transition,
)
from src.api.pipeline import TrajectoryPosterior
from src.geo import haversine_m
from src.model import FEATURE_DIM, Path, StateV1


def _path(edges, start_offset, end_offset, length_m, *, time_budget=10.0):
    return Path(
        edges=tuple(edges),
        start_offset=start_offset,
        end_offset=end_offset,
        expected_travel_time=10.0,
        length_meters=length_m,
        feature_vector=np.zeros(FEATURE_DIM),
        time_budget=time_budget,
    )


# ------------------------------------------------------------- Path properties


class TestPathProperties:
    """`inferred_dwell` clamps to 0; `slack_deficit` reports the excess."""

    def test_inferred_dwell_positive_when_budget_exceeds_travel(self):
        p = Path(
            edges=(1,), start_offset=0.0, end_offset=0.0,
            expected_travel_time=30.0, length_meters=0.0,
            feature_vector=np.zeros(FEATURE_DIM), time_budget=80.0,
        )
        assert p.inferred_dwell == pytest.approx(50.0)
        assert not p.is_overslacked
        assert p.slack_deficit == pytest.approx(0.0)

    def test_inferred_dwell_clamps_to_zero_when_overslacked(self):
        # expected_travel_time exceeds time_budget — path admitted via
        # budget_slack. inferred_dwell reports 0; slack_deficit is the gap.
        p = Path(
            edges=(1,), start_offset=0.0, end_offset=0.0,
            expected_travel_time=120.0, length_meters=0.0,
            feature_vector=np.zeros(FEATURE_DIM), time_budget=80.0,
        )
        assert p.inferred_dwell == pytest.approx(0.0)
        assert p.is_overslacked
        assert p.slack_deficit == pytest.approx(40.0)

    def test_exact_budget_match_means_zero_dwell_no_slack(self):
        p = Path(
            edges=(1,), start_offset=0.0, end_offset=0.0,
            expected_travel_time=60.0, length_meters=0.0,
            feature_vector=np.zeros(FEATURE_DIM), time_budget=60.0,
        )
        assert p.inferred_dwell == pytest.approx(0.0)
        assert not p.is_overslacked
        assert p.slack_deficit == pytest.approx(0.0)

    def test_perp_distance_fields_default_to_zero(self):
        # Constructed without perp_m → defaults preserve old test fixtures.
        p = Path(
            edges=(1,), start_offset=0.0, end_offset=0.0,
            expected_travel_time=10.0, length_meters=0.0,
            feature_vector=np.zeros(FEATURE_DIM), time_budget=10.0,
        )
        assert p.start_perp_m == 0.0
        assert p.end_perp_m == 0.0

    def test_perp_distance_fields_preserve_explicit_values(self):
        p = Path(
            edges=(1,), start_offset=0.0, end_offset=0.0,
            expected_travel_time=10.0, length_meters=0.0,
            feature_vector=np.zeros(FEATURE_DIM), time_budget=10.0,
            start_perp_m=12.5, end_perp_m=8.0,
        )
        assert p.start_perp_m == pytest.approx(12.5)
        assert p.end_perp_m == pytest.approx(8.0)


# ----------------------------------------------------- interpolate_along_path


class TestInterpolateAlongPath:
    def test_single_edge_frac_zero_is_start_offset(self, grid_network):
        path = _path([1], 100.0, 200.0, 100.0)
        lat, lon = interpolate_along_path(path, grid_network, 0.0)
        idx = grid_network.edge_index_for_link(1)
        exp_lat, exp_lon = _position_on_edge(grid_network, idx, 100.0)
        assert lat == pytest.approx(exp_lat)
        assert lon == pytest.approx(exp_lon)

    def test_single_edge_frac_one_is_end_offset(self, grid_network):
        path = _path([1], 100.0, 200.0, 100.0)
        lat, lon = interpolate_along_path(path, grid_network, 1.0)
        idx = grid_network.edge_index_for_link(1)
        exp_lat, exp_lon = _position_on_edge(grid_network, idx, 200.0)
        assert lat == pytest.approx(exp_lat)
        assert lon == pytest.approx(exp_lon)

    def test_multi_edge_frac_zero_is_path_start(self, grid_network):
        # Edges 1 (0→1) then 9 (1→4). Start at offset 50 on edge 1.
        e1_len = float(grid_network.lengths_m[grid_network.edge_index_for_link(1)])
        e9_len = float(grid_network.lengths_m[grid_network.edge_index_for_link(9)])
        path = _path([1, 9], 50.0, e9_len * 0.5, (e1_len - 50.0) + e9_len * 0.5)
        lat, lon = interpolate_along_path(path, grid_network, 0.0)
        idx = grid_network.edge_index_for_link(1)
        exp = _position_on_edge(grid_network, idx, 50.0)
        assert (lat, lon) == pytest.approx(exp)

    def test_multi_edge_frac_one_is_path_end(self, grid_network):
        e1_len = float(grid_network.lengths_m[grid_network.edge_index_for_link(1)])
        e9_len = float(grid_network.lengths_m[grid_network.edge_index_for_link(9)])
        path = _path([1, 9], 50.0, e9_len * 0.5, (e1_len - 50.0) + e9_len * 0.5)
        lat, lon = interpolate_along_path(path, grid_network, 1.0)
        idx = grid_network.edge_index_for_link(9)
        exp = _position_on_edge(grid_network, idx, e9_len * 0.5)
        assert (lat, lon) == pytest.approx(exp)

    def test_multi_edge_at_internal_boundary(self, grid_network):
        # Path runs full length of edge 1 then full length of edge 9.
        # At frac = (e1_len)/(e1_len + e9_len) we should be at the boundary
        # node, which is shared between the two edges.
        e1_len = float(grid_network.lengths_m[grid_network.edge_index_for_link(1)])
        e9_len = float(grid_network.lengths_m[grid_network.edge_index_for_link(9)])
        path = _path([1, 9], 0.0, e9_len, e1_len + e9_len)
        frac = e1_len / (e1_len + e9_len)
        lat, lon = interpolate_along_path(path, grid_network, frac)
        # The boundary node is N1 = (19.430, -99.135) per conftest.
        assert lat == pytest.approx(19.430, abs=1e-5)
        assert lon == pytest.approx(-99.135, abs=1e-5)

    def test_frac_is_clamped(self, grid_network):
        path = _path([1], 100.0, 200.0, 100.0)
        lo = interpolate_along_path(path, grid_network, -0.5)
        hi = interpolate_along_path(path, grid_network, 1.5)
        assert lo == interpolate_along_path(path, grid_network, 0.0)
        assert hi == interpolate_along_path(path, grid_network, 1.0)

    def test_empty_path_raises(self, grid_network):
        path = _path([], 0.0, 0.0, 0.0)
        with pytest.raises(ValueError):
            interpolate_along_path(path, grid_network, 0.5)

    def test_intermediate_frac_is_between_endpoints(self, grid_network):
        # Sanity: a midpoint should be roughly halfway by haversine distance
        # from the start to the end.
        e1_len = float(grid_network.lengths_m[grid_network.edge_index_for_link(1)])
        e9_len = float(grid_network.lengths_m[grid_network.edge_index_for_link(9)])
        path = _path([1, 9], 0.0, e9_len, e1_len + e9_len)
        lat0, lon0 = interpolate_along_path(path, grid_network, 0.0)
        lat1, lon1 = interpolate_along_path(path, grid_network, 1.0)
        latm, lonm = interpolate_along_path(path, grid_network, 0.5)
        d_total = haversine_m(lat0, lon0, lat1, lon1)
        d_mid_to_start = haversine_m(lat0, lon0, latm, lonm)
        # The two-edge path turns at the boundary node, so the midpoint
        # is not on the chord — but it should be at most as far from
        # start as the total chord length.
        assert d_mid_to_start < d_total


# --------------------------------------------------------- position_at_time


def _state(link_id, offset, t):
    return StateV1(link_id=link_id, offset=offset, entry_time=t)


def _segment(timestamps, most_likely, *, t_last=None):
    """Construct a synthetic Posterior. `t_last` defaults via the
    dataclass invariant (equals `canonical_timestamps` → zero confirmed
    dwell). Pass explicitly to exercise non-zero confirmed dwell."""
    return TrajectoryPosterior(
        state_marginals=[{} for _ in timestamps],
        path_marginals=[{} for _ in range(len(timestamps) - 1)],
        most_likely=most_likely,
        log_partition=0.0,
        canonical_timestamps=tuple(timestamps),
        canonical_t_last=tuple(t_last) if t_last is not None else (),
    )


class TestPositionAtTime:
    def test_exact_obs_timestamp_returns_state_position(
        self, grid_network, t0,
    ):
        e1_len = float(grid_network.lengths_m[grid_network.edge_index_for_link(1)])
        e9_len = float(grid_network.lengths_m[grid_network.edge_index_for_link(9)])
        s0 = _state(1, 0.0, t0)
        s1 = _state(9, e9_len, t0 + timedelta(seconds=10))
        p = _path([1, 9], 0.0, e9_len, e1_len + e9_len)
        seg = _segment([t0, t0 + timedelta(seconds=10)], [s0, p, s1])
        assert position_at_time([seg], t0, grid_network) == pytest.approx(
            _position_on_edge(grid_network, grid_network.edge_index_for_link(1), 0.0)
        )

    def test_mid_transition_interpolates(self, grid_network, t0):
        e1_len = float(grid_network.lengths_m[grid_network.edge_index_for_link(1)])
        e9_len = float(grid_network.lengths_m[grid_network.edge_index_for_link(9)])
        total_len = e1_len + e9_len
        s0 = _state(1, 0.0, t0)
        s1 = _state(9, e9_len, t0 + timedelta(seconds=10))
        p = _path([1, 9], 0.0, e9_len, total_len)
        seg = _segment([t0, t0 + timedelta(seconds=10)], [s0, p, s1])
        # At t0+5s, frac=0.5. Should match interpolate_along_path frac=0.5.
        got = position_at_time(
            [seg], t0 + timedelta(seconds=5), grid_network,
        )
        expected = interpolate_along_path(p, grid_network, 0.5)
        assert got == pytest.approx(expected)

    def test_outside_segment_span_returns_none(self, grid_network, t0):
        e1_len = float(grid_network.lengths_m[grid_network.edge_index_for_link(1)])
        s0 = _state(1, 0.0, t0)
        s1 = _state(1, e1_len, t0 + timedelta(seconds=10))
        p = _path([1], 0.0, e1_len, e1_len)
        seg = _segment([t0, t0 + timedelta(seconds=10)], [s0, p, s1])
        assert position_at_time(
            [seg], t0 - timedelta(seconds=1), grid_network,
        ) is None
        assert position_at_time(
            [seg], t0 + timedelta(seconds=11), grid_network,
        ) is None

    def test_gap_between_segments_returns_none(self, grid_network, t0):
        # Two segments with a 5s gap in the middle.
        e1_len = float(grid_network.lengths_m[grid_network.edge_index_for_link(1)])
        s_a0 = _state(1, 0.0, t0)
        s_a1 = _state(1, e1_len, t0 + timedelta(seconds=10))
        p_a = _path([1], 0.0, e1_len, e1_len)
        seg_a = _segment([t0, t0 + timedelta(seconds=10)], [s_a0, p_a, s_a1])

        s_b0 = _state(3, 0.0, t0 + timedelta(seconds=15))
        s_b1 = _state(3, e1_len, t0 + timedelta(seconds=25))
        p_b = _path([3], 0.0, e1_len, e1_len)
        seg_b = _segment(
            [t0 + timedelta(seconds=15), t0 + timedelta(seconds=25)],
            [s_b0, p_b, s_b1],
        )
        # t in the gap (12s after t0) → no segment covers it.
        assert position_at_time(
            [seg_a, seg_b], t0 + timedelta(seconds=12), grid_network,
        ) is None
        # t inside seg_b is covered.
        assert position_at_time(
            [seg_a, seg_b], t0 + timedelta(seconds=20), grid_network,
        ) is not None

    def test_single_obs_segment_only_resolves_at_its_timestamp(
        self, grid_network, t0,
    ):
        s0 = _state(1, 0.0, t0)
        seg = _segment([t0], [s0])
        assert position_at_time([seg], t0, grid_network) is not None
        assert position_at_time(
            [seg], t0 + timedelta(seconds=1), grid_network,
        ) is None


# ---------------------------------------------------- position_in_transition


class TestPositionInTransition:
    """Cover the three dwell-allocation rules on a single multi-edge path.

    Setup: a path along edges 1→9 with `expected_travel_time=20s` and
    `time_budget=80s`, so `D_p = 60s`. At τ=40s:

    - front: still in dwell window (40 ≤ 60) → at origin
    - back:  past transit window (40 ≥ 20) → at destination
    - spread: frac = 40/80 = 0.5 → midpoint along the path's geometry
    """

    def _multi_edge_path(self, network, *, t_p, budget):
        e1_len = float(network.lengths_m[network.edge_index_for_link(1)])
        e9_len = float(network.lengths_m[network.edge_index_for_link(9)])
        return Path(
            edges=(1, 9),
            start_offset=0.0,
            end_offset=e9_len,
            expected_travel_time=t_p,
            length_meters=e1_len + e9_len,
            feature_vector=np.zeros(FEATURE_DIM),
            time_budget=budget,
        )

    def test_front_rule_in_dwell_window_returns_origin(self, grid_network):
        p = self._multi_edge_path(grid_network, t_p=20.0, budget=80.0)
        link_id, offset = position_in_transition(
            p, grid_network, tau_s=40.0, rule="front",
        )
        assert link_id == 1
        assert offset == pytest.approx(0.0)

    def test_front_rule_past_dwell_walks_path(self, grid_network):
        p = self._multi_edge_path(grid_network, t_p=40.0, budget=80.0)
        # D_p = 40, t_p = 40. τ=60 → 20s into 40s of travel → frac=0.5.
        link_id, offset = position_in_transition(
            p, grid_network, tau_s=60.0, rule="front",
        )
        e1_len = float(grid_network.lengths_m[grid_network.edge_index_for_link(1)])
        total_len = e1_len + float(
            grid_network.lengths_m[grid_network.edge_index_for_link(9)],
        )
        midpoint_target = 0.5 * total_len
        if midpoint_target <= e1_len:
            assert link_id == 1
            assert offset == pytest.approx(midpoint_target)
        else:
            assert link_id == 9
            assert offset == pytest.approx(midpoint_target - e1_len)

    def test_back_rule_past_transit_returns_destination(self, grid_network):
        p = self._multi_edge_path(grid_network, t_p=20.0, budget=80.0)
        link_id, offset = position_in_transition(
            p, grid_network, tau_s=40.0, rule="back",
        )
        e9_len = float(grid_network.lengths_m[grid_network.edge_index_for_link(9)])
        assert link_id == 9
        assert offset == pytest.approx(e9_len)

    def test_back_rule_within_transit_walks_path(self, grid_network):
        p = self._multi_edge_path(grid_network, t_p=40.0, budget=80.0)
        # τ=20, t_p=40 → frac=0.5 along path.
        link_id, offset = position_in_transition(
            p, grid_network, tau_s=20.0, rule="back",
        )
        e1_len = float(grid_network.lengths_m[grid_network.edge_index_for_link(1)])
        e9_len = float(grid_network.lengths_m[grid_network.edge_index_for_link(9)])
        midpoint_target = 0.5 * (e1_len + e9_len)
        if midpoint_target <= e1_len:
            assert link_id == 1
        else:
            assert link_id == 9
            assert offset == pytest.approx(midpoint_target - e1_len)

    def test_spread_rule_uses_full_budget(self, grid_network):
        p = self._multi_edge_path(grid_network, t_p=20.0, budget=80.0)
        # frac = 40/80 = 0.5 regardless of D_p / t_p split.
        link_id, offset = position_in_transition(
            p, grid_network, tau_s=40.0, rule="spread",
        )
        e1_len = float(grid_network.lengths_m[grid_network.edge_index_for_link(1)])
        e9_len = float(grid_network.lengths_m[grid_network.edge_index_for_link(9)])
        midpoint_target = 0.5 * (e1_len + e9_len)
        if midpoint_target <= e1_len:
            assert link_id == 1
            assert offset == pytest.approx(midpoint_target)
        else:
            assert link_id == 9
            assert offset == pytest.approx(midpoint_target - e1_len)

    def test_zero_travel_time_returns_origin(self, grid_network):
        # Same-edge stay path: expected_travel_time=0.
        p = Path(
            edges=(1,),
            start_offset=50.0,
            end_offset=50.0,
            expected_travel_time=0.0,
            length_meters=0.0,
            feature_vector=np.zeros(FEATURE_DIM),
            time_budget=30.0,
        )
        for rule in ("front", "back", "spread"):
            link_id, offset = position_in_transition(
                p, grid_network, tau_s=15.0, rule=rule,
            )
            assert link_id == 1
            assert offset == pytest.approx(50.0)

    def test_unknown_rule_raises(self, grid_network):
        p = self._multi_edge_path(grid_network, t_p=20.0, budget=80.0)
        with pytest.raises(ValueError):
            position_in_transition(p, grid_network, 10.0, rule="middle")  # type: ignore[arg-type]


# --------------------------------- position_at_time confirmed-dwell anchor


class TestPositionAtTimeConfirmedDwell:
    """`canonical_t_last` splits the off-grid region into a confirmed-dwell
    half (vehicle at obs k's MLE state) and a transit half (dispatch via
    `position_in_transition` with τ anchored at t_last)."""

    def test_confirmed_dwell_returns_origin_state_position(
        self, grid_network, t0,
    ):
        # gap = 80s; first 60s is confirmed dwell at obs k. Transit budget
        # is 20s. Querying inside the confirmed window should return obs
        # k's MLE state position, regardless of path geometry.
        e1_len = float(grid_network.lengths_m[grid_network.edge_index_for_link(1)])
        e9_len = float(grid_network.lengths_m[grid_network.edge_index_for_link(9)])
        s0 = _state(1, 0.0, t0)
        s1 = _state(9, e9_len, t0 + timedelta(seconds=80))
        p = _path([1, 9], 0.0, e9_len, e1_len + e9_len, time_budget=20.0)
        seg = _segment(
            [t0, t0 + timedelta(seconds=80)], [s0, p, s1],
            t_last=[t0 + timedelta(seconds=60), t0 + timedelta(seconds=80)],
        )
        # τ=30s into the full gap is inside confirmed dwell.
        got = position_at_time(
            [seg], t0 + timedelta(seconds=30), grid_network, rule="front",
        )
        expected = _position_on_edge(
            grid_network, grid_network.edge_index_for_link(1), 0.0,
        )
        assert got == pytest.approx(expected)

    def test_transit_window_uses_t_last_anchor(self, grid_network, t0):
        # 60s confirmed dwell + 20s transit. Path travel_time=10, so
        # inferred_dwell=10 under the 20s budget. Under "front", τ=10 (i.e.
        # t0+70s) is exactly at the boundary of inferred dwell. τ=15
        # (i.e. t0+75s) lands at frac=(15-10)/10 = 0.5 along the path.
        e1_len = float(grid_network.lengths_m[grid_network.edge_index_for_link(1)])
        e9_len = float(grid_network.lengths_m[grid_network.edge_index_for_link(9)])
        s0 = _state(1, 0.0, t0)
        s1 = _state(9, e9_len, t0 + timedelta(seconds=80))
        p = _path([1, 9], 0.0, e9_len, e1_len + e9_len, time_budget=20.0)
        seg = _segment(
            [t0, t0 + timedelta(seconds=80)], [s0, p, s1],
            t_last=[t0 + timedelta(seconds=60), t0 + timedelta(seconds=80)],
        )
        got = position_at_time(
            [seg], t0 + timedelta(seconds=75), grid_network, rule="front",
        )
        expected = interpolate_along_path(p, grid_network, 0.5)
        assert got == pytest.approx(expected)

    def test_unset_t_last_defaults_to_zero_confirmed_dwell(
        self, grid_network, t0,
    ):
        # Fixture constructed without canonical_t_last → dataclass invariant
        # defaults it to canonical_timestamps (zero confirmed dwell). With
        # time_budget=10 and travel_time=10, inferred_dwell=0 → frac=tau/t_p.
        # At τ=5, frac=0.5.
        e1_len = float(grid_network.lengths_m[grid_network.edge_index_for_link(1)])
        e9_len = float(grid_network.lengths_m[grid_network.edge_index_for_link(9)])
        s0 = _state(1, 0.0, t0)
        s1 = _state(9, e9_len, t0 + timedelta(seconds=10))
        p = _path([1, 9], 0.0, e9_len, e1_len + e9_len)
        seg = _segment([t0, t0 + timedelta(seconds=10)], [s0, p, s1])
        got = position_at_time(
            [seg], t0 + timedelta(seconds=5), grid_network, rule="front",
        )
        expected = interpolate_along_path(p, grid_network, 0.5)
        assert got == pytest.approx(expected)
