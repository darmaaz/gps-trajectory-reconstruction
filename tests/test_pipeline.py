"""End-to-end `reconstruct_trajectory` and `TrajectoryPosterior`.

Exercises the orchestrator on the synthetic grid:
    - clean trip → single segment
    - off-network ping → split into two segments
    - all-junk input → empty list
    - single observation → emission-only segment
    - MarginalQuery API behaviour, including dwell-aware `at_time`
"""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pytest

from src.api import TrajectoryPosterior, reconstruct_trajectory
from src.config import Config
from src.model import (
    ExponentialFamilyTransition, FEATURE_DIM, Path, RawObservation, StateV1,
    StudentTEmission,
)


@pytest.fixture
def config(grid_network):
    emit = StudentTEmission(scale=15.0, network=grid_network, df=4.0)
    mu = np.zeros(FEATURE_DIM)
    mu[0] = -0.001    # mild length penalty
    mu[1] = mu[2] = -0.5    # turns
    mu[12] = -0.01    # mild travel-time penalty
    return Config(emission=emit, transition=ExponentialFamilyTransition(mu))


def _raw(t0, secs, lat, lon, hdop=2.0):
    return RawObservation(
        timestamp=t0 + timedelta(seconds=secs), lat=lat, lon=lon, hdop=hdop,
    )


def test_reconstruct_clean_trip_single_segment(grid_network, config, t0):
    raw = [
        _raw(t0, 0,   19.430, -99.1349),
        _raw(t0, 60,  19.430, -99.130),
        _raw(t0, 120, 19.430, -99.1251),
    ]
    segs = reconstruct_trajectory(raw, grid_network, config)
    assert len(segs) == 1
    seg = segs[0]
    assert len(seg.state_marginals) == 3
    assert len(seg.path_marginals) == 2
    assert len(seg.most_likely) == 5    # 2T-1
    for m in seg.state_marginals:
        assert sum(m.values()) == pytest.approx(1.0, abs=1e-9)
    assert seg.observation_indices == (0, 3)


def test_reconstruct_splits_at_off_network(grid_network, config, t0):
    raw = [
        _raw(t0, 0,   19.430, -99.1349),
        _raw(t0, 60,  19.430, -99.130),
        _raw(t0, 120, 19.450, -99.150),    # ~3 km off-network
        _raw(t0, 180, 19.430, -99.1255),
        _raw(t0, 240, 19.430, -99.1251),
    ]
    segs = reconstruct_trajectory(raw, grid_network, config)
    assert len(segs) == 2
    # Off-network observation (index 2) must not be in either segment
    for seg in segs:
        start, end = seg.observation_indices
        assert 2 not in range(start, end)


def test_reconstruct_empty_input_yields_empty_segments(grid_network, config, t0):
    junk = [
        RawObservation(timestamp=t0, lat=0.0, lon=0.0),    # sentinel
        RawObservation(timestamp=t0 + timedelta(seconds=1), lat=200.0, lon=0.0),    # OOR
        _raw(t0, 2, 19.43, -99.13, hdop=999.0),    # high HDOP
    ]
    assert reconstruct_trajectory(junk, grid_network, config) == []


def test_reconstruct_single_observation(grid_network, config, t0):
    segs = reconstruct_trajectory([_raw(t0, 0, 19.430, -99.130)],
                                  grid_network, config)
    assert len(segs) == 1
    seg = segs[0]
    assert len(seg.state_marginals) == 1
    assert seg.path_marginals == []
    assert len(seg.most_likely) == 1


def test_reconstruct_is_deterministic(grid_network, config, t0):
    raw = [
        _raw(t0, 0,   19.430, -99.1349),
        _raw(t0, 60,  19.430, -99.130),
        _raw(t0, 120, 19.430, -99.1251),
    ]
    a = reconstruct_trajectory(raw, grid_network, config)[0]
    b = reconstruct_trajectory(raw, grid_network, config)[0]
    assert a.log_partition == pytest.approx(b.log_partition, abs=1e-12)
    for ma, mb in zip(a.state_marginals, b.state_marginals):
        for state, p in ma.items():
            assert p == pytest.approx(mb[state], abs=1e-12)


# ---------------------------------------------------------- MarginalQuery


def test_marginal_query_at_observation_and_time(grid_network, config, t0):
    raw = [
        _raw(t0, 0,   19.430, -99.1349),
        _raw(t0, 60,  19.430, -99.130),
        _raw(t0, 120, 19.430, -99.1251),
    ]
    seg: TrajectoryPosterior = reconstruct_trajectory(
        raw, grid_network, config,
    )[0]
    assert seg.at_observation(0) is seg.state_marginals[0]
    assert seg.at_time(seg.canonical_timestamps[1]) is seg.state_marginals[1]


def test_marginal_query_at_time_out_of_segment_raises(
    grid_network, config, t0,
):
    raw = [
        _raw(t0, 0,   19.430, -99.1349),
        _raw(t0, 60,  19.430, -99.130),
    ]
    seg = reconstruct_trajectory(raw, grid_network, config)[0]
    with pytest.raises(ValueError):
        seg.at_time(t0 + timedelta(seconds=999))


# ------------------------------------------------------------ stale flag


def test_stale_flag_surfaces_in_posterior(grid_network, config, t0):
    """A trip whose first observation is a long frozen run followed by a
    short-gap recovery jump should produce a posterior whose first
    observation is in `stale_observation_indices`.

    Construction: 20 raw pings at the same anchor over 200 s (collapses to
    one observation with t_first..t_last spanning 200 s), then a single
    ping ~1.1 km away 10 s later. `min_travel(A→B)` is ~66 s; the 10 s
    gap from t_last is infeasible, the 210 s gap from t_first is feasible.
    """
    base_ts = t0
    raw = []
    for i in range(20):
        raw.append(_raw(base_ts, i * 10.0, 19.430, -99.1349))
    raw.append(_raw(base_ts, 210.0, 19.430, -99.1251))

    segs = reconstruct_trajectory(raw, grid_network, config)
    assert len(segs) == 1
    seg = segs[0]
    assert seg.stale_observation_indices == (0,)
    assert seg.stale_fraction == pytest.approx(0.5)


def test_stale_flag_absent_for_clean_trip(grid_network, config, t0):
    raw = [
        _raw(t0, 0,   19.430, -99.1349),
        _raw(t0, 60,  19.430, -99.130),
        _raw(t0, 120, 19.430, -99.1251),
    ]
    seg = reconstruct_trajectory(raw, grid_network, config)[0]
    assert seg.stale_observation_indices == ()
    assert seg.stale_fraction == 0.0


# ------------------------------------------------- dwell-aware at_time(t)
#
# These tests construct TrajectoryPosterior fixtures directly with manually
# populated path_marginals so the front-loaded dwell rule can be exercised
# without running the orchestrator. Conventions: edges 1 and 9 of the grid
# network (see conftest.py), two-edge paths whose `time_budget` exceeds
# `expected_travel_time` so `inferred_dwell > 0`.


def _path(edges, start_offset, end_offset, length_m, travel_time, time_budget):
    return Path(
        edges=tuple(edges),
        start_offset=start_offset,
        end_offset=end_offset,
        expected_travel_time=travel_time,
        length_meters=length_m,
        feature_vector=np.zeros(FEATURE_DIM),
        time_budget=time_budget,
    )


def _state(link_id, offset, t):
    return StateV1(link_id=link_id, offset=offset, entry_time=t)


def _posterior(
    timestamps, path_marginal_per_transition, network, *, t_last=None,
):
    """Synthetic Posterior with the given timestamps and path marginals.

    `t_last` defaults to `timestamps` (zero confirmed dwell). Pass an
    explicit tuple to exercise the confirmed-dwell window logic.
    """
    state_marginals = [{} for _ in timestamps]
    return TrajectoryPosterior(
        state_marginals=state_marginals,
        path_marginals=list(path_marginal_per_transition),
        most_likely=[],
        log_partition=0.0,
        canonical_timestamps=tuple(timestamps),
        canonical_t_last=tuple(t_last) if t_last is not None else tuple(timestamps),
        network=network,
    )


class TestPosteriorAtTime:
    def test_on_grid_returns_state_marginal_exactly(self, grid_network, t0):
        seg = _posterior([t0, t0 + timedelta(seconds=10)], [{}], grid_network)
        seg.state_marginals[0] = {_state(1, 0.0, t0): 1.0}
        assert seg.at_time(t0) is seg.state_marginals[0]

    def test_dwell_window_concentrates_at_path_origin(self, grid_network, t0):
        # Single path with travel_time=20s, time_budget=80s → dwell=60s.
        # At τ=30s (within dwell window), the vehicle should be at the
        # path's origin (link 1, offset 0).
        e1_len = float(grid_network.lengths_m[grid_network.edge_index_for_link(1)])
        e9_len = float(grid_network.lengths_m[grid_network.edge_index_for_link(9)])
        p = _path([1, 9], 0.0, e9_len, e1_len + e9_len, 20.0, 80.0)
        seg = _posterior(
            [t0, t0 + timedelta(seconds=80)], [{p: 1.0}], grid_network,
        )
        marg = seg.at_time(t0 + timedelta(seconds=30))
        assert len(marg) == 1
        ((state, weight),) = marg.items()
        assert state.link_id == 1
        assert state.offset == pytest.approx(0.0)
        assert weight == pytest.approx(1.0)

    def test_travel_window_uses_path_geometry(self, grid_network, t0):
        # Same path; at τ=50s we're 30s into travel (D=60), with travel_time=20s.
        # Wait, that's > travel_time — clamps to frac=1.0 → path end.
        # Use a longer travel_time to land mid-path: travel_time=40, dwell=40.
        e1_len = float(grid_network.lengths_m[grid_network.edge_index_for_link(1)])
        e9_len = float(grid_network.lengths_m[grid_network.edge_index_for_link(9)])
        p = _path([1, 9], 0.0, e9_len, e1_len + e9_len, 40.0, 80.0)
        seg = _posterior(
            [t0, t0 + timedelta(seconds=80)], [{p: 1.0}], grid_network,
        )
        # τ=60s, dwell=40s, so 20s into 40s of travel → frac=0.5 along path.
        marg = seg.at_time(t0 + timedelta(seconds=60))
        assert len(marg) == 1
        ((state, weight),) = marg.items()
        assert weight == pytest.approx(1.0)
        # frac=0.5 of (e1_len + e9_len) means we're past edge 1's full length
        # (the path starts at offset 0 on edge 1 so all of edge 1 belongs to it).
        midpoint_target = 0.5 * (e1_len + e9_len)
        if midpoint_target <= e1_len:
            assert state.link_id == 1
            assert state.offset == pytest.approx(midpoint_target)
        else:
            assert state.link_id == 9
            assert state.offset == pytest.approx(midpoint_target - e1_len)

    def test_mixed_dwell_and_travel_aggregates(self, grid_network, t0):
        # Two paths: p_short (travel=20s, dwell=60s) and p_long (travel=70s, dwell=10s).
        # At τ=40s, p_short is in travel (40-60=−20 → actually still dwelling),
        # p_long is in travel (40 > 10 → frac=(40-10)/70). So actually p_short
        # dwells (τ ≤ D_p), p_long travels.
        e1_len = float(grid_network.lengths_m[grid_network.edge_index_for_link(1)])
        e9_len = float(grid_network.lengths_m[grid_network.edge_index_for_link(9)])
        total = e1_len + e9_len
        p_short = _path([1, 9], 0.0, e9_len, total, 20.0, 80.0)
        p_long = _path([1, 9], 0.0, e9_len, total, 70.0, 80.0)
        seg = _posterior(
            [t0, t0 + timedelta(seconds=80)],
            [{p_short: 0.6, p_long: 0.4}],
            grid_network,
        )
        marg = seg.at_time(t0 + timedelta(seconds=40))
        # p_short's dwell = 60s, τ=40 ≤ 60 → at origin (link 1, offset 0)
        # p_long's dwell = 10s, τ=40 > 10 → traveling at frac=(40-10)/70 ≈ 0.4286
        assert len(marg) == 2
        origin_mass = sum(
            w for s, w in marg.items() if s.link_id == 1 and s.offset == 0.0
        )
        travel_mass = sum(
            w for s, w in marg.items() if not (s.link_id == 1 and s.offset == 0.0)
        )
        assert origin_mass == pytest.approx(0.6)
        assert travel_mass == pytest.approx(0.4)

    def test_weights_sum_to_one(self, grid_network, t0):
        e1_len = float(grid_network.lengths_m[grid_network.edge_index_for_link(1)])
        e9_len = float(grid_network.lengths_m[grid_network.edge_index_for_link(9)])
        total = e1_len + e9_len
        paths = {
            _path([1, 9], 0.0, e9_len, total, 20.0, 80.0): 0.5,
            _path([1, 9], 0.0, e9_len, total, 40.0, 80.0): 0.3,
            _path([1, 9], 0.0, e9_len, total, 70.0, 80.0): 0.2,
        }
        seg = _posterior(
            [t0, t0 + timedelta(seconds=80)], [paths], grid_network,
        )
        marg = seg.at_time(t0 + timedelta(seconds=50))
        assert sum(marg.values()) == pytest.approx(1.0)

    def test_out_of_segment_raises(self, grid_network, t0):
        e1_len = float(grid_network.lengths_m[grid_network.edge_index_for_link(1)])
        p = _path([1], 0.0, e1_len, e1_len, 10.0, 10.0)
        seg = _posterior(
            [t0, t0 + timedelta(seconds=10)], [{p: 1.0}], grid_network,
        )
        with pytest.raises(ValueError):
            seg.at_time(t0 - timedelta(seconds=1))
        with pytest.raises(ValueError):
            seg.at_time(t0 + timedelta(seconds=11))

    def test_off_grid_without_network_raises(self, grid_network, t0):
        e1_len = float(grid_network.lengths_m[grid_network.edge_index_for_link(1)])
        p = _path([1], 0.0, e1_len, e1_len, 10.0, 10.0)
        seg = _posterior(
            [t0, t0 + timedelta(seconds=10)], [{p: 1.0}], None,
        )
        # on-grid still works
        seg.state_marginals[0] = {_state(1, 0.0, t0): 1.0}
        assert seg.at_time(t0) is seg.state_marginals[0]
        # off-grid requires network
        with pytest.raises(RuntimeError):
            seg.at_time(t0 + timedelta(seconds=5))

    def test_empty_segment_raises(self, grid_network, t0):
        seg = _posterior([], [], grid_network)
        with pytest.raises(ValueError):
            seg.at_time(t0)

    def test_synthetic_state_carries_queried_t_as_entry_time(self, grid_network, t0):
        e1_len = float(grid_network.lengths_m[grid_network.edge_index_for_link(1)])
        p = _path([1], 0.0, e1_len, e1_len, 10.0, 40.0)
        seg = _posterior(
            [t0, t0 + timedelta(seconds=40)], [{p: 1.0}], grid_network,
        )
        # Dwell window: τ=5 ≤ D=30 → at origin, entry_time should be queried t.
        t_q = t0 + timedelta(seconds=5)
        marg = seg.at_time(t_q)
        ((state, _),) = marg.items()
        assert state.entry_time == t_q


class TestPosteriorAtTimeConfirmedDwell:
    """Exercise the confirmed-dwell anchor: `canonical_t_last` splits the
    region between two observations into a confirmed half (return at-obs
    marginal) and a transit half (dispatch via `position_in_transition`
    with τ anchored to `t_last`)."""

    def test_confirmed_dwell_window_returns_at_obs_marginal(
        self, grid_network, t0,
    ):
        # gap = 100s; first 60s confirmed dwell at obs k; remaining 40s is
        # the transit budget. Path has travel_time=40, so inferred_dwell=0.
        e1_len = float(grid_network.lengths_m[grid_network.edge_index_for_link(1)])
        e9_len = float(grid_network.lengths_m[grid_network.edge_index_for_link(9)])
        p = _path([1, 9], 0.0, e9_len, e1_len + e9_len, 40.0, 40.0)
        seg = _posterior(
            [t0, t0 + timedelta(seconds=100)], [{p: 1.0}], grid_network,
            t_last=[t0 + timedelta(seconds=60), t0 + timedelta(seconds=100)],
        )
        anchor_state = _state(1, 0.0, t0)
        seg.state_marginals[0] = {anchor_state: 1.0}
        # τ=30 ≤ 60 → inside confirmed dwell → return state_marginals[0].
        marg = seg.at_time(t0 + timedelta(seconds=30))
        assert marg is seg.state_marginals[0]
        # τ=60 exactly: still inside confirmed dwell (≤).
        assert seg.at_time(t0 + timedelta(seconds=60)) is seg.state_marginals[0]

    def test_transit_window_uses_t_last_anchored_tau(self, grid_network, t0):
        # 60s confirmed dwell + 40s transit budget. Path travel_time=40,
        # so inferred_dwell=0 — vehicle traverses the entire path at uniform
        # pace over the transit window. At t_last+20s (i.e. t0+80s),
        # τ=20 into a 40s travel → frac=0.5 along path.
        e1_len = float(grid_network.lengths_m[grid_network.edge_index_for_link(1)])
        e9_len = float(grid_network.lengths_m[grid_network.edge_index_for_link(9)])
        p = _path([1, 9], 0.0, e9_len, e1_len + e9_len, 40.0, 40.0)
        seg = _posterior(
            [t0, t0 + timedelta(seconds=100)], [{p: 1.0}], grid_network,
            t_last=[t0 + timedelta(seconds=60), t0 + timedelta(seconds=100)],
        )
        marg = seg.at_time(t0 + timedelta(seconds=80))
        ((state, weight),) = marg.items()
        assert weight == pytest.approx(1.0)
        midpoint = 0.5 * (e1_len + e9_len)
        if midpoint <= e1_len:
            assert state.link_id == 1
            assert state.offset == pytest.approx(midpoint)
        else:
            assert state.link_id == 9
            assert state.offset == pytest.approx(midpoint - e1_len)

    def test_back_rule_during_confirmed_dwell_still_at_origin(
        self, grid_network, t0,
    ):
        # Confirmed dwell is a data fact, not a rule choice. Under "back"
        # the inferred-dwell allocation would normally put the vehicle at
        # the destination late in the budget, but during *confirmed* dwell
        # the vehicle is unambiguously at obs k.
        e1_len = float(grid_network.lengths_m[grid_network.edge_index_for_link(1)])
        e9_len = float(grid_network.lengths_m[grid_network.edge_index_for_link(9)])
        p = _path([1, 9], 0.0, e9_len, e1_len + e9_len, 20.0, 80.0)
        seg = _posterior(
            [t0, t0 + timedelta(seconds=140)], [{p: 1.0}], grid_network,
            t_last=[t0 + timedelta(seconds=60), t0 + timedelta(seconds=140)],
        )
        seg.state_marginals[0] = {_state(1, 0.0, t0): 1.0}
        marg = seg.at_time(t0 + timedelta(seconds=40), rule="back")
        assert marg is seg.state_marginals[0]

    def test_unset_t_last_defaults_to_zero_confirmed_dwell(
        self, grid_network, t0,
    ):
        # When canonical_t_last is omitted at construction, the dataclass
        # invariant defaults it to canonical_timestamps — i.e., zero
        # confirmed dwell, with the transit window covering the entire gap.
        e1_len = float(grid_network.lengths_m[grid_network.edge_index_for_link(1)])
        e9_len = float(grid_network.lengths_m[grid_network.edge_index_for_link(9)])
        p = _path([1, 9], 0.0, e9_len, e1_len + e9_len, 40.0, 80.0)
        seg = TrajectoryPosterior(
            state_marginals=[{}, {}],
            path_marginals=[{p: 1.0}],
            most_likely=[],
            log_partition=0.0,
            canonical_timestamps=(t0, t0 + timedelta(seconds=80)),
            network=grid_network,
        )
        # Invariant kicks in.
        assert seg.canonical_t_last == seg.canonical_timestamps
        # τ=60 from t_last (=t_first since zero dwell) → past D_p=40 → frac=0.5.
        marg = seg.at_time(t0 + timedelta(seconds=60))
        assert len(marg) == 1

    def test_mismatched_t_last_length_raises(self, t0):
        with pytest.raises(ValueError, match="canonical_t_last length"):
            TrajectoryPosterior(
                state_marginals=[{}, {}],
                path_marginals=[{}],
                most_likely=[],
                log_partition=0.0,
                canonical_timestamps=(t0, t0 + timedelta(seconds=10)),
                canonical_t_last=(t0,),  # length 1 vs 2 — invalid
            )
