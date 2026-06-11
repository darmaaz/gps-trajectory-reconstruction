"""Path-feature vector ϕ(p) — schema and population."""

from __future__ import annotations

import numpy as np
import pytest

from src.model import FEATURE_DIM, Path
from src.model.features import path_features


def _path(
    edges, *, length_m=100.0, travel_time=10.0, budget=10.0,
    start_offset=0.0, end_offset=0.0,
    start_perp_m=0.0, end_perp_m=0.0,
):
    return Path(
        edges=tuple(edges),
        start_offset=start_offset,
        end_offset=end_offset,
        expected_travel_time=travel_time,
        length_meters=length_m,
        feature_vector=np.zeros(0),
        time_budget=budget,
        start_perp_m=start_perp_m,
        end_perp_m=end_perp_m,
    )


def test_feature_dim_is_nineteen():
    """The schema is co-versioned with `mu`. Bumping FEATURE_DIM is a
    breaking change — this test gates accidental schema drift.

    18 → 19 was a conscious bump: slot [18] = n_direction_violations
    (F5, direction-violation candidates). Stored 18-dim μ files are
    padded by `data.default_mu()` with the hand prior until retrain."""
    assert FEATURE_DIM == 19


def test_path_features_length_and_travel_slots(grid_network):
    p = _path([1], length_m=250.0, travel_time=18.0, budget=20.0)
    feats = path_features(p, grid_network)
    assert feats.shape == (FEATURE_DIM,)
    # Slots [0] and [12] are now scaled (length / 1000, travel / 60).
    assert feats[0] == pytest.approx(250.0 / 1000.0)    # length_km
    assert feats[12] == pytest.approx(18.0 / 60.0)      # travel_time_min


def test_path_features_dwell_slots_positive(grid_network):
    # budget=30, travel=20 → dwell=10s → 10/60 min ; ratio = 10/30 = 0.333
    p = _path([1], travel_time=20.0, budget=30.0)
    feats = path_features(p, grid_network)
    assert feats[13] == pytest.approx(10.0 / 60.0)     # dwell in minutes
    assert feats[14] == pytest.approx(10.0 / 30.0)     # ratio unchanged


def test_path_features_dwell_slots_zero_when_overslacked(grid_network):
    # travel exceeds budget → dwell is clamped to 0 → ratio is 0.
    p = _path([1], travel_time=120.0, budget=80.0)
    feats = path_features(p, grid_network)
    assert feats[13] == pytest.approx(0.0)
    assert feats[14] == pytest.approx(0.0)


def test_path_features_dwell_ratio_zero_when_budget_zero(grid_network):
    # Default test fixtures may construct Path with time_budget=0;
    # ratio must not divide-by-zero.
    p = _path([1], travel_time=10.0, budget=0.0)
    feats = path_features(p, grid_network)
    assert feats[14] == pytest.approx(0.0)
    assert np.isfinite(feats[14])


def test_path_features_perp_distance_slots(grid_network):
    p = _path([1], start_perp_m=7.5, end_perp_m=3.2)
    feats = path_features(p, grid_network)
    # Perp slots scaled by 1/10 (ten-metre units).
    assert feats[15] == pytest.approx(7.5 / 10.0)
    assert feats[16] == pytest.approx(3.2 / 10.0)


def test_default_mu_returns_feature_dim_vector():
    """`src.data.default_mu()` returns either the shipped trained vector
    or a zero fallback — but always shape `(FEATURE_DIM,)` float."""
    from src.data import default_mu
    mu = default_mu()
    assert mu.shape == (FEATURE_DIM,)
    assert mu.dtype == np.float64 or mu.dtype == float
    assert np.all(np.isfinite(mu))


def test_project_observation_populates_perp_m(grid_network, t0):
    """`project_observation` should store the projection distance on each
    returned State, not discard it as the old code did."""
    from datetime import timedelta
    from src.model import CollapsedObservation
    from src.candidates import project_observation

    obs = CollapsedObservation(
        lat=19.4305,
        lon=-99.132,
        t_first=t0,
        t_last=t0 + timedelta(seconds=1),
        collapsed_count=1,
    )
    states = project_observation(
        obs, grid_network, radius_meters=200.0, max_candidates=3,
    )
    assert len(states) > 0
    for s in states:
        assert hasattr(s, "perp_m")
        assert s.perp_m >= 0.0
        assert np.isfinite(s.perp_m)
    # Closest candidate should have the smallest perp_m.
    perps = [s.perp_m for s in states]
    assert perps == sorted(perps)
