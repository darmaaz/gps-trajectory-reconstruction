"""Validation primitives — Tier 1 sanity gates, Tier 3 baseline, Tier 4
supervised metrics. Tier 2 hold-out is exercised end-to-end via the
pipeline test in test_pipeline.py and via the synthetic fixture below.
"""

from __future__ import annotations

from collections import OrderedDict
from datetime import datetime, timedelta as _td

import numpy as np
import pytest

from src.model import FEATURE_DIM, Path, StateV1
from src.validation import (
    check_mu_signs,
    check_scale_bounds,
    credible_region_coverage,
    edge_disagreement_rate,
    path_miss_rate,
    point_miss_rate,
    posterior_top_k_rank,
)


# --------------------------------------------------------------------- sanity


def test_check_mu_signs_passes_on_calibration_priors():
    """Synthetic μ shaped like a successfully-trained driver model."""
    mu = np.zeros(FEATURE_DIM)
    mu[0] = -0.001    # length: prefer short
    mu[1] = -0.5      # turns: avoid
    mu[2] = -0.5
    mu[5] = 0.8       # motorway: prefer
    mu[6] = 0.5       # trunk: prefer
    mu[7] = 0.2       # primary: mildly prefer
    mu[10] = -0.4     # residential: avoid
    mu[11] = -0.8     # service: strongly avoid
    mu[12] = -0.01    # travel time: prefer fast
    results = check_mu_signs(mu)
    for slot in (
        "length_m", "n_left_turns", "n_right_turns",
        "frac_motorway", "frac_trunk", "frac_primary",
        "frac_residential", "frac_service", "expected_travel_time_s",
    ):
        assert results[slot]["passes"], (
            f"expected {slot} to pass, got {results[slot]}"
        )


def test_check_mu_signs_flags_wrong_signs():
    mu = np.zeros(FEATURE_DIM)
    mu[0] = +1.0      # length should be negative
    mu[5] = -1.0      # motorway should be positive
    mu[12] = +0.1     # travel time should be negative
    results = check_mu_signs(mu)
    assert not results["length_m"]["passes"]
    assert not results["frac_motorway"]["passes"]
    assert not results["expected_travel_time_s"]["passes"]
    # Slots with no sign expectation always pass
    assert results["n_signals"]["passes"]
    assert results["frac_secondary"]["passes"]


def test_check_mu_signs_zero_does_not_pass_signed_expectation():
    mu = np.zeros(FEATURE_DIM)
    results = check_mu_signs(mu)
    assert not results["length_m"]["passes"]
    assert not results["frac_motorway"]["passes"]


def test_check_scale_bounds():
    typical = check_scale_bounds(10.0)
    assert typical["passes"]
    assert "typical" in typical["note"]

    plausible = check_scale_bounds(40.0)
    assert plausible["passes"]
    assert "unusual" in plausible["note"]

    too_small = check_scale_bounds(0.1)
    assert not too_small["passes"]

    too_large = check_scale_bounds(500.0)
    assert not too_large["passes"]

    nan_or_zero = check_scale_bounds(0.0)
    assert not nan_or_zero["passes"]


# ------------------------------------------------------------------- baseline


def test_edge_disagreement_rate_basic(t0):
    snap = [
        StateV1(link_id=1, offset=0, entry_time=t0),
        StateV1(link_id=2, offset=0, entry_time=t0),
        StateV1(link_id=3, offset=0, entry_time=t0),
        None,
    ]
    viterbi = [
        StateV1(link_id=1, offset=10, entry_time=t0),    # agree
        StateV1(link_id=99, offset=0, entry_time=t0),    # disagree
        StateV1(link_id=3, offset=5, entry_time=t0),     # agree
        StateV1(link_id=4, offset=0, entry_time=t0),     # snap is None, skip
    ]
    rate = edge_disagreement_rate(snap, viterbi)
    assert rate == pytest.approx(1 / 3)


def test_edge_disagreement_rate_all_agree(t0):
    a = [StateV1(link_id=1, offset=0, entry_time=t0)] * 3
    b = [StateV1(link_id=1, offset=10, entry_time=t0)] * 3
    assert edge_disagreement_rate(a, b) == 0.0


def test_edge_disagreement_rate_empty():
    assert edge_disagreement_rate([], []) == 0.0


# ----------------------------------------------------------------- supervised


def _mk_path(edges, prob_for_test=None):
    """Synthetic Path object with empty feature vector — only `edges`
    matters for these tests."""
    return Path(
        edges=tuple(edges), start_offset=0.0, end_offset=0.0,
        expected_travel_time=0.0, length_meters=0.0,
        feature_vector=np.zeros(0),
    )


def test_path_miss_rate_perfect_match():
    rec = [1, 2, 3]
    gt = [1, 2, 3]
    assert path_miss_rate(rec, gt) == 0.0


def test_path_miss_rate_one_missing():
    assert path_miss_rate([1, 2], [1, 2, 3]) == pytest.approx(1 / 3)


def test_path_miss_rate_completely_wrong():
    assert path_miss_rate([99, 100], [1, 2, 3]) == 1.0


def test_path_miss_rate_empty_gt():
    assert path_miss_rate([1, 2], []) == 0.0


def test_posterior_top_k_rank_in_top():
    p1 = _mk_path([1, 2])
    p2 = _mk_path([1, 3])
    p3 = _mk_path([1, 4])
    marg = OrderedDict([(p1, 0.6), (p2, 0.3), (p3, 0.1)])
    assert posterior_top_k_rank(marg, (1, 2)) == 1
    assert posterior_top_k_rank(marg, (1, 3)) == 2
    assert posterior_top_k_rank(marg, (1, 4)) == 3


def test_posterior_top_k_rank_not_in_posterior():
    p1 = _mk_path([1, 2])
    marg = {p1: 1.0}
    assert posterior_top_k_rank(marg, (99, 100)) is None


def test_credible_region_coverage_at_top():
    p1 = _mk_path([1, 2])
    p2 = _mk_path([1, 3])
    marg = {p1: 0.6, p2: 0.4}
    assert credible_region_coverage(marg, (1, 2), level=0.5)
    assert credible_region_coverage(marg, (1, 2), level=0.9)
    assert credible_region_coverage(marg, (1, 3), level=0.9)


def test_credible_region_coverage_outside():
    p1 = _mk_path([1, 2])
    p2 = _mk_path([1, 3])
    p3 = _mk_path([1, 4])
    marg = {p1: 0.7, p2: 0.2, p3: 0.1}
    # Level 0.5 covers only p1
    assert credible_region_coverage(marg, (1, 2), level=0.5)
    assert not credible_region_coverage(marg, (1, 3), level=0.5)
    assert not credible_region_coverage(marg, (1, 4), level=0.5)


def test_credible_region_coverage_path_not_in_posterior():
    p1 = _mk_path([1, 2])
    marg = {p1: 1.0}
    assert not credible_region_coverage(marg, (99, 100), level=0.95)


# --------------------------------------------------------------- hold-out (e2e)


def test_endpoint_holdout_runs_without_error_on_synthetic(grid_network, t0):
    """Sanity-only: endpoint_holdout should run on a small synthetic trip
    and return a list (possibly empty if reconstruction fails)."""
    from src.api.pipeline import reconstruct_trajectory
    from src.config import Config
    from src.model import (
        ExponentialFamilyTransition, RawObservation, StudentTEmission,
    )
    from src.validation import endpoint_holdout

    raw = [
        RawObservation(timestamp=t0 + _td(seconds=i * 60),
                       lat=19.430, lon=-99.135 + i * 0.001, hdop=2.0)
        for i in range(6)
    ]
    config = Config(
        emission=StudentTEmission(scale=15.0, network=grid_network),
        transition=ExponentialFamilyTransition(np.zeros(FEATURE_DIM)),
    )
    out = endpoint_holdout(raw, grid_network, config)
    assert isinstance(out, list)
    for label, dist in out:
        assert label in ("head", "tail")
        assert dist >= 0
