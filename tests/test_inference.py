"""Forward-backward and Viterbi against hand-computed CRF posteriors.

The headline test (`test_fb_matches_analytical_2step_crf`) builds a 1×2×1
CRF where the posterior is closed-form. Two parallel branches A/B between
two states, with controllable emission penalties on the middle states and
length-only feature on each path. The expected ratio of state marginals at
step 1 is `exp(2.0) = 7.389`; the expected `log Z = log(e^-2.5 + e^-4.5)`.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta

import numpy as np
import pytest

from src.inference import forward_backward, most_likely_trajectory
from src.model import (
    CollapsedObservation, ExponentialFamilyTransition, FEATURE_DIM, Path,
    StateV1,
)


def _mk_path(edges, length, ttime):
    feats = np.zeros(FEATURE_DIM)
    feats[0] = float(length)
    feats[12] = float(ttime)
    return Path(
        edges=tuple(edges),
        start_offset=0.0, end_offset=0.0,
        expected_travel_time=ttime, length_meters=length,
        feature_vector=feats,
    )


class _FixedEmission:
    """Returns a predetermined log-potential per state object identity."""
    def __init__(self, scores):
        self._scores = {id(s): float(v) for s, v in scores.items()}

    def log_potential(self, obs, state):
        return self._scores.get(id(state), 0.0)


@pytest.fixture
def two_step_crf(t0):
    sk0 = StateV1(link_id=100, offset=0.0, entry_time=t0)
    sk1_a = StateV1(link_id=200, offset=0.0, entry_time=t0)
    sk1_b = StateV1(link_id=201, offset=0.0, entry_time=t0)
    sk2 = StateV1(link_id=300, offset=0.0, entry_time=t0)

    p_via_a = _mk_path([100, 200], length=100, ttime=10)
    p_via_b = _mk_path([100, 201], length=200, ttime=20)
    p_a_then = _mk_path([200, 300], length=50, ttime=5)
    p_b_then = _mk_path([201, 300], length=50, ttime=5)

    emit = _FixedEmission({sk0: 0.0, sk1_a: -1.0, sk1_b: -2.0, sk2: 0.0})

    mu = np.zeros(FEATURE_DIM)
    mu[0] = -0.01    # -1 per 100m of length
    trans = ExponentialFamilyTransition(mu)

    state_cands = [[sk0], [sk1_a, sk1_b], [sk2]]
    path_cands = [[p_via_a, p_via_b], [p_a_then, p_b_then]]
    obs = [
        CollapsedObservation(0.0, 0.0, t0, t0, 1),
        CollapsedObservation(0.0, 0.0, t0 + timedelta(seconds=10),
                             t0 + timedelta(seconds=10), 1),
        CollapsedObservation(0.0, 0.0, t0 + timedelta(seconds=20),
                             t0 + timedelta(seconds=20), 1),
    ]
    return state_cands, path_cands, obs, emit, trans, [10.0, 10.0], (
        sk0, sk1_a, sk1_b, sk2, p_via_a, p_via_b, p_a_then, p_b_then,
    )


def test_fb_matches_analytical_2step_crf(two_step_crf):
    state_cands, path_cands, obs, emit, trans, budgets, items = two_step_crf
    sk0, sk1_a, sk1_b, sk2, *_ = items

    state_marg, path_marg, log_z = forward_backward(
        state_cands, path_cands, obs, emit, trans, budgets,
    )

    # log Z analytical: logsumexp(-2.5, -4.5)
    expected_log_z = math.log(math.exp(-2.5) + math.exp(-4.5))
    assert log_z == pytest.approx(expected_log_z, abs=1e-9)

    # All state marginals sum to 1
    for k, m in enumerate(state_marg):
        assert sum(m.values()) == pytest.approx(1.0, abs=1e-9)

    # Posterior at step 1: ratio = exp(2.0)
    ratio = state_marg[1][sk1_a] / state_marg[1][sk1_b]
    assert ratio == pytest.approx(math.exp(2.0), abs=0.01)

    # Path marginals sum (across both paths in transition 0) to ≈ 1
    assert sum(path_marg[0].values()) == pytest.approx(1.0, abs=1e-9)


def test_viterbi_picks_dominant_branch(two_step_crf):
    state_cands, path_cands, obs, emit, trans, budgets, items = two_step_crf
    sk0, sk1_a, sk1_b, sk2, p_via_a, _, p_a_then, _ = items

    subs = most_likely_trajectory(
        state_cands, path_cands, obs, emit, trans, budgets,
    )
    # Clean segment: graceful Viterbi yields exactly one sub covering [0, T-1].
    assert len(subs) == 1
    sub = subs[0]
    assert (sub.start_obs_idx, sub.end_obs_idx) == (0, 2)
    ml = sub.most_likely
    assert len(ml) == 5
    assert ml[0] is sk0
    assert ml[1] is p_via_a
    assert ml[2] is sk1_a
    assert ml[3] is p_a_then
    assert ml[4] is sk2


def test_fb_t1_emission_only_marginal(t0):
    s_a = StateV1(link_id=1, offset=0.0, entry_time=t0)
    s_b = StateV1(link_id=2, offset=0.0, entry_time=t0)
    obs = [CollapsedObservation(0.0, 0.0, t0, t0, 1)]
    emit = _FixedEmission({s_a: 0.0, s_b: math.log(0.5) - math.log(0.5) - 1.0})
    # log emit: s_a = 0,  s_b = -1 → P(s_a) = e/(e+1), P(s_b) = 1/(e+1)
    state_marg, path_marg, log_z = forward_backward(
        [[s_a, s_b]], [], obs, emit,
        ExponentialFamilyTransition(np.zeros(FEATURE_DIM)),
        [],
    )
    assert path_marg == []
    p_a = state_marg[0][s_a]
    p_b = state_marg[0][s_b]
    assert p_a + p_b == pytest.approx(1.0, abs=1e-9)
    expected_pa = math.exp(0.0) / (math.exp(0.0) + math.exp(-1.0))
    assert p_a == pytest.approx(expected_pa, abs=1e-9)


def test_viterbi_splits_on_dead_transition(t0):
    sk0 = StateV1(link_id=100, offset=0.0, entry_time=t0)
    sk1 = StateV1(link_id=101, offset=0.0, entry_time=t0)
    obs = [
        CollapsedObservation(0.0, 0.0, t0, t0, 1),
        CollapsedObservation(0.0, 0.0, t0 + timedelta(seconds=1),
                             t0 + timedelta(seconds=1), 1),
    ]
    # No paths between them → graceful Viterbi splits into two single-obs
    # sub-trajectories rather than raising.
    subs = most_likely_trajectory(
        [[sk0], [sk1]], [[]], obs, _FixedEmission({sk0: 0.0, sk1: 0.0}),
        ExponentialFamilyTransition(np.zeros(FEATURE_DIM)), [1.0],
    )
    assert len(subs) == 2
    assert (subs[0].start_obs_idx, subs[0].end_obs_idx) == (0, 0)
    assert subs[0].most_likely == [sk0]
    assert (subs[1].start_obs_idx, subs[1].end_obs_idx) == (1, 1)
    assert subs[1].most_likely == [sk1]
