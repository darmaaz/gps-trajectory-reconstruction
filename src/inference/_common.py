"""Shared building blocks for forward-backward and Viterbi.

Both passes need:
    1. Per-observation emission log-potentials, indexed by state.
    2. Per-transition `(i, j) → list[(path, log_factor)]`, where i indexes
       `state_candidates[k]` and j indexes `state_candidates[k+1]`.

This module computes both up front so the two algorithms can share the
work; the output is consumed by `forward_backward` and
`most_likely_trajectory` differently (logsumexp vs max).
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy.special import logsumexp

from ..model import (
    CollapsedObservation, EmissionFactor, Path, State, TransitionFactor,
)


def emission_log_potentials(
    state_candidates: list[list[State]],
    observations: list[CollapsedObservation],
    emission: EmissionFactor,
) -> list[np.ndarray]:
    """Return a list of `(|X_k|,)` arrays of log-emission per observation."""
    out: list[np.ndarray] = []
    for k, states_k in enumerate(state_candidates):
        if not states_k:
            out.append(np.empty(0, dtype=float))
            continue
        out.append(np.array(
            [float(emission.log_potential(observations[k], s)) for s in states_k],
            dtype=float,
        ))
    return out


def transition_triples(
    state_candidates: list[list[State]],
    path_candidates: list[list[Path]],
    transition: TransitionFactor,
    time_budgets: list[float],
) -> list[dict[tuple[int, int], list[tuple[Path, float]]]]:
    """For each transition `k`, return `{(i, j): [(path, log_factor), ...]}`.

    Only entries with finite `log_factor` are recorded — the δ/δ̄ indicators
    on (state, path) compatibility eliminate any (i, j, path) triple where
    the path doesn't start at `state_candidates[k][i]` or end at
    `state_candidates[k+1][j]`. A triple list per (i, j) is needed because
    multiple paths can connect the same (i, j) pair (different middle
    edges, same offsets); their log-factors are logsumexp'd by FB and
    maxed by Viterbi.
    """
    out: list[dict[tuple[int, int], list[tuple[Path, float]]]] = []
    if len(state_candidates) < 2:
        return out

    for k in range(len(state_candidates) - 1):
        bucket: dict[tuple[int, int], list[tuple[Path, float]]] = defaultdict(list)
        sk_list = state_candidates[k]
        skp1_list = state_candidates[k + 1]
        # Pre-compute per-state matching predicates lazily inside the path
        # loop — paths usually match a single (i, j), so most checks short-
        # circuit on the first non-match.
        for path in path_candidates[k]:
            for i, sk in enumerate(sk_list):
                if not path.starts_at(sk):
                    continue
                for j, skp1 in enumerate(skp1_list):
                    if not path.ends_at(skp1):
                        continue
                    lp = float(transition.log_potential(
                        sk, path, skp1, time_budgets[k],
                    ))
                    if lp == float("-inf"):
                        continue
                    bucket[(i, j)].append((path, lp))
        out.append(dict(bucket))
    return out


def transition_logsumexp_matrix(
    triples: dict[tuple[int, int], list[tuple[Path, float]]],
    n_k: int, n_kp1: int,
) -> np.ndarray:
    """Per-(i, j) logsumexp over path log-factors. Used by forward-backward."""
    M = np.full((n_k, n_kp1), -np.inf, dtype=float)
    for (i, j), pairs in triples.items():
        if not pairs:
            continue
        M[i, j] = float(logsumexp([lp for _, lp in pairs]))
    return M


def transition_max_matrix(
    triples: dict[tuple[int, int], list[tuple[Path, float]]],
    n_k: int, n_kp1: int,
) -> tuple[np.ndarray, dict[tuple[int, int], Path]]:
    """Per-(i, j) max over path log-factors. Returns the matrix plus the
    argmax path per cell. Used by Viterbi.
    """
    M = np.full((n_k, n_kp1), -np.inf, dtype=float)
    best: dict[tuple[int, int], Path] = {}
    for (i, j), pairs in triples.items():
        if not pairs:
            continue
        b_path, b_lp = max(pairs, key=lambda x: x[1])
        M[i, j] = float(b_lp)
        best[(i, j)] = b_path
    return M, best
