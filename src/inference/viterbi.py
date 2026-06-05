"""Viterbi most-likely trajectory over the discrete CRF.

SPEC.md §inference.viterbi. Same recursion as forward-backward with
`max`/`argmax` replacing `logsumexp`/`sum`. The trajectory is the
interleaved sequence of states and paths

    [state_0, path_0, state_1, path_1, ..., state_{T-1}]

of length 2T-1; for the degenerate T=1 case the trajectory is `[state_0]`.

Graceful behavior on forward cliffs
-----------------------------------
A forward cliff occurs at transition `k` when
`delta[k][:, None] + log_trans[k]` has no finite entry. This means the
forward-alive subset at obs `k` has no compatible outgoing path to any
state at obs `k+1`. Rather than raising, this module splits the input
segment into multiple `MostLikelySubTrajectory` objects:

  - close the current sub at obs `k` by backtracking from the alive subset
    of `delta[k]`,
  - start a fresh sub at obs `k+1` with `delta = log_emit[k+1]` (no prior
    conditioning — provably correct because the bridge to anything before
    `k+1` is impossible).

A trip-clean segment yields a single `MostLikelySubTrajectory` covering
`[0, T-1]`. Multiple subs indicate one or more forward-pass cliffs.
Indices on the returned subs are local to the input segment (segment-local,
0-based, not trip-global). The orchestrator translates them into trip-global
indices and constructs `Discontinuity` metadata from the surrounding
candidate data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..model import (
    CollapsedObservation, EmissionFactor, Path, State, TransitionFactor,
)
from ._common import (
    emission_log_potentials,
    transition_max_matrix,
    transition_triples,
)


@dataclass(frozen=True)
class MostLikelySubTrajectory:
    """One coherent sub-segment from a graceful Viterbi pass.

    `start_obs_idx` and `end_obs_idx` are inclusive segment-local indices
    into the input `state_candidates`. `most_likely` is the interleaved
    `[State, Path, ..., State]` sequence of length
    `2 * (end_obs_idx - start_obs_idx + 1) - 1`.

    `end_alive_states` is the set of forward-alive state indices at
    `end_obs_idx` — i.e., those whose `delta` is finite when this sub
    closes. Useful for constructing a `Discontinuity` describing the
    gap that ends this sub (cliff or end-of-input). For a sub that
    reaches the end of the input cleanly, this is just the candidates
    with finite emission contribution to delta.
    """

    start_obs_idx: int
    end_obs_idx: int
    most_likely: list[State | Path]
    end_alive_state_indices: tuple[int, ...] = ()


def most_likely_trajectory(
    state_candidates: list[list[State]],
    path_candidates: list[list[Path]],
    observations: list[CollapsedObservation],
    emission: EmissionFactor,
    transition: TransitionFactor,
    time_budgets: list[float],
) -> list[MostLikelySubTrajectory]:
    T = len(state_candidates)
    if T == 0:
        return []

    if len(observations) != T:
        raise ValueError(
            f"observations length {len(observations)} != T={T}",
        )
    if T >= 2:
        if len(path_candidates) != T - 1:
            raise ValueError(
                f"path_candidates length {len(path_candidates)} != T-1={T - 1}",
            )
        if len(time_budgets) != T - 1:
            raise ValueError(
                f"time_budgets length {len(time_budgets)} != T-1={T - 1}",
            )

    log_emit = emission_log_potentials(state_candidates, observations, emission)

    # Pre-compute log_trans matrices and best-path-per-cell maps.
    triples = transition_triples(
        state_candidates, path_candidates, transition, time_budgets,
    ) if T >= 2 else []
    log_trans: list[np.ndarray] = []
    best_path: list[dict[tuple[int, int], Path]] = []
    for k in range(T - 1):
        if not state_candidates[k] or not state_candidates[k + 1] or not path_candidates[k]:
            log_trans.append(np.empty((0, 0)))
            best_path.append({})
            continue
        M, best = transition_max_matrix(
            triples[k], len(state_candidates[k]), len(state_candidates[k + 1]),
        )
        log_trans.append(M)
        best_path.append(best)

    return _viterbi_pass_with_cliffs(
        state_candidates, log_emit, log_trans, best_path,
    )


def _viterbi_pass_with_cliffs(
    state_candidates: list[list[State]],
    log_emit: list[np.ndarray],
    log_trans: list[np.ndarray],
    best_path: list[dict[tuple[int, int], Path]],
) -> list[MostLikelySubTrajectory]:
    """Walk forward; close-and-restart at every cliff. Backtrack each sub."""
    T = len(state_candidates)
    out: list[MostLikelySubTrajectory] = []

    sub_start = 0
    while sub_start < T:
        # Skip leading observations with empty state_cands (off-network).
        if not state_candidates[sub_start] or log_emit[sub_start].size == 0:
            sub_start += 1
            continue

        # Run forward pass from sub_start until cliff or end-of-input.
        delta: list[np.ndarray] = [log_emit[sub_start].copy()]
        bp_state: list[np.ndarray] = [
            np.full(len(state_candidates[sub_start]), -1, dtype=int),
        ]
        sub_end = sub_start
        for k in range(sub_start, T - 1):
            kk = k - sub_start    # local index into delta/bp_state
            if (
                log_trans[k].size == 0
                or not state_candidates[k + 1]
                or log_emit[k + 1].size == 0
            ):
                # Structural split — close sub at k, restart at k+1.
                break
            combined = delta[kk][:, None] + log_trans[k]
            if not np.isfinite(combined).any():
                # Forward cliff — close sub at k, restart at k+1.
                break
            bp_state.append(combined.argmax(axis=0))
            delta.append(log_emit[k + 1] + combined.max(axis=0))
            sub_end = k + 1

        # Backtrack from delta[-1] (the last finite delta in this sub).
        ml = _backtrack(
            sub_start, sub_end, state_candidates, delta, bp_state, best_path,
        )
        if ml:
            alive_indices = tuple(
                int(i) for i in np.where(np.isfinite(delta[-1]))[0]
            )
            out.append(MostLikelySubTrajectory(
                start_obs_idx=sub_start,
                end_obs_idx=sub_end,
                most_likely=ml,
                end_alive_state_indices=alive_indices,
            ))
        sub_start = sub_end + 1

    return out


def _backtrack(
    sub_start: int,
    sub_end: int,
    state_candidates: list[list[State]],
    delta: list[np.ndarray],
    bp_state: list[np.ndarray],
    best_path: list[dict[tuple[int, int], Path]],
) -> list[State | Path]:
    """Reconstruct the most-likely interleaved sequence for one sub."""
    if delta[-1].size == 0 or not np.isfinite(delta[-1]).any():
        return []

    j_star = int(np.nanargmax(np.where(np.isfinite(delta[-1]), delta[-1], -np.inf)))
    state_path_indices: list[int] = [j_star]
    n_steps = sub_end - sub_start
    for k_local in range(n_steps, 0, -1):
        i_star = int(bp_state[k_local][j_star])
        state_path_indices.append(i_star)
        j_star = i_star
    state_path_indices.reverse()

    out: list[State | Path] = []
    for k_local in range(n_steps + 1):
        global_k = sub_start + k_local
        out.append(state_candidates[global_k][state_path_indices[k_local]])
        if k_local < n_steps:
            i = state_path_indices[k_local]
            j = state_path_indices[k_local + 1]
            chosen = best_path[global_k].get((i, j))
            if chosen is None:
                # Should not happen: combined[i, j] was the argmax of a
                # finite-extension step, so the path map must have it.
                raise AssertionError(
                    f"viterbi backtrack picked (i={i}, j={j}) at "
                    f"transition {global_k} but no path is registered for it",
                )
            out.append(chosen)
    return out
