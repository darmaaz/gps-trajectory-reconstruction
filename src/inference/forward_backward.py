"""Log-domain forward-backward over the discrete trajectory CRF.

SPEC.md §inference.forward_backward. Returns:

    state_marginals : list[dict[State, float]]   length T
    path_marginals  : list[dict[Path, float]]    length T-1
    log_partition   : float                       log Z

All accumulation is in log-domain via `scipy.special.logsumexp`, which
handles `-inf` entries cleanly. The CRF factorisation is:

    φ(τ | g) = ∏ₖ ω(g_k|x_k) · δ(x_k, p_k) · η(p_k) · δ̄(p_k, x_{k+1})

Because the δ/δ̄ indicators are absorbed into `transition_triples`
(a path contributes a finite log_factor only for the (i, j) pair it actually
joins), the forward recursion collapses to:

    α[k+1][j] = ω(g_{k+1}|x_j^{k+1})
              + logsumexp_i (α[k][i] + log T[k][i, j])

with `log T[k][i, j] = logsumexp_p (μᵀϕ(p))` over paths p that connect
state i to state j. Backward is the symmetric reverse.

Path marginals use the FB identity:

    log r(p) = α[k][i] + log_factor(i, p, j) + ω(g_{k+1}|x_j^{k+1})
             + β[k+1][j] - log Z

Multiple (i, j) edges of the same `Path` (rare but possible if dedup ties
preserved them) are collapsed by logsumexp.
"""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

from ..model import (
    CollapsedObservation, EmissionFactor, Path, State, TransitionFactor,
)
from ._common import (
    emission_log_potentials,
    transition_logsumexp_matrix,
    transition_triples,
)


def forward_backward(
    state_candidates: list[list[State]],
    path_candidates: list[list[Path]],
    observations: list[CollapsedObservation],
    emission: EmissionFactor,
    transition: TransitionFactor,
    time_budgets: list[float],
) -> tuple[list[dict[State, float]], list[dict[Path, float]], float]:
    T = len(state_candidates)
    if T == 0:
        return [], [], float("-inf")
    if T == 1:
        log_emit = emission_log_potentials(state_candidates, observations, emission)
        log_z = float(logsumexp(log_emit[0])) if len(log_emit[0]) else float("-inf")
        if log_z == float("-inf"):
            return [{}], [], float("-inf")
        log_q = log_emit[0] - log_z
        return (
            [{state_candidates[0][i]: float(np.exp(log_q[i]))
              for i in range(len(state_candidates[0]))}],
            [],
            log_z,
        )

    if len(observations) != T:
        raise ValueError(
            f"observations length {len(observations)} != T={T}",
        )
    if len(path_candidates) != T - 1:
        raise ValueError(
            f"path_candidates length {len(path_candidates)} != T-1={T - 1}",
        )
    if len(time_budgets) != T - 1:
        raise ValueError(
            f"time_budgets length {len(time_budgets)} != T-1={T - 1}",
        )

    log_emit = emission_log_potentials(state_candidates, observations, emission)
    triples = transition_triples(
        state_candidates, path_candidates, transition, time_budgets,
    )
    log_trans = [
        transition_logsumexp_matrix(
            triples[k], len(state_candidates[k]), len(state_candidates[k + 1]),
        )
        for k in range(T - 1)
    ]

    # Forward pass.
    alpha: list[np.ndarray] = [np.empty(0)] * T
    alpha[0] = log_emit[0].copy()
    for k in range(T - 1):
        if alpha[k].size == 0 or log_trans[k].size == 0:
            alpha[k + 1] = np.full(len(state_candidates[k + 1]), -np.inf)
            continue
        # combined[i, j] = α[k][i] + log_trans[k][i, j]
        combined = alpha[k][:, None] + log_trans[k]
        alpha[k + 1] = log_emit[k + 1] + logsumexp(combined, axis=0)

    log_z = float(logsumexp(alpha[T - 1])) if alpha[T - 1].size else float("-inf")
    if not np.isfinite(log_z):
        # Pipeline-level discontinuity — the orchestrator should have split
        # this trip earlier. Surface as empty marginals + -inf partition.
        return (
            [{} for _ in range(T)],
            [{} for _ in range(T - 1)],
            log_z,
        )

    # Backward pass.
    beta: list[np.ndarray] = [np.empty(0)] * T
    beta[T - 1] = np.zeros(len(state_candidates[T - 1]))
    for k in range(T - 2, -1, -1):
        if log_trans[k].size == 0 or beta[k + 1].size == 0:
            beta[k] = np.full(len(state_candidates[k]), -np.inf)
            continue
        # combined[i, j] = log_trans[k][i, j] + log_emit[k+1][j] + beta[k+1][j]
        combined = log_trans[k] + (log_emit[k + 1] + beta[k + 1])[None, :]
        beta[k] = logsumexp(combined, axis=1)

    # State marginals.
    state_marginals: list[dict[State, float]] = []
    for k in range(T):
        log_q = alpha[k] + beta[k] - log_z
        q = np.exp(log_q)
        state_marginals.append({
            state_candidates[k][i]: float(q[i])
            for i in range(len(state_candidates[k]))
        })

    # Path marginals — accumulate per-path log-marginals across (i, j)
    # entries that share the same Path object.
    path_marginals: list[dict[Path, float]] = []
    for k in range(T - 1):
        per_path: dict[Path, float] = {}
        emit_kp1 = log_emit[k + 1]
        beta_kp1 = beta[k + 1]
        for (i, j), pairs in triples[k].items():
            base = alpha[k][i] + emit_kp1[j] + beta_kp1[j] - log_z
            for path, lp in pairs:
                log_marg = base + lp
                prev = per_path.get(path, -np.inf)
                per_path[path] = float(np.logaddexp(prev, log_marg))
        path_marginals.append({p: float(np.exp(lm)) for p, lm in per_path.items()})

    return state_marginals, path_marginals, log_z
