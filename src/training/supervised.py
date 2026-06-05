"""Supervised MLE for `(mu, log_emission_scale)` — SPEC.md §training.supervised.

Maximises the conditional log-likelihood of the labeled trajectories under
the CRF:

    L(mu, scale) = Σ_trips [log φ(τ_label | O; θ) - log Z(O; θ)]
                 = Σ_trips [Σ_k log ω(o_k | x_label_k; scale)
                          + Σ_k log η(p_label_k; mu)
                          - log Z(O; θ)]

Convex in `mu` for fixed scale; well-behaved jointly in `(mu, log_scale)`
in practice. Solved by L-BFGS-B with a hand-written gradient:

    ∂L/∂μ        = Σ_k ϕ(p_label_k) - Σ_k E_r[ϕ(p_k)]
    ∂L/∂log_scale = Σ_k g(d_label_k) - Σ_k E_q[g(d)]

where `g(d) = ∂log ω(d) / ∂log_scale` for Student-t equals
`-1 + (df+1) d² / (df·scale² + d²)`.

Forward-backward provides both the partition function `log Z` and the
state/path marginals needed for the expected terms. Each `negloglik_grad`
evaluation runs FB once per trip — for ~100 trips that's ≈ 200 ms;
L-BFGS-B typically converges in 30–60 iterations.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import scipy.optimize

from ..inference import forward_backward
from ..model import (
    ExponentialFamilyTransition, FEATURE_DIM, StudentTEmission,
)
from ..network import RoadNetwork
from .types import LabeledTrip


def _student_t_log_scale_gradient(d: float, scale: float, df: float) -> float:
    """∂ log f(d; scale, df) / ∂ log_scale for the Student-t emission."""
    if not math.isfinite(d):
        return 0.0
    d2 = d * d
    return -1.0 + (df + 1.0) * d2 / (df * scale * scale + d2)


def fit_supervised(
    labeled_trips: list[LabeledTrip],
    network: RoadNetwork,
    feature_dim: int = FEATURE_DIM,
    df: float = 4.0,
    initial_mu: np.ndarray | None = None,
    initial_log_scale: float | None = None,
    max_iter: int = 200,
    l2_reg: float = 1e-4,
    fix_scale: float | None = None,
) -> tuple[np.ndarray, float]:
    """Fit `(mu, scale)` by MLE on labeled trips with L2 regularisation on `mu`.

    The regulariser term `(l2_reg / 2) · ‖mu‖²` (default `1e-4`) breaks ties
    in flat directions of the CRF likelihood. Without it, tiny training
    sets with high feature dimensionality (length, travel-time, and dwell
    features can each take large values) drive L-BFGS-B into degenerate
    regions where ABNORMAL line-search failures occur. The penalty is
    small enough that it doesn't meaningfully bias the fit on well-
    conditioned production data; set to 0 to disable.

    Returns `(mu, scale)` where `scale > 0` is in metres. Emits a
    `RuntimeWarning` on non-convergence but still returns the best-so-far
    parameters.
    """
    if not labeled_trips:
        raise ValueError("at least one labeled trip required")

    mu0 = np.zeros(feature_dim) if initial_mu is None else np.asarray(initial_mu, dtype=float)
    log_scale0 = math.log(10.0) if initial_log_scale is None else float(initial_log_scale)

    def negloglik_and_grad(theta: np.ndarray) -> tuple[float, np.ndarray]:
        mu = theta[:feature_dim]
        log_scale = float(theta[feature_dim])
        scale = math.exp(log_scale)
        emission = StudentTEmission(scale=scale, network=network, df=df)
        transition = ExponentialFamilyTransition(mu)

        total_nll = 0.0
        grad_mu = np.zeros(feature_dim)
        grad_log_scale = 0.0

        for trip in labeled_trips:
            state_marg, path_marg, log_z = forward_backward(
                trip.state_candidates, trip.path_candidates,
                trip.observations, emission, transition, trip.time_budgets,
            )

            # log φ(τ_label) = sum of emission log-potentials at labeled
            # states + sum of transition log-potentials at labeled paths
            # (the latter is just μᵀ ϕ(p_label) since labels respect δ/δ̄).
            log_phi_label = 0.0
            empirical_phi = np.zeros(feature_dim)
            empirical_emit_g = 0.0
            for k, obs in enumerate(trip.observations):
                label_state = trip.state_candidates[k][trip.label_state_idx[k]]
                log_phi_label += emission.log_potential(obs, label_state)
                d = network.perpendicular_distance(
                    obs.lat, obs.lon, label_state.link_id, label_state.offset,
                )
                empirical_emit_g += _student_t_log_scale_gradient(d, scale, df)

            for k in range(len(trip.path_candidates)):
                if k >= len(trip.label_path_idx):
                    break
                label_path = trip.path_candidates[k][trip.label_path_idx[k]]
                empirical_phi += label_path.feature_vector
                # μᵀ ϕ(label_path) — δ/δ̄ are satisfied by construction.
                log_phi_label += float(mu @ label_path.feature_vector)

            # Expected features under path marginals (per trip).
            expected_phi = np.zeros(feature_dim)
            for pm in path_marg:
                for path, prob in pm.items():
                    expected_phi += prob * path.feature_vector

            # Expected emission gradient under state marginals.
            expected_emit_g = 0.0
            for k, sm in enumerate(state_marg):
                obs = trip.observations[k]
                for state, prob in sm.items():
                    d = network.perpendicular_distance(
                        obs.lat, obs.lon, state.link_id, state.offset,
                    )
                    expected_emit_g += prob * _student_t_log_scale_gradient(d, scale, df)

            total_nll += log_z - log_phi_label
            grad_mu += expected_phi - empirical_phi
            grad_log_scale += expected_emit_g - empirical_emit_g

        # L2 regularisation on mu — breaks flat-direction ties on small
        # training sets without biasing well-conditioned fits.
        if l2_reg > 0.0:
            total_nll += 0.5 * l2_reg * float(mu @ mu)
            grad_mu += l2_reg * mu

        grad = np.concatenate([grad_mu, np.array([grad_log_scale])])
        return total_nll, grad

    if fix_scale is not None:
        # Pin the emission scale; only μ is fit. Use when the joint
        # optimisation pegs at the scale lower bound (a degeneracy of
        # deterministic-label supervised CRF training) and the resulting
        # μ doesn't generalise.
        log_scale0 = math.log(float(fix_scale))
        scale_lo = log_scale0
        scale_hi = log_scale0
    else:
        # Standard bounds: 2.0 m noise floor, 1000 m overflow guard.
        scale_lo = math.log(2.0)
        scale_hi = math.log(1000.0)

    theta0 = np.concatenate([mu0, np.array([log_scale0])])
    bounds = [(None, None)] * feature_dim + [(scale_lo, scale_hi)]
    result = scipy.optimize.minimize(
        negloglik_and_grad, theta0,
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
        options={"maxiter": max_iter},
    )
    if not result.success:
        # L-BFGS-B's ABNORMAL termination typically means the line search
        # couldn't make further progress — common when the gradient is
        # small near optimum, or on very small synthetic problems with
        # limited signal in some feature dimensions. `result.x` still
        # holds the best-so-far parameters; warn rather than discard them.
        warnings.warn(
            f"L-BFGS-B did not fully converge: {result.message}. "
            f"Returning best-so-far parameters (nit={result.nit}, "
            f"nfev={result.nfev}, final_nll={result.fun:.6g}).",
            RuntimeWarning,
            stacklevel=2,
        )

    mu_star = result.x[:feature_dim]
    scale_star = math.exp(float(result.x[feature_dim]))
    return mu_star, scale_star
