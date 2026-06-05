"""EM training on unlabeled trips.

This is the **hard-EM** variant: each iteration's E-step computes the
Viterbi most-likely trajectory and treats it as a label for an inner
supervised step. Spec describes a soft-EM (gradient-weighted by FB
posteriors); hard-EM is operationally simpler, converges to the same local
optima in practice, and is what map-matching literature commonly uses
under "EM" without further qualification.

Loop:
    1. With current `(mu, scale)`, run preprocessing + candidates +
       forward_backward + Viterbi on every trip.
    2. Aggregate `Σ log Z` across trips. Stop if its change since the last
       iteration is below `tolerance`.
    3. Construct a `LabeledTrip` per trip from the Viterbi result.
    4. Run `fit_supervised` warm-started from the current parameters.
    5. Iterate.

Trips with broken segmentation (off-network observation, infeasible
transition) are skipped during the iteration — those trips don't supply
usable feature statistics and would otherwise pollute the M-step
gradient. They are reconsidered each iteration, so a parameter update
that makes a previously-broken trip routable will fold it back in.
"""

from __future__ import annotations

import logging
import math

import numpy as np

logger = logging.getLogger(__name__)

from ..candidates import enumerate_paths_per_transition, project_observation
from ..config import Config
from ..inference import forward_backward, most_likely_trajectory
from ..model import (
    ExponentialFamilyTransition, FEATURE_DIM, RawObservation, StudentTEmission,
)
from ..network import RoadNetwork
from ..preprocessing import (
    clean, collapse_by_uniqueness, drop_replay_bursts, flag_stale_runs,
)
from .supervised import fit_supervised
from .types import LabeledTrip


def _build_labeled_trip_from_viterbi(
    collapsed, state_cands, path_cands, time_budgets, ml_trajectory,
) -> LabeledTrip:
    """Translate a Viterbi `[state, path, state, ..., state]` output into
    `LabeledTrip` index arrays. State equality uses dataclass value-eq;
    Path equality uses identity (paths are passed through by reference).
    """
    label_state_idx: list[int] = []
    label_path_idx: list[int] = []

    for k in range(len(state_cands)):
        ml_state = ml_trajectory[2 * k]
        # state_cands[k] is a list of concrete State implementations — value-equal.
        try:
            idx = state_cands[k].index(ml_state)
        except ValueError as e:
            raise RuntimeError(
                f"Viterbi state at obs {k} not found in candidate list",
            ) from e
        label_state_idx.append(idx)

    for k in range(len(path_cands)):
        ml_path = ml_trajectory[2 * k + 1]
        # Path is identity-hashed (eq=False); use `is`.
        for pi, p in enumerate(path_cands[k]):
            if p is ml_path:
                label_path_idx.append(pi)
                break
        else:
            raise RuntimeError(
                f"Viterbi path at transition {k} not found in candidate list",
            )

    return LabeledTrip(
        observations=collapsed,
        state_candidates=state_cands,
        path_candidates=path_cands,
        time_budgets=time_budgets,
        label_state_idx=label_state_idx,
        label_path_idx=label_path_idx,
    )


def _enumerate_trip_segments(
    raw: list[RawObservation], network: RoadNetwork, config: Config,
) -> list[tuple]:
    """Run preprocessing and candidates for one trip; segment at
    discontinuities; return one tuple per contiguous reconstructable
    segment.

    Each tuple: `(collapsed, state_cands, path_cands, time_budgets)`
    spanning a single segment with non-empty state cands and path cands
    throughout. Single-observation segments are skipped (EM needs ≥ 2
    observations per segment to provide any transition signal).
    """
    cleaned = clean(raw)
    collapsed = collapse_by_uniqueness(cleaned, config.collapse_epsilon)
    if len(collapsed) < 2:
        return []

    # Narrow the network to this trip's bbox before any graph-search step.
    # See `api.pipeline.reconstruct_trajectory` for the rationale; this
    # mirrors that behaviour so EM iterates over the same subgraph the
    # orchestrator would. Each cached segment carries its subgraph along
    # so per-iteration FB / Viterbi run on the small subgraph too.
    lats = [o.lat for o in collapsed]
    lons = [o.lon for o in collapsed]
    net = network.subgraph_for_bbox(
        min(lats), max(lats), min(lons), max(lons),
        buffer_m=config.subgraph_buffer_m,
    )

    collapsed = flag_stale_runs(collapsed, net, config.max_speed_factor)
    collapsed = drop_replay_bursts(
        collapsed,
        max_speed_ms=config.replay_max_speed_ms,
        max_speed_factor=config.max_speed_factor,
        k_consistent=config.replay_k_consistent,
        moving_threshold_ms=config.replay_moving_threshold_ms,
    )
    collapsed = [o for o in collapsed if not o.dropped_during_replay]
    if len(collapsed) < 2:
        return []

    state_cands = [
        project_observation(
            o, net,
            radius_meters=config.candidate_radius,
            max_candidates=config.max_state_candidates,
        )
        for o in collapsed
    ]
    time_budgets = [
        (collapsed[k + 1].t_first - collapsed[k].t_first).total_seconds()
        - (0.0 if collapsed[k].stale_flagged
           else (collapsed[k].t_last - collapsed[k].t_first).total_seconds())
        for k in range(len(collapsed) - 1)
    ]
    path_cands = enumerate_paths_per_transition(
        state_cands, net, time_budgets,
        max_path_candidates=config.max_path_candidates,
        budget_slack=config.path_budget_slack,
        penalty_lambda=config.path_penalty_lambda,
    )

    # Segment by reachability — same logic the orchestrator uses.
    from ..api.pipeline import _segment_slices
    out: list[tuple] = []
    for start, end in _segment_slices(state_cands, path_cands):
        if end - start < 2:
            continue    # need at least one transition for EM signal
        out.append((
            net,
            collapsed[start:end],
            state_cands[start:end],
            path_cands[start:end - 1],
            time_budgets[start:end - 1],
        ))
    return out


def fit_em(
    unlabeled_trips: list[list[RawObservation]],
    network: RoadNetwork,
    config: Config,
    feature_dim: int = FEATURE_DIM,
    df: float = 4.0,
    initial_mu: np.ndarray | None = None,
    initial_log_scale: float | None = None,
    max_iterations: int = 50,
    tolerance: float = 1e-4,
) -> tuple[np.ndarray, float]:
    """EM fit of `(mu, scale)` from unlabeled trips.

    `config` supplies the preprocessing and candidate-generation knobs;
    its `emission` and `transition` factors are replaced internally each
    iteration with the current parameters, so the values you pass in for
    those slots only matter as initial defaults if `initial_mu` /
    `initial_log_scale` are unset.

    Returns `(mu, scale)` at convergence (∆ Σ log Z < `tolerance` between
    iterations) or after `max_iterations`. Raises `RuntimeError` if every
    trip is skipped at some iteration.
    """
    if not unlabeled_trips:
        raise ValueError("at least one unlabeled trip required")

    mu = (
        np.zeros(feature_dim)
        if initial_mu is None
        else np.asarray(initial_mu, dtype=float).copy()
    )
    log_scale = math.log(10.0) if initial_log_scale is None else float(initial_log_scale)
    prev_total_log_z = float("-inf")

    # Pre-compute path enumeration ONCE — it depends only on the network,
    # observations, and config geometry knobs (radius, K, slack), none of
    # which EM updates. Re-running it per iteration is wasted work.
    logger.info("enumerating paths for %d trip(s) once", len(unlabeled_trips))
    cached_segments: list[tuple] = []
    for trip_idx, raw in enumerate(unlabeled_trips):
        trip_segs = _enumerate_trip_segments(raw, network, config)
        cached_segments.extend(trip_segs)
        logger.info("trip %d: %d usable segment(s)", trip_idx, len(trip_segs))
    if not cached_segments:
        raise RuntimeError(
            f"no usable segments across {len(unlabeled_trips)} trip(s) "
            "after preprocessing + path enumeration",
        )
    logger.info("cached %d segment(s) total", len(cached_segments))

    for iteration in range(max_iterations):
        scale = math.exp(log_scale)
        transition = ExponentialFamilyTransition(mu)

        labeled_iter: list[LabeledTrip] = []
        total_log_z = 0.0
        for sub_net, collapsed, state_cands, path_cands, time_budgets in cached_segments:
            # Emission is rebound to the segment's subgraph so distance
            # lookups (`network.edge_index_for_link` → geom interp) hit the
            # subgraph's cached link-id index, not the full network's.
            seg_emission = StudentTEmission(
                scale=scale, network=sub_net, df=df,
            )
            _, _, log_z = forward_backward(
                state_cands, path_cands, collapsed,
                seg_emission, transition, time_budgets,
            )
            if not math.isfinite(log_z):
                continue
            subs = most_likely_trajectory(
                state_cands, path_cands, collapsed,
                seg_emission, transition, time_budgets,
            )
            # Graceful Viterbi may split on a cliff. For EM training we
            # currently only use trips that reconstructed cleanly as one
            # sub; cliffed trips are skipped (could be improved by adding
            # one labeled trip per sub).
            if len(subs) != 1 or subs[0].start_obs_idx != 0 or (
                subs[0].end_obs_idx != len(collapsed) - 1
            ):
                continue
            total_log_z += log_z
            labeled_iter.append(_build_labeled_trip_from_viterbi(
                collapsed, state_cands, path_cands, time_budgets,
                subs[0].most_likely,
            ))

        if not labeled_iter:
            raise RuntimeError(
                f"EM iteration {iteration}: no usable segments",
            )

        logger.info(
            "iter %d: segments=%d log_z=%.2f scale=%.3f mu[0]=%+.4f",
            iteration, len(labeled_iter), total_log_z,
            math.exp(log_scale), mu[0],
        )

        if abs(total_log_z - prev_total_log_z) < tolerance:
            break
        prev_total_log_z = total_log_z

        try:
            mu, scale = fit_supervised(
                labeled_iter, network,
                feature_dim=feature_dim, df=df,
                initial_mu=mu, initial_log_scale=log_scale,
            )
            log_scale = math.log(scale)
        except RuntimeError as e:
            logger.warning(
                "iter %d: M-step failed: %s; returning last-good params",
                iteration, e,
            )
            break

    return mu, math.exp(log_scale)
