"""Pipeline orchestrator.

`reconstruct_trajectory` glues the modules together:

    raw observations
        → hygiene.clean
        → drop_kinematic_spikes
        → drop_replay_bursts
        → collapse_by_uniqueness
        → flag_stale_runs               (uses A* on the network)
        → project_observation           (per CollapsedObservation)
        → enumerate_paths_per_transition (with features)
        → forward_backward + most_likely_trajectory (per contiguous segment)
        → list[TrajectoryPosterior]

Discontinuities (off-network observations, infeasible transitions) split
the trip into multiple contiguous segments. We return
`list[TrajectoryPosterior]` — one element per segment, single-element for
clean trips.

`TrajectoryPosterior` implements the `MarginalQuery` protocol directly, so
callers can use it as a query object without a separate wrapper. `at_time`
currently resolves only to canonical observation timestamps; the dwell-
aware interpolation across path posteriors is the next planned addition
(`QUESTIONS_DEFERRED.md` — intermediate-time queries).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Protocol

import networkx as nx

from ..candidates import enumerate_paths_per_transition, project_observation
from ..config import Config
from ..geo import haversine_m
from ..inference import (
    MostLikelySubTrajectory, forward_backward, most_likely_trajectory,
)
from ..model import (
    CollapsedObservation, Path, RawObservation, State, StateV1,
)
from ..network import RoadNetwork
from ..network.routing import _get_nx_graph
from ..preprocessing import (
    clean, collapse_by_uniqueness, drop_kinematic_spikes,
    drop_replay_bursts, flag_stale_runs,
)
from .interpolation import DwellRule, position_in_transition


class MarginalQuery(Protocol):
    """Read state marginals from a reconstructed segment.

    `at_observation(k)` returns the marginal at the k-th preprocessed
    observation within this segment. `at_time(t, rule)` returns a marginal
    at arbitrary `t` between (and at) the segment's canonical timestamps,
    aggregated over the path posterior under the chosen dwell allocation
    rule — see `TrajectoryPosterior.at_time` for the contract.
    """

    def at_observation(self, k: int) -> dict[State, float]: ...
    def at_time(
        self, t: datetime, rule: DwellRule = "front",
    ) -> dict[State, float]: ...


@dataclass(frozen=True)
class Discontinuity:
    """Structural facts about a segment boundary.

    Populated for any segment that doesn't start at trip-time-zero. Three
    causes share this dataclass: structural splits from `_segment_slices`
    (off-network observation or empty `path_cands[k]`), forward-pass cliffs
    inside Viterbi (where `delta` becomes all -inf), and post-backward
    sub-splits (where forward-defined segments contain interior all-`-inf`
    marginals).

    Fields are structural only. Downstream consumers that want a "reason"
    label should infer it from the structural data — e.g.,
    `len(last_alive_states) == 0 and not next_obs_candidates` indicates an
    off-network observation; `n_paths_in_transition == 0 and last_alive_states`
    indicates empty path enumeration; `n_paths_in_transition > 0 and
    last_alive_states` indicates a forward cliff with paths that the
    surviving alive set couldn't use.

    All trip-global indices.
    """

    after_obs_idx: int
    last_alive_states: tuple[State, ...]
    next_obs_candidates: tuple[State, ...]
    time_budget: float
    n_paths_in_transition: int
    speed_at_boundary: float
    min_unreachable_distance_s: float

    @property
    def n_alive_at_boundary(self) -> int:
        return len(self.last_alive_states)


@dataclass
class TrajectoryPosterior:
    """One contiguous segment of reconstructed trajectory.

    A trip with no discontinuities yields a single `TrajectoryPosterior`;
    discontinuities split it into multiple segments. `observation_indices`
    is a half-open slice `[start, end)` into the preprocessed
    `CollapsedObservation` sequence — useful for joining segments back to
    upstream metadata or for diagnosing where the splits happened.

    `preceded_by_discontinuity` is `None` for the very first segment of a
    trip and populated for every subsequent segment, regardless of whether
    the boundary is a structural split (off-network observation, empty
    `path_cands`) or a forward-pass cliff. Downstream consumers can iterate
    `segments` in trip-order; the discontinuity attached to each segment
    explains why this segment starts where it does.

    `stale_observation_indices` lists segment-local indices whose collapsed
    run was flagged stale by `flag_stale_runs`. Inference always uses
    `t_first` as the canonical timestamp regardless, so the flag is
    diagnostic only — a high `stale_fraction` signals the posterior is
    leaning heavily on the stale-data assumption and downstream consumers
    should treat the credible regions with proportional skepticism.

    Implements `MarginalQuery` directly via `at_observation` and `at_time`.
    """

    state_marginals: list[dict[State, float]]   # length T_seg
    path_marginals: list[dict[Path, float]]     # length T_seg - 1
    most_likely: list[State | Path]              # length 2*T_seg - 1
    log_partition: float
    canonical_timestamps: tuple[datetime, ...] = field(default_factory=tuple)
    canonical_t_last: tuple[datetime, ...] = field(default_factory=tuple)
    confirmed_dwell: tuple[float, ...] = field(default_factory=tuple)
    observation_indices: tuple[int, int] = (0, 0)
    stale_observation_indices: tuple[int, ...] = ()
    replay_dropped_count: int = 0
    preceded_by_discontinuity: Discontinuity | None = None
    network: RoadNetwork | None = None
    # ^ Subgraph used to reconstruct this segment. Required by `at_time(t)`
    # for off-grid queries (walking path geometry to (link_id, offset)).
    # Default `None` for tests that construct posteriors directly without
    # off-grid resolution; `reconstruct_trajectory` attaches the segment's
    # subgraph automatically.
    #
    # `replay_dropped_count`: number of original collapsed observations
    # whose `t_first` falls inside this segment's canonical-time span and
    # that were marked `dropped_during_replay` by `drop_replay_bursts`. The
    # segment's bookend bridge spans across them; a non-zero count signals
    # the segment includes a buffered-replay region whose intermediate
    # waypoints were discarded for data-quality reasons.

    def __post_init__(self) -> None:
        # Invariant: `canonical_t_last` is either empty (degenerate empty
        # segment) or the same length as `canonical_timestamps`. When the
        # caller didn't populate it and `canonical_timestamps` is non-empty,
        # default to `canonical_timestamps` — equivalent to zero confirmed
        # dwell at every observation. Consumers (`at_time`, `position_at_time`)
        # rely on this so they can index `canonical_t_last[k]` unconditionally.
        if not self.canonical_t_last and self.canonical_timestamps:
            self.canonical_t_last = self.canonical_timestamps
        elif (
            self.canonical_t_last
            and len(self.canonical_t_last) != len(self.canonical_timestamps)
        ):
            raise ValueError(
                f"canonical_t_last length {len(self.canonical_t_last)} "
                f"does not match canonical_timestamps length "
                f"{len(self.canonical_timestamps)}",
            )

    @property
    def stale_fraction(self) -> float:
        """Fraction of segment-local observations whose run was stale-flagged."""
        if not self.state_marginals:
            return 0.0
        return len(self.stale_observation_indices) / len(self.state_marginals)

    def at_observation(self, k: int) -> dict[State, float]:
        """State marginal for the k-th observation within this segment.

        `k` is local to the segment, not the original trip. Use
        `observation_indices` to map between segment-local and trip-global
        indices.
        """
        return self.state_marginals[k]

    def at_time(
        self, t: datetime, rule: DwellRule = "front",
    ) -> dict[State, float]:
        """Resolve a timestamp to a state marginal.

        Three regions are distinguished:

        - On-grid (`t` equals one of `canonical_timestamps`): returns the
          corresponding entry of `state_marginals` exactly — the contract
          matches `at_observation(k)`. The `rule` parameter is irrelevant.
        - Confirmed-dwell window (`canonical_timestamps[k] < t ≤
          canonical_t_last[k]`): the vehicle was *observed* to be at
          observation k's location throughout this interval. Returns
          `state_marginals[k]` unchanged — no path-allocation rule needed.
        - Transit window (`canonical_t_last[k] < t < canonical_timestamps[k+1]`):
          returns a marginal aggregated over the path posterior under
          `rule`, with `τ = (t − canonical_t_last[k])` — the time elapsed
          since the *end* of confirmed dwell, which is what the path's
          `time_budget` covers. Each candidate path's position is resolved
          by `position_in_transition`; positions resolving to the same
          `(link_id, offset)` get their weights summed.

        `rule` defaults to `"front"`, the project's convention. Pass
        `"back"` or `"spread"` to query alternative dwell allocations.

        Synthetic `State` objects carry the queried `t` as their
        `entry_time`. They are constructed fresh per query and are *not*
        elements of any `state_candidates[k]`; consumers must not rely on
        identity-equality against the candidate set.

        Raises `ValueError` if `t` falls outside the segment's time span
        (before `canonical_timestamps[0]` or after `canonical_timestamps[-1]`),
        and `RuntimeError` if an off-grid query is attempted on a Posterior
        constructed without a `network` reference.
        """
        if not self.canonical_timestamps:
            raise ValueError("at_time called on empty segment")

        for i, ts in enumerate(self.canonical_timestamps):
            if ts == t:
                return self.state_marginals[i]

        if t < self.canonical_timestamps[0] or t > self.canonical_timestamps[-1]:
            raise ValueError(
                f"t={t} is outside this segment's canonical-time span "
                f"[{self.canonical_timestamps[0]}, {self.canonical_timestamps[-1]}]",
            )

        # Locate the transition k where canonical_timestamps[k] < t < [k+1].
        k = next(
            i for i in range(len(self.canonical_timestamps) - 1)
            if self.canonical_timestamps[i] < t < self.canonical_timestamps[i + 1]
        )

        # End-of-confirmed-dwell anchor. The __post_init__ invariant
        # guarantees canonical_t_last[k] exists and matches t_first[k] when
        # the caller didn't supply confirmed-dwell info. Defensive clamp
        # protects against a malformed Posterior with t_last[k] > t_first[k+1].
        t_last_k = self.canonical_t_last[k]
        if t_last_k > self.canonical_timestamps[k + 1]:
            t_last_k = self.canonical_timestamps[k + 1]

        # Confirmed-dwell window: vehicle was at obs k's location.
        if t <= t_last_k:
            return self.state_marginals[k]

        if self.network is None:
            raise RuntimeError(
                "at_time off-grid resolution requires a RoadNetwork on the "
                "Posterior; reconstruct_trajectory attaches one automatically. "
                "Tests constructing Posteriors directly should pass network=...",
            )

        tau = (t - t_last_k).total_seconds()
        out: dict[State, float] = {}
        for path, weight in self.path_marginals[k].items():
            link_id, offset = position_in_transition(
                path, self.network, tau, rule,
            )
            state = StateV1(link_id=link_id, offset=offset, entry_time=t)
            out[state] = out.get(state, 0.0) + weight
        return out


def _segment_slices(
    state_cands: list[list[State]],
    path_cands: list[list[Path]],
) -> list[tuple[int, int]]:
    """Return half-open `[start, end)` slices into `state_cands` for the
    contiguous segments that survive splitting.

    A break occurs at:
        - any observation with empty `state_cands[k]` (off-network), or
        - any transition with empty `path_cands[k]` (no feasible path
          within the time budget).

    Single-observation segments are kept (they yield emission-only
    marginals via the T=1 branch in inference).
    """
    T = len(state_cands)
    segs: list[tuple[int, int]] = []
    i = 0
    while i < T:
        if not state_cands[i]:
            i += 1
            continue
        j = i
        while (
            j + 1 < T
            and state_cands[j + 1]
            and j < len(path_cands)
            and path_cands[j]
        ):
            j += 1
        segs.append((i, j + 1))
        i = j + 1
    return segs


def _min_routing_time_s(
    network: RoadNetwork,
    from_states: tuple[State, ...],
    to_states: tuple[State, ...],
) -> float:
    """Minimum graph travel time (seconds) from any source state's edge to
    any destination state's edge, ignoring start/end offsets.

    Used as a diagnostic on `Discontinuity` — distinguishes "barely missed
    the budget" from "wildly far in graph terms." Returns `inf` when no
    pair has a routable path or either side is empty.
    """
    if not from_states or not to_states:
        return float("inf")
    G = _get_nx_graph(network)
    best = float("inf")
    for src in from_states:
        try:
            src_idx = network.edge_index_for_link(src.link_id)
        except KeyError:
            continue
        src_to = int(network.to_node[src_idx])
        for dst in to_states:
            try:
                dst_idx = network.edge_index_for_link(dst.link_id)
            except KeyError:
                continue
            dst_from = int(network.from_node[dst_idx])
            if src_to == dst_from:
                d = 0.0
            elif src_to in G and dst_from in G:
                try:
                    d = nx.shortest_path_length(
                        G, src_to, dst_from, weight="weight",
                    )
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    continue
            else:
                continue
            if d < best:
                best = d
    return best


def _build_discontinuity(
    *,
    after_obs_idx: int,
    obs_before: CollapsedObservation,
    obs_after: CollapsedObservation,
    last_alive_states: tuple[State, ...],
    next_obs_candidates: tuple[State, ...],
    n_paths_in_transition: int,
    network: RoadNetwork,
) -> Discontinuity:
    """Assemble a `Discontinuity` from raw structural data."""
    dt = (obs_after.t_first - obs_before.t_first).total_seconds()
    distance_m = haversine_m(
        obs_before.lat, obs_before.lon, obs_after.lat, obs_after.lon,
    )
    speed = distance_m / dt if dt > 0 else 0.0
    return Discontinuity(
        after_obs_idx=after_obs_idx,
        last_alive_states=last_alive_states,
        next_obs_candidates=next_obs_candidates,
        time_budget=dt,
        n_paths_in_transition=n_paths_in_transition,
        speed_at_boundary=speed,
        min_unreachable_distance_s=_min_routing_time_s(
            network, last_alive_states, next_obs_candidates,
        ),
    )


def _run_segment(
    state_cands_seg: list[list[State]],
    path_cands_seg: list[list[Path]],
    observations_seg: list[CollapsedObservation],
    time_budgets_seg: list[float],
    confirmed_dwells_seg: list[float],
    config: Config,
    seg_indices: tuple[int, int],
    pre_filter_observations: list[CollapsedObservation],
    network: RoadNetwork,
) -> list[TrajectoryPosterior]:
    """Run graceful Viterbi + per-sub forward-backward.

    The graceful Viterbi splits the input segment at any forward-pass cliff
    (`delta` becoming all -inf). Each resulting `MostLikelySubTrajectory`
    gets its own forward-backward pass over the sub-range, yielding one
    `TrajectoryPosterior` per sub. Returns a list — typically length 1
    for clean segments, longer when cliffs split the input.
    """
    subs = most_likely_trajectory(
        state_cands_seg, path_cands_seg, observations_seg,
        config.emission, config.transition, time_budgets_seg,
    )
    if not subs:
        return []

    # Stale and replay diagnostics computed once per input segment; sliced
    # to each sub below.
    seg_start, _seg_end = seg_indices
    posteriors: list[TrajectoryPosterior] = []
    prev_sub: MostLikelySubTrajectory | None = None
    for sub in subs:
        # Cross-sub Discontinuity: the gap between this sub and the prior
        # one within the same input segment is a forward-pass cliff. The
        # surviving alive set at the end of `prev_sub` couldn't extend to
        # any state at obs `prev_sub.end_obs_idx + 1`.
        cliff_disc: Discontinuity | None = None
        if prev_sub is not None:
            after_local = prev_sub.end_obs_idx
            before_local = sub.start_obs_idx
            # Gap is the transition AT after_local (sub_local indexing).
            # The transition-paths list at that index is path_cands_seg[after_local].
            last_alive = tuple(
                state_cands_seg[after_local][i]
                for i in prev_sub.end_alive_state_indices
            )
            cliff_disc = _build_discontinuity(
                after_obs_idx=seg_start + after_local,
                obs_before=observations_seg[after_local],
                obs_after=observations_seg[before_local],
                last_alive_states=last_alive,
                next_obs_candidates=tuple(state_cands_seg[before_local]),
                n_paths_in_transition=len(path_cands_seg[after_local]),
                network=network,
            )

        s, e = sub.start_obs_idx, sub.end_obs_idx    # segment-local, inclusive
        sub_states = state_cands_seg[s:e + 1]
        sub_obs = observations_seg[s:e + 1]
        sub_paths = path_cands_seg[s:e] if e > s else []
        sub_budgets = time_budgets_seg[s:e] if e > s else []
        sub_dwells = confirmed_dwells_seg[s:e] if e > s else []

        sub_state_marg, sub_path_marg, sub_log_z = forward_backward(
            sub_states, sub_paths, sub_obs,
            config.emission, config.transition, sub_budgets,
        )
        sub_stale_idxs = tuple(
            i for i, o in enumerate(sub_obs) if o.stale_flagged
        )
        if sub_obs:
            sub_t_lo = sub_obs[0].t_first
            sub_t_hi = sub_obs[-1].t_first
            sub_replay_dropped = sum(
                1 for o in pre_filter_observations
                if o.dropped_during_replay
                and sub_t_lo <= o.t_first <= sub_t_hi
            )
        else:
            sub_replay_dropped = 0
        posteriors.append(TrajectoryPosterior(
            state_marginals=sub_state_marg,
            path_marginals=sub_path_marg,
            most_likely=sub.most_likely,
            log_partition=sub_log_z,
            canonical_timestamps=tuple(o.t_first for o in sub_obs),
            canonical_t_last=tuple(o.t_last for o in sub_obs),
            confirmed_dwell=tuple(sub_dwells),
            observation_indices=(seg_start + s, seg_start + e + 1),
            stale_observation_indices=sub_stale_idxs,
            replay_dropped_count=sub_replay_dropped,
            preceded_by_discontinuity=cliff_disc,
            network=network,
        ))
        prev_sub = sub
    return posteriors


def reconstruct_trajectory(
    raw_observations: list[RawObservation],
    network: RoadNetwork,
    config: Config,
) -> list[TrajectoryPosterior]:
    """Reconstruct the trajectory posterior over a road network.

    Returns a list of `TrajectoryPosterior` segments — one element for a
    trip with no discontinuities, multiple for trips that split. An empty
    list means no usable observations remained after preprocessing.
    """
    cleaned = clean(raw_observations)
    cleaned = drop_kinematic_spikes(
        cleaned,
        spike_speed_ms=config.spike_speed_ms,
        bridge_speed_ms=config.spike_bridge_speed_ms,
        max_spike_length=config.spike_max_length,
    )
    collapsed = collapse_by_uniqueness(cleaned, config.collapse_epsilon)

    # Narrow the network to the data's bbox before any operation that does
    # graph search. Stale detection routes A* per consecutive pair, and the
    # rest of the pipeline (state projection, path enumeration, FB) all hit
    # graph structures whose cost scales with edge count. Mexico-wide is
    # 17 M edges; a vehicle-day's bbox typically holds 50–200 K. Subgraph
    # construction itself is sub-second; net win is orders of magnitude.
    if collapsed:
        lats = [o.lat for o in collapsed]
        lons = [o.lon for o in collapsed]
        net = network.subgraph_for_bbox(
            min(lats), max(lats), min(lons), max(lons),
            buffer_m=config.subgraph_buffer_m,
        )
        # Apply caller-supplied typical-speed override (data-driven Porto
        # priors, custom region tuning, etc). Without this, `net` uses the
        # `V_TYPICAL_MS` defaults from `config.py`.
        if config.typical_speeds_by_class is not None:
            net.set_typical_speeds_by_class(config.typical_speeds_by_class)
        config = replace(
            config,
            emission=config.emission.rebind(net),
            transition=config.transition.rebind(net),
        )
    else:
        net = network

    collapsed = flag_stale_runs(collapsed, net, config.max_speed_factor)
    collapsed = drop_replay_bursts(
        collapsed,
        max_speed_ms=config.replay_max_speed_ms,
        max_speed_factor=config.max_speed_factor,
        k_consistent=config.replay_k_consistent,
        moving_threshold_ms=config.replay_moving_threshold_ms,
    )
    pre_filter_collapsed = collapsed
    collapsed = [o for o in collapsed if not o.dropped_during_replay]

    if not collapsed:
        return []

    state_cands = [
        project_observation(
            o, net,
            radius_meters=config.candidate_radius,
            max_candidates=config.max_state_candidates,
        )
        for o in collapsed
    ]

    if len(collapsed) >= 2:
        confirmed_dwells = [
            0.0 if collapsed[k].stale_flagged
            else (collapsed[k].t_last - collapsed[k].t_first).total_seconds()
            for k in range(len(collapsed) - 1)
        ]
        time_budgets = [
            (collapsed[k + 1].t_first - collapsed[k].t_first).total_seconds()
            - confirmed_dwells[k]
            for k in range(len(collapsed) - 1)
        ]
        path_cands = enumerate_paths_per_transition(
            state_cands, net, time_budgets,
            max_path_candidates=config.max_path_candidates,
            budget_slack=config.path_budget_slack,
            penalty_lambda=config.path_penalty_lambda,
            enable_offroad=config.enable_offroad_candidates,
            offroad_max_straight_m=config.offroad_max_straight_m,
            offroad_min_detour_ratio=config.offroad_min_detour_ratio,
            offroad_min_overslack=config.offroad_min_overslack,
        )
    else:
        confirmed_dwells = []
        time_budgets = []
        path_cands = []

    segments: list[TrajectoryPosterior] = []
    pre_seg_slices = _segment_slices(state_cands, path_cands)
    prev_pre_end: int | None = None    # trip-global obs index of last obs in
                                       # previous pre-segment's last sub
    prev_pre_alive: tuple[State, ...] = ()
    for start, end in pre_seg_slices:
        seg_states = state_cands[start:end]
        seg_paths = path_cands[start:end - 1] if end - start >= 2 else []
        seg_obs = collapsed[start:end]
        seg_budgets = time_budgets[start:end - 1] if end - start >= 2 else []
        seg_dwells = confirmed_dwells[start:end - 1] if end - start >= 2 else []
        sub_posteriors = _run_segment(
            seg_states, seg_paths, seg_obs, seg_budgets, seg_dwells, config,
            (start, end), pre_filter_collapsed, net,
        )
        # Cross-pre-segment Discontinuity: if there was a prior pre-segment,
        # the gap between its last sub and this pre-segment's first sub is
        # a structural split (empty state_cands or empty path_cands at the
        # boundary). Attach to this pre-segment's first sub.
        if (
            prev_pre_end is not None
            and sub_posteriors
            and sub_posteriors[0].preceded_by_discontinuity is None
        ):
            after_idx = prev_pre_end
            before_idx = start
            # The transition that triggered the split is `after_idx` →
            # `after_idx + 1`. Path candidates at that transition are
            # `path_cands[after_idx]` (might be empty by definition).
            n_paths = (
                len(path_cands[after_idx]) if after_idx < len(path_cands) else 0
            )
            sub_posteriors[0].preceded_by_discontinuity = _build_discontinuity(
                after_obs_idx=after_idx,
                obs_before=collapsed[after_idx],
                obs_after=collapsed[before_idx],
                last_alive_states=prev_pre_alive,
                next_obs_candidates=tuple(state_cands[before_idx]),
                n_paths_in_transition=n_paths,
                network=net,
            )
        segments.extend(sub_posteriors)
        # Track the alive set at the very end of THIS pre-segment for the
        # next iteration's structural Discontinuity construction.
        if sub_posteriors:
            last_post = sub_posteriors[-1]
            last_obs_global = last_post.observation_indices[1] - 1
            # Recover the alive states at last_obs_global from the input
            # state_cands; we don't track per-obs alive in the posterior,
            # but the most-likely State is in `most_likely[-1]`.
            ml_last = last_post.most_likely[-1] if last_post.most_likely else None
            prev_pre_alive = (
                (ml_last,) if ml_last is not None and not isinstance(ml_last, Path) else ()
            )
            prev_pre_end = last_obs_global
        else:
            prev_pre_alive = ()
            prev_pre_end = end - 1
    return segments
