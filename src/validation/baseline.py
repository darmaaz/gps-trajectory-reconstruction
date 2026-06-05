"""Tier 3 — comparison against the trivial snap-to-nearest baseline.

A "snap-to-nearest" matcher just projects each observation independently
to the closest edge with no inference, no transition factor, no
smoothing. Comparing the CRF's most-likely state assignment against this
baseline shows whether the inference layer is actually doing useful work
over raw projection.

If the CRF agrees with snap-to-nearest on every observation, the driver
model isn't pulling its weight. If they systematically disagree where the
CRF prefers smoother / more-on-road choices, the inference is contributing.
"""

from __future__ import annotations

from ..model import CollapsedObservation, Path, State, StateV1
from ..network import RoadNetwork


def snap_to_nearest_states(
    observations: list[CollapsedObservation],
    network: RoadNetwork,
    radius_meters: float = 200.0,
) -> list[State | None]:
    """Project each observation independently to its nearest edge."""
    out: list[State | None] = []
    for o in observations:
        hits = network.project_point(
            o.lat, o.lon,
            radius_meters=radius_meters, max_candidates=1,
        )
        if not hits:
            out.append(None)
            continue
        idx, offset_m, _ = hits[0]
        out.append(StateV1(
            link_id=int(network.edge_ids[idx]),
            offset=offset_m,
            entry_time=o.t_first,
        ))
    return out


def viterbi_states_per_obs(segments) -> list[State]:
    """Extract the most-likely state at each observation across all
    segments, in segment-then-obs order."""
    out: list[State] = []
    for seg in segments:
        # `most_likely` interleaves state, path, state, path, ..., state
        # at indices 0, 1, 2, ...; states are at even positions.
        for i, item in enumerate(seg.most_likely):
            if i % 2 == 0 and not isinstance(item, Path):
                out.append(item)
    return out


def edge_disagreement_rate(
    snap_states: list[State | None],
    viterbi_states: list[State | None],
) -> float:
    """Fraction of obs where the snap and Viterbi states differ on
    `link_id`. Pairs where either side is None are skipped (no projection
    available)."""
    n_compared = 0
    n_disagree = 0
    for s, v in zip(snap_states, viterbi_states):
        if s is None or v is None:
            continue
        n_compared += 1
        if int(s.link_id) != int(v.link_id):
            n_disagree += 1
    return n_disagree / n_compared if n_compared else 0.0
