"""Stale-jump detection — OVERVIEW.md §P2 / SPEC.md §stale_detection.

Compare the time available between consecutive collapsed observations
against the minimum feasible travel time on the network. A run is *stale*
when the available time looks infeasible if measured from the run's
`t_last`, but feasible if measured from `t_first` — i.e. the recovery jump
implies the static run was a frozen-replay artefact, not real dwelling.

The flag is informational. Inference always uses `t_first` as the canonical
timestamp, so the time budget for the transition naturally expands across
stale-flagged runs without any further bookkeeping. The flag's value is in
quality metrics and observability — surfaces the runs where the model is
relying on the staleness assumption rather than physical motion.
"""

from __future__ import annotations

from dataclasses import replace

from ..model import CollapsedObservation
from ..network import RoadNetwork


def flag_stale_runs(
    collapsed: list[CollapsedObservation],
    network: RoadNetwork,
    max_speed_factor: float = 1.2,
) -> list[CollapsedObservation]:
    """Return a new list of `CollapsedObservation` with `stale_flagged` set
    where the recovery-jump rule applies. Inputs are not mutated
    (`CollapsedObservation` is frozen).

    Rule (per OVERVIEW.md §P2):
        gap_from_last  = next.t_first - cur.t_last
        gap_from_first = next.t_first - cur.t_first
        if gap_from_last < min_travel and gap_from_first >= min_travel:
            cur is stale.
    """
    if len(collapsed) < 2:
        return list(collapsed)

    out: list[CollapsedObservation] = []
    for i in range(len(collapsed) - 1):
        cur = collapsed[i]
        nxt = collapsed[i + 1]
        min_travel = network.shortest_travel_time(
            cur.lat, cur.lon, nxt.lat, nxt.lon, max_speed_factor,
        )
        gap_last = (nxt.t_first - cur.t_last).total_seconds()
        gap_first = (nxt.t_first - cur.t_first).total_seconds()
        is_stale = gap_last < min_travel and gap_first >= min_travel
        out.append(replace(cur, stale_flagged=is_stale))
    out.append(collapsed[-1])
    return out
