"""Position-uniqueness collapse — OVERVIEW.md §P1 / SPEC.md §collapse.

Consecutive observations within ε metres of the run's anchor are merged into
a single `CollapsedObservation`. Stale runs and genuine stops both look like
static periods at the observation level, so the inference layer doesn't have
to distinguish them — the model treats them uniformly and stale-jump
detection (P2) handles the time-budget consequence separately.
"""

from __future__ import annotations

from ..geo import haversine_m
from ..model import CollapsedObservation, RawObservation


def collapse_by_uniqueness(
    obs: list[RawObservation],
    epsilon_meters: float = 5.0,
) -> list[CollapsedObservation]:
    """Walk the sequence anchor-by-anchor.

    A new run starts whenever an observation falls outside the current
    anchor's ε-disk. The anchor's lat/lon is the run's representative
    position; `t_first`/`t_last` span the run's first and last observation;
    `collapsed_count` records how many raw pings the run absorbed. ε defaults
    to ~the GPS noise scale.

    Tracks the maximum `reported_speed` across the raw pings absorbed into
    each run as `reported_speed_max_ms`. This signals the frozen-chip
    pattern downstream — a run with `collapsed_count > 1` and
    `reported_speed_max_ms` clearly above zero means the device kept
    reporting motion while the position stayed pinned, which is the
    fingerprint replay-burst detection looks for.
    """
    if not obs:
        return []

    def _max_speed(current: float | None, new: float | None) -> float | None:
        if new is None:
            return current
        if current is None:
            return new
        return max(current, new)

    out: list[CollapsedObservation] = []
    anchor = obs[0]
    t_last = anchor.timestamp
    count = 1
    max_reported = anchor.reported_speed

    for o in obs[1:]:
        d = float(haversine_m(anchor.lat, anchor.lon, o.lat, o.lon))
        if d <= epsilon_meters:
            t_last = o.timestamp
            count += 1
            max_reported = _max_speed(max_reported, o.reported_speed)
            continue
        out.append(CollapsedObservation(
            lat=anchor.lat, lon=anchor.lon,
            t_first=anchor.timestamp, t_last=t_last,
            collapsed_count=count,
            reported_speed_max_ms=max_reported,
        ))
        anchor = o
        t_last = o.timestamp
        count = 1
        max_reported = o.reported_speed

    out.append(CollapsedObservation(
        lat=anchor.lat, lon=anchor.lon,
        t_first=anchor.timestamp, t_last=t_last,
        collapsed_count=count,
        reported_speed_max_ms=max_reported,
    ))
    return out
