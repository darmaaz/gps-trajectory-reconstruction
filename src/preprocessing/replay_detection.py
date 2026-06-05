"""Buffered-replay burst detection — drop-and-bridge data-quality filter.

When a vehicle's cellular uplink is interrupted, fleet devices commonly:

    1. Continue emitting frozen pings (stale-frozen pattern, caught by
       `flag_stale_runs` and absorbed into one collapsed observation).
    2. After connection recovery, dump buffered positions one at a time
       at the device's heartbeat cadence — but with current upload
       timestamps, not the original capture times. The result is a
       "catch-up burst" that looks like impossible motion.

This module implements drop-and-bridge for these bursts:

    - Use a stale-flagged observation as the start sentinel for a
      potential replay zone.
    - Walk forward, marking observations as part of the burst zone until
      stability is restored. Stability = `K` consecutive observations
      that are (a) internally consistent (no frozen-chip pattern within
      the run) AND (b) connected by kinematically feasible transitions
      at canonical-time speeds.
    - Mark every observation between the stale-flagged anchor and the
      first stable observation as `dropped_during_replay=True`. The
      orchestrator filters these out before path enumeration; the
      bookend transition then sees the full clock-time budget, which
      is honest because both bookends are reliable.

`K=2` is the default closure-streak length. `K=1` would let a single
feasible-looking transition close the zone prematurely. `K=3` is more
conservative but rarely matters in practice.

"Internally consistent" means the run does not show the frozen-chip
pattern: either `collapsed_count <= 1` (single ping, no internal
motion to check) or `reported_speed_max_ms < moving_threshold_ms` (the
device wasn't claiming the vehicle was moving while the position stayed
pinned). The frozen pattern is `count > 1 AND max_reported > moving_threshold`.

Known limitation: if the upstream feed has buffered replay WITHOUT a
preceding stale-frozen run — i.e., the chip went
silent during outage instead of emitting frozen pings — the stale flag
never fires and this rule doesn't trigger. The fallback is the
pipeline's existing segmentation-on-infeasibility, which catches the
late catch-up burst but leaves the post-recovery zone fragmented.
Document the gap; revisit if production data shows the silent-outage
variant matters.
"""

from __future__ import annotations

from dataclasses import replace

from ..geo import haversine_m
from ..model import CollapsedObservation

DEFAULT_MAX_SPEED_MS: float = 40.0      # 144 km/h — V_MAX_FALLBACK in config
DEFAULT_MAX_SPEED_FACTOR: float = 1.2
DEFAULT_K_CONSISTENT: int = 2
DEFAULT_MOVING_THRESHOLD_MS: float = 1.0   # ~3.6 km/h: any reading above is "claiming motion"


def _internally_consistent(
    o: CollapsedObservation, moving_threshold_ms: float,
) -> bool:
    """Return False for the frozen-chip pattern only.

    `count == 1`: nothing internal to disagree about → consistent.
    `reported_speed_max_ms is None`: feed didn't expose speed → can't
        prove inconsistency → consistent (fail-open).
    `count > 1 AND max_reported >= moving_threshold`: chip claimed
        motion while position was pinned → inconsistent.
    """
    if o.collapsed_count <= 1:
        return True
    if o.reported_speed_max_ms is None:
        return True
    return o.reported_speed_max_ms < moving_threshold_ms


def _transition_feasible(
    a: CollapsedObservation, b: CollapsedObservation, speed_threshold_ms: float,
) -> bool:
    dt_s = (b.t_first - a.t_first).total_seconds()
    if dt_s <= 0:
        return False
    distance_m = float(haversine_m(a.lat, a.lon, b.lat, b.lon))
    return distance_m / dt_s <= speed_threshold_ms


def _is_consistent_chain(
    obs: list[CollapsedObservation],
    start: int,
    k: int,
    speed_threshold_ms: float,
    moving_threshold_ms: float,
) -> bool:
    """`obs[start..start+k-1]` are all internally consistent AND the
    transitions between them are kinematically feasible."""
    for offset in range(k):
        idx = start + offset
        if not _internally_consistent(obs[idx], moving_threshold_ms):
            return False
        if offset > 0 and not _transition_feasible(
            obs[idx - 1], obs[idx], speed_threshold_ms,
        ):
            return False
    return True


def _find_burst_end(
    obs: list[CollapsedObservation],
    stale_idx: int,
    speed_threshold_ms: float,
    moving_threshold_ms: float,
    k: int,
) -> int | None:
    """Smallest `j > stale_idx` such that `obs[j..j+k-1]` is a consistent
    chain. Returns None if no such window exists in the remaining sequence.
    """
    n = len(obs)
    for j in range(stale_idx + 1, n - k + 1):
        if _is_consistent_chain(
            obs, j, k, speed_threshold_ms, moving_threshold_ms,
        ):
            return j
    return None


def drop_replay_bursts(
    observations: list[CollapsedObservation],
    *,
    max_speed_ms: float = DEFAULT_MAX_SPEED_MS,
    max_speed_factor: float = DEFAULT_MAX_SPEED_FACTOR,
    k_consistent: int = DEFAULT_K_CONSISTENT,
    moving_threshold_ms: float = DEFAULT_MOVING_THRESHOLD_MS,
) -> list[CollapsedObservation]:
    """Mark observations belonging to a buffered-replay burst as dropped.

    Walks the sequence; each `stale_flagged` observation triggers a
    burst-zone candidate ending at the first `k_consistent`-length
    consistent chain. Observations strictly between the sentinel and
    the chain start get `dropped_during_replay=True` on a copy of the
    list.

    `CollapsedObservation` is frozen, so this returns a new list with
    `dataclasses.replace`-d entries; the input is not mutated.
    """
    if not observations:
        return list(observations)

    speed_threshold_ms = max_speed_ms * max_speed_factor
    out = list(observations)

    i = 0
    while i < len(out):
        if not out[i].stale_flagged:
            i += 1
            continue
        end_idx = _find_burst_end(
            out, i, speed_threshold_ms, moving_threshold_ms, k_consistent,
        )
        if end_idx is None or end_idx <= i + 1:
            # No closure within this trip, or stable region begins
            # immediately after the stale anchor (no burst). Move on.
            i += 1
            continue
        for m in range(i + 1, end_idx):
            out[m] = replace(out[m], dropped_during_replay=True)
        i = end_idx

    return out
