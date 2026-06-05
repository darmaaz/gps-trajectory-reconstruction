"""Kinematic-spike removal — drops GPS chip-glitch out-and-back artefacts.

Some upstream feeds (Porto Kaggle is a known example) emit short runs of
pings where the GPS chip momentarily reports a position several km away
from the true location, then snaps back. Cadence is preserved, the
upstream-quality flag stays clean, and per-ping values are internally
consistent — so plain-Gaussian outlier filters miss them. The signature
is purely *kinematic*: the implied vehicle speed entering and leaving the
spike is far above any feasible vehicle.

Algorithm
---------
For each ping `i` whose outbound transition speed `(i → i+1)` exceeds
`spike_speed_ms`, scan ahead at most `max_spike_length` pings for a
ping `j` whose inbound transition `(j-1 → j)` is also infeasible AND
whose bridging transition `(i → j)` is feasible (`≤ bridge_speed_ms`).
If such a `j` exists, the pings strictly between `i` and `j` are an
out-and-back chip glitch: drop them, resume scanning from `j`.

If no `j` is found within the run-length cap, the high-speed transition
is left alone — it might be sustained noise, a real long-distance
teleport in the data (which the rest of the pipeline already handles
via segment splitting), or a genuine fast vehicle on a highway.

Threshold rationale
-------------------
- `spike_speed_ms = 150 m/s` (540 km/h). Far above any real vehicle.
  Set deliberately above `MAX_REASONABLE_SPEED_MS = 100 m/s` so that
  pings whose implied speed sits in the 100–150 m/s "marginally
  infeasible" zone are *kept* — they're more likely to be real fixes
  with GPS along-track jitter than chip glitches, and the heavy-tailed
  Student-t emission can absorb them per-observation. Only the
  unambiguous garbage (chip glitches typically register at 200+ m/s)
  is removed here.
- `bridge_speed_ms = 100 m/s`. The surviving transition must itself
  be feasible for a vehicle. If the bridge isn't feasible we're
  papering over a real data problem; segment splitting handles that
  downstream.
- `max_spike_length = 3` pings. A two-ping spike (the canonical case
  observed in Porto) is well within this; longer runs are signs of
  sustained noise or an off-network detour that should NOT be silently
  patched. They survive this step and end up isolated in their own
  segment via empty path-enumeration → `_segment_slices` split.

The function is idempotent on already-clean data and a no-op when
`pings` has fewer than three elements.
"""

from __future__ import annotations

from ..geo import haversine_m
from ..model import RawObservation

DEFAULT_SPIKE_SPEED_MS: float = 150.0
DEFAULT_BRIDGE_SPEED_MS: float = 100.0
DEFAULT_MAX_SPIKE_LENGTH: int = 3


def drop_kinematic_spikes(
    pings: list[RawObservation],
    *,
    spike_speed_ms: float = DEFAULT_SPIKE_SPEED_MS,
    bridge_speed_ms: float = DEFAULT_BRIDGE_SPEED_MS,
    max_spike_length: int = DEFAULT_MAX_SPIKE_LENGTH,
) -> list[RawObservation]:
    """Drop out-and-back GPS chip-glitch spikes from a ping sequence.

    See module docstring for the algorithm and threshold rationale.
    Returns the surviving pings in original order; pings inside a
    detected spike run are removed.
    """
    n = len(pings)
    if n < 3:
        return pings

    def _speed(a: int, b: int) -> float:
        dt = (pings[b].timestamp - pings[a].timestamp).total_seconds()
        if dt <= 0:
            return float("nan")
        d = haversine_m(
            pings[a].lat, pings[a].lon, pings[b].lat, pings[b].lon,
        )
        return float(d) / dt

    drop = [False] * n
    i = 0
    while i < n - 1:
        v_out = _speed(i, i + 1)
        if v_out != v_out or v_out <= spike_speed_ms:    # NaN or feasible
            i += 1
            continue
        # Outbound infeasible. Scan candidate snap-back pings.
        # Spike pings = pings[i+1 .. j-1]; length = j - i - 1, must be
        # ≤ max_spike_length, so j ≤ i + 1 + max_spike_length.
        found_j = -1
        for j in range(i + 2, min(i + 2 + max_spike_length, n)):
            v_in = _speed(j - 1, j)
            if v_in != v_in or v_in <= spike_speed_ms:
                continue
            v_bridge = _speed(i, j)
            if v_bridge != v_bridge:
                continue
            if v_bridge <= bridge_speed_ms:
                found_j = j
                break
        if found_j >= 0:
            for k in range(i + 1, found_j):
                drop[k] = True
            i = found_j
        else:
            i += 1

    if not any(drop):
        return pings
    return [p for k, p in enumerate(pings) if not drop[k]]
