"""Universal observation hygiene — SPEC.md §preprocessing.hygiene.clean.

Drops sentinel coordinates, out-of-range lat/lon, high-HDOP fixes, and
timestamps that are not strictly later than the previously-kept observation.
These are universal cleanups that apply to any `RawObservation` source;
feed-specific filtering (deleted-vehicle list, geographic bbox, kinematic
caps) lives in the feed adapter that produced the observations and is not
repeated here.
"""

from __future__ import annotations

from ..model import RawObservation

DEFAULT_HDOP_MAX: float = 10.0


def _is_valid_coord(lat: float, lon: float) -> bool:
    if lat == 0.0 and lon == 0.0:    # sentinel "no fix"
        return False
    if not (-90.0 <= lat <= 90.0):
        return False
    if not (-180.0 <= lon <= 180.0):
        return False
    return True


def clean(
    obs: list[RawObservation],
    *,
    hdop_max: float = DEFAULT_HDOP_MAX,
) -> list[RawObservation]:
    """Drop sentinel coords, out-of-range lat/lon, fixes with HDOP > `hdop_max`,
    and observations whose timestamp is not strictly later than the previously
    kept observation.

    Behaviour: drop the offending observation. A configurable repair mode
    is not implemented — repair semantics for monotonicity are ambiguous
    (snap to prev+ε? interpolate?) and there's no evidence yet that any
    feed's monotonicity violations are recoverable.
    """
    out: list[RawObservation] = []
    last_ts = None
    for o in obs:
        if not _is_valid_coord(o.lat, o.lon):
            continue
        if o.hdop is not None and o.hdop > hdop_max:
            continue
        if last_ts is not None and o.timestamp <= last_ts:
            continue
        out.append(o)
        last_ts = o.timestamp
    return out
