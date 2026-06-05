"""hygiene.clean, collapse_by_uniqueness, flag_stale_runs."""

from __future__ import annotations

import math
from datetime import timedelta

import pytest

from src.model import CollapsedObservation, RawObservation
from src.preprocessing import clean, collapse_by_uniqueness, flag_stale_runs


# ---------------------------------------------------------------- hygiene


def _raw(t0, secs: float, lat: float = 19.43, lon: float = -99.13, **kw):
    return RawObservation(
        timestamp=t0 + timedelta(seconds=secs), lat=lat, lon=lon, **kw,
    )


def test_clean_drops_sentinel_oor_hdop_monotonicity(t0):
    raw = [
        _raw(t0, 0),                              # ok
        RawObservation(timestamp=t0 + timedelta(seconds=1), lat=0.0, lon=0.0),    # sentinel
        _raw(t0, 2, hdop=20.0),                   # high HDOP
        _raw(t0, 3, lat=200.0),                   # OOR lat
        _raw(t0, 4),                              # ok
        _raw(t0, 4),                              # tied → drop (strict-monotonic)
        _raw(t0, 3),                              # back-in-time → drop
        _raw(t0, 5),                              # ok
    ]
    out = clean(raw)
    assert len(out) == 3
    assert [o.timestamp.second for o in out] == [0, 4, 5]


def test_clean_passes_when_hdop_unset(t0):
    raw = [_raw(t0, 0, hdop=None), _raw(t0, 1, hdop=None)]
    assert len(clean(raw)) == 2


# -------------------------------------------------------------- collapse


def _nearby(t0, secs: float, base_lat: float, base_lon: float,
            dx_m: float = 0.0, dy_m: float = 0.0):
    dlat = dy_m / 111_320.0
    dlon = dx_m / (111_320.0 * math.cos(math.radians(base_lat)))
    return RawObservation(
        timestamp=t0 + timedelta(seconds=secs),
        lat=base_lat + dlat, lon=base_lon + dlon,
    )


def test_collapse_empty_and_single():
    assert collapse_by_uniqueness([]) == []
    raw = [RawObservation(
        timestamp=__import__("datetime").datetime(2026, 5, 5),
        lat=19.43, lon=-99.13,
    )]
    out = collapse_by_uniqueness(raw)
    assert len(out) == 1 and out[0].collapsed_count == 1


def test_collapse_merges_run_then_starts_new(t0):
    base_lat, base_lon = 19.43, -99.13
    raw = [
        _nearby(t0, 0, base_lat, base_lon, 0, 0),
        _nearby(t0, 1, base_lat, base_lon, 1.0, 0.5),
        _nearby(t0, 2, base_lat, base_lon, -0.7, 1.2),
        _nearby(t0, 3, base_lat, base_lon, 0.3, -0.5),
        _nearby(t0, 4, base_lat, base_lon, 1.5, 1.0),
        _nearby(t0, 20, base_lat, base_lon, 30.0, 0.0),    # ~30m east → new run
        _nearby(t0, 21, base_lat, base_lon, 30.5, 0.5),
    ]
    out = collapse_by_uniqueness(raw, epsilon_meters=5.0)
    assert [c.collapsed_count for c in out] == [5, 2]
    assert (out[0].t_last - out[0].t_first).total_seconds() == 4
    assert out[0].t_first == raw[0].timestamp
    assert out[1].t_first == raw[5].timestamp


# ---------------------------------------------------------- stale_detection


def _co(lat, lon, tf, tl, count=1):
    return CollapsedObservation(
        lat=lat, lon=lon, t_first=tf, t_last=tl, collapsed_count=count,
    )


def test_stale_detection_flags_stale_recovery_jump(grid_network, t0):
    cur = _co(19.430, -99.135, t0, t0 + timedelta(seconds=200), count=20)
    nxt = _co(19.430, -99.125,
              t0 + timedelta(seconds=210), t0 + timedelta(seconds=210))
    out = flag_stale_runs([cur, nxt], grid_network)
    assert out[0].stale_flagged is True
    assert out[1].stale_flagged is False


def test_stale_detection_does_not_flag_genuine_pause(grid_network, t0):
    cur = _co(19.430, -99.135, t0, t0 + timedelta(seconds=10), count=2)
    nxt = _co(19.430, -99.125,
              t0 + timedelta(seconds=80), t0 + timedelta(seconds=80))
    out = flag_stale_runs([cur, nxt], grid_network)
    assert out[0].stale_flagged is False


def test_stale_detection_passthrough_for_short_input(grid_network, t0):
    # 0 or 1 observation: nothing to compare → return inputs unchanged.
    assert flag_stale_runs([], grid_network) == []
    one = [_co(19.43, -99.13, t0, t0)]
    assert flag_stale_runs(one, grid_network)[0].stale_flagged is False


# ---------------------------------------------------------- replay bursts


from datetime import datetime, timedelta as _td  # noqa: E402

from src.preprocessing import drop_replay_bursts  # noqa: E402


def _co_with_speed(lat, lon, tf, tl, count, max_speed_ms=None, stale=False):
    return CollapsedObservation(
        lat=lat, lon=lon, t_first=tf, t_last=tl,
        collapsed_count=count, stale_flagged=stale,
        reported_speed_max_ms=max_speed_ms,
    )


def test_replay_bursts_drops_post_stale_catchup_zone():
    """A frozen run followed by an internally-inconsistent cluster and a
    string of impossible jumps to a stable point: drop the burst zone,
    keep both bookends."""
    base = datetime(2026, 3, 3, 1, 34, 56)
    obs = [
        # stale-flagged anchor at A — vehicle was here at t_first
        _co_with_speed(21.276, -101.288, base, base + _td(seconds=1440),
                       count=25, max_speed_ms=24.0, stale=True),
        # 3-ping cluster at B with reported_speed > 0 (frozen-chip fingerprint)
        _co_with_speed(21.167, -101.369,
                       base + _td(minutes=25), base + _td(minutes=27),
                       count=3, max_speed_ms=15.3),
        # single ping at C — internally consistent
        _co_with_speed(21.163, -101.371,
                       base + _td(minutes=28), base + _td(minutes=28),
                       count=1, max_speed_ms=18.0),
        # impossible jump to D
        _co_with_speed(21.114, -101.384,
                       base + _td(minutes=29), base + _td(minutes=29),
                       count=1, max_speed_ms=26.0),
        # impossible jump to E
        _co_with_speed(21.063, -101.396,
                       base + _td(minutes=30), base + _td(minutes=30),
                       count=1, max_speed_ms=12.8),
        # near-arrival at F — internally consistent
        _co_with_speed(20.974, -101.418,
                       base + _td(minutes=31), base + _td(minutes=31),
                       count=1, max_speed_ms=20.6),
        # G — consistent and feasibly reachable from F
        _co_with_speed(20.971, -101.420,
                       base + _td(minutes=32), base + _td(minutes=32),
                       count=1, max_speed_ms=17.5),
    ]
    out = drop_replay_bursts(obs, k_consistent=2)
    flags = [o.dropped_during_replay for o in out]
    # Bookends (stale anchor and last two consistent obs) are kept.
    assert flags[0] is False
    assert flags[-1] is False
    assert flags[-2] is False
    # Burst-zone members between are dropped.
    assert flags[1:-2] == [True, True, True, True]


def test_replay_bursts_no_stale_no_drops():
    """Without a stale-flagged anchor, the rule doesn't fire — the
    silent-outage failure mode documented in the module docstring."""
    base = datetime(2026, 3, 3, 0, 0)
    obs = [
        _co_with_speed(21.0, -101.0, base, base, count=1, max_speed_ms=20.0),
        _co_with_speed(21.1, -101.0,    # impossibly far in 1 s
                       base + _td(seconds=1), base + _td(seconds=1),
                       count=1, max_speed_ms=22.0),
        _co_with_speed(21.2, -101.0,
                       base + _td(seconds=2), base + _td(seconds=2),
                       count=1, max_speed_ms=24.0),
    ]
    out = drop_replay_bursts(obs)
    assert all(not o.dropped_during_replay for o in out)


def test_replay_bursts_k_consistent_avoids_premature_closure():
    """K=2 must not let a single feasible-looking transition close the
    zone too early."""
    base = datetime(2026, 3, 3, 1, 0)
    obs = [
        _co_with_speed(21.0, -101.0, base, base + _td(seconds=600),
                       count=10, max_speed_ms=22.0, stale=True),
        # B: frozen-chip fingerprint → inside the burst
        _co_with_speed(21.05, -101.0,
                       base + _td(seconds=700), base + _td(seconds=730),
                       count=3, max_speed_ms=15.0),
        # C: feasibly close to B but only one consistent step — K=2 wants two
        _co_with_speed(21.06, -101.0,
                       base + _td(seconds=800), base + _td(seconds=800),
                       count=1, max_speed_ms=22.0),
        # D: huge jump back into burstland
        _co_with_speed(22.0, -101.0,
                       base + _td(seconds=860), base + _td(seconds=860),
                       count=1, max_speed_ms=22.0),
        # E, F: two-step consistent run — closes the zone here
        _co_with_speed(22.001, -101.0,
                       base + _td(seconds=920), base + _td(seconds=920),
                       count=1, max_speed_ms=22.0),
        _co_with_speed(22.002, -101.0,
                       base + _td(seconds=980), base + _td(seconds=980),
                       count=1, max_speed_ms=22.0),
    ]
    flags_k2 = [o.dropped_during_replay for o in drop_replay_bursts(obs, k_consistent=2)]
    flags_k1 = [o.dropped_during_replay for o in drop_replay_bursts(obs, k_consistent=1)]
    # Under K=1, obs[2] (the lone consistent point) closes the zone, so
    # only obs[1] gets dropped.
    assert flags_k1 == [False, True, False, False, False, False]
    # Under K=2, obs[2] alone isn't enough — the chain needs obs[3] too,
    # but obs[2]→obs[3] is infeasible. Closure shifts to obs[3], so
    # obs[1] AND obs[2] drop. obs[3] is kept as the first post-burst
    # bookend; the orchestrator's segmentation handles the still-impossible
    # bookend bridge from obs[0] separately.
    assert flags_k2 == [False, True, True, False, False, False]


def test_replay_bursts_input_not_mutated():
    base = datetime(2026, 3, 3)
    obs = [
        _co_with_speed(21.0, -101.0, base, base, count=1, max_speed_ms=0.0,
                       stale=True),
        _co_with_speed(22.0, -101.0,
                       base + _td(seconds=10), base + _td(seconds=10),
                       count=1, max_speed_ms=0.0),
        _co_with_speed(22.001, -101.0,
                       base + _td(seconds=70), base + _td(seconds=70),
                       count=1, max_speed_ms=0.0),
        _co_with_speed(22.002, -101.0,
                       base + _td(seconds=130), base + _td(seconds=130),
                       count=1, max_speed_ms=0.0),
    ]
    snapshot = [o.dropped_during_replay for o in obs]
    drop_replay_bursts(obs)
    assert [o.dropped_during_replay for o in obs] == snapshot
