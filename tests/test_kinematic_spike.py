"""Tests for drop_kinematic_spikes — out-and-back GPS chip-glitch removal."""

from __future__ import annotations

from datetime import timedelta

from src.model import RawObservation
from src.preprocessing import drop_kinematic_spikes


def _raw(t0, secs: float, lat: float, lon: float) -> RawObservation:
    return RawObservation(
        timestamp=t0 + timedelta(seconds=secs), lat=lat, lon=lon,
    )


class TestDropKinematicSpikes:
    def test_porto_two_ping_spike_is_dropped(self, t0):
        # Reproduces the obs 33/34 fixture from Porto trip
        # 1372636951620000320: two pings teleport ~4.5 km away then snap back.
        pings = [
            _raw(t0, 0,  41.15430, -8.64994),     # central Porto, slow-moving
            _raw(t0, 15, 41.14174, -8.59938),     # +4.45 km in 15s — spike out
            _raw(t0, 30, 41.14057, -8.59653),     # spike continues
            _raw(t0, 45, 41.15429, -8.65008),     # snap back ~12m from start
            _raw(t0, 60, 41.15429, -8.65008),     # taxi continues
        ]
        out = drop_kinematic_spikes(pings)
        assert len(out) == 3
        # Pings 0, 3, 4 (zero-indexed) survive.
        assert out[0].timestamp == pings[0].timestamp
        assert out[1].timestamp == pings[3].timestamp
        assert out[2].timestamp == pings[4].timestamp

    def test_single_ping_spike_is_dropped(self, t0):
        pings = [
            _raw(t0, 0,  41.150, -8.650),
            _raw(t0, 15, 41.180, -8.620),         # +~4 km in 15s — spike
            _raw(t0, 30, 41.150, -8.650),         # snap back
            _raw(t0, 45, 41.150, -8.650),
        ]
        out = drop_kinematic_spikes(pings)
        assert len(out) == 3
        assert [o.lat for o in out] == [41.150, 41.150, 41.150]

    def test_real_highway_motion_is_not_flagged(self, t0):
        # A taxi cruising 110 km/h ≈ 30.5 m/s on a highway. Each 15s pair
        # covers ~458 m, well below the 100 m/s threshold. Must not trigger.
        # Six consecutive pings driving roughly east at this speed.
        pings = [
            _raw(t0, 15 * i, 41.150, -8.650 + 0.0055 * i)
            for i in range(6)
        ]
        out = drop_kinematic_spikes(pings)
        assert len(out) == 6

    def test_sustained_noise_run_is_left_alone(self, t0):
        # Four consecutive infeasible transitions with no clean snap-back —
        # this is a real data problem, not a chip glitch. We refuse to
        # silently drop it; segment splitting handles it downstream.
        pings = [
            _raw(t0, 0,  41.150, -8.650),
            _raw(t0, 15, 41.300, -8.500),         # spike
            _raw(t0, 30, 41.450, -8.350),         # still spiking, not back
            _raw(t0, 45, 41.600, -8.200),         # still drifting
            _raw(t0, 60, 41.750, -8.050),         # still drifting
            _raw(t0, 75, 41.900, -7.900),         # gone
        ]
        out = drop_kinematic_spikes(pings)
        assert len(out) == 6     # nothing dropped

    def test_spike_with_infeasible_bridge_is_left_alone(self, t0):
        # Outbound is infeasible AND inbound is infeasible BUT bridging from
        # ping 0 to ping 3 is itself infeasible (the snap-back is to a
        # different teleport, not the original path). Must not drop.
        pings = [
            _raw(t0, 0,  41.150, -8.650),
            _raw(t0, 15, 41.300, -8.500),         # spike out
            _raw(t0, 30, 41.000, -8.800),         # snap back, but to far away
            _raw(t0, 45, 41.500, -8.300),         # bridge 0→3 is infeasible
        ]
        out = drop_kinematic_spikes(pings)
        assert len(out) == 4

    def test_max_spike_length_caps_run(self, t0):
        # 5 spike pings in the middle — exceeds default max_spike_length=3.
        # The bridge is feasible, but the run is too long to call a chip
        # glitch confidently. Must not drop.
        spike_lat, spike_lon = 41.300, -8.500
        pings = [
            _raw(t0, 0, 41.150, -8.650),
            _raw(t0, 15, spike_lat, spike_lon),
            _raw(t0, 30, spike_lat, spike_lon),
            _raw(t0, 45, spike_lat, spike_lon),
            _raw(t0, 60, spike_lat, spike_lon),
            _raw(t0, 75, spike_lat, spike_lon),
            _raw(t0, 90, 41.151, -8.650),         # bridge would be feasible
        ]
        out = drop_kinematic_spikes(pings, max_spike_length=3)
        assert len(out) == 7

    def test_max_spike_length_can_be_raised(self, t0):
        # Same data as the cap test, but lifting the cap → drops.
        spike_lat, spike_lon = 41.300, -8.500
        pings = [
            _raw(t0, 0, 41.150, -8.650),
            _raw(t0, 15, spike_lat, spike_lon),
            _raw(t0, 30, spike_lat, spike_lon),
            _raw(t0, 45, spike_lat, spike_lon),
            _raw(t0, 60, spike_lat, spike_lon),
            _raw(t0, 75, spike_lat, spike_lon),
            _raw(t0, 90, 41.151, -8.650),
        ]
        out = drop_kinematic_spikes(pings, max_spike_length=5)
        assert len(out) == 2
        assert [o.lat for o in out] == [41.150, 41.151]

    def test_two_independent_spikes_are_both_dropped(self, t0):
        pings = [
            _raw(t0, 0,   41.150, -8.650),
            _raw(t0, 15,  41.300, -8.500),         # spike A out
            _raw(t0, 30,  41.150, -8.650),         # snap back
            _raw(t0, 45,  41.151, -8.651),
            _raw(t0, 60,  41.400, -8.400),         # spike B out
            _raw(t0, 75,  41.152, -8.652),         # snap back
            _raw(t0, 90,  41.153, -8.653),
        ]
        out = drop_kinematic_spikes(pings)
        assert len(out) == 5
        assert [round(o.lat, 3) for o in out] == [41.150, 41.150, 41.151, 41.152, 41.153]

    def test_short_input_is_returned_unchanged(self, t0):
        assert drop_kinematic_spikes([]) == []
        assert len(drop_kinematic_spikes([_raw(t0, 0, 41.0, -8.0)])) == 1
        two = [_raw(t0, 0, 41.0, -8.0), _raw(t0, 15, 41.5, -8.5)]
        assert len(drop_kinematic_spikes(two)) == 2

    def test_threshold_above_actual_spike_does_nothing(self, t0):
        # Bumping spike_speed_ms above the implied speed disarms the filter.
        pings = [
            _raw(t0, 0,  41.150, -8.650),
            _raw(t0, 15, 41.180, -8.620),         # ~4 km/15s ≈ 270 m/s
            _raw(t0, 30, 41.150, -8.650),
        ]
        out = drop_kinematic_spikes(pings, spike_speed_ms=500.0)
        assert len(out) == 3

    def test_marginal_speed_below_default_threshold_is_kept(self, t0):
        # ~1900 m in 15s ≈ 127 m/s = 457 km/h. Above the old 100 m/s
        # threshold, but below the current 150 m/s default — must be kept.
        # The intent: pings in this zone are more likely real fast fixes
        # with along-track GPS jitter than chip glitches; drop only the
        # unambiguous garbage and let the heavy-tailed emission absorb
        # the marginal cases per-observation.
        pings = [
            _raw(t0, 0,  41.150, -8.650),
            _raw(t0, 15, 41.167, -8.650),         # ≈ 1900 m north → ≈127 m/s
            _raw(t0, 30, 41.150, -8.650),         # snap back
        ]
        out = drop_kinematic_spikes(pings)
        assert len(out) == 3                      # at default threshold, kept
        # ...and dropped only if the operator explicitly tightens.
        out_strict = drop_kinematic_spikes(pings, spike_speed_ms=100.0)
        assert len(out_strict) == 2

    def test_zero_or_negative_dt_does_not_crash(self, t0):
        # Two pings at the same timestamp — clean() should usually have
        # already removed these, but the function must defend against it.
        pings = [
            _raw(t0, 0,  41.150, -8.650),
            _raw(t0, 0,  41.150, -8.650),         # tied timestamp
            _raw(t0, 15, 41.151, -8.651),
        ]
        out = drop_kinematic_spikes(pings)
        # No infeasible-speed transitions detectable → nothing dropped.
        assert len(out) == 3
