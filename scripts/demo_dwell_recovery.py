"""15s ↔ 120s dwell-recovery validation on the LONG Porto trip.

The question: at 120s sampling, does the path candidate set *contain* a
path whose `inferred_dwell` matches the 15s confirmed dwell? The
pipeline enumerates plausible (path, dwell) scenarios weighted by the
path posterior; validation passes if the true scenario is in the set.

Procedure:
    1. Native 15s reconstruction → per-transition `confirmed_dwell[k]`
       (ground-truth-ish; comes from collapse-by-uniqueness).
    2. Downsample to 120s, reconstruct → per-transition candidate set
       with `(inferred_dwell, posterior_weight)` per path.
    3. For each 120s window, sum the 15s `confirmed_dwell` values whose
       source observation falls inside the window.
    4. Plot: per-bucket scatter of all candidate `inferred_dwell` values
       (point size ∝ posterior weight), with the truth marked on each
       bucket. Validation passes when the truth falls inside the
       candidate range and at least one candidate has both
       `|inferred_dwell - truth|` small AND non-zero posterior weight.

Output: cache/demo_dwell_recovery.png
"""

from __future__ import annotations

import os
import sys
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("GPS_RECON_BBOX_LAT", "40.5,42.5")
os.environ.setdefault("GPS_RECON_BBOX_LON", "-9.5,-7.0")

from scripts._data_paths import osm_pbf_path, porto_csv_path    # noqa: E402

import matplotlib.pyplot as plt    # noqa: E402

from src.api.pipeline import reconstruct_trajectory    # noqa: E402
from src.config import Config    # noqa: E402
from src.feeds import iter_porto_trips    # noqa: E402
from src.data import default_mu    # noqa: E402
from src.model import (    # noqa: E402
    ExponentialFamilyTransition, FEATURE_DIM, StudentTEmission,
)
from src.network import load_osm_network    # noqa: E402
from src.preprocessing import (    # noqa: E402
    clean, collapse_by_uniqueness, drop_kinematic_spikes, flag_stale_runs,
)

PBF = osm_pbf_path()
CSV = porto_csv_path()
OSM_CACHE = Path(__file__).resolve().parents[1] / "cache" / "pt_edges.parquet"
OUT = Path(__file__).resolve().parents[1] / "cache" / "demo_dwell_recovery.png"

LONG_TRIP = "1372639536620000570"
DOWNSAMPLE_S = 120


def _log(m: str) -> None:
    print(f"[dwell-recovery] {m}", file=sys.stderr, flush=True)


def _load_trip(trip_id: str, min_pings: int):
    for tid, raw in iter_porto_trips(CSV, min_pings=min_pings):
        if tid == trip_id:
            return raw
    raise SystemExit(f"trip {trip_id} not found")


def _downsample(raw, target_dt_s: int):
    """Keep one ping every `target_dt_s` seconds (by timestamp)."""
    if not raw:
        return raw
    out = [raw[0]]
    last_t = raw[0].timestamp
    for o in raw[1:]:
        if (o.timestamp - last_t).total_seconds() >= target_dt_s:
            out.append(o)
            last_t = o.timestamp
    return out


def _native_15s_dwells(raw, network, config):
    """Run reconstruction at native 15s, return list of (t_first[k],
    t_first[k+1], confirmed_dwell[k]) per transition across all segments."""
    segments = reconstruct_trajectory(raw, network, config)
    out = []
    for seg in segments:
        ts = seg.canonical_timestamps
        dwells = seg.confirmed_dwell
        if len(ts) < 2 or not dwells:
            continue
        for k in range(len(ts) - 1):
            out.append((ts[k], ts[k + 1], dwells[k]))
    return out


def _candidate_dwells_120s(raw_120s, network, config):
    """Run reconstruction at 120s, return list of (t_first[k], t_first[k+1],
    [(inferred_dwell, normalised_weight), ...]) per transition."""
    segments = reconstruct_trajectory(raw_120s, network, config)
    out = []
    for seg in segments:
        ts = seg.canonical_timestamps
        for k in range(len(ts) - 1):
            pm = seg.path_marginals[k]
            total_w = sum(pm.values()) if pm else 0.0
            if total_w <= 0:
                out.append((ts[k], ts[k + 1], []))
                continue
            cands = [
                (p.inferred_dwell, w / total_w) for p, w in pm.items()
            ]
            out.append((ts[k], ts[k + 1], cands))
    return out


def _aggregate_15s_into_120s(dwells_15s, windows_120s):
    """Sum 15s confirmed_dwell values whose [t_first[k], t_first[k+1])
    interval overlaps each 120s window."""
    out = []
    for w_lo, w_hi, _ in windows_120s:
        total = 0.0
        for t_lo, t_hi, d in dwells_15s:
            if t_lo < w_hi and t_hi > w_lo:    # overlap
                total += d
        out.append(total)
    return out


def main() -> int:
    _log(f"loading network from {PBF.name}")
    network = load_osm_network(PBF, cache_path=OSM_CACHE)
    _log(f"  {len(network)} edges, {len(network.node_positions)} nodes")

    emission = StudentTEmission(scale=10.0, network=network)
    transition = ExponentialFamilyTransition(default_mu())
    config = Config(emission=emission, transition=transition)

    _log(f"loading LONG trip {LONG_TRIP}")
    raw = _load_trip(LONG_TRIP, 50)
    _log(f"  {len(raw)} pings, {raw[0].timestamp} → {raw[-1].timestamp}")

    raw_120s = _downsample(raw, DOWNSAMPLE_S)
    _log(f"  downsampled to {DOWNSAMPLE_S}s: {len(raw_120s)} pings")

    _log("reconstructing native 15s…")
    dwells_15s = _native_15s_dwells(raw, network, config)
    _log(f"  {len(dwells_15s)} 15s transitions, "
         f"non-zero dwells: {sum(1 for _,_,d in dwells_15s if d > 0)}")

    _log("reconstructing 120s downsampled…")
    cands_120s = _candidate_dwells_120s(raw_120s, network, config)
    _log(f"  {len(cands_120s)} 120s transitions, "
         f"avg candidates/transition: "
         f"{sum(len(c) for _,_,c in cands_120s) / max(1, len(cands_120s)):.1f}")

    truth_120s = _aggregate_15s_into_120s(dwells_15s, cands_120s)
    closeness_tol_s = 30.0    # within 30s = "consistent with truth"

    # Plot: per-bucket scatter of candidate inferred_dwells, sized by weight,
    # with truth marker. Validation = is truth inside the cloud?
    fig, ax = plt.subplots(figsize=(14, 7))
    in_range_count = 0
    consistent_count = 0
    for k, (_, _, cands) in enumerate(cands_120s):
        if not cands:
            continue
        dwells = [d for d, _ in cands]
        weights = [w for _, w in cands]
        # Scatter sized by posterior weight (min size 20 for visibility)
        sizes = [20 + 280 * w for w in weights]
        ax.scatter([k] * len(cands), dwells, s=sizes, c="#ff7f0e", alpha=0.55,
                   edgecolors="none",
                   label="candidate inferred_dwell (size ∝ posterior weight)"
                         if k == 0 else None)
        # Range bracket
        ax.plot([k, k], [min(dwells), max(dwells)], color="#ff7f0e",
                alpha=0.3, linewidth=1, zorder=0)
        # Truth marker
        truth = truth_120s[k]
        ax.scatter([k], [truth], marker="D", s=80, color="#1f77b4",
                   edgecolors="black", linewidth=0.7, zorder=5,
                   label="15s ground-truth dwell" if k == 0 else None)
        # Validation tally
        if min(dwells) - 0.5 <= truth <= max(dwells) + 0.5:
            in_range_count += 1
        if any(abs(d - truth) <= closeness_tol_s and w > 0.01
               for d, w in cands):
            consistent_count += 1

    n_buckets = len(cands_120s)
    ax.set_xlabel("120s transition index k")
    ax.set_ylabel("inferred_dwell (seconds)")
    ax.set_title(
        f"Dwell-recovery validation — LONG trip {LONG_TRIP}\n"
        f"orange = candidate paths' inferred_dwell at 120s "
        f"(size ∝ posterior weight); blue diamonds = 15s ground-truth dwell\n"
        f"validation: truth in candidate range = {in_range_count}/{n_buckets} buckets, "
        f"truth within ±{int(closeness_tol_s)}s of a posterior-weighted candidate "
        f"= {consistent_count}/{n_buckets} buckets",
        fontsize=11,
    )
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT, dpi=110, bbox_inches="tight")
    _log(f"wrote {OUT}")

    print()
    print("Per-bucket validation:")
    print("-" * 90)
    print(f"  {'k':>3} | {'truth':>6} | {'cands':>5} | {'min_dwell':>9} | "
          f"{'max_dwell':>9} | {'closest':>7} | {'in_range':>8} | {'consistent':>10}")
    print("-" * 90)
    for k, (_, _, cands) in enumerate(cands_120s):
        truth = truth_120s[k]
        if not cands:
            print(f"  {k:>3} | {truth:>6.1f} | (no candidates)")
            continue
        dwells = [d for d, _ in cands]
        d_min, d_max = min(dwells), max(dwells)
        closest = min(dwells, key=lambda d: abs(d - truth))
        in_range = d_min - 0.5 <= truth <= d_max + 0.5
        consistent = any(abs(d - truth) <= closeness_tol_s and w > 0.01
                         for d, w in cands)
        print(f"  {k:>3} | {truth:>6.1f} | {len(cands):>5d} | "
              f"{d_min:>9.1f} | {d_max:>9.1f} | {closest:>7.1f} | "
              f"{'yes' if in_range else 'NO':>8} | "
              f"{'yes' if consistent else 'NO':>10}")
    print()
    print(f"Validation summary:")
    print(f"  truth inside candidate range:                    "
          f"{in_range_count}/{n_buckets} buckets")
    print(f"  truth within ±{int(closeness_tol_s)}s of posterior-weighted candidate: "
          f"{consistent_count}/{n_buckets} buckets")
    print(f"\nFigure: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
