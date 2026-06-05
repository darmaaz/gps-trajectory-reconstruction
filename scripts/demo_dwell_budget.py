"""Visualise the confirmed-dwell vs transit-budget split per transition.

For each canonical Porto trip (SHORT / MEDIUM / LONG), reconstruct end-to-end
and render a stacked-bar chart of:
    confirmed_dwell  (lower bar, time the vehicle was confirmed stationary)
    time_budget      (upper bar, what's left for path enumeration)

Bar height = full t_first[k+1] - t_first[k] gap, partitioned by the
preprocessing's confirmed-dwell rule. Stale-flagged observations get
`confirmed_dwell = 0` (handled by the budget code's stale branch — visible
as a transition where t_last > t_first but the bar shows zero dwell).

Output: cache/demo_dwell_budget.png
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("GPS_RECON_BBOX_LAT", "40.5,42.5")
os.environ.setdefault("GPS_RECON_BBOX_LON", "-9.5,-7.0")

from scripts._data_paths import osm_pbf_path, porto_csv_path    # noqa: E402

import matplotlib.pyplot as plt    # noqa: E402

from src.config import Config    # noqa: E402
from src.feeds import iter_porto_trips    # noqa: E402
from src.data import default_mu    # noqa: E402
from src.model import (    # noqa: E402
    ExponentialFamilyTransition, FEATURE_DIM, StudentTEmission,
)
from src.network import load_osm_network    # noqa: E402
from src.preprocessing import (    # noqa: E402
    clean, collapse_by_uniqueness, drop_kinematic_spikes,
    drop_replay_bursts, flag_stale_runs,
)
from src.api.pipeline import reconstruct_trajectory    # noqa: E402

PBF = osm_pbf_path()
CSV = porto_csv_path()
OSM_CACHE = Path(__file__).resolve().parents[1] / "cache" / "pt_edges.parquet"
OUT = Path(__file__).resolve().parents[1] / "cache" / "demo_dwell_budget.png"

TRIPS = [
    ("SHORT", "1372637091620000337", 20),
    ("MEDIUM", "1372636951620000320", 30),
    ("LONG", "1372639536620000570", 50),
]


def _log(m: str) -> None:
    print(f"[demo] {m}", file=sys.stderr, flush=True)


def _load_trip(trip_id: str, min_pings: int):
    for tid, raw in iter_porto_trips(CSV, min_pings=min_pings):
        if tid == trip_id:
            return raw
    raise SystemExit(f"trip {trip_id} not found")


def _per_transition_breakdown(raw, network, config):
    """Mirror the orchestrator's preprocessing to recover the same collapsed
    sequence the budget code sees, then compute the gap, confirmed_dwell,
    and transit time_budget per transition."""
    cleaned = clean(raw)
    cleaned = drop_kinematic_spikes(
        cleaned,
        spike_speed_ms=config.spike_speed_ms,
        bridge_speed_ms=config.spike_bridge_speed_ms,
        max_spike_length=config.spike_max_length,
    )
    collapsed = collapse_by_uniqueness(cleaned, config.collapse_epsilon)
    if collapsed:
        lats = [o.lat for o in collapsed]
        lons = [o.lon for o in collapsed]
        net = network.subgraph_for_bbox(
            min(lats), max(lats), min(lons), max(lons),
            buffer_m=config.subgraph_buffer_m,
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
    collapsed = [o for o in collapsed if not o.dropped_during_replay]

    rows = []
    for k in range(len(collapsed) - 1):
        gap = (collapsed[k + 1].t_first - collapsed[k].t_first).total_seconds()
        run_len = (collapsed[k].t_last - collapsed[k].t_first).total_seconds()
        confirmed_dwell = 0.0 if collapsed[k].stale_flagged else run_len
        v2_budget = gap - confirmed_dwell
        rows.append({
            "k": k,
            "gap": gap,
            "v1_budget": gap,
            "v2_budget": v2_budget,
            "confirmed_dwell": confirmed_dwell,
            "run_len": run_len,
            "stale": collapsed[k].stale_flagged,
        })
    return collapsed, rows


def main() -> int:
    _log(f"loading network from {PBF.name}")
    network = load_osm_network(PBF, cache_path=OSM_CACHE)
    _log(f"  {len(network)} edges, {len(network.node_positions)} nodes")

    emission = StudentTEmission(scale=10.0, network=network)
    transition = ExponentialFamilyTransition(default_mu())
    config = Config(emission=emission, transition=transition)

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=False)
    summary_lines = []

    for ax, (label, trip_id, min_pings) in zip(axes, TRIPS):
        _log(f"loading trip {label} ({trip_id})")
        raw = _load_trip(trip_id, min_pings)
        collapsed, rows = _per_transition_breakdown(raw, network, config)
        segments = reconstruct_trajectory(raw, network, config)

        ks = [r["k"] for r in rows]
        dwells = [r["confirmed_dwell"] for r in rows]
        budgets = [r["v2_budget"] for r in rows]
        stales = [r["stale"] for r in rows]

        # Stacked: confirmed_dwell (bottom) + v2_budget (top) = original gap
        ax.bar(ks, dwells, color="#d62728", label="confirmed_dwell (C1)")
        ax.bar(ks, budgets, bottom=dwells, color="#2ca02c",
               label="time_budget (transit)")

        # Mark stale transitions with a hatch pattern on top
        for k, s in zip(ks, stales):
            if s:
                ax.bar(k, rows[k]["gap"], color="none", edgecolor="black",
                       hatch="///", linewidth=0.5)

        n_dwelled = sum(1 for d in dwells if d > 0)
        total_dwell = sum(dwells)
        total_gap = sum(r["gap"] for r in rows)
        n_stale = sum(stales)

        n_segs = len(segments)
        n_paths_total = sum(
            len(seg.path_marginals) and sum(
                1 for pm in seg.path_marginals for _ in pm
            ) for seg in segments
        )

        ax.set_title(
            f"{label}  ·  trip {trip_id}  ·  {len(raw)} raw / "
            f"{len(collapsed)} collapsed / {len(rows)} transitions  ·  "
            f"{n_segs} sub-segs reconstructed",
            fontsize=11,
        )
        ax.set_xlabel("transition index k")
        ax.set_ylabel("seconds")
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(axis="y", alpha=0.3)

        summary_lines.append(
            f"{label:6s} | gap_total={total_gap:6.1f}s "
            f"dwelled={n_dwelled:3d}/{len(rows):3d} "
            f"dwell_total={total_dwell:6.1f}s "
            f"stale_obs={n_stale:2d}"
        )

    fig.suptitle(
        "Per-transition dwell-vs-transit budget split on canonical Porto trips\n"
        "red = confirmed_dwell taken out of transit budget; "
        "green = time_budget for routing; "
        "hatched = stale-flagged (dwell forced to 0)",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT, dpi=110, bbox_inches="tight")
    _log(f"wrote {OUT}")

    print()
    print("Per-trip summary:")
    print("-" * 80)
    for line in summary_lines:
        print("  " + line)
    print(f"\nFigure: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
