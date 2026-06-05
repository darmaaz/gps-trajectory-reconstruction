"""15s vs 120s candidate-path overlap on the LONG Porto trip.

Renders a map figure showing:
    - Background: road network in the bbox
    - 15s reconstruction's most-likely path (the "ground-truth-ish" route)
    - 120s reconstruction's most-likely path (what coarse sampling picks)
    - 120s candidate alternatives (the rest of the candidate set, weighted
      by posterior — shows what 120s "considered")

Plus quantitative coverage stats: what fraction of the 15s MLE edges are
contained in the 120s candidate union? What about just the 120s MLE?

Output: cache/demo_path_overlap.png
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

from src.api.pipeline import reconstruct_trajectory    # noqa: E402
from src.config import Config    # noqa: E402
from src.feeds import iter_porto_trips    # noqa: E402
from src.data import default_mu    # noqa: E402
from src.model import (    # noqa: E402
    ExponentialFamilyTransition, FEATURE_DIM, Path as ModelPath,
    StudentTEmission,
)
from src.network import load_osm_network    # noqa: E402

PBF = osm_pbf_path()
CSV = porto_csv_path()
OSM_CACHE = Path(__file__).resolve().parents[1] / "cache" / "pt_edges.parquet"
OUT = Path(__file__).resolve().parents[1] / "cache" / "demo_path_overlap.png"

LONG_TRIP = "1372639536620000570"
DOWNSAMPLE_S = 120


def _log(m: str) -> None:
    print(f"[overlap] {m}", file=sys.stderr, flush=True)


def _load_trip(trip_id: str, min_pings: int):
    for tid, raw in iter_porto_trips(CSV, min_pings=min_pings):
        if tid == trip_id:
            return raw
    raise SystemExit(f"trip {trip_id} not found")


def _downsample(raw, target_dt_s: int):
    if not raw:
        return raw
    out = [raw[0]]
    last_t = raw[0].timestamp
    for o in raw[1:]:
        if (o.timestamp - last_t).total_seconds() >= target_dt_s:
            out.append(o)
            last_t = o.timestamp
    return out


def _path_edge_ids(p: ModelPath) -> tuple[int, ...]:
    return p.edges


def _mle_edges_per_segment(segments) -> list[set[int]]:
    """Per segment, the union of all edges in the MLE interleaved sequence."""
    out: list[set[int]] = []
    for seg in segments:
        edges = set()
        for item in seg.most_likely:
            if isinstance(item, ModelPath):
                edges.update(item.edges)
        out.append(edges)
    return out


def _all_candidate_edges_per_transition(segments) -> list[set[int]]:
    """Per transition (across all segs), union of all candidate paths' edges."""
    out: list[set[int]] = []
    for seg in segments:
        for pm in seg.path_marginals:
            edges = set()
            for p in pm:
                edges.update(p.edges)
            out.append(edges)
    return out


def _plot_path_on_ax(ax, network, edges_iter, color, alpha, lw, label=None):
    """Plot a sequence of edges as a linestring on the map."""
    plotted_label = False
    for eid in edges_iter:
        try:
            idx = network.edge_index_for_link(int(eid))
        except KeyError:
            continue
        geom = network.geoms[idx]
        xs, ys = geom.xy
        ax.plot(xs, ys, color=color, alpha=alpha, linewidth=lw,
                label=label if not plotted_label else None,
                solid_capstyle="round")
        plotted_label = True


def main() -> int:
    _log(f"loading network from {PBF.name}")
    network = load_osm_network(PBF, cache_path=OSM_CACHE)
    _log(f"  {len(network)} edges, {len(network.node_positions)} nodes")

    emission = StudentTEmission(scale=10.0, network=network)
    transition = ExponentialFamilyTransition(default_mu())
    config = Config(emission=emission, transition=transition)

    _log(f"loading LONG trip {LONG_TRIP}")
    raw = _load_trip(LONG_TRIP, 50)
    raw_120s = _downsample(raw, DOWNSAMPLE_S)
    _log(f"  15s: {len(raw)} pings, 120s: {len(raw_120s)} pings")

    _log("reconstructing native 15s…")
    segs_15s = reconstruct_trajectory(raw, network, config)
    _log("reconstructing 120s downsampled…")
    segs_120s = reconstruct_trajectory(raw_120s, network, config)

    # Edge sets
    mle_edges_15s_per = _mle_edges_per_segment(segs_15s)
    mle_edges_15s = set().union(*mle_edges_15s_per) if mle_edges_15s_per else set()

    mle_edges_120s_per = _mle_edges_per_segment(segs_120s)
    mle_edges_120s = set().union(*mle_edges_120s_per) if mle_edges_120s_per else set()

    cand_edges_120s_per = _all_candidate_edges_per_transition(segs_120s)
    cand_edges_120s = set().union(*cand_edges_120s_per) if cand_edges_120s_per else set()

    # Stats
    mle_inter = mle_edges_15s & mle_edges_120s
    cand_inter = mle_edges_15s & cand_edges_120s

    _log(f"  15s MLE: {len(mle_edges_15s)} unique edges")
    _log(f"  120s MLE: {len(mle_edges_120s)} unique edges")
    _log(f"  120s candidates (union): {len(cand_edges_120s)} unique edges")
    _log(f"  15s∩120s MLE: {len(mle_inter)} edges "
         f"({len(mle_inter)/max(1,len(mle_edges_15s)):.1%} of 15s MLE)")
    _log(f"  15s MLE ⊂ 120s candidates: {len(cand_inter)} edges "
         f"({len(cand_inter)/max(1,len(mle_edges_15s)):.1%} of 15s MLE)")

    # Per-transition posterior summary at 120s
    print()
    print("Per-transition posterior at 120s sampling:")
    print(f"{'k':>3} {'n_paths':>8} {'top_w':>8} {'2nd_w':>8} "
          f"{'entropy':>8} {'dwell_top':>10}")
    print("-" * 60)
    for seg in segs_120s:
        for k, pm in enumerate(seg.path_marginals):
            if not pm:
                print(f"{k:>3} {0:>8d}")
                continue
            weights = sorted(pm.values(), reverse=True)
            top_w = weights[0]
            second = weights[1] if len(weights) > 1 else 0.0
            ent = -sum(w * np.log(max(w, 1e-12)) for w in weights)
            mle_path = max(pm.items(), key=lambda kv: kv[1])[0]
            dwell_top = mle_path.inferred_dwell
            print(f"{k:>3} {len(pm):>8d} {top_w:>8.3f} {second:>8.3f} "
                  f"{ent:>8.3f} {dwell_top:>9.1f}s")

    # Bbox for the figure
    lats = [o.lat for o in raw]
    lons = [o.lon for o in raw]
    pad_lat = (max(lats) - min(lats)) * 0.05
    pad_lon = (max(lons) - min(lons)) * 0.05
    bbox = (min(lons) - pad_lon, max(lons) + pad_lon,
            min(lats) - pad_lat, max(lats) + pad_lat)

    # Render
    fig, ax = plt.subplots(figsize=(14, 12))

    # Background road network within bbox (light gray)
    _log("rendering background network…")
    bg_subnet = network.subgraph_for_bbox(
        bbox[2], bbox[3], bbox[0], bbox[1], buffer_m=200,
    )
    n_bg = 0
    for idx in range(len(bg_subnet)):
        geom = bg_subnet.geoms[idx]
        xs, ys = geom.xy
        ax.plot(xs, ys, color="#dddddd", linewidth=0.4, zorder=0)
        n_bg += 1
    _log(f"  {n_bg} edges in bbox")

    # 120s candidate alternatives weighted by posterior probability.
    # Each path's edges get alpha proportional to its posterior weight, so
    # higher-confidence alternatives appear bolder and low-confidence
    # candidates fade. The MLE path is overlaid separately on top.
    _log("rendering 120s candidate paths (alpha ∝ posterior weight)…")
    n_paths_plotted = 0
    max_weight_non_mle = 0.0
    for seg in segs_120s:
        for pm in seg.path_marginals:
            if not pm:
                continue
            mle_path = max(pm.items(), key=lambda kv: kv[1])[0]
            for path, weight in pm.items():
                if path is mle_path:
                    continue
                # Scale alpha so the weightiest non-MLE path reaches ~0.55,
                # weakest reaches ~0.05 — keeps the visual readable.
                alpha = 0.05 + 0.5 * float(weight)
                _plot_path_on_ax(
                    ax, network, path.edges,
                    color="#ff7f0e", alpha=alpha, lw=1.4,
                )
                n_paths_plotted += 1
                max_weight_non_mle = max(max_weight_non_mle, float(weight))
    _log(f"  plotted {n_paths_plotted} non-MLE candidate paths "
         f"(max non-MLE posterior weight: {max_weight_non_mle:.3f})")

    # 120s MLE path (solid orange)
    _log("rendering 120s MLE path…")
    _plot_path_on_ax(
        ax, network, mle_edges_120s,
        color="#ff7f0e", alpha=0.85, lw=2.5,
        label=f"120s MLE path ({len(mle_edges_120s)} edges)",
    )

    # 15s MLE path (solid blue, on top)
    _log("rendering 15s MLE path…")
    _plot_path_on_ax(
        ax, network, mle_edges_15s,
        color="#1f77b4", alpha=0.7, lw=1.6,
        label=f"15s MLE path ({len(mle_edges_15s)} edges)",
    )

    # Observation pings
    ax.scatter(lons, lats, c="black", s=8, zorder=4, alpha=0.6,
               label=f"raw pings ({len(raw)})")

    ax.set_xlim(bbox[0], bbox[1])
    ax.set_ylim(bbox[2], bbox[3])
    ax.set_aspect("equal")
    ax.set_title(
        f"Candidate-path overlap, 15s vs 120s — LONG trip {LONG_TRIP}\n"
        f"blue = 15s reconstruction's MLE path (richest data); "
        f"orange = 120s MLE; faint orange = other 120s candidates\n"
        f"15s MLE ⊂ 120s candidate union: "
        f"{len(cand_inter)/max(1,len(mle_edges_15s)):.1%} "
        f"({len(cand_inter)}/{len(mle_edges_15s)} edges)",
        fontsize=11,
    )
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUT, dpi=110, bbox_inches="tight")
    _log(f"wrote {OUT}")

    print()
    print("Edge-overlap summary:")
    print("-" * 70)
    print(f"  15s MLE edges:                   {len(mle_edges_15s)}")
    print(f"  120s MLE edges:                  {len(mle_edges_120s)}")
    print(f"  120s candidate-union edges:      {len(cand_edges_120s)}")
    print(f"  15s ∩ 120s MLE:                  {len(mle_inter)} "
          f"({len(mle_inter)/max(1,len(mle_edges_15s)):.1%} of 15s MLE)")
    print(f"  15s ∩ 120s candidate union:      {len(cand_inter)} "
          f"({len(cand_inter)/max(1,len(mle_edges_15s)):.1%} of 15s MLE)")
    print(f"\nFigure: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
