"""Reconstruct the same Porto trip at 15s and 120s sampling, overlaid.

Smoke-test view of how observation density shapes the path posterior.
The 15s case is the native Porto cadence. The 120s case keeps every Nth
ping (N = 120 / 15 = 8 by default), simulating a coarser fleet feed
where the model has to bridge much longer A→B time budgets.

Outputs:
    cache/compare_sampling.html — folium map with four toggleable layers:
        obs@15s, obs@120s, path@15s, path@120s
    cache/compare_sampling.png  — static figure with everything overlaid
        (no toggles)

Per-sampling colours: blue = 15s baseline, orange = 120s comparison.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import folium
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from shapely.geometry import box

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Loosen the geo bbox before src.config is evaluated.
os.environ.setdefault("GPS_RECON_BBOX_LAT", "40.5,42.5")
os.environ.setdefault("GPS_RECON_BBOX_LON", "-9.5,-7.0")

from scripts._data_paths import osm_pbf_path, porto_csv_path    # noqa: E402

from scripts.viz_helpers import compute_segments    # noqa: E402
from src.config import Config    # noqa: E402
from src.feeds import iter_porto_trips    # noqa: E402
from src.data import default_mu    # noqa: E402
from src.model import (    # noqa: E402
    ExponentialFamilyTransition, FEATURE_DIM, RawObservation, StudentTEmission,
)
from src.network import load_osm_network    # noqa: E402

CACHE_DIR = Path(__file__).resolve().parents[1] / "cache"


def _out_paths(trip_id: str, sampling_s: int) -> tuple[Path, Path]:
    base = CACHE_DIR / f"compare_sampling_{trip_id}_{sampling_s}s"
    return base.with_suffix(".png"), base.with_suffix(".html")
DEFAULT_PBF = osm_pbf_path()
DEFAULT_CSV = porto_csv_path()
OSM_CACHE = Path(__file__).resolve().parents[1] / "cache" / "pt_edges.parquet"

DEFAULT_TRIP = "1372636951620000320"
NATIVE_SAMPLING_S = 15

COLOR_15S = "#1f77b4"      # blue
COLOR_120S = "#ff7f0e"     # orange


def _log(msg: str) -> None:
    print(f"[cmp] {msg}", file=sys.stderr, flush=True)


def _pick_trip(
    csv_path: Path, trip_id: str, min_pings: int,
) -> list[RawObservation]:
    for tid, obs in iter_porto_trips(csv_path, min_pings=min_pings):
        if tid == trip_id:
            return obs
    raise SystemExit(
        f"trip {trip_id} not found (min_pings={min_pings})",
    )


def _downsample(
    pings: list[RawObservation], stride: int,
) -> list[RawObservation]:
    """Keep every `stride`-th ping; preserve the trailing ping so the
    reconstructed path covers the full trip even if stride doesn't divide
    evenly."""
    if stride <= 1:
        return list(pings)
    kept = pings[::stride]
    if pings and pings[-1] is not kept[-1]:
        kept.append(pings[-1])
    return kept


def _layer_for_run(
    *,
    label: str,
    colour: str,
    network: Any,
    collapsed: list[Any],
    seg_results: list[dict[str, Any]],
    marker_radius: int,
    polyline_weight: int,
    obs_show: bool,
    path_show: bool,
) -> tuple[folium.FeatureGroup, folium.FeatureGroup]:
    """Build one (obs_layer, path_layer) pair for a single sampling run."""
    obs_layer = folium.FeatureGroup(name=f"obs@{label}", show=obs_show)
    path_layer = folium.FeatureGroup(name=f"path@{label}", show=path_show)

    for k, c in enumerate(collapsed):
        folium.CircleMarker(
            location=[c.lat, c.lon],
            radius=marker_radius, color=colour, fill=True,
            fill_opacity=0.85, weight=1,
            popup=(
                f"<b>{label} obs {k}</b><br>"
                f"{c.t_first.strftime('%H:%M:%S')}<br>"
                f"({c.lat:.5f}, {c.lon:.5f})"
            ),
        ).add_to(obs_layer)

    for r in seg_results:
        if not r["ok"] or not r["edges"]:
            continue
        for eid in r["edges"]:
            try:
                idx = network.edge_index_for_link(int(eid))
            except KeyError:
                continue
            coords = [(lat, lon) for lon, lat in network.geoms[idx].coords]
            folium.PolyLine(
                locations=coords, color=colour,
                weight=polyline_weight, opacity=0.85,
            ).add_to(path_layer)

    return obs_layer, path_layer


def _render_html(
    *,
    network: Any,
    collapsed_15: list[Any], seg_results_15: list[dict[str, Any]],
    collapsed_120: list[Any], seg_results_120: list[dict[str, Any]],
    out_path: Path,
) -> None:
    all_obs = collapsed_15 + collapsed_120
    centre = [
        float(np.mean([c.lat for c in all_obs])),
        float(np.mean([c.lon for c in all_obs])),
    ]
    fmap = folium.Map(location=centre, zoom_start=14, tiles="OpenStreetMap")

    obs15, path15 = _layer_for_run(
        label="15s", colour=COLOR_15S, network=network,
        collapsed=collapsed_15, seg_results=seg_results_15,
        marker_radius=3, polyline_weight=4,
        obs_show=True, path_show=True,
    )
    obs120, path120 = _layer_for_run(
        label="120s", colour=COLOR_120S, network=network,
        collapsed=collapsed_120, seg_results=seg_results_120,
        marker_radius=6, polyline_weight=4,
        obs_show=True, path_show=True,
    )

    # Order: paths first (drawn underneath), then obs on top.
    fmap.add_child(path15)
    fmap.add_child(path120)
    fmap.add_child(obs15)
    fmap.add_child(obs120)

    folium.LayerControl(collapsed=False).add_to(fmap)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fmap.save(str(out_path))
    _log(f"wrote {out_path}")


def _render_png(
    *,
    network: Any,
    collapsed_15: list[Any], seg_results_15: list[dict[str, Any]],
    collapsed_120: list[Any], seg_results_120: list[dict[str, Any]],
    title: str, out_path: Path,
) -> None:
    all_lats = np.array([c.lat for c in collapsed_15 + collapsed_120])
    all_lons = np.array([c.lon for c in collapsed_15 + collapsed_120])
    pad = 0.005
    bbox = (
        all_lons.min() - pad, all_lats.min() - pad,
        all_lons.max() + pad, all_lats.max() + pad,
    )
    bbox_poly = box(*bbox)
    edge_idxs = np.asarray(network.tree.query(bbox_poly))

    fig, ax = plt.subplots(figsize=(14, 12), dpi=110)

    bg_segments = []
    for i in edge_idxs:
        geom = network.geoms[i]
        if not geom.intersects(bbox_poly):
            continue
        bg_segments.append(list(geom.coords))
    if bg_segments:
        ax.add_collection(LineCollection(
            bg_segments, colors="lightgray", linewidths=0.4, zorder=1,
        ))

    # Draw 120s first as a wide halo, then 15s thinner on top — when
    # both paths share the same edges (the common case) the 15s blue
    # rides on top with the 120s orange visible as a wider underlay.
    for results, colour, lw in (
        (seg_results_120, COLOR_120S, 5.0),
        (seg_results_15, COLOR_15S, 2.0),
    ):
        ml_segments = []
        for r in results:
            if not r["ok"] or not r["edges"]:
                continue
            for eid in r["edges"]:
                try:
                    idx = network.edge_index_for_link(int(eid))
                except KeyError:
                    continue
                ml_segments.append(list(network.geoms[idx].coords))
        if ml_segments:
            ax.add_collection(LineCollection(
                ml_segments, colors=[colour], linewidths=lw,
                zorder=2, alpha=0.9,
            ))

    obs15_lats = [c.lat for c in collapsed_15]
    obs15_lons = [c.lon for c in collapsed_15]
    obs120_lats = [c.lat for c in collapsed_120]
    obs120_lons = [c.lon for c in collapsed_120]

    # 120s drawn first (zorder=4) under 15s (zorder=5), so 15s small dots
    # are always visible even if a 120s ping landed on the same spot.
    ax.scatter(obs120_lons, obs120_lats, c=COLOR_120S, s=80, zorder=4,
               edgecolors="black", linewidths=0.6, label="obs@120s")
    ax.scatter(obs15_lons, obs15_lats, c=COLOR_15S, s=12, zorder=5,
               edgecolors="black", linewidths=0.3, label="obs@15s")

    ax.set_aspect("equal")
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_title(title)
    ax.legend(loc="upper right")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    _log(f"wrote {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pbf", type=Path, default=DEFAULT_PBF)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--trip-id", type=str, default=DEFAULT_TRIP)
    parser.add_argument("--sampling-s", type=int, default=120,
                        help="Comparison sampling rate in seconds. Native is "
                             f"{NATIVE_SAMPLING_S}s.")
    parser.add_argument("--min-pings", type=int, default=30)
    parser.add_argument("--scale", type=float, default=10.0)
    args = parser.parse_args()

    if args.sampling_s % NATIVE_SAMPLING_S != 0:
        raise SystemExit(
            f"--sampling-s ({args.sampling_s}) must be a multiple of native "
            f"{NATIVE_SAMPLING_S}s — Porto pings are at exactly that cadence.",
        )
    stride = args.sampling_s // NATIVE_SAMPLING_S

    _log(f"loading network from {args.pbf.name}"
         + (f" (cache: {OSM_CACHE.name})" if OSM_CACHE.exists() else ""))
    network = load_osm_network(args.pbf, cache_path=OSM_CACHE)
    _log(f"  {len(network)} edges")

    _log(f"picking trip {args.trip_id}…")
    raw_15 = _pick_trip(args.csv, args.trip_id, args.min_pings)
    raw_120 = _downsample(raw_15, stride)
    _log(f"  15s pings: {len(raw_15)}; {args.sampling_s}s pings: {len(raw_120)} "
         f"(stride={stride})")

    emission = StudentTEmission(scale=args.scale, network=network)
    config = Config(
        emission=emission,
        transition=ExponentialFamilyTransition(default_mu()),
    )

    _log("running 15s pipeline…")
    coll_15, _, seg_15 = compute_segments(raw_15, network, config)
    n_ok_15 = sum(1 for r in seg_15 if r["ok"])
    n_edges_15 = sum(len(r["edges"]) for r in seg_15 if r["ok"])

    _log(f"running {args.sampling_s}s pipeline…")
    coll_120, _, seg_120 = compute_segments(raw_120, network, config)
    n_ok_120 = sum(1 for r in seg_120 if r["ok"])
    n_edges_120 = sum(len(r["edges"]) for r in seg_120 if r["ok"])

    title = (
        f"Porto trip {args.trip_id}\n"
        f"15s: {len(coll_15)} obs, {len(seg_15)} segments "
        f"({n_ok_15} reconstructed, {n_edges_15} edges) · "
        f"{args.sampling_s}s: {len(coll_120)} obs, {len(seg_120)} segments "
        f"({n_ok_120} reconstructed, {n_edges_120} edges)"
    )

    png_path, html_path = _out_paths(args.trip_id, args.sampling_s)
    _render_html(
        network=network,
        collapsed_15=coll_15, seg_results_15=seg_15,
        collapsed_120=coll_120, seg_results_120=seg_120,
        out_path=html_path,
    )
    _render_png(
        network=network,
        collapsed_15=coll_15, seg_results_15=seg_15,
        collapsed_120=coll_120, seg_results_120=seg_120,
        title=title, out_path=png_path,
    )

    print()
    print(f"PNG : {png_path}")
    print(f"HTML: {html_path}")
    print()
    print(f"  15s : {len(coll_15):>3} obs · {len(seg_15)} segs "
          f"({n_ok_15} ok) · {n_edges_15} edges in path")
    print(f" {args.sampling_s}s : {len(coll_120):>3} obs · {len(seg_120)} segs "
          f"({n_ok_120} ok) · {n_edges_120} edges in path")
    return 0


if __name__ == "__main__":
    sys.exit(main())
