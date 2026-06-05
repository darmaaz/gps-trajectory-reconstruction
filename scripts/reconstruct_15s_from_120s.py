"""Reconstruct 15s positions from 120s observations on a Porto trip.

For the same trip used in `compare_sampling.py`, runs the pipeline at
120s sampling, then for each native 15s timestamp queries
`position_at_time` on the 120s segments to predict where the vehicle
was. Compares those predictions against the actual 15s pings (post
spike-filter, so we don't penalize the model for not matching known
GPS chip glitches).

Outputs:
    cache/recon_15s_from_120s.html  — folium map with toggleable layers
        for true 15s, 120s observations, predicted-at-15s, and the
        per-timestamp error vectors (red lines from prediction → truth)
    cache/recon_15s_from_120s.png   — static figure with everything
        overlaid

Stdout: per-timestamp error stats (count covered/uncovered, mean,
median, p95, max).

Caveats — this exercises the constant-speed-along-MLE-path estimator
in `src/api/interpolation.py`. Errors will concentrate at signalled
intersections and other dwell events; the dwell-aware posterior that
aggregates `inferred_dwell` across the path posterior to produce a
state marginal at arbitrary `t` is the next planned addition and is
expected to absorb most of that error.
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

# Loosen geo bbox before src.config is imported.
os.environ.setdefault("GPS_RECON_BBOX_LAT", "40.5,42.5")
os.environ.setdefault("GPS_RECON_BBOX_LON", "-9.5,-7.0")

from scripts._data_paths import osm_pbf_path, porto_csv_path    # noqa: E402

from src.api import position_at_time, reconstruct_trajectory    # noqa: E402
from src.config import Config    # noqa: E402
from src.feeds import iter_porto_trips    # noqa: E402
from src.geo import haversine_m    # noqa: E402
from src.data import default_mu    # noqa: E402
from src.model import (    # noqa: E402
    ExponentialFamilyTransition, FEATURE_DIM, RawObservation, StudentTEmission,
)
from src.network import load_osm_network    # noqa: E402
from src.preprocessing import clean, drop_kinematic_spikes    # noqa: E402

CACHE_DIR = Path(__file__).resolve().parents[1] / "cache"


def _out_paths(trip_id: str, sampling_s: int) -> tuple[Path, Path]:
    base = CACHE_DIR / f"recon_15s_from_{sampling_s}s_{trip_id}"
    return base.with_suffix(".png"), base.with_suffix(".html")
DEFAULT_PBF = osm_pbf_path()
DEFAULT_CSV = porto_csv_path()
OSM_CACHE = Path(__file__).resolve().parents[1] / "cache" / "pt_edges.parquet"
DEFAULT_TRIP = "1372636951620000320"
NATIVE_SAMPLING_S = 15

COLOR_TRUE = "#1f77b4"        # blue: true 15s
COLOR_120S = "#ff7f0e"        # orange: 120s observations
COLOR_PRED = "#2ca02c"        # green: predicted at 15s
COLOR_ERR = "#d62728"         # red: error vector


def _log(msg: str) -> None:
    print(f"[recon] {msg}", file=sys.stderr, flush=True)


def _pick_trip(
    csv_path: Path, trip_id: str, min_pings: int,
) -> list[RawObservation]:
    for tid, obs in iter_porto_trips(csv_path, min_pings=min_pings):
        if tid == trip_id:
            return obs
    raise SystemExit(f"trip {trip_id} not found")


def _downsample(
    pings: list[RawObservation], stride: int,
) -> list[RawObservation]:
    if stride <= 1:
        return list(pings)
    kept = pings[::stride]
    if pings and pings[-1] is not kept[-1]:
        kept.append(pings[-1])
    return kept


def _build_config(network) -> Config:
    emission = StudentTEmission(scale=10.0, network=network)
    return Config(
        emission=emission,
        transition=ExponentialFamilyTransition(default_mu()),
    )


def _truth_after_spike_filter(
    raw: list[RawObservation], config: Config,
) -> list[RawObservation]:
    """Same preprocessing as the orchestrator does upstream of collapse:
    clean() then drop_kinematic_spikes(). We compare against THIS rather
    than the raw input so spike pings (known garbage) don't poison the
    error stats — they'd inflate p95/max with multi-km errors that the
    model is rightly refusing to match.
    """
    cleaned = clean(raw)
    return drop_kinematic_spikes(
        cleaned,
        spike_speed_ms=config.spike_speed_ms,
        bridge_speed_ms=config.spike_bridge_speed_ms,
        max_spike_length=config.spike_max_length,
    )


def _print_stats(errors_m: list[float], n_uncovered: int) -> None:
    n = len(errors_m)
    print()
    print(f"Coverage:  {n} timestamps predicted, {n_uncovered} uncovered "
          f"(in segment-split gaps)")
    if not n:
        return
    arr = np.array(errors_m)
    print(f"Errors (m): mean={arr.mean():.1f}  median={np.median(arr):.1f}  "
          f"p95={np.percentile(arr, 95):.1f}  max={arr.max():.1f}")


def _render_html(
    *, network, truth_15: list[RawObservation], obs_120: list[RawObservation],
    predictions: list[tuple[RawObservation, tuple[float, float] | None]],
    out_path: Path,
) -> None:
    all_lats = [o.lat for o in truth_15] + [o.lat for o in obs_120]
    centre = [
        float(np.mean(all_lats)),
        float(np.mean([o.lon for o in truth_15] + [o.lon for o in obs_120])),
    ]
    fmap = folium.Map(location=centre, zoom_start=14, tiles="OpenStreetMap")

    truth_layer = folium.FeatureGroup(name="truth@15s", show=True)
    obs120_layer = folium.FeatureGroup(name="obs@120s", show=True)
    pred_layer = folium.FeatureGroup(name="predicted@15s", show=True)
    err_layer = folium.FeatureGroup(name="error vectors", show=True)

    for o in truth_15:
        folium.CircleMarker(
            location=[o.lat, o.lon], radius=3, color=COLOR_TRUE,
            fill=True, fill_opacity=0.85, weight=1,
            popup=f"truth {o.timestamp.strftime('%H:%M:%S')}",
        ).add_to(truth_layer)

    for o in obs_120:
        folium.CircleMarker(
            location=[o.lat, o.lon], radius=6, color=COLOR_120S,
            fill=True, fill_opacity=0.85, weight=1,
            popup=f"120s obs {o.timestamp.strftime('%H:%M:%S')}",
        ).add_to(obs120_layer)

    for truth, pred in predictions:
        if pred is None:
            continue
        folium.CircleMarker(
            location=[pred[0], pred[1]], radius=3, color=COLOR_PRED,
            fill=True, fill_opacity=0.85, weight=1,
            popup=(
                f"pred {truth.timestamp.strftime('%H:%M:%S')}<br>"
                f"err = {haversine_m(truth.lat, truth.lon, pred[0], pred[1]):.1f} m"
            ),
        ).add_to(pred_layer)
        folium.PolyLine(
            locations=[[truth.lat, truth.lon], [pred[0], pred[1]]],
            color=COLOR_ERR, weight=2, opacity=0.85,
        ).add_to(err_layer)

    fmap.add_child(err_layer)
    fmap.add_child(truth_layer)
    fmap.add_child(obs120_layer)
    fmap.add_child(pred_layer)
    folium.LayerControl(collapsed=False).add_to(fmap)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fmap.save(str(out_path))
    _log(f"wrote {out_path}")


def _render_png(
    *, network, truth_15, obs_120, predictions, title: str, out_path: Path,
) -> None:
    truth_lats = np.array([o.lat for o in truth_15])
    truth_lons = np.array([o.lon for o in truth_15])
    pad = 0.005
    bbox = (
        truth_lons.min() - pad, truth_lats.min() - pad,
        truth_lons.max() + pad, truth_lats.max() + pad,
    )
    bbox_poly = box(*bbox)
    edge_idxs = np.asarray(network.tree.query(bbox_poly))

    fig, ax = plt.subplots(figsize=(14, 12), dpi=110)
    bg = []
    for i in edge_idxs:
        geom = network.geoms[i]
        if not geom.intersects(bbox_poly):
            continue
        bg.append(list(geom.coords))
    if bg:
        ax.add_collection(LineCollection(
            bg, colors="lightgray", linewidths=0.4, zorder=1,
        ))

    # Error vectors first (under the points).
    err_segments = []
    for truth, pred in predictions:
        if pred is None:
            continue
        err_segments.append([(truth.lon, truth.lat), (pred[1], pred[0])])
    if err_segments:
        ax.add_collection(LineCollection(
            err_segments, colors=COLOR_ERR, linewidths=0.7,
            alpha=0.7, zorder=2,
        ))

    # Truth (small blue), 120s obs (big orange), predictions (small green).
    pred_lats = [p[1][0] for p in predictions if p[1] is not None]
    pred_lons = [p[1][1] for p in predictions if p[1] is not None]
    obs120_lats = [o.lat for o in obs_120]
    obs120_lons = [o.lon for o in obs_120]

    ax.scatter(obs120_lons, obs120_lats, c=COLOR_120S, s=80, zorder=3,
               edgecolors="black", linewidths=0.6, label="obs@120s")
    ax.scatter(truth_lons, truth_lats, c=COLOR_TRUE, s=14, zorder=4,
               edgecolors="black", linewidths=0.3, label="truth@15s")
    ax.scatter(pred_lons, pred_lats, c=COLOR_PRED, s=14, zorder=5,
               edgecolors="black", linewidths=0.3, label="predicted@15s")

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
    parser.add_argument("--sampling-s", type=int, default=120)
    parser.add_argument("--min-pings", type=int, default=30)
    args = parser.parse_args()

    if args.sampling_s % NATIVE_SAMPLING_S != 0:
        raise SystemExit(
            f"--sampling-s must be a multiple of {NATIVE_SAMPLING_S}",
        )
    stride = args.sampling_s // NATIVE_SAMPLING_S

    _log(f"loading network from {args.pbf.name}")
    network = load_osm_network(args.pbf, cache_path=OSM_CACHE)
    _log(f"  {len(network)} edges")

    raw_15 = _pick_trip(args.csv, args.trip_id, args.min_pings)
    _log(f"trip {args.trip_id}: {len(raw_15)} raw 15s pings")

    config = _build_config(network)
    truth_15 = _truth_after_spike_filter(raw_15, config)
    _log(f"  {len(truth_15)} truth pings after spike filter "
         f"({len(raw_15) - len(truth_15)} spikes removed)")

    raw_120 = _downsample(raw_15, stride)
    _log(f"  {len(raw_120)} pings at {args.sampling_s}s sampling "
         f"(stride={stride})")

    _log(f"running pipeline at {args.sampling_s}s sampling…")
    segments_120 = reconstruct_trajectory(raw_120, network, config)
    _log(f"  {len(segments_120)} segments produced")

    # For each truth timestamp, predict from the 120s reconstruction.
    predictions: list[tuple[RawObservation, tuple[float, float] | None]] = []
    errors_m: list[float] = []
    n_uncovered = 0
    for o in truth_15:
        pred = position_at_time(segments_120, o.timestamp, network, rule="front")
        predictions.append((o, pred))
        if pred is None:
            n_uncovered += 1
        else:
            errors_m.append(haversine_m(o.lat, o.lon, pred[0], pred[1]))

    _print_stats(errors_m, n_uncovered)

    title = (
        f"Porto {args.trip_id}: 15s reconstruction from {args.sampling_s}s obs\n"
        f"truth: {len(truth_15)}  ·  obs@{args.sampling_s}s: {len(raw_120)}  ·  "
        f"predicted: {len(errors_m)}  ·  uncovered: {n_uncovered}  ·  "
        f"mean err: {np.mean(errors_m):.1f} m  ·  "
        f"p95 err: {np.percentile(errors_m, 95):.1f} m" if errors_m else
        f"Porto {args.trip_id}: no covered predictions"
    )

    png_path, html_path = _out_paths(args.trip_id, args.sampling_s)
    _render_html(
        network=network, truth_15=truth_15, obs_120=raw_120,
        predictions=predictions, out_path=html_path,
    )
    _render_png(
        network=network, truth_15=truth_15, obs_120=raw_120,
        predictions=predictions, title=title, out_path=png_path,
    )

    print()
    print(f"PNG : {png_path}")
    print(f"HTML: {html_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
