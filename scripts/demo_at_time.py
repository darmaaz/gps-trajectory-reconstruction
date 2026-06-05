"""Dwell-aware posterior `at_time(t)` vs MLE constant-speed interpolation.

Same downsample-and-reconstruct setup as `reconstruct_15s_from_120s.py`,
but adds a side-by-side comparison between two position-at-time
predictors:

    (A) MLE constant-speed: existing `src/api/interpolation.position_at_time`.
        Single (lat, lon) from interpolating uniformly along the Viterbi
        most-likely path between two observations.

    (B) Dwell-aware posterior MAP: new `TrajectoryPosterior.at_time(t)`.
        Aggregates over the path posterior under a front-loaded dwell rule
        (dwell at origin for `inferred_dwell` seconds, then travel along
        the path at uniform speed). The marginal's highest-weight state is
        reported as the single-point summary; the full marginal is
        rendered as a calibrated-uncertainty cloud on the map.

For each held-out native 15 s timestamp on the LONG Porto trip,
downsampled to 120 s, predict the position both ways and compare to the
spike-filtered ground truth.

Outputs:
    cache/demo_at_time.png  — static figure: error CDF + map of a few
        example transitions with the full posterior marginal shown.
    cache/demo_at_time.html — folium map with truth, 120s observations,
        (A) and (B) predictions, and the full posterior marginal at each
        held-out timestamp (toggleable layers).

Stdout: paired error stats for (A) and (B).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import folium
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from shapely.geometry import box

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("GPS_RECON_BBOX_LAT", "40.5,42.5")
os.environ.setdefault("GPS_RECON_BBOX_LON", "-9.5,-7.0")

from scripts._data_paths import osm_pbf_path, porto_csv_path    # noqa: E402

from src.api import position_at_time, reconstruct_trajectory    # noqa: E402
from src.api.interpolation import _position_on_edge    # noqa: E402
from src.config import Config    # noqa: E402
from src.feeds import iter_porto_trips    # noqa: E402
from src.geo import haversine_m    # noqa: E402
from src.data import default_mu    # noqa: E402
from src.model import (    # noqa: E402
    ExponentialFamilyTransition, FEATURE_DIM, RawObservation, StudentTEmission,
)
from src.network import load_osm_network    # noqa: E402
from src.preprocessing import clean, drop_kinematic_spikes    # noqa: E402

CACHE = Path(__file__).resolve().parents[1] / "cache"
PBF = osm_pbf_path()
CSV = porto_csv_path()
OSM_CACHE = CACHE / "pt_edges.parquet"

LONG_TRIP = "1372639536620000570"
DOWNSAMPLE_S = 120
NATIVE_S = 15

COLOR_TRUTH = "#1f77b4"
COLOR_OBS = "#ff7f0e"
COLOR_MLE = "#2ca02c"
COLOR_POST = "#9467bd"
COLOR_ERR = "#d62728"

# How many illustrative transitions to highlight on the map.
N_HIGHLIGHT_TRANSITIONS = 3


def _log(m: str) -> None:
    print(f"[at_time] {m}", file=sys.stderr, flush=True)


def _load_trip(min_pings: int) -> list[RawObservation]:
    for tid, raw in iter_porto_trips(CSV, min_pings=min_pings):
        if tid == LONG_TRIP:
            return raw
    raise SystemExit(f"trip {LONG_TRIP} not found")


def _build_config(network) -> Config:
    emission = StudentTEmission(scale=10.0, network=network)
    return Config(
        emission=emission,
        transition=ExponentialFamilyTransition(default_mu()),
    )


def _downsample(pings: list[RawObservation], stride: int) -> list[RawObservation]:
    if stride <= 1:
        return list(pings)
    kept = pings[::stride]
    if pings and pings[-1] is not kept[-1]:
        kept.append(pings[-1])
    return kept


def _truth_after_spike_filter(
    raw: list[RawObservation], config: Config,
) -> list[RawObservation]:
    cleaned = clean(raw)
    return drop_kinematic_spikes(
        cleaned,
        spike_speed_ms=config.spike_speed_ms,
        bridge_speed_ms=config.spike_bridge_speed_ms,
        max_spike_length=config.spike_max_length,
    )


def _state_to_latlon(state, network) -> tuple[float, float]:
    idx = network.edge_index_for_link(state.link_id)
    return _position_on_edge(network, idx, state.offset)


def _posterior_map_prediction(segments, t, network):
    """MAP state of the dwell-aware at_time marginal → (lat, lon)."""
    for seg in segments:
        ts = seg.canonical_timestamps
        if not ts or t < ts[0] or t > ts[-1]:
            continue
        marg = seg.at_time(t, rule="front")
        if not marg:
            return None
        # Pick the highest-weight state. Ties broken arbitrarily.
        best_state = max(marg.items(), key=lambda kv: kv[1])[0]
        return _state_to_latlon(best_state, network)
    return None


def _posterior_full_marginal(segments, t, network):
    """Return [(lat, lon, weight)] for every state in at_time(t) marginal."""
    for seg in segments:
        ts = seg.canonical_timestamps
        if not ts or t < ts[0] or t > ts[-1]:
            continue
        marg = seg.at_time(t, rule="front")
        out = []
        for state, weight in marg.items():
            lat, lon = _state_to_latlon(state, network)
            out.append((lat, lon, float(weight)))
        return out
    return []


def _print_stats(name: str, errs: list[float]) -> None:
    if not errs:
        print(f"{name:>16s}: no covered timestamps")
        return
    arr = np.array(errs)
    print(
        f"{name:>16s}: n={len(errs):4d}  "
        f"mean={arr.mean():7.1f}m  median={np.median(arr):7.1f}m  "
        f"p95={np.percentile(arr, 95):7.1f}m  max={arr.max():7.1f}m"
    )


def _pick_highlight_indices(errs_a: list[float], errs_b: list[float], n: int) -> list[int]:
    """Pick `n` timestamps where (A) and (B) disagree most — useful for the map."""
    if not errs_a or not errs_b or len(errs_a) != len(errs_b):
        return []
    diffs = np.abs(np.array(errs_a) - np.array(errs_b))
    # Avoid clumping: pick well-separated indices from the top-k by diff.
    top = np.argsort(diffs)[::-1][:max(n * 3, n)]
    picked: list[int] = []
    for i in top:
        if all(abs(int(i) - p) > max(1, len(errs_a) // (n * 3)) for p in picked):
            picked.append(int(i))
            if len(picked) >= n:
                break
    return picked


def _render_png(
    *,
    network,
    truth: list[RawObservation],
    obs_120: list[RawObservation],
    pred_mle: list[tuple[float, float] | None],
    pred_post: list[tuple[float, float] | None],
    full_marginals: list[list[tuple[float, float, float]]],
    errs_a: list[float],
    errs_b: list[float],
    highlights: list[int],
    out_path: Path,
) -> None:
    fig = plt.figure(figsize=(16, 10), dpi=110)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.3, 1])

    # Left: error CDFs.
    ax_cdf = fig.add_subplot(gs[0, 0])
    if errs_a:
        a_sorted = np.sort(errs_a)
        y_a = np.arange(1, len(a_sorted) + 1) / len(a_sorted)
        ax_cdf.plot(
            a_sorted, y_a, color=COLOR_MLE, linewidth=2,
            label=f"(A) MLE constant-speed (n={len(errs_a)})",
        )
    if errs_b:
        b_sorted = np.sort(errs_b)
        y_b = np.arange(1, len(b_sorted) + 1) / len(b_sorted)
        ax_cdf.plot(
            b_sorted, y_b, color=COLOR_POST, linewidth=2,
            label=f"(B) Dwell-aware posterior MAP (n={len(errs_b)})",
        )
    ax_cdf.set_xlabel("position error (m)")
    ax_cdf.set_ylabel("CDF")
    ax_cdf.set_title("Per-timestamp prediction error — empirical CDF")
    ax_cdf.legend(loc="lower right")
    ax_cdf.grid(True, alpha=0.3)
    if errs_a or errs_b:
        xmax = float(np.percentile(np.array(errs_a + errs_b), 99))
        ax_cdf.set_xlim(0, max(xmax, 50))

    # Right: map of the highlight transitions.
    ax_map = fig.add_subplot(gs[0, 1])
    truth_lats = np.array([o.lat for o in truth])
    truth_lons = np.array([o.lon for o in truth])

    if highlights:
        h_lats = [truth[i].lat for i in highlights]
        h_lons = [truth[i].lon for i in highlights]
        pad = 0.003
        bbox = (
            min(h_lons) - pad, min(h_lats) - pad,
            max(h_lons) + pad, max(h_lats) + pad,
        )
    else:
        pad = 0.005
        bbox = (
            truth_lons.min() - pad, truth_lats.min() - pad,
            truth_lons.max() + pad, truth_lats.max() + pad,
        )

    bbox_poly = box(*bbox)
    bg_idxs = np.asarray(network.tree.query(bbox_poly))
    bg = []
    for i in bg_idxs:
        geom = network.geoms[i]
        if not geom.intersects(bbox_poly):
            continue
        bg.append(list(geom.coords))
    if bg:
        ax_map.add_collection(LineCollection(
            bg, colors="lightgray", linewidths=0.4, zorder=1,
        ))

    # All truth pings as faint background.
    ax_map.scatter(
        truth_lons, truth_lats, c=COLOR_TRUTH, s=8, alpha=0.25, zorder=2,
    )

    for i in highlights:
        t_o = truth[i]
        marg = full_marginals[i]
        for lat, lon, w in marg:
            ax_map.scatter(
                [lon], [lat],
                c=COLOR_POST, s=20 + 200 * w,
                alpha=0.4, zorder=4,
                edgecolors="none",
            )
        if pred_mle[i] is not None:
            ax_map.scatter(
                [pred_mle[i][1]], [pred_mle[i][0]],
                c=COLOR_MLE, s=80, zorder=5, marker="s",
                edgecolors="black", linewidths=0.6,
            )
        if pred_post[i] is not None:
            ax_map.scatter(
                [pred_post[i][1]], [pred_post[i][0]],
                c=COLOR_POST, s=80, zorder=6, marker="D",
                edgecolors="black", linewidths=0.6,
            )
        ax_map.scatter(
            [t_o.lon], [t_o.lat],
            c=COLOR_TRUTH, s=80, zorder=7, marker="o",
            edgecolors="black", linewidths=0.6,
        )

    ax_map.set_aspect("equal")
    ax_map.set_xlabel("longitude")
    ax_map.set_ylabel("latitude")
    ax_map.set_title(
        f"Posterior cloud + predictions at {len(highlights)} transition(s)"
    )

    # Legend with custom handles.
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_TRUTH,
               markersize=10, markeredgecolor="black", label="truth @ 15s"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=COLOR_MLE,
               markersize=10, markeredgecolor="black", label="(A) MLE const-speed"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor=COLOR_POST,
               markersize=10, markeredgecolor="black", label="(B) posterior MAP"),
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=COLOR_POST, markersize=12, alpha=0.4,
               markeredgecolor="none", label="posterior cloud (size ∝ weight)"),
    ]
    ax_map.legend(handles=legend_handles, loc="lower right", fontsize=9)

    fig.suptitle(
        f"Dwell-aware `at_time(t)` vs MLE const-speed on Porto LONG trip "
        f"({DOWNSAMPLE_S}s sampling → predict at native {NATIVE_S}s)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    _log(f"wrote {out_path}")


def _render_html(
    *,
    network,
    truth: list[RawObservation],
    obs_120: list[RawObservation],
    pred_mle: list[tuple[float, float] | None],
    pred_post: list[tuple[float, float] | None],
    full_marginals: list[list[tuple[float, float, float]]],
    out_path: Path,
) -> None:
    all_lats = [o.lat for o in truth] + [o.lat for o in obs_120]
    all_lons = [o.lon for o in truth] + [o.lon for o in obs_120]
    centre = [float(np.mean(all_lats)), float(np.mean(all_lons))]
    fmap = folium.Map(location=centre, zoom_start=14, tiles="OpenStreetMap")

    truth_layer = folium.FeatureGroup(name="truth @ 15s", show=True)
    obs120_layer = folium.FeatureGroup(name="obs @ 120s", show=True)
    mle_layer = folium.FeatureGroup(name="(A) MLE const-speed pred", show=True)
    post_layer = folium.FeatureGroup(name="(B) posterior MAP pred", show=True)
    cloud_layer = folium.FeatureGroup(name="posterior cloud", show=False)

    for o in truth:
        folium.CircleMarker(
            location=[o.lat, o.lon], radius=3, color=COLOR_TRUTH,
            fill=True, fill_opacity=0.85, weight=1,
            popup=f"truth {o.timestamp.strftime('%H:%M:%S')}",
        ).add_to(truth_layer)

    for o in obs_120:
        folium.CircleMarker(
            location=[o.lat, o.lon], radius=6, color=COLOR_OBS,
            fill=True, fill_opacity=0.85, weight=1,
            popup=f"obs@120s {o.timestamp.strftime('%H:%M:%S')}",
        ).add_to(obs120_layer)

    for o, pred in zip(truth, pred_mle):
        if pred is None:
            continue
        folium.CircleMarker(
            location=list(pred), radius=3, color=COLOR_MLE,
            fill=True, fill_opacity=0.85, weight=1,
            popup=f"MLE pred {o.timestamp.strftime('%H:%M:%S')}",
        ).add_to(mle_layer)

    for o, pred in zip(truth, pred_post):
        if pred is None:
            continue
        folium.CircleMarker(
            location=list(pred), radius=4, color=COLOR_POST,
            fill=True, fill_opacity=0.85, weight=1,
            popup=f"posterior MAP {o.timestamp.strftime('%H:%M:%S')}",
        ).add_to(post_layer)

    for o, marg in zip(truth, full_marginals):
        for lat, lon, w in marg:
            folium.CircleMarker(
                location=[lat, lon], radius=2 + 8 * w, color=COLOR_POST,
                fill=True, fill_opacity=0.35, weight=0,
                popup=f"{o.timestamp.strftime('%H:%M:%S')} w={w:.3f}",
            ).add_to(cloud_layer)

    fmap.add_child(cloud_layer)
    fmap.add_child(truth_layer)
    fmap.add_child(obs120_layer)
    fmap.add_child(mle_layer)
    fmap.add_child(post_layer)
    folium.LayerControl(collapsed=False).add_to(fmap)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fmap.save(str(out_path))
    _log(f"wrote {out_path}")


def main() -> int:
    stride = DOWNSAMPLE_S // NATIVE_S

    _log(f"loading network from {PBF.name}")
    network = load_osm_network(PBF, cache_path=OSM_CACHE)
    _log(f"  {len(network)} edges")

    raw = _load_trip(min_pings=30)
    _log(f"trip {LONG_TRIP}: {len(raw)} raw 15s pings")

    config = _build_config(network)
    truth = _truth_after_spike_filter(raw, config)
    _log(f"  {len(truth)} truth pings after spike filter "
         f"({len(raw) - len(truth)} spikes removed)")

    raw_120 = _downsample(raw, stride)
    _log(f"  {len(raw_120)} pings at {DOWNSAMPLE_S}s sampling (stride={stride})")

    _log(f"running pipeline at {DOWNSAMPLE_S}s sampling…")
    segments = reconstruct_trajectory(raw_120, network, config)
    _log(f"  {len(segments)} segments produced")

    pred_mle: list[tuple[float, float] | None] = []
    pred_post: list[tuple[float, float] | None] = []
    full_marginals: list[list[tuple[float, float, float]]] = []
    errs_a: list[float] = []
    errs_b: list[float] = []
    paired_a: list[float] = []
    paired_b: list[float] = []

    for o in truth:
        a = position_at_time(segments, o.timestamp, network, rule="front")
        b = _posterior_map_prediction(segments, o.timestamp, network)
        cloud = _posterior_full_marginal(segments, o.timestamp, network)
        pred_mle.append(a)
        pred_post.append(b)
        full_marginals.append(cloud)
        if a is not None:
            errs_a.append(haversine_m(o.lat, o.lon, a[0], a[1]))
        if b is not None:
            errs_b.append(haversine_m(o.lat, o.lon, b[0], b[1]))
        if a is not None and b is not None:
            paired_a.append(haversine_m(o.lat, o.lon, a[0], a[1]))
            paired_b.append(haversine_m(o.lat, o.lon, b[0], b[1]))

    print()
    print(f"Porto LONG trip {LONG_TRIP}: predict native {NATIVE_S}s "
          f"positions from {DOWNSAMPLE_S}s reconstruction")
    print(f"truth pings: {len(truth)}  ·  segments: {len(segments)}")
    print()
    _print_stats("(A) MLE const-speed", errs_a)
    _print_stats("(B) posterior MAP", errs_b)
    if paired_a and paired_b:
        diffs = np.array(paired_b) - np.array(paired_a)
        n_better = int((diffs < 0).sum())
        n_worse = int((diffs > 0).sum())
        n_tie = int((diffs == 0).sum())
        print()
        print(f"Paired (both covered, n={len(paired_a)}):  "
              f"(B) better at {n_better}  ·  worse at {n_worse}  ·  tie at {n_tie}")
        print(f"Mean Δ (B − A): {diffs.mean():+.1f} m  "
              f"(negative means dwell-aware posterior wins on average)")

    highlights = _pick_highlight_indices(
        [haversine_m(t.lat, t.lon, p[0], p[1]) if p else float("nan")
         for t, p in zip(truth, pred_mle)],
        [haversine_m(t.lat, t.lon, p[0], p[1]) if p else float("nan")
         for t, p in zip(truth, pred_post)],
        N_HIGHLIGHT_TRANSITIONS,
    )
    if highlights:
        _log(f"highlight transitions on map: {highlights}")

    png_path = CACHE / "demo_at_time.png"
    html_path = CACHE / "demo_at_time.html"
    _render_png(
        network=network, truth=truth, obs_120=raw_120,
        pred_mle=pred_mle, pred_post=pred_post,
        full_marginals=full_marginals,
        errs_a=errs_a, errs_b=errs_b,
        highlights=highlights, out_path=png_path,
    )
    _render_html(
        network=network, truth=truth, obs_120=raw_120,
        pred_mle=pred_mle, pred_post=pred_post,
        full_marginals=full_marginals,
        out_path=html_path,
    )

    print()
    print(f"PNG : {png_path}")
    print(f"HTML: {html_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
