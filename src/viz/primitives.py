"""Matplotlib rendering primitives.

All functions take an `ax: matplotlib.axes.Axes` and operate on it in
place. The caller controls figure construction, sizing, and saving; this
module only paints data onto axes.

Colour conventions:

    truth (ground-truth GPS at native sampling) : steel blue
    sparse observation (what the model sees)    : warm orange
    most-likely path (Viterbi)                  : firm orange
    candidate paths (posterior alternatives)    : Set1 categorical
    dwell-aware prediction                      : indigo
    no-dwell (constant-speed) prediction        : olive
    error connector                             : faint crimson
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

import numpy as np
from matplotlib.collections import LineCollection
from shapely.geometry import box

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from ..model import Path as ModelPath, RawObservation
    from ..network import RoadNetwork

# Categorical palette for candidate-path overlays. Set1 is high-contrast
# and reproduces well in print. The first 7 colours are well-separated.
CANDIDATE_COLORS: tuple[str, ...] = (
    "#e41a1c",  # red
    "#377eb8",  # blue
    "#4daf4a",  # green
    "#984ea3",  # purple
    "#ff7f00",  # orange
    "#a65628",  # brown
    "#f781bf",  # pink
)

COLOR_TRUTH = "#4682b4"          # steel blue
COLOR_OBS_SPARSE = "#d97706"     # warm orange
COLOR_MLE_PATH = "#ea580c"       # firm orange (saturated)
COLOR_PRED_DWELL = "#4338ca"     # indigo
COLOR_PRED_NODWELL = "#65a30d"   # olive
COLOR_ERR = "#dc2626"            # crimson
COLOR_NETWORK = "#cbd5e1"        # slate-200 (very light gray-blue)


def clean_map_axes(
    ax: "Axes",
    *,
    lats: Iterable[float] | None = None,
    lons: Iterable[float] | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    pad_frac: float = 0.05,
) -> tuple[float, float, float, float]:
    """Configure `ax` as a clean map: white background, no spines or ticks,
    equal aspect, bbox computed from input lat/lon if not given explicitly.

    Returns the bbox (lon_min, lat_min, lon_max, lat_max) actually applied.
    """
    if bbox is None:
        if lats is None or lons is None:
            raise ValueError("provide either bbox or both lats and lons")
        lats_arr = np.asarray(list(lats), dtype=float)
        lons_arr = np.asarray(list(lons), dtype=float)
        pad_lat = (lats_arr.max() - lats_arr.min()) * pad_frac
        pad_lon = (lons_arr.max() - lons_arr.min()) * pad_frac
        bbox = (
            float(lons_arr.min() - pad_lon),
            float(lats_arr.min() - pad_lat),
            float(lons_arr.max() + pad_lon),
            float(lats_arr.max() + pad_lat),
        )

    ax.set_facecolor("white")
    ax.set_xlim(bbox[0], bbox[2])
    ax.set_ylim(bbox[1], bbox[3])
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(False)
    return bbox


def draw_network_backdrop(
    ax: "Axes",
    network: "RoadNetwork",
    bbox: tuple[float, float, float, float],
    *,
    color: str = COLOR_NETWORK,
    linewidth: float = 0.5,
) -> int:
    """Render every road-network edge whose geometry intersects `bbox` as a
    faint backdrop on `ax`. Returns the number of edges drawn.
    """
    poly = box(*bbox)
    idxs = np.asarray(network.tree.query(poly))
    segs = []
    for i in idxs:
        geom = network.geoms[i]
        if not geom.intersects(poly):
            continue
        segs.append(list(geom.coords))
    if segs:
        ax.add_collection(LineCollection(
            segs, colors=color, linewidths=linewidth, zorder=1,
        ))
    return len(segs)


def draw_path_edges(
    ax: "Axes",
    path: "ModelPath",
    network: "RoadNetwork",
    *,
    color: str,
    linewidth: float = 2.0,
    alpha: float = 1.0,
    label: str | None = None,
    zorder: int = 3,
) -> None:
    """Render a single `Path`'s edge sequence as a polyline on `ax`."""
    plotted = False
    for eid in path.edges:
        try:
            idx = network.edge_index_for_link(int(eid))
        except KeyError:
            continue
        geom = network.geoms[idx]
        xs, ys = geom.xy
        ax.plot(
            xs, ys,
            color=color, linewidth=linewidth, alpha=alpha,
            solid_capstyle="round", zorder=zorder,
            label=label if not plotted else None,
        )
        plotted = True


def draw_candidate_paths(
    ax: "Axes",
    paths_with_weights: list[tuple["ModelPath", float]],
    network: "RoadNetwork",
    *,
    top_k: int = 5,
    rest_color: str = "#9ca3af",
    rest_alpha: float = 0.18,
) -> list[tuple[str, float]]:
    """Draw the top-K candidate paths in distinct categorical colors with
    linewidth proportional to posterior weight. Remaining low-weight
    candidates are drawn collectively as a faint gray backdrop.

    `paths_with_weights` is a list of `(path, weight)` sorted by descending
    weight (the caller is expected to sort).

    Returns the colour/weight pairs of the top-K paths so the caller can
    build a legend that matches.
    """
    legend_entries: list[tuple[str, float]] = []
    for i, (path, weight) in enumerate(paths_with_weights):
        if i < top_k:
            color = CANDIDATE_COLORS[i % len(CANDIDATE_COLORS)]
            # Linewidth ∝ sqrt(weight) — keeps low-weight paths visible
            # while still emphasising the heavyweights.
            lw = 1.5 + 4.0 * float(np.sqrt(max(weight, 0.0)))
            draw_path_edges(
                ax, path, network,
                color=color, linewidth=lw, alpha=0.9, zorder=4 + i,
                label=f"path {i+1}  (weight {weight:.3f})",
            )
            legend_entries.append((color, float(weight)))
        else:
            draw_path_edges(
                ax, path, network,
                color=rest_color, linewidth=1.0, alpha=rest_alpha, zorder=2,
            )
    return legend_entries


def draw_observations(
    ax: "Axes",
    observations: list["RawObservation"],
    *,
    color: str = COLOR_TRUTH,
    size: float = 12.0,
    marker: str = "o",
    edge: str = "white",
    label: str | None = None,
    zorder: int = 10,
) -> None:
    """Scatter observations on `ax` with a clean white edge so dots stand
    out cleanly over the backdrop."""
    lats = [o.lat for o in observations]
    lons = [o.lon for o in observations]
    ax.scatter(
        lons, lats, s=size, c=color, marker=marker,
        edgecolors=edge, linewidths=0.6, zorder=zorder, label=label,
    )


def draw_position_predictions(
    ax: "Axes",
    *,
    truth_lat: float, truth_lon: float,
    nodwell_pred: tuple[float, float] | None = None,
    dwell_pred: tuple[float, float] | None = None,
    annotate: bool = True,
) -> None:
    """Draw a single moment's predictions and ground truth on `ax`.

    Truth is a circle, no-dwell prediction is a square, dwell-aware
    prediction is a diamond. Faint error connectors are drawn from each
    prediction to the truth so the error magnitude is visible by eye.
    """
    ax.scatter(
        [truth_lon], [truth_lat],
        s=140, c=COLOR_TRUTH, marker="o",
        edgecolors="white", linewidths=1.0, zorder=10,
    )
    if annotate:
        ax.annotate(
            "truth",
            xy=(truth_lon, truth_lat), xytext=(6, 6),
            textcoords="offset points",
            fontsize=8, color=COLOR_TRUTH,
        )
    if nodwell_pred is not None:
        ax.plot(
            [truth_lon, nodwell_pred[1]], [truth_lat, nodwell_pred[0]],
            color=COLOR_ERR, linewidth=0.8, alpha=0.6, zorder=9,
        )
        ax.scatter(
            [nodwell_pred[1]], [nodwell_pred[0]],
            s=100, c=COLOR_PRED_NODWELL, marker="s",
            edgecolors="white", linewidths=1.0, zorder=11,
        )
    if dwell_pred is not None:
        ax.plot(
            [truth_lon, dwell_pred[1]], [truth_lat, dwell_pred[0]],
            color=COLOR_ERR, linewidth=0.8, alpha=0.6, zorder=9,
        )
        ax.scatter(
            [dwell_pred[1]], [dwell_pred[0]],
            s=100, c=COLOR_PRED_DWELL, marker="D",
            edgecolors="white", linewidths=1.0, zorder=12,
        )
