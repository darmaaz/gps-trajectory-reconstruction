"""Shared rendering and per-segment Viterbi for the visualize_* scripts.

Visualization scripts differ only in how they load `list[RawObservation]`
from their source dataset. Everything from preprocessing onward — collapse,
stale flagging, replay, candidate projection, path enumeration, per-segment
Viterbi, PNG, HTML — is data-source-agnostic and lives here.

Public surface:

    compute_segments(raw, network, config)
        → (collapsed, path_cands, seg_results)
        Drives the same preprocessing chain as `reconstruct_trajectory`,
        but per-segment Viterbi is wrapped in try/except so a single
        crashed segment doesn't block the whole figure.

    render_png(network, collapsed, path_cands, seg_results, title, path)
    render_html(network, collapsed, path_cands, seg_results, path)
        Same layout for every caller; only the title and output path differ.

The `dict` shape used for segments is:
    {"start": int, "end": int, "ok": bool, "edges": list[int], "error": str?}
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import folium
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from shapely.geometry import box

from src.api.pipeline import _segment_slices
from src.candidates import enumerate_paths_per_transition, project_observation
from src.inference import most_likely_trajectory
from src.model import Path as TPath
from src.preprocessing import (
    clean, collapse_by_uniqueness, drop_kinematic_spikes,
    drop_replay_bursts, flag_stale_runs,
)

if TYPE_CHECKING:
    from src.config import Config
    from src.model import CollapsedObservation, Path as ModelPath, RawObservation
    from src.network import RoadNetwork


def _log(msg: str) -> None:
    print(f"[viz] {msg}", file=sys.stderr, flush=True)


def compute_segments(
    raw: list["RawObservation"],
    network: "RoadNetwork",
    config: "Config",
) -> tuple[
    list["CollapsedObservation"],
    list[list["ModelPath"]],
    list[dict[str, Any]],
]:
    """Run preprocessing → projection → path enumeration → per-segment Viterbi.

    Returns the post-replay collapsed obs (with replay-dropped already
    filtered), per-transition path candidates, and the per-segment result
    dicts. Mirrors what `reconstruct_trajectory` does internally, but
    catches per-segment Viterbi exceptions so callers always get a complete
    `seg_results` list — the `ok=False` entries carry the error string.
    """
    cleaned = clean(raw)
    n_pre_spike = len(cleaned)
    cleaned = drop_kinematic_spikes(
        cleaned,
        spike_speed_ms=config.spike_speed_ms,
        bridge_speed_ms=config.spike_bridge_speed_ms,
        max_spike_length=config.spike_max_length,
    )
    n_dropped_spikes = n_pre_spike - len(cleaned)
    if n_dropped_spikes:
        _log(f"  {n_dropped_spikes} ping(s) dropped as kinematic spikes")
    collapsed = collapse_by_uniqueness(cleaned, config.collapse_epsilon)
    _log("stale-flagging…")
    collapsed = flag_stale_runs(collapsed, network, config.max_speed_factor)
    _log("replay-burst detection…")
    collapsed_full = drop_replay_bursts(
        collapsed,
        max_speed_ms=config.replay_max_speed_ms,
        max_speed_factor=config.max_speed_factor,
        k_consistent=config.replay_k_consistent,
        moving_threshold_ms=config.replay_moving_threshold_ms,
    )
    n_dropped = sum(1 for o in collapsed_full if o.dropped_during_replay)
    _log(f"  {n_dropped} obs marked dropped_during_replay")
    collapsed = [o for o in collapsed_full if not o.dropped_during_replay]

    state_cands = [
        project_observation(
            o, network,
            radius_meters=config.candidate_radius,
            max_candidates=config.max_state_candidates,
        )
        for o in collapsed
    ]
    budgets = [
        (collapsed[k + 1].t_first - collapsed[k].t_first).total_seconds()
        for k in range(len(collapsed) - 1)
    ]
    _log("enumerating paths…")
    path_cands = enumerate_paths_per_transition(
        state_cands, network, budgets,
        max_path_candidates=config.max_path_candidates,
        budget_slack=config.path_budget_slack,
    )

    segs = _segment_slices(state_cands, path_cands)
    _log(f"{len(collapsed)} obs, {len(segs)} segment slices")

    seg_results: list[dict[str, Any]] = []
    for start, end in segs:
        seg_states = state_cands[start:end]
        seg_paths = path_cands[start:end - 1] if end - start >= 2 else []
        seg_obs = collapsed[start:end]
        seg_budgets = budgets[start:end - 1] if end - start >= 2 else []
        try:
            subs = most_likely_trajectory(
                seg_states, seg_paths, seg_obs,
                config.emission, config.transition, seg_budgets,
            )
        except Exception as e:
            seg_results.append({
                "start": start, "end": end, "ok": False, "edges": [],
                "error": str(e),
            })
            continue
        # Graceful Viterbi: each input segment may produce 0+ sub-segments.
        # Each sub gets its own seg_results row with trip-global indices,
        # so the renderer treats them as independent segments.
        if not subs:
            seg_results.append({
                "start": start, "end": end, "ok": False, "edges": [],
                "error": "no sub-trajectory produced",
            })
            continue
        for sub in subs:
            edge_ids: list[int] = []
            for item in sub.most_likely:
                if isinstance(item, TPath):
                    edge_ids.extend(item.edges)
            seg_results.append({
                "start": start + sub.start_obs_idx,
                "end": start + sub.end_obs_idx + 1,
                "ok": True,
                "edges": edge_ids,
            })
    n_ok = sum(1 for r in seg_results if r["ok"])
    _log(f"viterbi succeeded on {n_ok}/{len(seg_results)} sub-segments")

    return collapsed, path_cands, seg_results


def render_png(
    *,
    network: "RoadNetwork",
    collapsed: list["CollapsedObservation"],
    path_cands: list[list["ModelPath"]],
    seg_results: list[dict[str, Any]],
    title: str,
    png_path: Path,
) -> None:
    """Static matplotlib figure: OSM background + obs colored by segment +
    Viterbi reconstructions overlaid in saturated segment colors."""
    obs_lats = np.array([c.lat for c in collapsed])
    obs_lons = np.array([c.lon for c in collapsed])

    pad = 0.02
    bbox = (
        obs_lons.min() - pad, obs_lats.min() - pad,
        obs_lons.max() + pad, obs_lats.max() + pad,
    )
    bbox_poly = box(*bbox)
    edge_idxs = np.asarray(network.tree.query(bbox_poly))
    _log(f"network edges in bbox: {len(edge_idxs)}")

    fig, ax = plt.subplots(figsize=(14, 12), dpi=110)

    segments_geom = []
    for i in edge_idxs:
        geom = network.geoms[i]
        if not geom.intersects(bbox_poly):
            continue
        segments_geom.append(list(geom.coords))
    if segments_geom:
        lc = LineCollection(
            segments_geom, colors="lightgray", linewidths=0.4, zorder=1,
        )
        ax.add_collection(lc)

    seg_of_obs = np.full(len(collapsed), -1, dtype=int)
    for s_idx, r in enumerate(seg_results):
        for k in range(r["start"], r["end"]):
            seg_of_obs[k] = s_idx

    cmap = plt.get_cmap("tab20", max(1, len(seg_results)))
    seg_ok = np.array([r["ok"] for r in seg_results])
    point_colors = []
    for k in range(len(collapsed)):
        sid = seg_of_obs[k]
        if sid < 0:
            point_colors.append("#888888")
        elif seg_ok[sid]:
            point_colors.append(cmap(sid))
        else:
            point_colors.append("#cccccc")

    for k in range(len(collapsed) - 1):
        x0, y0 = obs_lons[k], obs_lats[k]
        x1, y1 = obs_lons[k + 1], obs_lats[k + 1]
        if path_cands[k]:
            ax.plot([x0, x1], [y0, y1], color="#9bd09b", linewidth=0.8,
                    zorder=2, alpha=0.7)
        else:
            ax.plot([x0, x1], [y0, y1], color="#e88a8a", linewidth=1.0,
                    zorder=2, linestyle="--", alpha=0.9)

    for s_idx, r in enumerate(seg_results):
        if not r["ok"] or not r["edges"]:
            continue
        seg_color = cmap(s_idx)
        ml_segments = []
        for eid in r["edges"]:
            try:
                idx = network.edge_index_for_link(int(eid))
            except KeyError:
                continue
            ml_segments.append(list(network.geoms[idx].coords))
        if ml_segments:
            lc = LineCollection(
                ml_segments, colors=[seg_color], linewidths=2.2, zorder=3,
                alpha=0.9,
            )
            ax.add_collection(lc)

    is_stale = np.array([c.stale_flagged for c in collapsed])
    ax.scatter(
        obs_lons[~is_stale], obs_lats[~is_stale],
        c=[point_colors[i] for i in range(len(collapsed)) if not is_stale[i]],
        s=18, zorder=4, edgecolors="black", linewidths=0.3,
    )
    if is_stale.any():
        ax.scatter(
            obs_lons[is_stale], obs_lats[is_stale],
            c=[point_colors[i] for i in range(len(collapsed)) if is_stale[i]],
            marker="*", s=200, zorder=5, edgecolors="black", linewidths=0.6,
            label=f"stale-flagged ({is_stale.sum()})",
        )

    label_idx = list(range(0, len(collapsed), max(1, len(collapsed) // 12)))
    if (len(collapsed) - 1) not in label_idx:
        label_idx.append(len(collapsed) - 1)
    for k in label_idx:
        ax.annotate(
            str(k), (obs_lons[k], obs_lats[k]),
            xytext=(4, 4), textcoords="offset points",
            fontsize=7, color="#222222", zorder=6,
        )

    ax.set_aspect("equal")
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_title(title)

    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, bbox_inches="tight")
    _log(f"wrote {png_path}")
    plt.close(fig)


_FOLIUM_SEG_COLORS: list[str] = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


def render_html(
    *,
    network: "RoadNetwork",
    collapsed: list["CollapsedObservation"],
    path_cands: list[list["ModelPath"]],
    seg_results: list[dict[str, Any]],
    html_path: Path,
) -> None:
    """Pannable folium map: OSM tiles, observations as circle markers,
    Viterbi paths as heavy segment-coloured polylines."""
    obs_lats = np.array([c.lat for c in collapsed])
    obs_lons = np.array([c.lon for c in collapsed])

    seg_of_obs = np.full(len(collapsed), -1, dtype=int)
    for s_idx, r in enumerate(seg_results):
        for k in range(r["start"], r["end"]):
            seg_of_obs[k] = s_idx
    seg_ok = np.array([r["ok"] for r in seg_results])

    centre = [float(obs_lats.mean()), float(obs_lons.mean())]
    fmap = folium.Map(location=centre, zoom_start=13, tiles="OpenStreetMap")

    for k, c in enumerate(collapsed):
        sid = int(seg_of_obs[k])
        seg_ok_ = bool(seg_ok[sid]) if sid >= 0 else False
        color = "red" if c.stale_flagged else (
            "blue" if seg_ok_ else "gray"
        )
        radius = 7 if c.stale_flagged else 4
        seg_info = (
            f"segment {sid} (ok)" if seg_ok_
            else f"segment {sid} (viterbi failed)"
        )
        folium.CircleMarker(
            location=[c.lat, c.lon],
            radius=radius,
            color=color, fill=True, fill_opacity=0.8,
            popup=(
                f"<b>obs {k}</b><br>"
                f"{c.t_first.strftime('%H:%M:%S')}<br>"
                f"({c.lat:.5f}, {c.lon:.5f})<br>"
                f"count={c.collapsed_count}, "
                f"stale={c.stale_flagged}<br>"
                f"{seg_info}"
            ),
        ).add_to(fmap)

    for k in range(len(collapsed) - 1):
        c0, c1 = collapsed[k], collapsed[k + 1]
        ok = bool(path_cands[k])
        folium.PolyLine(
            locations=[[c0.lat, c0.lon], [c1.lat, c1.lon]],
            color="green" if ok else "red",
            weight=2 if ok else 3,
            opacity=0.5 if ok else 0.8,
            dash_array=None if ok else "6 6",
        ).add_to(fmap)

    for s_idx, r in enumerate(seg_results):
        if not r["ok"] or not r["edges"]:
            continue
        col = _FOLIUM_SEG_COLORS[s_idx % len(_FOLIUM_SEG_COLORS)]
        for eid in r["edges"]:
            try:
                idx = network.edge_index_for_link(int(eid))
            except KeyError:
                continue
            coords = [(lat, lon) for lon, lat in network.geoms[idx].coords]
            folium.PolyLine(
                locations=coords,
                color=col, weight=4, opacity=0.85,
            ).add_to(fmap)

    html_path.parent.mkdir(parents=True, exist_ok=True)
    fmap.save(str(html_path))
    _log(f"wrote {html_path}")


def print_segment_summary(seg_results: list[dict[str, Any]]) -> None:
    """Plain-text per-segment status table for stdout."""
    print()
    print("Segment results:")
    for s_idx, r in enumerate(seg_results):
        n = r["end"] - r["start"]
        flag = "OK " if r["ok"] else "X  "
        detail = (
            f"edges={len(r.get('edges', []))}" if r["ok"]
            else r.get("error", "")[:50]
        )
        print(
            f"  [{s_idx:>2}] obs[{r['start']:>3}:{r['end']:>3}] "
            f"({n:>3} obs)  {flag}  {detail}",
        )
