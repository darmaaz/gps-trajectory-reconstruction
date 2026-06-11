"""Offset-honouring path geometry.

`path_polyline` is the single sanctioned way to turn a `Path` into map
coordinates. Every previous consumer (figures, coverage metrics, truth-
distance scoring) concatenated the **full** geometry of every edge in
`path.edges`, including the terminal edges the path only partially
traverses — `start_offset` / `end_offset` were ignored. That exaggerates
paths past their anchoring observations (the fig-1 "overshoot"), credits
coverage to road the path never drove, and biases any geometry-based
metric. Use this instead.

Conventions (matching `RoadNetwork.perpendicular_distance`):
    metric offset → degree position via `(offset_m / lengths_m) * geom.length`.

Edge cases:
    - off-road path: straight segment between the two projected endpoints;
    - single-edge stay path with backward offsets (`end < start`): the
      short backward segment between the two projections — honest
      zero-motion jitter span, not the whole edge;
    - unknown `link_id` (subgraph consumers): skipped, like existing
      figure code.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from shapely.ops import substring

if TYPE_CHECKING:
    from ..model import Path
    from .loader import RoadNetwork


def _deg_pos(network: "RoadNetwork", edge_idx: int, offset_m: float) -> float:
    """Degree-space arc position for a metric offset on an edge."""
    m_len = float(network.lengths_m[edge_idx])
    deg_len = network.geoms[edge_idx].length
    if m_len <= 0.0 or deg_len <= 0.0:
        return 0.0
    return (float(offset_m) / m_len) * deg_len


def path_polyline(path: "Path", network: "RoadNetwork") -> np.ndarray:
    """`(N, 2)` array of `(lon, lat)` vertices for the path's driven
    geometry — terminal edges trimmed to `start_offset` / `end_offset`,
    middle edges in full.

    Always returns at least 2 vertices (degenerate zero-motion paths
    return the projected point twice).
    """
    idxs: list[int] = []
    for link in path.edges:
        try:
            idxs.append(network.edge_index_for_link(int(link)))
        except KeyError:
            idxs.append(-1)

    # Off-road: straight line between the two projected endpoints.
    if path.is_off_road:
        pts = []
        for idx, off in ((idxs[0], path.start_offset), (idxs[-1], path.end_offset)):
            if idx < 0:
                continue
            pt = network.geoms[idx].interpolate(_deg_pos(network, idx, off))
            pts.append((float(pt.x), float(pt.y)))
        if len(pts) < 2:
            pts = pts * 2 if pts else [(np.nan, np.nan)] * 2
        return np.asarray(pts, dtype=float)

    coords: list[tuple[float, float]] = []

    def _extend(xy_seq) -> None:
        for x, y in xy_seq:
            pt = (float(x), float(y))
            if not coords or coords[-1] != pt:
                coords.append(pt)

    # Direction-violation paths (`reversed_mask`): a reversed first edge
    # is exited backward toward its from_node (offset → 0); a reversed
    # last edge is entered backward from its to_node (length → offset);
    # reversed middles run end-to-start. `shapely.ops.substring` handles
    # the reversal natively when start_dist > end_dist.
    if len(idxs) == 1:
        idx = idxs[0]
        if idx >= 0:
            geom = network.geoms[idx]
            d0 = _deg_pos(network, idx, path.start_offset)
            d1 = _deg_pos(network, idx, path.end_offset)
            seg = substring(geom, d0, d1)
            _extend(np.asarray(seg.coords) if seg.geom_type == "LineString"
                    else [(seg.x, seg.y)])
    else:
        for pos, idx in enumerate(idxs):
            if idx < 0:
                continue
            geom = network.geoms[idx]
            rev = path.edge_reversed(pos)
            if pos == 0:
                d0 = _deg_pos(network, idx, path.start_offset)
                seg = substring(geom, d0, 0.0 if rev else geom.length)
            elif pos == len(idxs) - 1:
                d1 = _deg_pos(network, idx, path.end_offset)
                seg = substring(geom, geom.length if rev else 0.0, d1)
            else:
                seg = substring(geom, geom.length, 0.0) if rev else geom
            _extend(np.asarray(seg.coords) if seg.geom_type == "LineString"
                    else [(seg.x, seg.y)])

    if not coords:
        return np.full((2, 2), np.nan)
    if len(coords) == 1:
        coords = coords * 2
    return np.asarray(coords, dtype=float)
