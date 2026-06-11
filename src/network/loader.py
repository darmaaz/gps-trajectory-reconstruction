"""OSM road network loader and in-memory `RoadNetwork` graph.

The pipeline needs a routable graph: edges with length and max-speed,
adjacency at intersection nodes, plus a spatial index for state projection.
This module turns an OSM PBF into that representation and caches the parsed
edge table as parquet for fast reloads.

Pipeline:

    PBF (≈600 MB per country)
      → osmium two-pass:
          pass 1 — count node references in drivable ways
          pass 2 — split each way at intersection nodes (refs ≥ 2) into
                   routable edges; parse `maxspeed`; haversine-sum length
      → adjacency dict keyed by node id
      → STRtree over edge geometries
      → cached as parquet of (edge_id, road_class, max_speed_ms,
        from_node, to_node, length_m, wkt) for fast reloads.

Caveats:

- Spec §Storage names PostgreSQL + PostGIS as the canonical store. PBF +
  parquet is what's implemented here, matching the ported pipeline. The
  in-memory shape is identical regardless of source, so PostGIS migration is
  additive.
- Turn restrictions live in OSM `restriction` relations, which are not yet
  parsed. `turn_restrictions` is empty until that's added; routing must treat
  the empty set as "no restrictions".
- A* and Yen's K-shortest-paths are not yet implemented (`shortest_travel_time`
  and `candidate_paths` raise `NotImplementedError`). The graph data shape is
  in place so they can be filled in without changing callers.

Coordinate-space caveat for the spatial index
---------------------------------------------
The STRtree indexes geometry in degree space, not metric. At Mexico's
latitudes (~19–32°N), 1° lon ≈ 85–105 km, 1° lat ≈ 111 km. Tree picks
candidates by coordinate distance; final filtering and sorting use exact
equirectangular metres (`equirectangular_distance_m`). Sub-km worst-case
distortion <5%. If the network ever spans far from Mexico's latitudes, move
the tree to UTM-projected geometry.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd
import shapely
from shapely.geometry import LineString
from shapely.strtree import STRtree

from ..config import v_max_for, v_typical_for
from ..geo import M_PER_DEG_LAT, equirectangular_distance_m, haversine_m

DRIVABLE_HIGHWAY: frozenset[str] = frozenset({
    "motorway", "trunk", "primary", "secondary", "tertiary",
    "unclassified", "residential", "service", "living_street",
    "motorway_link", "trunk_link", "primary_link",
    "secondary_link", "tertiary_link",
})

NodeId = int
EdgeIdx = int    # internal index into the parallel arrays (0..E-1)


@dataclass
class RoadNetwork:
    """Routable in-memory graph plus a spatial index for state projection.

    Parallel arrays indexed by `EdgeIdx ∈ [0, E)`:
        edge_ids[e]      — stable id (way id, or synthetic for split segments)
        geoms[e]         — shapely LineString in (lon, lat) order
        lengths_m[e]     — metric arc length, haversine-summed
        road_classes[e]  — OSM `highway` value
        max_speeds_ms[e] — OSM `maxspeed` (m/s) if parseable, else
                           `v_max_for(class)`. Used by stale-jump detection
                           as a feasibility upper bound.
        typical_speeds_ms[e] — class-conditional typical realised speed
                           used by routing cost (`length / typical_speed`
                           graph weights, edge enumeration costs). Defaults
                           to `v_typical_for(class)`; can be overridden
                           per-network via `set_typical_speeds_by_class`.
        from_node[e]     — NodeId at edge start
        to_node[e]       — NodeId at edge end

    Topology:
        adjacency[node]    — outgoing edge indices from `node`
        turn_restrictions  — forbidden `(in_edge_idx, out_edge_idx)` pairs.
                             Empty until `restriction` relations are parsed.
        node_positions     — `NodeId → (lat, lon)`; needed for the A*
                             heuristic in routing.

    Spatial:
        tree              — STRtree over `geoms` (degree space)

    Caching:
        `_nx_graph_cache` is set lazily by `network.routing` on first use.
        Not part of the public surface; do not rely on it externally.
    """

    edge_ids: np.ndarray            # (E,) int64
    geoms: np.ndarray               # (E,) object array of LineStrings
    lengths_m: np.ndarray           # (E,) float64
    road_classes: np.ndarray        # (E,) object
    max_speeds_ms: np.ndarray       # (E,) float64 — feasibility upper bound
    typical_speeds_ms: np.ndarray   # (E,) float64 — routing-cost prior
    from_node: np.ndarray           # (E,) int64
    to_node: np.ndarray             # (E,) int64
    adjacency: dict[NodeId, list[EdgeIdx]]
    turn_restrictions: set[tuple[EdgeIdx, EdgeIdx]]
    node_positions: dict[NodeId, tuple[float, float]]
    tree: STRtree = field(repr=False)
    # Traffic-control feature data. `to_node_is_signal[e] == 1` iff
    # `to_node[e]` is tagged `highway=traffic_signals` in OSM; same for
    # `to_node_is_stop` with `highway=stop`. Used by `path_features` to
    # populate slots [3] (n_signals) and [4] (n_stop_signs). Defaults to
    # all zeros when the loader didn't extract signal/stop data
    # (synthetic test networks, older caches without the
    # pt_signal_stop_nodes.parquet sidecar).
    to_node_is_signal: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int8))
    to_node_is_stop:   np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int8))
    # `node_total_degree[node]` = total edges (in + out) touching that
    # node. Used by `path_features` to count traversed intersections
    # (slot [17] n_intersections): each consecutive edge-pair junction in
    # the path is an "intersection" if `node_total_degree ≥ 3` (i.e. the
    # path crosses a real multi-way junction, not just an OSM way split).
    # Computed by `build_network_from_records`; defaults to empty for
    # synthetic fixtures, in which case path_features falls back to
    # `len(path.edges) - 1` for slot [17].
    node_total_degree: dict[int, int] = field(default_factory=dict)
    _nx_graph_cache: object | None = field(default=None, repr=False, compare=False)
    # Lazy identity caches — see `twin_indices` / `undirected_segment_keys`.
    _twin_idx_cache: object | None = field(default=None, repr=False, compare=False)
    _seg_keys_cache: object | None = field(default=None, repr=False, compare=False)
    # `(cost_factor, DiGraph)` cache for the direction-violation routing
    # graph — set lazily by `network.routing` when
    # `enable_direction_violation` is on. Invalidated alongside
    # `_nx_graph_cache` (set_typical_speeds_by_class).
    _nx_graph_permissive_cache: object | None = field(default=None, repr=False, compare=False)

    def __len__(self) -> int:
        return len(self.geoms)

    def set_typical_speeds_by_class(
        self, class_dict: dict[str, float],
    ) -> None:
        """Recompute `typical_speeds_ms` from a `road_class → m/s` map.

        Mutates in place. Edges whose class is not in `class_dict` fall back
        to `v_typical_for(class)` (the module-level default). Invalidates
        `_nx_graph_cache` so subsequent routing rebuilds the graph against
        the new costs.
        """
        new_speeds = np.array(
            [class_dict.get(str(rc), v_typical_for(str(rc))) for rc in self.road_classes],
            dtype=float,
        )
        self.typical_speeds_ms = new_speeds
        self._nx_graph_cache = None    # graph weights depend on these
        self._nx_graph_permissive_cache = None

    def subgraph_for_bbox(
        self,
        lat_min: float, lat_max: float,
        lon_min: float, lon_max: float,
        buffer_m: float = 5000.0,
    ) -> "RoadNetwork":
        """Return a new `RoadNetwork` containing only edges whose geometry
        intersects the bbox padded by `buffer_m` metres.

        Built for per-vehicle-day calibration on country-wide networks:
        full Mexico has ~17 M edges and Yen's K-shortest scales poorly on
        that. A vehicle-day's bbox typically covers <100 km × 100 km, so
        the subgraph holds 50–200 K edges — Dijkstra/Yen's go from
        seconds-per-call to sub-millisecond.

        Caveat: the returned network is freshly built (new STRtree, new
        adjacency, new node-position dict). It does NOT share state with
        `self` — modifying one doesn't affect the other. The `_nx_graph_cache`
        on the returned network starts empty; routing builds it lazily on
        first call.

        The buffer should be wide enough that any plausible route between
        observations stays inside the bbox. 5 km default is generous for
        ~60-second sample cadence at highway speeds; bump it for sparser
        sampling or if routes diverge significantly.
        """
        # Convert metric buffer to degree padding at the bbox-centre latitude.
        mid_lat = 0.5 * (lat_min + lat_max)
        pad_lat = buffer_m / M_PER_DEG_LAT
        pad_lon = buffer_m / (M_PER_DEG_LAT * float(np.cos(np.radians(mid_lat))))
        from shapely.geometry import box as _shp_box
        bbox_poly = _shp_box(
            lon_min - pad_lon, lat_min - pad_lat,
            lon_max + pad_lon, lat_max + pad_lat,
        )
        cand_idxs = np.asarray(self.tree.query(bbox_poly))
        records: list[tuple] = []
        # Track the node IDs that carry signal/stop tags on the parent
        # network so the subgraph inherits the same per-edge flags.
        # Without this, subgraph_for_bbox produces a network whose
        # signal/stop arrays are all zero — feature slots 3 and 4 silently
        # become useless on any candidate enumerated within a subgraph.
        parent_signal_nodes: set[int] = set()
        parent_stop_nodes:   set[int] = set()
        for i in cand_idxs:
            i_int = int(i)
            geom = self.geoms[i_int]
            if not geom.intersects(bbox_poly):
                continue
            records.append((
                int(self.edge_ids[i_int]),
                str(self.road_classes[i_int]),
                float(self.max_speeds_ms[i_int]),
                geom,
                int(self.from_node[i_int]),
                int(self.to_node[i_int]),
                float(self.lengths_m[i_int]),
            ))
            if (
                len(self.to_node_is_signal) > i_int
                and self.to_node_is_signal[i_int]
            ):
                parent_signal_nodes.add(int(self.to_node[i_int]))
            if (
                len(self.to_node_is_stop) > i_int
                and self.to_node_is_stop[i_int]
            ):
                parent_stop_nodes.add(int(self.to_node[i_int]))
        # node_total_degree is recomputed from the subgraph's records
        # in build_network_from_records — passing parent's degrees would
        # over-count edges that crossed the bbox boundary.
        return build_network_from_records(
            records,
            signal_node_ids=parent_signal_nodes or None,
            stop_node_ids=parent_stop_nodes or None,
        )

    def perpendicular_distance(
        self, obs_lat: float, obs_lon: float,
        link_id: int, offset_m: float,
    ) -> float:
        """Distance in metres from `(obs_lat, obs_lon)` to the projected
        point at `offset_m` along the edge for `link_id`.

        Same linear degree↔metres scaling as `project_point`'s offset
        conversion. Returns `inf` if the link is unknown or has zero length.
        Used by training (analytical scale gradient) and other consumers
        that need to reproduce what `StudentTEmission.log_potential`
        computes internally.
        """
        try:
            idx = self.edge_index_for_link(link_id)
        except KeyError:
            return float("inf")
        geom = self.geoms[idx]
        deg_len = geom.length
        m_len = float(self.lengths_m[idx])
        if deg_len <= 0.0 or m_len <= 0.0:
            return float("inf")
        dist_along_deg = (offset_m / m_len) * deg_len
        pt = geom.interpolate(dist_along_deg)
        return float(equirectangular_distance_m(
            obs_lat, obs_lon, float(pt.y), float(pt.x),
        ))

    def edge_index_for_link(self, link_id: int) -> int:
        """Internal `EdgeIdx` for a stable `link_id`. Lazy dict cache built
        on first use; subsequent lookups are O(1). Raises `KeyError` for
        unknown ids.
        """
        cached = getattr(self, "_link_to_idx_cache", None)
        if cached is None:
            arr = self.edge_ids
            cached = {int(arr[i]): i for i in range(len(arr))}
            self._link_to_idx_cache = cached    # type: ignore[attr-defined]
        return cached[int(link_id)]

    # ── physical-road identity ──────────────────────────────────────────
    #
    # The PBF parser emits two-way streets as two *independent* directed
    # edges; the reverse twin gets a synthetic `edge_id` ≥ 10**12 with no
    # stored linkage to its forward sibling (and split segments lose the
    # OSM `way_id` entirely). Any consumer that counts, dedups, or
    # calibrates on raw `link_id` therefore treats one physical street as
    # two unrelated roads. The two methods below derive a physical-road
    # identity from the topology arrays alone — no PBF re-parse, no cache
    # schema change — and are the single sanctioned answer to "are these
    # two directed edges the same street?".
    #
    # Known limitation: distinct parallel ways that genuinely connect the
    # same node pair (rare; routing already collapses them in its DiGraph
    # mirror) share an undirected key. Acceptable for counting/metrics;
    # if it ever matters, refine the key with `way_id` once the parquet
    # cache schema carries it.

    def undirected_segment_keys(self) -> np.ndarray:
        """`(E, 2)` int64 array: `(min(from,to), max(from,to))` per edge.

        Canonical *undirected* segment identity — a two-way street's
        forward and reverse twins map to the same key. Lazily computed
        and cached; row `e` corresponds to `EdgeIdx e`.
        """
        cached = self._seg_keys_cache
        if cached is not None:
            return cached  # type: ignore[return-value]
        lo = np.minimum(self.from_node, self.to_node)
        hi = np.maximum(self.from_node, self.to_node)
        keys = np.column_stack([lo, hi]).astype(np.int64)
        keys.setflags(write=False)
        self._seg_keys_cache = keys
        return keys

    def segment_key(self, edge_idx: "EdgeIdx") -> tuple[int, int]:
        """Undirected segment key for an internal edge index."""
        row = self.undirected_segment_keys()[int(edge_idx)]
        return (int(row[0]), int(row[1]))

    def segment_key_for_link(self, link_id: int) -> tuple[int, int]:
        """Undirected segment key for a stable `link_id`.

        Raises `KeyError` for unknown ids (same contract as
        `edge_index_for_link`).
        """
        return self.segment_key(self.edge_index_for_link(link_id))

    def twin_indices(self) -> np.ndarray:
        """`(E,)` int64 array: `twin[e]` is the `EdgeIdx` of the
        opposite-direction twin of edge `e`, or `-1` when none exists
        (one-way street, or a degenerate self-loop).

        A twin is an edge with swapped `(from_node, to_node)` whose length
        matches within `max(1 m, 0.1 %)` — loader-emitted reverse twins are
        exact mirrors, so the tolerance only guards float round-trips
        through the parquet cache. When several reverse edges qualify
        (parallel ways), the closest length wins. Lazily computed, cached.
        """
        cached = self._twin_idx_cache
        if cached is not None:
            return cached  # type: ignore[return-value]
        e_count = len(self)
        twins = np.full(e_count, -1, dtype=np.int64)
        fr = self.from_node
        to = self.to_node
        lengths = self.lengths_m
        by_pair: dict[tuple[int, int], list[int]] = {}
        for e in range(e_count):
            by_pair.setdefault((int(fr[e]), int(to[e])), []).append(e)
        for e in range(e_count):
            u, v = int(fr[e]), int(to[e])
            if u == v:
                continue    # self-loop: direction is ill-defined
            reverse = by_pair.get((v, u))
            if not reverse:
                continue
            le = float(lengths[e])
            tol = max(1.0, 1e-3 * le)
            best = -1
            best_d = float("inf")
            for c in reverse:
                d = abs(float(lengths[c]) - le)
                if d <= tol and d < best_d:
                    best, best_d = c, d
            twins[e] = best
        twins.setflags(write=False)
        self._twin_idx_cache = twins
        return twins

    def project_point(
        self,
        lat: float,
        lon: float,
        radius_meters: float = 50.0,
        max_candidates: int = 5,
    ) -> list[tuple[EdgeIdx, float, float]]:
        """Top-K edge projections within `radius_meters`.

        Returns `[(edge_idx, offset_m, perp_distance_m), ...]` sorted by
        perpendicular distance ascending. `offset_m` is the arc-length
        position of the projected point along the edge, measured from
        `from_node`. See SPEC.md §candidates.project_observation.

        Implementation: rtree query with a generous degree-space buffer to
        gather candidates; then exact equirectangular metric distance to
        filter to `radius_meters` and sort.
        """
        # 1.5× overshoot on the degree buffer absorbs longitude scaling and
        # rtree's bbox-vs-geometry slack; metric filter prunes precisely.
        buffer_deg = radius_meters / M_PER_DEG_LAT * 1.5
        point = shapely.points([lon], [lat])[0]
        cand_idxs = np.asarray(self.tree.query(point.buffer(buffer_deg)))
        if cand_idxs.size == 0:
            return []

        cand_geoms = self.geoms[cand_idxs]
        dist_along_deg = shapely.line_locate_point(cand_geoms, point)
        nearest_pts = shapely.line_interpolate_point(cand_geoms, dist_along_deg)
        nlon = shapely.get_x(nearest_pts)
        nlat = shapely.get_y(nearest_pts)
        perp_m = equirectangular_distance_m(lat, lon, nlat, nlon)

        within = perp_m <= radius_meters
        if not within.any():
            return []
        cand_idxs = cand_idxs[within]
        dist_along_deg = dist_along_deg[within]
        perp_m = perp_m[within]

        # Convert arc-length from degree-space to metres by linear scaling
        # within each edge. Accurate while edges are short relative to the
        # scale at which lat/lon distortion varies (typical urban edges
        # <500m, error <0.1%).
        deg_lengths = np.array([self.geoms[i].length for i in cand_idxs])
        m_lengths = self.lengths_m[cand_idxs]
        with np.errstate(divide="ignore", invalid="ignore"):
            offset_m = np.where(
                deg_lengths > 0, dist_along_deg / deg_lengths * m_lengths, 0.0,
            )

        order = np.argsort(perp_m)[: max_candidates]
        return [
            (int(cand_idxs[i]), float(offset_m[i]), float(perp_m[i]))
            for i in order
        ]

    def shortest_travel_time(
        self,
        from_lat: float, from_lon: float,
        to_lat: float, to_lon: float,
        max_speed_factor: float = 1.0,
    ) -> float:
        """Lower-bound travel time in seconds between two points.

        Thin delegator to `network.routing.shortest_travel_time` so callers
        can use the spec's `network.shortest_travel_time(...)` API while the
        algorithm lives in `routing.py`.
        """
        from .routing import shortest_travel_time
        return shortest_travel_time(
            self, from_lat, from_lon, to_lat, to_lon, max_speed_factor,
        )


_MAXSPEED_NUMERIC = re.compile(r"\s*(\d+(?:\.\d+)?)\s*(km/h|kmh|kph|mph)?\s*$")


def _parse_maxspeed_ms(tag: str | None, road_class: str) -> float:
    """Parse OSM `maxspeed` to m/s, falling back to `v_max_for(road_class)`.

    OSM convention: bare numbers are km/h; "mph" is explicit. Non-numeric
    values ("none", "signals", "walk") fall back to the per-class envelope.
    Zero or negative parsed values are also treated as malformed and fall
    back — `maxspeed=0` shows up in real Mexico OSM data.
    """
    if not tag:
        return v_max_for(road_class)
    m = _MAXSPEED_NUMERIC.match(tag)
    if not m:
        return v_max_for(road_class)
    val = float(m.group(1))
    if val <= 0:
        return v_max_for(road_class)
    unit = (m.group(2) or "km/h").lower()
    if unit == "mph":
        return val * 1609.344 / 3600.0
    return val * 1000.0 / 3600.0


def _ls_length_m(line: LineString) -> float:
    """Metric length of a (lon, lat) LineString via haversine sum."""
    coords = np.asarray(line.coords)
    if len(coords) < 2:
        return 0.0
    lons, lats = coords[:, 0], coords[:, 1]
    return float(np.sum(haversine_m(lats[:-1], lons[:-1], lats[1:], lons[1:])))


def _build_arrays(
    records: list[tuple[int, str, float, LineString, int, int, float]],
) -> tuple[np.ndarray, ...]:
    return (
        np.array([r[0] for r in records], dtype="int64"),
        np.array([r[1] for r in records], dtype=object),
        np.array([r[2] for r in records], dtype=float),
        np.array([r[3] for r in records], dtype=object),
        np.array([r[4] for r in records], dtype="int64"),
        np.array([r[5] for r in records], dtype="int64"),
        np.array([r[6] for r in records], dtype=float),
    )


def _build_adjacency(
    from_node: np.ndarray,
) -> dict[NodeId, list[EdgeIdx]]:
    adj: dict[NodeId, list[EdgeIdx]] = defaultdict(list)
    for e, n in enumerate(from_node):
        adj[int(n)].append(e)
    return dict(adj)


def _build_node_positions(
    geoms: np.ndarray, from_node: np.ndarray, to_node: np.ndarray,
) -> dict[NodeId, tuple[float, float]]:
    """Extract `NodeId → (lat, lon)` from edge endpoints.

    Each node's position comes from the endpoint of any edge that touches it.
    OSM guarantees consistency, so multiple edges meeting at a node yield the
    same coordinates; later writes harmlessly reaffirm earlier ones.
    """
    positions: dict[NodeId, tuple[float, float]] = {}
    for i in range(len(geoms)):
        coords = list(geoms[i].coords)
        if not coords:
            continue
        # Geometry is in (lon, lat) order; we store (lat, lon).
        positions[int(from_node[i])] = (coords[0][1], coords[0][0])
        positions[int(to_node[i])] = (coords[-1][1], coords[-1][0])
    return positions


def build_network_from_records(
    records: list[tuple[int, str, float, LineString, int, int, float]],
    *,
    signal_node_ids: set[int] | None = None,
    stop_node_ids: set[int] | None = None,
) -> RoadNetwork:
    """Assemble a `RoadNetwork` from pre-parsed records.

    Each record:
        `(edge_id, road_class, max_speed_ms, geom, from_node, to_node, length_m)`.

    `signal_node_ids` / `stop_node_ids` are optional sets of OSM node IDs
    tagged `highway=traffic_signals` / `highway=stop`. When supplied, per-edge
    flags `to_node_is_signal` / `to_node_is_stop` are populated by
    membership lookup; when None, both default to all zeros (which is the
    behaviour for synthetic test networks).

    Primary use: tests and cache reloads.
    """
    edge_ids, road_classes, max_speeds_ms, geoms, from_node, to_node, lengths_m = (
        _build_arrays(records)
    )
    typical_speeds_ms = np.array(
        [v_typical_for(str(rc)) for rc in road_classes], dtype=float,
    )
    n_edges = len(edge_ids)
    if signal_node_ids is None:
        to_node_is_signal = np.zeros(n_edges, dtype=np.int8)
    else:
        to_node_is_signal = np.fromiter(
            (1 if int(to_node[e]) in signal_node_ids else 0 for e in range(n_edges)),
            dtype=np.int8, count=n_edges,
        )
    if stop_node_ids is None:
        to_node_is_stop = np.zeros(n_edges, dtype=np.int8)
    else:
        to_node_is_stop = np.fromiter(
            (1 if int(to_node[e]) in stop_node_ids else 0 for e in range(n_edges)),
            dtype=np.int8, count=n_edges,
        )
    # Total degree per node = incoming + outgoing edge count. A node with
    # degree ≥ 3 is a real intersection (multi-way junction). Degree 2
    # is just a through-segment OSM split.
    node_total_degree: dict[int, int] = {}
    for e in range(n_edges):
        u = int(from_node[e])
        v = int(to_node[e])
        node_total_degree[u] = node_total_degree.get(u, 0) + 1
        node_total_degree[v] = node_total_degree.get(v, 0) + 1
    return RoadNetwork(
        edge_ids=edge_ids,
        geoms=geoms,
        lengths_m=lengths_m,
        road_classes=road_classes,
        max_speeds_ms=max_speeds_ms,
        typical_speeds_ms=typical_speeds_ms,
        from_node=from_node,
        to_node=to_node,
        adjacency=_build_adjacency(from_node),
        turn_restrictions=set(),
        node_positions=_build_node_positions(geoms, from_node, to_node),
        tree=STRtree(list(geoms)),
        to_node_is_signal=to_node_is_signal,
        to_node_is_stop=to_node_is_stop,
        node_total_degree=node_total_degree,
    )


_MOTORWAY_IMPLICIT_ONEWAY: frozenset[str] = frozenset({"motorway", "motorway_link"})
_ONEWAY_YES: frozenset[str] = frozenset({"yes", "true", "1"})
_ONEWAY_REVERSE: frozenset[str] = frozenset({"-1", "reverse"})
_ONEWAY_NO: frozenset[str] = frozenset({"no", "false", "0"})


def _resolve_directions(highway: str, oneway_tag: str) -> tuple[bool, bool]:
    """Map `(highway, oneway)` to `(emit_forward, emit_reverse)`.

    Explicit `oneway` tags always win — including `oneway=no` on a
    `highway=motorway` (real Mexican OSM uses this for undivided highways
    that mappers want classified as motorway-class for speed but where
    traffic flows both ways). Unknown values (`alternating`, `reversible`)
    fall back to two-way: better to admit a few spurious paths the driver
    model can penalise than to miss a legitimate route.
    """
    tag = (oneway_tag or "").lower()
    if tag in _ONEWAY_REVERSE:
        return False, True
    if tag in _ONEWAY_YES:
        return True, False
    if tag in _ONEWAY_NO:
        return True, True
    # No explicit tag: apply highway-class default.
    if not tag and highway in _MOTORWAY_IMPLICIT_ONEWAY:
        return True, False
    # Unknown tag value, or no tag and non-motorway → two-way.
    return True, True


def _parse_pbf(
    pbf_path: Path,
) -> list[tuple[int, str, float, LineString, int, int, float]]:
    """Stream-parse a PBF and return routable edge records.

    Two passes:
        1. Count node references in drivable ways (intersection detection).
        2. Walk each drivable way; split at every node referenced ≥ 2 times,
           and emit one or two edges per segment depending on the way's
           directionality:
               oneway=yes  / motorway implicit  → forward edge only
               oneway=-1   / reverse            → reverse edge only
               oneway=no / unknown / unset on non-motorway → both directions

    Edge ids: original `way_id` for the first emitted segment in the
    canonical direction; synthetic ids (≥ 10**12) for additional split
    segments and for the reverse twins of two-way roads. Synthetic ids
    are deterministic given PBF iteration order.
    """
    import osmium

    node_refs: dict[int, int] = defaultdict(int)

    class _NodeCounter(osmium.SimpleHandler):  # type: ignore[misc]
        def way(self, w: Any) -> None:
            if w.tags.get("highway") not in DRIVABLE_HIGHWAY:
                return
            for n in w.nodes:
                node_refs[n.ref] += 1

    counter = _NodeCounter()
    counter.apply_file(str(pbf_path))

    records: list[tuple[int, str, float, LineString, int, int, float]] = []
    next_synth_id = [10 ** 12]

    class _EdgeBuilder(osmium.SimpleHandler):  # type: ignore[misc]
        def way(self, w: Any) -> None:
            hwy = w.tags.get("highway")
            if hwy not in DRIVABLE_HIGHWAY:
                return
            try:
                way_nodes = [
                    (n.ref, float(n.lon), float(n.lat))
                    for n in w.nodes if n.location.valid()
                ]
            except Exception as exc:
                logger.debug("skipping malformed OSM way id=%s: %s", w.id, exc)
                return
            if len(way_nodes) < 2:
                return
            max_speed_ms = _parse_maxspeed_ms(w.tags.get("maxspeed"), str(hwy))
            way_id = int(w.id)
            emit_fwd, emit_rev = _resolve_directions(
                str(hwy), w.tags.get("oneway", ""),
            )
            # The first emitted edge of the way (in either direction) inherits
            # `way_id`; everything else (additional split segments, reverse
            # twins) gets a synthetic id.
            canonical_used = False

            split_idxs = [0]
            for i in range(1, len(way_nodes) - 1):
                if node_refs[way_nodes[i][0]] >= 2:
                    split_idxs.append(i)
            split_idxs.append(len(way_nodes) - 1)

            for s, e in zip(split_idxs, split_idxs[1:]):
                segment = way_nodes[s:e + 1]
                from_n = segment[0][0]
                to_n = segment[-1][0]
                coords = [(lon, lat) for _, lon, lat in segment]
                geom = LineString(coords)
                length_m = _ls_length_m(geom)

                if emit_fwd:
                    if not canonical_used:
                        edge_id = way_id
                        canonical_used = True
                    else:
                        edge_id = next_synth_id[0]
                        next_synth_id[0] += 1
                    records.append(
                        (edge_id, str(hwy), max_speed_ms, geom,
                         from_n, to_n, length_m),
                    )

                if emit_rev:
                    rev_geom = LineString(list(reversed(coords)))
                    if not canonical_used:
                        # `oneway=-1`: the reverse direction is canonical.
                        edge_id = way_id
                        canonical_used = True
                    else:
                        edge_id = next_synth_id[0]
                        next_synth_id[0] += 1
                    records.append(
                        (edge_id, str(hwy), max_speed_ms, rev_geom,
                         to_n, from_n, length_m),
                    )

    builder = _EdgeBuilder()
    builder.apply_file(str(pbf_path), locations=True, idx="flex_mem")
    return records


def _parse_signal_stop_node_ids_from_pbf(
    pbf_path: Path,
) -> tuple[set[int], set[int]]:
    """Extract OSM node IDs tagged `highway=traffic_signals` and
    `highway=stop`. Both are commonly placed on the intersection node where
    the control applies; vehicles entering an edge whose `to_node` is one
    of these IDs traverse a signal/stop.
    """
    import osmium

    signal_ids: set[int] = set()
    stop_ids:   set[int] = set()

    class _NodeTagHandler(osmium.SimpleHandler):  # type: ignore[misc]
        def node(self, n: Any) -> None:
            hwy = n.tags.get("highway")
            if hwy == "traffic_signals":
                signal_ids.add(int(n.id))
            elif hwy == "stop":
                stop_ids.add(int(n.id))

    _NodeTagHandler().apply_file(str(pbf_path))
    return signal_ids, stop_ids


def _signal_stop_cache_path(edges_cache_path: Path) -> Path:
    """`pt_edges.parquet` → `pt_signal_stop_nodes.parquet` sidecar."""
    return edges_cache_path.with_name(
        edges_cache_path.stem.replace("_edges", "") + "_signal_stop_nodes.parquet",
    )


def _load_or_build_signal_stop_node_ids(
    pbf_path: Path, edges_cache_path: Path | None,
) -> tuple[set[int], set[int]]:
    """Return (signal_ids, stop_ids), reading the sidecar parquet when
    available and freshly parsing the PBF otherwise.
    """
    sidecar = (
        _signal_stop_cache_path(edges_cache_path)
        if edges_cache_path is not None else None
    )
    if sidecar is not None and sidecar.exists():
        df = pd.read_parquet(sidecar)
        signals = set(df.loc[df["kind"] == "signal", "node_id"].astype("int64").tolist())
        stops   = set(df.loc[df["kind"] == "stop",   "node_id"].astype("int64").tolist())
        return signals, stops

    signals, stops = _parse_signal_stop_node_ids_from_pbf(pbf_path)
    if sidecar is not None:
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({
            "kind": ["signal"] * len(signals) + ["stop"] * len(stops),
            "node_id": list(signals) + list(stops),
        }).to_parquet(sidecar, index=False)
    return signals, stops


def load_osm_network(
    pbf_path: Path,
    cache_path: Path | None = None,
) -> RoadNetwork:
    """Load a `RoadNetwork` from a PBF, with a parquet cache of parsed edges.

    `cache_path` exists  → load cached records (fast).
    `cache_path` set, missing → parse PBF, write cache, return.
    `cache_path` is None → parse PBF, no caching.

    A sidecar parquet at `cache_path.parent / <prefix>_signal_stop_nodes.parquet`
    is used (and created on demand) for OSM `highway=traffic_signals` and
    `highway=stop` node IDs. Existing edges caches without a sidecar
    trigger a one-time PBF re-parse on first load.
    """
    if cache_path is not None and cache_path.exists():
        df = pd.read_parquet(cache_path)
        geoms = shapely.from_wkt(df["wkt"].to_numpy())
        records = list(zip(
            df["edge_id"].astype("int64"),
            df["road_class"].astype(object),
            df["max_speed_ms"].astype(float),
            list(geoms),
            df["from_node"].astype("int64"),
            df["to_node"].astype("int64"),
            df["length_m"].astype(float),
        ))
        signals, stops = _load_or_build_signal_stop_node_ids(pbf_path, cache_path)
        return build_network_from_records(
            records, signal_node_ids=signals, stop_node_ids=stops,
        )

    records = _parse_pbf(pbf_path)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({
            "edge_id": [r[0] for r in records],
            "road_class": [r[1] for r in records],
            "max_speed_ms": [r[2] for r in records],
            "wkt": [r[3].wkt for r in records],
            "from_node": [r[4] for r in records],
            "to_node": [r[5] for r in records],
            "length_m": [r[6] for r in records],
        }).to_parquet(cache_path, index=False)

    signals, stops = _load_or_build_signal_stop_node_ids(pbf_path, cache_path)
    return build_network_from_records(
        records, signal_node_ids=signals, stop_node_ids=stops,
    )
