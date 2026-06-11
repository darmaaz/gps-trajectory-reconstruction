"""A* shortest path and penalty-diversified K paths over a `RoadNetwork`.

Two public entry points (SPEC.md §network.routing):

    `shortest_travel_time(network, p_from, p_to, max_speed_factor)`
        — A* on the node graph, accounting for partial-edge segments at the
          source (projection → `to_node`) and destination (`from_node` →
          projection). Used by stale-jump detection.

    `candidate_paths(src_states, dst_states, network, time_budget, max_paths,
                     k_per_pair=3)`
        — Penalty-diversified K paths (the Plateau / Penalty method) between
          every (src_state, dst_state) pair: each accepted path's edges are
          multiplicatively surcharged so subsequent searches prefer
          structurally different alternatives. Pruned by the time budget,
          deduplicated by edge sequence, sorted by expected travel time,
          capped at `max_paths`. Used by
          `path_candidates.enumerate_paths_per_transition`.

NetworkX is the algorithmic backend. The translation from `RoadNetwork`
(parallel-array data layout) to `nx.DiGraph` is computed once per
`RoadNetwork` and cached on the object — second call onward is free.

Multi-edge handling: a `DiGraph` keeps one edge per (u, v); when a road
network has parallel edges between the same node pair (rare for properly
split ways), the lower-cost edge wins. If parallel-edge support proves
necessary, swap to `MultiDiGraph` and replace the iterative shortest-path
call with a multi-graph K-shortest implementation.

Turn restrictions: `network.turn_restrictions` is currently always empty
(parsing not implemented). Once populated, restrictions can be enforced
either by graph rewriting (split nodes, forbid the disallowed edge pair) or
by a custom successor filter in the iterative search. Documented here so
the path is clear when restrictions land.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import networkx as nx
import numpy as np

from ..geo import equirectangular_distance_m
from ..model import Path, State
from .identity import truncate_with_route_diversity

if TYPE_CHECKING:
    from .loader import EdgeIdx, NodeId, RoadNetwork

# Cap on the number of candidate paths considered per (src, dst) pair before
# the cross-pair dedup-and-sort. 3 is a defensible default — captures the
# obvious primary + alternative, lets the cross-pair top-K from |src|×|dst|
# state pairs add additional diversity, and keeps the iterative search cheap.
DEFAULT_K_PER_PAIR: int = 3

# Upper bound speed used by the A* heuristic; admissibility requires this
# bound be ≥ any actual edge speed so the heuristic never overestimates time.
# Picked conservatively above any per-class V_MAX in `config.V_MAX_MS`.
_MAX_NETWORK_SPEED_MS: float = 60.0


def _build_nx_graph(network: "RoadNetwork") -> nx.DiGraph:
    """Build a `DiGraph` mirror of the road network for routing.

    Edge attributes:
        weight               — expected traversal time in seconds using
                               `typical_speeds_ms` (real-driving prior).
                               Used by path-enumeration cost.
        feasibility_weight   — minimum-feasible traversal time in seconds
                               using `max_speeds_ms`. Used by
                               `shortest_travel_time` (stale-jump detection
                               needs a lower bound, not a typical estimate).
        edge_idx             — index into the parallel arrays on `network`

    Edges with non-positive length or speed are skipped — these arise from
    malformed OSM data (degenerate geometries) and have no meaningful
    routing semantics. An edge is included only when *both* `weight` and
    `feasibility_weight` are well-defined.

    Parallel edges between the same node pair are collapsed to the
    lower-`weight` one (routing-cost-optimal); both attributes come from
    that surviving edge.
    """
    G: nx.DiGraph = nx.DiGraph()
    e_count = len(network)
    lengths = network.lengths_m
    typical = network.typical_speeds_ms
    maxs = network.max_speeds_ms
    fr = network.from_node
    to = network.to_node
    for e in range(e_count):
        length = float(lengths[e])
        t_speed = float(typical[e])
        m_speed = float(maxs[e])
        if length <= 0.0 or t_speed <= 0.0 or m_speed <= 0.0:
            continue
        u = int(fr[e])
        v = int(to[e])
        w = length / t_speed
        fw = length / m_speed
        existing = G.get_edge_data(u, v)
        if existing is None or w < existing["weight"]:
            G.add_edge(u, v, weight=w, feasibility_weight=fw, edge_idx=e)
    return G


def _get_nx_graph(network: "RoadNetwork") -> nx.DiGraph:
    """Return a cached `DiGraph` for `network`, building on first call."""
    cached = network._nx_graph_cache
    if cached is not None:
        return cached  # type: ignore[return-value]
    G = _build_nx_graph(network)
    network._nx_graph_cache = G
    return G


def _get_permissive_graph(
    network: "RoadNetwork", cost_factor: float,
) -> nx.DiGraph:
    """Legal routing graph plus penalized reverse arcs for one-way edges.

    For every edge with no opposite-direction twin (`twin_indices() == -1`,
    i.e. a mapped one-way), adds the arc `(to_node, from_node)` with
        weight   = legal weight × `cost_factor`   (search/diversification cost)
        illegal  = True                            (traversal is reversed)
        edge_idx = the underlying edge
    Two-way edges get nothing — their reverse direction already exists as a
    real twin edge. A reverse arc is skipped when a real arc `(v, u)`
    already occupies the slot in the DiGraph (parallel-way corner case).

    The inflated weight steers enumeration away from violations without
    forbidding them; physical admission stays on `min_traversal_time`
    computed from the arrays, and plausibility is priced by the
    `n_direction_violations` feature (μ slot [18]), not by this factor.

    Cached per `(network, cost_factor)`; invalidated with the legal graph.
    """
    cached = network._nx_graph_permissive_cache
    if cached is not None and cached[0] == cost_factor:  # type: ignore[index]
        return cached[1]  # type: ignore[index,return-value]
    G = _get_nx_graph(network).copy()
    twins = network.twin_indices()
    fr = network.from_node
    to = network.to_node
    for e in range(len(network)):
        if twins[e] != -1:
            continue
        u, v = int(fr[e]), int(to[e])
        if u == v or not G.has_edge(u, v):
            continue    # degenerate, or edge was skipped/collapsed in build
        if G.has_edge(v, u):
            continue    # real reverse arc exists (parallel ways); keep legal
        data = G[u][v]
        if data["edge_idx"] != e:
            continue    # parallel-edge collapse kept a different edge
        G.add_edge(
            v, u,
            weight=data["weight"] * cost_factor,
            feasibility_weight=data["feasibility_weight"],
            edge_idx=e,
            illegal=True,
        )
    network._nx_graph_permissive_cache = (cost_factor, G)
    return G


def _make_heuristic(network: "RoadNetwork", target_node: "NodeId"):
    """A* heuristic: euclidean distance to target / max network speed.

    Admissible because no edge can be traversed faster than
    `_MAX_NETWORK_SPEED_MS`, so the heuristic is always a lower bound on
    remaining travel time.
    """
    target_lat, target_lon = network.node_positions[target_node]

    def h(node: "NodeId", _ignored: "NodeId") -> float:
        lat, lon = network.node_positions[node]
        return float(
            equirectangular_distance_m(lat, lon, target_lat, target_lon)
            / _MAX_NETWORK_SPEED_MS,
        )

    return h


def _project_or_none(
    network: "RoadNetwork", lat: float, lon: float,
    *, radius_meters: float = 200.0,
) -> tuple[int, float] | None:
    """Best edge projection within radius, or None. Returns
    `(edge_idx, offset_m)`."""
    hits = network.project_point(lat, lon, radius_meters, max_candidates=1)
    if not hits:
        return None
    idx, offset_m, _ = hits[0]
    return idx, offset_m


def shortest_travel_time(
    network: "RoadNetwork",
    from_lat: float, from_lon: float,
    to_lat: float, to_lon: float,
    max_speed_factor: float = 1.0,
) -> float:
    """A* lower-bound travel time in seconds between two points on the
    network. Returns `inf` if either endpoint can't be projected within
    200 m of the network or the destination is unreachable from the source.

    Cost decomposition:
        - From the projection point on the source edge to that edge's
          `to_node`: `(length - src_offset) / (max_speed * factor)`
        - A* shortest path from source `to_node` to destination `from_node`,
          using per-edge `length / (max_speed * factor)` weights.
        - From destination `from_node` to its projection point:
          `dst_offset / (max_speed * factor)`

    `max_speed_factor` ≥ 1 inflates allowable speeds, lowering the bound
    (used by stale-jump detection to add slack so aggressive-but-feasible
    driving doesn't false-positive flag).
    """
    src = _project_or_none(network, from_lat, from_lon)
    dst = _project_or_none(network, to_lat, to_lon)
    if src is None or dst is None:
        return float("inf")

    src_idx, src_offset = src
    dst_idx, dst_offset = dst
    factor = max_speed_factor

    # Same edge, forward direction along it.
    if src_idx == dst_idx and dst_offset >= src_offset:
        speed = network.max_speeds_ms[src_idx] * factor
        return float((dst_offset - src_offset) / speed)

    src_to = int(network.to_node[src_idx])
    dst_from = int(network.from_node[dst_idx])
    src_partial = (network.lengths_m[src_idx] - src_offset) / (
        network.max_speeds_ms[src_idx] * factor
    )
    dst_partial = dst_offset / (network.max_speeds_ms[dst_idx] * factor)

    if src_to == dst_from:
        return float(src_partial + dst_partial)

    G = _get_nx_graph(network)
    if src_to not in G or dst_from not in G:
        return float("inf")

    # `shortest_travel_time` is a feasibility lower bound — use the
    # `feasibility_weight` edge attribute (length / max_speed), not the
    # routing-cost `weight` (length / typical_speed). Stored alongside on
    # each edge by `_build_nx_graph`; rescaled by `factor` so we don't
    # rebuild the graph per call when `factor` varies.
    try:
        nominal = nx.astar_path_length(
            G, src_to, dst_from,
            heuristic=_make_heuristic(network, dst_from),
            weight="feasibility_weight",
        )
    except nx.NetworkXNoPath:
        return float("inf")

    middle = nominal / factor
    return float(src_partial + middle + dst_partial)


def _path_weight(G: nx.DiGraph, node_path: list["NodeId"]) -> float:
    return sum(G[u][v]["weight"] for u, v in zip(node_path, node_path[1:]))


def _node_path_to_edges_with_direction(
    G: nx.DiGraph, node_path: list["NodeId"],
) -> tuple[list["EdgeIdx"], list[bool]]:
    """Edge indices plus per-edge reverse-traversal flags (permissive
    graphs mark wrong-way arcs with `illegal=True`)."""
    idxs: list["EdgeIdx"] = []
    revs: list[bool] = []
    for u, v in zip(node_path, node_path[1:]):
        data = G[u][v]
        idxs.append(data["edge_idx"])
        revs.append(bool(data.get("illegal", False)))
    return idxs, revs


def _penalty_diversified_paths(
    G: nx.DiGraph,
    src: "NodeId",
    dst: "NodeId",
    middle_budget: float,
    k_max: int,
    lambda_: float,
) -> list[list["NodeId"]]:
    """Iterative shortest-path with multiplicative edge penalty.

    Each accepted path multiplies its edges' penalties by `(1 + lambda_)`,
    so subsequent calls naturally avoid the already-used edges and produce
    structurally diverse alternatives. `lambda_=0.0` reduces to the same
    shortest path returned every iteration (no diversity).

    Lax cost cap: budget is checked against unpenalised travel time, so the
    penalty is purely a diversification mechanism — not a feasibility filter.
    """
    penalty: dict[tuple["NodeId", "NodeId"], float] = {}

    def weight_fn(u, v, edge_data):
        base = edge_data["weight"]
        return base * penalty.get((u, v), 1.0)

    out: list[list["NodeId"]] = []
    while len(out) < k_max:
        try:
            node_path = nx.shortest_path(G, src, dst, weight=weight_fn)
        except nx.NetworkXNoPath:
            break
        unpenalised = _path_weight(G, node_path)
        if unpenalised > middle_budget:
            break
        out.append(node_path)
        for u, v in zip(node_path, node_path[1:]):
            penalty[(u, v)] = penalty.get((u, v), 1.0) * (1.0 + lambda_)
    return out


def _build_path(
    network: "RoadNetwork",
    src_state: State,
    dst_state: State,
    middle_edge_idxs: list["EdgeIdx"],
    expected_travel_time: float,
    time_budget: float,
    min_traversal_time: float,
    *,
    middle_reversed: list[bool] | None = None,
    src_reversed: bool = False,
    dst_reversed: bool = False,
    src_backroll: bool = False,
) -> Path:
    """Assemble a `Path` dataclass from a routed edge sequence.

    The path's edge tuple is the source edge id (where the trip starts at
    `src_state.offset`), then the routed middle edges, then the destination
    edge id (ending at `dst_state.offset`). For the same-edge no-traversal
    case, the tuple collapses to a single edge id.

    `expected_travel_time` uses typical_speeds (the realistic-driving
    prior used for the CRF likelihood and dwell residual).
    `min_traversal_time` uses max_speeds (the physical lower bound on
    transit time, used for admission filtering).

    Direction-violation paths (`Config.enable_direction_violation`):
    `middle_reversed[i]` marks middle edge i as a wrong-way traversal;
    `src_reversed` means the source edge is exited backward via its
    `from_node` (span = `start_offset`); `dst_reversed` means the
    destination edge is entered backward via its `to_node`
    (span = `length − end_offset`). `src_backroll` marks the same-edge
    backward traversal (`dst.offset < src.offset` driven in reverse,
    span = `src.offset − dst.offset`). All default to the legal case,
    which produces `reversed_mask=None`.
    """
    src_idx_arr = network.edge_ids
    src_link = src_state.link_id
    dst_link = dst_state.link_id

    if not middle_edge_idxs and src_link == dst_link:
        edges: tuple[int, ...] = (src_link,)
        if src_backroll:
            # Same-edge wrong-way roll: real backward motion, not jitter.
            length_m = max(0.0, src_state.offset - dst_state.offset)
        else:
            # Backward case (dst.offset < src.offset) deliberately keeps
            # start_offset > end_offset so the path's `ends_at(dst_state)`
            # check passes — that's how this stay path attaches to the
            # (i, j) cell. Length is clamped to 0 for honest zero-motion
            # physics.
            length_m = max(0.0, dst_state.offset - src_state.offset)
        mask = (True,) if src_backroll else None
    else:
        middle_links = tuple(int(src_idx_arr[i]) for i in middle_edge_idxs)
        edges = (src_link, *middle_links, dst_link)
        src_internal_idx = _edge_index_for_link(network, src_link)
        dst_internal_idx = _edge_index_for_link(network, dst_link)
        src_span = (
            src_state.offset if src_reversed
            else network.lengths_m[src_internal_idx] - src_state.offset
        )
        dst_span = (
            network.lengths_m[dst_internal_idx] - dst_state.offset
            if dst_reversed else dst_state.offset
        )
        middle_total = float(sum(network.lengths_m[i] for i in middle_edge_idxs))
        length_m = float(src_span + middle_total + dst_span)
        mids = middle_reversed or [False] * len(middle_edge_idxs)
        if src_reversed or dst_reversed or any(mids):
            mask = (src_reversed, *mids, dst_reversed)
        else:
            mask = None

    return Path(
        edges=edges,
        start_offset=src_state.offset,
        end_offset=dst_state.offset,
        expected_travel_time=expected_travel_time,
        length_meters=length_m,
        feature_vector=np.zeros(0, dtype=float),    # populated by features.path_features
        time_budget=time_budget,
        start_perp_m=float(getattr(src_state, "perp_m", 0.0)),
        end_perp_m=float(getattr(dst_state, "perp_m", 0.0)),
        min_traversal_time=min_traversal_time,
        reversed_mask=mask,
    )


def _edge_index_for_link(network: "RoadNetwork", link_id: int) -> int:
    """Cached `link_id → EdgeIdx` lookup, delegating to `RoadNetwork`."""
    return network.edge_index_for_link(link_id)


def _paths_between(
    src_state: State,
    dst_state: State,
    network: "RoadNetwork",
    time_budget: float,
    k_per_pair: int,
    G: nx.DiGraph,
    *,
    penalty_lambda: float = 0.3,
    actual_budget: float | None = None,
    allow_violation: bool = False,
) -> list[Path]:
    """K time-feasible paths between a single (src, dst) state pair.

    Same-edge and adjacent-node cases short-circuit to a single path each.
    Multi-edge case uses penalty-diversified iterative shortest-path:
    `penalty_lambda` is the multiplicative surcharge per edge re-use,
    applied as `weight *= (1 + penalty_lambda)`.

    `time_budget` is the enumeration cap (slack-inflated by the caller).
    `actual_budget` is the unslacked transit budget stored on each `Path`
    for `inferred_dwell` semantics; defaults to `time_budget` when the
    caller doesn't distinguish.

    `allow_violation`: `G` is the permissive graph (wrong-way arcs on
    one-way edges, marked `illegal`) and the terminal edges may also be
    traversed against their direction — a one-way source edge can be
    exited backward via its `from_node`, a one-way destination edge
    entered backward via its `to_node`, and a same-edge pair with
    backward offsets on a one-way gets a real reverse-roll path alongside
    the zero-motion stay. Two-way terminals never get backward variants
    (their reverse twin is a separate projection candidate already).
    Resulting paths carry `reversed_mask`; the `n_direction_violations`
    feature prices them.
    """
    if actual_budget is None:
        actual_budget = time_budget
    src_idx = _edge_index_for_link(network, src_state.link_id)
    dst_idx = _edge_index_for_link(network, dst_state.link_id)

    # Two parallel travel-time accountings per path:
    #   typical_speeds  → `expected_travel_time` (CRF feature, dwell residual)
    #   max_speeds      → `min_traversal_time`   (physical lower bound)
    # Admission filters on `min_traversal_time` (anything physically
    # possible) while the typical-speed estimate feeds the likelihood
    # via `μᵀϕ(p)` and `inferred_dwell` arithmetic.
    src_typ = network.typical_speeds_ms[src_idx]
    dst_typ = network.typical_speeds_ms[dst_idx]
    src_max = network.max_speeds_ms[src_idx]
    dst_max = network.max_speeds_ms[dst_idx]

    twins = network.twin_indices() if allow_violation else None

    # Same-edge: vehicle stayed on this edge. Forward offsets are a partial
    # traversal; backward offsets are zero-motion (GPS along-track jitter at
    # near-stop). `_build_path` clamps `length_m` to `max(0, dst-src)`, so
    # the path correctly represents zero motion in the backward case.
    # Under `allow_violation`, backward offsets on a ONE-WAY edge also get
    # a genuine reverse-roll path (parking-lot pull-out / mid-edge U-turn
    # signature) alongside the stay — the posterior decides which story
    # the evidence supports.
    if src_idx == dst_idx:
        out_same: list[Path] = []
        delta_m = max(0.0, dst_state.offset - src_state.offset)
        ttime_typ = 0.0 if delta_m == 0.0 else delta_m / src_typ
        ttime_min = 0.0 if delta_m == 0.0 else delta_m / src_max
        if ttime_min <= time_budget:
            out_same.append(_build_path(
                network, src_state, dst_state, [],
                ttime_typ, actual_budget, ttime_min,
            ))
        back_m = src_state.offset - dst_state.offset
        if (
            twins is not None and twins[src_idx] == -1 and back_m > 0.0
        ):
            btyp = back_m / src_typ
            bmin = back_m / src_max
            if bmin <= time_budget:
                out_same.append(_build_path(
                    network, src_state, dst_state, [],
                    btyp, actual_budget, bmin, src_backroll=True,
                ))
        return out_same

    src_len = float(network.lengths_m[src_idx])
    dst_len = float(network.lengths_m[dst_idx])

    # Terminal traversal options: (boundary_node, span_m, reversed).
    # Legal: exit src via to_node, enter dst via from_node. Under
    # violation, a ONE-WAY terminal may also be traversed backward —
    # two-way terminals are skipped (their reverse twin is a separate
    # projection candidate; a backward variant would only mint a
    # duplicate spelling).
    exits: list[tuple[int, float, bool]] = [
        (int(network.to_node[src_idx]), src_len - src_state.offset, False),
    ]
    entries: list[tuple[int, float, bool]] = [
        (int(network.from_node[dst_idx]), dst_state.offset, False),
    ]
    if twins is not None:
        if twins[src_idx] == -1:
            exits.append((int(network.from_node[src_idx]), src_state.offset, True))
        if twins[dst_idx] == -1:
            entries.append((int(network.to_node[dst_idx]), dst_len - dst_state.offset, True))

    out: list[Path] = []
    for exit_node, src_span, src_rev in exits:
        for entry_node, dst_span, dst_rev in entries:
            fixed_overhead_typ = src_span / src_typ + dst_span / dst_typ
            fixed_overhead_min = src_span / src_max + dst_span / dst_max
            if fixed_overhead_min > time_budget:
                continue

            if exit_node == entry_node:
                out.append(_build_path(
                    network, src_state, dst_state, [],
                    fixed_overhead_typ, actual_budget, fixed_overhead_min,
                    src_reversed=src_rev, dst_reversed=dst_rev,
                ))
                continue

            if exit_node not in G or entry_node not in G:
                continue

            # Enumerate diverse paths under typical-speed graph cost. Cap
            # the internal search at a generous multiple of the max-speed-
            # based admission budget so we don't miss paths that look slow
            # under typical-speed but are feasible under max-speed
            # (residential roads have max/typical ratio up to ~2.0; 3×
            # covers that plus headroom).
            enum_cap_typ = (time_budget - fixed_overhead_min) * 3.0
            middle_budget = max(enum_cap_typ, 0.1)
            node_paths = _penalty_diversified_paths(
                G, exit_node, entry_node, middle_budget, k_per_pair,
                penalty_lambda,
            )
            for node_path in node_paths:
                edge_idxs, middle_rev = _node_path_to_edges_with_direction(
                    G, node_path,
                )
                # Typical/max-speed times from the arrays (direction-
                # symmetric), NOT graph weights — permissive-graph weights
                # carry the violation cost factor, which is a search
                # steering device, not a travel-time estimate.
                middle_time_typ = sum(
                    float(network.lengths_m[i]) / float(network.typical_speeds_ms[i])
                    for i in edge_idxs
                )
                middle_time_min = sum(
                    float(network.lengths_m[i]) / float(network.max_speeds_ms[i])
                    for i in edge_idxs
                )
                total_min = fixed_overhead_min + middle_time_min
                if total_min > time_budget:
                    continue
                out.append(_build_path(
                    network, src_state, dst_state, edge_idxs,
                    fixed_overhead_typ + middle_time_typ, actual_budget,
                    total_min,
                    middle_reversed=middle_rev,
                    src_reversed=src_rev, dst_reversed=dst_rev,
                ))
    return out


def _offroad_path(
    network: "RoadNetwork",
    src_state: State,
    dst_state: State,
    straight_m: float,
    actual_budget: float,
) -> Path:
    """Build a straight-line off-road candidate between two disconnected
    projected edges.

    Represents an off-network maneuver (parking, arrival, idling near a
    one-way pair) that legal routing can't model. `edges = (src, dst)`
    with NO topological adjacency — consumers must branch on
    `is_off_road`. `length` is the crow-flight distance; travel time uses
    a slow maneuver speed so the residual budget becomes a large inferred
    dwell. `min_traversal_time` is tiny (a straight-line move is always
    physically admissible).
    """
    # Maneuver speed: the lower of the two endpoint typical speeds — an
    # off-network creep is slow regardless of the road class nearby.
    src_idx = _edge_index_for_link(network, src_state.link_id)
    dst_idx = _edge_index_for_link(network, dst_state.link_id)
    maneuver_speed = max(
        1.0, min(
            float(network.typical_speeds_ms[src_idx]),
            float(network.typical_speeds_ms[dst_idx]),
        ),
    )
    ettime = straight_m / maneuver_speed
    return Path(
        edges=(src_state.link_id, dst_state.link_id),
        start_offset=src_state.offset,
        end_offset=dst_state.offset,
        expected_travel_time=ettime,
        length_meters=straight_m,
        feature_vector=np.zeros(0, dtype=float),    # populated by path_features
        time_budget=actual_budget,
        start_perp_m=float(getattr(src_state, "perp_m", 0.0)),
        end_perp_m=float(getattr(dst_state, "perp_m", 0.0)),
        min_traversal_time=straight_m / _MAX_NETWORK_SPEED_MS,
        is_off_road=True,
    )


def candidate_paths(
    src_states: list[State],
    dst_states: list[State],
    network: "RoadNetwork",
    time_budget_seconds: float,
    max_paths: int = 20,
    k_per_pair: int = DEFAULT_K_PER_PAIR,
    *,
    budget_slack: float = 1.0,
    penalty_lambda: float = 0.3,
    enable_offroad: bool = False,
    offroad_max_straight_m: float = 300.0,
    offroad_min_detour_ratio: float = 3.0,
    offroad_min_overslack: float = 1.0,
    diversify_truncation: bool = True,
    enable_direction_violation: bool = False,
    direction_violation_cost_factor: float = 3.0,
) -> list[Path]:
    """Top-K time-feasible paths from any `src_state` to any `dst_state`.

    For each (src, dst) pair, runs penalty-diversified iterative shortest-
    path with a per-pair cap of `k_per_pair`, pruned by
    `time_budget_seconds * budget_slack`. Results
    across pairs are deduplicated by edge tuple (best — i.e. lowest expected
    travel time — wins on ties), sorted by expected travel time ascending,
    and truncated to `max_paths`.

    `diversify_truncation` (default True) spends the `max_paths` cap on
    distinct *physical* routes: the |src|×|dst| state-pair sweep typically
    yields the same corridor under many directed spellings (terminal states
    on opposite-direction twins, neighbouring corridor edges), and plain
    expected-travel-time truncation fills the cap with those spellings while
    crowding out genuinely different routes. The diversified cut keeps the
    best path per `identity.canonical_route` first, then back-fills with the
    best remaining spellings; output stays sorted by expected travel time.
    Set False to recover the legacy behaviour.

    `budget_slack` accommodates the gap between OSM `maxspeed` tags and real
    driving speeds — at 1.5, paths the edge-time model thinks would take up
    to 1.5× the observed gap are still admitted (real drivers covered
    them by exceeding the posted limit). The model's `expected_travel_time`
    on the returned `Path` objects stays at the unscaled edge-based estimate
    so the driver model's feature can still penalise implausible paths.

    `enable_offroad`: when set, a (src, dst) pair whose only routed options
    are long detours (best routed length ≥ `offroad_min_detour_ratio` ×
    straight-line) despite a short straight-line gap (< `offroad_max_straight_m`)
    *also* gets a straight-line off-road candidate (`is_off_road=True`).
    This is purely additive — routed candidates are never removed — so the
    posterior decides between the detour and the maneuver. See
    `_offroad_path` and the diagnostic in `scripts/_diag_offroad.py`.

    `enable_direction_violation`: route on the permissive graph (penalized
    wrong-way arcs on one-way edges) and admit reversed terminal
    traversals — see `_paths_between`. Default off; resulting paths carry
    `reversed_mask` and are priced by the `n_direction_violations`
    feature rather than excluded.

    Returns `[]` when no source-destination combination produces a feasible
    path within the slacked budget — the orchestrator interprets this as a
    transition-level discontinuity and splits the trip per SPEC.md §Edge
    cases.
    """
    if not src_states or not dst_states:
        return []
    G = (
        _get_permissive_graph(network, direction_violation_cost_factor)
        if enable_direction_violation else _get_nx_graph(network)
    )
    # Dedup key includes the traversal-direction mask: a same-edge
    # reverse-roll and the zero-motion stay share an edge tuple but are
    # different physical stories.
    by_edges: dict[tuple, Path] = {}
    # Track, per (src, dst) pair, the shortest routed path's length and the
    # travel time OF THAT shortest path, so the off-road trigger can
    # measure both the detour ratio and whether the detour is overslacked.
    best_routed: dict[tuple[int, int], tuple[float, float]] = {}   # pair → (len, ett)
    effective_budget = time_budget_seconds * budget_slack
    for src in src_states:
        for dst in dst_states:
            for path in _paths_between(
                src, dst, network, effective_budget, k_per_pair, G,
                penalty_lambda=penalty_lambda,
                actual_budget=time_budget_seconds,
                allow_violation=enable_direction_violation,
            ):
                key = (path.edges, path.reversed_mask)
                prev = by_edges.get(key)
                if prev is None or path.expected_travel_time < prev.expected_travel_time:
                    by_edges[key] = path
                pair = (src.link_id, dst.link_id)
                cur = best_routed.get(pair)
                if cur is None or path.length_meters < cur[0]:
                    best_routed[pair] = (path.length_meters, path.expected_travel_time)

    if enable_offroad:
        _add_offroad_candidates(
            src_states, dst_states, network, time_budget_seconds,
            by_edges, best_routed,
            offroad_max_straight_m, offroad_min_detour_ratio,
            offroad_min_overslack,
        )

    ordered = sorted(by_edges.values(), key=lambda p: p.expected_travel_time)
    if diversify_truncation:
        return truncate_with_route_diversity(ordered, network, max_paths)
    return ordered[: max_paths]


def _add_offroad_candidates(
    src_states: list[State],
    dst_states: list[State],
    network: "RoadNetwork",
    actual_budget: float,
    by_edges: dict[tuple, Path],
    best_routed: dict[tuple[int, int], tuple[float, float]],
    max_straight_m: float,
    min_detour_ratio: float,
    min_overslack: float,
) -> None:
    """For each (src, dst) pair that triggers, add a straight-line off-road
    candidate to `by_edges` in place. ALL THREE gates must hold:
      - different edges (same-edge is the stay path's job)
      - short straight-line gap (< max_straight_m)
      - routed detour disproportionate to it (len ≥ min_detour_ratio × straight)
      - routed detour overslacked (ett > min_overslack × budget): the vehicle
        could not have driven the detour in the available time → implausible.
    The overslack gate is what spares genuine short drives through one-way
    loops (which fit the budget and must stay routed)."""
    for src in src_states:
        for dst in dst_states:
            if src.link_id == dst.link_id:
                continue    # same-edge handled by the stay path
            pair = (src.link_id, dst.link_id)
            routed = best_routed.get(pair)
            if routed is None:
                continue    # no routed path for this pair; conservative skip
            routed_len, routed_ett = routed
            ll_s = _state_position_latlon(network, src)
            ll_d = _state_position_latlon(network, dst)
            if ll_s is None or ll_d is None:
                continue
            straight = equirectangular_distance_m(ll_s[0], ll_s[1], ll_d[0], ll_d[1])
            if straight >= max_straight_m:
                continue
            if routed_len < min_detour_ratio * max(straight, 1.0):
                continue
            if routed_ett <= min_overslack * max(actual_budget, 1.0):
                continue    # detour fits the budget → plausibly real travel
            off = _offroad_path(network, src, dst, straight, actual_budget)
            key = (off.edges, off.reversed_mask)
            # Off-road key (src, dst) won't collide with a routed path:
            # a routed 2-edge (src, dst) would require adjacency, which
            # contradicts the detour-ratio trigger. Keep the better
            # (shorter) if a collision somehow occurs.
            prev = by_edges.get(key)
            if prev is None or off.length_meters < prev.length_meters:
                by_edges[key] = off


def _state_position_latlon(
    network: "RoadNetwork", state: State,
) -> tuple[float, float] | None:
    """`(lat, lon)` of a state's projected point on its edge, or None."""
    try:
        idx = _edge_index_for_link(network, state.link_id)
    except KeyError:
        return None
    geom = network.geoms[idx]
    m_len = float(network.lengths_m[idx])
    if geom.length <= 0.0 or m_len <= 0.0:
        return None
    pt = geom.interpolate((state.offset / m_len) * geom.length)
    return float(pt.y), float(pt.x)
