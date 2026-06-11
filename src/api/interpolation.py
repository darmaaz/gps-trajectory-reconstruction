"""Position-at-time helpers.

Three public surfaces, sharing one rule implementation:

- `link_offset_at_fraction(path, network, frac) -> (link_id, offset)` and
  `interpolate_along_path(path, network, frac) -> (lat, lon)`. Pure
  geometry walkers — given a fraction along a path's length, return the
  edge and offset (or the geographic point). No dwell semantics.

- `position_in_transition(path, network, tau_s, rule) -> (link_id, offset)`.
  Resolves the vehicle's position at `tau_s` seconds into a transition
  window, given a dwell allocation `rule`. This is the *single source of
  truth* for how transit-vs-dwell time gets allocated within a transition;
  both `TrajectoryPosterior.at_time` and `position_at_time` delegate here.

- `position_at_time(segments, t, network, rule) -> (lat, lon) | None`.
  Picks the segment whose canonical-time span contains `t` and returns
  the MLE-path's position under `rule`. MLE-only — collapses the path
  posterior to the Viterbi argmax for a single-point answer.

Dwell allocation rules
----------------------
- `"front"` (default): vehicle sits at the path's origin for the dwell
  window, then moves along the path. This is the project's convention.
- `"back"`: vehicle moves along the path immediately, then sits at the
  destination once it arrives.
- `"spread"`: constant speed across the whole budget — no concentrated
  dwell. Equivalent to the historical const-speed baseline.

The CRF inference is identical under all three rules — only intermediate-
time position queries differ. See `OVERVIEW.md` for the modelling stance.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Literal

from ..model import Path as ModelPath, State
from ..model.state import EdgeId

if TYPE_CHECKING:
    from ..network import RoadNetwork
    from .pipeline import TrajectoryPosterior

DwellRule = Literal["front", "back", "spread"]


def _position_on_edge(
    network: "RoadNetwork", edge_idx: int, offset_m: float,
) -> tuple[float, float]:
    """Return `(lat, lon)` at a metric offset along an edge.

    Same degree↔metres scaling as `RoadNetwork.perpendicular_distance`:
    fraction of metric length along the edge maps linearly to the same
    fraction in degree-space along the LineString.
    """
    geom = network.geoms[edge_idx]
    deg_len = geom.length
    m_len = float(network.lengths_m[edge_idx])
    if deg_len <= 0.0 or m_len <= 0.0:
        coords = list(geom.coords)
        return float(coords[0][1]), float(coords[0][0])
    offset_m = max(0.0, min(offset_m, m_len))
    dist_along_deg = (offset_m / m_len) * deg_len
    pt = geom.interpolate(dist_along_deg)
    return float(pt.y), float(pt.x)


def _state_position(
    network: "RoadNetwork", state: State,
) -> tuple[float, float]:
    idx = network.edge_index_for_link(state.link_id)
    return _position_on_edge(network, idx, state.offset)


def link_offset_at_fraction(
    path: ModelPath,
    network: "RoadNetwork",
    frac: float,
) -> tuple[EdgeId, float]:
    """`(link_id, offset_m)` at fraction `frac` along a path's total length.

    `frac=0` returns the path's start (`edges[0]`, `start_offset`); `frac=1`
    returns its end (`edges[-1]`, `end_offset`). Walks edge-by-edge,
    consuming `start_offset` worth of slack on the first edge and stopping
    at `end_offset` on the last. Out-of-range `frac` is clamped to `[0, 1]`.

    This is the underlying walker shared by `interpolate_along_path` (which
    converts to lat/lon) and the dwell-aware `TrajectoryPosterior.at_time`
    (which synthesises `State` objects).
    """
    if not path.edges:
        raise ValueError("cannot interpolate along an empty path")

    frac = max(0.0, min(1.0, frac))
    target = frac * path.length_meters
    edges = path.edges
    n = len(edges)

    # Single-edge path: the entire path lives on edges[0] between
    # start_offset and end_offset, no walking. A reverse-roll
    # (direction-violation backward traversal) walks offsets downward.
    if n == 1:
        if path.edge_reversed(0):
            return edges[0], path.start_offset - target
        return edges[0], path.start_offset + target

    # Multi-edge path. Each edge contributes a directed span
    # (from_offset → to_offset); reversed traversals
    # (`path.reversed_mask`) walk their span downward. Legal paths take
    # the original arithmetic: suffix of the first edge, full middles,
    # prefix of the last.
    edge_0_idx = network.edge_index_for_link(edges[0])
    edge_0_len = float(network.lengths_m[edge_0_idx])
    if path.edge_reversed(0):
        # Exited backward via from_node: span runs start_offset → 0.
        edge_0_remaining = path.start_offset
        if target <= edge_0_remaining:
            return edges[0], path.start_offset - target
    else:
        edge_0_remaining = edge_0_len - path.start_offset
        if target <= edge_0_remaining:
            return edges[0], path.start_offset + target
    target -= edge_0_remaining

    # Middle edges traversed end-to-end (reversed: to_node → from_node).
    for i in range(1, n - 1):
        idx = network.edge_index_for_link(edges[i])
        edge_len = float(network.lengths_m[idx])
        if target <= edge_len:
            return edges[i], (edge_len - target) if path.edge_reversed(i) else target
        target -= edge_len

    # Last edge. Legal: prefix up to end_offset. Reversed: entered via
    # to_node, span runs length → end_offset. Clamp defensively in case
    # rounding pushed `target` slightly past the span.
    if path.edge_reversed(n - 1):
        last_idx = network.edge_index_for_link(edges[-1])
        last_len = float(network.lengths_m[last_idx])
        return edges[-1], max(last_len - target, path.end_offset)
    return edges[-1], min(target, path.end_offset)


def interpolate_along_path(
    path: ModelPath,
    network: "RoadNetwork",
    frac: float,
) -> tuple[float, float]:
    """`(lat, lon)` at fraction `frac` along a path's total length.

    Thin wrapper over `link_offset_at_fraction` that converts the resulting
    `(link_id, offset)` to a position on the edge geometry.
    """
    link_id, offset_m = link_offset_at_fraction(path, network, frac)
    return _position_on_edge(
        network, network.edge_index_for_link(link_id), offset_m,
    )


def position_in_transition(
    path: ModelPath,
    network: "RoadNetwork",
    tau_s: float,
    rule: DwellRule = "front",
) -> tuple[EdgeId, float]:
    """Resolve `(link_id, offset)` at `tau_s` seconds into a transition.

    `tau_s` is measured from the source observation's canonical timestamp.
    The transition's total budget is `path.time_budget`; the path's expected
    transit time is `path.expected_travel_time`; the residual is the
    inferred dwell `D_p = path.time_budget − path.expected_travel_time`.

    `rule` selects how the dwell is allocated within the budget:

    - `"front"`: dwell first (vehicle at origin for τ ≤ D_p), then transit.
    - `"back"`: transit first (vehicle along path for τ < t_p), then dwell
      at destination.
    - `"spread"`: constant speed across the whole budget. Equivalent to the
      historical const-speed baseline.

    Edge cases:
    - `path.expected_travel_time ≤ 0`: the path implies zero motion (a
      same-edge stay path). Returns the path's origin under all rules.
    - `path.time_budget ≤ 0`: degenerate budget (e.g. a default-constructed
      Path without an explicit budget). Returns the path's origin.
    - `D_p < 0` (path enumerated under slack — travel time exceeds budget):
      under "front" the vehicle is treated as always travelling; under
      "back" likewise. "spread" still uses `frac = τ / budget`.
    """
    t_p = path.expected_travel_time
    B = path.time_budget

    # Off-road candidate: its two edges are disconnected, so the geometry
    # walker (`link_offset_at_fraction`) can't trace a continuous path.
    # The maneuver is near-stationary and small (< offroad_max_straight_m),
    # so we snap to whichever endpoint the rule places the vehicle nearest
    # — `(edges[0], start_offset)` = source projection,
    # `(edges[-1], end_offset)` = destination projection. Both are real
    # on-edge points, so downstream lat/lon resolution is unaffected.
    if getattr(path, "is_off_road", False):
        src_pt = (path.edges[0], path.start_offset)
        dst_pt = (path.edges[-1], path.end_offset)
        if t_p <= 0.0 or B <= 0.0:
            return src_pt
        d_p = B - t_p
        if rule == "front":
            return src_pt if tau_s <= d_p else dst_pt
        if rule == "back":
            if tau_s >= t_p:
                return dst_pt
            return src_pt if (tau_s / t_p) < 0.5 else dst_pt
        if rule == "spread":
            return src_pt if (tau_s / B) < 0.5 else dst_pt
        raise ValueError(f"unknown dwell rule: {rule!r}")

    if t_p <= 0.0 or B <= 0.0:
        return path.edges[0], path.start_offset
    d_p = B - t_p

    if rule == "front":
        if tau_s <= d_p:
            return path.edges[0], path.start_offset
        frac = (tau_s - d_p) / t_p
    elif rule == "back":
        if tau_s >= t_p:
            return path.edges[-1], path.end_offset
        frac = tau_s / t_p
    elif rule == "spread":
        frac = tau_s / B
    else:
        raise ValueError(f"unknown dwell rule: {rule!r}")

    return link_offset_at_fraction(path, network, frac)


def position_at_time(
    segments: "list[TrajectoryPosterior]",
    t: datetime,
    network: "RoadNetwork",
    rule: DwellRule = "front",
) -> tuple[float, float] | None:
    """Predicted `(lat, lon)` at timestamp `t`, or `None` if uncovered.

    Searches `segments` for one whose canonical-time span contains `t`:

    - Exact observation timestamp: returns the most-likely state's
      `(lat, lon)` from `seg.most_likely[2k]`.
    - Confirmed-dwell window (`t_k < t ≤ t_last[k]`): vehicle was
      observed stationary at obs k's location throughout — returns the
      MLE state's `(lat, lon)` from `seg.most_likely[2k]`.
    - Transit window (`t_last[k] < t < t_{k+1}`): resolves the position
      along the Viterbi most-likely path using `position_in_transition`
      under the requested `rule`, with `τ = (t − t_last[k])`.
    - `t` outside every segment's span (gap between segments, before
      first or after last observation): returns `None`.

    `TrajectoryPosterior.__post_init__` guarantees `canonical_t_last` is
    populated (defaulting to `canonical_timestamps` when the caller didn't
    supply confirmed-dwell info), so this function indexes it directly.

    MLE-only: aggregates a single point from the Viterbi argmax. For the
    full posterior over states use `TrajectoryPosterior.at_time(t, rule)`.

    `rule` defaults to `"front"`, the project's convention. Pass `"back"`
    or `"spread"` to compare alternative dwell allocations against truth.
    """
    for seg in segments:
        ts = seg.canonical_timestamps
        if not ts or t < ts[0] or t > ts[-1]:
            continue

        for k, t_k in enumerate(ts):
            if t == t_k:
                state = seg.most_likely[2 * k]
                return _state_position(network, state)

        for k in range(len(ts) - 1):
            if ts[k] < t < ts[k + 1]:
                t_last_k = seg.canonical_t_last[k]
                if t_last_k > ts[k + 1]:
                    t_last_k = ts[k + 1]
                # Confirmed-dwell window: vehicle at obs k's MLE state.
                if t <= t_last_k:
                    state = seg.most_likely[2 * k]
                    return _state_position(network, state)
                step = seg.most_likely[2 * k + 1]
                if not isinstance(step, ModelPath):
                    return None
                tau_s = (t - t_last_k).total_seconds()
                link_id, offset = position_in_transition(
                    step, network, tau_s, rule,
                )
                idx = network.edge_index_for_link(link_id)
                return _position_on_edge(network, idx, offset)
        return None
    return None
