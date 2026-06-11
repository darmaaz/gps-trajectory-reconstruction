"""Path-feature vector ϕ(p).

PIF's "Complex" feature set, plus dwell terms and traffic-control counts
that give the CRF a way to penalise implausible paths.

    [0]  length_km          — length / 1000  (most paths ∈ [0, 5])
    [1]  n_left_turns       — integer count, small
    [2]  n_right_turns      — integer count, small
    [3]  n_signals          — count of `highway=traffic_signals` nodes
                              traversed (= `to_node` of edges 0..n-2 in path)
    [4]  n_stop_signs       — count of `highway=stop` nodes traversed
    [5]  frac_motorway      — ∈ [0, 1]
    [6]  frac_trunk         — ∈ [0, 1]
    [7]  frac_primary       — ∈ [0, 1]
    [8]  frac_secondary     — ∈ [0, 1]
    [9]  frac_tertiary      — ∈ [0, 1]
    [10] frac_residential   — ∈ [0, 1]  (includes `unclassified`)
    [11] frac_service       — ∈ [0, 1]  (includes `living_street`)
    [12] travel_time_min    — expected_travel_time / 60  (typical ∈ [0, 10])
    [13] inferred_dwell_min — inferred_dwell / 60  (typical ∈ [0, 3])
    [14] dwell_ratio        — inferred_dwell / time_budget  ∈ [0, 1]
    [15] start_perp_10m     — start_perp_m / 10  (typical ∈ [0, 5])
    [16] end_perp_10m       — end_perp_m / 10  (typical ∈ [0, 5])
    [17] n_intersections    — count of multi-way (degree ≥ 3) junctions
                              traversed in the path. Catches accel/decel
                              cost at uncontrolled intersections that
                              `n_signals` / `n_stop_signs` / `n_turns`
                              miss (e.g. a 4-way uncontrolled where the
                              vehicle goes straight through).
    [18] n_direction_violation_runs — count of wrong-way *maneuvers*:
                              maximal RUNS of consecutive reversed edges
                              (`Path.reversed_mask`), NOT per-edge. OSM
                              splits a street at every intersection, so one
                              continuous wrong-way drive becomes several
                              reversed edges; counting runs makes the
                              feature invariant to that splitting (a single
                              maneuver costs once, however many edges it
                              spans). Always 0 unless `Config.
                              enable_direction_violation` admitted
                              wrong-way candidates. Expected weight
                              strongly ≤ 0: a violation is possible
                              (taxis run one-ways, pull out of lots, OSM
                              oneway tags are sometimes wrong) but must
                              cost real evidence to win. Until μ is
                              retrained at dim 19, `data.default_mu()`
                              pads a stored 18-dim vector with the hand
                              prior `DEFAULT_DIRECTION_VIOLATION_WEIGHT`.

All feature values are now in roughly [0, 10] range, keeping `μᵀϕ(p)` in
a numerically reasonable regime under modest L2 regularisation. Without
this scaling, the time/length features (raw seconds and metres, magnitudes
in 60-500) dominated the gradient and caused the unbounded-MLE
degeneracy observed when training at slack=1.2 with sharp labels.

Fixed dimensionality is 18. The driver-model `mu` vector has the same
dimension, learned by EM or supervised MLE. Schema changes must be
co-versioned with stored `mu` values; mismatched shapes raise loudly in
`ExponentialFamilyTransition.log_potential`. Scaling changes leave the
shape unchanged but invalidate the numerical interpretation of stored μ
— always retrain after editing the per-slot scaling.

Expected weight signs after training (sanity gates):
- `mu[0]` (length) ≤ 0: shorter paths are more likely.
- `mu[12]` (expected_travel_time): typically negative, but coupled with
  budget effects; not a strict sign gate.
- `mu[13]` (inferred_dwell_s) ≤ 0: less dwell is more likely.
- `mu[14]` (dwell ratio) ≤ 0: paths with most of the budget as dwell are
  less likely than paths that mostly transit.
- `mu[15]`, `mu[16]` (perp distances) ≤ 0: paths whose endpoints land
  close to their observations are more likely.
"""

from __future__ import annotations

import numpy as np

from ..geo import forward_azimuth_deg
from .state import Path, count_violation_runs

FEATURE_DIM: int = 19

# Hand prior for slot [18] when a pre-bump 18-dim μ file is padded by
# `data.default_mu()`. −2.0 ≈ e⁻² odds penalty per wrong-way *maneuver*
# (slot [18] counts runs, not edges): strong enough that a violation only
# wins when the legal alternatives are implausible (fig-1-style forced
# loops), weak enough to be winnable. Superseded by retraining μ at dim 19
# (scripts/compute_15s_labels.py + scripts/retrain_mu.py).
DEFAULT_DIRECTION_VIOLATION_WEIGHT: float = -2.0

# Indices for the road-class fraction block.
_CLASS_TO_SLOT: dict[str, int] = {
    "motorway": 5, "motorway_link": 5,
    "trunk": 6, "trunk_link": 6,
    "primary": 7, "primary_link": 7,
    "secondary": 8, "secondary_link": 8,
    "tertiary": 9, "tertiary_link": 9,
    "residential": 10, "unclassified": 10, "living_street": 11,
    "service": 11,
}

# Bearing-change thresholds (degrees, signed; positive = right turn). Below
# 30° absolute is treated as straight-through; ignores curvature within an
# intersection while still catching real turn moves.
_TURN_DEGREES: float = 30.0


def _edge_entry_bearing(network, edge_idx: int) -> float:
    coords = list(network.geoms[edge_idx].coords)
    if len(coords) < 2:
        return 0.0
    (lon0, lat0), (lon1, lat1) = coords[0], coords[1]
    return float(forward_azimuth_deg(lat0, lon0, lat1, lon1))


def _edge_exit_bearing(network, edge_idx: int) -> float:
    coords = list(network.geoms[edge_idx].coords)
    if len(coords) < 2:
        return 0.0
    (lon0, lat0), (lon1, lat1) = coords[-2], coords[-1]
    return float(forward_azimuth_deg(lat0, lon0, lat1, lon1))


def _signed_bearing_delta(b_in: float, b_out: float) -> float:
    """Signed angular change in degrees, ∈ (-180, 180]. Positive = right
    turn (bearings are clockwise from north)."""
    return ((b_out - b_in + 540.0) % 360.0) - 180.0


def _entry_bearing(network, edge_idx: int, reversed_: bool) -> float:
    """Entry bearing honouring traversal direction: a reverse traversal
    enters where the mapped geometry exits, heading flipped 180°."""
    if not reversed_:
        return _edge_entry_bearing(network, edge_idx)
    return (_edge_exit_bearing(network, edge_idx) + 180.0) % 360.0


def _exit_bearing(network, edge_idx: int, reversed_: bool) -> float:
    if not reversed_:
        return _edge_exit_bearing(network, edge_idx)
    return (_edge_entry_bearing(network, edge_idx) + 180.0) % 360.0


def path_features(path: Path, network) -> np.ndarray:
    """Return the ϕ(p) feature vector for `path` over `network`.

    Side-effect-free; safe to call repeatedly. `network` is required because
    turn counts and road-class fractions need per-edge metadata that the
    `Path` dataclass doesn't carry.
    """
    feats = np.zeros(FEATURE_DIM, dtype=float)
    feats[0]  = float(path.length_meters)       / 1000.0    # km
    feats[12] = float(path.expected_travel_time) /   60.0   # minutes
    feats[13] = float(path.inferred_dwell)      /   60.0    # minutes
    feats[14] = (
        float(path.inferred_dwell) / float(path.time_budget)
        if path.time_budget > 0.0 else 0.0
    )
    feats[15] = float(path.start_perp_m) / 10.0     # ten-metre units
    feats[16] = float(path.end_perp_m)   / 10.0

    # Off-road candidate: its two edges are NOT topologically adjacent.
    # Adjacency-dependent slots (turns, signals, stops, intersections)
    # are meaningless and stay zero. But the road-class fractions MUST be
    # populated from the two endpoint edges, not left zero: the driver
    # model carries large positive class-fraction weights, so a path with
    # zeroed class fractions is structurally handicapped and could never
    # win against a routed alternative regardless of its length/dwell.
    # The maneuver happens near both endpoint edges, so split the class
    # mass evenly between them (sums to 1, matching on-road paths).
    if getattr(path, "is_off_road", False):
        for link in path.edges:
            try:
                idx = network.edge_index_for_link(int(link))
            except KeyError:
                continue
            slot = _CLASS_TO_SLOT.get(str(network.road_classes[idx]))
            if slot is not None:
                feats[slot] += 0.5
        return feats

    if not path.edges:
        return feats

    edge_idxs: list[int] = []
    rev_flags: list[bool] = []
    for i, link in enumerate(path.edges):
        try:
            edge_idxs.append(network.edge_index_for_link(int(link)))
        except KeyError:
            continue
        rev_flags.append(path.edge_reversed(i))
    if not edge_idxs:
        return feats

    feats[18] = float(count_violation_runs(rev_flags))   # n_direction_violation_runs

    # Road-class fractions over edge length. Using `length_meters` (which
    # may include partial source/destination edges) as the denominator
    # would be slightly inconsistent with the per-edge full lengths used in
    # the numerator. Use the sum of full edge lengths as the denominator
    # for the fractions; the unscaled length still occupies feats[0].
    total_class_len = 0.0
    for idx in edge_idxs:
        rc = str(network.road_classes[idx])
        slot = _CLASS_TO_SLOT.get(rc)
        edge_len = float(network.lengths_m[idx])
        total_class_len += edge_len
        if slot is not None:
            feats[slot] += edge_len
    if total_class_len > 0:
        feats[5:12] /= total_class_len

    # Turn counts: bearing changes between consecutive edges. Skips the
    # case of a single-edge path (no internal joints).
    n_left = n_right = 0
    for (a, ra), (b, rb) in zip(
        zip(edge_idxs, rev_flags), zip(edge_idxs[1:], rev_flags[1:]),
    ):
        delta = _signed_bearing_delta(
            _exit_bearing(network, a, ra),
            _entry_bearing(network, b, rb),
        )
        if delta >= _TURN_DEGREES:
            n_right += 1
        elif delta <= -_TURN_DEGREES:
            n_left += 1
    feats[1] = float(n_left)
    feats[2] = float(n_right)

    # Traffic-signal and stop-sign counts. A signal/stop on the `to_node`
    # of edge e applies between e and the next edge in the path. Count
    # over `edge_idxs[:-1]` so we don't count the path's final destination
    # node as a traversed signal/stop (the vehicle is stopping there
    # regardless of whether it's controlled). Backward-compatible: when
    # the network was built without signal/stop data (e.g., test
    # fixtures), the per-edge arrays are zeros and feats[3]/feats[4] stay 0.
    # Reversed traversals are excluded from signal/stop counts: the
    # control sits on the edge's `to_node`, which a reverse traversal
    # *enters from* rather than exits through — counting it would charge
    # the path for a control it approaches from the uncontrolled side.
    if hasattr(network, "to_node_is_signal") and len(network.to_node_is_signal) > 0:
        feats[3] = float(sum(
            int(network.to_node_is_signal[idx])
            for idx, rev in zip(edge_idxs[:-1], rev_flags[:-1]) if not rev
        ))
    if hasattr(network, "to_node_is_stop") and len(network.to_node_is_stop) > 0:
        feats[4] = float(sum(
            int(network.to_node_is_stop[idx])
            for idx, rev in zip(edge_idxs[:-1], rev_flags[:-1]) if not rev
        ))

    # Intersection count: each consecutive edge pair in the path joins at
    # a node; that node is a "real intersection" if its total degree ≥ 3
    # (a multi-way junction, not just an OSM way split at degree=2).
    # Captures accel/decel cost at uncontrolled junctions that the
    # signal/stop/turn features don't catch. Backward-compatible:
    # synthetic fixtures with empty `node_total_degree` fall back to a
    # raw `len(edges) - 1` proxy.
    if hasattr(network, "node_total_degree") and network.node_total_degree:
        n_intersections = 0
        for a, rev in zip(edge_idxs[:-1], rev_flags[:-1]):
            # A reverse traversal exits via `from_node`, not `to_node`.
            connecting_node = int(network.from_node[a] if rev else network.to_node[a])
            if network.node_total_degree.get(connecting_node, 0) >= 3:
                n_intersections += 1
        feats[17] = float(n_intersections)
    else:
        feats[17] = float(max(0, len(edge_idxs) - 1))

    return feats
