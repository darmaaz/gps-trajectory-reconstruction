"""Tier 2 held-out cross-validation.

For each trip, mask some observations, reconstruct from the rest, and
measure the perpendicular distance from each masked observation to the
reconstructed path geometry. Aggregating across trips yields a CDF of
"how well does the reconstruction predict observations it didn't see."

Two flavours implemented here:

    `endpoint_holdout`         — mask the first and last raw obs of each
                                  trip; check distance from each held-out
                                  endpoint to the segment whose time span
                                  is closest.
    `random_interior_holdout`  — randomly mask `fraction` of the interior
                                  raw obs; check distance from each held-out
                                  obs to the segment that covers its
                                  timestamp.

Both return raw distance lists; the caller aggregates (P50, P90, etc.).
"""

from __future__ import annotations

import shapely
import numpy as np

from ..api import reconstruct_trajectory
from ..config import Config
from ..geo import equirectangular_distance_m
from ..model import Path, RawObservation
from ..network import RoadNetwork


def _min_distance_to_edges(
    lat: float, lon: float, edge_ids: list[int], network: RoadNetwork,
) -> float | None:
    """Min perpendicular distance (m) from `(lat, lon)` to any edge in
    `edge_ids`. None if no edge resolves."""
    if not edge_ids:
        return None
    point = shapely.points([lon], [lat])[0]
    min_d = float("inf")
    for eid in edge_ids:
        try:
            idx = network.edge_index_for_link(int(eid))
        except KeyError:
            continue
        geom = network.geoms[idx]
        dist_along_deg = geom.project(point)
        nearest = geom.interpolate(dist_along_deg)
        d = float(equirectangular_distance_m(
            lat, lon, float(nearest.y), float(nearest.x),
        ))
        if d < min_d:
            min_d = d
    return min_d if min_d < float("inf") else None


def _segment_edge_ids(segment) -> list[int]:
    """Flat list of edge IDs covering the segment's most-likely path."""
    edge_ids: list[int] = []
    for item in segment.most_likely:
        if isinstance(item, Path):
            edge_ids.extend(int(e) for e in item.edges)
    return edge_ids


def _segment_for_timestamp(ts, segments):
    """Return the segment whose canonical time span contains `ts`, or the
    closest segment by time if none does. None if no segments."""
    if not segments:
        return None
    for s in segments:
        if not s.canonical_timestamps:
            continue
        if s.canonical_timestamps[0] <= ts <= s.canonical_timestamps[-1]:
            return s

    def _gap(s):
        if not s.canonical_timestamps:
            return float("inf")
        return min(
            abs((s.canonical_timestamps[0] - ts).total_seconds()),
            abs((s.canonical_timestamps[-1] - ts).total_seconds()),
        )

    return min(segments, key=_gap)


def endpoint_holdout(
    raw_observations: list[RawObservation],
    network: RoadNetwork,
    config: Config,
) -> list[tuple[str, float]]:
    """Mask `raw_observations[0]` and `raw_observations[-1]`, reconstruct
    from the middle, and return `(label, distance_m)` for each held-out
    endpoint.

    Returns `[]` if the trip is too short or reconstruction returns no
    segments.
    """
    if len(raw_observations) < 4:
        return []
    head = raw_observations[0]
    tail = raw_observations[-1]
    middle = raw_observations[1:-1]

    segments = reconstruct_trajectory(middle, network, config)
    if not segments:
        return []

    out: list[tuple[str, float]] = []
    head_seg = _segment_for_timestamp(head.timestamp, segments)
    if head_seg is not None:
        d = _min_distance_to_edges(
            head.lat, head.lon, _segment_edge_ids(head_seg), network,
        )
        if d is not None:
            out.append(("head", d))
    tail_seg = _segment_for_timestamp(tail.timestamp, segments)
    if tail_seg is not None:
        d = _min_distance_to_edges(
            tail.lat, tail.lon, _segment_edge_ids(tail_seg), network,
        )
        if d is not None:
            out.append(("tail", d))
    return out


def random_interior_holdout(
    raw_observations: list[RawObservation],
    network: RoadNetwork,
    config: Config,
    fraction: float = 0.25,
    seed: int = 42,
) -> list[float]:
    """Randomly mask `fraction` of the interior observations (excluding
    boundaries), reconstruct from the rest, and return the distance from
    each masked observation to the reconstructed path of the segment that
    covers its timestamp.
    """
    if len(raw_observations) < 6:
        return []
    if not 0 < fraction < 1:
        raise ValueError("fraction must be in (0, 1)")

    n_interior = len(raw_observations) - 2
    k = max(1, int(round(n_interior * fraction)))
    rng = np.random.default_rng(seed)
    masked_idxs = set(rng.choice(
        np.arange(1, len(raw_observations) - 1), size=k, replace=False,
    ).tolist())

    kept = [o for i, o in enumerate(raw_observations) if i not in masked_idxs]
    held_out = [raw_observations[i] for i in sorted(masked_idxs)]

    segments = reconstruct_trajectory(kept, network, config)
    if not segments:
        return []

    distances: list[float] = []
    for ho in held_out:
        seg = _segment_for_timestamp(ho.timestamp, segments)
        if seg is None:
            continue
        d = _min_distance_to_edges(
            ho.lat, ho.lon, _segment_edge_ids(seg), network,
        )
        if d is not None:
            distances.append(d)
    return distances
