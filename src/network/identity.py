"""Physical-route identity for candidate paths.

A `Path`'s `edges` tuple is a *directed* edge-id sequence anchored to its
terminal state projections. That identity is correct for inference (paths
attach to specific CRF cells via `starts_at` / `ends_at`) but wrong for
counting: the same physical route appears under multiple directed spellings
— terminal states projected onto opposite-direction twins of the same
street, or onto neighbouring edges of the same corridor — and edge-id-based
metrics double-count or fragment it.

`canonical_route` maps a path to its sequence of *undirected* segment keys
(`RoadNetwork.undirected_segment_keys`), with consecutive duplicates
collapsed so a twin-anchored doubling-back onto the same street does not
mint a new identity. Two paths with equal canonical routes traversed the
same physical streets in the same order.

Intended consumers:
  - reporting / figures / calibration — aggregate by canonical route, never
    by raw `link_id`;
  - `routing.candidate_paths` — diversity-aware truncation, so the
    `max_paths` cap is spent on distinct physical routes rather than
    directed spellings of the same one.

NOT for inference dedup: paths with equal canonical routes can attach to
different `(src_state, dst_state)` cells; merging them would silently
change the CRF support.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..model import Path
    from .loader import RoadNetwork

SegmentKey = tuple[int, int]


def canonical_route(
    path: "Path", network: "RoadNetwork",
) -> tuple[SegmentKey, ...] | None:
    """Undirected, consecutive-deduped segment-key sequence for `path`.

    Returns `None` for off-road paths (their `edges` tuple is a
    non-topological `(src, dst)` marker pair, not a route) and treats a
    `link_id` unknown to `network` by skipping it (mirrors the tolerance
    of existing figure/metric code on subgraphs).
    """
    if path.is_off_road:
        return None
    out: list[SegmentKey] = []
    for link in path.edges:
        try:
            key = network.segment_key_for_link(int(link))
        except KeyError:
            continue
        if not out or out[-1] != key:
            out.append(key)
    return tuple(out)


def truncate_with_route_diversity(
    ordered: list["Path"],
    network: "RoadNetwork",
    max_paths: int,
) -> list["Path"]:
    """Keep up to `max_paths` from an expected-travel-time-sorted list,
    preferring paths whose canonical route is not yet represented.

    First pass admits the best (earliest) path per canonical route; if
    slots remain, they are filled with the best leftovers — so the result
    is never *smaller* than plain truncation would give. Off-road paths
    (`canonical_route is None`) are always novel. The returned list is
    re-sorted by `expected_travel_time`, preserving the public contract
    of `candidate_paths`.
    """
    if len(ordered) <= max_paths:
        return ordered
    seen: set[tuple[SegmentKey, ...]] = set()
    selected: list["Path"] = []
    leftovers: list["Path"] = []
    for p in ordered:
        route = canonical_route(p, network)
        if route is None or route not in seen:
            if route is not None:
                seen.add(route)
            selected.append(p)
            if len(selected) == max_paths:
                break
        else:
            leftovers.append(p)
    if len(selected) < max_paths:
        selected.extend(leftovers[: max_paths - len(selected)])
    selected.sort(key=lambda p: p.expected_travel_time)
    return selected
