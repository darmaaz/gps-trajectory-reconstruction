"""Tier 4 — supervised validation metrics, scaffolded for future use.

These metrics require ground-truth path labels per trip. The current
fleet doesn't have usable labels (business-trip metadata is too noisy);
this module is implemented and tested against synthetic input so it's
ready to fire as soon as a labeled dataset arrives.

Expected label format (commit when an actual schema lands):

    LabeledTripGroundTruth = TypedDict({
        "trip_id": str,
        "edges": tuple[int, ...],          # ordered OSM way ids actually driven
        "ground_truth_pings": list[dict] | None,
            # optional — per-time positions from a high-quality parallel logger
            # each dict: {"lat": float, "lon": float, "timestamp": datetime}
    })

Plug `validate.py --labels labels.parquet` at one of these and Tier 4
produces:
    - path-miss rate per trip (and aggregate)
    - point-miss rate per trip at threshold T (default 20 m)
    - posterior top-K rank distribution over trips
    - credible region coverage at level L (default 0.9)
"""

from __future__ import annotations

from typing import Any, Iterable

from .holdout import _min_distance_to_edges


def path_miss_rate(
    reconstructed_edges: Iterable[int],
    ground_truth_edges: Iterable[int],
) -> float:
    """Fraction of ground-truth edges NOT in the reconstructed edge set.

    PIF baseline: ~10–15 % at 60 s sampling, ~25 % at 120 s.
    """
    gt = set(int(e) for e in ground_truth_edges)
    if not gt:
        return 0.0
    rec = set(int(e) for e in reconstructed_edges)
    missed = gt - rec
    return len(missed) / len(gt)


def point_miss_rate(
    reconstructed_edge_ids: list[int],
    ground_truth_pings: list[dict[str, Any]],
    network,
    threshold_m: float = 20.0,
) -> float:
    """Fraction of ground-truth pings whose distance to the reconstructed
    path geometry exceeds `threshold_m`.
    """
    if not ground_truth_pings:
        return 0.0
    n = 0
    miss = 0
    for ping in ground_truth_pings:
        d = _min_distance_to_edges(
            float(ping["lat"]), float(ping["lon"]),
            reconstructed_edge_ids, network,
        )
        if d is None:
            continue
        n += 1
        if d > threshold_m:
            miss += 1
    return miss / n if n else 0.0


def posterior_top_k_rank(
    path_marginals: dict, ground_truth_edges: tuple[int, ...],
) -> int | None:
    """Rank (1-indexed) of the ground-truth path in the posterior, sorted
    by descending probability. Returns None if the GT path doesn't appear
    in the posterior at all."""
    gt = tuple(int(e) for e in ground_truth_edges)
    sorted_paths = sorted(
        path_marginals.items(), key=lambda kv: -kv[1],
    )
    for i, (path, _prob) in enumerate(sorted_paths):
        if tuple(int(e) for e in path.edges) == gt:
            return i + 1
    return None


def credible_region_coverage(
    path_marginals: dict,
    ground_truth_edges: tuple[int, ...],
    level: float = 0.9,
) -> bool:
    """True if the ground-truth path is inside the `level`-credible region
    of the posterior (the smallest set of paths whose cumulative posterior
    mass ≥ level, taken in descending-probability order)."""
    gt = tuple(int(e) for e in ground_truth_edges)
    sorted_paths = sorted(path_marginals.items(), key=lambda kv: -kv[1])
    cumulative = 0.0
    for path, prob in sorted_paths:
        cumulative += prob
        if tuple(int(e) for e in path.edges) == gt:
            return True
        if cumulative >= level:
            return False
    return False
