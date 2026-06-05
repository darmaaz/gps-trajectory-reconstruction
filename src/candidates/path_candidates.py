"""Per-transition path candidate enumeration — SPEC.md §candidates.path_candidates.

Wraps `routing.candidate_paths` over each consecutive
`(state_candidates[k], state_candidates[k+1])` pair, returning one path list
per transition. Optionally attaches PIF-style feature vectors via
`model.features.path_features` so paths are inference-ready.
"""

from __future__ import annotations

from dataclasses import replace

from ..model import Path, State
from ..model.features import path_features
from ..network import RoadNetwork
from ..network.routing import DEFAULT_K_PER_PAIR, candidate_paths


def enumerate_paths_per_transition(
    state_candidates: list[list[State]],
    network: RoadNetwork,
    time_budgets: list[float],
    max_path_candidates: int = 20,
    *,
    k_per_pair: int = DEFAULT_K_PER_PAIR,
    attach_features: bool = True,
    budget_slack: float = 1.0,
    penalty_lambda: float = 0.3,
    enable_offroad: bool = False,
    offroad_max_straight_m: float = 300.0,
    offroad_min_detour_ratio: float = 3.0,
    offroad_min_overslack: float = 1.0,
) -> list[list[Path]]:
    """Return one `list[Path]` per consecutive observation transition.

    `state_candidates` has length T (one per CollapsedObservation);
    `time_budgets` must have length T-1 (`time_budgets[k] = collapsed[k+1].t_first
    - collapsed[k].t_first` in seconds). Output has length T-1; an empty
    inner list signals a transition-level discontinuity (no feasible path
    within budget) — the orchestrator splits the trip there per
    SPEC.md §Edge cases.

    `attach_features=True` (default) populates each path's `feature_vector`
    via `model.features.path_features`. Skip when features aren't needed
    yet (e.g., ablations that defer feature computation).

    `budget_slack` is plumbed through to `routing.candidate_paths` to allow
    paths whose edge-time estimate exceeds the observed gap (e.g., real
    driving above posted speed limits).
    """
    if len(state_candidates) < 2:
        return []
    if len(time_budgets) != len(state_candidates) - 1:
        raise ValueError(
            f"time_budgets has length {len(time_budgets)}; "
            f"expected {len(state_candidates) - 1}",
        )

    out: list[list[Path]] = []
    for k in range(len(state_candidates) - 1):
        paths = candidate_paths(
            state_candidates[k], state_candidates[k + 1],
            network, time_budgets[k], max_path_candidates,
            k_per_pair=k_per_pair,
            budget_slack=budget_slack,
            penalty_lambda=penalty_lambda,
            enable_offroad=enable_offroad,
            offroad_max_straight_m=offroad_max_straight_m,
            offroad_min_detour_ratio=offroad_min_detour_ratio,
            offroad_min_overslack=offroad_min_overslack,
        )
        if attach_features and paths:
            paths = [
                replace(p, feature_vector=path_features(p, network))
                for p in paths
            ]
        out.append(paths)
    return out
