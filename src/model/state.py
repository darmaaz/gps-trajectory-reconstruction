"""State and Path types."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

import numpy as np

EdgeId = int


class State(Protocol):
    """Abstract state. Concrete representation is `(link_id, offset)`
    plus the canonical `entry_time` from the projected observation, plus
    the perpendicular distance from the observation to its projected
    point on the candidate edge.

    Algorithm code (`forward_backward`, `viterbi`, factor protocols) consumes
    `State`, never `StateV1` directly — the protocol seam lets the state
    representation evolve without touching the inference layer.
    """

    link_id: EdgeId
    offset: float            # metres along edge from `from_node`
    entry_time: datetime
    perp_m: float            # perpendicular distance, observation → edge


@dataclass(frozen=True)
class StateV1:
    link_id: EdgeId
    offset: float
    entry_time: datetime
    perp_m: float = 0.0
    # ^ Perpendicular distance in metres from the source observation to the
    # projected point at (link_id, offset). Populated by
    # `project_observation`; defaults to 0.0 for test fixtures that build
    # synthetic states without projection.


@dataclass(frozen=True, eq=False)
class Path:
    """A candidate path between two consecutive observations.

    Hashable by identity (`eq=False`) so `Path` can serve as a dict key in
    `path_marginals` outputs. Auto-generated value-equality is impossible
    here because `feature_vector` is a numpy array (unhashable). Identity
    semantics suffice — paths are produced once by `routing.candidate_paths`
    and threaded through the pipeline by reference.

    `time_budget` is the transition budget the path was enumerated under
    (seconds). `inferred_dwell` is `max(0, time_budget - expected_travel_time)`
    — the residual time, interpreted as dwell at the source state if the
    vehicle took this path. The `max(0, ...)` clamp matters when a path
    was admitted under `budget_slack > 1`: its `expected_travel_time` can
    exceed `time_budget`, which would mathematically yield negative dwell.
    That's physically uninterpretable (the path is "behind schedule"
    rather than dwelling), so the property reports 0 and the diagnostic
    is exposed via `is_overslacked` / `slack_deficit`.

    Default `0.0` for `time_budget` means the field wasn't set at
    construction (test fixtures); production paths from
    `routing.candidate_paths` always set it explicitly.
    """

    edges: tuple[EdgeId, ...]
    start_offset: float
    end_offset: float
    expected_travel_time: float    # seconds, using typical_speed
    length_meters: float
    feature_vector: np.ndarray     # ϕ(p)
    time_budget: float = 0.0
    start_perp_m: float = 0.0
    end_perp_m: float = 0.0
    min_traversal_time: float = 0.0
    is_off_road: bool = False
    # ^ True for a straight-line off-network candidate connecting two
    # *disconnected* projected edges (parking / arrival / idle-near-
    # one-way-pair maneuvers that legal routing can't represent). When
    # set, the `edges` tuple is (src_link, dst_link) with NO topological
    # adjacency between them — so feature extraction and `at_time` must
    # branch: adjacency-dependent features (turns, signals, intersections)
    # are meaningless and zeroed; `at_time` snaps to the endpoints rather
    # than walking edge geometry. Produced only by `routing.candidate_paths`
    # when `Config.enable_offroad_candidates` is set and the trigger fires.
    # ^ `expected_travel_time` uses typical_speeds (real-driving prior).
    # `min_traversal_time` uses max_speeds (physical upper bound on
    # speed): the path's minimum-possible traversal time. Admission to
    # the candidate set uses `min_traversal_time` (filter on physical
    # possibility); the CRF likelihood and dwell decomposition use
    # `expected_travel_time` (filter on typical plausibility). Default
    # 0.0 covers test fixtures; production paths from
    # `routing.candidate_paths` always set it explicitly.
    #
    # Perpendicular distance in metres from the path's start/end states
    # to their bracketing observations — i.e. how well the path's
    # geometry anchors to the actual GPS pings. Populated by
    # `_build_path` from `src_state.perp_m` / `dst_state.perp_m`.

    @property
    def inferred_dwell(self) -> float:
        """Non-negative residual time interpreted as dwell at the source.

        Clamped at 0: paths with `expected_travel_time > time_budget` (i.e.,
        admitted under budget_slack) report 0 here. Use `slack_deficit` to
        recover how much the path exceeds the budget.
        """
        return max(0.0, self.time_budget - self.expected_travel_time)

    @property
    def is_overslacked(self) -> bool:
        """True when the path's transit time exceeds its enumeration budget.

        Set by `budget_slack > 1` in `routing.candidate_paths`: the path is
        admitted because real driving may exceed posted speeds, but its
        nominal travel-time estimate doesn't fit the observed gap. These
        paths report `inferred_dwell == 0` and `slack_deficit > 0`.
        """
        return self.expected_travel_time > self.time_budget

    @property
    def slack_deficit(self) -> float:
        """Seconds by which `expected_travel_time` exceeds `time_budget`.

        Zero for non-overslacked paths. Positive for slacked admissions;
        the complement of `inferred_dwell` — exactly one of the two is
        non-zero for any path with positive `time_budget`.
        """
        return max(0.0, self.expected_travel_time - self.time_budget)

    def starts_at(self, state: State, tol_m: float = 1.0) -> bool:
        return (
            len(self.edges) > 0
            and self.edges[0] == state.link_id
            and abs(self.start_offset - state.offset) < tol_m
        )

    def ends_at(self, state: State, tol_m: float = 1.0) -> bool:
        return (
            len(self.edges) > 0
            and self.edges[-1] == state.link_id
            and abs(self.end_offset - state.offset) < tol_m
        )
