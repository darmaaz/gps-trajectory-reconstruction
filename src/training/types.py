"""Training data types — bridge between labeled trips and inference."""

from __future__ import annotations

from dataclasses import dataclass

from ..model import CollapsedObservation, Path, State


@dataclass
class LabeledTrip:
    """A trip with a known ground-truth trajectory.

    Construction expectation: the user runs the standard preprocessing +
    candidate enumeration to produce `state_candidates` and
    `path_candidates`, then identifies which candidate matches the truth at
    each step (e.g. by edge-id matching against high-quality parallel
    logging) and records those indices.

    Invariants:
        - `len(state_candidates) == len(observations) == len(label_state_idx)`
        - `len(path_candidates) == len(observations) - 1 == len(label_path_idx)`
        - `0 <= label_state_idx[k] < len(state_candidates[k])`
        - `0 <= label_path_idx[k] < len(path_candidates[k])`

    No checks are performed at construction; callers are responsible for
    validity. Mismatched indices will surface as `IndexError` during
    `fit_supervised`.
    """

    observations: list[CollapsedObservation]
    state_candidates: list[list[State]]
    path_candidates: list[list[Path]]
    time_budgets: list[float]
    label_state_idx: list[int]
    label_path_idx: list[int]
