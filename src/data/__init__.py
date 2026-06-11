"""Shipped data assets for the pipeline.

Currently exposes `default_mu()` — a Porto-trained driver-model weight
vector ϕ(p) → μᵀϕ(p) used when the caller doesn't supply their own
`ExponentialFamilyTransition`. Lineage:

    src/data/mu_default.npy  ← produced by `scripts/retrain_mu.py`
                              against `cache/labeled_trips_15s.pkl.gz`
                              (built by `scripts/compute_15s_labels.py`).

The vector is co-versioned with `FEATURE_DIM`. If the schema bumps,
regenerate via the script chain.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..model.features import DEFAULT_DIRECTION_VIOLATION_WEIGHT, FEATURE_DIM

_MU_PATH: Path = Path(__file__).resolve().parent / "mu_default.npy"

# Pre-direction-violation schema size. An 18-dim stored μ is forward-
# compatible: slots [0..17] kept their semantics when slot [18]
# (n_direction_violations) was added, so the trained weights stay valid
# and only the new slot needs a value. Any other mismatch is a real
# schema break and still raises.
_PRE_VIOLATION_DIM: int = 18


def default_mu() -> np.ndarray:
    """Return the shipped default `mu` vector of shape `(FEATURE_DIM,)`.

    Falls back to zeros (uniform driver model) if the trained file isn't
    present yet — useful during initial development before
    `scripts/retrain_mu.py` has been run.

    A stored 18-dim vector (trained before the `n_direction_violations`
    slot existed) is padded to 19 with
    `DEFAULT_DIRECTION_VIOLATION_WEIGHT`; the hand prior governs the new
    slot until a dim-19 retrain (`scripts/compute_15s_labels.py` +
    `scripts/retrain_mu.py`). Any other dimension mismatch raises
    `ValueError`.
    """
    if not _MU_PATH.exists():
        return np.zeros(FEATURE_DIM, dtype=float)
    mu = np.load(_MU_PATH)
    if mu.shape == (_PRE_VIOLATION_DIM,) and FEATURE_DIM == _PRE_VIOLATION_DIM + 1:
        return np.concatenate(
            [mu.astype(float), [DEFAULT_DIRECTION_VIOLATION_WEIGHT]],
        )
    if mu.shape != (FEATURE_DIM,):
        raise ValueError(
            f"{_MU_PATH} has shape {mu.shape}, expected ({FEATURE_DIM},). "
            f"Regenerate via scripts/retrain_mu.py after FEATURE_DIM changes.",
        )
    return mu


def generic_prior_mu(
    w_length: float = 2.0, w_time: float = 0.0, w_violation: float = 1.0,
) -> np.ndarray:
    """A model-independent, physics-only transition prior.

    Returns a sparse `mu` that penalises path length (slot [0]) and
    optionally travel time (slot [12]), with every learned behavioural slot
    (turns, road class, dwell, perp) left at zero. Intended as a reference /
    label prior where the trained `mu` would be circular — e.g. the 15 s
    truth reconstruction: it suppresses gratuitous detours ("spurs") that a
    pure-emission (`mu = 0`) reconstruction admits, without importing the
    learned driver model.

    The perp slots [15]/[16] are deliberately left at 0: a negative weight
    there would reward a path whose endpoints snap closest to each ping,
    which is exactly the nearest-edge preference that manufactures the spur.
    Length is the clean anti-spur signal — a detour is long for the gap it
    bridges, so [0] alone separates it from the direct connection.
    """
    mu = np.zeros(FEATURE_DIM, dtype=float)
    mu[0] = -float(w_length)    # length_km — shorter paths more likely
    mu[12] = -float(w_time)     # travel_time_min (optional)
    # Mild wrong-way penalty (slot [18]). Without it, direction violations
    # are FREE under this prior, and the 15 s truth reconstruction (which
    # uses it) could flip to a wrong-way shortcut whenever it is slightly
    # shorter than the legal route. At 15 s the emission dominates, so
    # this acts only as the tiebreak it should be — still physics-flavoured
    # (wrong-way driving is rare), not a learned behaviour.
    mu[18] = -float(w_violation)
    return mu


__all__ = ["default_mu", "generic_prior_mu"]
