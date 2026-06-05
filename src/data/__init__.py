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

from ..model.features import FEATURE_DIM

_MU_PATH: Path = Path(__file__).resolve().parent / "mu_default.npy"


def default_mu() -> np.ndarray:
    """Return the shipped default `mu` vector of shape `(FEATURE_DIM,)`.

    Falls back to zeros (uniform driver model) if the trained file isn't
    present yet — useful during initial development before
    `scripts/retrain_mu.py` has been run. Raises `ValueError` if the
    stored file's dimension doesn't match `FEATURE_DIM`.
    """
    if not _MU_PATH.exists():
        return np.zeros(FEATURE_DIM, dtype=float)
    mu = np.load(_MU_PATH)
    if mu.shape != (FEATURE_DIM,):
        raise ValueError(
            f"{_MU_PATH} has shape {mu.shape}, expected ({FEATURE_DIM},). "
            f"Regenerate via scripts/retrain_mu.py after FEATURE_DIM changes.",
        )
    return mu


def generic_prior_mu(w_length: float = 2.0, w_time: float = 0.0) -> np.ndarray:
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
    return mu


__all__ = ["default_mu", "generic_prior_mu"]
