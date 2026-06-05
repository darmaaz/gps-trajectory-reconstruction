"""Tier 1 sanity gates — fast, intrinsic checks on calibrated parameters.

No held-out data, no external truth. Just: does the calibrated `(mu, scale)`
look like the parameters of a working driver-and-emission model?

Failure of a Tier 1 gate is strong evidence calibration converged to a bad
local optimum. Investigate before relying on the parameters downstream.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..model import FEATURE_DIM

# Per-feature sign expectations after calibration on real fleet data.
# `expected` is one of "negative", "positive", "any" — slots flagged "any"
# either don't have a strong prior (e.g. secondary road class can go either
# way depending on fleet composition) or are placeholder features whose
# value depends on data extraction we don't currently do (signals, stops).
_FEATURE_EXPECTATIONS: dict[int, tuple[str, str]] = {
    0:  ("length_m",            "negative"),
    1:  ("n_left_turns",        "negative"),
    2:  ("n_right_turns",       "negative"),
    3:  ("n_signals",           "any"),       # placeholder, not extracted
    4:  ("n_stop_signs",        "any"),       # placeholder, not extracted
    5:  ("frac_motorway",       "positive"),
    6:  ("frac_trunk",          "positive"),
    7:  ("frac_primary",        "positive"),
    8:  ("frac_secondary",      "any"),
    9:  ("frac_tertiary",       "any"),
    10: ("frac_residential",    "negative"),
    11: ("frac_service",        "negative"),
    12: ("expected_travel_time_s", "negative"),
}

_SIGN_EPS = 1e-6


def _check_sign(value: float, expected: str) -> bool:
    if expected == "any":
        return True
    if expected == "negative":
        return value < -_SIGN_EPS
    if expected == "positive":
        return value > _SIGN_EPS
    if expected == "zero":
        return abs(value) < _SIGN_EPS
    raise ValueError(f"unknown sign expectation: {expected!r}")


def check_mu_signs(mu: np.ndarray) -> dict[str, dict[str, Any]]:
    """For each feature slot, report value and whether its sign matches the
    expected calibration outcome.

    Returns a dict keyed by feature name. Each value has `value`, `expected`,
    and `passes` keys.
    """
    if len(mu) < FEATURE_DIM:
        raise ValueError(
            f"mu has length {len(mu)}; expected {FEATURE_DIM}",
        )
    results: dict[str, dict[str, Any]] = {}
    for idx, (name, expected) in _FEATURE_EXPECTATIONS.items():
        value = float(mu[idx])
        results[name] = {
            "feature_idx": idx,
            "value": value,
            "expected": expected,
            "passes": _check_sign(value, expected),
        }
    return results


def check_scale_bounds(
    scale: float, lo: float = 0.5, hi: float = 100.0,
) -> dict[str, Any]:
    """Sanity-check the Student-t emission scale.

    Real consumer-GPS noise sits in 5–15 m. Our hard bound during
    optimisation is 0.1–1000 m. A calibrated scale outside [0.5, 100.0]
    is suspicious — either pegged to a bound (degenerate input) or fit to
    pathological residuals.
    """
    if not math.isfinite(scale) or scale <= 0:
        return {
            "value": scale, "passes": False,
            "note": "non-finite or non-positive",
        }
    if scale < lo:
        return {
            "value": scale, "passes": False,
            "note": f"below lower threshold {lo} m — likely pegged to optimiser bound",
        }
    if scale > hi:
        return {
            "value": scale, "passes": False,
            "note": f"above upper threshold {hi} m — emission too diffuse to be meaningful",
        }
    if scale < 1.0 or scale > 30.0:
        return {
            "value": scale, "passes": True,
            "note": f"in plausible range but unusual ({scale:.2f} m); typical consumer GPS is 5–15 m",
        }
    return {"value": scale, "passes": True, "note": "within typical consumer-GPS noise band"}
