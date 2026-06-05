"""Factor protocols and concrete implementations.

Protocols:

    EmissionFactor.log_potential(obs, state)
        Log-potential for the observation given the state. Heavy-tailed in
        perpendicular distance from the report to the candidate edge.

    TransitionFactor.log_potential(state_k, path, state_{k+1}, time_budget)
        Log-potential for taking `path` between two consecutive states.
        `time_budget` is in the signature so dwell-aware factors can
        consume it without changing call sites.

Concretes:

    StudentTEmission(scale, network, df=4.0)
    ExponentialFamilyTransition(mu)

`StudentTEmission` holds a `RoadNetwork` reference because the protocol
signature `(obs, state)` doesn't expose one — the orchestrator wires the
per-trip subgraph at construction time.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Protocol

import numpy as np

from ..geo import equirectangular_distance_m
from .observation import CollapsedObservation
from .state import Path, State

if TYPE_CHECKING:
    from ..network import RoadNetwork


class EmissionFactor(Protocol):
    def log_potential(
        self, obs: CollapsedObservation, state: State,
    ) -> float: ...

    def rebind(self, network: "RoadNetwork") -> "EmissionFactor": ...


class TransitionFactor(Protocol):
    def log_potential(
        self,
        state_k: State,
        path: Path,
        state_kplus1: State,
        time_budget: float,    # available for dwell-aware factors
    ) -> float: ...

    def rebind(self, network: "RoadNetwork") -> "TransitionFactor": ...


class StudentTEmission:
    """Heavy-tailed log-pdf of perpendicular distance under Student-t.

    Model: the report's perpendicular offset from the candidate edge is
    treated as a 1-D residual under Student-t with `df` degrees of freedom
    and `scale`. A Gaussian-with-bounded-uniform mixture is the spec's
    documented alternative; pick by injecting a different `EmissionFactor`
    (the protocol is the only contract).

    The Student-t pdf is symmetric over R; perpendicular distance is
    non-negative, so strictly we're using the right half. The factor-of-two
    half-distribution constant is omitted because it cancels in the
    forward-backward marginals — only relative log-potentials matter.

    `scale` is in metres; calibration sets it from residual magnitudes.
    `df=4` is a standard heavy-tail default; lower `df` is heavier tail.
    """

    def __init__(
        self,
        scale: float,
        network: "RoadNetwork",
        df: float = 4.0,
    ) -> None:
        if scale <= 0:
            raise ValueError(f"scale must be positive, got {scale}")
        if df <= 0:
            raise ValueError(f"df must be positive, got {df}")
        self.scale = float(scale)
        self.df = float(df)
        self.network = network
        # log of the Student-t normalising constant. Constant in (scale, df);
        # log_potential subtracts only the data-dependent piece per call.
        self._log_norm = (
            math.lgamma((self.df + 1) / 2)
            - math.lgamma(self.df / 2)
            - 0.5 * math.log(self.df * math.pi)
            - math.log(self.scale)
        )

    def log_potential(
        self, obs: CollapsedObservation, state: State,
    ) -> float:
        net = self.network
        try:
            idx = net.edge_index_for_link(int(state.link_id))
        except KeyError:
            return float("-inf")

        geom = net.geoms[idx]
        deg_len = geom.length
        m_len = float(net.lengths_m[idx])
        if deg_len <= 0.0 or m_len <= 0.0:
            return float("-inf")

        # Translate the metric offset back to degree-space arc length, then
        # interpolate. Same linear-scaling assumption as project_point —
        # accurate while edges aren't extremely long.
        dist_along_deg = (state.offset / m_len) * deg_len
        pt = geom.interpolate(dist_along_deg)
        proj_lat = float(pt.y)
        proj_lon = float(pt.x)

        d = float(equirectangular_distance_m(
            obs.lat, obs.lon, proj_lat, proj_lon,
        ))
        z = d / self.scale
        return self._log_norm - ((self.df + 1) / 2) * math.log1p(z * z / self.df)

    def rebind(self, network: "RoadNetwork") -> "StudentTEmission":
        return StudentTEmission(scale=self.scale, network=network, df=self.df)


class ExponentialFamilyTransition:
    """Driver model η(p) ∝ exp(μᵀϕ(p)).

    Returns `-inf` for (state_k, path, state_{k+1}) triples where the path
    doesn't start at `state_k` or end at `state_{k+1}` (the δ/δ̄ indicators
    in the CRF). Otherwise returns `μ · ϕ(p)`. `time_budget` is accepted
    for protocol conformance and ignored by this factor.
    """

    def __init__(self, mu: np.ndarray) -> None:
        self.mu = np.asarray(mu, dtype=float)

    def log_potential(
        self,
        state_k: State,
        path: Path,
        state_kplus1: State,
        time_budget: float,
    ) -> float:
        if not path.starts_at(state_k) or not path.ends_at(state_kplus1):
            return float("-inf")
        if path.feature_vector.shape != self.mu.shape:
            raise ValueError(
                f"path feature_vector shape {path.feature_vector.shape} "
                f"does not match mu shape {self.mu.shape}",
            )
        return float(self.mu @ path.feature_vector)

    def rebind(self, network: "RoadNetwork") -> "ExponentialFamilyTransition":
        return self
