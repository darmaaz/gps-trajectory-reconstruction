"""Per-observation state candidate enumeration.

Each `CollapsedObservation` gets up to K candidate `State` link+offset
projections within `radius_meters`. If no edges fall within the initial
radius, the search expands up to a maximum (default 200 m); an empty
return signals off-network to the orchestrator, which splits the
trajectory at that observation.

Each candidate's `entry_time` is set to `obs.t_first` so dwell-aware
consumers can read the same intermediate-time provenance without
re-projecting.
"""

from __future__ import annotations

from ..model import CollapsedObservation, State, StateV1
from ..network import RoadNetwork

DEFAULT_RADIUS_M: float = 50.0
DEFAULT_MAX_RADIUS_M: float = 200.0
DEFAULT_K: int = 5


def project_observation(
    obs: CollapsedObservation,
    network: RoadNetwork,
    radius_meters: float = DEFAULT_RADIUS_M,
    max_candidates: int = DEFAULT_K,
    *,
    max_radius_meters: float = DEFAULT_MAX_RADIUS_M,
) -> list[State]:
    """Top-K (link_id, offset) candidates within `radius_meters`, with the
    radius expanded by powers of two up to `max_radius_meters` if the initial
    query yields nothing. Returns `[]` to signal off-network.
    """
    radius = radius_meters
    while True:
        hits = network.project_point(
            obs.lat, obs.lon, radius, max_candidates,
        )
        if hits:
            return [
                StateV1(
                    link_id=int(network.edge_ids[idx]),
                    offset=offset_m,
                    entry_time=obs.t_first,
                    perp_m=float(perp_m),
                )
                for (idx, offset_m, perp_m) in hits
            ]
        if radius >= max_radius_meters:
            return []
        radius = min(radius * 2, max_radius_meters)
