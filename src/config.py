"""Pipeline configuration.

`Config` is the runtime configuration consumed by
`pipeline.reconstruct_trajectory`. Module-level constants cover the
kinematic caps and the per-road-class max-speed envelope used by the
network loader and stale-jump detection.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .model.factors import EmissionFactor, TransitionFactor

REPO_ROOT: Path = Path(__file__).resolve().parents[1]


# Kinematic sanity caps applied during ingest. Pairs outside these bounds are
# clock glitches or teleports; their derived per-pair kinematics are NaN'd so
# downstream consumers don't amplify the artefact.
MAX_REASONABLE_SPEED_MS: float = 100.0    # 360 km/h — above any real vehicle
MIN_REASONABLE_DT_S: float = 1.0          # sub-second pairs are clock glitches

# Per-class max-speed envelope, m/s. Used by the network loader as the
# `max_speed` fallback when an edge has no parseable OSM `maxspeed` tag, and
# by stale-jump detection in preprocessing as the kinematic envelope for
# `min_travel_time` (with `Config.max_speed_factor` slack on top).
V_MAX_MS: dict[str, float] = {
    "motorway": 45.0, "trunk": 45.0,
    "primary": 42.0,
    "secondary": 35.0,
    "tertiary": 25.0,
    "unclassified": 22.0,
    "residential": 12.0, "service": 12.0, "living_street": 12.0,
}
_LINK_PARENT: dict[str, str] = {
    "motorway_link": "motorway", "trunk_link": "trunk",
    "primary_link": "primary", "secondary_link": "secondary",
    "tertiary_link": "tertiary",
}
_V_MAX_FALLBACK: float = 40.0


def v_max_for(road_class: str | None) -> float:
    """Return v_max in m/s for a road class, resolving _link variants."""
    if road_class is None:
        return _V_MAX_FALLBACK
    rc = _LINK_PARENT.get(road_class, road_class)
    return V_MAX_MS.get(rc, _V_MAX_FALLBACK)


# Per-class TYPICAL speed envelope, m/s. Used as the routing-cost prior in
# path enumeration — "how fast do real vehicles actually drive on this
# class of road?" — distinct from the OSM `max_speed` cap (used by stale
# detection for feasibility). The bias being corrected: OSM `max_speed`
# tags are posted speed limits, and free-flow A* using them
# systematically underestimates expected travel time, which in turn
# overestimates `inferred_dwell`.
#
# Defaults are heuristic urban-driving best-guesses. For higher fidelity,
# compute per-class medians directly from Porto 15-second trajectories
# and override via `Config.typical_speeds_by_class`.
V_TYPICAL_MS: dict[str, float] = {
    "motorway": 27.5,        # ~99 km/h — realistic motorway free-flow
    "trunk": 22.0,           # ~79 km/h
    "primary": 13.5,         # ~49 km/h — urban primary with signals
    "secondary": 11.0,       # ~40 km/h
    "tertiary": 9.0,         # ~32 km/h
    "unclassified": 8.0,     # ~29 km/h
    "residential": 6.0,      # ~22 km/h
    "service": 4.0,          # ~14 km/h — parking-lot access
    "living_street": 3.0,    # ~11 km/h — shared pedestrian
}
_V_TYPICAL_FALLBACK: float = 8.0


def v_typical_for(road_class: str | None) -> float:
    """Return typical realised speed in m/s for a road class.

    Routing cost uses this rather than `v_max_for` so `expected_travel_time`
    reflects real driving, not posted limits. _link variants resolve to
    their parent class.
    """
    if road_class is None:
        return _V_TYPICAL_FALLBACK
    rc = _LINK_PARENT.get(road_class, road_class)
    return V_TYPICAL_MS.get(rc, _V_TYPICAL_FALLBACK)


@dataclass
class Config:
    """Runtime configuration for `pipeline.reconstruct_trajectory`.

    `emission` and `transition` are injected with no default so the dwell-aware
    transition factor can be swapped in without touching the orchestrator.
    """

    emission: "EmissionFactor"
    transition: "TransitionFactor"

    # Preprocessing
    collapse_epsilon: float = 5.0           # metres, position-uniqueness collapse
    max_speed_factor: float = 1.2           # slack on V_MAX for stale detection
    replay_max_speed_ms: float = 40.0       # 144 km/h — feasibility cap for
                                            # replay-burst transitions
    replay_k_consistent: int = 2            # consecutive consistent obs to
                                            # close a replay zone
    replay_moving_threshold_ms: float = 1.0  # ~3.6 km/h: above this, the
                                            # device is claiming motion
    spike_speed_ms: float = 150.0           # 540 km/h — entry/exit speed
                                            # threshold for chip-glitch
                                            # detection. Above any real
                                            # vehicle by a wide margin;
                                            # pings in the 100–150 m/s
                                            # zone are kept and absorbed
                                            # by the heavy-tailed emission.
    spike_bridge_speed_ms: float = 100.0    # bridge transition feasibility
                                            # cap for chip-glitch removal
    spike_max_length: int = 3               # max consecutive pings to treat
                                            # as one chip-glitch run; longer
                                            # runs survive and are isolated
                                            # downstream via segment splits

    # Network subgraph: bbox padding around the input observations when
    # narrowing the routing graph. Wide enough that any plausible route
    # between consecutive observations stays inside the bbox. 5 km is
    # generous for ~60 s sampling at highway speeds; bump for sparser
    # sampling or longer-distance gaps.
    subgraph_buffer_m: float = 5000.0

    # Candidate generation
    candidate_radius: float = 50.0          # metres, edge search around each obs
    max_state_candidates: int = 5
    max_path_candidates: int = 100
    # ^ Trip-level cap on candidate paths per transition after per-pair
    # enumeration and dedup. Sorted-by-fastest selection systematically
    # discards longer-but-feasible paths if the cap is tight, collapsing
    # coverage of slow real-driving routes. 100 keeps most penalty-
    # diversified paths intact while bounding forward-backward cost. Set
    # to a very large number to keep all feasible paths.
    path_budget_slack: float = 1.5
    # ^ Multiplier on the time budget passed to path enumeration; admission
    # cap uses `min_traversal_time ≤ slack × budget` where
    # `min_traversal_time = sum_edges(length / max_speed)` is the
    # physical lower bound on transit time for the path. Note: applying
    # slack to the budget is algebraically identical to allowing
    # `slack × max_speed` (a single factor moves freely across the
    # inequality), so slack = 1.5 ≡ "admit driving up to 1.5× the tagged
    # max speed."
    #
    # Why 1.5: the binding `max_speed` is usually the OSM `maxspeed` *tag*
    # (e.g. 50 km/h on Porto primaries), and real taxi driving there hits
    # 60-72 km/h — so a posted-limit cap spuriously rejects genuine
    # movement and shatters the (especially 15 s) reconstruction into
    # many empty-path segments. Admitting up to 1.5× the tag (75 km/h on a
    # 50 km/h road) covers observed behaviour. Per-transition anatomy on
    # the SHORT trip confirmed the splits were direct (non-detour) paths
    # failing purely on speed at slack 1.2.
    #
    # Admission is deliberately permissive: "is this physically possible",
    # not "is this likely". The likelihood (`μᵀϕ`) is meant to down-weight
    # unlikely-but-possible fast paths — though note that is currently
    # only indirect (via the dwell features); a graded speed-likelihood
    # factor `log P(implied_speed | road_class)` is the principled
    # extension (see QUESTIONS_DEFERRED.md). Admission on the **max-speed**
    # bound; `expected_travel_time` (typical-speed) feeds the likelihood
    # and dwell residual without doubling as the admission cap.
    path_penalty_lambda: float = 0.3
    # ^ Multiplicative surcharge per edge re-use during penalty-diversified
    # routing. Applied as `penalty *= (1 + λ)` so default 0.3 grows the
    # penalty 1.0 → 1.3 → 1.69 → …. Higher λ forces faster divergence
    # (more diverse paths, but the second-best path may become much longer
    # than the optimum).

    diversify_truncation: bool = True
    # ^ Spend the `max_path_candidates` cap on distinct *physical* routes:
    # the |src|×|dst| state-pair sweep yields the same corridor under many
    # directed spellings (terminal states projected onto opposite-direction
    # twins of the same street, or onto neighbouring corridor edges), and a
    # plain travel-time-sorted cut fills the cap with those spellings while
    # crowding out structurally different routes. When set, truncation keeps
    # the best path per `network.identity.canonical_route` first, then
    # back-fills with the best remaining spellings (set is never smaller
    # than the legacy cut). False recovers the legacy behaviour exactly.

    # Per-class typical speeds for routing cost. None ⇒ use the
    # `V_TYPICAL_MS` defaults. Override to inject data-driven Porto-derived
    # values (computed once from native 15s trajectories) without editing
    # module-level constants.
    typical_speeds_by_class: dict[str, float] | None = None

    # Direction-violation candidate paths. Default ON (promoted after the
    # held-out gate: capacity 11.0%→7.0% failing, edge-marginal 0.6-0.8 bin
    # 22pp→5pp gap, confident violation edges 96% truth-traversed; residual
    # cost = 1/200 windows over-eager violation. See diag_direction_conflict.py
    # for the prevalence diagnostic). Routing runs on a permissive graph where
    # every mapped one-way edge also has a penalized reverse arc, and one-way
    # terminal edges may be exited / entered backward. This admits the
    # maneuvers the legal directed graph cannot express — wrong-way streets
    # (taxis do this; OSM oneway tags are also sometimes wrong), parking-lot
    # pull-outs, mid-edge U-turns — instead of forcing every candidate into
    # around-the-block loops (the fig-1 failure). Violations are NOT free:
    # each wrong-way *maneuver* (run of consecutive reversed edges) counts
    # into feature slot [18] (n_direction_violation_runs), priced by μ — the
    # shipped dim-19 mu_default learned −2.78/maneuver.
    enable_direction_violation: bool = True
    direction_violation_cost_factor: float = 3.0
    # ^ Multiplicative weight surcharge on reverse arcs during search.
    # Pure steering: keeps enumeration preferring legal routes so
    # violations only surface when legal options are poor. Plausibility
    # pricing belongs to the feature/μ, not this factor.

    # Off-road / near-stationary candidate paths. Opt-in (default off):
    # when enabled, a transition whose only routed options are long
    # detours despite a short straight-line gap also gets a straight-line
    # off-road candidate, representing an off-network maneuver (parking,
    # arrival, idling near a one-way pair) that legal routing can't model.
    # See `routing.candidate_paths` for the trigger and OVERVIEW.md for
    # the rationale. Default ON as of the generic-prior μ retrain: Porto taxi
    # data has frequent parking/idle maneuvers, and the shipped μ + 15s labels
    # are co-generated with off-road enabled, so the production default must
    # match. (The low-level `candidate_paths`/`enumerate_paths_per_transition`
    # keep their own `enable_offroad=False` defaults for unit tests.)
    enable_offroad_candidates: bool = True
    offroad_max_straight_m: float = 120.0
    # ^ Trigger gate: only consider an off-road candidate when the
    # straight-line distance between the two states is below this. A
    # parking / arrival / idle maneuver is inherently small-radius; above
    # ~120 m a long routed path is plausibly real driving. Calibrated
    # against the 15s truth oracle (scripts/_diag_offroad_cutoff.py): on
    # an 80-trip Porto sample, transitions passing the detour+overslack
    # gates split cleanly — genuine maneuvers (vehicle max-excursion
    # < 150 m) had straight-line ≤ 100 m, real drives ≥ 130 m. 120 m sits
    # in that gap and gave 100% precision (only the true hallucination
    # fires, all real-drive regressions excluded).
    offroad_min_detour_ratio: float = 3.0
    # ^ Trigger gate: only when the best routed path's length is at least
    # this multiple of the straight-line distance — i.e. routing forces a
    # detour disproportionate to the crow-flight gap (the one-way-pair
    # signature).
    offroad_min_overslack: float = 1.0
    # ^ Trigger gate: only when the best routed detour is *overslacked* —
    # its typical-speed travel time exceeds `offroad_min_overslack ×
    # transit_budget`, meaning the vehicle could not have driven the
    # detour in the available time at typical speed (physically
    # implausible → likely an off-network maneuver). This gate is what
    # separates a hallucinated detour from a genuine short drive through
    # a one-way loop (which fits the budget and must stay routed). ALL
    # THREE gates must hold; the off-road candidate is then *added* to
    # compete, never replacing routed candidates.
