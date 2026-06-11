"""Compute labelled training trips by snapping 120s reconstructions to
native 15s Viterbi.

Slow side of the supervised-training pipeline. For each Porto trip:

  1. Reconstruct at native 15s cadence → Viterbi MLE gives a dense
     timestamp-to-state mapping treated as ground truth.
  2. Downsample raw observations to 120s (every 8th).
  3. Run preprocessing → state_candidates → path_candidates on the 120s
     version. These are what supervised training will fit against.
  4. For each 120s observation, label the state candidate matching the
     15s MLE state (link_id match, smallest offset distance).
  5. For each 120s transition, label the path candidate with highest
     Jaccard edge-overlap against the 15s MLE path edges in that window.
  6. Cache the resulting `LabeledTrip` to a pickle.

The fast side, `scripts/retrain_mu.py`, loads the cache and runs
`fit_supervised` in seconds.

Skip rules (a trip is dropped if):
  * fewer than 5 collapsed 120s observations
  * any 120s observation has no state candidate matching the 15s MLE
    state's link_id
  * any 120s transition has empty path candidates
  * the 15s reconstruction has a discontinuity inside the trip's time
    span that breaks the label-to-candidate mapping

Output: `cache/labeled_trips_15s.pkl.gz`.

The 15s reconstruction uses a generic length prior (w=2) and off-road
candidates by default — the recipe the shipped `mu_default.npy` is trained on,
so a no-argument run reproduces it. Pass `--prior zero --no-offroad` for the
legacy emission-only recipe.
"""

from __future__ import annotations

import gzip
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import shapely

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("GPS_RECON_BBOX_LAT", "40.5,42.5")
os.environ.setdefault("GPS_RECON_BBOX_LON", "-9.5,-7.0")

from scripts._data_paths import osm_pbf_path, porto_csv_path    # noqa: E402

from src.api import reconstruct_trajectory    # noqa: E402
from src.candidates import (    # noqa: E402
    enumerate_paths_per_transition, project_observation,
)
from src.config import Config    # noqa: E402
from src.feeds import iter_porto_trips    # noqa: E402
from src.model import (    # noqa: E402
    CollapsedObservation, ExponentialFamilyTransition, FEATURE_DIM,
    Path as ModelPath, RawObservation, State, StudentTEmission,
)
from src.geo import equirectangular_distance_m    # noqa: E402
from src.network import load_osm_network    # noqa: E402
from src.preprocessing import (    # noqa: E402
    clean, collapse_by_uniqueness, drop_kinematic_spikes,
    drop_replay_bursts, flag_stale_runs,
)
from src.training import LabeledTrip    # noqa: E402

CACHE = Path(__file__).resolve().parents[1] / "cache"
PBF = osm_pbf_path()
CSV = porto_csv_path()
OSM_CACHE = CACHE / "pt_edges.parquet"
LABELS_CACHE = CACHE / "labeled_trips_15s.pkl.gz"

N_TRIPS_TARGET = 400      # source-trip count to consume; each trip yields
                          # zero or more LabeledTrip segments depending on
                          # how many viable runs of transitions it has
MIN_PINGS = 40            # ≥ 5 obs after 8x downsampling
MAX_PINGS = 200           # bound per-trip cost; long trips are slow
DOWNSAMPLE_STRIDE = 8     # raw is 15s; 8x = 120s


def _log(m: str) -> None:
    print(f"[labels] {m}", file=sys.stderr, flush=True)


def _preprocess_to_collapsed(
    raw: list[RawObservation], net, config: Config,
) -> list[CollapsedObservation]:
    """Same prep stack as `reconstruct_trajectory`, exposed as a function
    so the labelling pipeline can build state/path candidates from the
    intermediates."""
    cleaned = clean(raw)
    cleaned = drop_kinematic_spikes(
        cleaned,
        spike_speed_ms=config.spike_speed_ms,
        bridge_speed_ms=config.spike_bridge_speed_ms,
        max_spike_length=config.spike_max_length,
    )
    collapsed = collapse_by_uniqueness(cleaned, config.collapse_epsilon)
    collapsed = flag_stale_runs(collapsed, net, config.max_speed_factor)
    collapsed = drop_replay_bursts(
        collapsed,
        max_speed_ms=config.replay_max_speed_ms,
        max_speed_factor=config.max_speed_factor,
        k_consistent=config.replay_k_consistent,
        moving_threshold_ms=config.replay_moving_threshold_ms,
    )
    return [o for o in collapsed if not o.dropped_during_replay]


def _build_time_to_state_lookup(
    segments,
) -> dict:
    """`{timestamp → (link_id, offset)}` from the 15s Viterbi MLE."""
    out: dict = {}
    for seg in segments:
        for k, t in enumerate(seg.canonical_timestamps):
            state = seg.most_likely[2 * k]
            out[t] = (int(state.link_id), float(state.offset))
    return out


def _closest_state(
    cands: list[State], truth_link: int, truth_offset: float, net,
) -> int | None:
    """Pick the candidate at this obs whose link matches the 15s truth
    and whose offset is closest. Falls back to the truth link's
    opposite-direction TWIN (same physical street; offset measured from
    the other end) — without the fallback, a 15s truth state on one twin
    and a 120s projection on the other silently dropped the whole
    segment. None if neither matches."""
    matches = [
        (i, c) for i, c in enumerate(cands) if int(c.link_id) == truth_link
    ]
    if matches:
        return min(
            matches, key=lambda ic: abs(float(ic[1].offset) - truth_offset),
        )[0]
    try:
        t_idx = net.edge_index_for_link(truth_link)
    except KeyError:
        return None
    twin = int(net.twin_indices()[t_idx])
    if twin == -1:
        return None
    twin_link = int(net.edge_ids[twin])
    flipped = float(net.lengths_m[twin]) - truth_offset
    matches = [
        (i, c) for i, c in enumerate(cands) if int(c.link_id) == twin_link
    ]
    if not matches:
        return None
    return min(
        matches, key=lambda ic: abs(float(ic[1].offset) - flipped),
    )[0]


def _segment_keys(net, edge_ids) -> set[tuple[int, int]]:
    """Undirected physical-road keys for a link-id iterable (twin-spelling
    safe — see src/network/identity.py). Unknown links skipped."""
    out: set[tuple[int, int]] = set()
    for e in edge_ids:
        try:
            out.add(net.segment_key_for_link(int(e)))
        except KeyError:
            continue
    return out


def _truth_segments_in_window(
    segments, t_lo, t_hi, net,
) -> set[tuple[int, int]]:
    """Undirected segment keys traversed by the 15s MLE inside
    `(t_lo, t_hi]`. Keyed on physical-road identity, NOT raw link ids:
    the 15s truth and a 120s candidate frequently spell the same street
    via opposite twins, which a directed-id Jaccard scores as disjoint."""
    edges: set[int] = set()
    for seg in segments:
        ts = seg.canonical_timestamps
        for j in range(len(ts) - 1):
            # Include transition j if any portion of it falls inside the
            # 120s window. Liberal: catches partial overlaps so short
            # window transitions still pick up the right path.
            if ts[j] >= t_hi or ts[j + 1] <= t_lo:
                continue
            step = seg.most_likely[2 * j + 1]
            if isinstance(step, ModelPath):
                edges.update(int(e) for e in step.edges)
    return _segment_keys(net, edges)


def _best_path_by_overlap(
    paths: list[ModelPath], truth_segs: set[tuple[int, int]], net,
) -> int | None:
    """Index of the candidate path with highest Jaccard overlap against
    the truth's undirected segment keys. Returns None on empty input."""
    if not paths:
        return None
    if not truth_segs:
        # No 15s edge info — fall back to shortest among candidates that
        # have any edges.
        return int(np.argmin([p.length_meters for p in paths]))
    best_score = -1.0
    best_i: int | None = None
    for i, p in enumerate(paths):
        cand_segs = _segment_keys(net, p.edges)
        if not cand_segs:
            continue
        inter = len(cand_segs & truth_segs)
        union = len(cand_segs | truth_segs)
        score = inter / union if union > 0 else 0.0
        # Tie-break by shortest length to prefer simpler paths.
        if score > best_score + 1e-9 or (
            abs(score - best_score) < 1e-9
            and best_i is not None
            and p.length_meters < paths[best_i].length_meters
        ):
            best_score = score
            best_i = i
    return best_i


def _nearest_state(cands: list[State]) -> int | None:
    """Raw-ping state label: index of the candidate nearest the observation
    (smallest perpendicular distance). No 15s reconstruction involved."""
    if not cands:
        return None
    return min(range(len(cands)), key=lambda i: float(cands[i].perp_m))


def _best_path_by_pings(
    paths: list[ModelPath], network, plons, plats,
) -> int | None:
    """Raw-ping path label: index of the candidate that best THREADS the raw
    15s pings in the window — minimises the largest ping→path distance (so all
    pings lie near one coherent path), tie-broken by shorter length. Falls back
    to the shortest path when the window has no raw pings. No 15s MLE involved.
    """
    valid = [(i, p) for i, p in enumerate(paths) if p.edges]
    if not valid:
        return None
    if len(plons) == 0:
        return min(valid, key=lambda ip: ip[1].length_meters)[0]
    pts = shapely.points(np.column_stack([plons, plats]))
    plats_a, plons_a = np.asarray(plats), np.asarray(plons)
    from src.network import path_polyline
    best_i, best_key = None, None
    for i, p in valid:
        # Offset-trimmed driven geometry — full edge unions credited
        # coverage to road the path never drove (terminal overshoot).
        pl = path_polyline(p, network)
        if pl.shape[0] < 2 or not np.isfinite(pl).all():
            continue
        geom = shapely.LineString(pl)
        near = shapely.get_point(shapely.shortest_line(pts, geom), 1)
        d = equirectangular_distance_m(
            plats_a, plons_a, shapely.get_y(near), shapely.get_x(near))
        key = (float(np.max(d)), float(p.length_meters))
        if best_key is None or key < best_key:
            best_key, best_i = key, i
    return best_i


MIN_SEGMENT_OBS = 3    # require ≥ 3 obs (i.e. ≥ 2 transitions) per segment


def _build_labeled_segments(
    trip_id: str, raw: list[RawObservation], network, base_config: Config,
    labels_from: str = "mle",
) -> list[tuple[str, LabeledTrip]]:
    """Return a list of `(segment_id, LabeledTrip)` — one per maximal run
    of consecutive transitions with non-empty `path_cands` and non-empty
    state_cands. Returns `[]` if no segment of ≥ `MIN_SEGMENT_OBS`
    observations qualifies.

    Transition-resilient labelling (replaces the older all-or-nothing
    `_build_one_labeled_trip`). A Porto trip with one failing transition
    in the middle previously contributed zero labelled transitions; now
    it contributes both flanking sub-segments. Diagnostic showed only
    ~13% of transitions fail individually but 44% of trips had ≥1
    failure, so this should recover roughly 3× the labelled data at the
    same slack.
    """
    if len(raw) < MIN_PINGS or len(raw) > MAX_PINGS:
        return []

    # 1) 15s reconstruction — only the MLE recipe needs it. The raw-pings
    #    recipe derives every label directly from the GPS pings, so it skips
    #    the 15s pass (and its spur/disconnection artifacts) entirely.
    segments_15s = None
    time_to_state = None
    if labels_from in ("mle", "hybrid"):    # both need 15s for STATE labels
        segments_15s = reconstruct_trajectory(raw, network, base_config)
        if not segments_15s:
            return []
        time_to_state = _build_time_to_state_lookup(segments_15s)
        if not time_to_state:
            return []

    # 2) Downsample to 120s.
    raw_120 = raw[::DOWNSAMPLE_STRIDE]
    if len(raw_120) < MIN_SEGMENT_OBS:
        return []

    # 3) Build the 120s subgraph + preprocess + candidates by hand.
    lats = [o.lat for o in raw_120]
    lons = [o.lon for o in raw_120]
    net = network.subgraph_for_bbox(
        min(lats), max(lats), min(lons), max(lons),
        buffer_m=base_config.subgraph_buffer_m,
    )
    if base_config.typical_speeds_by_class is not None:
        net.set_typical_speeds_by_class(base_config.typical_speeds_by_class)

    collapsed = _preprocess_to_collapsed(raw_120, net, base_config)
    if len(collapsed) < MIN_SEGMENT_OBS:
        return []

    state_cands = [
        project_observation(
            o, net,
            radius_meters=base_config.candidate_radius,
            max_candidates=base_config.max_state_candidates,
        )
        for o in collapsed
    ]

    confirmed_dwells = [
        0.0 if collapsed[k].stale_flagged
        else (collapsed[k].t_last - collapsed[k].t_first).total_seconds()
        for k in range(len(collapsed) - 1)
    ]
    time_budgets = [
        (collapsed[k + 1].t_first - collapsed[k].t_first).total_seconds()
        - confirmed_dwells[k]
        for k in range(len(collapsed) - 1)
    ]
    # The 120s candidate sets must be enumerated under the SAME
    # direction-violation regime the model will run with: if no candidate
    # ever carries a nonzero n_direction_violations feature, μ[18] is
    # unidentifiable in the supervised fit (zero gradient; L2 drags it to
    # 0 — which would make wrong-way traversal FREE at inference).
    path_cands = enumerate_paths_per_transition(
        state_cands, net, time_budgets,
        max_path_candidates=base_config.max_path_candidates,
        budget_slack=base_config.path_budget_slack,
        penalty_lambda=base_config.path_penalty_lambda,
        diversify_truncation=base_config.diversify_truncation,
        enable_direction_violation=base_config.enable_direction_violation,
        direction_violation_cost_factor=base_config.direction_violation_cost_factor,
    )

    # 4) Identify maximal runs of consecutive viable transitions. A
    #    transition k is viable iff state_cands[k], state_cands[k+1], and
    #    path_cands[k] are all non-empty. The maximal run [s_start, s_end)
    #    yields observation indices [s_start, s_end] (inclusive) — a
    #    segment of `s_end - s_start + 1` observations and
    #    `s_end - s_start` transitions.
    n_trans = len(path_cands)
    viable = [
        bool(path_cands[k]) and bool(state_cands[k]) and bool(state_cands[k + 1])
        for k in range(n_trans)
    ]
    runs: list[tuple[int, int]] = []
    k = 0
    while k < n_trans:
        if not viable[k]:
            k += 1
            continue
        start = k
        while k < n_trans and viable[k]:
            k += 1
        runs.append((start, k))    # (start_obs, end_obs); transitions [start, k)

    out: list[tuple[str, LabeledTrip]] = []
    for seg_idx, (s_start, s_end_excl) in enumerate(runs):
        s_end = s_end_excl    # observation end is inclusive at index s_end
        n_seg_obs = s_end - s_start + 1
        if n_seg_obs < MIN_SEGMENT_OBS:
            continue
        sub_obs = collapsed[s_start:s_end + 1]
        sub_state_cands = state_cands[s_start:s_end + 1]
        sub_path_cands = path_cands[s_start:s_end]
        sub_time_budgets = time_budgets[s_start:s_end]

        # 5) Label states. MLE = candidate matching the 15s truth link;
        #    raw-pings = candidate nearest the observation (min perp).
        label_state_idx: list[int] = []
        seg_ok = True
        for k, obs in enumerate(sub_obs):
            if labels_from == "raw-pings":
                idx = _nearest_state(sub_state_cands[k])
            else:
                truth = _nearest_truth(time_to_state, obs.t_first)
                if truth is None:
                    seg_ok = False
                    break
                idx = _closest_state(sub_state_cands[k], truth[0], truth[1], net)
            if idx is None:
                seg_ok = False
                break
            label_state_idx.append(idx)
        if not seg_ok:
            continue

        # 6) Label paths. MLE = edge-set Jaccard vs the 15s route; raw-pings =
        #    the candidate that best threads the raw GPS pings in the window.
        label_path_idx: list[int] = []
        for k in range(len(sub_obs) - 1):
            t_lo_seg = sub_obs[k].t_first
            t_hi_seg = sub_obs[k + 1].t_first
            if labels_from in ("raw-pings", "hybrid"):
                win = [o for o in raw if t_lo_seg < o.timestamp <= t_hi_seg]
                idx = _best_path_by_pings(
                    sub_path_cands[k], net,
                    [o.lon for o in win], [o.lat for o in win])
            else:
                truth_segs = _truth_segments_in_window(
                    segments_15s, t_lo_seg, t_hi_seg, net)
                idx = _best_path_by_overlap(sub_path_cands[k], truth_segs, net)
            if idx is None:
                seg_ok = False
                break
            label_path_idx.append(idx)
        if not seg_ok:
            continue

        out.append((
            f"{trip_id}_s{seg_idx}",
            LabeledTrip(
                observations=sub_obs,
                state_candidates=sub_state_cands,
                path_candidates=sub_path_cands,
                time_budgets=sub_time_budgets,
                label_state_idx=label_state_idx,
                label_path_idx=label_path_idx,
            ),
        ))

    return out


def _nearest_truth(time_to_state: dict, t) -> tuple[int, float] | None:
    """Return the (link_id, offset) at the timestamp in `time_to_state`
    closest to `t`. None if the lookup is empty.
    """
    if not time_to_state:
        return None
    if t in time_to_state:
        return time_to_state[t]
    # Linear scan over (typically) tens of timestamps — fine.
    best_key = min(time_to_state.keys(), key=lambda k: abs((k - t).total_seconds()))
    return time_to_state[best_key]


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prior", choices=["zero", "generic"], default="generic",
                    help="transition prior for the 15s reconstruction "
                         "(shipped default: generic; 'zero' = legacy "
                         "emission-only recipe)")
    ap.add_argument("--w-length", type=float, default=2.0,
                    help="length-penalty weight when --prior generic")
    ap.add_argument("--offroad", action=argparse.BooleanOptionalAction, default=True,
                    help="off-road candidates in the 15s reconstruction "
                         "(shipped default: on; --no-offroad = legacy)")
    ap.add_argument("--n", type=int, default=N_TRIPS_TARGET,
                    help="source-trip count to consume")
    ap.add_argument("--skip", type=int, default=0,
                    help="skip this many leading source trips (held-out splits)")
    ap.add_argument("--out", type=Path, default=LABELS_CACHE,
                    help="output label cache path")
    ap.add_argument("--labels-from", choices=["mle", "raw-pings", "hybrid"],
                    default="mle",
                    help="label source: 'mle' = 15s Viterbi MLE (shipped); "
                         "'raw-pings' = thread the raw GPS pings, no 15s pass; "
                         "'hybrid' = 15s-disambiguated STATE labels + raw-ping "
                         "PATH labels (diagnostic — isolates the state labeling)")
    ap.add_argument("--direction-violation",
                    action=argparse.BooleanOptionalAction, default=True,
                    help="direction-violation candidates in BOTH the 15s "
                         "truth reconstruction and the 120s candidate sets "
                         "(default on — required to identify μ[18]; "
                         "--no-direction-violation = pre-F5 recipe)")
    args = ap.parse_args()
    n_target, skip, out_path = args.n, args.skip, args.out

    _log(f"loading Portugal network (PBF cache: {OSM_CACHE})…")
    network = load_osm_network(PBF, cache_path=OSM_CACHE)
    _log(f"  network: {len(network)} edges")

    # Shipped recipe (these defaults reproduce mu_default.npy): a generic,
    # model-independent length prior + off-road candidates. At 15s the
    # emission dominates, but pure μ=0 (`--prior zero`) admits gratuitous
    # detour ("spur") labels at near-stationary junctions; the length prior
    # suppresses them without importing the trained μ (which would be
    # circular). Off-road covers Porto parking/idle maneuvers. Pass
    # `--prior zero --no-offroad` for the legacy emission-only recipe.
    if args.prior == "generic":
        from src.data import generic_prior_mu
        mu0 = generic_prior_mu(args.w_length)
    else:
        mu0 = np.zeros(FEATURE_DIM)
    emit = StudentTEmission(scale=15.0, network=network, df=4.0)
    trans = ExponentialFamilyTransition(mu0)
    # `enable_direction_violation` matters on BOTH sides: the 15s truth
    # reconstruction must be able to thread wrong-way corridors (else the
    # "truth" in direction-conflict windows is itself a forced loop and
    # the labels reward the wrong candidate), and the 120s candidates
    # must include violation paths so μ[18] has gradient.
    base_config = Config(emission=emit, transition=trans,
                         enable_offroad_candidates=args.offroad,
                         enable_direction_violation=args.direction_violation)
    _log(f"  prior={args.prior} w_length={args.w_length} "
         f"offroad={args.offroad} labels_from={args.labels_from} "
         f"direction_violation={args.direction_violation} "
         f"n={n_target} skip={skip}")

    labelled: list[tuple[str, LabeledTrip]] = []
    trips_consumed = 0
    trips_with_no_segments = 0
    total_transitions = 0
    t_start = time.time()
    seen = 0
    for tid, raw in iter_porto_trips(CSV, min_pings=MIN_PINGS):
        seen += 1
        if seen <= skip:
            continue    # held-out split: skip leading source trips
        if trips_consumed >= n_target:
            break    # trip-targeted: consume N source trips, take everything
        try:
            results = _build_labeled_segments(
                tid, raw, network, base_config, labels_from=args.labels_from)
        except Exception as exc:    # noqa: BLE001
            _log(f"  trip {tid}: error {type(exc).__name__}: {exc}")
            trips_consumed += 1
            continue
        trips_consumed += 1
        if not results:
            trips_with_no_segments += 1
            continue
        labelled.extend(results)
        total_transitions += sum(len(t.path_candidates) for _, t in results)
        if trips_consumed % 25 == 0:
            elapsed = time.time() - t_start
            rate = trips_consumed / elapsed
            need = max(0, n_target - trips_consumed)
            eta = need / max(rate, 1e-6)
            _log(
                f"  {trips_consumed}/{n_target} trips, "
                f"{len(labelled)} segments, "
                f"{total_transitions} transitions, "
                f"({trips_with_no_segments} trips no-segment), "
                f"{rate:.2f} trips/s, ETA {eta:.0f}s",
            )

    elapsed = time.time() - t_start
    _log(
        f"done: {len(labelled)} segments, {total_transitions} transitions, "
        f"{trips_consumed} source trips, "
        f"{trips_with_no_segments} no-segment, in {elapsed:.0f}s",
    )

    header = {
        "schema_version": 3,    # bumped — entries are segments, not trips
        "feature_dim": FEATURE_DIM,
        "n_segments": len(labelled),
        "n_transitions": total_transitions,
        "n_source_trips_consumed": trips_consumed,
        "downsample_stride": DOWNSAMPLE_STRIDE,
        "min_pings": MIN_PINGS,
        "max_pings": MAX_PINGS,
        "min_segment_obs": MIN_SEGMENT_OBS,
        "prior": args.prior,
        "w_length": args.w_length,
        "offroad": args.offroad,
        "labels_from": args.labels_from,
        "direction_violation": args.direction_violation,
        "path_match": "undirected-segment-jaccard",   # twin-spelling safe
        "skip": skip,
    }
    CACHE.mkdir(exist_ok=True)
    with gzip.open(out_path, "wb") as f:
        pickle.dump({"header": header, "trips": labelled}, f, protocol=5)
    _log(f"wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
