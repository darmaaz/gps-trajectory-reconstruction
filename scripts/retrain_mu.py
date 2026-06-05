"""Train `mu` from cached 15s-aligned labels, then demo opinionated paths.

Fast side of the supervised-training pipeline:

    1. Load `cache/labeled_trips_15s.pkl.gz`
       (build via `scripts/compute_15s_labels.py`).
    2. Run `fit_supervised` with light L2 regularisation.
    3. Save the trained vector to `src/data/mu_default.npy`.
    4. Reconstruct three canonical Porto trips with the new μ and print
       the top-3 candidate paths per transition with their posterior
       weights. Shows the path posterior really IS opinionated rather
       than uniform.

Run `compute_15s_labels.py` first; this script is a few seconds, that
one is the ~50-minute build.
"""

from __future__ import annotations

import gzip
import os
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("GPS_RECON_BBOX_LAT", "40.5,42.5")
os.environ.setdefault("GPS_RECON_BBOX_LON", "-9.5,-7.0")

from scripts._data_paths import osm_pbf_path, porto_csv_path    # noqa: E402

from src.api import reconstruct_trajectory    # noqa: E402
from src.config import Config    # noqa: E402
from src.data import default_mu    # noqa: E402
from src.feeds import iter_porto_trips    # noqa: E402
from src.model import (    # noqa: E402
    ExponentialFamilyTransition, FEATURE_DIM, StudentTEmission,
)
from src.network import load_osm_network    # noqa: E402
from src.training import fit_supervised    # noqa: E402

CACHE = Path(__file__).resolve().parents[1] / "cache"
PBF = osm_pbf_path()
CSV = porto_csv_path()
OSM_CACHE = CACHE / "pt_edges.parquet"
LABELS_CACHE = CACHE / "labeled_trips_15s.pkl.gz"

MU_OUT = Path(__file__).resolve().parents[1] / "src" / "data" / "mu_default.npy"

DEMO_TRIPS = {
    "SHORT":  "1372637091620000337",
    "MEDIUM": "1372636951620000320",
    "LONG":   "1372639536620000570",
}

FEATURE_NAMES = [
    "length_km", "n_left_turns", "n_right_turns",
    "n_signals", "n_stop_signs",
    "frac_motorway", "frac_trunk", "frac_primary",
    "frac_secondary", "frac_tertiary",
    "frac_residential", "frac_service",
    "travel_time_min",
    "dwell_min", "dwell_ratio",
    "start_perp_10m", "end_perp_10m",
    "n_intersections",
]


def _log(m: str) -> None:
    print(f"[retrain] {m}", file=sys.stderr, flush=True)


def _load_labels(path: Path = LABELS_CACHE):
    if not path.exists():
        raise SystemExit(
            f"label cache not found at {path}. "
            f"Run `python scripts/compute_15s_labels.py` first.",
        )
    with gzip.open(path, "rb") as f:
        payload = pickle.load(f)
    header = payload["header"]
    if header["feature_dim"] != FEATURE_DIM:
        raise SystemExit(
            f"label cache feature_dim={header['feature_dim']} but "
            f"current FEATURE_DIM={FEATURE_DIM}. Regenerate the cache.",
        )
    return payload


def _print_mu(mu: np.ndarray, scale: float) -> None:
    print()
    print(f"Trained mu (FEATURE_DIM={FEATURE_DIM}):")
    for i, (name, val) in enumerate(zip(FEATURE_NAMES, mu)):
        print(f"  mu[{i:2d}] {name:25s} = {val: .6f}")
    print(f"  scale = {scale:.4f} metres")
    print()
    sign_gates = {
        " 0 length_m":              mu[0],
        "13 inferred_dwell_s":      mu[13],
        "14 inferred_dwell/budget": mu[14],
        "15 start_perp_m":          mu[15],
        "16 end_perp_m":            mu[16],
    }
    print("Sanity sign checks (≤ 0 expected for a real driver-preference fit):")
    for label, val in sign_gates.items():
        marker = "PASS" if val <= 0.0 else "WARN"
        print(f"  [{marker}] mu[{label:30s} = {val: .6f}")


def _demo_opinionated_posteriors(network, mu: np.ndarray, scale: float) -> None:
    """Reconstruct three Porto trips with the trained μ and print the
    top-3 candidate paths per transition with their posterior weights.

    If the posterior is genuinely informed by μ, the weights should be
    skewed (e.g. 0.7 / 0.2 / 0.1) rather than near-uniform.
    """
    print()
    print("=" * 72)
    print("Opinionated posteriors with the trained μ — top 3 paths per transition")
    print("=" * 72)

    emission = StudentTEmission(scale=scale, network=network)
    transition = ExponentialFamilyTransition(mu)
    config = Config(emission=emission, transition=transition)

    target_ids = set(DEMO_TRIPS.values())
    raws_by_id = {}
    for tid, raw in iter_porto_trips(CSV, min_pings=10):
        if tid in target_ids:
            raws_by_id[tid] = raw
            if len(raws_by_id) == len(target_ids):
                break

    for label, tid in DEMO_TRIPS.items():
        if tid not in raws_by_id:
            print(f"\n[{label}] trip {tid} not found in CSV.")
            continue
        print(f"\n[{label}] trip {tid} ({len(raws_by_id[tid])} raw pings)")
        segs = reconstruct_trajectory(raws_by_id[tid], network, config)
        if not segs:
            print("  empty reconstruction")
            continue
        for s_i, seg in enumerate(segs):
            print(f"  segment {s_i}: {len(seg.canonical_timestamps)} obs, "
                  f"{len(seg.path_marginals)} transitions, "
                  f"log_Z = {seg.log_partition:.3f}")
            for k, marg in enumerate(seg.path_marginals[:3]):    # first 3 transitions
                ranked = sorted(marg.items(), key=lambda kv: kv[1], reverse=True)[:3]
                if not ranked:
                    continue
                top_w = ranked[0][1]
                entropy_check = "OPINIONATED" if top_w >= 0.6 else (
                    "MILD" if top_w >= 0.35 else "UNIFORM-ISH"
                )
                print(f"    transition {k} ({entropy_check}, top weight {top_w:.3f}):")
                for r, (path, weight) in enumerate(ranked):
                    print(f"      [{r}] weight={weight:.3f}  "
                          f"edges={len(path.edges)}  "
                          f"length={path.length_meters:.0f}m  "
                          f"travel={path.expected_travel_time:.1f}s  "
                          f"dwell={path.inferred_dwell:.1f}s")
            if len(seg.path_marginals) > 3:
                print(f"    ... ({len(seg.path_marginals) - 3} more transitions)")


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", type=Path, default=LABELS_CACHE,
                    help="input label cache path")
    ap.add_argument("--out", type=Path, default=MU_OUT,
                    help="output mu .npy path")
    args = ap.parse_args()

    _log(f"loading label cache {args.labels}…")
    payload = _load_labels(args.labels)
    trips = [trip for _tid, trip in payload["trips"]]
    _log(f"  {len(trips)} labelled segments, header: {payload['header']}")

    _log("loading Portugal network…")
    network = load_osm_network(PBF, cache_path=OSM_CACHE)
    _log(f"  network: {len(network)} edges")

    _log("running supervised fit…")
    mu_star, scale_star = fit_supervised(
        trips, network,
        max_iter=300,
        l2_reg=1.0,
        fix_scale=10.0,    # pin emission scale to inference value; avoids
                            # the deterministic-label degeneracy that
                            # pegged scale at its lower bound previously.
    )

    _print_mu(mu_star, scale_star)

    args.out.parent.mkdir(exist_ok=True)
    np.save(args.out, mu_star.astype(float))
    _log(f"wrote {args.out}")

    _demo_opinionated_posteriors(network, mu_star, scale_star)


if __name__ == "__main__":
    main()
