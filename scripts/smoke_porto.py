"""End-to-end smoke run on the Porto Kaggle taxi dataset.

Loads:
    - Portugal OSM PBF (cached parquet of edges on subsequent runs)
    - Porto Kaggle CSV, streamed trip-by-trip
    - Builds a Config with un-calibrated heuristic priors (same as smoke.py)

Runs `reconstruct_trajectory` for the first `--limit-trips` trips that have
at least `--min-pings` pings. Prints one diagnostic row per trip — enough to
confirm the pipeline produces sensible output on Porto data before
committing time to calibration. NOT a benchmark.

Defaults pick a small batch (5 trips, ≥30 pings each) so the first run is
quick. Override with --limit-trips / --min-pings.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

# Make `src.*` importable when run as a script from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Loosen the geo bbox before src.config evaluates it. Porto is at ~41°N /
# -8°W, way outside the Mexico default. Setting the env var here keeps
# the config module data-source-agnostic.
os.environ.setdefault("GPS_RECON_BBOX_LAT", "40.5,42.5")
os.environ.setdefault("GPS_RECON_BBOX_LON", "-9.5,-7.0")

from scripts._data_paths import osm_pbf_path, porto_csv_path    # noqa: E402

from src.api import reconstruct_trajectory    # noqa: E402
from src.config import Config    # noqa: E402
from src.feeds import iter_porto_trips    # noqa: E402
from src.model import (    # noqa: E402
    ExponentialFamilyTransition, FEATURE_DIM, StudentTEmission,
)
from src.network import load_osm_network    # noqa: E402

DEFAULT_PBF = osm_pbf_path()
DEFAULT_CSV = porto_csv_path()
OSM_CACHE = Path(__file__).resolve().parents[1] / "cache" / "pt_edges.parquet"


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pbf", type=Path, default=DEFAULT_PBF,
                        help="Portugal OSM PBF path.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV,
                        help="Porto Kaggle train.csv path.")
    parser.add_argument("--limit-trips", type=int, default=5,
                        help="Number of trips to reconstruct.")
    parser.add_argument("--min-pings", type=int, default=30,
                        help="Skip trips shorter than this (Porto is 15s "
                             "sampled, so 30 pings ≈ 7.5 min).")
    parser.add_argument("--scale", type=float, default=10.0,
                        help="Initial Student-t scale (m). Heuristic.")
    args = parser.parse_args()

    # --- stage 1: OSM ----------------------------------------------------
    if not args.pbf.exists():
        _log(f"FATAL: OSM PBF not found at {args.pbf}")
        _log("Get it from https://download.geofabrik.de/europe/portugal.html")
        return 2
    _log(f"loading network from {args.pbf.name}"
         + (f" (cache: {OSM_CACHE.name})" if OSM_CACHE.exists() else ""))
    t0 = time.perf_counter()
    network = load_osm_network(args.pbf, cache_path=OSM_CACHE)
    _log(f"  network: {len(network)} edges, "
         f"{len(network.node_positions)} nodes "
         f"({time.perf_counter() - t0:.1f}s)")

    # --- stage 2: build config -------------------------------------------
    emission = StudentTEmission(scale=args.scale, network=network)
    mu = np.zeros(FEATURE_DIM)
    mu[0] = -0.001    # mild length penalty (uncalibrated)
    mu[1] = mu[2] = -0.5
    mu[12] = -0.01
    transition = ExponentialFamilyTransition(mu)
    config = Config(emission=emission, transition=transition)

    # --- stage 3: stream trips ------------------------------------------
    if not args.csv.exists():
        _log(f"FATAL: Porto CSV not found at {args.csv}")
        _log("Get it from "
             "https://www.kaggle.com/competitions/pkdd-15-predict-taxi-"
             "service-trajectory-i/data")
        return 2

    print()
    print(f"{'trip_id':>20} | {'pings':>5} | {'segs':>4} | "
          f"{'paths':>5} | {'log Z':>9} | {'most_likely link sequence (first 6)'}")
    print("-" * 100)

    n_done = 0
    trip_iter = iter_porto_trips(
        args.csv, min_pings=args.min_pings,
    )
    for trip_id, raw in trip_iter:
        if n_done >= args.limit_trips:
            break
        t0 = time.perf_counter()
        segments = reconstruct_trajectory(raw, network, config)
        elapsed = time.perf_counter() - t0

        if not segments:
            print(f"{trip_id:>20} | {len(raw):>5} | {'0':>4} | "
                  f"{'-':>5} | {'-':>9} | (no usable segments)")
            n_done += 1
            continue

        # Aggregate across segments for the one-line summary.
        seg_summaries = []
        for seg in segments:
            n_obs = len(seg.state_marginals)
            n_paths = (
                np.mean([len(pm) for pm in seg.path_marginals])
                if seg.path_marginals else 0.0
            )
            states_in_ml = [seg.most_likely[2 * k] for k in range(n_obs)]
            link_preview = ", ".join(str(s.link_id) for s in states_in_ml[:6])
            if len(states_in_ml) > 6:
                link_preview += " …"
            seg_summaries.append((n_obs, n_paths, seg.log_partition, link_preview))

        biggest = max(seg_summaries, key=lambda s: s[0])
        print(
            f"{trip_id:>20} | {len(raw):>5} | {len(segments):>4} | "
            f"{biggest[1]:>5.1f} | {biggest[2]:>9.2f} | {biggest[3]} "
            f"({elapsed:.1f}s)"
        )
        n_done += 1

    print()
    _log(f"done; {n_done} trip(s) processed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
