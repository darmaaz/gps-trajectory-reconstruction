"""Visualise a single Porto Kaggle taxi trip.

Two outputs:
    cache/visualization_porto.png   — static matplotlib figure
    cache/visualization_porto.html  — pannable folium map

Same rendering as `scripts/visualize.py`; only the data-loading layer
differs. Pass a trip id with `--trip-id`, or rely on the default which
walks the CSV and grabs the first trip with at least `--min-pings` pings.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Loosen the geo bbox before src.config evaluates it (Porto is at ~41°N,
# outside the Mexico default).
os.environ.setdefault("GPS_RECON_BBOX_LAT", "40.5,42.5")
os.environ.setdefault("GPS_RECON_BBOX_LON", "-9.5,-7.0")

from scripts._data_paths import osm_pbf_path, porto_csv_path    # noqa: E402

from scripts.viz_helpers import (    # noqa: E402
    compute_segments, print_segment_summary, render_html, render_png,
)
from src.config import Config    # noqa: E402
from src.feeds import iter_porto_trips    # noqa: E402
from src.data import default_mu    # noqa: E402
from src.model import (    # noqa: E402
    ExponentialFamilyTransition, FEATURE_DIM, StudentTEmission,
)
from src.network import load_osm_network    # noqa: E402

CACHE_DIR = Path(__file__).resolve().parents[1] / "cache"


def _out_paths(trip_id: str) -> tuple[Path, Path]:
    """PNG + HTML paths namespaced by trip-id so multiple runs preserve
    their outputs side by side rather than overwriting each other."""
    base = CACHE_DIR / f"visualization_porto_{trip_id}"
    return base.with_suffix(".png"), base.with_suffix(".html")
DEFAULT_PBF = osm_pbf_path()
DEFAULT_CSV = porto_csv_path()
OSM_CACHE = Path(__file__).resolve().parents[1] / "cache" / "pt_edges.parquet"


def _log(msg: str) -> None:
    print(f"[viz] {msg}", file=sys.stderr, flush=True)


def _pick_trip(
    csv_path: Path,
    trip_id: str | None,
    min_pings: int,
) -> tuple[str, list]:
    """Return `(trip_id, list[RawObservation])` for the requested trip.

    If `trip_id` is None, walks the CSV until the first trip with enough
    pings shows up — fast on the chunked reader, no full materialisation.
    """
    for tid, obs in iter_porto_trips(csv_path, min_pings=min_pings):
        if trip_id is None or tid == trip_id:
            return tid, obs
    raise SystemExit(
        f"no matching trip (trip_id={trip_id}, min_pings={min_pings})",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pbf", type=Path, default=DEFAULT_PBF)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--trip-id", type=str, default=None,
                        help="Specific TRIP_ID to visualise; default is the "
                             "first trip with ≥ --min-pings pings.")
    parser.add_argument("--min-pings", type=int, default=50,
                        help="Skip trips shorter than this. 50 ≈ 12.5 min.")
    parser.add_argument("--scale", type=float, default=10.0,
                        help="Initial Student-t scale (m). Heuristic.")
    args = parser.parse_args()

    _log(f"loading network from {args.pbf.name}"
         + (f" (cache: {OSM_CACHE.name})" if OSM_CACHE.exists() else ""))
    network = load_osm_network(args.pbf, cache_path=OSM_CACHE)
    _log(f"  {len(network)} edges, {len(network.node_positions)} nodes")

    _log("picking trip…")
    trip_id, raw = _pick_trip(args.csv, args.trip_id, args.min_pings)
    _log(f"  trip {trip_id}: {len(raw)} pings, "
         f"{raw[0].timestamp} → {raw[-1].timestamp}")

    emission = StudentTEmission(scale=args.scale, network=network)
    transition = ExponentialFamilyTransition(default_mu())
    config = Config(emission=emission, transition=transition)

    collapsed, path_cands, seg_results = compute_segments(raw, network, config)
    n_ok = sum(1 for r in seg_results if r["ok"])

    title = (
        f"Porto trip {trip_id}\n"
        f"{len(raw)} raw pings · {len(collapsed)} collapsed obs · "
        f"{len(seg_results)} segments ({n_ok} reconstructed) · "
        f"green = transition has paths, red dashed = transition empty"
    )
    png_path, html_path = _out_paths(trip_id)
    render_png(
        network=network, collapsed=collapsed, path_cands=path_cands,
        seg_results=seg_results, title=title, png_path=png_path,
    )
    render_html(
        network=network, collapsed=collapsed, path_cands=path_cands,
        seg_results=seg_results, html_path=html_path,
    )

    print()
    print(f"PNG : {png_path}")
    print(f"HTML: {html_path}")
    print_segment_summary(seg_results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
