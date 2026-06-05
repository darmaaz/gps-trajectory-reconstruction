"""DRAFT — path-matched oracle: isolate dwell-rule error from path error.

The §4 oracle currently averages position error over *all* truth pings.
But a transition whose MLE path is the wrong road entirely contributes
huge error to every ping in its window — error that no dwell-rule choice
can fix. That pollutes the "rule-choice cost" story: it mixes
path-selection error with dwell-timing error.

This draft conditions on **path-matched** transitions — those where the
120 s MLE path's edges coincide with the native-15 s MLE edges in the
same window (Jaccard ≥ threshold). On that subset the path is
essentially right, so the remaining per-ping error is almost purely
*dwell timing*. Comparing the all-transitions oracle against the
path-matched oracle shows how much of the headline error was really
path-selection, not rule choice.

Run:  python scripts/demo_oracle_pathmatched_draft.py
Prints, per canonical trip: per-rule + oracle medians over ALL pings vs
over PATH-MATCHED pings, plus the path-match rate. Saves a comparison
bar chart to cache/demo_oracle_pathmatched_draft.png.
"""

from __future__ import annotations

import os
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("GPS_RECON_BBOX_LAT", "40.5,42.5")
os.environ.setdefault("GPS_RECON_BBOX_LON", "-9.5,-7.0")

from scripts._data_paths import osm_pbf_path, porto_csv_path    # noqa: E402

from src.api import position_at_time, reconstruct_trajectory     # noqa: E402
from src.config import Config                                    # noqa: E402
from src.data import default_mu                                  # noqa: E402
from src.feeds import iter_porto_trips                           # noqa: E402
from src.geo import haversine_m                                  # noqa: E402
from src.model import (                                          # noqa: E402
    ExponentialFamilyTransition, Path as ModelPath, StudentTEmission,
)
from src.network import load_osm_network                         # noqa: E402
from src.preprocessing import clean, drop_kinematic_spikes       # noqa: E402

CSV = porto_csv_path()
PBF = osm_pbf_path()
OSM_CACHE = Path("cache/pt_edges.parquet")
OUT_PNG = Path("cache/demo_oracle_pathmatched_draft.png")

TRIPS = {"SHORT": "1372637091620000337",
         "MEDIUM": "1372636951620000320",
         "LONG": "1372639536620000570"}
RULES = ("front", "back", "spread")
MATCH_JACCARD = 0.5   # MLE-vs-truth edge overlap to call a transition "path-matched"


def _cfg(network):
    return Config(
        emission=StudentTEmission(scale=10.0, network=network),
        transition=ExponentialFamilyTransition(default_mu()),
        enable_offroad_candidates=True,
    )


def _truth_edges_in_window(segs_15, t_lo, t_hi):
    """Edge ids traversed by the native-15 s Viterbi MLE inside (t_lo, t_hi]."""
    edges = set()
    for seg in segs_15:
        ts = seg.canonical_timestamps
        for j in range(len(ts) - 1):
            if ts[j] >= t_hi or ts[j + 1] <= t_lo:
                continue
            step = seg.most_likely[2 * j + 1]
            if isinstance(step, ModelPath):
                edges.update(int(e) for e in step.edges)
    return edges


def _jaccard(a, b):
    a, b = set(a), set(b)
    u = a | b
    return len(a & b) / len(u) if u else 0.0


def _median(xs):
    return float(np.median(xs)) if xs else float("nan")


def main() -> None:
    print("loading network…", file=sys.stderr)
    network = load_osm_network(PBF, cache_path=OSM_CACHE)

    summary = []   # (trip, n_all, n_matched, match_rate, all_meds, matched_meds)
    for label, tid in TRIPS.items():
        raw = next(r for t, r in iter_porto_trips(CSV, min_pings=20) if t == tid)
        truth = drop_kinematic_spikes(clean(raw))
        segs_15 = reconstruct_trajectory(raw, network, _cfg(network))
        segs_120 = reconstruct_trajectory(raw[::8], network, _cfg(network))

        # Per-ping error buckets, split by whether the transition is path-matched.
        buckets = {"all": {r: [] for r in RULES}, "matched": {r: [] for r in RULES}}
        oracle = {"all": [], "matched": []}
        n_trans_all = n_trans_matched = 0

        for seg in segs_120:
            for k in range(len(seg.path_marginals)):
                mle = seg.most_likely[2 * k + 1]
                if not isinstance(mle, ModelPath):
                    continue
                n_trans_all += 1
                t_lo = seg.canonical_timestamps[k]
                t_hi = seg.canonical_timestamps[k + 1]
                truth_edges = _truth_edges_in_window(segs_15, t_lo, t_hi)
                matched = (
                    bool(truth_edges)
                    and _jaccard(mle.edges, truth_edges) >= MATCH_JACCARD
                )
                if matched:
                    n_trans_matched += 1
                for o in truth:
                    if not (t_lo < o.timestamp < t_hi):
                        continue
                    pe = {}
                    for r in RULES:
                        p = position_at_time([seg], o.timestamp, network, rule=r)
                        if p is not None:
                            pe[r] = haversine_m(o.lat, o.lon, *p)
                    if len(pe) != len(RULES):
                        continue
                    for r, e in pe.items():
                        buckets["all"][r].append(e)
                        if matched:
                            buckets["matched"][r].append(e)
                    best = min(pe.values())
                    oracle["all"].append(best)
                    if matched:
                        oracle["matched"].append(best)

        all_meds = {r: _median(buckets["all"][r]) for r in RULES}
        all_meds["oracle"] = _median(oracle["all"])
        matched_meds = {r: _median(buckets["matched"][r]) for r in RULES}
        matched_meds["oracle"] = _median(oracle["matched"])
        rate = n_trans_matched / max(n_trans_all, 1)
        summary.append((label, n_trans_all, n_trans_matched, rate,
                        all_meds, matched_meds))

    # Print.
    print()
    print(f"Path-matched := 120s MLE edges vs 15s-truth edges Jaccard ≥ {MATCH_JACCARD}")
    print()
    hdr = f"{'trip':8s} {'set':9s} {'matched%':>8s}  " + "  ".join(f"{c:>8s}" for c in ("front", "back", "spread", "ORACLE"))
    print(hdr); print("-" * len(hdr))
    for label, n_all, n_m, rate, am, mm in summary:
        for setname, meds in (("all", am), ("path-match", mm)):
            rate_s = f"{100*rate:.0f}%" if setname == "all" else ""
            row = "  ".join(f"{meds[c]:>8.1f}" for c in ("front", "back", "spread", "oracle"))
            print(f"{label:8s} {setname:9s} {rate_s:>8s}  {row}")
        print()

    # Bar chart: oracle (all) vs oracle (path-matched) per trip, plus
    # front (all) for reference.
    trips = [s[0] for s in summary]
    x = np.arange(len(trips)); width = 0.27
    fig, ax = plt.subplots(figsize=(9, 5))
    series = [
        ("front (all)",        [s[4]["front"] for s in summary],  "#94a3b8"),
        ("oracle (all)",       [s[4]["oracle"] for s in summary], "#f59e0b"),
        ("oracle (path-match)",[s[5]["oracle"] for s in summary], "#16a34a"),
    ]
    for j, (name, vals, color) in enumerate(series):
        bars = ax.bar(x + (j - 1) * width, vals, width, label=name,
                      color=color, edgecolor="white")
        for b, v in zip(bars, vals):
            if np.isfinite(v):
                ax.text(b.get_x() + b.get_width()/2, v + 3, f"{v:.0f}",
                        ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(trips)
    ax.set_ylabel("median position error (m)")
    ax.set_title("Oracle floor: all transitions vs path-matched only\\n"
                 "(path-matched isolates pure dwell-timing error)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    OUT_PNG.parent.mkdir(exist_ok=True)
    plt.savefig(OUT_PNG, dpi=130, bbox_inches="tight")
    print(f"wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
