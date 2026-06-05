"""EXPLORATORY — what kind of segments populate the overconfident mid bin?

The edge-marginal calibration is overconfident in the moderate band and the
overconfidence survived both widening admission and max-vs-sum aggregation —
so it is intrinsic to *which* segments land there. This maps every candidate
segment in the trained-μ baseline back to its road class and metric length,
then anatomises the most overconfident mid bin (stated P in [0.4,0.6)):
within it, the traversed-vs-not split by road class and by length says what
the model over-credits.

No reconstruction — loads the network and the cached baseline records.
Run:  python scripts/_explore_midbin_anatomy.py
Saves cache/_explore_midbin_anatomy.png
"""

from __future__ import annotations

import pickle
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts._data_paths import osm_pbf_path    # noqa: E402
from src.network import load_osm_network          # noqa: E402

REPO = Path(__file__).resolve().parents[1]
OSM_CACHE = REPO / "cache" / "pt_edges.parquet"
PBF = osm_pbf_path()
RECORDS = REPO / "cache" / "_cluster_calib_records_indeptruth.pkl"   # trained μ, narrow
OUT_PNG = REPO / "cache" / "_explore_midbin_anatomy.png"

BIN_LO, BIN_HI = 0.4, 0.6          # the most overconfident mid bin (50% -> 37%)
LINK_PARENT = {"motorway_link": "motorway", "trunk_link": "trunk",
               "primary_link": "primary", "secondary_link": "secondary",
               "tertiary_link": "tertiary"}
LEN_BUCKETS = [0, 30, 60, 120, 250, 1e9]
LEN_LAB = ["<30m", "30-60", "60-120", "120-250", ">250m"]


def parent(rc):
    rc = str(rc)
    return LINK_PARENT.get(rc, rc)


def main():
    print("loading network…", file=sys.stderr)
    net = load_osm_network(PBF, cache_path=OSM_CACHE)
    fn, tn, rc, ln = net.from_node, net.to_node, net.road_classes, net.lengths_m
    key2meta = {}
    for i in range(len(fn)):
        a, b = int(fn[i]), int(tn[i])
        key = (a, b) if a <= b else (b, a)
        if key not in key2meta:
            key2meta[key] = (parent(rc[i]), float(ln[i]))

    with open(RECORDS, "rb") as f:
        records = pickle.load(f)["records"]

    # (P, traversed, road_class, length, n_paths) per candidate segment
    rows = []
    for r in records:
        if not r["paths"] or not r["truth_edges"]:
            continue
        P = defaultdict(float)
        for seg_set, w in r["paths"]:
            for e in seg_set:
                P[e] += w
        for e, p in P.items():
            meta = key2meta.get(e)
            if meta is None:
                continue
            rows.append((min(p, 1.0), e in r["truth_edges"],
                         meta[0], meta[1], len(r["paths"])))

    inbin = [x for x in rows if BIN_LO <= x[0] < BIN_HI]
    obs = np.mean([x[1] for x in inbin])
    print(f"\nbin [{BIN_LO},{BIN_HI}): {len(inbin)} segments | stated ~50% | "
          f"observed traversed {100*obs:.0f}%  (overconfidence = the gap)\n")

    # ---- by road class ----
    print("by road class (within the bin):")
    print(f"  {'class':>13}  {'n':>6}  {'bin share':>9}  {'traversed':>9}  "
          f"{'med len':>8}")
    cls = defaultdict(list)
    for p, t, c, L, npa in inbin:
        cls[c].append((t, L))
    for c in sorted(cls, key=lambda c: -len(cls[c])):
        ts = [t for t, _ in cls[c]]; ls = [L for _, L in cls[c]]
        print(f"  {c:>13}  {len(ts):>6}  {100*len(ts)/len(inbin):>8.0f}%  "
              f"{100*np.mean(ts):>8.0f}%  {np.median(ls):>7.0f}m")

    # ---- by length bucket ----
    print("\nby length (within the bin):")
    print(f"  {'length':>10}  {'n':>6}  {'traversed':>9}")
    bidx = np.digitize([x[3] for x in inbin], LEN_BUCKETS) - 1
    for b in range(len(LEN_LAB)):
        sel = [inbin[i] for i in range(len(inbin)) if bidx[i] == b]
        if sel:
            print(f"  {LEN_LAB[b]:>10}  {len(sel):>6}  "
                  f"{100*np.mean([x[1] for x in sel]):>8.0f}%")

    # traversed vs not: length + candidate count
    trav = [x for x in inbin if x[1]]; non = [x for x in inbin if not x[1]]
    print(f"\ntraversed segments:     median len {np.median([x[3] for x in trav]):.0f}m"
          f"   mean candidates/transition {np.mean([x[4] for x in trav]):.0f}")
    print(f"non-traversed segments: median len {np.median([x[3] for x in non]):.0f}m"
          f"   mean candidates/transition {np.mean([x[4] for x in non]):.0f}")

    # ---- context: class mix of overconfident bin vs the calibrated high bin ----
    hi = [x for x in rows if x[0] >= 0.8]
    def mix(rs):
        d = defaultdict(int)
        for x in rs:
            d[x[2]] += 1
        return {c: v / len(rs) for c, v in d.items()}
    mb, hb = mix(inbin), mix(hi)
    print(f"\nclass mix: mid bin [{BIN_LO},{BIN_HI}) vs calibrated high bin [0.8,1.0]:")
    print(f"  {'class':>13}  {'mid %':>6}  {'high %':>7}")
    for c in sorted(set(mb) | set(hb), key=lambda c: -mb.get(c, 0)):
        print(f"  {c:>13}  {100*mb.get(c,0):>5.0f}%  {100*hb.get(c,0):>6.0f}%")

    # ---- figure ----
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(14, 5.2))
    order = sorted(cls, key=lambda c: -len(cls[c]))[:7]
    rates = [100*np.mean([t for t, _ in cls[c]]) for c in order]
    ns = [len(cls[c]) for c in order]
    axL.bar(range(len(order)), rates, color="#0e7490", alpha=0.85, edgecolor="white")
    for i, (rr, nn) in enumerate(zip(rates, ns)):
        axL.text(i, rr + 1, f"n={nn}", ha="center", fontsize=7)
    axL.axhline(100*obs, ls="--", color="#dc2626", lw=1, label=f"bin avg {100*obs:.0f}%")
    axL.axhline(50, ls="--", color="#64748b", lw=1, label="stated ~50%")
    axL.set_xticks(range(len(order))); axL.set_xticklabels(order, rotation=35, fontsize=8)
    axL.set_ylabel("traversed (%)"); axL.set_ylim(0, 100); axL.legend(fontsize=8)
    axL.set_title(f"Mid bin [{BIN_LO},{BIN_HI}): traversal by road class")

    rates_l, ns_l = [], []
    for b in range(len(LEN_LAB)):
        sel = [inbin[i] for i in range(len(inbin)) if bidx[i] == b]
        rates_l.append(100*np.mean([x[1] for x in sel]) if sel else np.nan)
        ns_l.append(len(sel))
    axR.bar(range(len(LEN_LAB)), rates_l, color="#7c3aed", alpha=0.85, edgecolor="white")
    for i, (rr, nn) in enumerate(zip(rates_l, ns_l)):
        if not np.isnan(rr):
            axR.text(i, rr + 1, f"n={nn}", ha="center", fontsize=7)
    axR.axhline(100*obs, ls="--", color="#dc2626", lw=1, label=f"bin avg {100*obs:.0f}%")
    axR.axhline(50, ls="--", color="#64748b", lw=1, label="stated ~50%")
    axR.set_xticks(range(len(LEN_LAB))); axR.set_xticklabels(LEN_LAB, fontsize=8)
    axR.set_ylabel("traversed (%)"); axR.set_ylim(0, 100); axR.legend(fontsize=8)
    axR.set_title(f"Mid bin [{BIN_LO},{BIN_HI}): traversal by segment length")
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=130, bbox_inches="tight")
    print(f"\nwrote {OUT_PNG}")


if __name__ == "__main__":
    main()
