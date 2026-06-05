"""EXPLORATORY — directed marginals, undirected by max(fwd,rev) vs sum.

Current edge marginal collapses a two-way road's two directed links to one
undirected segment and SUMS the path mass through it. This asks: compute the
marginal on the *directed* links (no twin collapse), then fold each two-way
road to undirected by taking max(P_forward, P_reverse) instead of the sum.
Does the per-bin calibration change?

Run on both the untrained (μ=0) and trained eval, both graded by the neutral
15 s ruler (μ=0/scale15/df4). First run collects + caches
cache/_explore_directed_records.pkl; later runs re-bin instantly (--recollect
to redo). Saves cache/_explore_directed_max.png.
"""

from __future__ import annotations

import gzip
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.demo_cluster_calibration_draft import (   # noqa: E402
    CSV, LABELS, OSM_CACHE, PBF, TARGET,
    _cfg, _eval_config_mu0, _truth_config_indep,
    _truth_edges_in_window, _undirected_keyer,
)
from src.api import reconstruct_trajectory              # noqa: E402
from src.feeds import iter_porto_trips                   # noqa: E402
from src.network import load_osm_network                 # noqa: E402

REPO = Path(__file__).resolve().parents[1]
RECORDS = REPO / "cache" / "_explore_directed_records.pkl"
OUT_PNG = REPO / "cache" / "_explore_directed_max.png"

BINS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0001]
LAB = ["[0.0,0.2)", "[0.2,0.4)", "[0.4,0.6)", "[0.6,0.8)", "[0.8,1.0]"]


def collect(network):
    with gzip.open(LABELS, "rb") as f:
        training_ids = {tid.split("_s")[0] for tid, _ in pickle.load(f)["trips"]}
    key_of = _undirected_keyer(network)
    eval_cfgs = {"mu0": _eval_config_mu0(network), "trained": _cfg(network)}
    truth_cfg = _truth_config_indep(network)
    out = {"mu0": [], "trained": []}
    n = 0
    for tid, raw in iter_porto_trips(CSV, min_pings=40):
        if tid in training_ids or len(raw) > 200:
            continue
        try:
            segs_15 = reconstruct_trajectory(raw, network, truth_cfg)
        except Exception:
            continue
        ok = True
        staged = {"mu0": [], "trained": []}
        for label, cfg in eval_cfgs.items():
            try:
                segs_120 = reconstruct_trajectory(raw[::8], network, cfg)
            except Exception:
                ok = False
                break
            for seg in segs_120:
                for k, marg in enumerate(seg.path_marginals):
                    if not marg:
                        continue
                    total = sum(marg.values())
                    if total <= 0:
                        continue
                    paths = []
                    for p, w in marg.items():
                        pairs = tuple((int(e), key_of(int(e))) for e in p.edges
                                      if key_of(int(e)) is not None)
                        paths.append((pairs, float(w) / total))
                    t_lo = seg.canonical_timestamps[k]
                    t_hi = seg.canonical_timestamps[k + 1]
                    truth = _truth_edges_in_window(segs_15, t_lo, t_hi, key_of)
                    staged[label].append({"paths": paths, "truth_edges": truth})
        if not ok:
            continue
        out["mu0"].extend(staged["mu0"])
        out["trained"].extend(staged["trained"])
        n += 1
        print(f"  {n}/{TARGET} trips", end="\r", file=sys.stderr)
        if n >= TARGET:
            break
    print(f"\ncollected {n} trips; mu0 {len(out['mu0'])} / trained "
          f"{len(out['trained'])} transitions")
    with open(RECORDS, "wb") as f:
        pickle.dump(out, f)
    return out


def sum_max_rows(records):
    """Per (transition, undirected segment): (P_sum, P_max, on_truth).
    Also returns the count of segments where the two directed links both
    carried mass (max < sum)."""
    rows = []
    split = 0
    for r in records:
        Pdir, lkey = {}, {}
        for pairs, w in r["paths"]:
            for lid, key in pairs:
                Pdir[lid] = Pdir.get(lid, 0.0) + w
                lkey[lid] = key
        groups = {}
        for lid, key in lkey.items():
            groups.setdefault(key, []).append(Pdir[lid])
        truth = r["truth_edges"]
        for key, ps in groups.items():
            t = key in truth
            s, m = min(sum(ps), 1.0), min(max(ps), 1.0)
            if len(ps) > 1 and m < s - 1e-9:
                split += 1
            rows.append((s, m, t))
    return rows, split


def bins(vals, truth):
    vals = np.array(vals); truth = np.array(truth, float)
    idx = np.digitize(vals, BINS) - 1
    n, pred, obs = [], [], []
    for b in range(len(LAB)):
        sel = idx == b
        n.append(int(sel.sum()))
        pred.append(vals[sel].mean() if sel.any() else np.nan)
        obs.append(truth[sel].mean() if sel.any() else np.nan)
    return n, pred, obs


def gap(vals, truth):
    n, pred, obs = bins(vals, truth)
    N = sum(n)
    return 100 * sum(n[i] / N * abs(pred[i] - obs[i])
                     for i in range(len(LAB)) if n[i])


def report(data):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4))
    for ax, label in zip(axes, ("mu0", "trained")):
        rows, split = sum_max_rows(data[label])
        s = [r[0] for r in rows]; m = [r[1] for r in rows]; t = [r[2] for r in rows]
        ns, ps, os_ = bins(s, t)
        nm, pm, om = bins(m, t)
        title = "untrained (μ=0)" if label == "mu0" else "trained μ"
        print(f"\n=== {title} ===")
        print(f"  {len(rows)} (transition, segment) pairs; "
              f"{split} ({100*split/len(rows):.0f}%) have both directions carrying mass")
        print(f"  gap: sum {gap(s,t):.1f}%   max {gap(m,t):.1f}%")
        print(f"  {'bin':>11}  {'n':>6}  {'sum pred':>8} {'sum obs':>7}  | "
              f"{'max pred':>8} {'max obs':>7}")
        for i in range(len(LAB)):
            print(f"  {LAB[i]:>11}  {ns[i]:>6}  {100*ps[i]:>7.0f}% {100*os_[i]:>6.0f}%  | "
                  f"{100*pm[i]:>7.0f}% {100*om[i]:>6.0f}%")

        x = np.arange(len(LAB))
        ax.plot([100*p for p in ps], [100*o for o in os_], "o-",
                color="#7c3aed", label="sum (fwd+rev)")
        ax.plot([100*p for p in pm], [100*o for o in om], "s-",
                color="#dc2626", label="max(fwd, rev)")
        ax.plot([0, 100], [0, 100], "--", color="#64748b", lw=1, label="honest")
        ax.set_xlim(0, 100); ax.set_ylim(0, 105); ax.legend(fontsize=8)
        ax.set_xlabel("stated P(segment) (%)")
        ax.set_ylabel("share on true route (%)")
        ax.set_title(f"{title}: sum vs max aggregation")
    plt.tight_layout()
    OUT_PNG.parent.mkdir(exist_ok=True)
    plt.savefig(OUT_PNG, dpi=130, bbox_inches="tight")
    print(f"\nwrote {OUT_PNG}")


def main():
    if RECORDS.exists() and "--recollect" not in sys.argv:
        print(f"loading {RECORDS.name}")
        with open(RECORDS, "rb") as f:
            data = pickle.load(f)
    else:
        print("loading network…")
        network = load_osm_network(PBF, cache_path=OSM_CACHE)
        data = collect(network)
    report(data)


if __name__ == "__main__":
    main()
