"""EXPLORATORY — does widening admission change edge-marginal calibration?

Isolated A/B: the eval reconstruction is run with max_path_candidates=300
(default 100) and path_budget_slack=2.0 (default 1.5); everything else is
identical. The neutral 15 s truth ruler (μ=0/scale15/df4) keeps its baseline
admission, so only the eval's candidate breadth changes. Run for both the
untrained (μ=0) and trained eval; overlay the wide-config calibration on the
narrow baseline already collected.

Baselines reused:
  cache/_cluster_calib_records_indeptruth.pkl   (trained μ, narrow)
  cache/_cluster_calib_records_eval_mu0.pkl      (μ=0, narrow)

First run collects + caches cache/_explore_wide_records.pkl; reruns re-bin
instantly (--recollect to redo). Saves cache/_explore_wide.png.
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
    _edges_undirected, _truth_config_indep, _truth_edges_in_window,
    _undirected_keyer,
)
from src.api import reconstruct_trajectory              # noqa: E402
from src.config import Config                            # noqa: E402
from src.data import default_mu                          # noqa: E402
from src.feeds import iter_porto_trips                   # noqa: E402
from src.model import (                                  # noqa: E402
    ExponentialFamilyTransition, FEATURE_DIM, StudentTEmission,
)
from src.network import load_osm_network                 # noqa: E402

REPO = Path(__file__).resolve().parents[1]
RECORDS = REPO / "cache" / "_explore_wide_records.pkl"
OUT_PNG = REPO / "cache" / "_explore_wide.png"
BASE_TRAINED = REPO / "cache" / "_cluster_calib_records_indeptruth.pkl"
BASE_MU0 = REPO / "cache" / "_cluster_calib_records_eval_mu0.pkl"

N_CANDIDATES = 300
SLACK = 2.0
BINS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0001]
LAB = ["[0.0,0.2)", "[0.2,0.4)", "[0.4,0.6)", "[0.6,0.8)", "[0.8,1.0]"]


def eval_wide(network, mu):
    return Config(
        emission=StudentTEmission(scale=10.0, network=network),
        transition=ExponentialFamilyTransition(mu),
        enable_offroad_candidates=True,
        max_path_candidates=N_CANDIDATES,
        path_budget_slack=SLACK,
    )


def collect(network):
    with gzip.open(LABELS, "rb") as f:
        training_ids = {tid.split("_s")[0] for tid, _ in pickle.load(f)["trips"]}
    key_of = _undirected_keyer(network)
    eval_cfgs = {"mu0": eval_wide(network, np.zeros(FEATURE_DIM)),
                 "trained": eval_wide(network, default_mu())}
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
        staged = {"mu0": [], "trained": []}
        ok = True
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
                    paths = [(_edges_undirected(p.edges, key_of), float(w) / total)
                             for p, w in marg.items()]
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
          f"{len(out['trained'])} transitions (candidates={N_CANDIDATES}, slack={SLACK})")
    with open(RECORDS, "wb") as f:
        pickle.dump(out, f)
    return out


def edge_rows(recs):
    rows = []
    for r in recs:
        if not r["paths"] or not r["truth_edges"]:
            continue
        P = {}
        for seg_set, w in r["paths"]:
            for e in seg_set:
                P[e] = P.get(e, 0.0) + w
        for e, p in P.items():
            rows.append((min(p, 1.0), 1.0 if e in r["truth_edges"] else 0.0))
    return rows


def by_bin(rows):
    p = np.array([a for a, _ in rows]); y = np.array([b for _, b in rows])
    idx = np.digitize(p, BINS) - 1
    n, pred, obs = [], [], []
    for b in range(len(LAB)):
        s = idx == b
        n.append(int(s.sum()))
        pred.append(p[s].mean() if s.any() else np.nan)
        obs.append(y[s].mean() if s.any() else np.nan)
    return n, pred, obs


def gap(rows):
    n, pred, obs = by_bin(rows); N = sum(n)
    return 100 * sum(n[i] / N * abs(pred[i] - obs[i]) for i in range(len(LAB)) if n[i])


def _load(p):
    with open(p, "rb") as f:
        d = pickle.load(f)
    return d["records"] if isinstance(d, dict) and "records" in d else d


def report(wide):
    base = {"mu0": _load(BASE_MU0), "trained": _load(BASE_TRAINED)}
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4))
    for ax, label in zip(axes, ("mu0", "trained")):
        title = "untrained (μ=0)" if label == "mu0" else "trained μ"
        wr, br = edge_rows(wide[label]), edge_rows(base[label])
        nw, pw, ow = by_bin(wr); nb, pb, ob = by_bin(br)
        print(f"\n=== {title} ===  (narrow: 100/1.5   wide: {N_CANDIDATES}/{SLACK})")
        print(f"  segments: narrow {len(br)}  wide {len(wr)}   |   "
              f"gap: narrow {gap(br):.1f}%  wide {gap(wr):.1f}%")
        print(f"  {'bin':>11}  {'narrow n':>9} {'pred':>5} {'obs':>5}  | "
              f"{'wide n':>8} {'pred':>5} {'obs':>5}")
        for i in range(len(LAB)):
            print(f"  {LAB[i]:>11}  {nb[i]:>9} {100*pb[i]:>4.0f}% {100*ob[i]:>4.0f}%  | "
                  f"{nw[i]:>8} {100*pw[i]:>4.0f}% {100*ow[i]:>4.0f}%")
        ax.plot([100*p for p in pb], [100*o for o in ob], "o-",
                color="#94a3b8", label="narrow (100 / 1.5)")
        ax.plot([100*p for p in pw], [100*o for o in ow], "s-",
                color="#dc2626", label=f"wide ({N_CANDIDATES} / {SLACK})")
        ax.plot([0, 100], [0, 100], "--", color="#64748b", lw=1, label="honest")
        ax.set_xlim(0, 100); ax.set_ylim(0, 105); ax.legend(fontsize=8)
        ax.set_xlabel("stated P(segment) (%)")
        ax.set_ylabel("share on true route (%)")
        ax.set_title(f"{title}: candidate breadth")
    plt.tight_layout()
    OUT_PNG.parent.mkdir(exist_ok=True)
    plt.savefig(OUT_PNG, dpi=130, bbox_inches="tight")
    print(f"\nwrote {OUT_PNG}")


def main():
    if RECORDS.exists() and "--recollect" not in sys.argv:
        print(f"loading {RECORDS.name}")
        with open(RECORDS, "rb") as f:
            wide = pickle.load(f)
    else:
        print("loading network…")
        network = load_osm_network(PBF, cache_path=OSM_CACHE)
        wide = collect(network)
    report(wide)


if __name__ == "__main__":
    main()
