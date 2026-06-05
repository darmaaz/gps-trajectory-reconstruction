"""DRAFT — does path-posterior confidence calibrate at the EDGE level?

We cut the STORY §5 calibration chart because top-1 *path* weight is a
broken confidence axis: the posterior over-resolves the path partition,
splitting ~1.0 corridor mass across 5 near-identical paths (0.25 / 0.20 /
0.20 / 0.20 / 0.15). Raw top-1 weight then reads "weakly confident" when
the corridor itself is near-certain.

Two candidate fixes, both here:

  1. CLUSTER marginalization — merge near-duplicate paths by edge-set
     Jaccard, sum weight per cluster, calibrate on cluster top-weight.
     (Verdict from the first pass: doesn't calibrate, and the weight-sum
     hurts on rep-flips. Kept for the comparison.)

  2. EDGE marginalization — for a transition, P(E in path) = sum_p r(p)
     1[E in p]. Corridor degeneracy contributes the *same total mass* to a
     corridor's edges, so a corridor that's "really 0.6" reads 0.6 at the
     edge level even when split 0.15x4 at the path level. The summing IS
     the marginalization, performed where structural alternatives become
     well-defined. Test: bin edges by P(E); per bin, the fraction of edges
     traversed by the 15 s truth. If this lands on the diagonal, the
     calibration story exists — at the edge level, which is what most
     downstream consumers want anyway ("did the vehicle go through X?").

THE FOOTGUN (load-bearing — see memory coverage-validation-draft): any
path edge-overlap metric MUST key on **undirected road identity**
`tuple(sorted((from_node[idx], to_node[idx])))`, not raw `link_id`. A
two-way road's reverse twin is a synthetic link_id with no arithmetic
relation to the forward id; directed keying counts same-road-opposite-
direction as a miss (~16-20pp of pure artifact on recall). All keying
below is undirected.

Run:  python scripts/demo_cluster_calibration_draft.py
First run collects + pickles to cache/_cluster_calib_records_v2.pkl (the
40-trip x (120 s + 15 s) reconstruction loop is the slow part); later runs
re-bin from the pickle instantly. Pass --recollect to redo.
Saves cache/demo_cluster_calibration_draft.png (cluster) and
      cache/demo_cluster_calibration_edge.png (edge marginal).
"""

from __future__ import annotations

import gzip
import os
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("GPS_RECON_BBOX_LAT", "40.5,42.5")
os.environ.setdefault("GPS_RECON_BBOX_LON", "-9.5,-7.0")

from scripts._data_paths import osm_pbf_path, porto_csv_path    # noqa: E402

from src.api import reconstruct_trajectory                       # noqa: E402
from src.config import Config                                    # noqa: E402
from src.data import default_mu                                  # noqa: E402
from src.feeds import iter_porto_trips                           # noqa: E402
from src.model import (                                          # noqa: E402
    ExponentialFamilyTransition, FEATURE_DIM, StudentTEmission,
)
from src.network import load_osm_network                         # noqa: E402

REPO = Path(__file__).resolve().parents[1]
CSV = porto_csv_path()
PBF = osm_pbf_path()
OSM_CACHE = REPO / "cache" / "pt_edges.parquet"
LABELS = REPO / "cache" / "labeled_trips_15s.pkl.gz"
RECORDS = REPO / "cache" / "_cluster_calib_records_v2.pkl"   # v2 = undirected keys
RECORDS_INDEP = REPO / "cache" / "_cluster_calib_records_indeptruth.pkl"
RECORDS_EVAL_MU0 = REPO / "cache" / "_cluster_calib_records_eval_mu0.pkl"
OUT_PNG = REPO / "cache" / "demo_cluster_calibration_draft.png"
OUT_PNG_EDGE = REPO / "cache" / "demo_cluster_calibration_edge.png"
KEYING = "undirected_nodepair"

TARGET = 40
MATCH_JACCARD = 0.5
TAUS = (0.2, 0.25, 0.3)
TAU_MAIN = 0.25
BINS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0001]
BIN_LABELS = ["[0.0,0.2)", "[0.2,0.4)", "[0.4,0.6)", "[0.6,0.8)", "[0.8,1.0]"]


# --------------------------------------------------------------------------
# collection
# --------------------------------------------------------------------------
def _cfg(network):
    """The 120 s model UNDER EVALUATION: trained μ, scale=10, off-road on."""
    return Config(
        emission=StudentTEmission(scale=10.0, network=network),
        transition=ExponentialFamilyTransition(default_mu()),
        enable_offroad_candidates=True,
    )


def _eval_config_mu0(network):
    """Ablation eval: SAME as _cfg (scale=10, off-road on) but μ=0 — the
    flat-prior model, to test whether the trained μ shapes the calibration."""
    return Config(
        emission=StudentTEmission(scale=10.0, network=network),
        transition=ExponentialFamilyTransition(np.zeros(FEATURE_DIM)),
        enable_offroad_candidates=True,
    )


def _truth_config_indep(network):
    """Independent ground-truth reference, mirroring the coverage draft /
    compute_15s_labels.py: μ=0, scale=15, df=4, off-road off. Decouples the
    truth route from the trained μ being evaluated, so it can't flatter the
    model at the ambiguous forks (where a shared μ tips eval and truth the
    same way)."""
    return Config(
        emission=StudentTEmission(scale=15.0, network=network, df=4.0),
        transition=ExponentialFamilyTransition(np.zeros(FEATURE_DIM)),
    )


def _undirected_keyer(network):
    """link_id -> undirected road identity tuple(sorted((from_node, to_node))).

    Collapses a two-way road's synthetic reverse-twin link_id onto the same
    key as its forward edge. Memoised. Unmappable link_ids (none expected)
    are skipped by returning None.
    """
    fn, tn = network.from_node, network.to_node
    cache: dict[int, tuple[int, int] | None] = {}

    def key_of(link_id):
        lid = int(link_id)
        k = cache.get(lid, 0)
        if k != 0:
            return k
        try:
            idx = network.edge_index_for_link(lid)
            a, b = int(fn[idx]), int(tn[idx])
            k = (a, b) if a <= b else (b, a)
        except KeyError:
            k = None
        cache[lid] = k
        return k

    return key_of


def _edges_undirected(edge_ids, key_of):
    return frozenset(k for k in (key_of(e) for e in edge_ids) if k is not None)


def _truth_edges_in_window(segs_15, t_lo, t_hi, key_of):
    """Undirected road keys on the 15 s Viterbi MLE overlapping (t_lo, t_hi]."""
    edges = set()
    for s in segs_15:
        ts = s.canonical_timestamps
        ml = s.most_likely
        for j in range(len(ts) - 1):
            if ts[j] >= t_hi or ts[j + 1] <= t_lo:
                continue
            idx = 2 * j + 1
            if idx >= len(ml):
                continue
            step = ml[idx]
            if hasattr(step, "edges"):
                edges |= _edges_undirected(step.edges, key_of)
    return frozenset(edges)


def collect(network, truth_cfg=None, out_path=RECORDS, truth_label="same-model",
            eval_cfg=None, eval_label="trained-mu"):
    """Eval (120 s) uses `eval_cfg` (None ⇒ _cfg, trained μ scale 10). `truth_cfg`
    is the 15 s reference config; None ⇒ same as eval (same-model truth)."""
    with gzip.open(LABELS, "rb") as f:
        training_ids = {tid.split("_s")[0] for tid, _ in pickle.load(f)["trips"]}
    print(f"training set holds {len(training_ids)} source trips; collecting held-out "
          f"(eval={eval_label}, truth={truth_label})…")

    key_of = _undirected_keyer(network)
    records = []   # one dict per 120 s transition
    n_collected = 0
    eval_cfg = eval_cfg or _cfg(network)
    truth_cfg = truth_cfg or eval_cfg
    for tid, raw in iter_porto_trips(CSV, min_pings=40):
        if tid in training_ids or len(raw) > 200:
            continue
        try:
            segs_120 = reconstruct_trajectory(raw[::8], network, eval_cfg)
            segs_15 = reconstruct_trajectory(raw, network, truth_cfg)
        except Exception:
            continue
        for seg in segs_120:
            for k, marg in enumerate(seg.path_marginals):
                if not marg:
                    continue
                paths = [(_edges_undirected(p.edges, key_of), float(w))
                         for p, w in marg.items()]
                total = sum(w for _, w in paths)
                if total <= 0:
                    continue
                paths = [(e, w / total) for e, w in paths]   # normalise
                t_lo = seg.canonical_timestamps[k]
                t_hi = seg.canonical_timestamps[k + 1]
                records.append({
                    "trip": tid, "k": int(k),
                    "paths": paths,
                    "truth_edges": _truth_edges_in_window(segs_15, t_lo, t_hi, key_of),
                })
        n_collected += 1
        if n_collected >= TARGET:
            break
    print(f"collected {n_collected} trips, {len(records)} transitions "
          f"(keying={KEYING}, eval={eval_label}, truth={truth_label})")
    with open(out_path, "wb") as f:
        pickle.dump({"keying": KEYING, "eval": eval_label, "truth": truth_label,
                     "records": records}, f)
    return records


def load_records(path=RECORDS):
    with open(path, "rb") as f:
        data = pickle.load(f)
    if data.get("keying") != KEYING:
        raise ValueError(f"{path} keying={data.get('keying')} != {KEYING}; "
                         "re-run with --recollect")
    return data["records"]


# --------------------------------------------------------------------------
# clustering + scoring  (operates on undirected edge-key sets)
# --------------------------------------------------------------------------
def _jaccard(a, b):
    u = a | b
    return len(a & b) / len(u) if u else 0.0


def _jaccard_dist(a, b):
    return 1.0 - _jaccard(a, b)


def cluster_indices(paths, tau):
    """Single-linkage union-find: merge path pairs with Jaccard dist <= tau."""
    n = len(paths)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        for j in range(i + 1, n):
            if _jaccard_dist(paths[i][0], paths[j][0]) <= tau:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[ri] = rj
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def score(records, tau, match_jac=MATCH_JACCARD):
    out = []
    for rec in records:
        paths, truth = rec["paths"], rec["truth_edges"]
        if not paths:
            continue
        raw_idx = max(range(len(paths)), key=lambda i: paths[i][1])
        groups = cluster_indices(paths, tau)
        cl_weights = [sum(paths[i][1] for i in g) for g in groups]
        top_c = max(range(len(groups)), key=lambda c: cl_weights[c])
        rep_idx = max(groups[top_c], key=lambda i: paths[i][1])
        scorable = len(truth) > 0
        pw = np.array([w for _, w in paths]); pw = pw[pw > 0]
        cw = np.array(cl_weights); cw = cw[cw > 0]
        out.append({
            "raw_w": paths[raw_idx][1], "cl_w": cl_weights[top_c],
            "raw_m": _jaccard(paths[raw_idx][0], truth) >= match_jac if scorable else None,
            "cl_m": _jaccard(paths[rep_idx][0], truth) >= match_jac if scorable else None,
            "scorable": scorable,
            "n_paths": len(paths), "n_clusters": len(groups),
            "rep_differs": rep_idx != raw_idx,
            "H_raw": float(-np.sum(pw * np.log(pw))),
            "H_cl": float(-np.sum(cw * np.log(cw))),
        })
    return out


def bin_rates(scored, key_w, key_m):
    rows = [s for s in scored if s["scorable"]]
    idx = np.digitize([s[key_w] for s in rows], BINS) - 1
    rates, means, counts = [], [], []
    for b in range(len(BIN_LABELS)):
        sel = [rows[i] for i in range(len(rows)) if idx[i] == b]
        counts.append(len(sel))
        rates.append(np.mean([s[key_m] for s in sel]) if sel else float("nan"))
        means.append(np.mean([s[key_w] for s in sel]) if sel else float("nan"))
    return rates, means, counts


# --------------------------------------------------------------------------
# EDGE-MARGINAL calibration
# --------------------------------------------------------------------------
def edge_marginal_rows(records):
    """Per (transition, candidate-edge): (P(E in path), traversed_by_truth).

    Scored over the candidate-set support (P>0). Truth edges with P=0 are a
    recall gap (separate axis) — counted and returned, not binned here.
    """
    rows = []                 # (P_edge, traversed_bool)
    truth_total = truth_missed = 0
    for rec in records:
        paths, truth = rec["paths"], rec["truth_edges"]
        if not paths or not truth:
            continue
        P = {}
        for edges, w in paths:
            for e in edges:
                P[e] = P.get(e, 0.0) + w
        for e, p in P.items():
            rows.append((min(p, 1.0), e in truth))
        truth_total += len(truth)
        truth_missed += sum(1 for e in truth if e not in P)
    return rows, truth_total, truth_missed


def edge_bins(rows):
    """Per bin: n edges, empirical traversed-fraction, mean predicted P."""
    ps = np.array([p for p, _ in rows])
    tv = np.array([1.0 if t else 0.0 for _, t in rows])
    idx = np.digitize(ps, BINS) - 1
    frac, pred, counts = [], [], []
    for b in range(len(BIN_LABELS)):
        sel = idx == b
        counts.append(int(sel.sum()))
        frac.append(float(tv[sel].mean()) if sel.any() else float("nan"))
        pred.append(float(ps[sel].mean()) if sel.any() else float("nan"))
    return frac, pred, counts


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------
def report(records):
    print(f"\n(keying = {KEYING}; reverse-twin link_ids collapsed)")
    print("\n" + "=" * 70)
    print("CLUSTER-COUNT COLLAPSE  (n_paths -> n_clusters, mean over transitions)")
    print("=" * 70)
    multi = [r for r in records if len(r["paths"]) >= 2]
    print(f"  transitions with >=2 candidate paths: {len(multi)} / {len(records)}")
    print(f"  {'tau':>5}  {'mean n_paths':>12}  {'mean n_clusters':>15}  "
          f"{'% collapsed to 1':>16}")
    for tau in TAUS:
        ncl = [len(cluster_indices(r["paths"], tau)) for r in multi]
        npaths = np.mean([len(r["paths"]) for r in multi])
        print(f"  {tau:>5.2f}  {npaths:>12.2f}  {np.mean(ncl):>15.2f}  "
              f"{100*np.mean([c == 1 for c in ncl]):>15.0f}%")

    scored = score(records, TAU_MAIN)
    scorable = [s for s in scored if s["scorable"]]
    print(f"\n  (scoring at tau={TAU_MAIN}; {len(scorable)}/{len(scored)} "
          f"transitions have 15 s truth in-window)")

    raw_rates, raw_means, raw_n = bin_rates(scored, "raw_w", "raw_m")
    cl_rates, cl_means, cl_n = bin_rates(scored, "cl_w", "cl_m")

    def _pct(x):
        return f"{100*x:.0f}%" if not np.isnan(x) else "  -"

    print("\n" + "=" * 70)
    print("PATH/CLUSTER CALIBRATION — match rate per top-weight bin")
    print("=" * 70)
    print(f"  {'bin':>11}  {'RAW n':>6} {'RAW':>6}  |  {'CL n':>6} {'CL':>6}")
    for i, lab in enumerate(BIN_LABELS):
        print(f"  {lab:>11}  {raw_n[i]:>6d} {_pct(raw_rates[i]):>6}  |  "
              f"{cl_n[i]:>6d} {_pct(cl_rates[i]):>6}")
    print("\n  RELIABILITY (pred vs emp; equal = calibrated)")
    print(f"  {'bin':>11}  {'RAW pred':>9} {'RAW emp':>9}  |  {'CL pred':>9} {'CL emp':>9}")
    for i, lab in enumerate(BIN_LABELS):
        print(f"  {lab:>11}  {_pct(raw_means[i]):>9} {_pct(raw_rates[i]):>9}  |  "
              f"{_pct(cl_means[i]):>9} {_pct(cl_rates[i]):>9}")

    diff = [s for s in scorable if s["rep_differs"]]
    print("\n  HELP-OR-HURT (top-cluster rep != global MLE):")
    if diff:
        rm, cm = np.mean([s["raw_m"] for s in diff]), np.mean([s["cl_m"] for s in diff])
        print(f"    {len(diff)}/{len(scorable)} flip; raw-MLE {100*rm:.0f}% -> "
              f"cluster-rep {100*cm:.0f}%  "
              f"({'HELPS' if cm > rm else 'HURTS' if cm < rm else 'neutral'})")
    else:
        print("    none")

    # ---- EDGE MARGINAL ----
    rows, truth_total, truth_missed = edge_marginal_rows(records)
    frac, pred, counts = edge_bins(rows)
    print("\n" + "=" * 70)
    print("EDGE-MARGINAL CALIBRATION — bin edges by P(E in path)")
    print("=" * 70)
    print(f"  {len(rows)} candidate edges scored across "
          f"{sum(1 for r in records if r['truth_edges'])} transitions")
    print(f"  truth-edge coverage: {truth_total - truth_missed}/{truth_total} "
          f"truth edges have P>0 "
          f"({100*(truth_total-truth_missed)/max(truth_total,1):.0f}%); "
          f"the rest are a recall gap (separate axis)")
    print(f"\n  {'P(E) bin':>11}  {'n edges':>8}  {'pred P':>7}  "
          f"{'emp traversed':>13}")
    for i, lab in enumerate(BIN_LABELS):
        p = f"{100*pred[i]:.0f}%" if not np.isnan(pred[i]) else "  -"
        fr = f"{100*frac[i]:.0f}%" if not np.isnan(frac[i]) else "  -"
        print(f"  {lab:>11}  {counts[i]:>8d}  {p:>7}  {fr:>13}")

    # entropy
    print("\n" + "=" * 70)
    print("ENTROPY  — raw path distribution vs cluster distribution (nats)")
    print("=" * 70)
    print(f"  mean entropy:  raw {np.mean([s['H_raw'] for s in scored]):.3f}  ->  "
          f"cluster {np.mean([s['H_cl'] for s in scored]):.3f}")
    print(f"  fraction opinionated (top weight >= 0.6):  "
          f"raw {100*np.mean([s['raw_w'] >= 0.6 for s in scored]):.0f}%  ->  "
          f"cluster {100*np.mean([s['cl_w'] >= 0.6 for s in scored]):.0f}%")

    _plot_cluster(scored, raw_rates, raw_means, raw_n, cl_rates, cl_means, cl_n)
    _plot_edge(rows, frac, pred, counts)


def _plot_cluster(scored, raw_rates, raw_means, raw_n, cl_rates, cl_means, cl_n):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))
    x = np.arange(len(BIN_LABELS)); w = 0.4
    ax1.bar(x - w/2, [100*r for r in raw_rates], w, label="raw top-path",
            color="#94a3b8", edgecolor="white")
    ax1.bar(x + w/2, [100*r for r in cl_rates], w, label="cluster top-weight",
            color="#16a34a", edgecolor="white")
    for i in range(len(BIN_LABELS)):
        if not np.isnan(raw_rates[i]):
            ax1.text(x[i]-w/2, 100*raw_rates[i]+1, f"{raw_n[i]}", ha="center",
                     fontsize=7, color="#475569")
        if not np.isnan(cl_rates[i]):
            ax1.text(x[i]+w/2, 100*cl_rates[i]+1, f"{cl_n[i]}", ha="center",
                     fontsize=7, color="#166534")
    ax1.set_xticks(x); ax1.set_xticklabels(BIN_LABELS, rotation=30, fontsize=8)
    ax1.set_ylabel("path-match rate (%)"); ax1.set_ylim(0, 105)
    ax1.set_xlabel("top-weight bin"); ax1.legend(fontsize=8)
    ax1.set_title(f"Path/cluster calibration (tau={TAU_MAIN})\n(undirected; n above bars)")
    for means, rates, color, lab in (
        (raw_means, raw_rates, "#94a3b8", "raw"),
        (cl_means, cl_rates, "#16a34a", "cluster"),
    ):
        ax2.plot([100*m for m in means], [100*r for r in rates], "o-",
                 color=color, label=lab)
    ax2.plot([0, 100], [0, 100], "--", color="#64748b", lw=1, label="calibrated")
    ax2.set_xlabel("mean predicted top-weight (%)")
    ax2.set_ylabel("empirical match rate (%)")
    ax2.set_xlim(0, 100); ax2.set_ylim(0, 105); ax2.legend(fontsize=8)
    ax2.set_title("Reliability (path/cluster)")
    sc = [s for s in scored if s["scorable"]]
    ax3.scatter([100*s["raw_w"] for s in sc], [100*s["cl_w"] for s in sc],
                c=["#16a34a" if s["cl_m"] else "#dc2626" for s in sc],
                s=18, alpha=0.6, edgecolor="white", linewidth=0.3)
    ax3.plot([0, 100], [0, 100], "--", color="#64748b", lw=1)
    ax3.set_xlabel("raw top-path weight (%)")
    ax3.set_ylabel("cluster top-weight (%)")
    ax3.set_xlim(0, 100); ax3.set_ylim(0, 105)
    ax3.set_title("Weight axis shift\n(green=rep matches truth, red=not)")
    plt.tight_layout()
    OUT_PNG.parent.mkdir(exist_ok=True)
    plt.savefig(OUT_PNG, dpi=130, bbox_inches="tight")
    print(f"\nwrote {OUT_PNG}")


def _plot_edge(rows, frac, pred, counts):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(len(BIN_LABELS))
    bars = ax1.bar(x, [100*f for f in frac], color="#7c3aed", alpha=0.85,
                   edgecolor="white")
    for i, b in enumerate(bars):
        if not np.isnan(frac[i]):
            ax1.text(b.get_x()+b.get_width()/2, 100*frac[i]+1, f"{counts[i]}",
                     ha="center", fontsize=7, color="#5b21b6")
    # diagonal target: bin midpoints
    mids = [100*(BINS[i] + min(BINS[i+1], 1.0))/2 for i in range(len(BIN_LABELS))]
    ax1.plot(x, mids, "--", color="#64748b", lw=1, label="calibrated (bin mid)")
    ax1.set_xticks(x); ax1.set_xticklabels(BIN_LABELS, rotation=30, fontsize=8)
    ax1.set_ylabel("fraction of edges traversed by truth (%)")
    ax1.set_xlabel("P(E in path) bin"); ax1.set_ylim(0, 105); ax1.legend(fontsize=8)
    ax1.set_title("Edge-marginal calibration\n(undirected; n above bars)")

    ax2.plot([100*p for p in pred], [100*f for f in frac], "o-",
             color="#7c3aed", label="edge marginal")
    ax2.plot([0, 100], [0, 100], "--", color="#64748b", lw=1, label="calibrated")
    ax2.set_xlabel("mean predicted P(E) (%)")
    ax2.set_ylabel("empirical traversed fraction (%)")
    ax2.set_xlim(0, 100); ax2.set_ylim(0, 105); ax2.legend(fontsize=8)
    ax2.set_title("Reliability (edge marginal)")
    plt.tight_layout()
    plt.savefig(OUT_PNG_EDGE, dpi=130, bbox_inches="tight")
    print(f"wrote {OUT_PNG_EDGE}")


def main():
    recollect = "--recollect" in sys.argv
    indep = "--indep-truth" in sys.argv
    eval_mu0 = "--eval-mu0" in sys.argv
    if eval_mu0:
        # μ=0 eval ablation, graded by the neutral independent ruler (the
        # truth is μ-insensitive, so this is the clean "μ=0 both" comparison).
        path, eval_msg, truth_msg = RECORDS_EVAL_MU0, "MU=0 (scale10)", "INDEPENDENT (mu=0, scale15, df4)"
    elif indep:
        path, eval_msg, truth_msg = RECORDS_INDEP, "trained mu (scale10)", "INDEPENDENT (mu=0, scale15, df4)"
    else:
        path, eval_msg, truth_msg = RECORDS, "trained mu (scale10)", "SAME-MODEL (trained mu, scale10)"
    if path.exists() and not recollect:
        print(f"loading cached records from {path.name} (pass --recollect to redo)")
        records = load_records(path)
    else:
        print("loading network…")
        network = load_osm_network(PBF, cache_path=OSM_CACHE)
        if eval_mu0:
            records = collect(network, _truth_config_indep(network), RECORDS_EVAL_MU0,
                              "independent_mu0_scale15_df4",
                              eval_cfg=_eval_config_mu0(network), eval_label="mu0_scale10")
        elif indep:
            records = collect(network, _truth_config_indep(network), RECORDS_INDEP,
                              "independent_mu0_scale15_df4")
        else:
            records = collect(network, None, RECORDS, "same-model")
    print(f"\n### eval: {eval_msg}   truth: {truth_msg} ###")
    report(records)


if __name__ == "__main__":
    main()
