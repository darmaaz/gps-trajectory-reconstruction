"""Compute top-K Viterbi diagnostics on the canonical Porto trips.

Heavy-lifting half (CLI-iterable): runs the pipeline at 15/60/120 s on the
three canonical trips, extracts cliff-free spans, runs top-K (parallel list)
Viterbi, gates rank-1 against the production MLE score, and computes both
diagnostics. Dumps `cache/_topk_records.pkl` + figures; the notebook builder
(`build_topk_explore_notebook.py`) renders them inline.

Run (Portfolio venv):
    .../python scripts/_topk_explore_run.py            # all trips, figures
    .../python scripts/_topk_explore_run.py --gate-only # extraction gate only
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt    # noqa: E402
import numpy as np
from matplotlib.collections import LineCollection    # noqa: E402
from shapely.geometry import box    # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("GPS_RECON_BBOX_LAT", "40.5,42.5")
os.environ.setdefault("GPS_RECON_BBOX_LON", "-9.5,-7.0")

from scripts._data_paths import osm_pbf_path, porto_csv_path    # noqa: E402

from scripts._topk_viterbi_explore import (    # noqa: E402
    diversity_filter, extract_spans, interleaved_edges, jaccard_dist,
    recompute_score, state_hamming_frac, state_links_undirected,
    topk_viterbi_span,
)
from src.config import Config    # noqa: E402
from src.data import default_mu    # noqa: E402
from src.feeds import iter_porto_trips    # noqa: E402
from src.model import (    # noqa: E402
    ExponentialFamilyTransition, Path as MPath, RawObservation, StudentTEmission,
)
from src.network import load_osm_network    # noqa: E402

REPO = Path(__file__).resolve().parents[1]
CACHE = REPO / "cache"
DEFAULT_PBF = osm_pbf_path()
DEFAULT_CSV = porto_csv_path()
OSM_CACHE = CACHE / "pt_edges.parquet"

TRIPS = {
    "SHORT": "1372637091620000337",
    "MEDIUM": "1372636951620000320",
    "LONG": "1372639536620000570",
}
SAMPLINGS = [15, 60, 120]
NATIVE_S = 15
K_RAW = 15        # raw pool, then we report top-5 / diversity-filter to 5
K_REPORT = 5
TAU = 0.25


def _log(m):
    print(f"[topk] {m}", file=sys.stderr, flush=True)


def _pick_trip(csv, trip_id):
    for tid, obs in iter_porto_trips(csv, min_pings=2):
        if tid == trip_id:
            return obs
    raise SystemExit(f"trip {trip_id} not found")


def _downsample(pings, stride):
    if stride <= 1:
        return list(pings)
    kept = pings[::stride]
    if pings and pings[-1] is not kept[-1]:
        kept.append(pings[-1])
    return kept


def _build_config(network):
    return Config(
        emission=StudentTEmission(scale=10.0, network=network),
        transition=ExponentialFamilyTransition(default_mu()),
    )


def _state_marg_mass(span, obs_k, state_idx):
    st = span.state_cands[obs_k][state_idx]
    return float(span.state_marginals[obs_k].get(st, 0.0))


def _edge_marginal_for_edge(span, trans_k, edge_link_id, network, cache):
    """P(undirected-edge ∈ path) at transition trans_k from FB path marginals."""
    from scripts._topk_viterbi_explore import undirected_edge
    target = undirected_edge(network, edge_link_id, cache)
    tot = 0.0
    for path, w in span.path_marginals[trans_k].items():
        if any(undirected_edge(network, e, cache) == target for e in path.edges):
            tot += w
    return float(tot)


def _cell_of(path, state_cands_k, state_cands_kp1):
    """(i, j) the path joins, or None if it doesn't match the candidate sets."""
    i = next((ii for ii, s in enumerate(state_cands_k) if path.starts_at(s)), None)
    j = next((jj for jj, s in enumerate(state_cands_kp1) if path.ends_at(s)), None)
    return None if (i is None or j is None) else (i, j)


def composition_check(span, network, ecache):
    """Does the per-transition marginal COMPOSE into a coherent global story?

    The falsifying direction (marginal → coherence). Build the marginal-greedy
    trajectory — argmax `state_marginals[k]` independently per obs — and ask
    whether consecutive picks are actually connected by an enumerated path
    (`(i*, j*)` registered in `best_path[k]`). Breaks ⇒ marginals do not
    compose; top-K is more than presentation.

    Also spot-checks the max-path-per-cell blind spot: transitions where a
    single (i, j) cell carries real marginal mass on ≥2 paths with materially
    different undirected edge sets — structural alternative-ness top-K cannot see.
    """
    L = len(span.state_cands)
    m_star = []
    for k in range(L):
        sm = span.state_marginals[k]
        masses = [sm.get(s, 0.0) for s in span.state_cands[k]]
        m_star.append(int(np.argmax(masses)) if masses else 0)

    breaks = []
    for k in range(L - 1):
        if (m_star[k], m_star[k + 1]) not in span.best_path[k]:
            breaks.append(k)
    composed_feasible = (len(breaks) == 0)

    # equals MLE / some top-K member?
    mle_idx = [
        span.state_cands[k].index(x)
        for k, x in enumerate(s for s in span.mle_interleaved if not isinstance(s, MPath))
    ]
    equals_mle = (m_star == mle_idx)

    # within-cell structural split: same (i,j), different undirected edge sets, both with mass
    wc_splits = 0
    for k in range(L - 1):
        from collections import defaultdict
        cells = defaultdict(list)
        for p, w in span.path_marginals[k].items():
            c = _cell_of(p, span.state_cands[k], span.state_cands[k + 1])
            if c is not None:
                cells[c].append((p, w))
        for c, plist in cells.items():
            big = [(p, w) for p, w in plist if w > 0.10]
            if len(big) < 2:
                continue
            esets = [frozenset(undirected_edge(network, e, ecache) for e in p.edges) for p, _ in big]
            # any two materially different (edge-Jaccard >= 0.5) ?
            if any(jaccard_dist(esets[a], esets[b]) >= 0.5
                   for a in range(len(esets)) for b in range(a + 1, len(esets))):
                wc_splits += 1
                break  # count the transition once

    return dict(
        composed_feasible=composed_feasible, n_breaks=len(breaks),
        equals_mle=equals_mle, within_cell_split_transitions=wc_splits,
        m_star=m_star,
    )


# late import so the helper above can use it
from scripts._topk_viterbi_explore import undirected_edge    # noqa: E402


def analyse_span(span, network, ecache):
    """Top-K on a span → per-rank distinctness, score gaps, diag-2 forks."""
    ranked = topk_viterbi_span(
        span.log_emit, span.log_trans, span.best_path, span.state_cands, K_RAW,
    )
    # ---- extraction gate: rank-1 score == production MLE score (recomputed) ----
    mle_idx = [
        span.state_cands[k].index(x)
        for k, x in enumerate(s for s in span.mle_interleaved if not isinstance(s, MPath))
    ]
    mle_score = recompute_score(mle_idx, span.log_emit, span.log_trans)
    gate_ok = abs(ranked[0].score - mle_score) < 1e-6

    rank1 = ranked[0]
    e1 = interleaved_edges(rank1.interleaved, network, ecache)
    l1 = state_links_undirected(rank1.interleaved, network, ecache)

    # ---- diag 1: distinctness + score gap of ranks 2..K vs rank-1 ----
    per_rank = []
    for rt in ranked[1:K_REPORT]:
        es = interleaved_edges(rt.interleaved, network, ecache)
        ls = state_links_undirected(rt.interleaved, network, ecache)
        per_rank.append(dict(
            jaccard=jaccard_dist(e1, es),
            hamming=state_hamming_frac(l1, ls),
            score_gap=rank1.score - rt.score,        # nats; >=0
        ))
    surv = diversity_filter(ranked, network, ecache, k_keep=K_REPORT, tau=TAU)

    # ---- diag 2: where rank-2 disagrees with rank-1 ----
    forks = []
    if len(ranked) >= 2:
        idx1, idx2 = ranked[0].state_indices, ranked[1].state_indices
        # state-level forks (headline): obs positions with different state
        for k in range(len(idx1)):
            if idx1[k] != idx2[k]:
                forks.append(dict(
                    kind="state", obs_k=k,
                    mass1=_state_marg_mass(span, k, idx1[k]),
                    mass2=_state_marg_mass(span, k, idx2[k]),
                    same_road=(state_links_undirected(ranked[0].interleaved, network, ecache)[k]
                               == state_links_undirected(ranked[1].interleaved, network, ecache)[k]),
                ))
        # transition-level path forks (illustration of path over-resolution)
        for t in range(len(idx1) - 1):
            cell1 = (idx1[t], idx1[t + 1])
            cell2 = (idx2[t], idx2[t + 1])
            if cell1 == cell2:
                continue
            p1 = span.best_path[t].get(cell1)
            p2 = span.best_path[t].get(cell2)
            if p1 is None or p2 is None or p1 is p2:
                continue
            w1 = float(span.path_marginals[t].get(p1, 0.0))
            w2 = float(span.path_marginals[t].get(p2, 0.0))
            # edge-marginal corroboration on a distinguishing edge of each
            d1 = [e for e in p1.edges if e not in set(p2.edges)]
            d2 = [e for e in p2.edges if e not in set(p1.edges)]
            em1 = _edge_marginal_for_edge(span, t, d1[0], network, ecache) if d1 else None
            em2 = _edge_marginal_for_edge(span, t, d2[0], network, ecache) if d2 else None
            forks.append(dict(
                kind="path", trans_t=t, path_w1=w1, path_w2=w2,
                edge_marg1=em1, edge_marg2=em2,
            ))

    comp = composition_check(span, network, ecache)
    m_star = comp.pop("m_star")
    comp["marginal_greedy_topk_rank"] = next(
        (r for r, rt in enumerate(ranked) if rt.state_indices == m_star), -1,
    )

    return dict(
        trip=span.trip_id, sampling=span.sampling_s, seg=span.seg_idx,
        sub=span.sub_idx, n_trans=span.n_trans, L=len(span.state_cands),
        n_ranked=len(ranked), gate_ok=bool(gate_ok),
        rank1_score=rank1.score,
        scores=[rt.score for rt in ranked[:K_REPORT]],
        per_rank=per_rank, n_survivors=len(surv),
        forks=forks, **comp,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pbf", type=Path, default=DEFAULT_PBF)
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    ap.add_argument("--gate-only", action="store_true")
    ap.add_argument("--trips", nargs="*", default=list(TRIPS))
    args = ap.parse_args()

    _log(f"loading network {args.pbf.name}")
    network = load_osm_network(args.pbf, cache_path=OSM_CACHE)
    _log(f"  {len(network)} edges")
    config = _build_config(network)
    ecache: dict = {}

    records = []
    all_spans = []          # aligned with `records`
    for name in args.trips:
        trip_id = TRIPS[name]
        raw15 = _pick_trip(args.csv, trip_id)
        _log(f"{name} {trip_id}: {len(raw15)} raw 15s pings")
        for s in SAMPLINGS:
            stride = s // NATIVE_S
            raw = _downsample(raw15, stride)
            spans = extract_spans(trip_id, s, raw, network, config)
            _log(f"  {s:>3}s (stride {stride}, {len(raw)} pings): {len(spans)} span(s)")
            for sp in spans:
                if sp.n_trans < 1:
                    continue
                rec = analyse_span(sp, network, ecache)
                rec["trip_name"] = name
                records.append(rec)
                all_spans.append(sp)
                flag = "ok" if rec["gate_ok"] else "GATE-FAIL"
                _log(f"      seg{rec['seg']}.sub{rec['sub']}  "
                     f"T={rec['n_trans']:>2}  ranks={rec['n_ranked']:>2}  "
                     f"surv={rec['n_survivors']}  forks={len(rec['forks'])}  [{flag}]")

    n_fail = sum(1 for r in records if not r["gate_ok"])
    _log(f"GATE: {len(records) - n_fail}/{len(records)} spans rank-1==MLE")
    if n_fail:
        _log("!!! gate failures present — extraction or top-K is wrong")

    if args.gate_only:
        return 0

    out = CACHE / "_topk_records.pkl"
    with open(out, "wb") as f:
        pickle.dump(dict(records=records, K_REPORT=K_REPORT, TAU=TAU,
                         SAMPLINGS=SAMPLINGS, TRIPS=TRIPS), f)
    _log(f"wrote {out}  ({len(records)} span records)")
    make_figures(records, all_spans, network, ecache)
    _print_summary(records)
    return 0


# ──────────────────────────────────────────────────────────────────── figures
SAMP_COLOR = {15: "#1f77b4", 60: "#ff7f0e", 120: "#2ca02c"}


def _jitter(x, n, rng, w=0.12):
    return np.full(n, x) + rng.uniform(-w, w, n)


def make_figures(records, spans, network, ecache):
    rng = np.random.default_rng(0)
    _fig1_distinctness(records, rng)
    _fig2_convergence(records, rng)
    _fig3_maps(records, spans, network, ecache)


def _fig1_distinctness(records, rng):
    """Diag 1: structural distinctness + near-tie score gaps + diversity collapse."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), dpi=120)
    samplings = sorted({r["sampling"] for r in records})

    # (A) edge-Jaccard of ranks 2..5 vs rank-1
    axA = axes[0]
    for si, s in enumerate(samplings):
        vals = [pr["jaccard"] for r in records if r["sampling"] == s for pr in r["per_rank"]]
        axA.scatter(_jitter(si, len(vals), rng), vals, s=26, alpha=0.6,
                    color=SAMP_COLOR[s], edgecolors="k", linewidths=0.3)
    axA.axhline(0.5, ls="--", color="grey", lw=1)
    axA.text(0.02, 0.52, "Jaccard≥0.5 = structurally distinct route", fontsize=8,
             color="grey", transform=axA.get_yaxis_transform())
    axA.text(0.02, 0.03, "≈0 = same roads (offset/twin only)", fontsize=8, color="grey",
             transform=axA.get_yaxis_transform())
    axA.set_xticks(range(len(samplings)))
    axA.set_xticklabels([f"{s}s" for s in samplings])
    axA.set_ylabel("edge-Jaccard distance, rank 2–5 vs MLE")
    axA.set_ylim(-0.05, 1.02)
    axA.set_title("(A) Are top-K structurally distinct?")

    # (B) score gap (nats) of ranks 2..5 vs rank-1
    axB = axes[1]
    for si, s in enumerate(samplings):
        vals = [pr["score_gap"] for r in records if r["sampling"] == s for pr in r["per_rank"]]
        axB.scatter(_jitter(si, len(vals), rng), vals, s=26, alpha=0.6,
                    color=SAMP_COLOR[s], edgecolors="k", linewidths=0.3)
    axB.set_xticks(range(len(samplings)))
    axB.set_xticklabels([f"{s}s" for s in samplings])
    axB.set_ylabel("log-prob gap to MLE (nats)")
    axB.set_title("(B) How much less likely?  (small = near-tie)")
    axB.axhline(0, color="grey", lw=0.6)

    # (C) diversity survivors out of 5 per span
    axC = axes[2]
    for si, s in enumerate(samplings):
        rs = [r for r in records if r["sampling"] == s]
        vals = [r["n_survivors"] for r in rs]
        axC.scatter(_jitter(si, len(vals), rng, w=0.14), vals, s=44, alpha=0.7,
                    color=SAMP_COLOR[s], edgecolors="k", linewidths=0.3)
    axC.set_xticks(range(len(samplings)))
    axC.set_xticklabels([f"{s}s" for s in samplings])
    axC.set_yticks(range(0, 6))
    axC.set_ylabel("distinct stories after τ=0.25 diversity filter (/5)")
    axC.set_title("(C) Diversity-filtered top-15 → how many survive?")
    axC.axhline(1, color="grey", ls=":", lw=1)

    fig.suptitle(
        "Diagnostic 1 — top-K Viterbi paths are offset/twin near-duplicates of the MLE; "
        "genuine route alternatives only on short ambiguous fragments",
        fontsize=11, y=1.02,
    )
    fig.tight_layout()
    p = CACHE / "_topk_fig1_distinctness.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    _log(f"wrote {p}")


def _fig2_convergence(records, rng):
    """Diag 2: state-marginal mass on both fork options (headline) vs raw path
    weights (illustration of over-resolution)."""
    sf = [(f["mass1"], f["mass2"], f["same_road"])
          for r in records for f in r["forks"] if f["kind"] == "state"]
    pf = [(f["path_w1"], f["path_w2"])
          for r in records for f in r["forks"] if f["kind"] == "path"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.4), dpi=120, sharex=True, sharey=True)

    for ax, data, title, sub in [
        (ax1, sf, "(A) STATE marginal at the fork  (headline)",
         "mass the per-transition marginal puts on each top-2 option"),
        (ax2, pf, "(B) raw PATH-object weight  (illustration)",
         "same forks — path posterior splits/under-resolves the same mass"),
    ]:
        if ax is ax1:
            for m1, m2, same in data:
                c = "#9467bd" if same else "#d62728"
                ax.scatter(m1, m2, s=70, alpha=0.8, color=c, edgecolors="k", linewidths=0.4)
        else:
            for w1, w2 in data:
                ax.scatter(w1, w2, s=70, alpha=0.8, color="#7f7f7f", edgecolors="k", linewidths=0.4)
        ax.plot([0, 1], [0, 1], ls="--", color="grey", lw=1)
        ax.axhline(0.15, ls=":", color="grey", lw=1)
        ax.axvline(0.15, ls=":", color="grey", lw=1)
        ax.set_xlim(0, 0.85)
        ax.set_ylim(0, 0.85)
        ax.set_xlabel("mass on rank-1's option")
        ax.set_title(title, fontsize=10)
        ax.text(0.5, -0.16, sub, fontsize=8, color="grey", ha="center", transform=ax.transAxes)
    ax1.set_ylabel("mass on rank-2's option")

    # legend for same_road
    from matplotlib.lines import Line2D
    ax1.legend(handles=[
        Line2D([], [], marker="o", ls="", color="#d62728", label="different road"),
        Line2D([], [], marker="o", ls="", color="#9467bd", label="same road (offset/twin)"),
    ], loc="upper right", fontsize=8)

    fig.suptitle(
        "Diagnostic 2 — where top-2 disagrees with top-1, the STATE marginal already puts real mass on BOTH "
        "(convergent);\nraw path-object weights under-resolve the same forks (the documented over-resolution)",
        fontsize=10.5, y=1.04,
    )
    fig.tight_layout()
    p = CACHE / "_topk_fig2_convergence.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    _log(f"wrote {p}")


def _plot_edges(ax, edge_links, network, **kw):
    segs = []
    for e in edge_links:
        try:
            idx = network.edge_index_for_link(int(e))
        except KeyError:
            continue
        segs.append(list(network.geoms[idx].coords))
    if segs:
        ax.add_collection(LineCollection(segs, **kw))


def _interleaved_edge_links(interleaved):
    out = []
    for x in interleaved:
        if not isinstance(x, MPath):
            continue
        out.extend(x.edges)
    return out


def _draw_span_map(ax, sp, network, title):
    from scripts._topk_viterbi_explore import topk_viterbi_span
    ranked = topk_viterbi_span(sp.log_emit, sp.log_trans, sp.best_path, sp.state_cands, K_RAW)
    obs = sp.observations
    lats = [o.lat for o in obs]
    lons = [o.lon for o in obs]
    pad = 0.0015
    bbox = (min(lons) - pad, min(lats) - pad, max(lons) + pad, max(lats) + pad)
    bpoly = box(*bbox)
    bg = [list(network.geoms[int(i)].coords)
          for i in np.asarray(network.tree.query(bpoly))
          if network.geoms[int(i)].intersects(bpoly)]
    if bg:
        ax.add_collection(LineCollection(bg, colors="0.86", linewidths=0.5, zorder=1))
    # ranks 2..5 (alternatives) in red, then rank-1 in blue on top
    for rt in ranked[1:K_REPORT]:
        _plot_edges(ax, _interleaved_edge_links(rt.interleaved), network,
                    colors="#d62728", linewidths=2.2, alpha=0.55, zorder=2)
    _plot_edges(ax, _interleaved_edge_links(ranked[0].interleaved), network,
                colors="#1f77b4", linewidths=1.4, alpha=0.95, zorder=3)
    ax.scatter(lons, lats, c="k", s=14, zorder=4)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=9.5)


def _fig3_maps(records, spans, network, ecache):
    """Two representative spans: a near-dup corridor and a genuine-fork fragment."""
    # near-dup: longest contiguous span with exactly 1 survivor
    contiguous = [(r, sp) for r, sp in zip(records, spans)
                  if r["n_survivors"] == 1 and r["n_trans"] >= 7]
    near = max(contiguous, key=lambda rs: rs[0]["n_trans"]) if contiguous else None
    # genuine fork: span with the most survivors (ties → fewest transitions = a fragment)
    fork = max(zip(records, spans), key=lambda rs: (rs[0]["n_survivors"], -rs[0]["n_trans"]))

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.2), dpi=120)
    if near is not None:
        r, sp = near
        _draw_span_map(axes[0], sp, network,
                       f"NEAR-DUPLICATE corridor\n{r['trip_name']} @ {r['sampling']}s  "
                       f"(T={r['n_trans']}, survivors=1)\nblue=MLE, red=ranks 2–5 (overlap → one story)")
    r, sp = fork
    _draw_span_map(axes[1], sp, network,
                   f"GENUINE FORK fragment\n{r['trip_name']} @ {r['sampling']}s  "
                   f"(T={r['n_trans']}, survivors={r['n_survivors']})\nblue=MLE, red=distinct alternatives")
    fig.suptitle("Diagnostic 1, geographic view — near-duplicate corridor (left) vs genuine structural fork (right)",
                 fontsize=11, y=1.0)
    fig.tight_layout()
    p = CACHE / "_topk_fig3_maps.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    _log(f"wrote {p}")


def _print_summary(records):
    print("\n=== top-K Viterbi exploration — summary ===")
    print(f"{'trip':<7}{'samp':>5}{'T':>4}{'ranks':>6}{'surv5':>6}"
          f"{'jac2':>7}{'gap2':>8}{'forks':>7}")
    for r in sorted(records, key=lambda x: (x["trip_name"], x["sampling"], x["seg"], x["sub"])):
        j2 = r["per_rank"][0]["jaccard"] if r["per_rank"] else float("nan")
        g2 = r["per_rank"][0]["score_gap"] if r["per_rank"] else float("nan")
        print(f"{r['trip_name']:<7}{r['sampling']:>5}{r['n_trans']:>4}"
              f"{r['n_ranked']:>6}{r['n_survivors']:>6}{j2:>7.2f}{g2:>8.2f}"
              f"{len(r['forks']):>7}")
    # composition check (marginal -> coherence, the falsifying direction)
    nfeas = sum(1 for r in records if r["composed_feasible"])
    neqmle = sum(1 for r in records if r["equals_mle"])
    nwc = sum(1 for r in records if r["within_cell_split_transitions"] > 0)
    print(f"\ncomposition (marginal-greedy → coherent story):")
    print(f"  feasible (connected) : {nfeas}/{len(records)} spans")
    print(f"  == MLE               : {neqmle}/{len(records)} spans")
    print(f"  spans with within-cell structural mass-splits (top-K blind spot): {nwc}/{len(records)}")
    in_topk = sum(1 for r in records if r["marginal_greedy_topk_rank"] >= 0)
    print(f"  marginal-greedy path is itself a top-15 Viterbi member: {in_topk}/{len(records)}")
    for r in records:
        if not r["composed_feasible"] or r["within_cell_split_transitions"] > 0 or not r["equals_mle"]:
            print(f"    {r['trip_name']:<7}{r['sampling']:>4}s seg{r['seg']}.sub{r['sub']} "
                  f"T={r['n_trans']:>2}: feasible={r['composed_feasible']} "
                  f"==MLE={r['equals_mle']} breaks={r['n_breaks']} "
                  f"wc_splits={r['within_cell_split_transitions']}")

    # aggregate by sampling
    print("\nby sampling (rank-2 vs rank-1, span-level means):")
    for s in sorted({r["sampling"] for r in records}):
        rs = [r for r in records if r["sampling"] == s and r["per_rank"]]
        jac = np.mean([r["per_rank"][0]["jaccard"] for r in rs])
        gap = np.mean([r["per_rank"][0]["score_gap"] for r in rs])
        surv = np.mean([r["n_survivors"] for r in rs])
        nf = np.mean([len([f for f in r["forks"] if f["kind"] == "state"]) for r in rs])
        print(f"  {s:>3}s: jaccard2={jac:.2f}  gap2={gap:.2f} nats  "
              f"diversity-survivors/5={surv:.1f}  state-forks(r1vr2)={nf:.1f}")


if __name__ == "__main__":
    sys.exit(main())
