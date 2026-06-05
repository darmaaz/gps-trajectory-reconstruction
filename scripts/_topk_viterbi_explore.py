"""Top-K (parallel list) Viterbi over the trajectory CRF — exploration lib.

Temp exploration module (underscore-prefixed, not part of the public API).
Backs `TOPK_VITERBI_EXPLORE.ipynb`. Answers two diagnostics:

  (1) Are the top-K globally-coherent reconstructions structurally distinct,
      or near-duplicates of the MLE? (does the article need a diversity
      penalty, and how does that change with sampling granularity?)
  (2) At transitions where top-2 Viterbi disagrees with top-1, do the
      per-transition forward-backward marginals also put real mass on both
      options? (do globally-coherent alternatives and marginalised objects
      tell the same story?)

Scope of the K-best object
--------------------------
The trellis cell `log_trans[k][i, j]` is the **max** over paths joining
state i to state j (same matrix the production top-1 Viterbi uses, via
`transition_max_matrix`). So `topk_viterbi_span` enumerates the K best
*state* sequences, each annotated with the argmax path per (i, j) cell. It
does NOT enumerate within-cell path alternatives (same endpoints, different
middle edges) — that is the textbook list-Viterbi object and the right one
for "coherent end-to-end stories". Within-cell mass is structurally
invisible to top-K, which is one reason diagnostic 2 aggregates marginals
to the state/edge level rather than reading raw path-object weights.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from itertools import product

import numpy as np

import src.api.pipeline as pipeline_mod
from src.api import reconstruct_trajectory
from src.inference import forward_backward
from src.inference._common import (
    emission_log_potentials,
    transition_max_matrix,
    transition_triples,
)
from src.model import Path

NEG = -np.inf


# ───────────────────────────────────────────────────── trellis construction
def build_trellis(state_cands, path_cands, observations, budgets, emission, transition):
    """Build `(log_emit, log_trans, best_path)` exactly as production Viterbi.

    `log_emit[k]`  : (|X_k|,) emission log-potentials.
    `log_trans[k]` : (|X_k|, |X_{k+1}|) max-over-path log-factors.
    `best_path[k]` : {(i, j): Path} argmax path per cell.
    """
    log_emit = emission_log_potentials(state_cands, observations, emission)
    L = len(state_cands)
    triples = (
        transition_triples(state_cands, path_cands, transition, budgets)
        if L >= 2 else []
    )
    log_trans, best_path = [], []
    for k in range(L - 1):
        if not state_cands[k] or not state_cands[k + 1] or not path_cands[k]:
            log_trans.append(np.empty((0, 0)))
            best_path.append({})
            continue
        M, best = transition_max_matrix(
            triples[k], len(state_cands[k]), len(state_cands[k + 1]),
        )
        log_trans.append(M)
        best_path.append(best)
    return log_emit, log_trans, best_path


# ───────────────────────────────────────────────────────── top-K list Viterbi
@dataclass
class RankedTrajectory:
    score: float
    state_indices: list[int]       # candidate index per obs, length L
    interleaved: list              # [State, Path, ..., State], length 2L-1


def topk_viterbi_span(log_emit, log_trans, best_path, state_cands_span, K):
    """Parallel list-Viterbi: the K best globally-coherent state sequences.

    Per state j at step k, keeps up to K best partial scores with
    (prev_state, prev_rank) backpointers. Assumes a cliff-free span (every
    consumed transition has at least one finite cell) — callers slice spans
    from `most_likely_trajectory`'s subs so this holds.
    """
    L = len(log_emit)
    if L == 0:
        return []

    # delta[k][j] : list of (score, prev_j, prev_rank) sorted desc, len <= K.
    delta: list[list[list[tuple[float, int, int]]]] = [None] * L  # type: ignore
    delta[0] = [
        [(float(log_emit[0][j]), -1, -1)] if np.isfinite(log_emit[0][j]) else []
        for j in range(len(log_emit[0]))
    ]
    for k in range(L - 1):
        M = log_trans[k]
        nkp1 = len(log_emit[k + 1])
        dk1: list[list[tuple[float, int, int]]] = []
        for j in range(nkp1):
            ej = log_emit[k + 1][j]
            if not np.isfinite(ej) or M.size == 0:
                dk1.append([])
                continue
            cand: list[tuple[float, int, int]] = []
            for i in range(len(delta[k])):
                lij = M[i, j]
                if not np.isfinite(lij):
                    continue
                for r, (s, _, _) in enumerate(delta[k][i]):
                    cand.append((s + lij, i, r))
            cand.sort(key=lambda x: x[0], reverse=True)
            dk1.append([(s + float(ej), i, r) for (s, i, r) in cand[:K]])
        delta[k + 1] = dk1

    finals: list[tuple[float, int, int]] = []
    for j, lst in enumerate(delta[L - 1]):
        for r, (s, _, _) in enumerate(lst):
            if np.isfinite(s):
                finals.append((s, j, r))
    finals.sort(key=lambda x: x[0], reverse=True)

    out: list[RankedTrajectory] = []
    for (s, j, r) in finals[:K]:
        idxs = [0] * L
        cj, cr = j, r
        for k in range(L - 1, -1, -1):
            idxs[k] = cj
            _, pj, pr = delta[k][cj][cr]
            cj, cr = pj, pr
        inter: list = []
        for k in range(L):
            inter.append(state_cands_span[k][idxs[k]])
            if k < L - 1:
                inter.append(best_path[k][(idxs[k], idxs[k + 1])])
        out.append(RankedTrajectory(float(s), idxs, inter))
    return out


# ──────────────────────────────────────────────────────────── correctness gates
def recompute_score(state_indices, log_emit, log_trans):
    """Re-sum emission+transition along a backtracked path from the trellis.

    Independent of the DP's stored scores/backpointers — catches backpointer
    threading bugs (a returned sequence whose score doesn't match its states).
    """
    s = float(log_emit[0][state_indices[0]])
    for k in range(len(state_indices) - 1):
        i, j = state_indices[k], state_indices[k + 1]
        s += float(log_trans[k][i, j]) + float(log_emit[k + 1][j])
    return s


def brute_force_topk(log_emit, log_trans, K):
    """Enumerate ALL state sequences, score, sort. Gold standard for tiny spans."""
    L = len(log_emit)
    scored: list[tuple[float, tuple[int, ...]]] = []
    for combo in product(*[range(len(e)) for e in log_emit]):
        s = float(log_emit[0][combo[0]])
        ok = np.isfinite(s)
        for k in range(L - 1):
            if not ok:
                break
            s += float(log_trans[k][combo[k], combo[k + 1]]) + float(log_emit[k + 1][combo[k + 1]])
            ok = np.isfinite(s)
        if ok:
            scored.append((s, combo))
    scored.sort(key=lambda x: -x[0])
    return scored[:K]


# ───────────────────────────────────────────── pipeline-internal extraction
@dataclass
class Span:
    """One cliff-free sub-segment with everything top-K and FB need."""
    trip_id: str
    sampling_s: int
    seg_idx: int                   # which captured pre-segment
    sub_idx: int                   # which sub within it
    state_cands: list              # span-local list[list[State]]
    path_cands: list               # span-local list[list[Path]]
    observations: list
    budgets: list
    emission: object
    transition: object
    mle_interleaved: list          # production most_likely for this sub
    # filled lazily:
    log_emit: object = None
    log_trans: object = None
    best_path: object = None
    state_marginals: object = None
    path_marginals: object = None
    log_partition: float = 0.0

    @property
    def n_trans(self):
        return len(self.state_cands) - 1


@contextlib.contextmanager
def _capture_viterbi_inputs():
    """Capture exact inputs/outputs of every `most_likely_trajectory` call.

    Monkeypatches the name in the pipeline module so we get the production
    rebound emission/transition and the seg-level candidate/budget arrays
    with ZERO reproduction of the orchestrator's front half.
    """
    captured: list[dict] = []
    orig = pipeline_mod.most_likely_trajectory

    def shim(state_candidates, path_candidates, observations, emission, transition, time_budgets):
        subs = orig(state_candidates, path_candidates, observations, emission, transition, time_budgets)
        captured.append(dict(
            state_cands=state_candidates, path_cands=path_candidates,
            observations=observations, budgets=time_budgets,
            emission=emission, transition=transition, subs=subs,
        ))
        return subs

    pipeline_mod.most_likely_trajectory = shim
    try:
        yield captured
    finally:
        pipeline_mod.most_likely_trajectory = orig


def extract_spans(trip_id, sampling_s, raw_obs, network, config):
    """Run the real pipeline, returning one `Span` per cliff-free sub.

    Each Span carries the production MLE (`sub.most_likely`) plus the exact
    trellis inputs, so the caller can (a) gate top-K rank-1 against the
    production Viterbi score and (b) run FB on the identical slice.
    """
    with _capture_viterbi_inputs() as captured:
        reconstruct_trajectory(raw_obs, network, config)

    spans: list[Span] = []
    for seg_idx, cap in enumerate(captured):
        for sub_idx, sub in enumerate(cap["subs"]):
            s, e = sub.start_obs_idx, sub.end_obs_idx     # inclusive, seg-local
            sc = cap["state_cands"][s:e + 1]
            pc = cap["path_cands"][s:e] if e > s else []
            ob = cap["observations"][s:e + 1]
            bg = cap["budgets"][s:e] if e > s else []
            sp = Span(
                trip_id=trip_id, sampling_s=sampling_s, seg_idx=seg_idx,
                sub_idx=sub_idx, state_cands=sc, path_cands=pc, observations=ob,
                budgets=bg, emission=cap["emission"], transition=cap["transition"],
                mle_interleaved=sub.most_likely,
            )
            le, lt, bp = build_trellis(sc, pc, ob, bg, sp.emission, sp.transition)
            sp.log_emit, sp.log_trans, sp.best_path = le, lt, bp
            sm, pm, lz = forward_backward(sc, pc, ob, sp.emission, sp.transition, bg)
            sp.state_marginals, sp.path_marginals, sp.log_partition = sm, pm, lz
            spans.append(sp)
    return spans


# ────────────────────────────────────────────────── structural-distinctness
def undirected_edge(network, link_id, _cache):
    """Undirected road identity = frozenset of the edge's two endpoint nodes.

    A two-way road's forward edge and its reverse twin (a synthetic link_id)
    share the node pair — the only recoverable shared identity. Compare on
    this, never raw link_ids (the reverse-twin keying footgun).
    """
    key = int(link_id)
    hit = _cache.get(key)
    if hit is None:
        i = network.edge_index_for_link(key)
        hit = frozenset((int(network.from_node[i]), int(network.to_node[i])))
        _cache[key] = hit
    return hit


def interleaved_edges(interleaved, network, cache):
    es = set()
    for x in interleaved:
        if isinstance(x, Path):
            for e in x.edges:
                es.add(undirected_edge(network, e, cache))
    return es


def state_links_undirected(interleaved, network, cache):
    return [
        undirected_edge(network, x.link_id, cache)
        for x in interleaved if not isinstance(x, Path)
    ]


def jaccard_dist(a, b):
    if not a and not b:
        return 0.0
    union = len(a | b)
    return 1.0 - len(a & b) / union if union else 0.0


def state_hamming_frac(links_a, links_b):
    """Fraction of obs positions where the (undirected) chosen link differs."""
    n = min(len(links_a), len(links_b))
    if n == 0:
        return 0.0
    return sum(1 for i in range(n) if links_a[i] != links_b[i]) / n


def diversity_filter(ranked, network, cache, k_keep=5, tau=0.25):
    """Greedy: accept the next ranked path only if its edge-Jaccard distance
    to every already-accepted path is >= tau. Needs a raw pool larger than
    k_keep so the filter can actually drop near-duplicates.
    """
    kept, kept_edges = [], []
    for rt in ranked:
        es = interleaved_edges(rt.interleaved, network, cache)
        if all(jaccard_dist(es, ke) >= tau for ke in kept_edges):
            kept.append(rt)
            kept_edges.append(es)
            if len(kept) >= k_keep:
                break
    return kept


# ─────────────────────────────────────────── Part B: diversity + coupling probe
#
# Coupling flows ONLY through shared states: transition k and k+1 share the state
# at obs k+1, so a different path at k propagates only if it changes a bracketing
# state. Within-cell alternatives (same (i, j), different middle road) are
# zero-coupling BY DEFINITION — same endpoints, neighbours untouched — and the
# max-path trellis cannot even represent them. So every probe below is a
# CROSS-CELL counterfactual: force a different cell at k, and coupling = how far
# the re-optimised road sequence deviates from the MLE *outside* {k, k+1}.

def mle_state_indices(span):
    from src.model import Path as _P
    return [
        span.state_cands[k].index(x)
        for k, x in enumerate(s for s in span.mle_interleaved if not isinstance(s, _P))
    ]


def _road_per_obs(state_indices, span, network, cache):
    return [
        undirected_edge(network, span.state_cands[o][state_indices[o]].link_id, cache)
        for o in range(len(state_indices))
    ]


def state_marginal_entropy(span):
    """Per-obs entropy (nats) of the FB state marginal — the chain's local
    pinned-ness. Peaked (≈0) ⇒ neighbours conditionally independent;
    diffuse ⇒ the choice here constrains its neighbours."""
    out = []
    for k in range(len(span.state_cands)):
        q = np.array([span.state_marginals[k].get(s, 0.0) for s in span.state_cands[k]])
        q = q[q > 0]
        out.append(float(-(q * np.log(q)).sum()) if q.size else 0.0)
    return out


def force_alt_and_reopt(span, k, network, cache):
    """Force the best alternative to the MLE cell at transition k, re-optimise.

    Excludes the single MLE cell (i*, j*) at k and re-runs Viterbi over the span
    — the best globally-coherent story that makes a *different* choice at k.
    Spillover = road-level diffs at obs OUTSIDE {k, k+1} (the endpoints change
    mechanically; they are not the signal). Returns None bookkeeping when k has
    no alternative cell (forced ⇒ disconnects)."""
    le, lt0, bp = span.log_emit, span.log_trans, span.best_path
    mle = mle_state_indices(span)
    istar, jstar = mle[k], mle[k + 1]
    n_cells_k = len(bp[k])
    lt = [m.copy() for m in lt0]
    lt[k][istar, jstar] = NEG
    sol = topk_viterbi_span(le, lt, bp, span.state_cands, 1)
    if not sol or not np.isfinite(sol[0].score):
        return dict(k=k, feasible=False, n_cells_k=n_cells_k)
    idx = sol[0].state_indices
    rmle = _road_per_obs(mle, span, network, cache)
    ralt = _road_per_obs(idx, span, network, cache)
    diff = [o for o in range(len(rmle)) if rmle[o] != ralt[o]]
    spill = [o for o in diff if o not in (k, k + 1)]
    reach = max((abs(o - k) for o in spill), default=0)
    road_changed = any(o in (k, k + 1) for o in diff) or bool(spill)
    H = state_marginal_entropy(span)
    return dict(
        k=k, feasible=True, n_cells_k=n_cells_k,
        diff_obs=diff, spillover_obs=spill, reach=reach, n_spill=len(spill),
        road_changed_at_k=any(o in (k, k + 1) for o in diff),
        score_gap=recompute_score(mle, le, lt0) - recompute_score(idx, le, lt0),
        fork_entropy=max(H[k], H[k + 1]),
    )


def divmbest_viterbi(span, network, cache, lam, M):
    """DivMBest: iteratively re-run Viterbi penalising reuse of roads from prior
    solutions (−λ per shared undirected edge per cell). Yields M structurally
    diverse, each-internally-coherent global stories. Scores are recomputed on
    the ORIGINAL (unpenalised) trellis for honest comparison."""
    le, lt0, bp = span.log_emit, span.log_trans, span.best_path
    cell_edges = [
        {(i, j): frozenset(undirected_edge(network, e, cache) for e in p.edges)
         for (i, j), p in bp[k].items()}
        for k in range(len(bp))
    ]
    used: set = set()
    sols = []
    for _m in range(M):
        lt = []
        for k in range(len(lt0)):
            Mk = lt0[k].copy()
            if used and Mk.size:
                for (i, j), es in cell_edges[k].items():
                    ov = len(es & used)
                    if ov:
                        Mk[i, j] -= lam * ov
            lt.append(Mk)
        sol = topk_viterbi_span(le, lt, bp, span.state_cands, 1)
        if not sol or not np.isfinite(sol[0].score):
            break
        idx = sol[0].state_indices
        sols.append((idx, recompute_score(idx, le, lt0), sol[0].interleaved))
        for x in sol[0].interleaved:
            if isinstance(x, Path):
                for e in x.edges:
                    used.add(undirected_edge(network, e, cache))
    return sols


def road_diff_runs(idx, span, network, cache):
    """Contiguous obs-runs where `idx`'s road differs from the MLE. Returns a
    list of (start, end) inclusive runs — long runs = a coupled re-route,
    many length-1 runs = isolated decoupled flips."""
    mle = mle_state_indices(span)
    rmle = _road_per_obs(mle, span, network, cache)
    ralt = _road_per_obs(idx, span, network, cache)
    diff = [o for o in range(len(rmle)) if rmle[o] != ralt[o]]
    runs = []
    for o in diff:
        if runs and o == runs[-1][1] + 1:
            runs[-1][1] = o
        else:
            runs.append([o, o])
    return [tuple(r) for r in runs]


if __name__ == "__main__":
    # Self-test: the K-best gates on a hand-built synthetic trellis. No network
    # needed — exercises topk_viterbi_span vs brute force + recompute-score.
    log_emit = [
        np.array([0.0, -0.2, -1.0]),
        np.array([-0.1, -0.3, -2.0]),
        np.array([0.0, -0.5]),
        np.array([-0.2, -0.4, -0.9]),
    ]
    rng = np.random.default_rng(0)
    log_trans = []
    for k in range(3):
        M = -rng.random((len(log_emit[k]), len(log_emit[k + 1]))) * 2.0
        # punch a few -inf to make it interesting
        M[0, -1] = NEG
        log_trans.append(M)
    state_cands = [[("s", k, j) for j in range(len(log_emit[k]))] for k in range(4)]
    best_path = [
        {(i, j): None for i in range(M.shape[0]) for j in range(M.shape[1])
         if np.isfinite(M[i, j])}
        for M in log_trans
    ]
    K = 6
    # topk vs brute force
    bf = brute_force_topk(log_emit, log_trans, K)
    tk = topk_viterbi_span(log_emit, log_trans, best_path, state_cands, K)
    print("brute :", [round(s, 4) for s, _ in bf])
    print("topk  :", [round(t.score, 4) for t in tk])
    assert len(tk) == len(bf), (len(tk), len(bf))
    for (bs, bcombo), t in zip(bf, tk):
        assert abs(bs - t.score) < 1e-9, (bs, t.score)
        assert tuple(t.state_indices) == bcombo, (bcombo, t.state_indices)
        rs = recompute_score(t.state_indices, log_emit, log_trans)
        assert abs(rs - t.score) < 1e-9, (rs, t.score)
    # monotone non-increasing + distinct
    scores = [t.score for t in tk]
    assert scores == sorted(scores, reverse=True)
    seqs = {tuple(t.state_indices) for t in tk}
    assert len(seqs) == len(tk)
    print("OK — topk == brute force, scores recompute, monotone, distinct")
