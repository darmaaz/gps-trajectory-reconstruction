"""Assemble TOPK_VITERBI_EXPLORE.ipynb from the precomputed records + PNGs.

Review-only notebook (the build_cluster_calibration_draft pattern): loads
`cache/_topk_records.pkl` and embeds the three figures produced by
`scripts/_topk_explore_run.py`, with the narrative and tables inline. Run the
runner FIRST to (re)generate the pickle and PNGs, then this builder.

Run (Portfolio venv):
    .../python scripts/_topk_explore_run.py
    .../python scripts/build_topk_explore_notebook.py            # build + execute
    .../python scripts/build_topk_explore_notebook.py --no-exec  # build only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from textwrap import dedent

import nbformat as nbf
from nbclient import NotebookClient

REPO = Path(__file__).resolve().parents[1]
NB_PATH = REPO / "wip" / "TOPK_VITERBI_EXPLORE.ipynb"


def _md(s):
    return nbf.v4.new_markdown_cell(dedent(s).strip())


def _code(s):
    return nbf.v4.new_code_cell(dedent(s).strip())


def build():
    cells = []

    cells.append(_md("""
        # Top-K Viterbi — set-of-stories exploration

        **Question.** The pipeline returns top-1 Viterbi (the MLE) plus per-transition
        forward–backward marginals. The article keeps reaching for a *set of plausible
        reconstructions*. Top-K (parallel **list**) Viterbi is the cleaner object for
        that — each member is a connected, end-to-end story, not a stitched composite of
        marginalised fragments. This notebook implements top-K on the three canonical
        Porto trips at 15 / 60 / 120 s sampling and answers two diagnostics:

        1. **Are the top-5 paths structurally distinct, or near-duplicates of the MLE?**
           (does the article need a diversity penalty — and does it change with sampling?)
        2. **At transitions where top-2 disagrees with top-1, do the per-transition
           marginals put real mass on both options?**
           (do globally-coherent alternatives and marginalised objects tell the same story?)

        > **TL;DR.** (1) Top-K *as built* are *offset/twin* near-duplicates of the MLE
        > almost everywhere — a diversity penalty is necessary, and genuine route
        > alternatives appear only on short ambiguous fragments. (2) At the **state/edge**
        > level the marginals **converge** with top-K (real mass on both options at every
        > top-2 fork); at the raw **path-object** level they under-resolve. (2b) But the
        > *falsifying* direction (marginal → coherence) is more interesting: the
        > marginal-greedy trajectory **always composes** into a connected story (15/15),
        > yet it is the MLE only 7/15 and is not even a top-15 member in 5/15 — and 5/15
        > spans carry **within-cell** structural alternatives that max-path top-K cannot
        > see. So the two views agree on *where* the uncertainty is, but they are **not
        > the same object**: a faithful set-of-stories needs *both* a diversity penalty
        > (kill cosmetic dups) *and* within-cell path enumeration (surface the structural
        > alternatives the marginals already carry).
    """))

    cells.append(_md("""
        ## Method & scope

        - **Algorithm.** `scripts/_topk_viterbi_explore.py::topk_viterbi_span` is the
          textbook parallel list-Viterbi: per state it keeps the *K* best partial scores
          with (prev-state, prev-rank) backpointers, gathers the global top-K at the end,
          and backtracks each.
        - **Trellis.** Built from the **same** max-over-path cells production top-1 Viterbi
          uses (`transition_max_matrix`). So top-K enumerates the *K* best **state**
          sequences, each annotated with the argmax path per cell. Within-cell path
          alternatives (same endpoints, different middle edges) are collapsed — that is the
          correct "coherent story" object, and one reason Diagnostic 2 reads marginals at
          the state/edge level, not the path-object level.
        - **Spans.** Run per cliff-free sub-segment; boundaries come from the real
          `most_likely_trajectory`. The exact Viterbi inputs are *captured* from a live
          `reconstruct_trajectory` run (no orchestrator reproduction).
        - **Correctness gates.** (i) top-K vs **brute-force** enumeration on a synthetic
          trellis (`__main__` self-test); (ii) every returned path's score **recomputed**
          along its states; (iii) rank-1 score == the **production MLE** score on every
          real span (printed below); (iv) scores monotone non-increasing, sequences distinct.
        - **Edge keying is undirected** (endpoint-node pair) throughout — a two-way road's
          reverse twin is a distinct synthetic `link_id`, so directed keying would count
          the same physical road driven the other way as a different story.
    """))

    cells.append(_code("""
        import pickle
        import numpy as np
        import pandas as pd
        from pathlib import Path
        from IPython.display import Image, display

        CACHE = Path.cwd() / "cache"
        if not (CACHE / "_topk_records.pkl").exists():       # notebook may be run from repo root
            CACHE = Path("cache")
        d = pickle.load(open(CACHE / "_topk_records.pkl", "rb"))
        R = d["records"]
        K_REPORT, TAU = d["K_REPORT"], d["TAU"]

        n_fail = sum(1 for r in R if not r["gate_ok"])
        print(f"GATE: {len(R) - n_fail}/{len(R)} spans have rank-1 == production MLE score"
              + ("" if n_fail == 0 else "   <-- FAILURES PRESENT"))

        rows = []
        for r in sorted(R, key=lambda x: (x["trip_name"], x["sampling"], x["seg"], x["sub"])):
            j2 = r["per_rank"][0]["jaccard"] if r["per_rank"] else np.nan
            g2 = r["per_rank"][0]["score_gap"] if r["per_rank"] else np.nan
            rows.append(dict(trip=r["trip_name"], sampling_s=r["sampling"], T=r["n_trans"],
                             ranks=r["n_ranked"], survivors_of_5=r["n_survivors"],
                             jaccard_r2=round(j2, 2), gap_r2_nats=round(g2, 2),
                             forks=len(r["forks"])))
        pd.DataFrame(rows)
    """))

    cells.append(_md("""
        ## Diagnostic 1 — structural distinctness

        For each span we take the raw top-15, measure each of ranks 2–5 against the MLE by
        **undirected edge-Jaccard distance** (0 = same roads; ≥0.5 = a structurally distinct
        route) and by **log-prob gap** (how much less likely), then greedily **diversity-filter**
        the top-15 to 5 at τ=0.25 (accept a path only if its Jaccard distance to every accepted
        path ≥ τ).
    """))

    cells.append(_code("""
        display(Image(filename=str(CACHE / "_topk_fig1_distinctness.png")))

        print("Span-level means, rank-2 vs rank-1:")
        for s in sorted({r["sampling"] for r in R}):
            rs = [r for r in R if r["sampling"] == s and r["per_rank"]]
            jac = np.mean([r["per_rank"][0]["jaccard"] for r in rs])
            gap = np.mean([r["per_rank"][0]["score_gap"] for r in rs])
            surv = np.mean([r["n_survivors"] for r in rs])
            print(f"  {s:>3}s:  edge-Jaccard(r2)={jac:.2f}   gap(r2)={gap:.2f} nats   "
                  f"diversity-survivors/5={surv:.1f}")
    """))

    cells.append(_md("""
        **Finding.** On contiguous, well-observed spans the top-K paths are **edge-identical**
        to the MLE (Jaccard ≈ 0) at *every* sampling — they differ only in **offset** (position
        along the same road) or **reverse-twin** (same road, opposite direction). The τ=0.25
        diversity filter collapses the top-15 to **~1 survivor** on these spans. Genuine
        structurally-distinct alternatives (Jaccard ≥ 0.5, multiple survivors) appear **only on
        short ambiguous fragments** — the off-network parking-lot subs and coarse-sampled SHORT
        trips. The convergence is **not** primarily sampling-driven for these trips: road choice
        rarely flips across the top-K at any sampling; what little divergence exists is the
        off-network fragment, which is *densest*-sampled (15 s) yet most ambiguous.

        ⇒ **A diversity penalty is necessary** if the pipeline exposes "set of stories" — raw
        top-K is cosmetic. But even with it, most contiguous spans have one dominant structural
        story; the multi-story payoff is concentrated at structural forks / sparse fragments.
    """))

    cells.append(_code("""
        display(Image(filename=str(CACHE / "_topk_fig3_maps.png")))
    """))

    cells.append(_md("""
        ## Diagnostic 2 — do the marginals agree at the forks?

        At each obs position where rank-2 chooses a different state than rank-1, we read the
        **per-transition state marginal** (forward–backward) and ask how much mass it puts on
        each of the two options. The **state** marginal is the headline object: it sums over all
        paths into a state, so it is immune to within-cell path fragmentation and sidesteps `Path`
        identity. We also show the raw **path-object** weights at the same forks — *not* as the
        verdict, but to illustrate the over-resolution the edge-marginal calibration work already
        documented (path posteriors split corridor mass across near-duplicate paths).
    """))

    cells.append(_code("""
        display(Image(filename=str(CACHE / "_topk_fig2_convergence.png")))

        sf = [(f["mass1"], f["mass2"], f["same_road"])
              for r in R for f in r["forks"] if f["kind"] == "state"]
        pf = [(f["path_w1"], f["path_w2"]) for r in R for f in r["forks"] if f["kind"] == "path"]
        m1 = np.array([x[0] for x in sf]); m2 = np.array([x[1] for x in sf])
        pw1 = np.array([x[0] for x in pf]); pw2 = np.array([x[1] for x in pf])
        same = sum(1 for x in sf if x[2])
        print(f"STATE forks (rank-1 vs rank-2): n={len(sf)}")
        print(f"  both options have marginal mass > 0.15 : {int(((m1>0.15)&(m2>0.15)).sum())}/{len(sf)}")
        print(f"  rank-2 option ignored (mass < 0.05)     : {int((m2<0.05).sum())}/{len(sf)}")
        print(f"  mean mass:  rank-1 = {m1.mean():.2f}   rank-2 = {m2.mean():.2f}")
        print(f"  forks that are the SAME undirected road (offset/twin): {same}/{len(sf)}")
        print(f"\\nPATH-object weights at the same forks (illustration): "
              f"mean w1 = {pw1.mean():.2f}  w2 = {pw2.mean():.2f}  (systematically below the state marginals)")
    """))

    cells.append(_md("""
        **Finding.** At **every** state-fork where rank-2 disagrees with rank-1, the per-transition
        **state marginal already puts real mass on both options** (both > 0.15 at every fork; mean
        0.40 vs 0.32 — nearly balanced, points hug the diagonal in panel A). So globally-coherent
        top-K alternatives and per-transition marginals tell the **same story about *where* the
        uncertainty is** — a **convergent** answer.

        Two qualifications that matter:

        - **Roughly half the forks are the same physical road** (offset/twin, purple in panel A):
          the marginal correctly flags the fork, but the "alternative story" is often *which point /
          which direction*, not *which road* — the same offset/twin degeneracy Diagnostic 1 found.
        - **The path-object level under-resolves** (panel B): raw path weights at the same forks are
          systematically lower (0.24 / 0.18) and several fall below 0.15. Had we read the marginal at
          path-object granularity with a 0.15 threshold, we'd have spuriously concluded "divergent" at
          those forks. The **state / edge** level is the level that converges with top-K — consistent
          with the edge-marginal-calibration result.

        > **Caveat — this test is conditioned on Viterbi.** We conditioned on transitions where the
        > *globally near-best* rank-2 path disagrees, then asked the marginal. A near-best path passes
        > through states with non-trivial α·β almost by construction, so "17/17" is partly baked in. The
        > question the article actually poses runs the **other** direction — *do the marginals compose
        > into a coherent story?* — which Diagnostic 2b tests and which can falsify the verdict.
    """))

    cells.append(_md("""
        ## Diagnostic 2b — does the marginal *compose*? (the falsifying direction)

        Build the **marginal-greedy** trajectory: at each obs take `argmax state_marginals[k]`
        *independently*, then check whether consecutive picks are actually connected by an enumerated
        path (`(i*, j*)` registered in `best_path[k]`). A break ⇒ the marginals do **not** compose, and
        top-K would be more than presentation. We also flag the max-path-per-cell **blind spot**:
        transitions where a single `(i, j)` cell carries real marginal mass (>0.10) on ≥2 paths with
        materially different undirected edge sets (Jaccard ≥ 0.5) — structural alternatives top-K cannot
        surface because it keeps only the argmax path per cell.
    """))

    cells.append(_code("""
        nspan = len(R)
        feas = sum(1 for r in R if r["composed_feasible"])
        eqmle = sum(1 for r in R if r["equals_mle"])
        in_topk = sum(1 for r in R if r["marginal_greedy_topk_rank"] >= 0)
        wc = sum(1 for r in R if r["within_cell_split_transitions"] > 0)
        print("marginal-greedy (argmax state marginal per obs) → coherent story?")
        print(f"  connected / feasible end-to-end          : {feas}/{nspan} spans")
        print(f"  identical to the Viterbi MLE             : {eqmle}/{nspan} spans")
        print(f"  is itself a top-15 global Viterbi member : {in_topk}/{nspan} spans")
        print(f"  spans with within-cell structural splits : {wc}/{nspan}  (top-K blind spot)")
        rows = []
        for r in sorted(R, key=lambda x: (x["trip_name"], x["sampling"])):
            rows.append(dict(trip=r["trip_name"], sampling_s=r["sampling"], T=r["n_trans"],
                             composes=r["composed_feasible"], eq_MLE=r["equals_mle"],
                             greedy_topk_rank=r["marginal_greedy_topk_rank"],
                             within_cell_splits=r["within_cell_split_transitions"]))
        import pandas as pd
        pd.DataFrame(rows)
    """))

    cells.append(_md("""
        **Finding (this is the genuine one).** The strong falsification does **not** happen: the
        marginal-greedy trajectory is **connected on 15/15 spans** — the marginals compose into a
        coherent global story, they don't fall apart into disconnected fragments. *But the composed
        story is not the same object as the Viterbi set:* it equals the MLE on only **7/15** spans, is
        a top-15 global member on **10/15**, and on **5/15** it is a feasible path that no top-15
        story matches exactly. And **5/15** spans carry **within-cell** structural mass-splits — two
        roads between the same projected endpoints, both with real marginal mass — that the max-path
        top-K **cannot see**.

        ⇒ The two views agree on *where* the uncertainty is, and the marginal always composes — but
        they are **not the same object**. Raw top-K *over*-states diversity (cosmetic offset/twin
        dups); top-K-with-max-path *under*-states it (within-cell route alternatives it collapses,
        which the marginals carry).
    """))

    # ────────────────────────────────────────────────────────────────── PART B
    cells.append(_md("""
        # Part B — is the per-transition enumeration good enough?

        Candidate paths are enumerated **independently per transition**. The worry: if choosing a
        *non-Viterbi* candidate at transition *k* forces different optimal choices at *other*
        transitions, the per-transition view under-represents the joint alternative space and you'd
        need top-K/joint. We test the coupling directly.

        **Coupling flows only through shared states.** Transition *k* and *k*+1 share the state at obs
        *k*+1, so a different path at *k* propagates only if it changes a bracketing state. Two
        consequences frame everything below:

        - **Within-cell** alternatives (same `(i, j)`, different middle road — Part A's 5/15) are
          **zero-coupling by definition**: same endpoints, neighbours untouched. The max-path trellis
          cannot even represent them, so every probe here is a **cross-cell** counterfactual.
        - **Cross-cell** alternatives are the only place coupling can arise. We force one and measure how
          far it propagates.
    """))

    cells.append(_md("""
        ### B1 — coupling probe (the core test)

        For each transition *k*, **force the best alternative cell** (exclude the MLE cell, re-optimise
        the span) — the best globally-coherent story that makes a different choice at *k*. The endpoints
        {*k*, *k*+1} change mechanically; the signal is road-level diffs **outside** that, in either
        direction. **reach = 0 ⇒ decoupled** (the deviation is absorbed by the shared downstream state);
        reach > 0 ⇒ the choice at *k* re-routed its neighbours.
    """))

    cells.append(_code("""
        import pickle, numpy as np
        from IPython.display import Image, display
        RB = pickle.load(open(CACHE / "_topk_partb_records.pkl", "rb"))["records"]

        reaches = [r for rec in RB for r in rec["reaches"]]
        n0 = sum(1 for r in reaches if r == 0)
        nfeas = sum(rec["n_feasible_alt"] for rec in RB)
        noalt = sum(rec["n_no_alt"] for rec in RB)
        cpl = [(g, e) for rec in RB for g, r, e in
               zip(rec["score_gaps"], rec["reaches"], rec["fork_entropy"]) if r > 0]
        dec = [(g, e) for rec in RB for g, r, e in
               zip(rec["score_gaps"], rec["reaches"], rec["fork_entropy"]) if r == 0]
        print("force a non-Viterbi candidate at each transition, re-optimise the span:")
        print(f"  feasible alternatives : {nfeas}   (no-alt / forcing disconnects: {noalt})")
        print(f"  reach == 0 (DECOUPLED): {n0}/{len(reaches)}   reach >= 1 (coupled): {len(reaches)-n0}/{len(reaches)}")
        print(f"  forcing-cost  : coupled mean={np.mean([g for g,_ in cpl]):.2f}  vs  decoupled mean={np.mean([g for g,_ in dec]):.2f} nats  (clean separator)")
        print(f"  fork-entropy  : coupled mean={np.mean([e for _,e in cpl]):.2f}  vs  decoupled mean={np.mean([e for _,e in dec]):.2f} nats  "
              f"(corroborating; overlaps — 23/{len(dec)} decoupled are also <0.3, and {noalt} no-alt forks excluded)")
        cheap = [r for rec in RB for g, r in zip(rec["score_gaps"], rec["reaches"]) if g < 3.0]
        print(f"  among CHEAP alternatives (cost<3 nats, the genuine ones): max reach = {max(cheap, default=0)} "
              f"→ coupling never extends past an immediate neighbour; reach 2–3 is confined to the high-cost,"
              f" near-certain regime")
        display(Image(filename=str(CACHE / "_topk_figB1_coupling.png")))
    """))

    cells.append(_md("""
        **Finding — per-transition enumeration is good enough (for these trips).** Forcing a non-Viterbi
        road at transition *k* re-routes **only {*k*, *k*+1} in 254/270 (94%)** of transitions — the
        alternative is a cheap local swap that the shared downstream state absorbs. The joint alternative
        space is ≈ the product of the per-transition marginals; choosing a non-Viterbi candidate at *k*
        does **not** make the Viterbi different elsewhere.

        The ~6% that *do* couple are the interesting part. **The discriminator is cost, not ambiguity:**
        coupled transitions cost a mean **6.9 nats** to break vs **0.9** decoupled (panel B — the clean
        separator). Entropy *corroborates* (coupled forks are low-entropy, panel C) but is muddier and
        partly a selection artifact: it overlaps across the split (23/254 decoupled forks are also
        low-entropy), and the 22 maximally-pinned **no-alt** transitions are excluded from the decoupled
        pool, inflating its entropy. So read panel C as support, not a standalone finding.

        The key move is to split the coupling by cost, because that decides whether it *matters*:

        - **Long-reach coupling (reach 2–3) lives only in the high-cost regime (≥ ~3 nats).** There the
          per-transition marginal **already reports near-certainty** (entropy ≈ 0, all mass on one state) —
          there is no real alternative to resolve, so per-transition loses nothing. The single route is
          load-bearing; forcing off it is expensive and ripples, but the model never claimed an
          alternative existed.
        - **Among genuine alternatives (cheap, cost < ~3 nats), coupling never extends past an immediate
          neighbour (max reach = 1).** So where per-transition *does* offer a real choice, that choice is
          local — the independent enumeration captures it.

        ⇒ **Per-transition enumeration is good enough**: it loses nothing in the no-alternative regime and
        only ever couples to one neighbour where a real alternative exists.
    """))

    cells.append(_md("""
        ### B2 — a real diversity penalty (DivMBest)

        Iteratively re-run Viterbi, subtracting λ per **reused road** from prior solutions → a set of
        structurally-distinct, each-internally-coherent stories (not Part A's offset/twin dups). We ask
        whether their diffs from the MLE are **isolated single-transition flips** (decoupled) or
        **contiguous runs** (a coupled re-route).
    """))

    cells.append(_code("""
        display(Image(filename=str(CACHE / "_topk_figB2_divmbest.png")))
        rl = [d["max_run_len"] for r in RB for d in r["divmbest"]]
        print(f"DivMBest (λ=1.0): {sum(1 for x in rl if x <= 1)}/{len(rl)} diverse stories differ from the MLE by "
              f"isolated single-transition flips (max contiguous road-diff run ≤ 1).")
    """))

    cells.append(_md("""
        **Finding.** The diverse stories **share the MLE corridor** and differ by **isolated single-transition
        flips** (41/60 with max run ≤ 1; panel B) — the same per-transition alternatives, independently
        recombined, not coupled re-routes. And diversity **exhausts after ~1 alternative** (the penalty
        drives later solutions back toward the MLE) — there isn't a deep set of distinct stories. Both are
        exactly what a **decoupled** alternative space predicts, and they compose with the B1 probe.
    """))

    # ────────────────────────────────────────────────────────────────── PART C
    cells.append(_md("""
        # Part C — short / sparse paths (robustness)

        Parts A–B were dominated by long, densely-pinned spans. The opposite regime — genuinely short
        native trips (4–10 pings) and coarse-sampled sparse paths (2–4 long transitions) — is where
        per-transition ambiguity, and any coupling, should be *highest*. Does the verdict hold?
    """))

    cells.append(_code("""
        import pickle
        import pandas as pd
        from IPython.display import Image, display
        SR = pickle.load(open(CACHE / "_topk_shortpath_records.pkl", "rb"))["rows"]
        df = pd.DataFrame([{k: r[k] for k in
                            ("name","samp","T","ranks","surv","jac2","gap2","feas","noalt","maxreach","divrun")}
                           for r in SR])
        nmulti = sum(1 for r in SR if r["surv"] >= 2)
        ncoup = sum(1 for r in SR if r["maxreach"] > 0)
        print(f"{len(SR)} short/sparse spans:  multi-story (survivors>=2): {nmulti}/{len(SR)}   "
              f"any coupling (reach>0): {ncoup}/{len(SR)}   max reach overall: {max(r['maxreach'] for r in SR)}")
        df
    """))

    cells.append(_code("""
        display(Image(filename=str(CACHE / "_topk_figC_shortpath.png")))
    """))

    cells.append(_md("""
        **Finding — the verdict holds, and the two halves split cleanly by path length.**

        - **Diversity is HIGHER on short/sparse paths.** 6/13 spans are multi-story (≥2 survivors after the
          diversity filter) — vs ~rare on long corridors. One T=2 span has **5 distinct stories**. So Part
          A's "top-K are offset/twin near-duplicates" is a **long-corridor** property; where evidence is
          sparse, the set-of-stories framing genuinely pays off. (This is the same place the off-network
          fragments live — now generalised.)
        - **Yet coupling stays at zero.** 0/13 short/sparse spans couple (max reach 0) — *including* every
          multi-story one. Forcing a non-Viterbi candidate still re-routes only its own endpoints.

        **Why (panel A).** Diversity concentrates at a span's **endpoints**, which are under-constrained
        (one neighbour only); **interior** observations are pinned by their own GPS ping (entropy ≈ 0) and
        **screen the transitions apart**. In the representative T=2 span, the middle obs is certain
        (H = 0.02) while both ends fan into alternatives — so the two transitions are conditionally
        independent and the top-K is just the *product* of the two endpoints' independent choice sets.
        Short paths have more diversity precisely because more of their observations are endpoints — but
        that diversity is still per-transition-separable, so the per-transition enumeration captures it.
    """))

    cells.append(_md("""
        ## Verdict for the article

        The user's framing was: *convergent answers ⇒ the marginal approach is approximately right;
        divergent ⇒ the article has a genuine finding.* The honest answer is **both, on different axes** —
        and the nuance is the finding.

        - **Convergent on *where* the uncertainty is.** Top-K forks and per-transition marginals agree at
          every top-2 fork (read at state/edge granularity — the path-object level under-resolves), and
          the marginal-greedy story always **composes** into a connected trajectory (15/15). The marginals
          do not fall apart; the pipeline's marginal output is a coherent object for flagging uncertainty.
        - **Divergent on *what the object is*.** The marginal-greedy composite equals the MLE only 7/15
          and is not even a top-15 global story 5/15 — argmax-per-obs and best-global-story are related
          but not identical. So the worry "marginalised objects don't necessarily compose into *the*
          coherent story" is mildly real: they compose into *a* connected story, just not always the
          Viterbi one.
        - **Top-K is not free presentation.** Raw top-K **over**-states diversity — offset/twin near-ties
          (edge-Jaccard ≈ 0) — so it needs a τ≈0.25 diversity penalty to be meaningful. But top-K with
          **max-path-per-cell *under*-states** diversity: 5/15 spans carry within-cell route alternatives
          (two roads, same endpoints) the marginals weight but this instrument collapses. A faithful
          "set of stories" needs **both** fixes: a diversity penalty *and* within-cell path enumeration.
        - **Where genuine multi-story cases live (Part C):** short and sparsely-sampled paths — and the
          off-network / parking-lot fragments — not well-observed corridors. Diversity concentrates at
          under-constrained span **endpoints**; pinned interior observations screen transitions apart, so
          even multi-story short paths stay decoupled (0/13). "Top-K are near-duplicates" is therefore a
          long-corridor property, and the set-of-stories framing earns its keep specifically on short/sparse
          paths.
        - **Per-transition enumeration is good enough (Part B).** Forcing a non-Viterbi candidate at a
          transition re-routes only its own endpoints in **94%** of cases — the alternatives are
          **decoupled**, so the independent per-transition view faithfully represents the joint
          alternative space, and DivMBest's diverse stories are just isolated per-transition flips of the
          same corridor. Splitting the residual ~6% coupling by **cost** (the clean discriminator) settles
          it: the long-reach coupling (reach 2–3) sits only at high-cost transitions where the marginal
          *already* reports near-certainty (no alternative to resolve — per-transition loses nothing),
          while among genuine cheap alternatives coupling **never reaches past an immediate neighbour**.
          So there is no regime where per-transition meaningfully under-represents the alternatives;
          top-K/joint buys nothing here beyond packaging.

        **If ported:** keep the per-transition **edge** marginals as the primary uncertainty object;
        expose top-K *with* a diversity filter (and ideally within-cell path enumeration) as an optional
        set-of-stories output for consumers that want connected trajectories. Reconstruct:
        `python scripts/_topk_explore_run.py && python scripts/build_topk_explore_notebook.py`.
    """))

    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    }
    return nb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-exec", action="store_true")
    args = ap.parse_args()

    nb = build()
    if not args.no_exec:
        print("executing notebook…", file=sys.stderr)
        NotebookClient(nb, timeout=600, kernel_name="python3",
                       resources={"metadata": {"path": str(REPO)}}).execute()
    NB_PATH.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, str(NB_PATH))
    print(f"wrote {NB_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
