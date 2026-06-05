"""Build STORY_edge_calibration_draft.ipynb — edge-level route calibration.

Loads the held-out records collected by scripts/demo_cluster_calibration_draft.py
(both eval configurations, graded against the same neutral 15 s reference) and
renders the calibration analysis inline. Records are produced by:
    python scripts/demo_cluster_calibration_draft.py --indep-truth   # trained prior
    python scripts/demo_cluster_calibration_draft.py --eval-mu0       # no prior
"""

from __future__ import annotations

import sys
from pathlib import Path
from textwrap import dedent

import nbformat as nbf
from nbclient import NotebookClient

REPO_ROOT = Path(__file__).resolve().parents[1]
NB_PATH = REPO_ROOT / "wip" / "STORY_edge_calibration_draft.ipynb"
REC_TRAINED = REPO_ROOT / "cache" / "_cluster_calib_records_indeptruth.pkl"
REC_NOPRIOR = REPO_ROOT / "cache" / "_cluster_calib_records_eval_mu0.pkl"


def _md(s: str) -> dict:
    return nbf.v4.new_markdown_cell(dedent(s).strip() + "\n")


def _code(s: str) -> dict:
    return nbf.v4.new_code_cell(dedent(s).strip() + "\n")


def build_notebook() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.13"},
    }
    cells: list[dict] = []

    cells.append(_md("""
        # Do the route probabilities mean what they say?

        Between two sparse GPS fixes the path is ambiguous: many road
        sequences fit the same few points. The pipeline does not commit to
        one. It returns a set of candidate routes, each with a probability
        that sums to one across the set. Add up those probabilities over
        every candidate route that uses a given road segment and you get,
        for that segment, the model's belief that the vehicle actually drove
        it — call it **P(segment)**.

        The question: is P(segment) honest? When the model stamps a segment
        with 0.7, was that segment on the true route 70 % of the time?
    """))

    cells.append(_md("""
        **Construction.**

        - 297 transitions drawn from 40 trips held out of training.
        - The true route is the same trip reconstructed from the full
          15-second trace (eight times denser than the input). At that
          density the GPS points pin the route on their own.
        - A road segment is named by its two endpoints, direction ignored —
          a two-way street is one segment, not two.
        - For each segment touched by any candidate route, P(segment) is the
          summed probability of the routes through it. Every
          (transition, segment) pair is one observation. Bin the observations
          by P(segment); within each bin, measure the share that lie on the
          true route. A model whose probabilities are honest has, in the bin
          centred on 0.7, about 70 % of its segments on the true route.
    """))

    cells.append(_code("""
        import pickle, sys
        from pathlib import Path
        REPO_ROOT = Path.cwd()
        if str(REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(REPO_ROOT))
        import numpy as np
        import matplotlib.pyplot as plt
        %matplotlib inline
        plt.rcParams["figure.dpi"] = 110
        plt.rcParams["font.size"] = 10

        def _load(p):
            with open(p, "rb") as f:
                return pickle.load(f)["records"]
        TRAINED = _load(REPO_ROOT / "cache" / "_cluster_calib_records_indeptruth.pkl")
        NOPRIOR = _load(REPO_ROOT / "cache" / "_cluster_calib_records_eval_mu0.pkl")

        BINS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0001]
        LAB  = ["[0.0,0.2)", "[0.2,0.4)", "[0.4,0.6)", "[0.6,0.8)", "[0.8,1.0]"]

        def edge_rows(recs):
            # (P(segment), on_true_route) for every candidate segment.
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
            # average distance between stated probability and observed share,
            # weighted by how many segments fall in each bin (lower is better).
            n, pred, obs = by_bin(rows); N = sum(n)
            return 100 * sum(n[i] / N * abs(pred[i] - obs[i])
                             for i in range(len(LAB)) if n[i])

        def _pava(y):
            v, w = [], []
            for yi in y:
                v.append(float(yi)); w.append(1.0)
                while len(v) > 1 and v[-2] > v[-1] + 1e-15:
                    nv = (v[-2]*w[-2] + v[-1]*w[-1]) / (w[-2] + w[-1]); nw = w[-2] + w[-1]
                    del v[-2:], w[-2:]; v.append(nv); w.append(nw)
            out = []
            for vi, wi in zip(v, w):
                out += [vi] * int(round(wi))
            return out

        def rescaled(recs):
            # Fit one monotone rescaling on half the transitions, apply to the
            # other half; return the held-out (rescaled P, on_true_route).
            sc = [r for r in recs if r["truth_edges"] and r["paths"]]
            tr = edge_rows(sc[0::2]); te = edge_rows(sc[1::2])
            tr.sort(key=lambda r: r[0])
            xs = np.array([a for a, _ in tr]); fit = np.array(_pava([b for _, b in tr]))
            te_p = np.array([a for a, _ in te]); te_y = np.array([b for _, b in te])
            q = np.array([fit[min(max(np.searchsorted(xs, p, "right")-1, 0), len(fit)-1)]
                          for p in te_p])
            return list(zip(q, te_y))

        print(f"trained prior: {len(edge_rows(TRAINED))} segments | "
              f"no prior: {len(edge_rows(NOPRIOR))} segments")
    """))

    cells.append(_md("## The result"))

    cells.append(_code("""
        for name, recs in (("no route prior", NOPRIOR), ("trained prior", TRAINED)):
            n, pred, obs = by_bin(edge_rows(recs))
            print(f"\\n{name}:")
            print(f"  {'P(segment)':>11}  {'segments':>9}  {'stated':>7}  {'on true route':>14}")
            for i in range(len(LAB)):
                print(f"  {LAB[i]:>11}  {n[i]:>9}  {100*pred[i]:>6.0f}%  {100*obs[i]:>13.0f}%")

        print("\\nstated-vs-observed gap, measured on a held-out half "
              "(rescale fit on the other half):")
        for name, recs in (("no route prior", NOPRIOR), ("trained prior", TRAINED)):
            sc = [r for r in recs if r["truth_edges"] and r["paths"]]
            raw = gap(edge_rows(sc[1::2])); res = gap(rescaled(recs))
            print(f"  {name:>14}:  stated {raw:.1f}%   ->   rescaled {res:.1f}%")
    """))

    cells.append(_code("""
        fig, (axL, axR) = plt.subplots(1, 2, figsize=(14, 5.4))

        # Left: stated probability vs observed share, no route prior.
        n, pred, obs = by_bin(edge_rows(NOPRIOR))
        x = np.arange(len(LAB))
        bars = axL.bar(x, [100*o for o in obs], color="#7c3aed", alpha=0.85,
                       edgecolor="white")
        for i, b in enumerate(bars):
            axL.text(b.get_x()+b.get_width()/2, 100*obs[i]+1, f"{n[i]}",
                     ha="center", fontsize=7, color="#5b21b6")
        mids = [100*(BINS[i]+min(BINS[i+1], 1.0))/2 for i in range(len(LAB))]
        axL.plot(x, mids, "--", color="#64748b", lw=1, label="honest")
        axL.set_xticks(x); axL.set_xticklabels(LAB, rotation=30, fontsize=8)
        axL.set_xlabel("P(segment) the model states")
        axL.set_ylabel("share actually on the true route (%)")
        axL.set_ylim(0, 105); axL.legend(fontsize=8)
        axL.set_title("Stated probability vs reality (no route prior)\\nbars = observed share, n above")

        # Right: reliability lines for both models, stated vs observed.
        def line(rows):
            _, pred, obs = by_bin(rows)
            return [100*p for p in pred], [100*o for o in obs]
        for rows, color, lab in (
            (edge_rows(NOPRIOR), "#7c3aed", "no route prior"),
            (edge_rows(TRAINED), "#16a34a", "trained prior"),
        ):
            xs, ys = line(rows); axR.plot(xs, ys, "o-", color=color, label=lab)
        axR.plot([0, 100], [0, 100], "--", color="#64748b", lw=1, label="honest")
        axR.set_xlim(0, 100); axR.set_ylim(0, 105)
        axR.set_xlabel("stated probability (%)")
        axR.set_ylabel("share on true route (%)")
        axR.legend(fontsize=8)
        axR.set_title("Stated vs observed, with and without the prior")
        plt.tight_layout(); plt.show()
    """))

    cells.append(_md("""
        ## What it says

        **The ordering is correct.** More stated probability means a higher
        chance the segment was driven, with no reversals, and the ambiguous
        middle is dense, not sparse: thousands of segments sit near one-half.
        Those are real forks — places where the sparse data genuinely leaves
        two ways open — and the model ranks them in the right order.

        **The raw numbers are overstated, and overstated predictably.** A
        segment the prior-free model calls 0.49 was driven 35 % of the time;
        one it calls 0.29, 9 % of the time. But the observed share climbs
        monotonically with the stated one, so the error is a consistent
        rescaling, not noise. Fit that rescaling on one half of the
        transitions, apply it to the other, and the average gap falls from
        11 % to 3 %. The order was already right; only the scale was off.

        **The trained route prior is not what makes this work.** Strip the
        prior out entirely — treat every candidate route as equally likely
        before the GPS evidence — and the same rescale lands at the same
        place (3 % against the trained prior's 2 %). What the prior buys is
        the overstatement correction, and the rescale supplies that for free.
        The trustworthy part — the ordering — comes from the candidate
        generation and the fit-to-GPS term, which the prior never touches.
        The one range where the two part is 0.6–0.8, and there the prior-free
        model is exactly right (it states 69 %, and 69 % were driven) while
        the trained prior overstates (71 % stated, 59 % driven).

        **Probability belongs on segments, not whole routes.** A corridor
        with one obvious path still splits its probability across many routes
        that differ only in trivia, so no single route carries the corridor's
        full weight and route-level confidence reads low where the model is
        in fact certain. Summing to the segment cancels that split — and the
        segment is the unit a consumer asks about: *was the vehicle on this
        stretch of road?* To that question the pipeline gives a calibrated
        answer, and gives it without a learned prior.
    """))

    nb.cells = cells
    return nb


def main() -> int:
    for p in (REC_TRAINED, REC_NOPRIOR):
        if not p.exists():
            print(f"ERROR: {p} missing. Collect it with "
                  "scripts/demo_cluster_calibration_draft.py "
                  "(--indep-truth and --eval-mu0).", file=sys.stderr)
            return 1
    print(f"building {NB_PATH.name}")
    nb = build_notebook()
    print(f"executing {len(nb.cells)} cells…")
    NotebookClient(nb, timeout=300, kernel_name="python3",
                   resources={"metadata": {"path": str(REPO_ROOT)}}).execute()
    NB_PATH.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, str(NB_PATH))
    print(f"wrote {NB_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
