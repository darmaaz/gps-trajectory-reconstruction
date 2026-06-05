"""Construct COVERAGE_DRAFT.ipynb — capacity of the 120s candidate set.

Coverage is asked **model-independently**, against the raw GPS pings (NOT the
15s reconstruction, which carries spurs/disconnections of its own): for each
120s transition, can the candidate set construct the path the vehicle actually
drove — i.e. is there ONE candidate path within δ of ALL the raw pings in the
window?

Failures are decomposed honestly, none dropped from the denominator:
- data_gap                — the raw polyline teleports (no ground truth across it)
- capacity_ok             — a candidate threads every ping
- off_footprint_excursion — only DROPPED pings missed, and they fall off EVERY
                            candidate's footprint: the vehicle drove roads no
                            candidate covers — a non-simple excursion the
                            simple-path enumerator can't produce. Verified: relaxed
                            A* (slack 3.0/λ 0.1/3× budget) recovers 0/23 → genuinely
                            un-constructible. The ONLY bucket excluded as un-constructible.
- on_footprint_split      — missed dropped pings ARE within δ of some candidate
                            (road in the set), but no single path threads them all.
                            A coverage gap; counted against the score.
- generator_gap           — an OBSERVED/kept ping is unreachable by any candidate.
                            The genuine candidate-generation gap.

Built cell by cell via nbformat, executed via nbclient so outputs embed inline.

Run:
    python scripts/build_coverage_draft.py            # build + execute (N=50)
    python scripts/build_coverage_draft.py --n 3      # smaller batch
    python scripts/build_coverage_draft.py --no-exec  # build JSON only
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from textwrap import dedent

import nbformat as nbf
from nbclient import NotebookClient

REPO_ROOT = Path(__file__).resolve().parents[1]
NB_PATH = REPO_ROOT / "wip" / "COVERAGE_DRAFT.ipynb"


def _md(source: str) -> dict:
    return nbf.v4.new_markdown_cell(dedent(source).strip() + "\n")


def _code(source: str) -> dict:
    return nbf.v4.new_code_cell(dedent(source).strip() + "\n")


def build_notebook() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    cells: list[dict] = []

    cells.append(_md("""
        # Capacity of the 120s candidate set

        Draft for a section of `scripts/build_story_notebook.py`; shares its
        `make_config` / `downsample` helpers.

        **The question, model-independently:** for each 120s transition, can the
        candidate set construct the path the vehicle actually drove? Ground truth
        is the **raw GPS pings** — not the 15s reconstruction, which has spurs and
        disconnections of its own. A window is **capacity-covered** iff one
        candidate path passes within δ of *all* the raw pings in it.

        Failures are decomposed, **nothing dropped from the denominator** except
        what is provably un-constructible:

        - **data_gap** — the raw polyline teleports (no ground truth across it).
        - **capacity_ok** — a candidate threads every ping.
        - **off_footprint_excursion** — only *dropped* pings are missed, and they
          fall off *every* candidate's footprint. The vehicle drove roads no
          candidate covers — a non-simple excursion (parking loop, out-and-back)
          the simple-path enumerator structurally can't produce. **Verified:**
          relaxed A* (slack 3.0, λ 0.1, 3× budget) recovers **0 / 23** of these,
          so they're genuinely un-constructible — the *only* bucket excluded.
        - **on_footprint_split** — missed dropped pings *are* within δ of some
          candidate (the road is in the set), but no *single* path threads them
          all. A coverage/representation gap; **counted against** the score.
        - **generator_gap** — an *observed* (kept) ping is unreachable by any
          candidate. The genuine candidate-generation gap; counted against.

        Headline = **capacity given recoverable info** = capacity_ok /
        (capacity_ok + on_footprint_split + generator_gap).
    """))

    # ───────────────────────────────────────────────────────────── setup
    cells.append(_md("## Setup"))

    cells.append(_code("""
        from __future__ import annotations

        import os
        import pickle
        import sys
        from pathlib import Path

        REPO_ROOT = Path.cwd()
        if str(REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(REPO_ROOT))

        os.environ.setdefault("GPS_RECON_BBOX_LAT", "40.5,42.5")
        os.environ.setdefault("GPS_RECON_BBOX_LON", "-9.5,-7.0")

        import numpy as np
        import matplotlib.pyplot as plt
        import shapely
        from shapely.ops import unary_union

        %matplotlib inline
        plt.rcParams["figure.dpi"] = 110
        plt.rcParams["savefig.dpi"] = 110
        plt.rcParams["font.size"] = 10

        from src.api import reconstruct_trajectory
        from src.config import Config
        from src.data import default_mu
        from src.feeds import iter_porto_trips
        from src.geo import M_PER_DEG_LAT, equirectangular_distance_m
        from src.model import ExponentialFamilyTransition, FEATURE_DIM, StudentTEmission
        from src.network import load_osm_network

        from scripts._data_paths import osm_pbf_path, porto_csv_path

        print("imports ok; FEATURE_DIM =", FEATURE_DIM)
    """))

    cells.append(_code("""
        PBF = osm_pbf_path()
        CSV = porto_csv_path()
        OSM_CACHE = REPO_ROOT / "cache" / "pt_edges.parquet"
        DRAFT_CACHE = REPO_ROOT / "cache" / "coverage_draft_records.pkl"

        print("loading Portugal network…")
        network = load_osm_network(PBF, cache_path=OSM_CACHE)
        print(f"  network: {len(network)} edges")

        mu_trained = default_mu()
        print(f"trained mu loaded; ||mu|| = {np.linalg.norm(mu_trained):.3f}")
    """))

    cells.append(_code("""
        # Helpers — names match scripts/build_story_notebook.py.

        def make_config(network, mu=None, scale=10.0, offroad=True):
            # The 120s model under test: trained μ, scale=10, off-road on.
            if mu is None:
                mu = mu_trained
            return Config(
                emission=StudentTEmission(scale=scale, network=network),
                transition=ExponentialFamilyTransition(mu),
                enable_offroad_candidates=offroad,
            )

        def downsample(raw, stride):
            return raw[::stride]

        def edge_geom(lid):
            return network.geoms[network.edge_index_for_link(int(lid))]

        def pings_in_window(raw, t_lo, t_hi):
            return [(i, o) for i, o in enumerate(raw) if t_lo <= o.timestamp <= t_hi]

        def max_consecutive_jump_m(pings):
            mx = 0.0
            for (_, a), (_, b) in zip(pings, pings[1:]):
                d = np.hypot((a.lon - b.lon) * M_PER_DEG_LAT * np.cos(np.radians(a.lat)),
                             (a.lat - b.lat) * M_PER_DEG_LAT)
                mx = max(mx, float(d))
            return mx

        def ping_dists_m(geom, lons, lats):
            # Accurate per-ping distance (metres) to a candidate geometry, via
            # nearest point + equirectangular — not anisotropic degree distance.
            pts = shapely.points(np.column_stack([lons, lats]))
            near = shapely.get_point(shapely.shortest_line(pts, geom), 1)
            return equirectangular_distance_m(
                np.asarray(lats), np.asarray(lons),
                shapely.get_y(near), shapely.get_x(near))
    """))

    cells.append(_md("""
        ### Knobs

        `N_TRIPS` is read from `$COVERAGE_DRAFT_N`. δ (`PING_TOL_M`) is the
        ping-to-road tolerance — pings on the Porto feed sit ~dead on the road,
        so a small δ (~20 m for GPS noise / minor off-road) is appropriate; the
        records store per-ping distances so δ sweeps for free.
    """))

    cells.append(_code("""
        N_TRIPS    = int(os.environ.get("COVERAGE_DRAFT_N", "50"))
        MIN_PINGS  = 40
        MAX_PINGS  = 200
        STRIDE     = 8          # 15s → 120s

        TELEPORT_M = 250.0      # consecutive raw-ping jump above this = data gap
        PING_TOLS  = [10, 15, 20, 25, 30, 40]   # δ sweep (m)
        HEADLINE_TOL = 25       # headline δ (m)
        CAP_THRESH = 0.999      # capacity = a candidate threads ALL pings

        FORCE_RECOMPUTE = False
        print(f"N_TRIPS={N_TRIPS}  STRIDE={STRIDE}  HEADLINE_TOL={HEADLINE_TOL}m")
    """))

    # ───────────────────────────────────────────────────── build records
    cells.append(_md("""
        ---
        ## Build records

        One 120s reconstruction per trip (no 15s pass — the metric is
        ping-grounded). Per window we store the per-candidate × per-ping distance
        matrix (metres) and the kept/dropped mask, so capacity, the bucket
        decomposition, and the δ-sweep are all computed in analysis without
        re-reconstructing. Data-gap (teleport) windows are flagged.
    """))

    cells.append(_code("""
        def window_record(pm, pings):
            kept = np.array([gi % STRIDE == 0 for gi, _ in pings])
            lons = np.array([o.lon for _, o in pings])
            lats = np.array([o.lat for _, o in pings])
            dists = []
            for p in pm:
                if not p.edges:
                    continue
                geom = unary_union([edge_geom(e) for e in p.edges])
                dists.append(ping_dists_m(geom, lons, lats))
            return dict(kept=kept, n_pings=len(pings),
                        cand_dists=[np.asarray(d) for d in dists])

        def build_records():
            eval_cfg = make_config(network)
            recs = []
            n = 0
            for tid, raw in iter_porto_trips(CSV, min_pings=MIN_PINGS):
                if n >= N_TRIPS:
                    break
                if len(raw) > MAX_PINGS:
                    continue
                try:
                    segs = reconstruct_trajectory(downsample(raw, STRIDE), network, eval_cfg)
                    for seg in segs:
                        ts = seg.canonical_timestamps
                        for k, pm in enumerate(seg.path_marginals):
                            pw = pings_in_window(raw, ts[k], ts[k + 1])
                            if len(pw) < 2:
                                continue
                            if max_consecutive_jump_m(pw) > TELEPORT_M:
                                recs.append(dict(trip=tid, kind="data_gap",
                                                 n_pings=len(pw)))
                                continue
                            r = window_record(pm, pw)
                            r.update(trip=tid, kind="scoreable", n_cands=len(pm))
                            recs.append(r)
                except Exception as exc:
                    print(f"  trip {tid}: {type(exc).__name__}: {exc}")
                n += 1
                ns = sum(r["kind"] == "scoreable" for r in recs)
                print(f"  [{n}/{N_TRIPS}] {tid}: {ns} scoreable windows")
            return dict(records=recs,
                        meta=dict(N_TRIPS=N_TRIPS, MIN_PINGS=MIN_PINGS,
                                  MAX_PINGS=MAX_PINGS, STRIDE=STRIDE,
                                  TELEPORT_M=TELEPORT_M, REC_SCHEMA="v5-capacity"))

        need = dict(N_TRIPS=N_TRIPS, MIN_PINGS=MIN_PINGS, MAX_PINGS=MAX_PINGS,
                    STRIDE=STRIDE, TELEPORT_M=TELEPORT_M, REC_SCHEMA="v5-capacity")
        DATA = None
        if (not FORCE_RECOMPUTE) and DRAFT_CACHE.exists():
            with open(DRAFT_CACHE, "rb") as f:
                DATA = pickle.load(f)
            if DATA.get("meta") != need:
                print(f"cache meta != need; recomputing")
                DATA = None
        if DATA is None:
            DATA = build_records()
            with open(DRAFT_CACHE, "wb") as f:
                pickle.dump(DATA, f)
            print(f"wrote {DRAFT_CACHE.name}")
        else:
            print(f"loaded cached {DRAFT_CACHE.name}")
        print(f"  {len(DATA['records'])} window records")
    """))

    # ───────────────────────────────────────────────── capacity analysis
    cells.append(_md("""
        ---
        ## Capacity (headline)

        At tolerance δ, classify each scoreable window four ways: `capacity_ok`
        (one candidate threads all pings); `generator_gap` (an *observed* ping is
        unreachable); else the best candidate misses only *dropped* pings, split by
        candidate-footprint reachability into `off_footprint_excursion` (off every
        candidate → un-constructible) vs `on_footprint_split` (road in set, no
        single path). Headline excludes *only* the off-footprint excursions.
    """))

    cells.append(_code("""
        def classify(rec, tol):
            on = [d <= tol for d in rec["cand_dists"]]
            if not on:
                return "generator_gap", 0.0
            best = max(on, key=lambda m: m.sum())
            frac = float(best.mean())
            if frac >= CAP_THRESH:
                return "capacity_ok", frac
            kept = rec["kept"]
            # any candidate thread all the KEPT (observed) pings?
            kept_ok = max(float(np.mean(m[kept])) if kept.any() else 1.0
                          for m in on) >= CAP_THRESH
            if int(np.sum((~best) & kept)) > 0 or not kept_ok:
                return "generator_gap", frac          # an OBSERVED ping is unreachable
            # best misses only DROPPED pings → footprint reachability decides:
            footprint = np.min(np.vstack(rec["cand_dists"]), axis=0)
            missed_dropped = (~best) & (~kept)
            if np.any(footprint[missed_dropped] > tol):
                return "off_footprint_excursion", frac  # off EVERY candidate → un-constructible
            return "on_footprint_split", frac           # road IS in set, no single path threads it

        scoreable = [r for r in DATA["records"] if r["kind"] == "scoreable"]
        gaps = [r for r in DATA["records"] if r["kind"] == "data_gap"]
        d = HEADLINE_TOL
        buckets = {}
        gen_cases = []
        for r in scoreable:
            cat, frac = classify(r, d)
            buckets[cat] = buckets.get(cat, 0) + 1
            if cat == "generator_gap":
                gen_cases.append((r["trip"], round(frac, 2), r["n_cands"]))

        n_s = len(scoreable)
        ok = buckets.get("capacity_ok", 0)
        off = buckets.get("off_footprint_excursion", 0)
        split = buckets.get("on_footprint_split", 0)
        gen = buckets.get("generator_gap", 0)
        print(f"δ={d} m   scoreable windows={n_s}   data-gap windows={len(gaps)} "
              f"({len(gaps)/(n_s+len(gaps)):.0%} of all)\\n")
        print(f"  capacity_ok              {ok:>4}  ({ok/n_s:.1%})")
        print(f"  off_footprint_excursion  {off:>4}  ({off/n_s:.1%})  "
              f"← dropped pings off EVERY candidate → un-constructible")
        print(f"  on_footprint_split       {split:>4}  ({split/n_s:.1%})  "
              f"← road IS in the set, no single path threads it")
        print(f"  generator_gap            {gen:>4}  ({gen/n_s:.1%})  "
              f"← an OBSERVED ping unreachable")
        recov = ok + split + gen
        print(f"\\nCapacity given recoverable info: {ok/recov:.1%} "
              f"({ok}/{recov}; excludes ONLY off-footprint excursions + data gaps).")
        print("  off-footprint excursions verified non-enumerable under relaxed A* "
              "(slack=3.0, λ=0.1, max_cands=300): 0/23 rescued —\\n"
              "  wip/diag_15s_truth/diag_offfootprint_relaxed.py")
    """))

    cells.append(_code("""
        # δ sweep — the four buckets as functions of tolerance.
        print(f"  {'δ (m)':>6} {'capacity':>9} {'off_excur':>10} {'split':>7} {'gen_gap':>8}")
        rows = {}
        for tol in PING_TOLS:
            b = {}
            for r in scoreable:
                cat, _ = classify(r, tol)
                b[cat] = b.get(cat, 0) + 1
            rows[tol] = b
            print(f"  {tol:>6} {b.get('capacity_ok',0)/n_s:>8.1%} "
                  f"{b.get('off_footprint_excursion',0)/n_s:>9.1%} "
                  f"{b.get('on_footprint_split',0)/n_s:>6.1%} "
                  f"{b.get('generator_gap',0)/n_s:>7.1%}")

        fig, ax = plt.subplots(figsize=(7, 4.2))
        ax.plot(PING_TOLS, [rows[t].get('capacity_ok',0)/n_s for t in PING_TOLS],
                marker="o", label="capacity_ok")
        ax.plot(PING_TOLS, [rows[t].get('off_footprint_excursion',0)/n_s for t in PING_TOLS],
                marker="^", color="#7f7f7f", label="off-footprint excursion")
        ax.plot(PING_TOLS, [(rows[t].get('on_footprint_split',0)
                             + rows[t].get('generator_gap',0))/n_s for t in PING_TOLS],
                marker="s", color="#d62728", label="recoverable gap (split+gen)")
        ax.axvline(HEADLINE_TOL, color="gray", ls=":", lw=0.8)
        ax.set_xlabel("δ (m)"); ax.set_ylabel("fraction of scoreable windows")
        ax.set_title("Capacity vs ping tolerance"); ax.grid(alpha=0.3); ax.legend()
        fig.tight_layout(); plt.show()
    """))

    cells.append(_code("""
        # The genuine generation gaps — the only windows worth chasing.
        print(f"generator_gap windows at δ={HEADLINE_TOL} m "
              f"({len(gen_cases)} of {n_s} = {len(gen_cases)/n_s:.1%}):")
        print(f"  {'trip':>22} {'best_ping_frac':>14} {'n_cands':>8}")
        for tid, frac, nc in sorted(gen_cases, key=lambda x: x[1])[:15]:
            print(f"  {tid:>22} {frac:>14.2f} {nc:>8}")
    """))

    cells.append(_md("""
        ---
        ## Porting notes

        Fold into `scripts/build_story_notebook.py`: `make_config` / `downsample`
        already exist; bring `ping_dists_m` and the record-build cell. The metric
        is **ping-grounded** — no 15s reconstruction, so it sidesteps the 15s-truth
        spurs/disconnections entirely. Headline = capacity at δ={HEADLINE_TOL} m
        with the four-way decomposition (nothing dropped from the denominator);
        the generation gap is the small residual. `cache/coverage_draft_records.pkl`
        (`v5-capacity`) keeps re-runs cheap; δ sweeps without recompute.
    """))

    nb.cells = cells
    return nb


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--no-exec", action="store_true")
    args = parser.parse_args()

    if args.n is not None:
        os.environ["COVERAGE_DRAFT_N"] = str(args.n)

    print(f"building notebook at {NB_PATH}")
    nb = build_notebook()

    if not args.no_exec:
        n = os.environ.get("COVERAGE_DRAFT_N", "50")
        print(f"executing {len(nb.cells)} cells (N_TRIPS={n})…")
        client = NotebookClient(
            nb, timeout=2400, kernel_name="python3",
            resources={"metadata": {"path": str(REPO_ROOT)}},
        )
        client.execute()

    NB_PATH.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, str(NB_PATH))
    print(f"wrote {NB_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
