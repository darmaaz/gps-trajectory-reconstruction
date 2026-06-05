"""Construct STORY.ipynb — the comprehensive demonstration notebook.

The notebook is built cell by cell via `nbformat`, then executed via
`nbclient` so outputs (figures, tables, stdout) are embedded inline.
The result is written to wip/ (gitignored while the notebook is in
progress); running this script regenerates it from scratch.

Run:
    /path/to/python scripts/build_story_notebook.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from textwrap import dedent

import nbformat as nbf
from nbclient import NotebookClient

REPO_ROOT = Path(__file__).resolve().parents[1]
NB_PATH = REPO_ROOT / "wip" / "STORY.ipynb"


def _md(source: str) -> dict:
    return nbf.v4.new_markdown_cell(dedent(source).strip() + "\n")


def _code(source: str) -> dict:
    return nbf.v4.new_code_cell(dedent(source).strip() + "\n")


def build_notebook() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.13"},
    }
    cells: list[dict] = []

    # ============================================================== TITLE
    cells.append(_md("""
        # Sparse-GPS Trajectory Reconstruction — A Walkthrough

        A vehicle traverses a road network. We observe sparse, noisy GPS
        pings — say, one every two minutes. Conventional map-matching
        commits to a single most-likely path and discards the rest.
        Under sparse sampling that's often wrong: the same two
        observations are consistent with many real-world driving stories
        — a fast direct route, a slow scenic route, a quick run followed
        by a long wait at the destination.

        This walkthrough builds the case for a different output: a
        **calibrated set of paths**, each annotated with the implied
        dwell time, so downstream consumers can either pick the educated
        guess or reason about the full range of possibilities.

        Data: the [Porto Kaggle taxi dataset](https://www.kaggle.com/c/pkdd-15-predict-taxi-service-trajectory-i)
        at native 15-second sampling, paired with the [OSM Portugal extract](https://download.geofabrik.de/europe/portugal.html).
        We treat the native 15 s pings as ground truth and downsample the
        same trips to 120 s and 300 s to simulate sparser fleet feeds.

        **Sections:**
        1. Sampling regimes — what 15 s, 60 s, 120 s actually look like
        2. The path posterior — enumerated routes with calibrated weights
        3. Confirmed dwell vs inferred dwell
        4. Position at intermediate times — the dwell-allocation choice
           and the oracle floor
        5. Posterior entropy — where the model is confident
        6. Off-road candidates — when routing can't represent the maneuver
        7. 60 s vs 120 s — how the reconstruction degrades with sparsity
        8. Take-aways

        For methodology, see `OVERVIEW.md`. For the module-level spec,
        see `SPEC.md`.
    """))

    # ============================================================== SETUP
    cells.append(_md("## Setup"))

    cells.append(_code("""
        from __future__ import annotations

        import os
        import sys
        from datetime import timedelta
        from pathlib import Path

        REPO_ROOT = Path.cwd()
        if str(REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(REPO_ROOT))

        os.environ.setdefault("GPS_RECON_BBOX_LAT", "40.5,42.5")
        os.environ.setdefault("GPS_RECON_BBOX_LON", "-9.5,-7.0")

        import numpy as np
        import matplotlib.pyplot as plt

        %matplotlib inline
        plt.rcParams["figure.dpi"] = 110
        plt.rcParams["savefig.dpi"] = 110
        plt.rcParams["font.size"] = 10

        from src.api import position_at_time, reconstruct_trajectory
        from src.api.interpolation import interpolate_along_path, position_in_transition
        from src.config import Config
        from src.data import default_mu
        from src.feeds import iter_porto_trips
        from src.geo import haversine_m
        from src.model import (
            ExponentialFamilyTransition, FEATURE_DIM, StudentTEmission,
        )
        from src.network import load_osm_network
        from src.preprocessing import clean, drop_kinematic_spikes
        from src.viz.primitives import (
            CANDIDATE_COLORS, COLOR_TRUTH, COLOR_OBS_SPARSE, COLOR_MLE_PATH,
            COLOR_PRED_DWELL, COLOR_PRED_NODWELL, COLOR_NETWORK,
            clean_map_axes, draw_network_backdrop, draw_path_edges,
        )

        from scripts._data_paths import osm_pbf_path, porto_csv_path

        print("imports ok; FEATURE_DIM =", FEATURE_DIM)
    """))

    cells.append(_code("""
        PBF = osm_pbf_path()
        CSV = porto_csv_path()
        OSM_CACHE = REPO_ROOT / "cache" / "pt_edges.parquet"

        SHORT_TRIP  = "1372637091620000337"   # ~7 min   29 pings
        MEDIUM_TRIP = "1372636951620000320"   # ~16 min  65 pings
        LONG_TRIP   = "1372639536620000570"   # ~36 min  145 pings

        print("loading Portugal network…")
        network = load_osm_network(PBF, cache_path=OSM_CACHE)
        print(f"  network: {len(network)} edges")

        mu_trained = default_mu()
        print(f"trained mu (FEATURE_DIM={FEATURE_DIM}) loaded; "
              f"||mu||={np.linalg.norm(mu_trained):.3f}")
    """))

    cells.append(_code("""
        def make_config(network, mu=None, scale=10.0, offroad=True):
            if mu is None:
                mu = mu_trained
            # path_budget_slack defaults to 1.2 in src/config.py (matches
            # the slack used for the labelled cache + the trained μ).
            # enable_offroad_candidates is on throughout this notebook so
            # near-stationary / one-way-pair maneuvers don't hallucinate
            # block-loops (see the dedicated off-road section). It is
            # purely additive and gated conservatively — net-neutral on
            # the trips that don't need it. Pass offroad=False to compare.
            return Config(
                emission=StudentTEmission(scale=scale, network=network),
                transition=ExponentialFamilyTransition(mu),
                enable_offroad_candidates=offroad,
            )

        def load_trip(trip_id, min_pings=20):
            for tid, raw in iter_porto_trips(CSV, min_pings=min_pings):
                if tid == trip_id:
                    return raw
            raise RuntimeError(f"trip {trip_id} not found in CSV")

        def downsample(raw, stride):
            return raw[::stride]
    """))

    # ========================================================== SECTION 1
    cells.append(_md("""
        ---

        ## 1. Sampling regimes

        Porto data is logged every 15 s — dense enough that we treat it as
        ground truth. Operational fleets report far less often. We
        downsample the same trip to **60 s** and **120 s**: both are sparse
        relative to 15 s, and the whole walkthrough asks how much of the
        native trajectory the pipeline can recover from each.

        Below: the same trip, sampled three ways.
    """))

    cells.append(_code("""
        long_raw = load_trip(LONG_TRIP, min_pings=50)
        print(f"LONG trip {LONG_TRIP}: {len(long_raw)} raw 15 s pings, "
              f"duration {(long_raw[-1].timestamp - long_raw[0].timestamp).total_seconds()/60:.1f} min")

        raw_15  = long_raw
        raw_60  = downsample(long_raw, 4)    #  15 s × 4 = 60 s
        raw_120 = downsample(long_raw, 8)    #  15 s × 8 = 120 s

        print(f"  15 s sampling: {len(raw_15)} pings")
        print(f"  60 s sampling: {len(raw_60)} pings")
        print(f" 120 s sampling: {len(raw_120)} pings")
    """))

    cells.append(_code("""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        for ax, (raw, label, color) in zip(axes, [
            (raw_15,  f"15 s — truth ({len(raw_15)} pings)",  COLOR_TRUTH),
            (raw_60,  f"60 s — sparse ({len(raw_60)} pings)", "#0e7490"),
            (raw_120, f"120 s — sparser ({len(raw_120)} pings)", COLOR_OBS_SPARSE),
        ]):
            lats = [o.lat for o in raw]
            lons = [o.lon for o in raw]
            bbox = clean_map_axes(ax, lats=lats, lons=lons, pad_frac=0.08)
            draw_network_backdrop(ax, network, bbox)
            ax.scatter(lons, lats, c=color, s=18, zorder=3, edgecolor="white",
                       linewidth=0.5)
            # Faint line connecting in temporal order to show the path shape.
            ax.plot(lons, lats, c=color, alpha=0.25, lw=1.0, zorder=2)
            ax.set_title(label, fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])

        fig.suptitle(
            f"LONG trip {LONG_TRIP}: raw observations at three sampling rates\\n"
            f"dots = GPS pings; faint line = straight segments between "
            f"consecutive pings (NOT a route — just the raw data)",
            fontsize=11, y=1.04,
        )
        plt.tight_layout()
        plt.show()
    """))

    cells.append(_md("""
        At 15 s the dense pings pin almost every transition through the
        emission factor alone. At 60 s the overall route shape is intact
        but the per-intersection detail starts to blur — the model has to
        reason about what happened between pings. At 120 s the gaps are
        wide enough that genuinely different routes become consistent with
        the same two endpoints, which is where the path posterior, the
        dwell accounting, and the off-road handling all earn their keep.
        The rest of the walkthrough works mostly at 120 s — the harder of
        the two realistic rates — and §7 returns to compare 60 s against
        120 s head-to-head.
    """))

    cells.append(_md("""
        ### Preview: what the pipeline reconstructs

        The straight segments above are *not* what the pipeline produces —
        they're just the raw data. Below is the same trip at the same three
        rates, now connected by the pipeline's **reconstructed most-likely
        path** (Viterbi over the CRF, snapped to real roads). This is the
        end product; the rest of the notebook unpacks how it's built and
        how much to trust it at each rate.
    """))

    cells.append(_code("""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        for ax, (raw, label) in zip(axes, [
            (raw_15,  f"15 s — reconstructed ({len(raw_15)} pings)"),
            (raw_60,  f"60 s — reconstructed ({len(raw_60)} pings)"),
            (raw_120, f"120 s — reconstructed ({len(raw_120)} pings)"),
        ]):
            lats = [o.lat for o in raw]
            lons = [o.lon for o in raw]
            bbox = clean_map_axes(ax, lats=lats, lons=lons, pad_frac=0.08)
            draw_network_backdrop(ax, network, bbox)
            segs = reconstruct_trajectory(raw, network, make_config(network))
            # Draw each segment's Viterbi MLE path edges (routed geometry).
            for seg in segs:
                for step in seg.most_likely:
                    if hasattr(step, "edges"):
                        for link_id in step.edges:
                            try:
                                idx = network.edge_index_for_link(link_id)
                            except KeyError:
                                continue
                            xs, ys = zip(*list(network.geoms[idx].coords))
                            ax.plot(xs, ys, c=COLOR_MLE_PATH, lw=2.2,
                                    alpha=0.8, zorder=2)
            ax.scatter(lons, lats, c=COLOR_OBS_SPARSE, s=18, zorder=4,
                       edgecolor="white", linewidth=0.5)
            ax.set_title(label, fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])

        fig.suptitle(
            "Same trip, reconstructed: Viterbi most-likely path on the road "
            "network\\n(orange line = reconstructed route; dots = the "
            "observations the model saw)",
            fontsize=11, y=1.04,
        )
        plt.tight_layout()
        plt.show()
    """))

    cells.append(_md("""
        Even at 120 s — eight-fold sparser than native — the reconstructed
        route recovers the trip's real shape on the road network, not a
        chord across blocks. Where the rates differ is in *confidence* and
        in the handful of ambiguous transitions; that's what the following
        sections quantify.
    """))

    # ========================================================== SECTION 2
    cells.append(_md("""
        ---

        ## 2. The path posterior

        Between two consecutive 120 s observations the vehicle could have
        taken several genuinely different routes. The pipeline enumerates
        them with penalty-diversified A\\* — each accepted path's edges are
        surcharged so the next search finds a structurally distinct
        alternative, not a near-copy.

        The CRF weighs each enumerated path by `exp(μᵀϕ(p))` × the
        bracketing emissions, normalised over the candidate set — a
        calibrated **path posterior** per transition. Below we pick a
        transition that actually offers distinct routes (a real fork in
        the network, not one corridor with cosmetic variants) and show how
        the posterior splits its mass across them.
    """))

    cells.append(_code("""
        from src.preprocessing import collapse_by_uniqueness as _collapse

        def _edge_jaccard_distance(a, b):
            sa, sb = set(a.edges), set(b.edges)
            u = sa | sb
            return 1.0 - (len(sa & sb) / len(u)) if u else 0.0

        def _route_diversity(marg, top_n=4):
            # Mean pairwise edge-set Jaccard distance among the top routed
            # candidates. High → the posterior is spread over structurally
            # different routes (a genuine fork), not offset variants of one
            # corridor.
            routed = [p for p, _ in sorted(marg.items(), key=lambda kv: kv[1],
                                           reverse=True)
                      if not getattr(p, "is_off_road", False)]
            top = routed[:top_n]
            if len(top) < 2:
                return 0.0, top
            dists = [
                _edge_jaccard_distance(top[i], top[j])
                for i in range(len(top)) for j in range(i + 1, len(top))
            ]
            return float(np.mean(dists)), top

        # Scan the canonical trips for the transition whose top candidates
        # are the most structurally diverse, with the posterior meaningfully
        # split (top weight < 0.9 so the alternatives carry real mass).
        config = make_config(network)
        best = None   # (diversity, trip_raw_120, seg, k, marg)
        for tid in (MEDIUM_TRIP, LONG_TRIP, SHORT_TRIP):
            traw = load_trip(tid, min_pings=20)
            t120 = downsample(traw, 8)
            for s in reconstruct_trajectory(t120, network, config):
                for k in range(len(s.path_marginals)):
                    marg_k = s.path_marginals[k]
                    if len(marg_k) < 3:
                        continue
                    top_w = max(marg_k.values())
                    if top_w >= 0.9:
                        continue
                    div, _ = _route_diversity(marg_k)
                    if best is None or div > best[0]:
                        best = (div, t120, s, k, marg_k)

        _, chosen_120, seg, k_pick, marg = best
        print(f"chosen transition: {len(marg)} candidate paths, "
              f"route-diversity {best[0]:.2f} (mean pairwise edge Jaccard distance)")
    """))

    cells.append(_code("""
        # Build a ranked table of candidates.
        ranked = sorted(marg.items(), key=lambda kv: kv[1], reverse=True)
        print(f"\\nTop {min(len(ranked), 6)} of {len(ranked)} candidate paths "
              f"(transition {k_pick}, time_budget = {ranked[0][0].time_budget:.0f} s):")
        print(f"  {'rank':>4}  {'weight':>7}  {'length_m':>9}  {'travel_s':>9}  "
              f"{'dwell_s':>8}  {'edges':>6}")
        for r, (path, w) in enumerate(ranked[:6]):
            print(f"  {r:>4d}  {w:>7.3f}  {path.length_meters:>9.0f}  "
                  f"{path.expected_travel_time:>9.1f}  {path.inferred_dwell:>8.1f}  "
                  f"{len(path.edges):>6d}")
    """))

    cells.append(_code("""
        # Two-part rendering for clarity:
        #   (top)    weight bar chart for the top-N candidates
        #   (bottom) small-multiples: one map per candidate showing its geometry
        #
        # Overlaying many candidates on a single map is illegible because they
        # share most edges; splitting them apart makes the structural
        # differences readable.

        TOP_N = min(5, len(ranked))
        top_paths = ranked[:TOP_N]

        # Geographic span (union of all top candidates' edges).
        all_lats, all_lons = [], []
        for path, _ in top_paths:
            for link_id in path.edges:
                idx = network.edge_index_for_link(link_id)
                for x, y in network.geoms[idx].coords:
                    all_lons.append(x); all_lats.append(y)

        # Bracketing observation positions for endpoint markers.
        obs_pre_t  = seg.canonical_timestamps[k_pick]
        obs_post_t = seg.canonical_timestamps[k_pick + 1]
        collapsed_chosen = _collapse(clean(chosen_120))
        pre_pt = next((o.lat, o.lon) for o in collapsed_chosen if o.t_first == obs_pre_t)
        post_pt = next((o.lat, o.lon) for o in collapsed_chosen if o.t_first == obs_post_t)

        fig = plt.figure(figsize=(15, 7.5))
        gs = fig.add_gridspec(2, TOP_N, height_ratios=[1, 3.5], hspace=0.35)

        # Top row: weight distribution as a bar chart spanning the whole width.
        ax_bar = fig.add_subplot(gs[0, :])
        weights = [w for _, w in top_paths]
        labels = [f"#{i}" for i in range(TOP_N)]
        colors = [CANDIDATE_COLORS[i % len(CANDIDATE_COLORS)] for i in range(TOP_N)]
        ax_bar.bar(labels, weights, color=colors, edgecolor="white", linewidth=1.5)
        ax_bar.set_ylabel("posterior weight")
        ax_bar.set_title(
            f"Posterior over top {TOP_N} of {len(ranked)} candidate paths "
            f"(transition {k_pick}, budget = {top_paths[0][0].time_budget:.0f} s)",
            loc="left",
        )
        ax_bar.set_ylim(0, max(weights) * 1.45)
        for i, (path, w) in enumerate(top_paths):
            ax_bar.text(
                i, w + max(weights) * 0.03,
                f"{w:.3f}\\nlen={path.length_meters:.0f}m\\n"
                f"travel={path.expected_travel_time:.0f}s\\n"
                f"dwell={path.inferred_dwell:.0f}s",
                ha="center", va="bottom", fontsize=8,
            )

        # Bottom row: one panel per candidate showing just its edges.
        for i, (path, w) in enumerate(top_paths):
            ax_map = fig.add_subplot(gs[1, i])
            bbox = clean_map_axes(ax_map, lats=all_lats, lons=all_lons, pad_frac=0.15)
            draw_network_backdrop(ax_map, network, bbox)
            color = CANDIDATE_COLORS[i % len(CANDIDATE_COLORS)]
            for link_id in path.edges:
                idx = network.edge_index_for_link(link_id)
                geom = network.geoms[idx]
                xs, ys = zip(*list(geom.coords))
                ax_map.plot(xs, ys, c=color, lw=3.0, alpha=0.95, zorder=3)
            ax_map.scatter(
                [pre_pt[1], post_pt[1]], [pre_pt[0], post_pt[0]],
                c=COLOR_OBS_SPARSE, s=90, marker="o",
                zorder=5, edgecolor="white", linewidth=1.3,
            )
            ax_map.set_title(f"#{i}  weight = {w:.3f}", fontsize=10)
            ax_map.set_xticks([]); ax_map.set_yticks([])

        plt.show()
        print(f"top weight: {ranked[0][1]:.3f};  "
              f"top-{TOP_N} sum: {sum(w for _, w in ranked[:TOP_N]):.3f};  "
              f"total candidates: {len(ranked)}")
    """))

    cells.append(_md("""
        The bar chart shows the posterior weight distribution at a
        glance — clearly opinionated when one bar dominates, more
        spread-out when several bars are similar. Each panel below
        shows that candidate's actual edge geometry in isolation, so
        structural differences (which intersection turn, which shortcut)
        are readable rather than obscured by overlap.

        The `inferred_dwell` annotation on each bar shows the dwell
        story each path commits to: a path with shorter
        `expected_travel_time` implies more dwell at the origin to fill
        the same budget; a longer path implies less.
    """))

    # ========================================================== SECTION 3
    cells.append(_md("""
        ---

        ## 3. Confirmed dwell, and a sparse-sampling caveat

        A transition's wall-clock interval has two components:

        - **Confirmed dwell** (`t_last − t_first` at the source
          observation, on non-stale runs): the vehicle was *observed*
          to be at this location across multiple raw pings before
          departure.
        - **Transit budget** (the rest): the unknown-allocation window
          during which the vehicle drove to the next observation. Each
          candidate path splits this further into `expected_travel_time`
          and `inferred_dwell`.

        The pipeline tracks both. `confirmed_dwell` is removed from the
        time budget before path enumeration, so candidate paths compete
        only over the actual transit window. The inferred dwell is what's
        left over for each path within that window.

        **One caveat first.** "Confirmed dwell" is a fact about the
        *collapse step*, not necessarily about the vehicle. The collapse
        merges consecutive observations within a small radius (default
        ε=5 m). At 120 s sampling, "same position twice 120 s apart" is
        indistinguishable from "stationary the whole time" — even when
        the vehicle drove away and came back in between. We'll see one
        such case below, then move to a transition where the dwell
        accounting is unambiguous.
    """))

    cells.append(_code("""
        # Pick a transition on the LONG trip that actually has a confirmed dwell.
        long_120 = downsample(long_raw, 8)
        seg_long_list = reconstruct_trajectory(long_120, network, make_config(network))
        # Find the segment+transition with the largest confirmed_dwell.
        best_seg, best_k, best_cd = None, None, -1.0
        for s in seg_long_list:
            for k, cd in enumerate(s.confirmed_dwell):
                if cd > best_cd:
                    best_seg, best_k, best_cd = s, k, cd
        cd = best_seg.confirmed_dwell[best_k]
        gap = (best_seg.canonical_timestamps[best_k + 1]
               - best_seg.canonical_timestamps[best_k]).total_seconds()
        transit_budget = gap - cd

        print(f"LONG trip — segment with {len(best_seg.canonical_timestamps)} obs")
        print(f"transition {best_k}:")
        print(f"  wall-clock gap   = {gap:7.1f} s")
        print(f"  confirmed dwell  = {cd:7.1f} s  (data fact)")
        print(f"  transit budget   = {transit_budget:7.1f} s")

        if best_seg.path_marginals[best_k]:
            top_path, top_w = max(
                best_seg.path_marginals[best_k].items(), key=lambda kv: kv[1],
            )
            print(f"  top path:        weight {top_w:.3f}, "
                  f"travel {top_path.expected_travel_time:.1f} s, "
                  f"inferred dwell {top_path.inferred_dwell:.1f} s")
    """))

    cells.append(_code("""
        # Visualise the budget split for this transition.
        fig, ax = plt.subplots(figsize=(11, 2.6))
        cd_val = cd
        tb_val = transit_budget
        # Within transit_budget, the top path attributes some of it to inferred dwell.
        top_path = max(best_seg.path_marginals[best_k].items(),
                       key=lambda kv: kv[1])[0]
        inferred = top_path.inferred_dwell
        travel = top_path.expected_travel_time

        bars = [
            (0,                        cd_val,        "confirmed dwell\\n(observed)",
             "#94a3b8"),
            (cd_val,                   inferred,      "inferred dwell\\n(top path)",
             COLOR_PRED_DWELL),
            (cd_val + inferred,        travel,        "expected travel\\n(top path)",
             COLOR_MLE_PATH),
        ]
        for x, w, label, color in bars:
            ax.barh(0, w, left=x, color=color, edgecolor="white", linewidth=1.2)
            ax.text(x + w/2, 0, f"{w:.1f}s\\n{label}",
                    ha="center", va="center", fontsize=9, color="white")

        ax.set_yticks([])
        ax.set_xlabel("seconds since obs k's t_first")
        ax.set_xlim(0, cd_val + inferred + travel + 5)
        ax.set_title(
            f"LONG trip transition {best_k}: budget split (front-loaded convention)",
            loc="left",
        )
        plt.tight_layout()
        plt.show()
    """))

    cells.append(_md("""
        That's what the model thinks happened. Now compare against what
        the native 15 s pings show.

        For this transition, three raw 120 s observations are relevant:
        one at `obs k.t_first` (t=0 s in the chart below), one at
        t=120 s (the second 120 s ping that landed in the same place,
        triggering the collapse), and one at t=240 s (the next
        un-collapsed observation = obs k+1). The collapse step merged
        the first two into a single `CollapsedObservation` with
        `t_last − t_first = 120 s` — that's the model's claim of a
        confirmed dwell.
    """))

    cells.append(_code("""
        # Sparse-vs-dense temporal view. The two 120 s observations both
        # land at position A, so the collapse step reports a 120 s
        # confirmed dwell. The 15 s truth pings tell a different story.

        truth_all = drop_kinematic_spikes(clean(long_raw))
        t_lo = best_seg.canonical_timestamps[best_k]
        t_hi = best_seg.canonical_timestamps[best_k + 1]
        t_last_k = best_seg.canonical_t_last[best_k]
        truth_in = [o for o in truth_all if t_lo <= o.timestamp <= t_hi]

        # Distance from the obs-k position (use first truth ping in the
        # window as the anchor — it lands at obs k's collapsed position).
        ref_lat, ref_lon = truth_in[0].lat, truth_in[0].lon
        offsets = [(o.timestamp - t_lo).total_seconds() for o in truth_in]
        dists   = [haversine_m(ref_lat, ref_lon, o.lat, o.lon) for o in truth_in]

        # The THREE 120 s observations that bracket this transition. The
        # first two were collapsed into a single observation because they
        # landed within ε of each other; the third is the next un-collapsed
        # observation. The chart shows all three so the collapse is visible.
        # Find the 120 s pings nearest to t_lo, t_lo+120s, t_hi by timestamp.
        long_120 = downsample(long_raw, 8)
        from datetime import timedelta
        targets = [t_lo, t_lo + timedelta(seconds=120), t_hi]
        obs_120 = []
        for target in targets:
            closest = min(long_120, key=lambda o: abs((o.timestamp - target).total_seconds()))
            obs_120.append(closest)
        obs_120_offsets = [
            (o.timestamp - t_lo).total_seconds() for o in obs_120
        ]
        obs_120_dists = [
            haversine_m(ref_lat, ref_lon, o.lat, o.lon) for o in obs_120
        ]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5))

        # Left: temporal plot.
        ax1.plot(offsets, dists, "-o", c=COLOR_TRUTH, lw=1.8, ms=6,
                 markeredgecolor="white", markeredgewidth=0.7,
                 label=f"15 s truth pings ({len(truth_in)} pings)")
        ax1.axvspan(0, (t_last_k - t_lo).total_seconds(),
                    alpha=0.18, color="#94a3b8",
                    label="confirmed-dwell window")
        ax1.scatter(obs_120_offsets, obs_120_dists,
                    c=COLOR_OBS_SPARSE, s=200, marker="o",
                    edgecolor="white", linewidth=1.8, zorder=5,
                    label="120 s observations")
        # Annotate the two pings that got collapsed and the next ping.
        ax1.annotate(
            "obs k.t_first\\n(collapse anchor)", xy=(obs_120_offsets[0], obs_120_dists[0]),
            xytext=(10, 25), textcoords="offset points", fontsize=8,
            arrowprops=dict(arrowstyle="-", color="gray", lw=0.5),
        )
        ax1.annotate(
            "obs k.t_last\\n(collapsed into obs k\\n— same position)",
            xy=(obs_120_offsets[1], obs_120_dists[1]),
            xytext=(10, 35), textcoords="offset points", fontsize=8,
            arrowprops=dict(arrowstyle="-", color="gray", lw=0.5),
        )
        ax1.annotate(
            "obs k+1\\n(next observation)",
            xy=(obs_120_offsets[2], obs_120_dists[2]),
            xytext=(-90, 10), textcoords="offset points", fontsize=8,
            arrowprops=dict(arrowstyle="-", color="gray", lw=0.5),
        )
        ax1.set_xlabel("seconds since obs k's t_first")
        ax1.set_ylabel("distance from obs k's position (m)")
        ax1.set_title("Temporal view: 15 s truth vs the three 120 s observations")
        ax1.legend(loc="upper left", fontsize=8)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(-15, (t_hi - t_lo).total_seconds() + 15)

        # Right: map view — all three 120 s observations + the 15 s trace.
        all_lats = [o.lat for o in truth_in]
        all_lons = [o.lon for o in truth_in]
        bbox = clean_map_axes(ax2, lats=all_lats, lons=all_lons, pad_frac=0.35)
        draw_network_backdrop(ax2, network, bbox)
        ax2.plot(all_lons, all_lats, "-o", c=COLOR_TRUTH, lw=1.5, ms=5,
                 markeredgecolor="white", markeredgewidth=0.5,
                 alpha=0.85, label="15 s truth trace", zorder=3)
        ax2.scatter([o.lon for o in obs_120], [o.lat for o in obs_120],
                    c=COLOR_OBS_SPARSE, s=200, marker="o",
                    edgecolor="white", linewidth=1.8, zorder=5,
                    label="120 s observations (3 pings)")
        ax2.set_title("Map view: same window")
        ax2.set_xticks([]); ax2.set_yticks([])
        ax2.legend(loc="lower right", fontsize=8)

        plt.tight_layout()
        plt.show()

        print(f"Three 120 s observations relevant to this transition:")
        for i, (off, dist) in enumerate(zip(obs_120_offsets, obs_120_dists)):
            tag = ["obs k (t_first)", "obs k (t_last, collapsed)", "obs k+1"][i]
            print(f"  t+{off:5.0f}s  dist_from_obs_k={dist:5.1f} m   [{tag}]")

        # Separate the two phases for an accurate narrative.
        cd_end_offset = (t_last_k - t_lo).total_seconds()
        dists_in_cd = [d for off, d in zip(offsets, dists) if off <= cd_end_offset]
        dists_in_transit = [
            d for off, d in zip(offsets, dists) if off > cd_end_offset
        ]
        print()
        print("Confirmed-dwell window (0 → 120 s):")
        print(f"  peak excursion = {max(dists_in_cd):.0f} m from obs k")
        print(f"  position at end of confirmed dwell ≈ {dists_in_cd[-1]:.0f} m")
        print("Transit window (120 → 240 s):")
        if dists_in_transit:
            print(f"  start at ≈ {dists_in_transit[0]:.0f} m, end at {dists_in_transit[-1]:.0f} m")
        else:
            print("  (no 15 s pings inside)")
    """))

    cells.append(_md("""
        The first two orange dots both sit at distance ≈ 0 m from obs k
        — that's why the collapse merged them into one observation with
        a 120 s confirmed dwell. The 15 s trace in between tells a
        different story: the vehicle drove ~70 m south, came back to
        obs k's location by t=120 s, then sat for ~75 s before driving
        away to ~100 m north by t=240 s (where obs k+1 sits).

        The model sees only the three orange dots. The blue out-and-back
        between t=15 s and t=120 s is invisible at 120 s sampling.

        **This is a fundamental limitation of sparse data, not a
        pipeline bug.** Any reconstruction algorithm that only sees
        these three 120 s observations has the same problem. The
        take-away is that `confirmed_dwell` should be read as "the
        model's best-effort inference from what it saw," not as ground
        truth.

        In §4 we'll pick a different transition — one where the model's
        view actually has transit — so the dwell-rule comparisons are
        meaningful.
    """))

    # ========================================================== SECTION 4
    cells.append(_md("""
        ---

        ## 4. Position at intermediate times — the dwell-allocation choice

        A reconstruction isn't just *which path* — downstream consumers
        want *where was the vehicle at time t*. Between two observations
        the budget splits into transit and dwell, and the posterior is
        silent on **when** within the window the dwell happens. Three
        conventions are equally consistent with the model:

        - **front**: dwell first, then transit.
        - **back**: transit first, then dwell at destination.
        - **spread**: constant speed across the whole window.

        We first show, on one transition, how differently the three place
        the vehicle; then (the oracle subsection) quantify across trips
        how much reconstruction error is due to picking the *wrong rule*
        rather than the wrong path.

        The picker scans Porto trips for a transition with a confident MLE
        (weight ≥ 0.5), real transit and dwell (travel > 0, inferred dwell
        ≥ 30 s, not overslacked), and selects the one where the three
        rules' predicted positions **diverge the most spatially** — a
        long-enough path where the choice is visible, not a stop-heavy
        short path where all three collapse to the same spot.

        For each native 15 s ping inside the window we predict its
        position under each rule and measure haversine error against the
        actual ping — so the "right" rule is the one whose error is
        lowest on this transition.
    """))

    cells.append(_code("""
        # Picker for the rule-comparison transition. Hard gates:
        #   - top-1 posterior weight >= 0.5 (model has a confident MLE)
        #   - top path expected_travel_time > 0 (real transit, not stay)
        #   - top path inferred_dwell >= 30 s (real dwell to allocate)
        #   - expected_travel_time <= time_budget (not overslacked)
        #   - not an off-road MLE (covered in §6)
        #   - rule divergence >= 80 m
        # then SELECT the max-divergence transition (clearest visual).
        #   - >= 5 truth pings inside the window
        #   - truth pings span >= 50 m end-to-end (visible motion)
        # Scan Porto trips until we find at least one match.
        CANONICAL_IDS = {SHORT_TRIP, MEDIUM_TRIP, LONG_TRIP}
        TRIP_SCAN_CAP = 80
        candidates_for_rule_demo = []
        trips_scanned = 0

        # Start with canonicals (preferred for narrative continuity), then
        # widen to other Porto trips if needed.
        canonical_pre = [
            ("LONG",   load_trip(LONG_TRIP,   min_pings=50)),
            ("MEDIUM", load_trip(MEDIUM_TRIP, min_pings=20)),
            ("SHORT",  load_trip(SHORT_TRIP,  min_pings=20)),
        ]

        def _rule_divergence(seg, k, pings):
            # Mean, over the interior truth pings, of the max pairwise
            # distance between the front / back / spread predicted
            # positions. This is exactly what we want to demonstrate
            # visually: a transition is a *good* front-vs-back-vs-spread
            # example when the three rules place the vehicle at clearly
            # different map points. Maximising raw inferred dwell does NOT
            # do this — a stop-heavy short path has huge dwell but tiny
            # spatial extent, so all three rules collapse to nearly the
            # same spot. Divergence captures spatial extent × dwell/transit
            # balance directly.
            divs = []
            for o in pings:
                preds = {}
                ok = True
                for r in ("front", "back", "spread"):
                    p = position_at_time([seg], o.timestamp, network, rule=r)
                    if p is None:
                        ok = False
                        break
                    preds[r] = p
                if not ok:
                    continue
                pts = list(preds.values())
                pair_max = max(
                    haversine_m(pts[i][0], pts[i][1], pts[j][0], pts[j][1])
                    for i in range(len(pts)) for j in range(i + 1, len(pts))
                )
                divs.append(pair_max)
            return float(np.mean(divs)) if divs else 0.0

        def evaluate_trip(label, trip_raw):
            try:
                trip_120 = downsample(trip_raw, 8)
                trip_segs = reconstruct_trajectory(trip_120, network, make_config(network))
                trip_truth = drop_kinematic_spikes(clean(trip_raw))
            except Exception:
                return []
            hits = []
            for s in trip_segs:
                for k in range(len(s.path_marginals)):
                    marg = s.path_marginals[k]
                    if not marg:
                        continue
                    top_path, top_w = max(marg.items(), key=lambda kv: kv[1])
                    if top_w < 0.5:
                        continue
                    if top_path.expected_travel_time <= 0:
                        continue
                    if top_path.expected_travel_time > top_path.time_budget:
                        continue
                    # Exclude off-road MLEs here: their straight-line
                    # endpoint-snapping makes the rules degenerate, and the
                    # off-road story has its own section. We want a genuine
                    # routed transit with real dwell.
                    if getattr(top_path, "is_off_road", False):
                        continue
                    d_p = top_path.time_budget - top_path.expected_travel_time
                    if d_p < 30:
                        continue
                    t_lo_c = s.canonical_timestamps[k]
                    t_hi_c = s.canonical_timestamps[k + 1]
                    pings = [o for o in trip_truth if t_lo_c < o.timestamp < t_hi_c]
                    if len(pings) < 5:
                        continue
                    end_to_end = haversine_m(
                        pings[0].lat, pings[0].lon,
                        pings[-1].lat, pings[-1].lon,
                    )
                    if end_to_end < 50:
                        continue
                    divergence = _rule_divergence(s, k, pings)
                    # Require the three rules to be meaningfully separated
                    # (≥ 80 m mean max-pairwise) so the demo reads clearly.
                    if divergence < 80.0:
                        continue
                    hits.append({
                        "trip_label": label,
                        "trip_raw": trip_raw,
                        "seg": s,
                        "k": k,
                        "top_w": top_w,
                        "top_path": top_path,
                        "n_pings": len(pings),
                        "end_to_end": end_to_end,
                        "divergence": divergence,
                    })
            return hits

        # Scan canonicals first, then widen — but keep scanning extra trips
        # even after the first hit so the divergence-maximising pick has a
        # real pool to choose from (the first match is rarely the clearest).
        for label, traw in canonical_pre:
            candidates_for_rule_demo += evaluate_trip(label, traw)
        trips_scanned += 3
        for tid, traw in iter_porto_trips(CSV, min_pings=40):
            if tid in CANONICAL_IDS:
                continue
            if len(traw) > 200:
                continue
            trips_scanned += 1
            candidates_for_rule_demo += evaluate_trip(f"Porto:{tid}", traw)
            # Keep going until we have a decent pool or hit the cap.
            if (len(candidates_for_rule_demo) >= 8
                    or trips_scanned >= TRIP_SCAN_CAP):
                break

        if not candidates_for_rule_demo:
            raise RuntimeError(
                f"no transition matching the rule-demo criteria found "
                f"after scanning {trips_scanned} trips",
            )

        # Pick the transition where the three dwell rules diverge the most
        # spatially — the clearest visual demonstration.
        chosen = max(candidates_for_rule_demo, key=lambda c: c["divergence"])
        comp_seg = chosen["seg"]
        comp_k = chosen["k"]
        comp_trip_raw = chosen["trip_raw"]
        comp_top_path = chosen["top_path"]
        t_lo = comp_seg.canonical_timestamps[comp_k]
        t_hi = comp_seg.canonical_timestamps[comp_k + 1]
        comp_d_p = comp_top_path.time_budget - comp_top_path.expected_travel_time
        print(f"scanned {trips_scanned} trips, {len(candidates_for_rule_demo)} matches")
        print(f"chose {chosen['trip_label']}, transition {comp_k}:")
        print(f"  top-1 weight    = {chosen['top_w']:.3f}")
        print(f"  travel time     = {comp_top_path.expected_travel_time:.1f} s")
        print(f"  inferred dwell  = {comp_d_p:.1f} s")
        print(f"  budget          = {comp_top_path.time_budget:.1f} s")
        print(f"  truth pings     = {chosen['n_pings']}")
        print(f"  end-to-end disp = {chosen['end_to_end']:.0f} m")
        print(f"  rule divergence = {chosen['divergence']:.0f} m  "
              f"(mean max-pairwise front/back/spread separation)")

        # Expose the chosen transition as best_seg/best_k/top_path for §4.
        best_seg = comp_seg
        best_k = comp_k
        top_path = comp_top_path
        long_raw_for_truth = comp_trip_raw
        truth_pings = drop_kinematic_spikes(clean(long_raw_for_truth))
        dropped_15 = [o for o in truth_pings if t_lo < o.timestamp < t_hi]
        print(f"\\ndropped 15 s pings inside this transition: {len(dropped_15)}")

        # Predict each dropped timestamp two ways.
        rows = []
        for o in dropped_15:
            pred_front = position_at_time([best_seg], o.timestamp, network, rule="front")
            pred_spread = position_at_time([best_seg], o.timestamp, network, rule="spread")
            err_front = haversine_m(o.lat, o.lon, *pred_front) if pred_front else None
            err_spread = haversine_m(o.lat, o.lon, *pred_spread) if pred_spread else None
            rows.append({
                "tau_s": (o.timestamp - t_lo).total_seconds(),
                "truth": (o.lat, o.lon),
                "pred_front": pred_front,
                "pred_spread": pred_spread,
                "err_front_m": err_front,
                "err_spread_m": err_spread,
            })

        # Summary line.
        ef = [r["err_front_m"] for r in rows if r["err_front_m"] is not None]
        es = [r["err_spread_m"] for r in rows if r["err_spread_m"] is not None]
        print(f"front-loaded  : median err = {np.median(ef):6.1f} m, mean = {np.mean(ef):6.1f} m")
        print(f"spread (no dw): median err = {np.median(es):6.1f} m, mean = {np.mean(es):6.1f} m")
    """))

    cells.append(_code("""
        # Visualise as map: truth pings + two prediction series + the MLE path edges.
        fig, ax = plt.subplots(figsize=(10, 8))
        # Bbox from truth pings.
        lats_pred = [r["truth"][0] for r in rows]
        lons_pred = [r["truth"][1] for r in rows]
        # Include MLE path geometry.
        top_path = max(best_seg.path_marginals[best_k].items(),
                       key=lambda kv: kv[1])[0]
        for link_id in top_path.edges:
            idx = network.edge_index_for_link(link_id)
            for x, y in network.geoms[idx].coords:
                lons_pred.append(x); lats_pred.append(y)

        bbox = clean_map_axes(ax, lats=lats_pred, lons=lons_pred, pad_frac=0.2)
        draw_network_backdrop(ax, network, bbox)
        # MLE path.
        for link_id in top_path.edges:
            idx = network.edge_index_for_link(link_id)
            geom = network.geoms[idx]
            xs, ys = zip(*list(geom.coords))
            ax.plot(xs, ys, c=COLOR_MLE_PATH, lw=2.5, alpha=0.7, zorder=2)

        # Truth pings + predictions.
        for r in rows:
            ty, tx = r["truth"]
            ax.scatter(tx, ty, c=COLOR_TRUTH, s=60, edgecolor="white",
                       linewidth=1.0, zorder=5)
            if r["pred_front"]:
                py, px = r["pred_front"]
                ax.scatter(px, py, c=COLOR_PRED_DWELL, s=45, marker="^",
                           edgecolor="white", linewidth=0.8, zorder=4)
                ax.plot([tx, px], [ty, py], c=COLOR_PRED_DWELL,
                        alpha=0.35, lw=0.8)
            if r["pred_spread"]:
                py, px = r["pred_spread"]
                ax.scatter(px, py, c=COLOR_PRED_NODWELL, s=45, marker="s",
                           edgecolor="white", linewidth=0.8, zorder=4)
                ax.plot([tx, px], [ty, py], c=COLOR_PRED_NODWELL,
                        alpha=0.35, lw=0.8)

        ax.set_xticks([]); ax.set_yticks([])
        # Legend.
        from matplotlib.lines import Line2D
        handles = [
            Line2D([0], [0], color=COLOR_MLE_PATH, lw=2.5, label="MLE path"),
            Line2D([0], [0], marker="o", color="white", markerfacecolor=COLOR_TRUTH,
                   markersize=8, label="truth ping", linestyle=""),
            Line2D([0], [0], marker="^", color="white",
                   markerfacecolor=COLOR_PRED_DWELL, markersize=8,
                   label="front-loaded prediction", linestyle=""),
            Line2D([0], [0], marker="s", color="white",
                   markerfacecolor=COLOR_PRED_NODWELL, markersize=8,
                   label="spread (const-speed) prediction", linestyle=""),
        ]
        ax.legend(handles=handles, loc="best", fontsize=9)
        ax.set_title(
            f"Predicted positions inside transition {best_k} "
            f"(dwell={best_seg.confirmed_dwell[best_k] + top_path.inferred_dwell:.0f} s)",
        )
        plt.show()
    """))

    cells.append(_md("""
        Where the vehicle is genuinely stationary (the dwell window),
        front-loaded predictions cluster at the path's origin while
        const-speed smears them along the path. If the vehicle truly did
        sit at the origin for the dwell window, front-loaded wins; if it
        moved continuously, spread wins. The 15 s truth pings disambiguate.
    """))

    # ----- §4 (cont.): front vs back vs spread on the chosen transition
    cells.append(_md("""
        ### Front vs back vs spread on the chosen transition

        Here's how the predicted position diverges per dropped 15 s ping
        under each rule. The three panels below should land the vehicle at
        visibly different points along the path — the dwell-allocation
        choice is a real modelling decision with metric consequences, not
        a cosmetic convention.
    """))

    cells.append(_code("""
        rule_rows = []
        for o in dropped_15:
            tau_s = (o.timestamp - t_lo).total_seconds()
            preds = {
                rule: position_at_time([best_seg], o.timestamp, network, rule=rule)
                for rule in ("front", "back", "spread")
            }
            errs = {
                rule: haversine_m(o.lat, o.lon, *p) if p else None
                for rule, p in preds.items()
            }
            rule_rows.append({"tau_s": tau_s, "truth": (o.lat, o.lon),
                              "preds": preds, "errs": errs})

        for rule in ("front", "back", "spread"):
            es = [r["errs"][rule] for r in rule_rows if r["errs"][rule] is not None]
            print(f"  rule={rule:>6s}: median err = {np.median(es):6.1f} m  "
                  f"mean = {np.mean(es):6.1f} m  max = {np.max(es):6.1f} m")

        # Per-rule "winner" count.
        winners = {"front": 0, "back": 0, "spread": 0, "tie": 0}
        for r in rule_rows:
            if any(v is None for v in r["errs"].values()):
                continue
            best = min(r["errs"], key=r["errs"].get)
            winners[best] += 1
        print(f"\\nPer-ping winner counts (n={sum(winners.values())}): {winners}")
    """))

    cells.append(_code("""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        rule_colors = {"front": COLOR_PRED_DWELL, "back": "#0891b2",
                       "spread": COLOR_PRED_NODWELL}

        for ax, rule in zip(axes, ("front", "back", "spread")):
            bbox = clean_map_axes(ax, lats=lats_pred, lons=lons_pred, pad_frac=0.2)
            draw_network_backdrop(ax, network, bbox)
            for link_id in top_path.edges:
                idx = network.edge_index_for_link(link_id)
                geom = network.geoms[idx]
                xs, ys = zip(*list(geom.coords))
                ax.plot(xs, ys, c=COLOR_MLE_PATH, lw=2.0, alpha=0.5, zorder=2)
            for r in rule_rows:
                ty, tx = r["truth"]
                ax.scatter(tx, ty, c=COLOR_TRUTH, s=45, edgecolor="white",
                           linewidth=0.7, zorder=5)
                p = r["preds"][rule]
                if p:
                    py, px = p
                    ax.scatter(px, py, c=rule_colors[rule], s=40, marker="X",
                               edgecolor="white", linewidth=0.7, zorder=4)
                    ax.plot([tx, px], [ty, py], c=rule_colors[rule],
                            alpha=0.45, lw=0.9)
            es = [r["errs"][rule] for r in rule_rows if r["errs"][rule] is not None]
            ax.set_title(f"rule = {rule}\\nmedian err = {np.median(es):.0f} m",
                         fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])

        fig.suptitle(
            f"Three dwell-allocation rules on the same transition "
            f"(truth pings = blue dots; prediction = colored X)", y=1.02)
        plt.tight_layout()
        plt.show()
    """))

    cells.append(_md("""
        The relative ordering of error medians tells you which rule
        matches the actual dwell timing on this transition. If front
        wins, the vehicle paused near the start of the budget; if back
        wins, near the end; if spread wins, it was continuously moving.
        Different transitions favour different rules — and the model
        can't know which without information it doesn't have. Front-loaded
        is the shipped convention; the others stay available.
    """))

    # ----- §4 (cont.): the oracle floor across trips
    cells.append(_md("""
        ### The oracle floor — how much error is the rule choice?

        A single transition is anecdotal. To see how much the rule choice
        *systematically* costs, we evaluate every truth ping on the three
        canonical trips under all three rules and compute, per ping, an
        **oracle** error `= min(front, back, spread)`. The oracle is the
        floor reachable by perfect per-transition rule choice, holding the
        path selection fixed. The gap between any single fixed rule and the
        oracle is the price of committing to one convention.
    """))

    cells.append(_code("""
        from collections import Counter
        RULES = ("front", "back", "spread")

        oracle_rows = []
        for label, tid in [("SHORT", SHORT_TRIP), ("MEDIUM", MEDIUM_TRIP),
                           ("LONG", LONG_TRIP)]:
            raw = load_trip(tid, min_pings=20)
            truth = drop_kinematic_spikes(clean(raw))
            segs = reconstruct_trajectory(downsample(raw, 8), network, make_config(network))
            per_rule = {r: [] for r in RULES}
            oracle = []
            winners = Counter()
            for o in truth:
                pe = {}
                for r in RULES:
                    p = position_at_time(segs, o.timestamp, network, rule=r)
                    if p is not None:
                        pe[r] = haversine_m(o.lat, o.lon, *p)
                if len(pe) != len(RULES):
                    continue
                for r, e in pe.items():
                    per_rule[r].append(e)
                best = min(pe, key=pe.get)
                oracle.append(pe[best]); winners[best] += 1
            oracle_rows.append({"trip": label, "front": per_rule["front"],
                                "back": per_rule["back"], "spread": per_rule["spread"],
                                "oracle": oracle, "winners": dict(winners)})

        print(f"{'trip':8s}  " + "  ".join(f"{c:>8s}" for c in ("front","back","spread","ORACLE")))
        print("-" * 50)
        for r in oracle_rows:
            meds = [float(np.median(r[c])) if r[c] else float('nan')
                    for c in ("front","back","spread","oracle")]
            print(f"{r['trip']:8s}  " + "  ".join(f"{m:>8.1f}" for m in meds))
        print()
        print(f"{'trip':8s}  per-ping rule winners (% of pings)")
        for r in oracle_rows:
            tot = sum(r["winners"].values())
            print(f"{r['trip']:8s}  " + "  ".join(
                f"{rule}={100*r['winners'].get(rule,0)/max(tot,1):>4.1f}%" for rule in RULES))
    """))

    cells.append(_code("""
        trips = ["SHORT", "MEDIUM", "LONG"]
        series = ["front", "back", "spread", "oracle"]
        colors = {"front": COLOR_PRED_DWELL, "back": "#0891b2",
                  "spread": COLOR_PRED_NODWELL, "oracle": "#16a34a"}
        fig, ax = plt.subplots(figsize=(9, 5))
        x = np.arange(len(trips)); width = 0.20
        for j, s in enumerate(series):
            meds = [float(np.median(next(r for r in oracle_rows if r["trip"] == t)[s]))
                    for t in trips]
            bars = ax.bar(x + (j - 1.5) * width, meds, width, label=s,
                          color=colors[s], edgecolor="white")
            for b, v in zip(bars, meds):
                ax.text(b.get_x() + b.get_width()/2, v + 4, f"{v:.0f}",
                        ha="center", fontsize=8)
        ax.set_xticks(x); ax.set_xticklabels(trips)
        ax.set_ylabel("median position error (m)")
        ax.set_title("Per-rule and oracle medians — gap to oracle = rule-choice cost")
        ax.legend(loc="upper right", fontsize=8)
        plt.tight_layout(); plt.show()
    """))

    cells.append(_md("""
        Two things to read off:

        - **The oracle is below every single rule.** No one rule wins all
          pings — front, back, and spread each take a meaningful share —
          so per-transition rule choice would beat any fixed convention.

        - **On the denser trips the oracle approaches the GPS + projection
          noise floor.** Where it does, the path enumeration is essentially
          right and the remaining error is almost entirely *dwell timing* —
          knowing when, not where. Where the oracle is still high, path
          selection (not the rule) is the bottleneck.

        This is the strongest single lever left: making the dwell
        allocation a modelled latent — e.g. an empirical `P(dwell_bin)`
        the CRF marginalises over per transition — rather than a fixed
        convention. That's the headline open direction (see
        `QUESTIONS_DEFERRED.md`).
    """))

    # ========================================================== SECTION 5
    cells.append(_md("""
        ---

        ## 5. Posterior entropy — where is the model confident?

        Per-transition Shannon entropy of the path posterior is a clean
        confidence summary:

        - `H = 0`: one candidate has all the weight; the model is sure.
        - `H = log K`: weight uniformly spread across K candidates;
          maximum uncertainty for a set of that size.

        Plotting `H` along the timeline of the LONG trip surfaces where
        the model needed to rely on priors vs where the GPS chain
        constrained it firmly.
    """))

    cells.append(_code("""
        # Compute per-transition entropy across all segments of the LONG trip at 120s.
        entropy_records = []
        for seg in seg_long_list:
            for k, marg in enumerate(seg.path_marginals):
                t_k = seg.canonical_timestamps[k]
                weights = np.array(list(marg.values()))
                weights = weights[weights > 0]
                if weights.size == 0:
                    continue
                H = float(-np.sum(weights * np.log(weights)))
                H_max = float(np.log(weights.size))
                entropy_records.append({
                    "trip_t": (t_k - seg_long_list[0].canonical_timestamps[0]).total_seconds() / 60,
                    "H": H,
                    "H_norm": H / H_max if H_max > 0 else 0.0,
                    "n_cands": int(weights.size),
                    "top_w": float(weights.max()),
                })

        ts = [r["trip_t"] for r in entropy_records]
        Hs = [r["H"] for r in entropy_records]
        H_norm = [r["H_norm"] for r in entropy_records]
        n_cands = [r["n_cands"] for r in entropy_records]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        ax1.bar(ts, Hs, width=1.8, color=COLOR_PRED_DWELL, alpha=0.7,
                edgecolor="white", linewidth=0.5)
        ax1.set_ylabel("Shannon entropy (nats)")
        ax1.set_title(f"LONG trip — posterior entropy across {len(entropy_records)} transitions")

        ax2.bar(ts, n_cands, width=1.8, color="#0e7490", alpha=0.6,
                edgecolor="white", linewidth=0.5)
        ax2.set_ylabel("# candidate paths")
        ax2.set_xlabel("minutes since trip start")
        plt.tight_layout()
        plt.show()

        print(f"  mean entropy:           {np.mean(Hs):.3f} nats")
        print(f"  mean normalised H/H_max: {np.mean(H_norm):.3f}  (0 = certain, 1 = uniform)")
        print(f"  fraction OPINIONATED (top weight ≥ 0.6): "
              f"{sum(1 for r in entropy_records if r['top_w'] >= 0.6) / len(entropy_records):.0%}")
        print(f"  fraction UNIFORM-ISH (top weight < 0.35): "
              f"{sum(1 for r in entropy_records if r['top_w'] < 0.35) / len(entropy_records):.0%}")
    """))

    cells.append(_md("""
        Low-entropy transitions are usually short urban hops with one
        dominant path. High-entropy moments are either genuinely
        ambiguous (parking lots, dense intersections) or places where
        the candidate set itself is large (many enumerated alternatives).
        A well-trained μ should drop average entropy without flattening
        out — being decisive when there's signal, honest when there isn't.
    """))

    # ========================================================== SECTION 6
    cells.append(_md("""
        ---

        ## 6. Off-road candidates — when routing can't represent the maneuver

        Standard map-matching assumes two consecutive observations are
        joined by a *routed* path on the network. Under sparse sampling on
        a one-way street grid, that assumption breaks. A vehicle that
        idles, parks, or does an arrival maneuver can be observed at two
        points that project — correctly, by the emission factor — onto two
        **disconnected one-way edges**. Legal routing between them needs a
        long block-loop the vehicle never drove. The model hallucinates a
        2–3 km detour; the 15 s ground truth shows it barely moved.

        The fix is an **off-road candidate**: a straight-line path between
        the two well-fitting endpoints, with the residual budget as dwell,
        representing the off-network maneuver OSM routing can't. It's
        *additive* (routed candidates are never removed) and gated by three
        conservative, truth-calibrated conditions — short straight-line gap
        (< 120 m), a routed detour ≥ 3× that gap, and that detour being
        *overslacked* (un-driveable in the available time). On an 80-trip
        Porto sample these gates fire on ~1 % of transitions with 100 %
        precision against the 15 s oracle.

        Below: the failure, the fix, and a guardrail showing a genuine
        long detour is untouched.
    """))

    cells.append(_code("""
        from src.model import Path as _ModelPath

        def _transition_window_error(seg, k, truth):
            t_lo = seg.canonical_timestamps[k]; t_hi = seg.canonical_timestamps[k + 1]
            errs = []
            for o in truth:
                if t_lo < o.timestamp < t_hi:
                    pred = position_at_time([seg], o.timestamp, network, rule="front")
                    if pred:
                        errs.append(haversine_m(o.lat, o.lon, *pred))
            return errs

        def _state_ll(state):
            idx = network.edge_index_for_link(state.link_id)
            g = network.geoms[idx]; ml = float(network.lengths_m[idx])
            if g.length <= 0 or ml <= 0: return None
            p = g.interpolate((state.offset / ml) * g.length)
            return float(p.y), float(p.x)

        def _scan(segs, truth, want_halluc):
            best = None; best_score = -1.0
            for seg in segs:
                for k in range(len(seg.path_marginals)):
                    mle = seg.most_likely[2*k+1]
                    if not isinstance(mle, _ModelPath): continue
                    t_lo = seg.canonical_timestamps[k]; t_hi = seg.canonical_timestamps[k+1]
                    pings = [o for o in truth if t_lo <= o.timestamp <= t_hi]
                    if len(pings) < 3: continue
                    mx = max(haversine_m(pings[0].lat, pings[0].lon, o.lat, o.lon) for o in pings)
                    ratio = mle.length_meters / max(2*mx, 1.0)
                    score = ratio if want_halluc else (mx if ratio < 1.3 else -1.0)
                    if score > best_score:
                        best_score = score; best = (seg, k, mx, ratio)
            return best

        # The canonical hallucination case: LONG trip, obs project to
        # disconnected one-way edges; model routes a block-loop.
        long_raw_or = load_trip(LONG_TRIP, min_pings=50)
        truth_or = drop_kinematic_spikes(clean(long_raw_or))
        raw120_or = downsample(long_raw_or, 8)

        segs_off = reconstruct_trajectory(raw120_or, network, make_config(network, offroad=False))
        seg_b, k_b, max_exc, ratio = _scan(segs_off, truth_or, True)
        mle_b = seg_b.most_likely[2*k_b+1]
        err_off = _transition_window_error(seg_b, k_b, truth_or)

        print("BEFORE (off-road disabled):")
        print(f"  MLE: {len(mle_b.edges)}-edge routed detour, {mle_b.length_meters:.0f} m")
        print(f"  vehicle's actual max excursion (15s truth): {max_exc:.0f} m")
        print(f"  route / excursion ratio: {ratio:.1f}x  (hallucination)")
        print(f"  position error: median {np.median(err_off):.0f} m, max {np.max(err_off):.0f} m")

        segs_on = reconstruct_trajectory(raw120_or, network, make_config(network, offroad=True))
        t0_b = seg_b.canonical_timestamps[k_b]
        mle_a = None; err_on = None; seg_a = None; k_a = None
        for seg in segs_on:
            for k in range(len(seg.path_marginals)):
                if seg.canonical_timestamps[k] == t0_b:
                    mle_a = seg.most_likely[2*k+1]; err_on = _transition_window_error(seg, k, truth_or)
                    seg_a, k_a = seg, k
        print()
        print("AFTER (off-road enabled):")
        tag = "OFF-ROAD" if getattr(mle_a, "is_off_road", False) else "routed"
        print(f"  MLE is now: {tag}, {len(mle_a.edges)}-edge, {mle_a.length_meters:.0f} m")
        print(f"  position error: median {np.median(err_on):.0f} m, max {np.max(err_on):.0f} m")
        print(f"  improvement: median {np.median(err_off):.0f} -> {np.median(err_on):.0f} m")
    """))

    cells.append(_code("""
        # Guardrail: a genuine long detour (vehicle really drove the long
        # way, large 15s excursion) must NOT be converted to off-road.
        legit = _scan(segs_off, truth_or, False)
        seg_l, k_l, exc_l, ratio_l = legit
        mle_l = seg_l.most_likely[2*k_l+1]
        t0_l = seg_l.canonical_timestamps[k_l]
        still_routed = True
        for seg in segs_on:
            for k in range(len(seg.path_marginals)):
                if seg.canonical_timestamps[k] == t0_l:
                    m = seg.most_likely[2*k+1]
                    still_routed = not (isinstance(m, _ModelPath) and getattr(m, "is_off_road", False))
        print("GUARDRAIL — legitimate detour:")
        print(f"  {len(mle_l.edges)}-edge routed, {mle_l.length_meters:.0f} m; "
              f"vehicle excursion {exc_l:.0f} m (real driving)")
        print(f"  with off-road enabled: "
              f"{'OK — still routed' if still_routed else 'FAIL — wrongly converted'}")
    """))

    cells.append(_code("""
        # Before/after map: the hallucinated detour vs the off-road MLE,
        # with the 15s truth pings (viridis = time) showing the vehicle
        # barely moved.
        pings = [o for o in truth_or
                 if seg_b.canonical_timestamps[k_b] <= o.timestamp <= seg_b.canonical_timestamps[k_b+1]]
        lats = [o.lat for o in pings]; lons = [o.lon for o in pings]
        for link_id in mle_b.edges:
            idx = network.edge_index_for_link(int(link_id))
            for x, y in network.geoms[idx].coords:
                lons.append(x); lats.append(y)

        fig, axes = plt.subplots(1, 2, figsize=(14, 7))
        bbox = clean_map_axes(axes[0], lats=lats, lons=lons, pad_frac=0.12)
        cmap = plt.cm.viridis
        for ax, (mle, color, title) in zip(axes, [
            (mle_b, COLOR_MLE_PATH, f"BEFORE: hallucinated detour\\n{mle_b.length_meters:.0f} m, median err {np.median(err_off):.0f} m"),
            (mle_a, "#16a34a", f"AFTER: off-road MLE\\n{mle_a.length_meters:.0f} m, median err {np.median(err_on):.0f} m"),
        ]):
            clean_map_axes(ax, bbox=bbox); draw_network_backdrop(ax, network, bbox)
            ax.set_xticks([]); ax.set_yticks([])
            for link_id in mle.edges:
                idx = network.edge_index_for_link(int(link_id))
                xs, ys = zip(*list(network.geoms[idx].coords))
                ax.plot(xs, ys, c=color, lw=3.0, alpha=0.85, zorder=3)
            for i, o in enumerate(pings):
                ax.scatter(o.lon, o.lat, c=[cmap(i / max(len(pings)-1, 1))],
                           s=55, edgecolor="white", linewidth=0.6, zorder=5)
            ax.set_title(title, fontsize=10)
        fig.suptitle("Off-road candidate: 2.7 km hallucinated loop → honest 100 m near-stationary path", y=1.02)
        plt.tight_layout(); plt.show()
    """))

    cells.append(_md("""
        The 15 s pings (coloured by time) sit in a tight cluster — the
        vehicle was effectively stationary, doing an arrival maneuver near
        a one-way pair. With off-road disabled the model is *forced* to
        connect the two correctly-projected endpoints by the only legal
        route, a multi-block loop, and the predicted positions scatter
        across that loop. With off-road enabled the straight-line
        candidate wins the posterior on its short length and large dwell,
        collapsing the error from ~66 m to ~25 m on this transition while
        leaving genuine detours untouched.

        This is the calibrated-set philosophy doing exactly its job: we
        *add* a hypothesis the router structurally couldn't express, and
        let the posterior choose. The feature is opt-in
        (`Config.enable_offroad_candidates`) and enabled throughout this
        notebook.
    """))

    # ========================================================== SECTION 7
    cells.append(_md("""
        ---

        ## 7. 60 s vs 120 s — how the reconstruction degrades with sparsity

        Everything so far worked at 120 s. The companion question for a
        fleet operator is: how much does halving the report interval to
        60 s buy you? We reconstruct the three canonical trips at both
        rates and measure, against the 15 s truth, three things per rate:
        candidate-set size (how much ambiguity the model faces), top
        posterior weight (how decisively it resolves it), and median
        position error (how close it lands).
    """))

    cells.append(_code("""
        def rate_stats(segments, truth_pings):
            n_paths, top_w, errs = [], [], []
            for seg in segments:
                for k, marg in enumerate(seg.path_marginals):
                    if not marg:
                        continue
                    n_paths.append(len(marg))
                    top_w.append(max(marg.values()))
                    t_lo, t_hi = seg.canonical_timestamps[k], seg.canonical_timestamps[k+1]
                    es = []
                    for o in truth_pings:
                        if t_lo < o.timestamp < t_hi:
                            p = position_at_time([seg], o.timestamp, network, rule="front")
                            if p:
                                es.append(haversine_m(o.lat, o.lon, *p))
                    if es:
                        errs.append(float(np.mean(es)))
            return {
                "n_trans": len(n_paths),
                "mean_n_paths": float(np.mean(n_paths)) if n_paths else 0.0,
                "mean_top_w": float(np.mean(top_w)) if top_w else 0.0,
                "median_err": float(np.median(errs)) if errs else float("nan"),
            }

        degr = []   # (trip, stats_60, stats_120)
        for label, tid in [("SHORT", SHORT_TRIP), ("MEDIUM", MEDIUM_TRIP),
                           ("LONG", LONG_TRIP)]:
            raw = load_trip(tid, min_pings=20)
            truth = drop_kinematic_spikes(clean(raw))
            s60 = rate_stats(reconstruct_trajectory(downsample(raw, 4), network,
                                                    make_config(network)), truth)
            s120 = rate_stats(reconstruct_trajectory(downsample(raw, 8), network,
                                                     make_config(network)), truth)
            degr.append((label, s60, s120))

        print(f"{'trip':8s} {'metric':26s} {'60s':>8s} {'120s':>8s}")
        print("-" * 54)
        for label, s60, s120 in degr:
            for key, name in [("mean_n_paths", "mean # candidate paths"),
                              ("mean_top_w",   "mean top posterior weight"),
                              ("median_err",   "median position error (m)")]:
                print(f"{label:8s} {name:26s} {s60[key]:8.2f} {s120[key]:8.2f}")
            print()
    """))

    cells.append(_code("""
        # Grouped bars: median position error at 60s vs 120s per trip.
        trips = [d[0] for d in degr]
        err60 = [d[1]["median_err"] for d in degr]
        err120 = [d[2]["median_err"] for d in degr]
        x = np.arange(len(trips)); width = 0.36
        fig, ax = plt.subplots(figsize=(9, 5))
        b1 = ax.bar(x - width/2, err60, width, label="60 s",
                    color="#0e7490", edgecolor="white")
        b2 = ax.bar(x + width/2, err120, width, label="120 s",
                    color=COLOR_OBS_SPARSE, edgecolor="white")
        for bars, vals in [(b1, err60), (b2, err120)]:
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2, v + 2, f"{v:.0f}",
                        ha="center", fontsize=9)
        ax.set_xticks(x); ax.set_xticklabels(trips)
        ax.set_ylabel("median position error (m)")
        ax.set_title("Reconstruction error: 60 s vs 120 s sampling (front-loaded)")
        ax.legend()
        plt.tight_layout(); plt.show()
    """))

    cells.append(_md("""
        The pattern is graceful, not cliff-edged: halving the interval
        from 120 s to 60 s shrinks the candidate set (less ambiguity per
        transition), raises decisiveness, and lowers error — without any
        regime change. 60 s is sparse relative to the 15 s native cadence,
        but the pipeline accommodates it cleanly; 120 s is harder but
        still produces usable, calibrated reconstructions rather than
        breaking down. That graceful degradation across realistic fleet
        rates is the property an operator actually cares about.
    """))

    # ========================================================== SECTION 8
    cells.append(_md("""
        ---

        ## 8. Take-aways

        - **The pipeline outputs a distribution over paths, not a single
          path.** That distribution honestly tracks uncertainty as
          sampling sparsens.
        - **Each path carries an implied dwell.** Dwell is a derived
          quantity from path enumeration, not a learned latent. The
          residual time between the time gap and the path's travel time
          is allocated to dwell at the path's origin.
        - **Confirmed dwell is a data fact; inferred dwell is a model
          claim.** The pipeline tracks both, and the time accounting
          subtracts confirmed dwell from the transit budget before
          candidate paths compete.
        - **Dwell-allocation rule is a modelling choice, not a deduction.**
          Front, back, and spread are equally consistent with the CRF; we
          ship front-loaded as the convention. The oracle analysis shows
          the *rule choice* — not the path — is the dominant remaining
          error on well-sampled transitions, which is the strongest open
          lever (a modelled dwell-timing latent).
        - **Degradation is graceful across realistic fleet rates.** From
          60 s to 120 s the candidate set widens and error rises smoothly,
          with no regime change — the pipeline accommodates sparse feeds
          rather than breaking at a cliff.
        - **The driver model `μ` is a pluggable component.** It's a learned
          CRF factor (`src/data/mu_default.npy`); its value depends on the
          data and feature richness. On Porto-at-120 s its effect is
          modest — see the README for how to train and apply it, and the
          honest note on where it helps.
        - **Off-road candidates handle what routing structurally can't.**
          When two observations project onto disconnected one-way edges,
          legal routing invents a block-loop the vehicle never drove. A
          gated, additive straight-line candidate lets the posterior
          recover the real near-stationary maneuver — fixing the worst
          single-transition outliers without touching genuine detours.
        - **What's deliberately not promised here.** Learned dwell
          distributions, context-aware dwell, a full off-network state
          regime (beyond the straight-line maneuver candidate), and
          real-time / streaming inference are out of scope. See
          `QUESTIONS_DEFERRED.md` for the running list of open threads.

        For methodology in full see `OVERVIEW.md`; for the module-level
        spec see `SPEC.md`; for the training pipeline see
        `scripts/compute_15s_labels.py` and `scripts/retrain_mu.py`.
    """))

    nb.cells = cells
    return nb


def main() -> int:
    print(f"building notebook at {NB_PATH}")
    nb = build_notebook()

    print(f"executing {len(nb.cells)} cells (this takes several minutes)…")
    client = NotebookClient(
        nb,
        timeout=1800,
        kernel_name="python3",
        resources={"metadata": {"path": str(REPO_ROOT)}},
    )
    client.execute()

    NB_PATH.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, str(NB_PATH))
    print(f"wrote {NB_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
