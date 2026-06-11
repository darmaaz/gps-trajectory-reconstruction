# Trajectory Reconstruction on a Road Network

Probabilistic trajectory reconstruction from sparse, noisy GPS observations on an OSM road network. The output is not a single most-likely path but a calibrated set of feasible paths plus a dwell-aware annotation of how each candidate splits its time between transit and stationary periods.

## Why

Under sparse sampling — one GPS ping every minute or more — a single pair of consecutive pings is consistent with many driving stories: a fast direct route, a slow scenic one, or a quick run followed by a long wait at the destination. Conventional map-matching commits to one and discards the rest. For consumers that reason about uncertainty (fleet analytics, dwell-at-customer estimation, route auditing), those alternatives are the signal.

The pipeline produces:

- a **path posterior** per transition (probability over enumerated candidate paths),
- a **per-observation state marginal** (probability over each ping's road-edge projections),
- a **dwell annotation** per path — `inferred_dwell = time_budget − expected_travel_time`,
- a **most-likely interleaved trajectory** from Viterbi, when a single best guess is needed,
- **structural discontinuities**, with diagnostic context, when the data can't support a contiguous reconstruction.

## Methodology in one paragraph

Preprocess raw pings (hygiene, kinematic-spike removal, replay-burst collapse, position-uniqueness collapse, stale-jump detection). Project each observation onto its top-K nearest road edges. Between consecutive observations, enumerate top-K feasible paths with A* under edge-penalty diversification, so the candidate set spans structurally different routes rather than near-duplicates of the optimum. Carry a confirmed-dwell floor through the time-budget arithmetic: the static span at an observation is transit time *removed*, not transit time *allowed*. Run forward-backward over the resulting CRF in log domain for state marginals and path posteriors; run Viterbi for the most-likely path; split the trip where the data can't support a contiguous reconstruction. See `OVERVIEW.md` for the full methodology.

## Repository layout

```
src/                  # the pipeline
  preprocessing/      # hygiene → spikes → replay → collapse → stale-jump
  network/            # OSM PBF loader + A* + penalty-diversified routing
  candidates/         # observation → state projections; per-transition paths
  model/              # State, Path, factor protocols, ϕ(p)
  inference/          # forward-backward, Viterbi
  training/           # supervised MLE, hard-EM
  validation/         # sanity gates, holdout, baseline comparison
  feeds/              # data-source adapters (Porto Kaggle)
  api/                # reconstruct_trajectory + TrajectoryPosterior
scripts/              # demo and visualisation scripts (Porto)
tests/                # pytest suite (149 tests)
cache/                # OSM parquet cache + generated demo outputs
OVERVIEW.md           # the methodology document
```

## Running

The pipeline is validated on the [Porto Kaggle taxi dataset](https://www.kaggle.com/c/pkdd-15-predict-taxi-service-trajectory-i) at its native 15-second cadence, downsampled to 120 s / 300 s to simulate sparse sampling.

### Setup

```bash
pip install -e .
```

You'll need two external inputs the repo doesn't ship. By default they're read from `~/Documents/shared_data/`; point the pipeline elsewhere with environment variables:

- **OSM PBF** for the region of interest (Portugal for the Porto demos) — default `~/Documents/shared_data/osm/portugal-latest.osm.pbf`, override with `GPS_RECON_PBF`.
- **Porto Kaggle CSV** (`train.csv`) — default `~/Documents/shared_data/porto/train.csv`, override with `GPS_RECON_CSV`.

```bash
# only if your data lives elsewhere
export GPS_RECON_DATA=/path/to/shared_data            # base dir for both, or
export GPS_RECON_PBF=/path/to/portugal-latest.osm.pbf  # override individually
export GPS_RECON_CSV=/path/to/porto/train.csv
```

A parsed-edge parquet cache (`cache/pt_edges.parquet`) short-circuits the PBF parse after the first run.

### Demos

| Script | What it shows |
|---|---|
| `scripts/smoke_porto.py` | End-to-end pipeline run on a small batch of Porto trips. Diagnostic stats only — confirms the pipeline produces sensible output before any calibration. |
| `scripts/visualize_porto.py` | Map + static plot of a single trip's reconstruction (road network, observations coloured by segment, Viterbi MLE overlay). |
| `scripts/compare_sampling.py` | Reconstruct the same trip at 15 s and 120 s; overlay observations and most-likely paths. Shows how the candidate set widens with sparser sampling. |
| `scripts/reconstruct_15s_from_120s.py` | Predict positions at the dropped 15 s timestamps from the 120 s reconstruction; compare against the actual 15 s pings. Per-timestamp error stats. |
| `scripts/demo_dwell_budget.py` | Per-transition dwell-vs-transit budget split on the canonical SHORT/MEDIUM/LONG Porto trips. |
| `scripts/demo_dwell_recovery.py` | 15 s ↔ 120 s dwell-recovery validation on the LONG trip: does the 120 s path candidate set contain a path whose `inferred_dwell` matches the 15 s ground truth? |
| `scripts/demo_path_overlap.py` | 15 s vs 120 s candidate-path overlap on the LONG trip; quantitative edge-coverage stats. |

### Training the driver model

The shipped default `μ` (`src/data/mu_default.npy`) is fit by supervised MLE on Porto trips where the native 15 s reconstruction acts as ground truth for the 120 s downsampling. The 15 s reconstruction uses a generic, model-independent length prior (which suppresses near-stationary "spur" artifacts the old `μ=0` recipe admitted) and off-road candidates — which is also why `Config.enable_offroad_candidates` now defaults to `True`.

**Schema note (dim 19):** `FEATURE_DIM` is now 19 — slot [18] counts direction-violation *maneuvers* (runs of consecutive reversed edges in `Path.reversed_mask`, so OSM's mid-corridor splits don't over-count a single wrong-way drive; F5). A pre-bump 18-dim `mu_default.npy` is transparently padded with a hand prior (−2.0/maneuver) by `default_mu()`, so inference works unchanged; the dim-19 retrain below replaces the prior with a learned weight (shipped: −2.78/maneuver). The label recipe correspondingly enumerates direction-violation candidates on *both* the 15 s and 120 s sides (without this, μ[18] has zero gradient — `retrain_mu.py` detects that and re-pins the hand prior) and matches paths by undirected segment-key Jaccard, so opposite-twin spellings of the same street no longer score as disjoint. The no-argument commands below reproduce the shipped artifacts:

```bash
# Slow side (~50 min): reconstruct each trip at native 15 s, downsample,
# label the 120 s candidates against the 15 s Viterbi MLE.
python scripts/compute_15s_labels.py    # writes cache/labeled_trips_15s.pkl.gz

# Fast side (~seconds): supervised fit.
python scripts/retrain_mu.py            # writes src/data/mu_default.npy
```

`compute_15s_labels.py` also accepts `--labels-from raw-pings` (label the 120 s candidates by threading the raw GPS pings, skipping the 15 s pass entirely — under evaluation), `--prior zero --no-offroad` (the legacy emission-only recipe), and `--no-direction-violation` (the pre-F5 recipe). Re-run whenever `FEATURE_DIM` changes — `src/data/default_mu()` enforces the shape and falls back to zeros if the file is absent.

### Tests

```bash
pytest
```

Test suite covers preprocessing, network loading, A* + penalty-diversified routing, forward-backward (against hand-computed posteriors on toy graphs), Viterbi cliff-handling, training, the Porto feed, the orchestrator's discontinuity behaviour, and the dwell-allocation rule (`front` / `back` / `spread`) for off-grid `at_time` queries.

## Status

The pipeline produces calibrated path posteriors and per-path dwell annotations for arbitrary trips. `MarginalQuery.at_time(t)` resolves arbitrary `t` under three selectable dwell-allocation conventions (front-loaded default, back-loaded, constant-speed spread), with the confirmed-dwell window anchored to `canonical_t_last`.

## References

The driver-model formulation follows Newson & Krumm's *Hidden Markov Map Matching Through Noise and Sparseness* (2009) and the PIF (*Path Inference Filter*) line of work from Hunter, Abbeel & Bayen (Berkeley) for the CRF + path-feature exponential family. The candidate-path diversification uses the Plateau / Penalty method common in alternative-routes literature.
