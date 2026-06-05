# Trajectory Reconstruction on a Road Network

## Problem

A vehicle traverses a road network over an interval of interest. We observe sparse, noisy GPS reports and want to reconstruct the **path taken on the network** as a probabilistic object — a distribution over edge-sequences, plus a most-likely path, per-observation state marginals, and a dwell-aware position estimate at intermediate times.

Under sparse observation (one ping every minute or more), a single pair of consecutive pings is consistent with many driving stories — a fast direct route, a slow scenic one, a quick run followed by a long wait at the destination. Standard map-matching commits to one; we want a calibrated set, so downstream consumers can pick the educated guess or reason about the full range.

## What's observed

`O = {oᵢ}` with `oᵢ = (τᵢ, yᵢ, qᵢ)`: timestamp, reported position, and auxiliary fields (fix type, HDOP, reported speed/heading where exposed). Reports may be **stale** (frozen from a past freeze time) or **fresh**; staleness is handled in preprocessing rather than as a model latent.

Road network `G = (V, E)`: directed graph with edge geometries, lengths, classes, speed limits, turn restrictions. Approximately correct; not authoritative.

Vehicle-class priors: kinematic envelope (max speed, typical accelerations) appropriate to the vehicle.

## What's latent

For each preprocessed observation index *k*: a candidate **state** set `xᵏ = {xᵏᵢ}` of (edge, offset) projections. Between consecutive observations: a candidate **path** set `pᵏ = {pᵏⱼ}` of feasible paths in `G`. A trajectory is the interleaved sequence `τ = x¹ p¹ x² p² … pᵀ⁻¹ xᵀ`.

Each state carries an `entry_time`. Each path additionally carries `time_budget`, `expected_travel_time`, and an `inferred_dwell` annotation — the residual time the vehicle is implied to have spent stationary at the path's origin state if it took that path.

## Preprocessing

Observation-level pathology is offloaded to preprocessing, so the inference layer sees a clean, conventional GPS sequence.

**P1 — Hygiene.** Sentinel coordinates `(0, 0)`, out-of-range lat/lon, HDOP above threshold, and timestamp monotonicity violations are dropped (or repaired, configurably).

**P2 — Collapse by positional uniqueness.** Consecutive reports within ε meters are collapsed into a single observation tagged `(position, t_first, t_last, count)`. This treats stale runs and genuine stops uniformly — both look like static periods at the observation level. The `t_last − t_first` span on a non-stale run becomes the **confirmed dwell**: a floor on the time the vehicle spent stationary here.

**P3 — Stale-jump detection.** For each transition from collapsed observation `A` to next `B`, compute the minimum feasible travel time given `G` and the vehicle's max-speed envelope. If `B.t_first − A.t_last < min_travel_time(A→B)` but `B.t_first − A.t_first ≥ min_travel_time(A→B)`, the static run at `A` is flagged stale: its internal pings are an artifact of device caching, not real dwell. Stale-flagged runs contribute zero confirmed dwell; the full `t_first → t_first` gap becomes transit budget.

**P4 — Kinematic-spike removal.** Single pings (or runs up to a configured length) whose entry/exit speeds exceed a chip-glitch threshold are removed; the surrounding pings are bridged.

**P5 — Replay-burst detection.** Buffered-replay regions — long stretches of high-frequency identical or near-identical pings emitted in burst after a backhaul gap — are detected and collapsed.

The output is a sequence of `(position, t_first, t_last, stale_flagged)` observations that inference can consume as if it were ordinary low-frequency GPS data.

## Methodological layering (for context)

The pipeline was built in two conceptual passes:

- **Phase 1 — candidate states and paths.** Project each observation onto top-K edges; enumerate top-K feasible paths between consecutive states; run forward-backward over the resulting CRF for state marginals and path posteriors.
- **Phase 2 — dwell-aware extension.** The time between two observations partitions into transit and dwell, and any candidate path makes an implicit claim about that split. Surface `inferred_dwell` as a first-class annotation per path, diversify the candidate set so structurally different transit/dwell stories are represented, and expose intermediate-time position queries that honour the allocation.

The layering matters because the dwell-aware features are *additive*: the same forward-backward consumes the dwell-annotated candidates, with no separate dwell-latent. The rest of this document describes the pipeline as it stands, not as it was built.

## Assumptions

**A1.** Each preprocessed observation is a noisy projection of the vehicle's true road position at the canonical timestamp: `yᵢ = h(xᵢ) + ηᵢ`, with `ηᵢ` heavy-tailed (e.g., Student-t) on perpendicular distance to the candidate edge.

**A2.** The vehicle is on `G` for the duration of inference. Genuine off-network driving snaps to the nearest road as best-effort; modeling off-network as a separate regime is deferred.

**A3.** Driver behaviour follows preferences over paths in `G`, modelled as an exponential family over path features `ϕ(p)`: `η(p) ∝ exp(μᵀϕ(p))`. Features include path length, travel-time fit, number of turns, signals, stop signs, and road-class composition.

**A4.** `G` is approximately correct. Edge-existence uncertainty and geometry error are not separately modelled; mass that fits poorly is absorbed by the heavy-tailed emission.

**A5.** Speed limits and class-conditional speed priors give upper bounds on feasible travel times per link, used during candidate path discovery.

## Model

A PIF-style Conditional Random Field over the discrete trajectory `τ`:

```
φ(τ | y¹:ᵀ) = [ ∏ₜ₌₁ᵀ⁻¹ ω(yᵗ|xᵗ) · δ(xᵗ, pᵗ) · η(pᵗ) · δ̄(pᵗ, xᵗ⁺¹) ] · ω(yᵀ|xᵀ)
π(τ | y¹:ᵀ) = Z⁻¹ φ(τ | y¹:ᵀ)
```

where `δ` and `δ̄` are start/end compatibility indicators between states and paths.

**Emission `ω(y|x)`:** heavy-tailed in perpendicular distance from the report to the candidate edge. A robust family (Student-t, or a Gaussian-with-bounded-uniform mixture) is preferred over plain Gaussian to absorb residual upstream-pipeline pathology not caught by preprocessing.

**Driver model `η(p)`:** exponential family parameterised by `μ`, implemented as a pluggable factor.

## Time accounting

The transit window between consecutive observations is

```
time_budget[k] = (t_first[k+1] − t_first[k]) − confirmed_dwell[k]
```

where `confirmed_dwell[k] = t_last[k] − t_first[k]` on non-stale runs and `0` on stale-flagged runs — confirmed-stationary periods do not contribute to transit time.

For any candidate path `p` enumerated under `time_budget[k]`,

```
inferred_dwell(p) = time_budget[k] − expected_travel_time(p)
```

— the residual time the vehicle would have spent stationary at state *k* if it took path `p`. `inferred_dwell` is deterministic from the budget; it is not learned. The path posterior `r^k(p)` weights the candidates by their feature-induced plausibility, so

```
E[dwell_k] = Σ_p r^k(p) · inferred_dwell(p)
```

is an expected-dwell estimate over the posterior, calibrated to the candidate set.

## Candidate path discovery

A* search on `G` from each source state to each destination state, with travel-time cost (`length / max_speed` per edge) and an admissible euclidean heuristic. Each accepted path is pruned against an inflated cost cap (`path_budget_slack × time_budget[k]`).

A single shortest-path search collapses the candidate set onto near-duplicates of the optimum. Two mechanisms diversify it:

- **Same-edge stay paths and same-edge backward paths** are enumerated up-front, before any graph search, and emitted unconditionally. They handle parking-lot / dwell-at-state-k scenarios that no graph search produces.
- **Edge-penalty diversification** (the Plateau / Penalty method): after each accepted routed path, multiplicatively surcharge that path's edges by `(1 + λ)`. The next search penalises re-using those edges, so the second-best path tends to be structurally different (highway vs surface street, detour around a bottleneck) rather than a near-duplicate of the first. Penalty is used for diversification only; feasibility pruning runs on unpenalised travel time.

Each accepted path carries `(edges, start_offset, end_offset, expected_travel_time, length_meters, time_budget, feature_vector)`, with `inferred_dwell` as a derived property.

## Inference

**Forward-backward** on the CRF yields per-observation state marginals `qᵏ` and per-transition path marginals `rᵏ`, both normalised, in log-domain throughout. **Viterbi** yields the most-likely interleaved sequence of states and paths.

When a transition has zero feasible paths, or an observation has no on-network candidate states, the trajectory is split at that boundary: the segment terminates, a `Discontinuity` is recorded with diagnostic context (last-alive states, min routable time, speed at the boundary), and a new segment begins at the next observation that does have candidates. Trips with no breaks yield a single `TrajectoryPosterior`; trips with breaks yield a list of segments tied to their preceding `Discontinuity`.

Complexity is `O(T · U · V)` where `U` is the maximum number of paths at one step and `V` is the maximum number of paths originating from a single point.

## Outputs

For each contiguous segment:

- Path posterior: per-transition `{(p, r^k(p))}` over enumerated paths.
- Most-likely interleaved trajectory `τ*` from Viterbi.
- Per-observation state marginals `q̄ᵏ`.
- Confirmed-dwell vector across transitions, persisted alongside `time_budgets` and `canonical_timestamps`.
- `MarginalQuery` interface: `at_observation(k)` returns the on-grid state marginal; `at_time(t)` resolves intermediate timestamps (interpolation across the dwell+transit allocation of each path, aggregated over the posterior).

## Training

If labelled trajectories are available, `μ` and the emission scale are fit by maximum likelihood (convex in `μ`). Otherwise, EM on unlabelled data, with the forward-backward pass providing the E-step expectations and a convex M-step in `(μ, log_emission_scale)`.

## Validation

The pipeline is validated on the Porto Kaggle taxi dataset, which provides 15-second-cadence trips. Validation is structural and visual:

- **Downsample-and-reconstruct.** Take a native 15 s trip; downsample to 120 s, 300 s. Reconstruct from the sparse version. Confirm the most-likely path covers the native 15 s ground-truth-ish path, and the candidate set's union covers the regions the most-likely misses.
- **Dwell recovery.** For each sparse-window transition, aggregate the native 15 s confirmed-dwell values into the matching window, and compare to the expected dwell `Σ_p r^k(p) · inferred_dwell(p)` from the sparse posterior. Validates that the path candidate set contains a path whose dwell story matches reality — i.e., the truth is *in the set*, even when the posterior is uncertain about which member.
- **Sanity gates and held-out NLL** on labelled-validation trips (Tier 1–4 in `src/validation/`), available when labels exist.

## Calibration

Parameters: emission scale and tail, driver weights `μ`, stale-detection threshold, collapse radius, path-budget slack, penalty-diversification `λ`.

Identifiability is narrower than in the original problem framing because (a) staleness is a preprocessing decision with explicit detection rules, not a model latent; (b) off-network is tolerated by the heavy-tailed emission rather than modelled. The remaining tension is between emission scale and driver model: a wider emission lets more paths fit; a tighter one sharpens the posterior. EM on representative trips converges reliably for this two-way tension.

If labelled traces are available (parallel high-quality logging on a small fleet for tens to hundreds of trips), supervised fitting is preferred. Otherwise, EM on the production fleet at scale, validated against a small hand-labelled set.

## Scope notes

- Real-time online matching is not a goal; the pipeline is offline batch.
- Stale-detection failure cases (chip jitter rather than exact freeze, staleness with kinematically borderline jumps) are absorbed by the heavy-tailed emission rather than handled with a dedicated latent.
- Off-network state and graph-error modelling are out of scope.
- Latency is unconstrained; accuracy and calibrated uncertainty are the priorities.
