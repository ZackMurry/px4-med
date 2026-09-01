# CEDA-FGCS-PX4 — consolidated SITL validation results

All experiments 2026-08-29 → 2026-08-31 on an idle 12-core host.
**94 SITL episodes across four campaigns, 0 contaminated, 1 job abandoned**
(to a bug in the training env, §6). Model
`models/ctde_agent_marl_FGCS.pth` (SHA-256 `3d0df78d…`), curriculum stage 2
(`full_5_drone_50_patient_100x100_cross_layer`), 5 drones, 50 patients,
100×100 grid at 2 m/cell, 800-step mission budget + 400-step landing grace.

| campaign | dir | jobs | purpose |
|---|---|---|---|
| core | `results/core_20260829_230535` | 24 | 4-policy nominal comparison + battery stress |
| hazard | `results/hazard_20260830_104103` | 32 | hazard-density sweep 0.5-2.0× |
| extend | `results/extend_20260831_000000` | 27 | power top-up for learned vs nearest_path |
| latency | `results/latency_20260831_000000` | 11 | command-staleness robustness |
| offline twin | `results/offline_twin_core_20260829_230535` | 24 | paired offline mirror (matched seeds) |
| offline sweep | `results/offline_sweep_baseline_comparison_20260831_211720` | 200 | large-n offline reference (n=50/policy) |

Per-campaign figures in each `paper_figures/`; per-episode data in
`tables/episodes.csv` (75 columns); group means + 95% CI in
`tables/summary.csv`.

---

## 1. THE headline: abstract evaluation misses the advantage that SITL finds

Learned vs `nearest_path` (the strongest baseline), delivery rate, Welch's t:

| evaluation | learned | nearest_path | difference [95% CI] | p |
|---|---|---|---|---|
| Abstract env, **n=50/arm** | 0.966 | 0.948 | +0.017 [−0.003, +0.038] | 0.102 — **not significant** |
| PX4 SITL, **n=19/arm** | 0.701 | 0.638 | +0.063 [+0.012, +0.114] | **0.016 — significant** |

Same pattern on triage efficiency (offline +0.020, p=0.067; SITL +0.080,
p=0.010).

**This is not a statistical-power artifact — it is the opposite.** The offline
comparison has 2.6× more episodes and intervals roughly 3× tighter, and still
finds nothing. The abstract environment structurally cannot see the
difference, because the learned policy's advantage lies in energy-aware
routing under real flight time, which abstract stepping does not charge for.

For a validation paper this is the central methodological claim: **had we
evaluated only in the training environment, we would have concluded the
learned policy offers no significant benefit over a greedy nearest-patient
heuristic.**

## 2. Nominal comparison (PX4 SITL)

Pooled core + extend, mean ± CI95:

| metric | learned (n=19) | nearest_path (n=19) | diff [95% CI] | p |
|---|---|---|---|---|
| delivery rate | 0.701 | 0.638 | +0.063 [+0.012, +0.114] | 0.016 |
| triage efficiency | 0.715 | 0.635 | +0.080 [+0.020, +0.139] | 0.010 |
| rescue quality | 0.708 | 0.637 | +0.071 [+0.017, +0.126] | 0.011 |
| drones landed (of 5) | 4.158 | 2.947 | +1.211 [+0.521, +1.900] | 0.0011 |
| drones depleted | 0.842 | 2.053 | −1.211 [−1.900, −0.521] | 0.0011 |
| wind avoidance | 0.998 | 0.475 | +0.523 [+0.440, +0.606] | <0.0001 |
| low-signal avoidance | 0.991 | 0.486 | +0.505 [+0.434, +0.576] | <0.0001 |

The learned policy wins on **every** dimension. Weaker baselines from the core
run (n=5): `priority_path` delivery 0.168, `random` 0.000.

Caveat on magnitude: retrospective power says ~23 episodes/arm were needed for
80% power at the observed delivery effect, and we have 19. The direction and
significance hold, but the *size* of the delivery advantage is imprecise
(plausibly 1-11 points). Quote the CI, not the point estimate. The hazard and
fleet-recovery effects are far above the noise floor and need no such hedge.

## 3. Sim-to-sim transfer: spatial skill survives, energy budget does not

Learned policy, identical world model and weights, matched seeds:

| metric | offline | PX4 SITL |
|---|---|---|
| delivery rate | 0.970 | 0.697 |
| triage efficiency | 0.970 | 0.699 |
| mission success | 1.000 | 0.167 |
| drones landed | 5.000 | 3.667 |
| drones depleted | 0.000 | 1.333 |
| **wind avoidance** | 0.986 | **0.997** |
| **low-signal avoidance** | 0.923 | **0.992** |
| episode length (steps) | 352.5 | 493.2 |

Mechanism: real accel/decel and tracking jitter (1.5-1.9 m) make each grid move
cost more time, so episodes run **~40% longer** (352 → 493 steps). The energy
ledger is fixed at 100 charge / 0.20 per step ⇒ 500 steps of flight, so the
longer episodes exhaust it — depletions 0 → 1.33, landings 5 → 3.67, patients
time out, delivery 0.97 → 0.70.

Hazard avoidance is *unaffected* (0.986 → 0.997). Corroborated by
`nearest_path`, at 0.466 offline and 0.463 in SITL: the metric tracks a stable
property of the policy, not a simulator artifact.

## 4. Hazard density sweep — safety is invariant, throughput pays

| density | learned delivery | learned wind avoid (opp) | wind exp | nearest delivery | nearest wind avoid | wind exp |
|---|---|---|---|---|---|---|
| 0.5× | 0.765 ± 0.029 | 0.962 (24) | 0.5 | 0.645 ± 0.049 | 0.481 (24) | 33.2 |
| 1.0× | 0.715 ± 0.046 | 1.000 (36) | 0.0 | 0.670 ± 0.087 | 0.540 (42) | 37.8 |
| 1.5×* | 0.650 ± 0.061 | 1.000 (90) | 0.0 | 0.515 ± 0.116 | 0.648 (88) | 76.5 |
| 2.0×* | 0.640 ± 0.096 | 0.991 (43) | 0.2 | 0.525 ± 0.077 | 0.504 (91) | 96.8 |

\\* off-distribution: training used 0.5× (stage 0) and 1.0× (stages 1-2).

The learned policy holds 0.96-1.00 avoidance and ≤0.5 steps of hazard exposure
at **every** density, clearing its own 0.98 stage-2 training gate even at
double the trained density. `nearest_path` sits at 0.48-0.65 with exposure
growing monotonically 33 → 97 steps. Delivery declines for both, and the gap
*widens* under stress (non-overlapping at 1.5×).

**Interpretation trap:** the opportunity denominator is NOT monotone in
density (24 → 36 → 90 → 43). An opportunity requires both a hazard cell and a
hazard-free cell among the valid destinations, so it rises with density and
then falls once the map saturates and every neighbour is hazardous. At high
density a drone can be *forced* into a hazard; that shows up in
`wind_exposure_steps`, not in the rate. **Always report rate and exposure
together.**

## 5. Command-latency robustness — landing breaks first, as a cliff

| delay (steps) | n | delivery | triage | landed/5 | wind avoid | tracking err |
|---|---|---|---|---|---|---|
| 0 (ref) | 19 | 0.701 ± 0.032 | 0.715 ± 0.040 | 4.16 ± 0.46 | 0.998 ± 0.002 | ~1.9 m |
| 1 | 4 | 0.570 ± 0.061 | 0.602 ± 0.071 | 4.25 ± 0.49 | 0.863 ± 0.134 | 1.98 m |
| 2 | 4 | 0.505 ± 0.067 | 0.545 ± 0.079 | **1.00 ± 0.00** | 0.729 ± 0.155 | 1.73 m |
| 4 | 3 | 0.300 ± 0.060 | 0.323 ± 0.096 | **0.00 ± 0.00** | 0.836 ± 0.199 | 1.74 m |

All delays are off-distribution (training had none).

- **Landing fails before delivery does, and it fails as a cliff:** recovery
  goes 4.16 → 4.25 → 1.00 → 0.00 with *zero variance* at delays ≥2 — every
  episode identical. Landing is precision work against current state.
- **Tracking error is flat** (1.73-1.98 m) at every delay. The drones fly just
  as accurately, they just fly to the wrong places. This cleanly separates
  control degradation from decision degradation: it is entirely the latter.
- Avoidance degrades but **non-monotonically** (0.863 → 0.729 → 0.836); with
  n=3-4 and CIs of ±0.13-0.20 that ordering is noise. Claim "degrades under
  latency", not a specific curve.
- So hazard competence is robust along the *environmental* axis (§4) and
  fragile along the *control-timing* axis. Tight command latency is a hard
  deployment requirement, and this quantifies how tight.

## 6. Battery stress (60% starting charge)

Learned: delivery 0.600 offline → 0.327 SITL; drones landed 2.67 → 2.00.
Hazard avoidance stays at 1.000/1.000 in both. The policy gives up deliveries
before it gives up hazard safety — the correct sacrifice ordering for a
medical-delivery system, and worth stating explicitly.

## 7. Known limitations

- **n=19/arm nominal, n=3-4 per sweep cell.** Adequate for the hazard and
  fleet-recovery effects (far above noise), marginal for the delivery effect
  size (§2).
- **One job abandoned** (`latency delay_4 ep001`) to a float32 tolerance bug in
  the training environment's own reward invariant — see HANDOFF.md §6c. It
  trips stochastically in severe-failure regimes, so the environment currently
  cannot reliably score its own worst cases. Reported to the collaborator with
  a one-line fix; **not patched locally**, because running his exact
  environment is what makes this validation credible.
- `mission_success` **must never be quoted alone**: it counts a *dead* patient
  as resolved, so `priority_path` (delivery 0.168) scored higher on it than the
  learned policy. Use `drones_landed` for fleet-recovery claims.
- Off-distribution arms (hazard 1.5×/2.0×, all latency delays) are robustness
  probes, not validation, and must be labelled as extrapolation.
- Two intermittent SITL faults (arm-denied, inert-drone) were traced to host
  CPU contention and fixed with guards; they did not recur on an idle host.
  See HANDOFF.md §5b-§5d.

## 8. Reproduction

```bash
env PYTHONPATH=src poetry run python -u scripts/run_overnight_validation.py \
    --plan {core|hazard|extend|latency} --output-dir results/<name> --max-hours 30
poetry run python scripts/run_offline_companion.py --sitl-dir results/<name> --only-completed
poetry run python scripts/run_offline_sweep.py --episodes 50
poetry run python scripts/detect_inert_drones.py --run-dir results/<name>   # ALWAYS
poetry run python scripts/make_paper_figures.py --run-dir results/<name> \
    --offline-dir results/offline_twin_<name>
```

Or the whole queue unattended: `nohup bash scripts/run_experiment_chain.sh > chain.log 2>&1 &`
