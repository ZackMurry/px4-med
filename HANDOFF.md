# px4-med Agent Handoff — 2026-08-29 ~21:50 local (rev 2: TRUE-WORLD stack)

Complete context for continuing this project. Written after a full working
session (model verification → 5-drone adapter → SITL debugging → first
complete episode → experiment runner port → pilot suite in flight).
Persistent memory files also exist at
`~/.claude/projects/-home-zack-px4-med/memory/` and largely mirror this.

## 1. Project goal

Run PX4 SITL validation experiments for a paper (internally "RouteMED",
target venue FGCS) on the **CEDA-FGCS-PX4** model: a 5-drone, 50-patient
CTDE multi-agent RL policy for medical delivery with triage, wind/low-signal
hazards, and a mission-energy ledger, on a 100×100 abstract grid
(2 m/cell → 200×200 m flight area). The repo (`/home/zack/px4-med`, git,
branch main) deploys it via MAVSDK against 5 PX4 SITL instances + one
headless Gazebo in a single Docker container.

The previous paper iteration used a 2-drone model; its archived repo +
results live in `/mnt/data/px4-med` (see `results/paper_core_12h_run/analysis.md`
there for the old findings and recommended follow-ups). The old 2-drone code
was replaced in-place; git history has it.

## 2. Model package (`models/`) — VERIFIED GOOD

- `ctde_agent_marl_FGCS.pth` — final checkpoint, SHA-256
  `3d0df78d5291edb730bfb507568d2a9c0c5b0f03cf70842a14e329bc1abd6275`
  (matches README), 28.9 MB, 4,835,653 env steps, 149,551 learner updates.
- `CEDA-FGCS.py` — matching loader (`CEDAFGCSPX4Policy`), strict-load +
  smoke test pass. Load via importlib (hyphen in filename) — wrapped by
  `src/px4med/fgcs_policy.py`.
- `README.md` — the observation/action contract. Verified consistent with
  the checkpoint: drones (5,22), patients (50,10), mission (12,),
  local_grids (5,3,21,21), action_masks (5,6); actions
  N,S,W,E,hover,land; battery ledger 0.20/step, +2.30 wind, 0.02 pad
  standby; safe-return Dijkstra with +18 buffer; movement failure p:
  ×0.50 low-signal, ×0.85 wind; triage service-debt targets W1 .5 / W2 .7 / W3 .9.
- History: an earlier intermediate checkpoint (`77076a2b…`, 13.5 MB,
  mid-curriculum) circulated first and caused confusion; it's gone.
  If dims ever look wrong, check you have the 28.9 MB file.

## 3. Code architecture (all rewritten for 5 drones)

Everything under `src/px4med/`:

- `fgcs_policy.py` — importlib wrapper; `FGCSPolicy.select_actions(obs) -> [5 ints]`.
- `fgcs_state.py` — builds the model's dict observation from `WorldEnvironment`.
  Feature indices documented in its docstring; matches models/README.md.
- `environment.py` — world model mirroring training: 100×100 grid, 50 patient
  slots (20 initial, dynamic spawn budget 30 total @ every 40 steps —
  **guessed params**, see §7), logistic patient decay, hazard rectangles,
  energy ledger, Dijkstra expected-energy safe-return maps, per-drone
  energy-return phase, action masks, no early termination on drone death.
  Also: already-overlapping drones (telemetry drift) resolve by lower-index
  moves / higher yields (prevents a deadlock).
- `coordinator.py` — episode driver: telemetry sync (clamped to grid bounds
  — critical, see §5), obs → policy → world.step → waypoint dispatch at
  2 Hz; jump validator with one re-sample.
- `drone.py` — MAVSDK wrapper. Arm retries (90 s), stream-rate re-requests,
  battery/landed reads have cached fallback + 30 s backoff after stalls
  (position reads stay strict). Health wait falls back to verifying a
  stable `position_velocity_ned` stream. `SDLOG_MODE=-1` disables ULog.
- `boot.py` — the hardened boot: `check_ports_free()`, `settle_and_gate()`
  (settle `PX4_ATTACH_SETTLE_S` default 120 s → per-drone ephemeral probe
  subprocess (scripts/probe_health.py) → on failure: dump PX4 internals,
  restart that instance, re-probe).
- `docker_manager.py` — N-drone container lifecycle; `instance_diagnostics(i)`
  (px4-ekf2 status via daemon socket), `restart_instance(i)` (kills by
  binary name via /proc, relaunches attaching to existing gz model via
  `PX4_GZ_MODEL_NAME`); deletes instance logs on stop.
- `baselines.py` — priority_path / nearest_path / random for 5 drones with
  distinct target claiming + safe-return (same interface as FGCSPolicy).
- `episode_budget.py` — THE step budget (800 mission + 400 landing grace,
  loop cap 1200). Import-light on purpose so main.py can read it at argparse
  time. Must match the training module; see §6.
- `experiments.py` — suites (baseline_comparison, hazard_sweep,
  battery_sweep), EpisodeResult/StepResult schemas, CSV writers,
  mean+CI95 summaries, `run_offline_episode()` (no SITL) and
  `run_sitl_episode()` backends.
- `metrics.py` — jsonlines step/episode records (per-drone lists).
- `main.py` — single/multi-episode CLI (`poetry run px4med`). NOTE:
  multi-episode in ONE container is broken (re-arm denied after depletion
  landings); use the runner for real experiments (fresh container/episode).
- `docker/start_multi.sh` — no-`make` launch: gz server started directly
  (needs `GZ_SIM_SERVER_CONFIG_PATH` — see §5), private `GZ_PARTITION`,
  all 5 PX4 instances standalone with `px4 -d`. PX4 stdout discarded unless
  `PX4_VERBOSE_LOGS=1`.
- `scripts/run_overnight_validation.py` — THE experiment runner.
  Parent/worker: per attempt boots fresh container + gate, worker runs one
  episode, heartbeat watchdog (kill on episode timeout 60 min default /
  300 s stale heartbeat), 3 attempts, resumable via same `--output-dir`,
  incremental `tables/{episodes,summary,steps}.csv`. Plans: `pilot`
  (4 jobs) and `core` (24 jobs: 6 learned + 5 priority + 5 nearest + 2 random
  nominal, then 3+3 battery_60).
- `scripts/detect_inert_drones.py` — post-hoc data-integrity check for the
  §5c inert-drone fault, from `tables/steps.csv`. ALWAYS run this before
  believing a run's aggregates.
- `scripts/make_paper_figures.py` — figures + `analysis.md` + `table_main.tex`
  from a run's `tables/`. Safe on a PARTIAL run (missing policies/metrics are
  skipped, not fatal), so it can be run mid-suite. Palette = validated
  categorical slots 1-4 by policy identity, hatching as secondary encoding for
  greyscale print, value labels double as the contrast relief.
  `--offline-dir` adds the transfer figure.
- `scripts/run_offline_companion.py` — the offline TWIN of a SITL run: reads
  its `plan.csv` and replays every job at the identical (suite, scenario,
  policy, seed) with no PX4, into a parallel dir with the same `tables/`
  layout. This is what makes the transfer comparison paired rather than
  two unrelated samples. `--only-completed` restricts it to jobs SITL actually
  finished. A few minutes of CPU for 24 jobs — but do NOT run it while a SITL
  fleet is booting.
- `scripts/offline_rollout.py` (now drives TrueWorld; it used to drive the
  dead reconstructed `environment.py` and reported off-distribution garbage
  — ~10/50 delivered — which reads like a model regression but isn't),
  `scripts/probe_directions.py`,
  `scripts/probe_health.py`, `scripts/sniff_mavlink.py` — diagnostics.
- Tests: `tests/` — 30 pass (`PYTHONPATH=src poetry run pytest`).

`results/` and `logs/` are symlinks to `/mnt/data/px4-med-runs/` (root disk
is at 96%; keep bulky outputs on /mnt/data).

## 4. Environments

**Local laptop (primary experiment host; 12 cores, 22 GB RAM, Arch):**
- Python via poetry venv (py3.14) WITH system-site-packages: torch comes
  from the SYSTEM (`python-pytorch` 2.12); torch/numpy must NOT be
  installed in the venv (see §5 crash history). `poetry run` for everything.
- Docker image `zackmurry/dronevalkit-sim:latest` local.

**dvk-server (secondary lane; `ssh dvk-server`, root@178.104.184.37):**
- 4 cores, 7.6 GB RAM. Minecraft server was stopped to free RAM (user did
  this; don't restart it, but don't assume RAM stays free either).
- Repo at `/root/px4-med`, venv `.venv` (pip: CPU torch 2.13, mavsdk,
  numpy, pytest). All current code synced. Boot stack NOT yet re-validated
  there since the final fixes; treat as available but unproven.
- Deploy with `rsync -az --exclude .git --exclude __pycache__ ... /home/zack/px4-med/ dvk-server:/root/px4-med/`.

## 5. Hard-won root causes (DO NOT re-learn these)

1. **`PX4_SIM_SPEED_FACTOR` must never default to a value.** Passing it
   (even `=1`) makes each PX4 instance's rc call gz `set_physics` during
   boot, which randomly and permanently corrupts sibling instances'
   MAVSDK-visible telemetry (PX4 EKF itself stays healthy — verified via
   `px4-ekf2 status`). DockerManager passes it only if explicitly set.
2. **Gazebo needs `GZ_SIM_SERVER_CONFIG_PATH`** =
   `$PX4_ROOT/Tools/simulation/gz/server.config` when launched directly;
   otherwise NO sensor systems load — topics exist but carry no data,
   preflight says "Accel Sensor 0 missing" forever.
3. **`--network host` containers share the host's hostname**, and gz
   transport partitions on hostname → stale/concurrent sim containers
   cross-talk. `start_multi.sh` sets a private `GZ_PARTITION`. Any manual
   `gz topic` debugging must set the same partition (read it from
   `/proc/<gz pid>/environ`), else you get empty/stale listings.
4. **`px4 -d` (daemon mode) is mandatory**: without it the pxh console
   busy-loops on /dev/null stdin — ~1 core each and 450 MB of log spam per
   instance in minutes (this was the historical disk-filler too).
5. **Never install torch/numpy/nvidia packages into the poetry venv.**
   An aborted CUDA torch install left truncated `nvidia_*_cu12` libs that
   system torch ctypes-loaded → bus error → silent process death. Venv must
   get numpy+torch from system-site-packages only.
6. **Telemetry sync must clamp to grid bounds** (done in coordinator):
   boundary braking overshoot otherwise creates a runaway waypoint
   feedback loop off the map.
7. **`pkill -f PATTERN` self-matches** the invoking shell/script cmdline —
   always bracket-class patterns (`"px4med.mai[n]"`), including inside
   docker-exec'd scripts (where the *launch line later in the same script*
   can be the self-match).
8. **Orphaned `mavsdk_server` processes** (from killed/crashed runs) hold
   udp 14540-14544 and silently eat PX4 telemetry for the next run.
   `boot.check_ports_free()` fails fast; cleanup:
   `pkill -9 -f "mavsdk/bi[n]"`. main.py also has a hard-exit watchdog
   that reaps children.
9. Python f-strings written through layered bash heredocs lose backslashes
   (`tr "\0"` became a literal NUL → "embedded null byte" crash). Prefer
   the Edit/Write tools over heredoc patchers for tricky strings.

## 5b. Known intermittent failure: arm COMMAND_DENIED (diagnosed 2026-08-29 23:15)

One boot in roughly several comes up with a single drone whose health reads
`global=False home=False` (local position fine). `drone.py`'s health wait sees
the flags stuck, logs "health flags frozen — falling back to direct
position-stream verification", and passes the gate on the local position
stream alone. PX4 then refuses to arm that drone forever (no home position),
so `arm()` burns its full 90 s retry window and the attempt dies with
`mavsdk.action.ActionError: COMMAND_DENIED`. The other four drones arm on
attempt 1.

Observed: job 1 of `core_20260829_230535`, attempt 1 (drone 0). The runner's
3-attempt retry recovered it on a fresh container. Cost ≈ 9 min per
occurrence (boot + gate + 90 s arm + teardown).

**Deliberately NOT fixed mid-run** — the parent process has `boot.py` loaded
at startup, so a gate change needs a runner restart, and the frozen-flags
fallback exists to work around the documented MAVSDK frozen-view bug (§5),
so tightening it risks failing EVERY boot instead of one in several.

Follow-up worth doing when the runner is idle: have `settle_and_gate()` also
require a home/global position (or an explicit arm dry-run) per drone, and on
failure use the existing `DockerManager.restart_instance(i)` + re-probe path
that already handles probe failures. That converts a wasted 9-minute attempt
into a ~1-minute instance restart. Validate over several boots before
trusting it overnight.

### 5b.0 WHY it started happening: host contention (2026-08-30 00:19)

The arm-denied fault was NOT present in the 21:52 and 22:10 runs and then hit
four consecutive boots (drone 0, drone 0, drones 1+2, drone 3). The trigger was
external: at ~23:32 an unrelated container stack came up on this host —
`openreel-admin`, `openreel-feedgen`, `openreel-appview`,
`ghcr.io/bluesky-social/pds:0.4`, `postgres:16.9`, `redis:7.4`. Load average
went to 7.08 (12 cores) and SITL step time went from 1.7-2.2 s to ~3.4 s.

This is the documented sensitivity from §5 / the memory file: heavy concurrent
work during SITL boot starves EKF convergence. Consequences to expect while
that stack is up:
- boots need more settle time, hence `PX4_ATTACH_SETTLE_S=240` (from 120) on
  the 00:20 relaunch;
- episodes take ~1.5x longer, so a 24-job core plan does NOT fit in ~10 h.
  Plan order is fortunate: baseline_comparison is jobs 1-18 and battery_sweep
  is 19-24, so a short night loses the less important suite.

**RESOLVED 2026-08-30 00:32: the user stopped that stack themselves.** Host
is back to just the sim container + rabbitmq, so the run was relaunched with
the DEFAULT 120 s settle (saves ~2 min/job) and `--max-hours 30`, i.e. no
practical cap — the user explicitly said not to time-box it.

Lesson for future runs: if boot timings or arm failures look wrong, check
`docker ps` and `uptime` BEFORE suspecting the PX4 stack. This cost several
failed attempts before I thought to look outside the project.

### 5b.1 GATE FIX for arm-denied (2026-08-30 00:16)

Root cause located: `scripts/probe_health.py` returned success on
`is_local_position_ok` ALONE, so the convergence gate happily passed an
instance whose MAVSDK view had `global=False home=False`. PX4 then denied
arming and the attempt died ~18 min later.

Investigated inside the live container: PX4 itself is entirely healthy on the
affected instance — `px4-ekf2 status` gives `attitude: 1, local position: 1,
global position: 1`, `vehicle_gps_position` is publishing, and
`home_position` is present, IDENTICAL to a working instance. So this is the
MAVSDK-view fault of §5, not a PX4 or Gazebo sensor problem. (`px4-commander
check` prints "Preflight check: FAILED" on ALL five instances during settle,
including ones that go on to arm fine — it is NOT a usable discriminator, do
not chase it.)

The probe now requires local AND global AND home position. A healthy drone
satisfies all three within ~1 s; an affected one never does. Failure routes
into the gate's existing `restart_instance()` + re-probe recovery, i.e. a
~1-minute instance restart before the worker starts, instead of an 18-minute
dead attempt. `--allow-local-only` restores the old behaviour if global/home
ever proves legitimately unavailable.

**Deploys with no runner restart** — `boot._probe()` spawns the script as a
subprocess, so edits take effect at the next gate. (Contrast the §5c
coordinator guard, which is imported and did need a restart.)

`health.is_armable` also exists and is the most direct arming signal; the
probe now LOGS it but deliberately does not gate on it, because we have no
evidence yet that it ever goes true in this SITL setup and gating on a signal
that never fires would fail every boot. Check the probe output in a few runs;
if `armable=True` shows up reliably on healthy drones, gate on that instead.

## 5c. Known intermittent failure: INERT DRONE (found 2026-08-29 23:36)

Worse than §5b because it is silent and corrupts data rather than failing the
attempt. A drone arms, reports "airborne at 20.0 m AGL", logs "local position
+ home position ready", passes the convergence gate — and then its telemetry
reports the map ORIGIN for the entire episode. The coordinator syncs the world
from telemetry (clamped to grid bounds), so the env believes that drone is
parked in the (0,0) corner. It never moves, never delivers, and can never
reach its landing pad.

Observed: `core_20260829_230535` job 1 attempt 2, drone 0 — commanded to
reposition to grid (29, 80), read (1,0) then (0,1) for 269+ steps while being
commanded east then south. The other four drones still delivered 46/50, which
is a nice robustness anecdote but NOT a valid 5-drone measurement:
`mission_success` and `all_landed` are forced to 0 by the inert drone.

**GUARD ADDED 2026-08-29 23:54** (`coordinator.find_stranded_drones` +
`InertDroneError`, pinned by `tests/test_inert_drone_guard.py`): the
reposition step already knew — it logged
`expected=[(29,80),(11,49),...] actual=[(0,0),(11,49),...]` and then continued
anyway. Now, if any drone is still more than `_INERT_START_MAX_CELLS` (12)
from its assigned start after the settle timeout, the episode raises instead
of proceeding, so the runner retries on a fresh container. Healthy drones
settle on the EXACT cell (4 of 5 did in the observed failure) and the fault
sat 109 cells out, so the threshold is far from both. This turns a 30-minute
contaminated episode into an ~8-minute failed attempt AND keeps corrupt data
out of the CSVs entirely. Note the guard can only fail the attempt; a better
fix would restart just that instance via `DockerManager.restart_instance(i)`,
but the worker does not own the container.

**Detection is also post-hoc and needs no code change to a running suite**,
because `tables/steps.csv` records every drone's grid cell every step:

```bash
poetry run python scripts/detect_inert_drones.py --run-dir results/core_...
poetry run python scripts/detect_inert_drones.py --run-dir R --print-rerun
```

A drone counts as inert when it occupies <= 2 distinct cells all episode; a
drone that flew and then parked on its pad is not flagged (only pre-parking
movement is judged). Verified clean on the offline twin and on the two
`SUPERSEDED_oldschema_*` SITL episodes, so the fault is intermittent, not
universal — as of 23:40 the incidence is 1 of 3 SITL episodes attempted.

Remediation for a contaminated job: delete its `jobs/<job_id>/` directory (or
just its `status.json`) and re-run the runner with the same `--output-dir`;
resume skips only jobs whose status is `completed`.

Root cause is NOT yet established. It is presumably the same MAVSDK
frozen-view family as §5, but note this variant reports position data
(the origin) rather than no data, which is why the gate's position-stream
check passes. Worth probing whether the drone is physically at home in Gazebo
(i.e. offboard setpoints never took effect) or actually flying while telemetry
lies — `gz topic` with the sim's GZ_PARTITION (§5.3) would settle it.

## 5d. GATE PROBE COULD HANG FOREVER — fixed 2026-08-30 20:44 (cost 6 h)

**The single most expensive bug so far.** The hazard sweep stalled from 14:47
to 20:41 — six hours — completing zero jobs while Gazebo burned 83% CPU.

Mechanism: `scripts/probe_health.py` waited for connection with
`async for state in s.core.connection_state()` and no timeout. Its
`mavsdk_server` child died (visible as `<defunct>`), so that stream never
yielded and the probe blocked forever. `boot._probe()` did a bare
`await proc.wait()`, so the gate blocked with it. The `--timeout` argument
only ever bounded the health loop AFTER connection, and the runner's heartbeat
watchdog covers WORKER processes — nothing watched the parent's gate phase.
Pre-existing latent bug; my §5b.1 probe change did not introduce it.

Fixed in two layers:
1. `boot._probe()` wraps `proc.wait()` in `asyncio.wait_for(budget + 60)`,
   kills the probe on timeout and returns failure, so the gate's existing
   `restart_instance()` recovery runs. **This is the layer that actually
   fires** — verified against a dead port: rc=1 after 80 s, process reaped.
2. `probe_health.py` bounds its connect phase too. Note this did NOT fire in
   testing, because `await s.connect(...)` itself can block before the loop is
   reached — which is exactly why layer 1 is the load-bearing one.

**A hang emits no log lines, so a log-grep monitor never fires.** That is why
this ran for 6 h. Going forward, watch file mtimes, not log content:

```bash
/tmp/px4med_chain_stall.sh 25    # minutes; exits nonzero on stall
```

**Two bugs made the first stall detector silently useless** (found 2026-08-31
when it fired a false alarm on a perfectly healthy run):
1. `results` is a SYMLINK to `/mnt/data/px4-med-runs/results`, and plain
   `find results -type f` does NOT traverse a symlink argument — it matched
   0 files. Needs `find -L`.
2. `-newermt '-1200 seconds'` matches nothing in this environment (verified:
   `-newermt '-300 seconds'` and `'5 minutes ago'` both return 0 while
   `-mmin -5` and `-newermt "@<epoch>"` correctly return 4). Use `-mmin`.

Both failure modes produce FALSE stalls rather than missed ones, so the run
was never at risk — but the detector was checking nothing. If you write any
similar watcher, verify its predicate returns >0 on a known-healthy run before
trusting it.

It alerts when NOTHING in the run dir or the parent log has been modified for
20 min. Neither signal alone works: during an episode the parent log is silent
(steps go to `worker.log`) while `heartbeat.json` updates every step; during
boot the run dir is silent while the parent log gets gate lines. The max of
the two is never stale beyond ~10 min (worst legitimate case: an
instance-restart boot). **Arm this alongside the log monitor for every long
run.**

## 6. Current state — TRUE-WORLD stack (major update)

**The collaborator's real training script arrived**: `models/CEDA-FGCS-new.py`
(12,324 lines — full Environment, curriculum, rewards, training loop). This
replaced all my reconstructed world guesses and closed the performance
mystery: **the model is excellent; my approximated world was off-distribution.**

- Native env, offline, seed 0: **48/50 delivered, triage 0.962,
  mission_success at step 335, 5/5 landed.**
- Offline baselines in the true world (seed 11): learned 0.954 triage +
  only policy with mission success; priority_path 0.574; nearest_path 0.954
  but fails to land all. The learned policy now clearly wins — paper story
  flipped from the old 2-drone iteration.
- **SITL validation episode (true world): 35/50 delivered, triage 0.72,
  4/5 landed on pads, +196 total reward.** The offline→SITL gap
  (48→35) is a genuine sim-transfer effect (flight time + tracking jitter
  consume patient timers) — a paper finding, not a bug.

**Architecture change:** `src/px4med/true_world.py` wraps the training
script's `Environment` directly (importlib, `CEDA_HEADLESS=1`,
`Environment(fixed_layout=False)`, `reset(curriculum_stage=2)` = final
stage). Coordinator/experiments/main all consume it via the same duck-typed
interface. `environment.py` and `fgcs_state.py` are now DEAD CODE (kept for
reference; their unit tests still pass but test the dead module).
Observations come from his `get_state()`; transitions/rewards/masks/
termination from his `step()` → `(next_states, rewards, done, step_data)`.

TrueWorld adapter gotchas (all handled, don't re-break):
- env's `patients_delivered` flag means RESOLVED (died patients also set it);
  true deliveries = delivered AND not died. Authoritative counts:
  `env.mission_outcome_metrics()`.
- step_data: deliveries in `patient_delivery_events`; `land_actions` is a
  per-step COUNT and carries no success info — **every** ACTION_LAND emits
  exactly one `landing_events` entry (post-death lands included, with
  `successful=False`), so `land_actions == len(landing_events)` always.
  Wrong-lands = landing_events with `successful=False`; per-agent
  attempts/successes come from each event's `agent`/`successful` fields.
  (Fixed 2026-08-29 in `true_world._landing_outcomes`; the earlier
  `land_actions − len(landing_events)` formula was always 0, and feeding the
  scalar count through `_per_agent_bool` aliased it as an agent index, which
  made offline `wrong_land_attempts` ≈ step count. `_per_agent_bool` is
  gone.)
- Start positions and landing pads are RANDOMIZED mid-map per episode
  (reposition timeout raised to 45 s to cover ~200 m flights).
- Telemetry sync writes `env.agents` (clamped to grid).
- Episodes: 800 mission steps + 400 landing-grace, defined ONCE in
  `src/px4med/episode_budget.py` and pinned to the training module by
  `tests/test_episode_budget.py`. Successful learned episodes self-terminate
  ~310-450 steps. Wall time ~20-35 min/job incl. boot.
  **Why this is load-bearing (bug found+fixed 2026-08-29 22:00):** the suites
  had been running a 500-step budget. Mission observation feature 0 is
  `min(1.0, episode_step / current_episode_deadline())`, so a 500 budget makes
  the policy perceive the mission clock running 1.6x faster than in training —
  off-distribution and invisible in the logs. Separately the driver loop was
  capped at the mission budget, which truncates episodes that resolve late
  before their 400-step landing grace elapses (under-reporting `all_landed` /
  `mission_success`). The env self-terminates by itself (`rescue_timeout` at
  the mission budget, `landing_timeout` at `resolution_step + grace`), so the
  loop cap is now `MISSION_MAX_STEPS + LANDING_GRACE_STEPS` = 1200 and the env
  decides. main.py had the inverse mismatch (loop 500 / env deadline 800) and
  is fixed too — so the earlier one-off SITL number (35/50, triage 0.72) was
  measured under a truncated budget and should NOT be quoted; the relaunched
  core run supersedes it.
  Offline at the corrected budget (seed 0): learned 48/50 triage 0.962 5/5
  landed mission-success @335; nearest_path 49/50 triage 0.981 but 3/5 landed;
  priority_path 17/50 triage 0.337; random 2/50 triage 0.048.
- Hazard sweep suite is DEFERRED (no knob mapping into his env yet);
  suites currently: baseline_comparison (nominal) + battery_sweep
  (battery_60 via post-reset battery override).

**Runner validated live**: old-world pilot ran 4 jobs cleanly (14-21 min
each) incl. one automatic retry (battery-stall timeout, then fixed via
read-backoff). A 24-job core plan was launched on the true-world stack at
21:38 and then STOPPED at the user's request (hand off to a fresh session
before overnight runs). `results/core_20260829_213852` was deleted;
old-world pilot data in `results/pilot_20260829_185954` is off-distribution
— do not use for the paper.

**ALL FOUR CAMPAIGNS COMPLETE (2026-08-29 → 2026-08-31).**

### >>> READ `RESULTS.md` IN THE REPO ROOT FIRST <<<

94 SITL episodes, **0 contaminated**, 1 job abandoned (env bug §6c).
Campaigns: `core_20260829_230535` (24), `hazard_20260830_104103` (32),
`extend_20260831_000000` (27), `latency_20260831_000000` (11), plus
`offline_twin_core_20260829_230535` (24 paired) and
`offline_sweep_baseline_comparison_20260831_211720` (200, n=50/policy).
30 unit tests pass.

**The headline is methodological.** Learned vs nearest_path on delivery:
abstract env at n=50/arm gives +0.017 [-0.003,+0.038] p=0.102 (NOT
significant); PX4 SITL at n=19/arm gives +0.063 [+0.012,+0.114] p=0.016
(significant). The offline comparison has 2.6x MORE data and ~3x tighter
intervals and still sees nothing — so this is not a power artifact, the
abstract env structurally cannot see the difference. Evaluating only in the
training environment would have concluded the learned policy offers no
significant benefit over a greedy heuristic.

Everything else — the full per-campaign tables, the transfer analysis, the
hazard and latency sweeps, and all interpretation traps — is in `RESULTS.md`.

Re-run or extend with the same command (resumable via the same
`--output-dir`; completed jobs are skipped):

```bash
cd /home/zack/px4-med
OUT="results/core_$(date +%Y%m%d_%H%M%S)"   # or reuse a dir to resume
nohup env PYTHONPATH=src poetry run python -u scripts/run_overnight_validation.py \
  --plan core --output-dir "$OUT" --max-hours 30 > core_run.log 2>&1 &
```

Monitor `core_run.log` (parent: boot/job/retry lines) and
`$OUT/jobs/<job_id>/attempts/attempt_NN/worker.log` (per-step episode detail —
the parent log does NOT carry step output). The runner self-retries
(3 attempts); intervene only on systematic failures.

**NEXT STEPS (in priority order, 2026-08-30 morning):**
1. Send the collaborator `FINDINGS.md` + his open questions from §7 (his
   eval-time numbers; confirmation that stage 2 is the intended eval config).
   The training-vs-offline-vs-SITL table needs his numbers to be complete.
2. If a reviewer would press on delivery (learned 0.697 vs nearest_path
   0.656, overlapping at n=5-6), extend ONLY those two arms. Cheapest
   decisive experiment: ~10 more episodes each ≈ 6 h on an idle host.
   The hazard difference is already resolved many times over — no more data
   needed there.
3. Consider the hazard_sweep suite (still DEFERRED, no knob mapping into his
   env). Given hazard avoidance is the paper's differentiator, an explicit
   hazard-density sweep is now the highest-value new experiment.
4. Optional: a labelled off-distribution battery arm (e.g. battery_140) as a
   sensitivity check. Keep nominal at 100 for headline numbers — the ledger is
   part of the trained contract (see §"battery" reasoning).

Older follow-ups: review
`$OUT/tables/summary.csv`, consider adding hazard-sweep knobs to TrueWorld,
and port figure generation (see `/mnt/data/px4-med/scripts/make_paper_core_12h_figures.py`
for the old paper's pgfplots pipeline).

## 6b. Follow-up experiment queue (2026-08-30, deadline extended)

User approved three follow-ups; running sequentially on the idle host. Launch
the next when the monitor reports "Plan finished." for the previous.

| # | plan | jobs | est. | status |
|---|---|---|---|---|
| 1 | `--plan hazard` | 32 | ~10.5 h | RUNNING from 10:41, `results/hazard_20260830_104103`, log `hazard_run.log` |
| 2 | `--plan extend` | 27 | ~9 h | queued |
| 3 | `--plan latency` | 12 | ~4 h | queued |
| 4 | offline sweep n=50 | — | ~1.5-2.5 h CPU | queued, run LAST (no SITL concurrently) |

```bash
# 2. resolve the learned vs nearest_path delivery difference (own output dir!)
nohup env PYTHONPATH=src poetry run python -u scripts/run_overnight_validation.py \
  --plan extend --output-dir "results/extend_$(date +%Y%m%d_%H%M%S)" \
  --max-hours 30 > extend_run.log 2>&1 &

# 3. command-latency robustness
nohup env PYTHONPATH=src poetry run python -u scripts/run_overnight_validation.py \
  --plan latency --output-dir "results/latency_$(date +%Y%m%d_%H%M%S)" \
  --max-hours 30 > latency_run.log 2>&1 &

# 4. large-n offline (CPU only; nearest_path is ~75 s/episode)
poetry run python scripts/run_offline_sweep.py --episodes 50 \
  --suite baseline_comparison --scenario nominal
```

**New knobs/suites added for these:**
- `TrueWorld` now accepts `{"hazard": {"fraction": X}}`, applied post-reset
  exactly like the battery override. The env scales its rectangle counts by
  `hazard_fraction` (`NUM_WIND_ZONE_RECTANGLES=6`,
  `NUM_LOW_SIGNAL_ZONE_RECTANGLES=5`) and re-reads the attribute on every
  periodic regeneration, so one forced regeneration at reset holds for the
  episode. Verified: 0.5→(3,2) rects, 1.0→(6,5), 2.0→(12,9); default
  untouched. `hazard_rectangle_counts` exposes the counts for provenance.
  **Training used 0.5 (stage 0) and 1.0 (stages 1-2), so 1.5 and 2.0 are
  off-distribution extrapolation and must be labelled as such.**
- `hazard_sweep` suite: densities 0.5/1.0/1.5/2.0 x {learned, nearest_path},
  4 episodes each. 1.0 is included rather than borrowed from the core run so
  the curve is self-contained rather than spliced across suites. Jobs are
  interleaved by density so a truncated run still yields a curve at every
  density.
- `latency_sweep` suite: `action_delay_steps` 1/2/4 (already plumbed through
  ScenarioDef → Coordinator action queues) x learned. Training had zero delay,
  so all points are off-distribution robustness checks.
- `extend` plan: +13 learned, +14 nearest_path. Power calc on the core effect
  (delta 0.041, pooled sigma ~0.045) → n~=19/arm for 80% power. Episode
  indices start at 100 via the new `_expand(episode_offset=)` so they cannot
  collide with core's ep000-ep005 when the two episode sets are concatenated.
  **Must use its own `--output-dir`** — resume keys on job_id.
- `scripts/run_offline_sweep.py`: arbitrary (suite, scenario, policy) cells at
  any n, distinct from `run_offline_companion.py` (which mirrors a specific
  SITL plan at matched seeds for paired comparison).

Offline preview of the hazard sweep (n=1/cell, so indicative only) — the
learned policy holds wind avoidance 0.94-1.00 with 0-2 steps of wind exposure
across all four densities, while nearest_path sits at 0.41-0.61 with 66-100
exposure steps. `wind_exposure_steps` / `low_signal_exposure_steps` are a more
intuitive companion to the avoidance rate ("time spent in hazardous airspace")
and are already columns in `episodes.csv`.

## 6c. BUG IN THE COLLABORATOR'S ENV — float32 reward invariant (2026-08-31)

`models/CEDA-FGCS-new.py:3214` raises
`RuntimeError('Per-agent rewards do not sum to team reward')` during
`env.step()`. First seen in the latency sweep at `delay_4`, after ~90 clean
episodes.

It is a floating-point tolerance bug, not a logic error:

```python
local_rewards = np.zeros(NUM_AGENTS, dtype=np.float32)   # line 2113
...
unattributed_reward = team_reward - float(local_rewards.sum())
local_rewards += unattributed_reward / NUM_AGENTS        # makes it exact...
if not np.isclose(local_rewards.sum(), team_reward, atol=1e-5):  # ...then checks
    raise RuntimeError(...)
```

The redistribution makes the identity hold by construction, so the check can
only fail on rounding. But `local_rewards` is float32: spacing at |value|~1e3
is 6.1e-05 and at 1e4 is 9.8e-04 — 6x and 100x the fixed `atol=1e-5`. So the
assertion is guaranteed to trip once |team_reward| grows past a few hundred.
The check is PER STEP, so what matters is per-step team reward, not the
episode total — which makes it stochastic: each large-penalty step has a
chance to trip, and over ~500 steps the odds compound. Observed episode
totals: delay_1 ~-7k, delay_2 ~-12k, delay_4 ~-34k; `delay_4` ep000 completed
fine while ep001 failed twice, exactly the coin-flip this predicts. Any
severe-failure regime is affected, not just latency.

**Do NOT patch his script to work around this.** Running his exact training
environment is what makes the whole validation credible; a local edit would
forfeit that. Let the runner retry, and report affected cells as partial.

Suggested fix FOR HIM (one line): use a relative tolerance or float64, e.g.
`np.isclose(local_rewards.sum(), team_reward, rtol=1e-5, atol=1e-4)` or
accumulate `local_rewards` in float64. Ask him to confirm before we re-run any
affected cell.

## 7. Collaborator status

RESOLVED: the training script (`models/CEDA-FGCS-new.py`) contains the full
environment — no more guessed parameters. Remaining asks (non-urgent):
- his eval-time performance numbers for the final checkpoint (for a
  training-vs-SITL comparison table);
- confirmation that `curriculum_stage=2` ("full_5_drone_50_patient_100x100_
  cross_layer") is the intended evaluation configuration;
- (later) preferred hazard-sweep knobs if we want that suite — NOTE we have
  since found `hazard_fraction` ourselves and used it (§6b); worth confirming
  he considers that the right knob.
- **the float32 reward-invariant bug in §6c** — this one blocks off-distribution
  latency data, so it is the most actionable ask.

## 8. Quick command reference

```bash
# unit tests
PYTHONPATH=src poetry run pytest tests/ -q
# offline sanity (no SITL)
poetry run python scripts/offline_rollout.py --episodes 1 --policy learned
poetry run python scripts/probe_directions.py
# one SITL episode
env PYTHONPATH=src poetry run python -u -m px4med.main --episodes 1 --max-steps 500
# experiment suite (fresh container per episode, resumable)
env PYTHONPATH=src poetry run python -u scripts/run_overnight_validation.py --plan pilot
# full cleanup between runs
pkill -9 -f "px4med.mai[n]"; pkill -9 -f "mavsdk/bi[n]"
docker ps -q --filter ancestor=zackmurry/dronevalkit-sim:latest | xargs -r docker stop
# verify model package
python3 models/CEDA-FGCS.py --device cpu --show-metadata --smoke-test
```

## 9. Constraints from the user

- No git commits/pushes unless asked.
- One heavy job per host (don't run offline sweeps during SITL boot).
- Storage-heavy outputs go to /mnt/data (results/ and logs/ symlinks).
- The old 2-drone reference material: `/mnt/data/px4-med` (archive),
  `/mnt/data/px4-med-results` (orphan run), `AneeshMARL5.py`/`train.py`
  in repo root (old training code, kept for reference).
