#!/bin/bash
# Run the approved follow-up experiment queue back-to-back, unattended.
#
# Stages run strictly sequentially; each SITL stage gets a fresh container per
# job from the runner itself, and we scrub stray processes/containers between
# stages so one stage's leftovers can never degrade the next. The large-n
# offline sweep runs LAST because it is CPU-heavy and must not compete with a
# SITL boot (HANDOFF.md §5).
#
# Resumable by design: every SITL stage uses a fixed --output-dir, so re-running
# this script skips jobs already marked completed. Safe to kill and restart.
#
# Usage: nohup bash scripts/run_experiment_chain.sh > chain.log 2>&1 &
set -u

cd /home/zack/px4-med || exit 1
export PYTHONPATH=src

HAZARD_DIR="results/hazard_20260830_104103"          # resume the paused sweep
EXTEND_DIR="results/extend_20260831_000000"
LATENCY_DIR="results/latency_20260831_000000"

log() { echo "=== [chain $(date '+%Y-%m-%d %H:%M:%S')] $*"; }

scrub() {
  log "scrubbing between stages"
  pkill -9 -f "probe_healt[h]" 2>/dev/null
  pkill -9 -f "px4med.mai[n]" 2>/dev/null
  pkill -9 -f "mavsdk/bi[n]" 2>/dev/null
  sleep 2
  docker ps -q --filter ancestor=zackmurry/dronevalkit-sim:latest | xargs -r docker stop
  sleep 3
}

run_sitl_stage() {
  local plan="$1" outdir="$2" stagelog="$3"
  if [ -f "$outdir/.chain_done" ]; then
    log "stage $plan already marked done, skipping"
    return 0
  fi
  log "starting stage: $plan -> $outdir (log $stagelog)"
  poetry run python -u scripts/run_overnight_validation.py \
      --plan "$plan" --output-dir "$outdir" --max-hours 30 \
      > "$stagelog" 2>&1
  local rc=$?
  log "stage $plan exited rc=$rc; completed jobs: $(grep -c 'completed in' "$stagelog" 2>/dev/null)"
  # Mark done only if the runner reported finishing the plan, so an interrupted
  # stage is retried rather than silently skipped on the next invocation.
  if grep -q "Plan finished" "$stagelog" 2>/dev/null; then
    touch "$outdir/.chain_done"
    log "stage $plan marked complete"
  else
    log "stage $plan did NOT report 'Plan finished' — leaving unmarked for retry"
  fi
  scrub
  return 0
}

log "chain starting"
scrub

run_sitl_stage hazard  "$HAZARD_DIR"  hazard_run.log
run_sitl_stage extend  "$EXTEND_DIR"  extend_run.log
run_sitl_stage latency "$LATENCY_DIR" latency_run.log

# Large-n offline last: CPU-only, no SITL running by now.
if [ ! -f "results/.offline_sweep_done" ]; then
  log "starting offline sweep (n=50, baseline_comparison/nominal)"
  poetry run python -u scripts/run_offline_sweep.py \
      --episodes 50 --suite baseline_comparison --scenario nominal \
      > offline_sweep.log 2>&1
  log "offline sweep exited rc=$?"
  touch "results/.offline_sweep_done"
fi

log "chain complete"
