#!/usr/bin/env bash
# Launch NUM_DRONES PX4 SITL instances inside one container.
#
# Architecture: a headless Gazebo server is started directly (no `make`, so no
# per-container ninja rebuild), then every PX4 instance runs the prebuilt px4
# binary with PX4_GZ_STANDALONE=1 and spawns its own model into that world.
#
# Port assignments (offsets from PX4_BASE_INSTANCE, default 0):
#   Instance i: MAVLink API UDP 14540+i, MAV_SYS_ID i+1
#
# Environment variables:
#   NUM_DRONES           number of PX4 instances (default 5)
#   PX4_BASE_INSTANCE    base instance number (default 0)
#   DRONE_MODEL          Gazebo model name    (default gz_x500)
#   GZ_WORLD             world SDF file       (default default.sdf)
#   GZ_READY_TIMEOUT     max seconds to wait for Gazebo (default 300)
#   INSTANCE_STAGGER     seconds between PX4 launches (default 2)
#   PX4_VERBOSE_LOGS     set to 1 to keep gz/PX4 stdout logs on the mounted
#                        volume (default: discard; the console spam previously
#                        grew to hundreds of MB per episode)
set -euo pipefail

PX4_ROOT=/root/PX4-Autopilot
cd "${PX4_ROOT}"

NUM_DRONES="${NUM_DRONES:-5}"
BASE_INSTANCE="${PX4_BASE_INSTANCE:-0}"
DRONE_MODEL="${DRONE_MODEL:-gz_x500}"
GZ_WORLD="${GZ_WORLD:-default.sdf}"
GZ_READY_TIMEOUT="${GZ_READY_TIMEOUT:-300}"
INSTANCE_STAGGER="${INSTANCE_STAGGER:-2}"
PX4_VERBOSE_LOGS="${PX4_VERBOSE_LOGS:-0}"
SIM_SPEED_FACTOR="${PX4_SIM_SPEED_FACTOR:-}"
HOME_LAT="${PX4_HOME_LAT:-38.8983889}"
HOME_LON="${PX4_HOME_LON:--92.2156389}"
HOME_ALT="${PX4_HOME_ALT:-220.0}"
ROOTFS_LOG_DIR="${PX4_ROOT}/build/px4_sitl_default/rootfs/log"
PX4_BIN="${PX4_ROOT}/build/px4_sitl_default/bin/px4"

# --network host makes every container share the host's hostname, and gz
# transport partitions on hostname by default — so concurrent/stale sim
# containers would all see each other. Use an explicit private partition.
export GZ_PARTITION="${GZ_PARTITION:-px4med_sim_$$}"
echo "[start_multi] GZ_PARTITION=${GZ_PARTITION}"
export GZ_SIM_RESOURCE_PATH="${PX4_ROOT}/Tools/simulation/gz/models:${PX4_ROOT}/Tools/simulation/gz/worlds"
# Without this config the gz server loads no sensor systems (imu, baro, mag,
# navsat) and PX4's EKF never receives data. PX4's make flow exports it too.
export GZ_SIM_SERVER_CONFIG_PATH="${PX4_ROOT}/Tools/simulation/gz/server.config"
# Opt-in sub/super-realtime lockstep. Only exported when explicitly requested:
# the rc's set_physics call is skipped entirely otherwise.
if [ -n "${SIM_SPEED_FACTOR}" ]; then
  export PX4_SIM_SPEED_FACTOR="${SIM_SPEED_FACTOR}"
fi

log_target() {
  # $1 = log file name; prints the redirect target path
  if [[ "${PX4_VERBOSE_LOGS}" == "1" ]]; then
    mkdir -p "${ROOTFS_LOG_DIR}"
    local path="${ROOTFS_LOG_DIR}/$1"
    : >"${path}"
    echo "${path}"
  else
    echo /dev/null
  fi
}

# ── Gazebo server ─────────────────────────────────────────────────────────────
GZ_LOG="$(log_target gz_server.log)"
echo "[start_multi] Starting headless Gazebo server (world=${GZ_WORLD}) ..."
gz sim -r -s "${PX4_ROOT}/Tools/simulation/gz/worlds/${GZ_WORLD}" \
  >"${GZ_LOG}" 2>&1 &

echo "[start_multi] Waiting for Gazebo world (timeout ${GZ_READY_TIMEOUT}s) ..."
waited=0
until gz topic -l 2>/dev/null | grep -q "/clock"; do
  sleep 2
  waited=$((waited + 2))
  if (( waited >= GZ_READY_TIMEOUT )); then
    echo "[start_multi] ERROR: Gazebo not ready after ${GZ_READY_TIMEOUT}s" >&2
    exit 1
  fi
done
echo "[start_multi] Gazebo ready after ~${waited}s."

# ── PX4 instances ─────────────────────────────────────────────────────────────
for ((i=0; i<NUM_DRONES; i++)); do
  INSTANCE=$((BASE_INSTANCE + i))
  X_OFFSET=$((i * 3))
  INSTANCE_DIR="/tmp/instance_${INSTANCE}"
  rm -rf "${INSTANCE_DIR}"
  mkdir -p "${INSTANCE_DIR}"
  ln -sfn "${PX4_ROOT}/build/px4_sitl_default/etc" "${INSTANCE_DIR}/etc"
  PX4_LOG="$(log_target "px4_${INSTANCE}.log")"

  echo "[start_multi] Starting PX4 instance ${INSTANCE} (pose ${X_OFFSET},0) ..."
  (
    PX4_SYS_AUTOSTART=4001 \
    PX4_SIM_MODEL="${DRONE_MODEL}" \
    PX4_GZ_MODEL_POSE="${X_OFFSET},0" \
    PX4_GZ_STANDALONE=1 \
    PX4_HOME_LAT="${HOME_LAT}" \
    PX4_HOME_LON="${HOME_LON}" \
    PX4_HOME_ALT="${HOME_ALT}" \
    HEADLESS=1 \
    "${PX4_BIN}" -d -i "${INSTANCE}" -w "${INSTANCE_DIR}" \
      >"${PX4_LOG}" 2>&1
  ) &
  sleep "${INSTANCE_STAGGER}"
done

echo "[start_multi] All ${NUM_DRONES} instances launched. Waiting ..."
wait
