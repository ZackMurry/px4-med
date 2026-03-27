#!/usr/bin/env bash
# Launch exactly 2 PX4 SITL instances inside one container.
#
# Instance 0 starts Gazebo and PX4; instance 1 attaches to that Gazebo with
# PX4_GZ_STANDALONE=1.  A sleep between the two gives Gazebo time to fully
# initialise before instance 1 tries to connect.
#
# Port assignments (offsets from PX4_BASE_INSTANCE, default 0):
#   Instance 0: MAVLink API UDP 14540, simulator TCP 4560, MAV_SYS_ID 1
#   Instance 1: MAVLink API UDP 14541, simulator TCP 4561, MAV_SYS_ID 2
#
# Environment variables:
#   PX4_BASE_INSTANCE  base instance number (default 0)
#   DRONE_MODEL        Gazebo model name    (default gz_x500)
#   GZ_STARTUP_WAIT    seconds to wait for Gazebo before starting instance 1 (default 15)
set -euo pipefail

cd /root/PX4-Autopilot

BASE_INSTANCE="${PX4_BASE_INSTANCE:-0}"
DRONE_MODEL="${DRONE_MODEL:-gz_x500}"
GZ_STARTUP_WAIT="${GZ_STARTUP_WAIT:-15}"
HOME_LAT="${PX4_HOME_LAT:-38.8983889}"
HOME_LON="${PX4_HOME_LON:--92.2156389}"
HOME_ALT="${PX4_HOME_ALT:-220.0}"
ROOTFS_LOG_DIR="/root/PX4-Autopilot/build/px4_sitl_default/rootfs/log"
PX4_BIN="/root/PX4-Autopilot/build/px4_sitl_default/bin/px4"

# Instance 0: starts Gazebo + first PX4
INSTANCE_0=$((BASE_INSTANCE + 0))
LOG_DIR_0="${ROOTFS_LOG_DIR}/instance_${INSTANCE_0}"
mkdir -p "${LOG_DIR_0}"
: >"${LOG_DIR_0}/px4_stdout.log"

# Ensure instance-0 runtime state from previous runs does not carry over.
# Keep log files on the mounted volume, but clear rootfs state under the build tree.
rm -rf "/root/PX4-Autopilot/build/px4_sitl_default/rootfs/instance_${INSTANCE_0}"

echo "[start_multi] Starting instance ${INSTANCE_0} (Gazebo + PX4) ..."
(
  PX4_INSTANCE="${INSTANCE_0}" \
  PX4_SYS_AUTOSTART=4001 \
  PX4_SIM_MODEL="${DRONE_MODEL}" \
  PX4_GZ_MODEL_POSE="0,0" \
  PX4_HOME_LAT="${HOME_LAT}" \
  PX4_HOME_LON="${HOME_LON}" \
  PX4_HOME_ALT="${HOME_ALT}" \
  HEADLESS=1 \
  make px4_sitl "${DRONE_MODEL}" \
    >>"${LOG_DIR_0}/px4_stdout.log" 2>&1
) &

# Wait for Gazebo to be ready
echo "[start_multi] Waiting ${GZ_STARTUP_WAIT}s for Gazebo to initialise ..."
sleep "${GZ_STARTUP_WAIT}"

# Instance 1: attaches to the running Gazebo
INSTANCE_1=$((BASE_INSTANCE + 1))
LOG_DIR_1="${ROOTFS_LOG_DIR}/instance_${INSTANCE_1}"
INSTANCE_DIR_1="/tmp/instance_${INSTANCE_1}"
mkdir -p "${LOG_DIR_1}"
rm -rf "${INSTANCE_DIR_1}"
mkdir -p "${INSTANCE_DIR_1}"
ln -sfn /root/PX4-Autopilot/build/px4_sitl_default/etc "${INSTANCE_DIR_1}/etc"
ln -sfn "${LOG_DIR_1}" "${INSTANCE_DIR_1}/log"
: >"${LOG_DIR_1}/px4_stdout.log"

echo "[start_multi] Starting instance ${INSTANCE_1} (standalone PX4, attaching to Gazebo) ..."
(
  PX4_SYS_AUTOSTART=4001 \
  PX4_SIM_MODEL="${DRONE_MODEL}" \
  PX4_GZ_MODEL_POSE="3,0" \
  PX4_GZ_STANDALONE=1 \
  PX4_HOME_LAT="${HOME_LAT}" \
  PX4_HOME_LON="${HOME_LON}" \
  PX4_HOME_ALT="${HOME_ALT}" \
  HEADLESS=1 \
  "${PX4_BIN}" -i "${INSTANCE_1}" -w "${INSTANCE_DIR_1}" \
    >>"${LOG_DIR_1}/px4_stdout.log" 2>&1
) &

echo "[start_multi] Both instances launched. Waiting ..."
wait
