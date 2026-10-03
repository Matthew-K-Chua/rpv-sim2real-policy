#!/usr/bin/env bash
# =============================================================================
# The policy half: VLM model servers + the RPV policy server, in conda.
#
# The workstation runs BOTH halves of the stack: ROS (Python 3.10, rclpy) and
# this one (Python 3.12, the VLFM/SAM3 stack). They share a box, and the three
# things this script does beyond starting the servers exist because of that:
#
#   1. It scrubs ROS's environment before activating conda. Sourcing
#      /opt/ros/humble/setup.bash puts /opt/ros/humble/lib/python3.10/
#      site-packages on PYTHONPATH, and a conda Python 3.12 that inherits it
#      will import ROS's numpy/cv2 and fail in ways that look like a broken
#      install. This is the single most likely reason a first bring-up here
#      fails.
#   2. It sets NO_PROXY. If this machine has http_proxy/https_proxy set, requests
#      will happily try to proxy 127.0.0.1:13000 and the bridge's POST dies with
#      a connection error that names the proxy, not the policy.
#   3. It binds LOOPBACK, not 0.0.0.0. The bridge is on this host; there is
#      nothing to expose. Set RPV_POLICY_HOST=0.0.0.0 only if the GUI console
#      polls /status from another machine.
#
# NO ROS RUNS HERE. SLAM / Nav2 / the bridge run in the system ROS environment,
# started by rpv-ros2-bridge's scripts/launch_offboard.sh, which also calls this script.
#
# Usage:  ./scripts/launch_offboard_ai.sh
#   stop: tmux kill-session -t offboard_ai
#  timed: TIMED=1 ./scripts/launch_offboard_ai.sh   (drops per-step PNG/JPEG work)
# =============================================================================
set -u

SESSION=${SESSION:-offboard_ai}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=${REPO:-$(dirname "$SCRIPT_DIR")}
CONDA_ROOT=${CONDA_ROOT:-$HOME/miniconda3}
CONDA_ENV=${CONDA_ENV:-rpv-emb}
RPV_POLICY_PORT=${RPV_POLICY_PORT:-13000}
# Loopback: the bridge shares this host. See header note 3.
RPV_POLICY_HOST=${RPV_POLICY_HOST:-127.0.0.1}
CLIP_PORT=${CLIP_PORT:-12182}; SAM3_PORT=${SAM3_PORT:-12183}; YOLOE_PORT=${YOLOE_PORT:-12184}
YOLOE_CLASSES=${YOLOE_CLASSES:-data/shortvis.yaml}
SAM3_CONF=${SAM3_CONF:-0.35}
MAP_DEBUG=${MAP_DEBUG:-0}
SKIP_YOLOE=${SKIP_YOLOE:-1}
# Set to 0 to keep ROS's environment inside the conda windows. Only do this if
# you have a specific reason -- see header note 1.
SCRUB_ROS_ENV=${SCRUB_ROS_ENV:-1}
# REUSE_SESSION=1: attach these windows to a tmux session that ALREADY EXISTS
# instead of creating one and refusing if it is there. CEAI's
# rpv-ros2-bridge's launch_offboard.sh sets it so the whole workstation side of the
# run -- zenoh, SLAM/Nav2, these servers, ZoeDepth and the bridge -- lives in one
# session the operator console can start and stop as a unit. Default 0 keeps the
# old behaviour exactly: a stale session is still an error, because silently
# adding a second policy server to it is worse.
REUSE_SESSION=${REUSE_SESSION:-0}

fail() { echo "ERROR: $*" >&2; exit 1; }
command -v tmux >/dev/null || fail "tmux not installed"
[ -d "$REPO" ] || fail "repo not found: $REPO (set REPO=...)"
[ -f "$CONDA_ROOT/etc/profile.d/conda.sh" ] || fail "conda not found at $CONDA_ROOT (set CONDA_ROOT=...)"
[ -d "$CONDA_ROOT/envs/$CONDA_ENV" ] || echo "WARN: conda env '$CONDA_ENV' not found under $CONDA_ROOT/envs"
if tmux has-session -t "$SESSION" 2>/dev/null && [ "$REUSE_SESSION" != "1" ]; then
    fail "session '$SESSION' exists (tmux kill-session -t $SESSION)"
fi

# --- ROS environment scrub ---------------------------------------------------
# LD_LIBRARY_PATH is included deliberately: ROS puts /opt/ros/humble/lib and its
# rviz_ogre_vendor libs there, which shadow conda's libtiff/libffi and produce
# import errors far from the cause. Torch in a conda env does not need it.
ROS_VARS="PYTHONPATH AMENT_PREFIX_PATH AMENT_CURRENT_PREFIX CMAKE_PREFIX_PATH COLCON_PREFIX_PATH LD_LIBRARY_PATH PKG_CONFIG_PATH ROS_DISTRO ROS_VERSION ROS_PYTHON_VERSION ROS_LOCALHOST_ONLY RMW_IMPLEMENTATION"
if [ "$SCRUB_ROS_ENV" = "1" ]; then
    SCRUB="unset $ROS_VARS;"
    if [ -n "${ROS_DISTRO:-}" ]; then
        echo "NOTE: this shell has ROS '$ROS_DISTRO' sourced — scrubbing it from the conda windows."
        echo "      (that is correct; the ROS half belongs in a separate, base-environment shell)"
    fi
else
    SCRUB=""
    echo "WARN: SCRUB_ROS_ENV=0 — ROS's PYTHONPATH/LD_LIBRARY_PATH will be inherited by conda."
fi

# --- proxy ------------------------------------------------------------------
# Only matters if a proxy is configured, but costs nothing when it is not.
NO_PROXY_VAL="localhost,127.0.0.1,::1,${no_proxy:-}"
PROXY_ENV="no_proxy=$NO_PROXY_VAL NO_PROXY=$NO_PROXY_VAL"
if [ -n "${http_proxy:-}${HTTP_PROXY:-}" ]; then
    echo "NOTE: http_proxy is set — exporting NO_PROXY so loopback POSTs bypass it."
fi

# --- port pre-flight ---------------------------------------------------------
# A stale server from a previous run holding :13000 is invisible until the
# bridge POSTs into it and gets answers from the wrong policy.
for pp in "$CLIP_PORT:CLIP" "$SAM3_PORT:SAM3" "$RPV_POLICY_PORT:policy"; do
    p=${pp%%:*}; lbl=${pp##*:}
    if ss -ltn 2>/dev/null | grep -q ":$p "; then
        fail "port $p ($lbl) is already in use — kill the old server first (ss -ltnp | grep :$p)"
    fi
done

# --- GPU report --------------------------------------------------------------
# Printed rather than gated: SAM3's weights are ~3.4 GB and this same stack has
# run on an 8 GB card. Nothing in the ROS half uses the GPU
# (spot_driver, depth_image_proc, pointcloud_to_laserscan, slam_toolbox and Nav2
# are all CPU, and it passes launch_rviz:=False), so ROS is not your competition
# for VRAM here -- it is your competition for CPU and RAM.
if command -v nvidia-smi >/dev/null; then
    echo "GPU: $(nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader | paste -sd' | ')"
else
    echo "WARN: nvidia-smi not found — cannot report VRAM."
fi
echo "RAM: $(free -h | awk '/^Mem:/ {print $3" used / "$2" total"}')  CPU: $(nproc) cores"
if ! systemctl is-active --quiet earlyoom; then
    echo "WARN: earlyoom is NOT active — OOM freeze safety net off (sudo systemctl enable --now earlyoom)"
    echo "      ROS and the AI stack share this machine's RAM."
    read -r -p "Continue without it? [y/N] " a; [ "${a:-N}" = "y" ] || exit 1
fi

CONDA_SETUP="$SCRUB source $CONDA_ROOT/etc/profile.d/conda.sh && conda activate $CONDA_ENV"

new_win() { tmux new-window -t "$SESSION" -n "$1"; tmux send-keys -t "$SESSION:$1" "$2" C-m; }
wait_for_port() {
    local port=$1 label=$2 timeout=$3 waited=0
    echo -n "  waiting for $label (:$port) "
    while ! ss -ltn 2>/dev/null | grep -q ":$port "; do
        sleep 2; waited=$((waited+2)); echo -n "."
        [ "$waited" -ge "$timeout" ] && { echo " TIMEOUT (continuing)"; return 1; }
    done
    echo " up (${waited}s)"
}

echo "Starting '$SESSION' (VLM servers + the RPV policy server) ..."
CLIP_CMD="$CONDA_SETUP && cd $REPO && python -m vlfm.vlm.clip --port $CLIP_PORT"
if tmux has-session -t "$SESSION" 2>/dev/null; then
    # REUSE_SESSION: the session was made by whoever is orchestrating us.
    new_win vlm-clip "$CLIP_CMD"
else
    tmux new-session -d -s "$SESSION" -n vlm-clip
    tmux send-keys -t "$SESSION:vlm-clip" "$CLIP_CMD" C-m
fi
new_win vlm-sam3 "$CONDA_SETUP && cd $REPO && python -m vlfm.vlm.sam3 --port $SAM3_PORT --conf_threshold $SAM3_CONF"
if [ "$SKIP_YOLOE" != "1" ]; then
    new_win vlm-yoloe "$CONDA_SETUP && cd $REPO && YOLOE_CLASSES=$YOLOE_CLASSES python -m vlfm.vlm.yoloe --port $YOLOE_PORT"
else
    echo "SKIP_YOLOE=1 — not starting the YOLO-E server; SAM3 will be prompted directly."
fi

echo "VLM servers loading (ports LISTEN only once weights are ready)..."
wait_for_port "$CLIP_PORT" CLIP 240
wait_for_port "$SAM3_PORT" SAM3 240
[ "$SKIP_YOLOE" != "1" ] && wait_for_port "$YOLOE_PORT" YOLOE 240

# --- value-map propagation ---------------------------------------------------
# DECAY_SIGMA_M shapes the gradient the
# frontier ranking actually reads, MAX_PROPAGATION_M is a hard cutoff kept larger
# than the arena so the cliff never decides anything.
#
# UNVERIFIED FOR THIS SITE: 1.5 / 10.0 were tuned for the TurtleBot4's small
# indoor arena. If Spot's space is materially larger, sigma should grow with it
# -- a sigma small relative to the arena leaves distant frontiers all scoring ~0,
# the mirror of the flat-plateau failure that motivated dropping 5.0.
DECAY_SIGMA_M=${DECAY_SIGMA_M:-2.0}
MAX_PROPAGATION_M=${MAX_PROPAGATION_M:-6.0}

# --- deviations from the benchmarked RPV defaults ----------------------------
# Report them with any result.
ROOM_TYPES_FILE=${ROOM_TYPES_FILE:-data/room_types_lab.yaml}
SKIP_YOLOE_CLASSES=${SKIP_YOLOE_CLASSES:-data/shortvis.yaml}
STEPS_BETWEEN_DETECTIONS=${STEPS_BETWEEN_DETECTIONS:-1}
MIN_BLOB_AREA_M2=${MIN_BLOB_AREA_M2:-0.1}
PROPAGATION_SIGMA_CUTOFF=${PROPAGATION_SIGMA_CUTOFF:-3.0}

# --- Spot-specific -----------------------------------------------------------
# Depth range: the policy's depth is the bridge's ranging curtain, built from
# /scan, which on Spot is five body-camera depth images flattened with
# range_min / range_max in rpv-ros2-bridge's ceai/cameras.py. Its launch_offboard.sh
# derives both from that file and passes them in; the values below are only the
# fallback when this script is run on its own.
RPV_MIN_DEPTH_M=${RPV_MIN_DEPTH_M:-0.3}
RPV_MAX_DEPTH_M=${RPV_MAX_DEPTH_M:-2.0} # confirmed at https://support.bostondynamics.com/s/article/Spot-Specifications-49916
# D435 mount tilt, used only when the bridge sends no camera_pitch of its own.
# rpv-ros2-bridge's launch_offboard.sh passes the measured mount from the robot
# profile (its ceai/robots/<robot>.py); positive = lens down. The value below
# is only the fallback when this script is run on its own.
RPV_CAMERA_PITCH_RAD=${RPV_CAMERA_PITCH_RAD:-0.12}
RPV_CAMERA_ROLL_RAD=${RPV_CAMERA_ROLL_RAD:-0.0}
# Re-aim the goal heading at the value-map peak from further out than the
# TurtleBot4's 1.0 m: Spot pivots cheaply, so the heading change costs less.
RPV_SCAN_ARRIVAL_M=${RPV_SCAN_ARRIVAL_M:-1.5}
RPV_HEADING_RADIUS_M=${RPV_HEADING_RADIUS_M:-3.0}

RPV_DEBUG=${RPV_DEBUG:-0}

# --- per-step work the policy loop does not need -----------------------------
TIMED=${TIMED:-0}
RPV_STATUS_IMAGE_EVERY=${RPV_STATUS_IMAGE_EVERY:-3}
VALUE_MAP_IMG_EVERY=${VALUE_MAP_IMG_EVERY:-5}
if [ "$TIMED" = "1" ]; then
    VALUE_MAP_IMG_EVERY=0
    RPV_STATUS_IMAGE_EVERY=${RPV_STATUS_IMAGE_EVERY_TIMED:-10}
    echo "TIMED=1 — value-map dump off, status images every $RPV_STATUS_IMAGE_EVERY steps."
fi

# What the console's camera panel / run-record video is drawn on: `raw` = the
# camera frame with SAM3's segments outlined and labelled (target red); `yoloe`
# = YOLO-E's own box plot with the contours over it (upstream's, and only
# different from raw when SKIP_YOLOE=0). vlfm/policy/base_objectnav_policy.py.
ANNOTATED_RGB_BASE=${ANNOTATED_RGB_BASE:-raw}

VLM_REQUEST_RETRIES=${VLM_REQUEST_RETRIES:-2}
VLM_REQUEST_BACKOFF=${VLM_REQUEST_BACKOFF:-0.5}
VLM_BUSY_TIMEOUT=${VLM_BUSY_TIMEOUT:-15.0}
RPV_TIMING=${RPV_TIMING:-0}

# RPV_STATUS_IMAGE_EVERY / VALUE_MAP_IMG_EVERY / RPV_TIMING keep their RPV_ names
# on purpose: they are process-level logging knobs read by shared code, not robot
# geometry, so there is nothing Spot-specific to give them.
POLICY_ENV="$PROXY_ENV SKIP_MASK2FORMER=1 SKIP_YOLOE=$SKIP_YOLOE"
POLICY_ENV="$POLICY_ENV RPV_ROBOT=${RPV_ROBOT:-spot}"
POLICY_ENV="$POLICY_ENV RPV_POLICY_PORT=$RPV_POLICY_PORT RPV_POLICY_HOST=$RPV_POLICY_HOST"
POLICY_ENV="$POLICY_ENV DECAY_SIGMA_M=$DECAY_SIGMA_M MAX_PROPAGATION_M=$MAX_PROPAGATION_M"
POLICY_ENV="$POLICY_ENV RPV_STATUS_IMAGE_EVERY=$RPV_STATUS_IMAGE_EVERY VALUE_MAP_IMG_EVERY=$VALUE_MAP_IMG_EVERY"
POLICY_ENV="$POLICY_ENV VLM_REQUEST_RETRIES=$VLM_REQUEST_RETRIES VLM_REQUEST_BACKOFF=$VLM_REQUEST_BACKOFF VLM_BUSY_TIMEOUT=$VLM_BUSY_TIMEOUT"
POLICY_ENV="$POLICY_ENV RPV_TIMING=$RPV_TIMING ANNOTATED_RGB_BASE=$ANNOTATED_RGB_BASE"
POLICY_ENV="$POLICY_ENV ROOM_TYPES_FILE=$ROOM_TYPES_FILE SKIP_YOLOE_CLASSES=$SKIP_YOLOE_CLASSES YOLOE_CLASSES=$YOLOE_CLASSES"
POLICY_ENV="$POLICY_ENV STEPS_BETWEEN_DETECTIONS=$STEPS_BETWEEN_DETECTIONS MIN_BLOB_AREA_M2=$MIN_BLOB_AREA_M2 PROPAGATION_SIGMA_CUTOFF=$PROPAGATION_SIGMA_CUTOFF"
POLICY_ENV="$POLICY_ENV RPV_MIN_DEPTH_M=$RPV_MIN_DEPTH_M RPV_MAX_DEPTH_M=$RPV_MAX_DEPTH_M"
POLICY_ENV="$POLICY_ENV RPV_CAMERA_PITCH_RAD=$RPV_CAMERA_PITCH_RAD RPV_CAMERA_ROLL_RAD=$RPV_CAMERA_ROLL_RAD"
POLICY_ENV="$POLICY_ENV RPV_SCAN_ARRIVAL_M=$RPV_SCAN_ARRIVAL_M RPV_HEADING_RADIUS_M=$RPV_HEADING_RADIUS_M"
POLICY_ENV="$POLICY_ENV RPV_DEBUG=$RPV_DEBUG"
[ "$MAP_DEBUG" = "1" ] && POLICY_ENV="$POLICY_ENV MAP_DEBUG=true"

LOGDIR=${LOGDIR:-$REPO/logs}; mkdir -p "$LOGDIR"
new_win policy "$CONDA_SETUP && cd $REPO && $POLICY_ENV python -u -m vlfm.policy.rpv_policy_server 2>&1 | tee $LOGDIR/rpv_policy_\$(date +%Y%m%d_%H%M%S).log"
wait_for_port "$RPV_POLICY_PORT" policy-server 120

# No resource monitor window since 2026-09-19: the run record the console's
# Record button writes (rosbag + policy decision log + panel videos) is what a
# run is inspected from. scripts/monitor_resources.sh still exists for a
# RAM/VRAM hunt by hand.

cat <<EOF

Policy stack '$SESSION' is up.
  policy server : http://$RPV_POLICY_HOST:$RPV_POLICY_PORT/rpv_policy
  depth range   : $RPV_MIN_DEPTH_M–$RPV_MAX_DEPTH_M m   camera pitch: $RPV_CAMERA_PITCH_RAD rad (lens down)
  attach: tmux attach -t $SESSION     stop: tmux kill-session -t $SESSION

Normally this script is called by rpv-ros2-bridge's launch_offboard.sh, which also
starts zenoh, SLAM, Nav2 and the bridge. Run on its own, start the ROS half
in a SEPARATE shell with ROS sourced (NOT this one):
  START_AI=0 <rpv-ros2-bridge>/scripts/launch_offboard.sh
EOF
