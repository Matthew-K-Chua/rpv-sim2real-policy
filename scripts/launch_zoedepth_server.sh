#!/usr/bin/env bash
# =============================================================================
# ZoeDepth metric-depth server (the GPU half of the monocular-depth pipeline).
#
# For a robot whose camera publishes colour only: this server turns the colour
# stream into a metric depth raster in the colour camera's frame. With the
# D435 streaming real depth it is not needed and launch_offboard.sh does not
# start it (DEPTH_SOURCE=realsense, the default).
#
# THIS SCRIPT STARTS ONLY THE MODEL. The ROS side -- the node that subscribes to
# the colour stream, posts frames here and republishes 16UC1 depth -- runs in a
# separate, ROS-sourced shell, printed at the end. Same split as
# launch_offboard_ai.sh, and for the same reason: rclpy is Python 3.10 and
# this is not.
#
# Usage:  ./scripts/launch_zoedepth_server.sh
#         CONDA_ENV=rpv-emb ./scripts/launch_zoedepth_server.sh   # if the pre-flight passes
#   stop: tmux kill-session -t zoedepth
# =============================================================================
set -u

SESSION=${SESSION:-zoedepth}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=${REPO:-$(dirname "$SCRIPT_DIR")}
CONDA_ROOT=${CONDA_ROOT:-$HOME/miniconda3}
# rpv-emb, the SAME env the policy runs in. VERIFIED (2026-09-03): its
# transformers 5.2.0 ships ZoeDepth (ZoeDepthForDepthEstimation +
# post_process_depth_estimation), and the checkpoint uses transformers' own BEiT
# backbone rather than a timm one, so there is no second environment to build and
# nothing to install beyond the weights.
CONDA_ENV=${CONDA_ENV:-rpv-emb}
ZOEDEPTH_PORT=${ZOEDEPTH_PORT:-12185}
# Loopback by default: the ROS node is on this machine. Set 0.0.0.0 only if the
# camera's ROS side lives elsewhere, and then put it behind a VPN rather
# than the open network -- this endpoint takes an image and has no auth.
ZOEDEPTH_HOST=${ZOEDEPTH_HOST:-127.0.0.1}
ZOEDEPTH_BACKEND=${ZOEDEPTH_BACKEND:-hf}
# Indoor NYU head. Intel/zoedepth-nyu-kitti the moment the arena includes
# daylight or sightlines past ~10 m.
ZOEDEPTH_HF_MODEL=${ZOEDEPTH_HF_MODEL:-Intel/zoedepth-nyu}
ZOEDEPTH_HALF=${ZOEDEPTH_HALF:-1}
ZOEDEPTH_FLIP_AUG=${ZOEDEPTH_FLIP_AUG:-0}
SCRUB_ROS_ENV=${SCRUB_ROS_ENV:-1}
# REUSE_SESSION=1: put the window in an EXISTING session rather than making one.
# See the same flag in launch_offboard_ai.sh; rpv-ros2-bridge's launch_offboard.sh
# uses both so the whole workstation side lives in one session. Default 0 is
# the old behaviour.
REUSE_SESSION=${REUSE_SESSION:-0}

fail() { echo "ERROR: $*" >&2; exit 1; }
command -v tmux >/dev/null || fail "tmux not installed"
[ -d "$REPO" ] || fail "repo not found: $REPO (set REPO=...)"
[ -f "$CONDA_ROOT/etc/profile.d/conda.sh" ] || fail "conda not found at $CONDA_ROOT"
[ -d "$CONDA_ROOT/envs/$CONDA_ENV" ] || fail "conda env '$CONDA_ENV' not found under $CONDA_ROOT/envs"
if tmux has-session -t "$SESSION" 2>/dev/null && [ "$REUSE_SESSION" != "1" ]; then
    fail "session '$SESSION' exists (tmux kill-session -t $SESSION)"
fi
ss -ltn 2>/dev/null | grep -q ":$ZOEDEPTH_PORT " && fail "port $ZOEDEPTH_PORT is in use (ss -ltnp | grep :$ZOEDEPTH_PORT)"

# --- ROS environment scrub (see launch_offboard_ai.sh header note 1) -------
ROS_VARS="PYTHONPATH AMENT_PREFIX_PATH AMENT_CURRENT_PREFIX CMAKE_PREFIX_PATH COLCON_PREFIX_PATH LD_LIBRARY_PATH PKG_CONFIG_PATH ROS_DISTRO ROS_VERSION ROS_PYTHON_VERSION ROS_LOCALHOST_ONLY RMW_IMPLEMENTATION"
SCRUB=""
if [ "$SCRUB_ROS_ENV" = "1" ]; then
    SCRUB="unset $ROS_VARS;"
    [ -n "${ROS_DISTRO:-}" ] && echo "NOTE: scrubbing ROS '$ROS_DISTRO' from the conda window."
fi

NO_PROXY_VAL="localhost,127.0.0.1,::1,${no_proxy:-}"
PROXY_ENV="no_proxy=$NO_PROXY_VAL NO_PROXY=$NO_PROXY_VAL"

CONDA_SETUP="$SCRUB source $CONDA_ROOT/etc/profile.d/conda.sh && conda activate $CONDA_ENV"

# --- pre-flight: is a second env even needed? --------------------------------
# Cheap, and it answers the question the env file raises. A failure here is only
# fatal for the env actually selected.
echo -n "checking ZoeDepth import in '$CONDA_ENV' ... "
if bash -lc "$CONDA_SETUP >/dev/null 2>&1 && python -c 'from transformers import ZoeDepthForDepthEstimation' 2>/dev/null"; then
    echo "ok"
else
    if [ "$ZOEDEPTH_BACKEND" = "hf" ]; then
        fail "transformers in '$CONDA_ENV' has no ZoeDepthForDepthEstimation.
       This is expected to PASS on rpv-emb (transformers 5.2.0). If it does not, the env has drifted."
    fi
    echo "absent (backend=$ZOEDEPTH_BACKEND, continuing)"
fi

if command -v nvidia-smi >/dev/null; then
    echo "GPU: $(nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader | paste -sd' | ')"
    echo "     ESTIMATE, verify on first load: the BEiT-L backbone is ~1-2 GB in fp16,"
    echo "     and that is ON TOP of SAM3's ~3.4 GB in the same GPU."
else
    echo "WARN: nvidia-smi not found -- ZoeDepth on CPU is seconds per frame, not milliseconds."
fi

SERVER_ENV="$PROXY_ENV ZOEDEPTH_BACKEND=$ZOEDEPTH_BACKEND ZOEDEPTH_HF_MODEL=$ZOEDEPTH_HF_MODEL"
SERVER_ENV="$SERVER_ENV ZOEDEPTH_HALF=$ZOEDEPTH_HALF ZOEDEPTH_FLIP_AUG=$ZOEDEPTH_FLIP_AUG"

LOGDIR=${LOGDIR:-$REPO/logs}; mkdir -p "$LOGDIR"
ZOE_CMD="$CONDA_SETUP && cd $REPO && $SERVER_ENV python -u -m vlfm.vlm.zoedepth --port $ZOEDEPTH_PORT --host $ZOEDEPTH_HOST 2>&1 | tee $LOGDIR/zoedepth_\$(date +%Y%m%d_%H%M%S).log"
if tmux has-session -t "$SESSION" 2>/dev/null; then
    tmux new-window -t "$SESSION" -n zoedepth
    tmux send-keys -t "$SESSION:zoedepth" "$ZOE_CMD" C-m
else
    tmux new-session -d -s "$SESSION" -n zoedepth
    tmux send-keys -t "$SESSION:zoedepth" "$ZOE_CMD" C-m
fi

echo -n "waiting for the server (:$ZOEDEPTH_PORT) "
waited=0
while ! ss -ltn 2>/dev/null | grep -q ":$ZOEDEPTH_PORT "; do
    sleep 2; waited=$((waited+2)); echo -n "."
    [ "$waited" -ge 300 ] && { echo " TIMEOUT -- check: tmux attach -t $SESSION"; exit 1; }
done
echo " up (${waited}s)"

cat <<EOF

ZoeDepth server is up: http://$ZOEDEPTH_HOST:$ZOEDEPTH_PORT/zoedepth
  attach: tmux attach -t $SESSION     stop: tmux kill-session -t $SESSION

THEN, in a SEPARATE shell with ROS sourced (NOT this one):
  source /opt/ros/humble/setup.bash
  ros2 launch $REPO/ros_nodes/zoedepth_chain.launch.py \\
      zoedepth_url:=http://$ZOEDEPTH_HOST:$ZOEDEPTH_PORT/zoedepth scale_align:=true

  ros2 topic hz /zoedepth/image_raw
  ros2 topic echo /zoedepth/diagnostics --once     # [scale, residual_m, n_pairs, infer_ms]

FINALLY, to give the POLICY this depth: DEPTH_SOURCE=zoedepth <rpv-ros2-bridge>/scripts/launch_offboard.sh
  # which starts this server, the chain and the bridge together and sets
  # RPV_MAX_DEPTH_M to match the node's trust_max_m.
EOF
