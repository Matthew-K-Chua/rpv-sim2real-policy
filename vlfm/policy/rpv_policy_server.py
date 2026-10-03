# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

"""
Robot-agnostic Flask server that hosts an RPVITMPolicyV2 for the ROS-side bridge
(``ceai.rpv_bridge``). One server, any robot: the RPV policy, perception, value
map, frontier scoring and goal-heading refinement are identical across platforms;
everything that differs between robots -- depth band, camera tilt, footprint
(``agent_radius``), stop radius and the near-detection gate -- is read from a
robot PROFILE selected by the ``RPV_ROBOT`` environment variable.

Why a server: the VLFM/AI stack runs in Python 3.12, but ``rclpy`` is only built
for Python 3.10, so the ROS node and the policy cannot share an interpreter. This
mirrors how the VLM models (Mask2Former / YOLO-E / SAM3 / CLIP) are hosted -- one
more Flask service alongside them. On some rigs (a laptop ROS brain + a GPU host)
that split is also a machine boundary; on others (the Spot workstation runs both,
over loopback) it is not. The transport is unchanged either way.

Request (POST /rpv_policy), JSON:
    rgb            : base64 JPEG, (H, W, 3) uint8
    rgb_order      : "rgb" (default) or "bgr" — channel order the JPEG decodes to.
                     The bridge forwards the camera's own JPEG bytes unmodified
                     (one less lossy generation than re-encoding), and OpenCV
                     decodes those to BGR, so the live bridge sends "bgr".
    depth_row      : base64 raw float32, (w,) — ONE row of the LiDAR depth curtain.
                     Every row of the curtain is identical (it is a vertical
                     "curtain" built from a 2D scan), so only the row travels;
                     the server tiles it to depth_shape. ~250 kB/step -> ~1 kB.
    depth_m        : base64 raw float32 (h*w) — legacy full raster. Used only when
                     depth_row is absent (older bridge).
    depth_png16    : base64 16-bit PNG of MILLIMETRES, registered to the RGB frame,
                     0 = invalid. Takes precedence over both of the above and
                     carries its own shape, so depth_shape is not needed with it.
                     This is the real-depth path: a dense
                     raster where the curtain has one distance per column. ~8x
                     smaller than depth_m and lossless, unlike a JPEG.
    depth_shape    : [h, w]   (required only for depth_row / depth_m)
    occ_hash       : hex digest identifying this occupancy grid (see occupancy).
    occupancy      : base64 raw int8 Nav2 OccupancyGrid data (-1 / 0..100).
                     OMITTED when the grid is byte-identical to the one already
                     cached here under occ_hash — SLAM republishes an unchanged
                     grid far more often than it changes it. If the hash misses
                     the cache (server restarted mid-episode) the response is
                     {"occ_cache_miss": true} and the bridge re-POSTs in full.
    occ_shape      : [H, W]
    origin_x/y     : OccupancyGrid origin (map frame, meters)
    resolution     : OccupancyGrid resolution (m/cell)
    robot_map_x/y/yaw : robot pose in the map frame
    start_x/y/yaw  : pinned episodic-frame origin (robot start pose, map frame)
    fx, fy, img_w  : RGB camera intrinsics (pixels)
    min_depth, max_depth : metric depth range used for [0,1] normalization
    camera_height  : camera height above the floor (meters)
    camera_pitch   : OPTIONAL float, radians, POSITIVE TILTS THE LENS DOWN. When
                     absent, the server profile's default is used. A level camera
                     (pitch=roll=0) reduces the transform to yaw-only.
    camera_roll    : OPTIONAL float, radians. Defaults to the profile.
    objectgoal     : target class string
    is_first       : bool, True on the first step of an episode (-> reset)
    finish_init    : bool, True once the base scan spin has completed

Response, JSON:
    mode              : "initialize" | "explore" | "navigate"
    goal_map_xy       : [x, y] in the map frame, or None while initializing
    goal_yaw          : desired heading in the map frame (radians)
    called_stop       : bool, target reached
    done_initializing : bool

ROUTES: ``POST /rpv_policy``, ``GET /status`` (for the GUI console).

WHAT A ROBOT PROFILE CHANGES, AND WHY EACH ONE MATTERS
------------------------------------------------------
1. **Depth range.** The policy's depth is the bridge's ranging curtain, built from
   ``/scan``, whose usable band is a property of the robot's depth sensor. It must
   track the flattener's ``range_min`` / ``range_max`` in ceai's launch: too wide a
   range compresses real readings into the bottom of the [0, 1] normalisation AND
   makes ``ObjectPointCloudMap``'s "out of range" test (``<= max_depth * 0.95``)
   accept the saturation value as a genuine far detection.

2. **Camera tilt.** ``_camera_tf`` builds the camera->episodic transform from the
   full roll/pitch/yaw. A level camera (Spot's D435 aside, the TurtleBot 4's OAK-D)
   has pitch=roll=0 and the transform reduces to yaw-only. A pitched camera cannot
   fold its tilt into ``camera_height``: a yaw-only transform back-projects every
   detection to the wrong ground range, with an error that grows with the row
   offset from the image centre. The bridge sends the live mount angles when it
   has them; the profile supplies the fallback.

3. **Geometry.** ``agent_radius`` (the obstacle-map dilation, i.e. the narrowest
   gap the frontier search will offer), ``pointnav_stop_radius`` and the
   near-detection gate (``min_object_distance``) are all measured from the robot's
   base origin and so differ per platform. ``agent_radius`` must match Nav2's
   inscribed radius in ceai's ``config/nav2_<robot>.yaml`` and ``agent_radius_m``
   in ceai's ``ceai/robots/<robot>.py`` -- change them together.

WHAT IT DELIBERATELY DOES NOT CHANGE
------------------------------------
``pixels_per_meter`` stays 20, pinned to the SLAM grid's 0.05 m cell for every
robot -- ``resample_to_vlfm`` NEAREST-upsamples that same source, so a finer map
adds no information and costs O(radius^2) in propagation. The perception stack,
the value map, the frontier scoring and the goal-heading refinement are inherited
untouched, so two robots' runs differ only in the profile deltas above.
"""

import base64
import math
import os
import threading
from time import perf_counter, time
from datetime import datetime
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import torch

from vlfm.policy.base_objectnav_policy import VLFMConfig
from vlfm.policy.rpv_timing import StepTimer, install_step_timers, TIMING_ENABLED
from vlfm.policy.rpv_policies import RPVITMPolicyV2
from vlfm.utils.geometry_utils import get_fov, wrap_heading, xyz_rpy_to_tf_matrix, xyz_yaw_to_tf_matrix
from vlfm.vlm.depth_codec import decode_png16_mm
from vlfm.vlm.server_wrapper import ServerMixin, host_model, image_to_str, str_to_image

# ===================================================================== #
# robot profiles -- the only per-robot values. Selected by RPV_ROBOT.
# ===================================================================== #
# Each numeric can be overridden by the matching RPV_* environment variable, so a
# launch script can DERIVE the depth band and camera tilt from ceai's own robot
# profile (one source of truth) rather than duplicating the numbers here.
ROBOT_PROFILES: Dict[str, Dict[str, Any]] = {
    "turtlebot4": dict(
        name="TurtleBot4",
        # RPLIDAR A1 curtain usable to ~12 m; OAK-D stream clipped to it. NOT the
        # Habitat sensor range (5 m) -- that stays upstream so sim stays comparable.
        depth_min_m=0.5,
        depth_max_m=12.0,
        # OAK-D looks straight ahead.
        camera_pitch_rad=0.0,
        camera_roll_rad=0.0,
        # The base pivot ate a visible slice of each step, so only re-aim the
        # heading at the value-map peak once within 1.0 m of the goal.
        heading_scan_arrival_m=1.0,
        heading_peak_radius_m=3.0,
        # radius of the TurtleBot 4 plus leeway (turtlebot4-user-manual/overview).
        agent_radius=0.19,
        pointnav_stop_radius=0.9,
        # Targets sit ~0.5-1.5 m away and base_link is at the rim.
        min_object_distance=0.3,
        # 0.0.0.0 so a ROS brain on another machine can reach it.
        host_default="0.0.0.0",
    ),
    "spot": dict(
        name="Spot",
        # /scan built from the D435 depth stream; keep in lockstep with ceai's
        # flattener range_min / range_max. See point 1 of the module docstring.
        depth_min_m=0.3,
        depth_max_m=2.0,
        # D435 mount pose relative to Spot's body, used when the bridge does not
        # send the angles itself (roll/yaw 0, pitch 0.12 rad lens-down).
        camera_pitch_rad=0.12,
        camera_roll_rad=0.0,
        # Spot turns in place cheaply, so re-aim from further out than the TB4.
        heading_scan_arrival_m=1.5,
        heading_peak_radius_m=3.0,
        # Half-WIDTH of Spot's ~1.1 x 0.5 m footprint (the inscribed radius Nav2's
        # inflation uses). 0.55 (the half-length, until 2026-09-19) closed every
        # gap under 1.15 m -- every doorway. Matches agent_radius_m in ceai's
        # ceai/robots/spot.py and the inscribed radius in config/nav2_spot.yaml.
        agent_radius=0.25,
        # From the body origin, which sits ~0.55 m behind the nose; parks the nose
        # ~0.85 m from the object. UNVERIFIED -- derived from the standoff, not
        # tape-measured (derived from the standoff; test_md/13 tape-measures it).
        pointnav_stop_radius=1.4,
        # Body origin ~0.55 m behind the nose: a detection closer than this from
        # the ORIGIN is level with or behind the nose, geometry the back-projection
        # cannot be trusted for. The TB4's 0.3 m gate would admit exactly those.
        min_object_distance=0.5,
        # Loopback: on Spot the bridge and this server share the workstation.
        host_default="127.0.0.1",
    ),
}
DEFAULT_ROBOT: str = "turtlebot4"


def _resolve_profile() -> Dict[str, Any]:
    """Pick the robot profile from RPV_ROBOT and apply any RPV_* env overrides.

    A wrong robot must never be silent: an unset RPV_ROBOT is logged loudly (and
    falls back to the default), and an unknown value is a hard error rather than a
    stack that comes up clean and then navigates with the wrong footprint.
    """
    robot = os.environ.get("RPV_ROBOT", "").strip().lower()
    if not robot:
        robot = DEFAULT_ROBOT
        print(
            f"[profile] RPV_ROBOT unset; defaulting to {robot!r}. "
            f"Set RPV_ROBOT to one of {sorted(ROBOT_PROFILES)} to be explicit.",
            flush=True,
        )
    if robot not in ROBOT_PROFILES:
        raise SystemExit(
            f"[profile] RPV_ROBOT={robot!r} is not a known robot "
            f"(one of {sorted(ROBOT_PROFILES)})."
        )
    p = dict(ROBOT_PROFILES[robot])
    p["robot"] = robot
    # Per-value env overrides. The bridge/launch scripts derive these from ceai's
    # own robot profile so the depth band and camera tilt have one source.
    p["depth_min_m"] = float(os.environ.get("RPV_MIN_DEPTH_M", p["depth_min_m"]))
    p["depth_max_m"] = float(os.environ.get("RPV_MAX_DEPTH_M", p["depth_max_m"]))
    p["camera_pitch_rad"] = float(os.environ.get("RPV_CAMERA_PITCH_RAD", p["camera_pitch_rad"]))
    p["camera_roll_rad"] = float(os.environ.get("RPV_CAMERA_ROLL_RAD", p["camera_roll_rad"]))
    p["heading_scan_arrival_m"] = float(os.environ.get("RPV_SCAN_ARRIVAL_M", p["heading_scan_arrival_m"]))
    p["heading_peak_radius_m"] = float(os.environ.get("RPV_HEADING_RADIUS_M", p["heading_peak_radius_m"]))
    return p


_depth_range_warned: bool = False

# Save a single value-map snapshot the instant the 360 deg base scan completes
# (done_initializing flips True), once per episode. On by default and independent
# of any per-step map dump.
SAVE_SCAN_MAP: bool = os.environ.get("RPV_SAVE_SCAN_MAP", "1") == "1"
# Unset -> the post-scan snapshot lands in this run's value-map folder alongside
# the per-step frames. Set RPV_MAP_LOG_DIR to restore the old flat layout.
MAP_LOG_DIR: Optional[str] = os.environ.get("RPV_MAP_LOG_DIR")

# Per-step value-map dump. Every run (= one server process) gets its own
# timestamped folder, so all the frames belonging to one experiment stay
# together and no two runs can interleave into the same directory.
#   value_map_imgs/<YYYY-MM-DD>-<HH-MM-SS>/ep000_step_000042.png
# VALUE_MAP_IMG_EVERY = 0 disables the dump entirely.
VALUE_MAP_IMG_DIR: str = os.environ.get("VALUE_MAP_IMG_DIR", "value_map_imgs")
VALUE_MAP_IMG_EVERY: int = int(os.environ.get("VALUE_MAP_IMG_EVERY", "5"))
_RUN_IMG_DIR: Optional[str] = None


def _run_img_dir() -> str:
    """Path of this run's image folder, created (and stamped) on first use.

    The timestamp is taken once and cached for the lifetime of the process:
    stamping per call would scatter one run's frames across many folders.
    """
    global _RUN_IMG_DIR
    if _RUN_IMG_DIR is None:
        stamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        _RUN_IMG_DIR = os.path.join(VALUE_MAP_IMG_DIR, stamp)
        os.makedirs(_RUN_IMG_DIR, exist_ok=True)
        print(f"[value-map] saving every {VALUE_MAP_IMG_EVERY} steps -> {_RUN_IMG_DIR}/", flush=True)
    return _RUN_IMG_DIR


# ===================================================================== #
# map image stacking helpers
# ===================================================================== #
def _resize_to_height(image: np.ndarray, height: int) -> np.ndarray:
    if image.shape[0] == height:
        return image
    scale = float(height) / float(image.shape[0])
    width = max(1, int(round(image.shape[1] * scale)))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def _stack_map_images(images: list[np.ndarray]) -> np.ndarray:
    valid_images = [img for img in images if img is not None]
    if not valid_images:
        raise ValueError("No images provided for stacking")
    target_height = max(img.shape[0] for img in valid_images)
    resized = [_resize_to_height(img, target_height) for img in valid_images]
    return np.hstack(resized)


# ===================================================================== #
# map <-> episodic rigid transform  (pure numpy, identical to the ROS node)
# ===================================================================== #
def make_T_map_to_epi(start_x: float, start_y: float, start_yaw: float):
    """Returns (map_to_epi, epi_to_map) callables, each mapping (N,2) -> (N,2).

    The episodic frame is the robot's start pose: origin at (start_x, start_y),
    +x along start_yaw. So the forward transform is

        epi = Rot(-start_yaw) @ (map - t)

    which MUST match how headings are converted elsewhere in this module
    (``robot_heading = wrap_heading(robot_map_yaw - start_yaw)``, i.e. also a
    -start_yaw rotation). This previously applied Rot(+start_yaw) to positions
    while headings used -start_yaw, so positions and headings lived in mirrored
    frames: every camera detection was placed at a bearing rotated by
    2*start_yaw about the robot, and the goal handed back to Nav2 was rotated by
    -2*start_yaw. At start_yaw = +/-90 deg that is a full 180 deg -- the robot
    saw the target ahead, mapped it behind itself, and drove away from it. At
    start_yaw = 0 or 180 deg the error vanishes, which is why it only showed up
    on some runs.

    Note ``pts @ M`` applies ``M.T`` to each row vector, hence R / R.T read
    "backwards" relative to the maths above.
    """
    c, s = np.cos(start_yaw), np.sin(start_yaw)
    R = np.array([[c, -s], [s, c]])  # Rot(+start_yaw)
    t = np.array([start_x, start_y])

    def map_to_epi(pts: np.ndarray) -> np.ndarray:
        pts = np.atleast_2d(pts)
        return (pts - t) @ R  # == Rot(-start_yaw) @ (pts - t)

    def epi_to_map(pts: np.ndarray) -> np.ndarray:
        pts = np.atleast_2d(pts)
        return pts @ R.T + t  # == Rot(+start_yaw) @ pts + t

    return map_to_epi, epi_to_map


# ===================================================================== #
# OccupancyGrid -> VLFM obstacle-map layers
# ===================================================================== #
def resample_to_vlfm(occ_bool: np.ndarray, src_resolution: float, pixels_per_meter: int = 20) -> np.ndarray:
    """Rescale a bool grid from src_resolution m/cell to VLFM's 1/ppm m/cell (NEAREST only)."""
    vlfm_cell = 1.0 / pixels_per_meter
    if abs(src_resolution - vlfm_cell) < 1e-9:
        return occ_bool
    scale = src_resolution / vlfm_cell
    h, w = occ_bool.shape
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(occ_bool.astype(np.uint8), (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    return resized.astype(bool)


def _src_bool_to_canvas_px(src_bool, resolution, origin_x, origin_y, obstacle_map, map_to_epi):
    """Rasterize a source occupancy bool grid onto cropped VLFM canvas pixels (N,2)."""
    occ = resample_to_vlfm(src_bool, resolution, obstacle_map.pixels_per_meter)
    cell = 1.0 / obstacle_map.pixels_per_meter
    rows, cols = np.nonzero(occ)
    world_x = origin_x + (cols + 0.5) * cell
    world_y = origin_y + (rows + 0.5) * cell
    world_xy = np.column_stack([world_x, world_y])
    if len(world_xy) == 0:
        return np.empty((0, 2), dtype=int)
    px = obstacle_map._xy_to_px(map_to_epi(world_xy))
    size = obstacle_map.size
    keep = (px[:, 0] >= 0) & (px[:, 0] < size) & (px[:, 1] >= 0) & (px[:, 1] < size)
    return px[keep]


# Embodied-RPV-NOTE: ObstacleMap.update_map keeps only the explored contour that
# contains (or is nearest to) the agent, so disconnected explored islands can't
# grow their own frontiers. populate_obstacle_map replaces update_map on the
# Nav2 path and originally skipped that step; a SLAM grid that reveals space
# behind a wall or through a window then produced frontiers in "unknown" space.
# On by default because it restores upstream behaviour; set
# RPV_PRUNE_EXPLORED_ISLANDS=0 to fall back to the unpruned map.
PRUNE_EXPLORED_ISLANDS: bool = os.environ.get("RPV_PRUNE_EXPLORED_ISLANDS", "1") == "1"


def keep_agent_component(explored_area: np.ndarray, agent_px: np.ndarray) -> np.ndarray:
    """Return ``explored_area`` reduced to the external contour containing the agent.

    Port of the findContours block in ObstacleMap.update_map. If no contour
    contains ``agent_px`` (x, y pixels), the nearest one is kept. A map with
    one or zero contours is returned unchanged.
    """
    contours, _ = cv2.findContours(
        explored_area.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if len(contours) <= 1:
        return explored_area
    agent_pt = (int(agent_px[0]), int(agent_px[1]))
    min_dist = np.inf
    best_idx = 0
    for idx, cnt in enumerate(contours):
        dist = cv2.pointPolygonTest(cnt, agent_pt, True)
        if dist >= 0:
            best_idx = idx
            break
        elif abs(dist) < min_dist:
            min_dist = abs(dist)
            best_idx = idx
    new_area = np.zeros_like(explored_area, dtype=np.uint8)
    cv2.drawContours(new_area, contours, best_idx, 1, -1)  # type: ignore
    return new_area.astype(bool)


def populate_obstacle_map(obstacle_map, grid, origin_x, origin_y, resolution, map_to_epi, agent_xy=None):
    """Rebuild all three layers VLFM frontiers need from a Nav2 trinary grid.

      * ``_map``           : occupied cells (== 100)
      * ``_navigable_map`` : 1 - dilate(obstacles) by agent radius (VLFM convention)
      * ``explored_area``  : known cells (!= -1), intersected with navigable,
                             then pruned to the connected component holding
                             ``agent_xy`` (episodic metres) when it is given
                             and PRUNE_EXPLORED_ISLANDS is set

    Then recompute ``frontiers`` (mirrors obstacle_map.py:148-153) so the policy
    can read them. This is the §2 logic, living where the ObstacleMap object is.
    """
    om = obstacle_map

    om._map.fill(False)
    obstacle_px = _src_bool_to_canvas_px(grid == 100, resolution, origin_x, origin_y, om, map_to_epi)
    if len(obstacle_px):
        om._map[obstacle_px[:, 1], obstacle_px[:, 0]] = True

    om._navigable_map = (
        1 - cv2.dilate(om._map.astype(np.uint8), om._navigable_kernel, iterations=1)
    ).astype(bool)

    known_px = _src_bool_to_canvas_px(grid != -1, resolution, origin_x, origin_y, om, map_to_epi)
    known_canvas = np.zeros((om.size, om.size), dtype=bool)
    if len(known_px):
        known_canvas[known_px[:, 1], known_px[:, 0]] = True
    om.explored_area = known_canvas & om._navigable_map
    if PRUNE_EXPLORED_ISLANDS and agent_xy is not None:
        agent_px = om._xy_to_px(np.asarray(agent_xy, dtype=float).reshape(1, 2))[0]
        om.explored_area = keep_agent_component(om.explored_area, agent_px)

    om._frontiers_px = om._get_frontiers()
    om.frontiers = om._px_to_xy(om._frontiers_px) if len(om._frontiers_px) else np.array([])


# ===================================================================== #
# payload decoders
# ===================================================================== #
def _b64_to_array(s: str, dtype, shape) -> np.ndarray:
    raw = base64.b64decode(s)
    return np.frombuffer(raw, dtype=dtype).reshape(shape).copy()


# ===================================================================== #
# the hosted policy
# ===================================================================== #
# GUI status panels (value/obstacle map + annotated camera) refresh every few
# steps; encoding all three JPEGs on every step is wasted work the policy loop
# doesn't need to pay. Env-tunable.
STATUS_IMAGE_EVERY: int = int(os.environ.get("RPV_STATUS_IMAGE_EVERY", "1"))


def _as_float_or_none(value: Any) -> Optional[float]:
    """JSON-safe float for the status snapshot (numpy scalars included)."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _fmt_conf(value: Any) -> str:
    conf = _as_float_or_none(value)
    return "none" if conf is None else f"{conf:.2f}"


class RPVPolicyServer(ServerMixin):
    """Hosts an RPVITMPolicyV2 for the ceai bridge. Robot-agnostic: the per-robot
    depth band, camera tilt and step-loop tunables come from the ``profile`` dict
    (see ``ROBOT_PROFILES`` / ``_resolve_profile``), resolved once at start-up."""

    def __init__(self, policy: RPVITMPolicyV2, profile: Dict[str, Any]) -> None:
        super().__init__()
        self.policy = policy
        self.robot_name: str = profile["name"]
        # Step-loop tunables, resolved once per server from the robot profile.
        self.depth_min_m: float = profile["depth_min_m"]
        self.depth_max_m: float = profile["depth_max_m"]
        self.camera_pitch_rad: float = profile["camera_pitch_rad"]
        self.camera_roll_rad: float = profile["camera_roll_rad"]
        self.heading_peak_radius_m: float = profile["heading_peak_radius_m"]
        self.heading_scan_arrival_m: float = profile["heading_scan_arrival_m"]
        # Logged once rather than per step: a wrong pitch is invisible in the
        # per-step output (every bearing stays self-consistent) and shows up only
        # as a systematic range error, so the value in force has to be on record.
        self._mount_warned = False
        # Once-per-episode guard + counter for the post-360 value-map snapshot.
        self._scan_map_saved = False
        self._episode_idx = 0
        # Per-episode transition memory for the [event] log lines: the mode the
        # last step ended in, and whether "target found" / "stop" were already
        # announced. Reset on is_first.
        self._last_mode: Optional[str] = None
        self._target_announced = False
        self._stop_announced = False
        # Last occupancy grid received, keyed by the bridge's content hash, so an
        # unchanged grid costs a 32-char string instead of a base64 raster (A-9).
        # (hash, grid) — replaced wholesale, never mutated, so the /status thread
        # cannot observe a half-updated pair.
        self._occ_cache: Optional[tuple] = None
        # Per-stage step timing (RPV_TIMING=1). The timer is replaced every step, so
        # the shims are handed a getter rather than one object. None between steps,
        # which is also what makes the GET /status thread free of timing overhead.
        self._timer: Optional[StepTimer] = None
        install_step_timers(policy, lambda: self._timer)
        # Latest reasoning snapshot for the GUI console's GET /status poll. Held
        # under a lock and only ever handed out as a copy, so the polling thread
        # (Flask threaded=True) never races the step that mutates policy state.
        self._status_lock = threading.Lock()
        self._status: Dict[str, Any] = {"mode": "starting", "ready": False, "step": 0}
        self._status_step: int = 0

    def get_status(self) -> Dict[str, Any]:
        """Return a shallow copy of the latest reasoning snapshot (GET /status)."""
        with self._status_lock:
            return dict(self._status)

    def _save_scan_maps(self, policy_info: Dict[str, Any]) -> None:
        """Save a one-shot value-map snapshot taken right after the 360 base scan.

        Reads the already-rendered images from ``policy_info`` (the snapshot the
        policy took during ``act()``); does NOT re-call ``_get_policy_info``,
        which would KeyError on the now-cleared observations cache.
        """
        value_map_rgb = policy_info.get("value_map")
        if value_map_rgb is None:
            # Only present when the policy's _visualize flag is on. Skip quietly
            # rather than crash the bridge.
            print("[scan-map] value_map missing (policy _visualize off?); skipping save", flush=True)
            return
        obstacle_map_rgb = policy_info.get("obstacle_map")
        obstacle_map_detections_rgb = policy_info.get("obstacle_map_detections", obstacle_map_rgb)

        # The value map auto-crops to the explored area, but the obstacle map
        # renders the full canvas with the arena as a tiny speck. Crop each
        # obstacle panel to its content (anything not near-white) so it fills the
        # frame and is comparable in size to the value map.
        def _crop_to_content(img: np.ndarray, pad: int = 6) -> np.ndarray:
            if img is None:
                return img
            content = np.any(img < 250, axis=2)  # not near-white background
            if not content.any():
                return img
            rows = np.where(np.any(content, axis=1))[0]
            cols = np.where(np.any(content, axis=0))[0]
            r0, r1 = max(0, rows[0] - pad), min(img.shape[0], rows[-1] + pad + 1)
            c0, c1 = max(0, cols[0] - pad), min(img.shape[1], cols[-1] + pad + 1)
            return img[r0:r1, c0:c1]

        # The value map is already rendered to a fixed output canvas, so it is
        # saved as-is -- re-zooming it would only multiply the file size. The
        # obstacle panels are still raw canvas pixels, so they are cropped to
        # content and matched to the value map's height.
        # policy_info renders in RGB; OpenCV imwrite expects BGR.
        value_map_bgr = cv2.cvtColor(value_map_rgb, cv2.COLOR_RGB2BGR)
        target_h = value_map_bgr.shape[0]

        def _match(img: np.ndarray) -> np.ndarray:
            cropped = _crop_to_content(img)
            scale = target_h / float(max(1, cropped.shape[0]))
            interp = cv2.INTER_NEAREST if scale >= 1.0 else cv2.INTER_AREA
            w = max(1, int(round(cropped.shape[1] * scale)))
            return cv2.resize(cropped, (w, target_h), interpolation=interp)

        images = [value_map_bgr]
        if obstacle_map_rgb is not None:
            images.append(_match(cv2.cvtColor(obstacle_map_rgb, cv2.COLOR_RGB2BGR)))
        if obstacle_map_detections_rgb is not None:
            images.append(_match(cv2.cvtColor(obstacle_map_detections_rgb, cv2.COLOR_RGB2BGR)))

        # Written into this run's folder alongside the per-step frames, so one
        # experiment leaves exactly one directory behind. RPV_MAP_LOG_DIR still
        # overrides the location if the old flat layout is wanted.
        out_dir = MAP_LOG_DIR or _run_img_dir()
        os.makedirs(out_dir, exist_ok=True)
        tag = f"scan_complete_ep{self._episode_idx:03d}"
        cv2.imwrite(os.path.join(out_dir, f"{tag}_value_map.png"), value_map_bgr)
        cv2.imwrite(os.path.join(out_dir, f"{tag}_side_by_side.png"), _stack_map_images(images))
        print(f"[scan-map] saved post-360 snapshot -> {out_dir}/{tag}_*.png", flush=True)

    def _save_value_map_frame(self, policy_info: Dict[str, Any]) -> None:
        """Dump this step's value map into the run folder, every n steps.

        Saves the same frame the GUI shows -- ValueMap.visualize already renders
        to a fixed output canvas, so no extra zoom or resize is applied here and
        every frame in the folder has identical dimensions and scale.
        """
        if VALUE_MAP_IMG_EVERY <= 0 or self._status_step % VALUE_MAP_IMG_EVERY != 0:
            return
        value_map_rgb = policy_info.get("value_map")
        if value_map_rgb is None:
            return  # policy._visualize is off; nothing rendered to save
        name = f"ep{self._episode_idx:03d}_step_{self._status_step:06d}.png"
        path = os.path.join(_run_img_dir(), name)
        bgr = cv2.cvtColor(np.asarray(value_map_rgb, dtype=np.uint8), cv2.COLOR_RGB2BGR)
        if not cv2.imwrite(path, bgr):
            print(f"[value-map] FAILED to write {path}", flush=True)

    def _log_state(
        self,
        result: Dict[str, Any],
        objectgoal: str,
        robot_map_x: float,
        robot_map_y: float,
        goal_map_xy: Optional[List[float]],
    ) -> None:
        """One `[state]` line per step, plus a `[event]` line the first time the
        target is seen, each time the mode changes, and when the policy calls
        stop. Both carry wall-clock seconds so the policy log can be lined up
        with the resource CSV and the bridge's timing log. read_policy_log.sh
        lists the `[event]` lines under "Events"."""
        info = result.get("info", {}) or {}
        mode = str(result.get("mode", "?"))
        detected = bool(info.get("target_detected", False))
        conf = _fmt_conf(info.get("target_confidence"))
        stopped = bool(result.get("called_stop", False))
        now = time()
        wall = datetime.fromtimestamp(now).strftime("%H:%M:%S.%f")[:-3]
        goal = "none" if goal_map_xy is None else f"({goal_map_xy[0]:+.2f},{goal_map_xy[1]:+.2f})"
        step = self._status_step
        head = f"t={now:.3f} {wall} ep={self._episode_idx} step={step}"

        print(
            f"[state] {head} mode={mode} target={objectgoal!r} target_detected={detected} "
            f"target_conf={conf} "
            f"called_stop={stopped} robot_map=({robot_map_x:+.2f},{robot_map_y:+.2f}) "
            f"goal_map={goal}",
            flush=True,
        )
        if detected and not self._target_announced:
            self._target_announced = True
            print(f"[event] {head} TARGET FOUND {objectgoal!r} sam3_conf={conf}", flush=True)
        if mode != self._last_mode:
            if self._last_mode is not None:
                print(f"[event] {head} MODE {self._last_mode} -> {mode}", flush=True)
            if mode == "navigate":
                print(f"[event] {head} NAVIGATING to {objectgoal!r} goal_map={goal}", flush=True)
            self._last_mode = mode
        if stopped and not self._stop_announced:
            self._stop_announced = True
            print(
                f"[event] {head} STOPPED target reached ({objectgoal!r}) at "
                f"robot_map=({robot_map_x:+.2f},{robot_map_y:+.2f})",
                flush=True,
            )

    def _publish_status(self, result: Dict[str, Any], objectgoal: str) -> None:
        """Build the GUI status snapshot from a finished policy step."""
        info = result.get("info", {}) or {}
        self._status_step += 1
        self._save_value_map_frame(info)
        snapshot: Dict[str, Any] = {
            "ready": True,
            "step": self._status_step,
            "mode": result.get("mode", "?"),
            "objectgoal": objectgoal,
            "target_detected": bool(info.get("target_detected", False)),
            # SAM3's score for the target on THIS step's frame; None on steps
            # where SAM3 did not run or did not segment the target. The GUI's
            # run recorder logs it beside target_detected.
            "target_confidence": _as_float_or_none(info.get("target_confidence")),
            "called_stop": bool(result.get("called_stop", False)),
            "done_initializing": bool(result.get("done_initializing", False)),
            # Text panels (already formatted by the policy for human reading).
            "detections": info.get("detections", ""),
            "debug": info.get("debug", ""),
        }
        # Image panels: encode at most every STATUS_IMAGE_EVERY steps to keep the
        # step loop cheap. Keep the previous frames between encodes so the GUI
        # never blanks out.
        if self._status_step % STATUS_IMAGE_EVERY == 0:
            # Prefer the obstacle map with detection markers when present.
            obstacle = info.get("obstacle_map_detections", info.get("obstacle_map"))
            panels = {
                "value_map": info.get("value_map"),
                "obstacle_map": obstacle,
                "annotated_rgb": info.get("annotated_rgb"),
            }
            for key, img in panels.items():
                if img is not None:
                    # The policy stores these RGB-ordered; image_to_str/cv2.imencode
                    # expect BGR, so convert first. Any standard JPEG decoder (the
                    # GUI's QImage.loadFromData) then shows true colours.
                    bgr = cv2.cvtColor(np.asarray(img, dtype=np.uint8), cv2.COLOR_RGB2BGR)
                    # The maps are hard-edged synthetic images (inferno ramp
                    # against flat white/grey); JPEG rings visibly around those
                    # edges at 80, so the map panels get a higher quality than a
                    # photographic frame would need.
                    quality = 80 if key == "annotated_rgb" else 92
                    snapshot[f"{key}_jpeg"] = image_to_str(bgr, quality=quality)
        with self._status_lock:
            # Carry forward the most recent images if this step skipped encoding.
            if self._status_step % STATUS_IMAGE_EVERY != 0:
                for k, v in self._status.items():
                    if k.endswith("_jpeg"):
                        snapshot.setdefault(k, v)
            self._status = snapshot

    def _camera_tf(
        self,
        robot_xy: np.ndarray,
        robot_heading: float,
        camera_height: float,
        payload: dict,
    ) -> np.ndarray:
        """Camera-optical -> episodic transform for this step, including any tilt.

        Prefers the roll/pitch the bridge sends (it has the live TF and therefore
        the authoritative mount pose) and falls back to the robot profile, so a
        bridge that has not been taught to send them still produces a correct
        transform as long as the two ends are configured alike. A level camera
        (pitch=roll=0) reduces this to the yaw-only transform.
        """
        pitch = payload.get("camera_pitch")
        roll = payload.get("camera_roll")
        from_payload = pitch is not None or roll is not None
        pitch = self.camera_pitch_rad if pitch is None else float(pitch)
        roll = self.camera_roll_rad if roll is None else float(roll)

        if not self._mount_warned:
            source = "bridge payload" if from_payload else "server profile (bridge sent none)"
            print(
                f"[mount] camera tilt from {source}: "
                f"pitch={np.rad2deg(pitch):+.2f}deg roll={np.rad2deg(roll):+.2f}deg "
                f"height={camera_height:.3f}m. Positive pitch = lens down.",
                flush=True,
            )
            self._mount_warned = True

        # Camera yaw relative to the body is folded into robot_heading: the
        # bridge already projects the ranging curtain into the camera's own
        # optical frame, so a non-zero mount YAW would double-count. Roll and
        # pitch do not have that problem -- the curtain carries forward distance,
        # not elevation -- which is why only those two are applied here.
        return xyz_rpy_to_tf_matrix(
            np.array([robot_xy[0], robot_xy[1], camera_height]), roll, pitch, robot_heading
        )

    def process_payload(self, payload: dict) -> Dict[str, Any]:
        timer = StepTimer()
        self._timer = timer if TIMING_ENABLED else None
        t_step0 = perf_counter()

        # --- decode sensors ---
        t_decode0 = perf_counter()
        rgb = str_to_image(payload["rgb"])
        # cv2.imdecode always yields BGR. The bridge used to decode + re-encode the
        # frame so the bytes on the wire were channel-swapped and this came out RGB
        # by luck of the double swap; it now forwards the camera's own JPEG and says
        # so. Default "rgb" keeps an older bridge working unchanged.
        if str(payload.get("rgb_order", "rgb")).lower() == "bgr":
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]

        # Depth, in preference order: a real raster (16-bit PNG of millimetres),
        # one row of the /scan curtain expanded here, or a legacy float32 raster.
        if payload.get("depth_png16") is not None:
            # The ZoeDepth path. Shape comes from the PNG
            # itself, so depth_shape is not required alongside it.
            mm = decode_png16_mm(payload["depth_png16"])
            depth_m = mm.astype(np.float32) / 1000.0
            # ZERO MEANS INVALID on the wire, but zero here would mean "an object
            # touching the lens": the normalisation below maps it to 0.0, which
            # ObjectPointCloudMap reads as a detection at min_depth. The curtain
            # path expresses invalid as max_depth precisely because the policy
            # rejects saturation, so convert to that convention rather than
            # inventing a second one.
            depth_m[mm == 0] = self.depth_max_m
        elif payload.get("depth_row") is not None:
            dh, dw = (int(x) for x in payload["depth_shape"])
            row = _b64_to_array(payload["depth_row"], np.float32, (dw,))
            # np.broadcast_to alone returns a read-only, non-contiguous view, which
            # cv2.resize below refuses; materialise it (250 kB) here instead.
            depth_m = np.ascontiguousarray(np.broadcast_to(row, (dh, dw)))
        else:
            dh, dw = (int(x) for x in payload["depth_shape"])
            depth_m = _b64_to_array(payload["depth_m"], np.float32, (dh, dw))

        # Occupancy: full grid, or the cached one when the bridge says it is unchanged.
        occ_shape = tuple(int(x) for x in payload["occ_shape"])
        occ_hash = payload.get("occ_hash")
        if payload.get("occupancy") is not None:
            grid = _b64_to_array(payload["occupancy"], np.int8, occ_shape)
            if occ_hash is not None:
                self._occ_cache = (occ_hash, grid)
        else:
            cached = self._occ_cache
            if cached is None or cached[0] != occ_hash or cached[1].shape != occ_shape:
                # Server restarted, or a hash we never saw. Ask for the real thing
                # rather than navigating on a stale map.
                print(f"[occ] cache miss for {occ_hash}; requesting a full grid", flush=True)
                self._timer = None      # this step does no work; don't leave a live timer
                return {"occ_cache_miss": True, "mode": None, "goal_map_xy": None,
                        "goal_yaw": 0.0, "called_stop": False, "done_initializing": False}
            grid = cached[1]
        timer.add("decode", (perf_counter() - t_decode0) * 1e3)

        fx = float(payload["fx"])
        fy = float(payload["fy"])
        img_w = int(payload.get("img_w", w))
        min_depth, max_depth = self.depth_min_m, self.depth_max_m
        global _depth_range_warned
        if not _depth_range_warned:
            bridge_min = float(payload.get("min_depth", min_depth))
            bridge_max = float(payload.get("max_depth", max_depth))
            if (bridge_min, bridge_max) != (min_depth, max_depth):
                print(
                    f"[depth] bridge declares {bridge_min:.2f}-{bridge_max:.2f} m but this "
                    f"server normalises to {min_depth:.2f}-{max_depth:.2f} m "
                    f"({self.robot_name} profile / RPV_MIN_DEPTH_M / RPV_MAX_DEPTH_M); "
                    f"using the server's range.",
                    flush=True,
                )
            _depth_range_warned = True
        camera_height = float(payload["camera_height"])

        # --- frame transforms (server owns all map<->episodic math) ---
        map_to_epi, epi_to_map = make_T_map_to_epi(
            float(payload["start_x"]), float(payload["start_y"]), float(payload["start_yaw"])
        )

        # --- robot pose in episodic frame ---
        robot_map_x = float(payload["robot_map_x"])
        robot_map_y = float(payload["robot_map_y"])
        robot_map_yaw = float(payload["robot_map_yaw"])
        robot_xy = map_to_epi([robot_map_x, robot_map_y])[0]
        robot_heading = wrap_heading(robot_map_yaw - float(payload["start_yaw"]))

        # --- obstacle map / frontiers from the Nav2 grid ---
        with timer.stage("obstacle"):
            populate_obstacle_map(
                self.policy._obstacle_map,
                grid,
                float(payload["origin_x"]),
                float(payload["origin_y"]),
                float(payload["resolution"]),
                map_to_epi,
                agent_xy=robot_xy,
            )

        # --- normalize + register depth to the RGB frame ---
        if depth_m.shape[:2] != (h, w):
            # /stereo/depth is a different resolution/FOV than the RGB preview;
            # nearest-resize is an approximation (see plan "open items").
            depth_m = cv2.resize(depth_m, (w, h), interpolation=cv2.INTER_NEAREST)
        depth_norm = np.clip((depth_m - min_depth) / (max_depth - min_depth), 0.0, 1.0).astype(np.float32)

        # --- camera->episodic transform (x-forward convention, matches Habitat) ---
        tf = self._camera_tf(robot_xy, robot_heading, camera_height, payload)
        camera_fov = get_fov(fx, img_w)

        print(f"[target] objectgoal={payload['objectgoal']!r}")

        observations = {
            "objectgoal": payload["objectgoal"],
            "robot_xy": np.asarray(robot_xy, dtype=float),
            "robot_heading": float(robot_heading),
            "object_map_rgbd": [(rgb, depth_norm, tf, min_depth, max_depth, fx, fy)],
            "value_map_rgbd": [(rgb, depth_norm, tf, min_depth, max_depth, camera_fov)],
        }

        if payload.get("finish_init", False):
            self.policy.finish_initializing()

        # An episode reset is the ONLY thing that transfers payload["objectgoal"]
        # into the policy (BaseObjectNavPolicy._pre_step sets _target_object there).
        # If the bridge never sends is_first, the policy silently keeps whatever
        # target it last had -- including "chair" left behind by _warmup_policy --
        # and pursues that instead, while this server still logs the requested
        # goal from the payload. Treat a target mismatch as a reset so the two can
        # never diverge.
        is_first = bool(payload.get("is_first", False))
        current_target = getattr(self.policy, "_target_object", "")
        if not is_first and current_target != payload["objectgoal"]:
            print(
                f"[reset] policy target {current_target!r} != requested "
                f"{payload['objectgoal']!r} and is_first was not set -- forcing an "
                f"episode reset. (Is the bridge sending is_first on step 0?)",
                flush=True,
            )
            is_first = True

        if is_first:
            # New episode: re-arm the post-360 snapshot guard and bump the index.
            self._scan_map_saved = False
            self._episode_idx += 1
            self._last_mode = None
            self._target_announced = False
            self._stop_announced = False
            # Pin down the episodic frame in the log. A map<->episodic convention
            # error is invisible in the per-step debug output (every bearing is
            # self-consistent) but scales with start_yaw, so record it.
            print(
                f"[episode {self._episode_idx}] start pose map=("
                f"{float(payload['start_x']):+.2f},{float(payload['start_y']):+.2f}) "
                f"yaw={np.rad2deg(float(payload['start_yaw'])):+.1f}deg "
                f"| robot map=({robot_map_x:+.2f},{robot_map_y:+.2f}) "
                f"yaw={np.rad2deg(robot_map_yaw):+.1f}deg "
                f"-> epi=({robot_xy[0]:+.2f},{robot_xy[1]:+.2f}) "
                f"heading={np.rad2deg(robot_heading):+.1f}deg",
                flush=True,
            )
        masks = torch.tensor([[0 if is_first else 1]], dtype=torch.bool)

        with timer.stage("policy"):
            result = self.policy.step(observations, masks)

        # --- save a value-map snapshot the moment the 360 base scan completes ---
        if SAVE_SCAN_MAP and result["done_initializing"] and not self._scan_map_saved:
            self._save_scan_maps(result["info"])
            self._scan_map_saved = True

        # --- stash reasoning snapshot for the GUI console (GET /status) ---
        with timer.stage("status"):
            self._publish_status(result, payload["objectgoal"])

        # --- goal back to map frame ---
        goal_map_xy = None
        goal_yaw = robot_map_yaw
        if result["goal_epi"] is not None:
            goal_epi = result["goal_epi"]
            goal_map = epi_to_map(goal_epi)[0]
            goal_map_xy = [float(goal_map[0]), float(goal_map[1])]

            # Default heading: face straight toward the goal point.
            goal_yaw = float(math.atan2(goal_map[1] - robot_map_y, goal_map[0] - robot_map_x))

            # Better heading (explore only, on arrival): once the robot is near
            # the frontier, face the value-map peak so the narrow-FOV camera
            # sweeps where the target is most likely instead of straight at the
            # frontier boundary. While still travelling, the default goal-facing
            # heading is kept so Nav2 drives forward rather than pivoting in
            # place. Navigate mode aims at the object itself, so leave it alone.
            if result["mode"] == "explore":
                rho_to_goal = float(np.linalg.norm(np.asarray(robot_xy) - goal_epi))
                if rho_to_goal <= self.heading_scan_arrival_m:
                    peak_epi = self.policy._value_map.peak_world_xy_within_radius(
                        robot_xy, radius_m=self.heading_peak_radius_m
                    )
                    if peak_epi is not None and np.linalg.norm(peak_epi - robot_xy) > 0.25:
                        peak_map = epi_to_map(peak_epi)[0]
                        goal_yaw = float(
                            math.atan2(peak_map[1] - robot_map_y, peak_map[0] - robot_map_x)
                        )

        if TIMING_ENABLED:
            print(
                timer.line(self._status_step, str(result["mode"]),
                           (perf_counter() - t_step0) * 1e3),
                flush=True,
            )
        self._timer = None

        self._log_state(result, payload["objectgoal"], robot_map_x, robot_map_y, goal_map_xy)

        return {
            "mode": result["mode"],
            "goal_map_xy": goal_map_xy,
            "goal_yaw": goal_yaw,
            "called_stop": bool(result["called_stop"]),
            "done_initializing": bool(result["done_initializing"]),
        }


def _warmup_policy(policy: RPVITMPolicyV2) -> None:
    """Run one synthetic step so the GPU vision models (SAM3 / CLIP) pay their
    first-inference CUDA/cuDNN init cost now, at server startup, instead of
    stalling the bridge's first real scan step by ~30 s.

    The synthetic step stays in 'initialize' mode (finish_init is never sent), and
    the state it leaves behind is explicitly wiped below -- do NOT rely on the real
    episode's first is_first=True step to do it. If the bridge omits is_first the
    policy keeps this function's synthetic objectgoal ("chair") and pursues that for
    the whole run while the log shows the requested goal, which is a very confusing
    failure. Best-effort: a failure (e.g. a VLM server not up yet) is logged and
    ignored so the policy server still starts.
    """
    try:
        import time

        h = w = 224
        rgb = np.full((h, w, 3), 127, dtype=np.uint8)
        depth_norm = np.full((h, w), 0.5, dtype=np.float32)
        fx = fy = 200.0
        min_depth, max_depth = 0.5, 3.0
        tf = xyz_yaw_to_tf_matrix(np.array([0.0, 0.0, 0.25]), 0.0)
        camera_fov = get_fov(fx, w)
        observations = {
            "objectgoal": "chair",
            "robot_xy": np.zeros(2, dtype=float),
            "robot_heading": 0.0,
            "object_map_rgbd": [(rgb, depth_norm, tf, min_depth, max_depth, fx, fy)],
            "value_map_rgbd": [(rgb, depth_norm, tf, min_depth, max_depth, camera_fov)],
        }
        masks = torch.tensor([[0]], dtype=torch.bool)  # is_first -> reset
        print("[warmup] priming vision pipeline (SAM3/CLIP) ...", flush=True)
        t0 = time.time()
        policy.step(observations, masks)
        print(f"[warmup] done in {time.time() - t0:.1f}s", flush=True)
    except Exception as e:
        print(f"[warmup] skipped ({type(e).__name__}: {e})", flush=True)
    finally:
        # Drop the synthetic episode's state unconditionally (including after a
        # partial failure). _reset() leaves _did_reset=True, which would suppress
        # the real episode's is_first reset, so clear it again afterwards.
        try:
            policy._reset()
            policy._did_reset = False
            policy._target_object = ""
            policy._num_steps = 0
        except Exception as e:  # never let cleanup stop the server from starting
            print(f"[warmup] state cleanup failed ({type(e).__name__}: {e})", flush=True)


def _build_policy(profile: Dict[str, Any]) -> RPVITMPolicyV2:
    """Construct the policy with VLFM defaults plus the robot profile's geometry."""
    cfg = VLFMConfig()
    kwargs = {k: getattr(cfg, k) for k in VLFMConfig.kwaarg_names}
    # reality.yaml-style overrides for an indoor robot behind a Nav2 bridge.
    # Unused on this path (the obstacle map is rebuilt from the SLAM grid, not
    # projected from depth) but kept at reality values so nothing shifts silently.
    kwargs["min_obstacle_height"] = 0.1
    kwargs["max_obstacle_height"] = 1.5
    # This scalar dilates the obstacle map by a (2 * agent_radius) square
    # (mapping/obstacle_map.py) and frontiers are only found in what is left, so
    # it is the NARROWEST GAP the policy will ever send the robot through. Matches
    # the inscribed radius in ceai's config/nav2_<robot>.yaml and agent_radius_m
    # in ceai/robots/<robot>.py -- change all three together.
    kwargs["agent_radius"] = profile["agent_radius"]
    kwargs["pointnav_stop_radius"] = profile["pointnav_stop_radius"]
    # Map resolution is deliberately pinned to the Nav2 OccupancyGrid's 0.05 m
    # cell: 1/0.05 = 20 px/m. Going finer does NOT add information on this path
    # -- resample_to_vlfm NEAREST-upsamples the same 5 cm source -- but it costs,
    # because signal propagation is O(radius_px^2). Override with
    # VLFM_PIXELS_PER_METER (and raise VLFM_MAP_SIZE with it) only if a future
    # sensor actually supplies sub-5 cm occupancy.
    kwargs["pixels_per_meter"] = 20
    kwargs["map_size"] = 1000
    # Thin structures (chair legs/backs, bin rims) are eroded away entirely by
    # the Habitat-tuned erosion of 5 px, producing "[object_map] DROPPED
    # '<target>': empty cloud" even when the target is plainly visible.
    kwargs["object_map_erosion_size"] = 2
    policy = RPVITMPolicyV2(**kwargs)
    # Near-detection gate is measured from the base origin, so it is per-robot.
    policy._object_map.min_object_distance = profile["min_object_distance"]  # type: ignore
    return policy


def main() -> None:
    profile = _resolve_profile()
    port = int(os.environ.get("RPV_POLICY_PORT", "13000"))
    # host defaults per profile: loopback for a single-machine rig (Spot), 0.0.0.0
    # for a laptop-ROS + GPU-host rig (TurtleBot 4). Override with RPV_POLICY_HOST.
    host = os.environ.get("RPV_POLICY_HOST", profile["host_default"])

    policy = _build_policy(profile)
    _warmup_policy(policy)
    server = RPVPolicyServer(policy, profile)
    print(
        f"Hosting RPV policy server for {profile['name']} on {host}:{port} "
        f"(POST /rpv_policy, GET /status)\n"
        f"  depth range {profile['depth_min_m']:.2f}-{profile['depth_max_m']:.2f} m "
        f"| agent_radius {profile['agent_radius']:.2f} m | stop radius {profile['pointnav_stop_radius']:.2f} m "
        f"| min object distance {profile['min_object_distance']:.2f} m "
        f"| camera pitch {profile['camera_pitch_rad']:+.2f} rad",
        flush=True,
    )
    host_model(
        server,
        "rpv_policy",
        port=port,
        host=host,
        status_provider=server.get_status,
    )


if __name__ == "__main__":
    main()
