# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

"""
Robot-agnostic reality port of ITMPolicyV2, behind a Nav2 bridge.

This is the *Nav2-bridge* RPV policy. Unlike ``RealityITMPolicyV2`` (upstream
VLFM's Spot port), which builds its obstacle map by projecting depth and drives
the base with the pointnav velocity output, this policy is designed to run
*behind* a Nav2 bridge (``ceai.rpv_bridge``), and the *deployment shape* is the
same for every robot it runs on -- Spot, TurtleBot 4, or the next platform:

  * The obstacle map / frontiers come from Nav2 LiDAR SLAM. The bridge populates
    this policy's ``self._obstacle_map`` layers (``_map`` / ``_navigable_map`` /
    ``explored_area``) every time a new ``/map`` arrives, so this policy does NOT
    project depth into the obstacle map -- ``_cache_observations`` just reads the
    already-populated ``frontiers``.

  * Motion is executed by Nav2 via ``/goal_pose``. The pointnav ResNet is
    therefore bypassed: ``_pointnav`` only records the chosen goal
    (``self._last_goal``, episodic xy) and the stop condition. The bridge reads
    ``self._mode`` / ``self._last_goal`` / ``self._called_stop`` after each
    ``act()`` and converts the goal to a Nav2 waypoint.

  * The init "scan" rotates the *base*. The bridge spins the robot ~360 deg,
    calling ``act()`` to capture frames; it then calls ``finish_initializing()``
    (or the step-count safety net trips) to leave initialize mode.

Depth comes from the RGB-D hardware stream (assembled by the bridge into
``object_map_rgbd``), so object detection + value-map signal placement work the
same way on every robot. The value map shares this policy's ``_obstacle_map`` for
its free-space mask, so the Nav2-derived free space propagates automatically.

Everything a robot actually changes -- footprint (``agent_radius``), stop radius,
depth band, near-detection gate, camera tilt -- is supplied by the *server* from
its robot profile (see ``rpv_policy_server.py``); nothing in this file is
robot-specific.
"""

from typing import Any, Dict, Union

import numpy as np
import torch
from omegaconf import DictConfig
from torch import Tensor

from vlfm.policy.base_objectnav_policy import VLFMConfig
from vlfm.policy.itm_policy import ITMPolicyV2
from vlfm.utils.geometry_utils import rho_theta

# Number of ``act()`` calls to remain in "initialize" mode if the bridge never
# calls ``finish_initializing()``. Acts purely as a safety net so the policy
# cannot get stuck scanning forever; the bridge normally ends init explicitly
# when its 360 deg base spin completes.
INIT_SCAN_STEPS: int = 12

# Default "too close to trust" gate (metres). The experiment on a small robot
# places targets ~0.5-1.5 m away and its base origin sits at the rim, so anything
# under 1.0 m (the Spot/Habitat default) would be silently dropped and the policy
# would never leave explore mode. The server overrides this per robot from its
# profile (e.g. Spot raises it, its body origin sitting ~0.55 m behind the nose).
DEFAULT_MIN_OBJECT_DISTANCE_M: float = 0.3


class RPVMixin:
    """Endows ITMPolicyV2 with the behaviour needed to run behind a Nav2 bridge
    (see module docstring), independent of which robot the bridge drives."""

    # Dummy 2-DOF action; its numeric value is never executed (Nav2 drives the
    # robot), but the base ``act()`` still indexes it, so keep the (1, 2) shape.
    _stop_action: Tensor = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
    _load_yolo: bool = False
    _load_pointnav: bool = False
    # Embodied-RPV-NOTE: added for the Nav2-bridge deployment. RPV drops sparse
    # point signals rather than painting VLFM's camera cone, so most cells in a
    # frontier's disc carry no signal at all: the upstream median reduction
    # reports ~0 for every frontier, and the -1 "nothing in the disc" sentinel
    # makes them all tie (np.argsort then falls back to contour order and
    # BaseITMPolicy latches onto the first frontier for the whole episode).
    # These two overrides are a DEVIATION from the benchmarked policy -- they
    # change how goals are chosen, so report them alongside any result.
    _waypoint_reduction: str = "max"
    _clamp_waypoint_values: bool = True
    _observations_cache: Dict[str, Any] = {}
    _policy_info: Dict[str, Any] = {}
    _done_initializing: bool = False

    def __init__(self: Union["RPVMixin", ITMPolicyV2], *args: Any, **kwargs: Any) -> None:
        # sync_explored_areas=True keeps the object map's explored cone aligned
        # with the obstacle map, matching the reality config.
        super().__init__(sync_explored_areas=True, *args, **kwargs)  # type: ignore
        # DBSCAN clustering assumes dense simulator point clouds; disable for the
        # sparser/noisier real-world object map (same as RealityMixin).
        self._object_map.use_dbscan = False  # type: ignore
        # Near-detection gate. The default suits a small robot with near targets;
        # the server retunes it per robot (see DEFAULT_MIN_OBJECT_DISTANCE_M).
        self._object_map.min_object_distance = DEFAULT_MIN_OBJECT_DISTANCE_M  # type: ignore
        self._init_steps: int = 0

    @classmethod
    def from_config(cls, config: DictConfig, *args_unused: Any, **kwargs_unused: Any) -> Any:
        policy_config: VLFMConfig = config.policy
        kwargs = {k: policy_config[k] for k in VLFMConfig.kwaarg_names}  # type: ignore
        return cls(**kwargs)

    # ------------------------------------------------------------------ #
    # Bridge-facing convenience wrapper
    # ------------------------------------------------------------------ #
    def step(
        self: Union["RPVMixin", ITMPolicyV2],
        observations: Dict[str, Any],
        masks: Tensor,
        deterministic: bool = True,
    ) -> Dict[str, Any]:
        """Run one ``act()`` and return the high-level decision for the bridge.

        Returns a dict with:
          * ``mode``: "initialize" | "explore" | "navigate"
          * ``goal_epi``: (2,) episodic xy goal, or ``None`` while initializing
          * ``called_stop``: True once within the stop radius of the target
          * ``done_initializing``: whether the scan phase is complete
          * ``info``: the policy's ``_policy_info`` (for viz / logging)
        """
        self.act(observations, None, None, masks, deterministic=deterministic)
        mode = getattr(self, "_mode", "initialize")
        goal_epi = None if mode == "initialize" else np.asarray(self._last_goal, dtype=float).copy()
        return {
            "mode": mode,
            "goal_epi": goal_epi,
            "called_stop": self._called_stop,
            "done_initializing": self._done_initializing,
            "info": self._policy_info,
        }

    def finish_initializing(self) -> None:
        """Called by the bridge when its 360 deg base scan completes."""
        self._done_initializing = True

    # ------------------------------------------------------------------ #
    # Overrides of the base state machine
    # ------------------------------------------------------------------ #
    def _reset(self: Union["RPVMixin", ITMPolicyV2]) -> None:
        super()._reset()  # type: ignore
        self._init_steps = 0
        self._done_initializing = False

    def _initialize(self) -> Tensor:
        """Stay in initialize mode while the bridge spins the base to scan.

        The bridge ends init explicitly via ``finish_initializing()``; the
        step counter is only a safety net so we cannot scan forever.
        """
        self._init_steps += 1
        if self._init_steps >= INIT_SCAN_STEPS:
            self._done_initializing = True
        return self._stop_action

    def _pointnav(self: Union["RPVMixin", ITMPolicyV2], goal: np.ndarray, stop: bool = False) -> Tensor:
        """Record the chosen goal instead of running the pointnav ResNet.

        Nav2 executes the motion, so we only need ``self._last_goal`` (episodic
        xy) and the stop condition; the returned tensor is never executed.
        """
        self._last_goal = goal
        robot_xy = self._observations_cache["robot_xy"]
        heading = self._observations_cache["robot_heading"]
        rho, theta = rho_theta(robot_xy, heading, goal)
        self._policy_info["rho_theta"] = np.array([rho, theta])
        if rho < self._pointnav_stop_radius and stop:
            self._called_stop = True
        # The goal and the stop decision are what the bridge acts on, so make them
        # visible: an episode that ends without the robot moving is indistinguishable
        # from one that never got a goal unless rho and called_stop are logged.
        print(
            f"[goal] target={self._target_object!r} epi=({goal[0]:+.2f},{goal[1]:+.2f}) "
            f"rho={rho:.2f}m theta={np.rad2deg(theta):+.0f}deg "
            f"stop_radius={self._pointnav_stop_radius:.2f}m "
            f"stop_armed={stop} called_stop={self._called_stop}",
            flush=True,
        )
        return self._stop_action

    def _infer_depth(self, rgb: np.ndarray, min_depth: float, max_depth: float) -> np.ndarray:
        """No monocular depth estimator on the Nav2-bridge path.

        The base ``_update_object_map`` calls this only when the incoming depth
        frame is all-ones, which on this fork means the Nav2/RGB-D bridge had no
        usable depth for the frame (e.g. an empty LiDAR curtain band). Without a
        depth model we simply return all-ones again: every masked pixel then
        projects to ``max_depth`` and is flagged out-of-range by
        ``ObjectPointCloudMap`` (``<= max_depth * 0.95``), so nothing is placed.
        The base class stub raises ``NotImplementedError``; this override keeps
        the policy running on no-depth frames instead of 500-ing the bridge.
        """
        return np.ones(rgb.shape[:2], dtype=np.float32)

    def _cache_observations(self: Union["RPVMixin", ITMPolicyV2], observations: Dict[str, Any]) -> None:
        """Read frontiers from the Nav2-populated obstacle map; cache RGB-D.

        The bridge writes the obstacle/navigable/explored layers of
        ``self._obstacle_map`` from the Nav2 occupancy grid before calling
        ``act()``, so here we only update the agent trajectory and read the
        frontiers -- no depth projection into the obstacle map.

        Expected keys in ``observations``:
          * ``objectgoal``   : target class string (consumed by ``_pre_step``)
          * ``robot_xy``     : (2,) episodic xy (np.ndarray)
          * ``robot_heading``: float, episodic yaw in radians
          * ``object_map_rgbd``: ``[(rgb, depth, tf_cam_to_epi, min_d, max_d, fx, fy)]``
          * ``value_map_rgbd`` : same layout with ``fov`` in place of ``fx, fy``
                                 (kept for interface parity; unused by this fork)
        """
        if len(self._observations_cache) > 0:
            return

        self._obstacle_map.update_agent_traj(observations["robot_xy"], observations["robot_heading"])
        frontiers = self._obstacle_map.frontiers

        self._observations_cache = {
            "frontier_sensor": frontiers,
            "robot_xy": observations["robot_xy"],
            "robot_heading": observations["robot_heading"],
            "object_map_rgbd": observations["object_map_rgbd"],
            "value_map_rgbd": observations["value_map_rgbd"],
        }


class RPVITMPolicyV2(RPVMixin, ITMPolicyV2):
    pass
