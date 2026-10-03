# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

import glob
import json
import os
import os.path as osp
import shutil
import time
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

try:
    import skfmm
    _HAS_SKFMM = True
except ImportError:
    _HAS_SKFMM = False

from vlfm.mapping.base_map import BaseMap
from vlfm.utils.geometry_utils import extract_yaw, get_rotation_matrix
from vlfm.utils.debug_log import dbg
from vlfm.utils.img_utils import (
    monochannel_to_inferno_rgb,
    pixel_value_within_radius,
    place_img_in_img,
    rotate_image,
)

DEBUG = False
SAVE_VISUALIZATIONS = False
RECORDING = os.environ.get("RECORD_VALUE_MAP", "0") == "1"
PLAYING = os.environ.get("PLAY_VALUE_MAP", "0") == "1"
RECORDING_DIR = "value_map_recordings"
JSON_PATH = osp.join(RECORDING_DIR, "data.json")
KWARGS_JSON = osp.join(RECORDING_DIR, "kwargs.json")


class ValueMap(BaseMap):
    """Generates a map representing how valuable explored regions of the environment
    are with respect to finding and navigating to the target object."""

    _confidence_masks: Dict[Tuple[float, float], np.ndarray] = {}
    _camera_positions: List[np.ndarray] = []
    _last_camera_yaw: float = 0.0
    _min_confidence: float = 0.25
    _decision_threshold: float = 0.35
    _map: np.ndarray

    def __init__(
        self,
        value_channels: int,
        size: int = 1000,
        pixels_per_meter: int = 20,
        use_max_confidence: bool = True,
        fusion_type: str = "default",
        obstacle_map: Optional["ObstacleMap"] = None,  # type: ignore # noqa: F821
        signal_mode: str = "fmm",                 # Propagation method: "gaussian" or "fmm" (geodesic flood fill)
        decay_sigma_m: float = 2.0,                    # Gaussian decay std-dev in metres (for both modes; for FMM this controls decay over geodesic distance)
        max_propagation_m: float = 6.0,                # Maximum signal propagation radius in metres (for both modes; for FMM this is a geodesic distance limit)
    ) -> None:
        """
        Args:
            value_channels: The number of channels in the value map.
            size: The size of the value map in pixels.
            use_max_confidence: Whether to use the maximum confidence value in the value
                map or a weighted average confidence value.
            fusion_type: The type of fusion to use when combining the value map with the
                obstacle map.
            obstacle_map: An optional obstacle map to use for overriding the occluded
                areas of the FOV
            signal_mode: "gaussian" for isotropic gaussians (then zeroed by free-space
                mask), or "fmm" for wall-respecting flood-fill via Fast Marching.
            decay_sigma_m: Gaussian decay std-dev in metres.  For ``fmm`` mode this
                controls how fast the signal decays with *geodesic* distance.
            max_propagation_m: Maximum propagation distance in metres.  For ``fmm``
                mode this is a geodesic distance limit; for ``gaussian`` mode it is
                the hard Euclidean cut-off radius (default ≈ 3*sigma).
        """
        if PLAYING:
            size = 2000
        super().__init__(size, pixels_per_meter)
        self._value_map = np.zeros((size, size, value_channels), np.float32)
        self._value_channels = value_channels
        self._use_max_confidence = use_max_confidence
        self._fusion_type = fusion_type
        self._obstacle_map = obstacle_map

        # Signal propagation configuration, can set variables as environment variables for easy ablation
        self._signal_mode: str = os.environ.get("SIGNAL_MODE", signal_mode).lower()
        print(f"USING SIGNAL MODE: {self._signal_mode}")
        self._decay_sigma_m: float = float(os.environ.get("DECAY_SIGMA_M", decay_sigma_m))
        self._max_propagation_m: float = float(os.environ.get("MAX_PROPAGATION_M", max_propagation_m))
        if (self._signal_mode == "fmm" or self._signal_mode == "raycast") and not _HAS_SKFMM:
            warnings.warn("scikit-fmm not installed — falling back to gaussian mode.")
            self._signal_mode = "gaussian"

        # FMM no-decay: when True, FMM flood fill uses flat peak_value
        # within radius instead of Gaussian decay over geodesic distance.
        self._fmm_no_decay: bool = os.environ.get("FMM_NO_DECAY", "0").lower() in ("1", "true", "yes")
        if self._fmm_no_decay:
            print("FMM NO DECAY: enabled — flat score within radius")

        # Truncate propagation at N sigma instead of the full radius. Crop cost is
        # O(radius^2), so this is a latency lever -- at 3 sigma the discarded tail
        # is <1.2% of peak. It still CHANGES the field, so it is off (0) by
        # default and the TurtleBot4 stack opts in.
        # Embodied-RPV-NOTE: added for TurtleBot4 deployment (latency).
        self._propagation_sigma_cutoff: float = float(
            os.environ.get("PROPAGATION_SIGMA_CUTOFF", 0.0)
        )
        if self._propagation_sigma_cutoff > 0:
            print(f"PROPAGATION SIGMA CUTOFF: {self._propagation_sigma_cutoff:.1f} sigma")

        # FMM / raycast multi-seed: when centroid is inside an obstacle, all
        # free cells within this radius of the centroid become seeds.  Covers
        # both sides of furniture-sized objects (sofas, beds etc.).
        self._fmm_seed_radius_m: float = float(os.environ.get("FMM_SEED_RADIUS_M", 1.0))

        # Variables for signal-based value tracking
        # Per-label signal maps: for each label, a 2D array storing the max
        # gaussian signal from all detections of that label.
        self._label_signal_maps: Dict[str, np.ndarray] = {}

        # Each tracked object dict has keys:
        #   label, world_xy, score, dirty (bool), signal_cache (optional crop tuple)
        # Updated in-place on each new detection; used to prevent duplicate
        # signals for the same object across multiple detections.
        self._tracked_objects: List[Dict[str, Any]] = []
        # same-label, same-position merge threshold. Env-tunable: in small envs
        # (e.g. a 3x3 m room) the 2.0 m default is larger than the whole map and
        # collapses distinct objects onto one wandering centroid.
        self._redetection_dist_m: float = float(os.environ.get("REDETECT_DIST_M", 2.0))
        self._centroid_dirty_dist_m: float = 0.5  # centroid shift that triggers signal recomputation

        # Previous step's free-space mask — used for dirty-flag diffing
        self._prev_free_mask: Optional[np.ndarray] = None
        # Last blob-gate drop count, so the gate only logs when it changes
        self._last_blob_drop_count: int = 0

        if self._obstacle_map is not None:
            assert self._obstacle_map.pixels_per_meter == self.pixels_per_meter
            assert self._obstacle_map.size == self.size
        if os.environ.get("MAP_FUSION_TYPE", "") != "":
            self._fusion_type = os.environ["MAP_FUSION_TYPE"]

        if RECORDING:
            if osp.isdir(RECORDING_DIR):
                warnings.warn(f"Recording directory {RECORDING_DIR} already exists. Deleting it.")
                shutil.rmtree(RECORDING_DIR)
            os.mkdir(RECORDING_DIR)
            # Dump all args to a file
            with open(KWARGS_JSON, "w") as f:
                json.dump(
                    {
                        "value_channels": value_channels,
                        "size": size,
                        "use_max_confidence": use_max_confidence,
                    },
                    f,
                )
            # Create a blank .json file inside for now
            with open(JSON_PATH, "w") as f:
                f.write("{}")

    def reset(self) -> None:
        super().reset()
        self._value_map.fill(0)
        self._label_signal_maps = {}
        self._tracked_objects = []
        self._prev_free_mask = None
        self._last_blob_drop_count = 0


    # Tracked-object centroid positions (for visualisation markers)
    def get_object_centroids(self) -> List[Dict[str, Any]]:
        """Return a copy of tracked object records.

        Each dict has keys ``label``, ``world_xy`` (2-element ndarray in
        episodic metres), and ``score``.
        """
        return list(self._tracked_objects)


    # Object-signal API: gaussian heat sources at detection centroids
    def _world_to_map_pixel(self, world_xy: np.ndarray) -> Tuple[int, int]:
        """Convert episodic world (x, y) in metres to value-map pixel (row, col).

        Uses the same convention as ``sort_waypoints`` so that signal placement
        and frontier evaluation are consistent.
        """
        x, y = world_xy
        px = int(-x * self.pixels_per_meter) + self._episode_pixel_origin[0]
        py = int(-y * self.pixels_per_meter) + self._episode_pixel_origin[1]
        row = self._value_map.shape[0] - 1 - px     # Zero-indexing
        col = py
        return row, col


    def add_object_signal(
        self,
        world_xy: np.ndarray,
        score: float,
        label: str,
        room: Optional[str] = None,
    ) -> None:
        """Register or update a detected object for signal propagation.

        The actual signal map is (re-)computed in ``recompute_value_map``
        so that obstacle-map changes are always reflected.

        Args:
            world_xy: Detection position in episodic metres (x, y).
            score: Peak value (e.g. CLIP dot-product in [0, 1]).
            label: Semantic label of the detected object.
            room: Optional predicted room type for this detection (used only
                for visualisation — pinning the label and labelling frontiers).
        """
        if score <= 0:
            return

        row, col = self._world_to_map_pixel(world_xy)
        if row < 0 or row >= self.size or col < 0 or col >= self.size:
            # Off-canvas. Loud, because it is otherwise invisible: the detection
            # is logged by [signal-dbg] upstream and then silently vanishes.
            # Means map_size / pixels_per_meter is too small for the arena.
            half_extent = 0.5 * self.size / self.pixels_per_meter
            print(
                f"[redetect] DROPPED '{label}' at world_xy=({world_xy[0]:+.2f},{world_xy[1]:+.2f}) "
                f"-> px=({row},{col}) outside the {self.size}px canvas "
                f"(+/-{half_extent:.1f} m). Increase map_size or lower pixels_per_meter."
            )
            return

        # Re-detection handling
        is_redetection = False
        best_match_dist = float("inf")
        best_match_obj = None
        for obj in self._tracked_objects:
            if obj["label"] == label:
                dist = np.linalg.norm(obj["world_xy"] - world_xy)
                if dist < self._redetection_dist_m and dist < best_match_dist:
                    best_match_dist = dist
                    best_match_obj = obj

        if best_match_obj is not None:
            is_redetection = True
            best_match_obj["score"] = max(best_match_obj["score"], score)
            if best_match_dist >= self._centroid_dirty_dist_m:
                best_match_obj["dirty"] = True
            best_match_obj["world_xy"] = world_xy.copy()
            if room is not None:
                best_match_obj["room"] = room
            dbg(f"[redetect] MERGED '{label}' (dist={best_match_dist:.2f}m, score={score:.3f})")

        if not is_redetection:
            self._tracked_objects.append(
                {"label": label, "world_xy": world_xy.copy(), "score": score,
                 "room": room, "dirty": True, "signal_cache": None}
            )
            dbg(f"[redetect] NEW '{label}' (score={score:.3f}, total={len(self._tracked_objects)})")

    def _place_gaussian(
        self,
        target_map: np.ndarray,
        center_row: int,
        center_col: int,
        peak_value: float,
        sigma_px: float,
        radius_px: int,
    ) -> None:
        """Stamp a circular 2D gaussian onto *target_map* using max semantics."""
        h, w = target_map.shape

        r_min = max(0, center_row - radius_px)
        r_max = min(h, center_row + radius_px + 1)
        c_min = max(0, center_col - radius_px)
        c_max = min(w, center_col + radius_px + 1)

        if r_min >= r_max or c_min >= c_max:
            return

        rows = np.arange(r_min, r_max)
        cols = np.arange(c_min, c_max)
        rr, cc = np.meshgrid(rows, cols, indexing="ij")

        dist_sq = (rr - center_row) ** 2 + (cc - center_col) ** 2
        gaussian = peak_value * np.exp(-dist_sq / (2.0 * sigma_px ** 2))
        gaussian[dist_sq > radius_px ** 2] = 0.0  # hard circular cutoff

        # Max update: keep the higher of old / new at each pixel
        patch = target_map[r_min:r_max, c_min:c_max]
        target_map[r_min:r_max, c_min:c_max] = np.maximum(patch, gaussian)



    #  Crop-returning signal computations (used by recompute_value_map)
    def _compute_gaussian_crop(
        self,
        center_row: int,
        center_col: int,
        peak_value: float,
        sigma_px: float,
        radius_px: int,
    ) -> Optional[Tuple[int, int, int, int, np.ndarray]]:
        """
        Compute an isotropic gaussian signal crop.

        Returns:
            ``(r0, r1, c0, c1, signal_crop)`` or ``None`` if out of bounds.
        """
        H = W = self.size
        r0 = max(0, center_row - radius_px)
        r1 = min(H, center_row + radius_px + 1)
        c0 = max(0, center_col - radius_px)
        c1 = min(W, center_col + radius_px + 1)
        if r0 >= r1 or c0 >= c1:
            return None

        rows = np.arange(r0, r1)
        cols = np.arange(c0, c1)
        rr, cc = np.meshgrid(rows, cols, indexing="ij")
        dist_sq = (rr - center_row) ** 2 + (cc - center_col) ** 2
        gaussian = peak_value * np.exp(-dist_sq / (2.0 * sigma_px ** 2))
        gaussian[dist_sq > radius_px ** 2] = 0.0
        return (r0, r1, c0, c1, gaussian.astype(np.float32))


    def _compute_fmm_crop(
        self,
        center_row: int,
        center_col: int,
        peak_value: float,
        sigma_px: float,
        radius_px: int,
        free_mask: np.ndarray,
    ) -> Optional[Tuple[int, int, int, int, np.ndarray]]:
        """
        Wall-respecting signal via Fast Marching Method (Flood Fill)

        When the centroid falls inside occupied space (common — it *is* the
        detected object's centre) every free cell within
        ``_fmm_seed_radius_m`` of the centroid is used as an FMM seed.  This
        removes the side-bias of the old nearest-single-cell snap and lets the
        signal propagate symmetrically around furniture-sized objects.

        Returns:
            ``(r0, r1, c0, c1, signal_crop)`` or ``None`` on failure.
        """
        H = W = self.size

        # Crop to local window (centre ± radius_px with margin)
        # e.g. max_prop=6 m, ppm=20 → radius_px=120, margin=12, crop ≈ 262×262
        margin = max(10, int(radius_px * 0.1))
        r0 = max(0, center_row - radius_px - margin)
        r1 = min(H, center_row + radius_px + margin + 1)
        c0 = max(0, center_col - radius_px - margin)
        c1 = min(W, center_col + radius_px + margin + 1)
        if r0 >= r1 or c0 >= c1:
            return None

        # Local seed coords inside the crop
        lr = center_row - r0
        lc = center_col - c0

        # Speed map: free = 1.0, obstacle/unexplored = 1e-6 (effectively
        # impassable; skfmm requires strictly positive speed).
        local_free = free_mask[r0:r1, c0:c1]
        speed = np.where(local_free > 0, 1.0, 1e-6).astype(np.float64)

        # Build level-set function (phi < 0 = seed cells)
        phi = np.ones_like(speed)

        if speed[lr, lc] >= 0.5:
            # Centroid is already in free space — single seed
            phi[lr, lc] = -1.0
        else:
            # Centroid is inside an obstacle (common — it IS the detected
            # object's centre).  Use morphological dilation to find the
            # boundary cells: free cells immediately adjacent to the
            # occupied cells near the centroid.  This seeds the FMM wave
            # at the object's surface rather than at arbitrary free cells.
            seed_radius_px = int(self._fmm_seed_radius_m * self.pixels_per_meter)

            # Identify occupied cells within the seed radius of centroid
            rows_idx = np.arange(speed.shape[0])
            cols_idx = np.arange(speed.shape[1])
            rr, cc = np.meshgrid(rows_idx, cols_idx, indexing="ij")
            l1_dist = np.abs(rr - lr) + np.abs(cc - lc)
            nearby_occupied = (l1_dist <= seed_radius_px) & (speed < 0.5)

            if nearby_occupied.any():
                # Dilate the occupied region by 1 pixel (3×3 kernel)
                occupied_u8 = nearby_occupied.astype(np.uint8)
                kernel = np.ones((3, 3), dtype=np.uint8)
                dilated = cv2.dilate(occupied_u8, kernel, iterations=1)
                # Boundary = dilated minus original, intersected with free space
                boundary = (dilated > 0) & ~nearby_occupied & (speed > 0.5)
                seed_mask = boundary
            else:
                seed_mask = np.zeros_like(speed, dtype=bool)

            if not seed_mask.any():
                # No boundary cells found — fall back to isotropic gaussian
                # rather than seeding the entire crop (which would produce
                # uniform travel time = 0 everywhere, meaningless).
                return self._compute_gaussian_crop(
                    center_row, center_col, peak_value, sigma_px, radius_px
                )

            phi[seed_mask] = -1.0

        # Run FMM — O(n log n) on the crop, optimised C under the hood
        try:
            travel = skfmm.travel_time(phi, speed, dx=1.0)
        except Exception:
            # Seed is isolated, fall back to isotropic gaussian
            print(f"WARNING: FMM failed for signal placement at ({center_row}, {center_col}) — likely isolated seed. Falling back to gaussian.")
            return self._compute_gaussian_crop(
                center_row, center_col, peak_value, sigma_px, radius_px
            )

        # Decay over geodesic distance + hard cutoff
        if self._fmm_no_decay:
            # Flat value: every reachable cell within radius gets full score
            signal = np.where(
                (travel <= radius_px) & (speed > 0.5),
                peak_value,
                0.0,
            )
        else:
            # Gaussian decay
            signal = peak_value * np.exp(-(travel ** 2) / (2.0 * sigma_px ** 2))
            signal[travel > radius_px] = 0.0
            signal[speed < 0.5] = 0.0  # zero wall / unexplored cells

        return (r0, r1, c0, c1, signal.astype(np.float32))


    def _compute_raycast_crop(
        self,
        center_row: int,
        center_col: int,
        peak_value: float,
        radius_px: int,
        free_mask: np.ndarray,
    ) -> Optional[Tuple[int, int, int, int, np.ndarray]]:
        """Line-of-sight raycasting: flat score, blocked by obstacles.

        Casts rays outward from **every seed** in all directions.  Each ray
        walks pixel-by-pixel until it hits an obstacle or the radius limit.
        All directly-visible free cells within ``radius_px`` get the full
        ``peak_value``.  Unlike FMM, signals do **not** wrap around corners.

        When the centroid is in free space there is a single seed (the
        centroid itself).  When the centroid is inside occupied space,
        morphological dilation is used to find **all** free-space boundary
        cells around the obstacle — exactly like the FMM mode — so that
        rays emanate from every side of the object, not just the nearest
        free pixel.

        Returns:
            ``(r0, r1, c0, c1, signal_crop)`` or ``None`` on failure.
        """
        H = W = self.size

        margin = max(10, int(radius_px * 0.1))
        r0 = max(0, center_row - radius_px - margin)
        r1 = min(H, center_row + radius_px + margin + 1)
        c0 = max(0, center_col - radius_px - margin)
        c1 = min(W, center_col + radius_px + margin + 1)
        if r0 >= r1 or c0 >= c1:
            return None

        lr = center_row - r0  # local row of centroid in crop
        lc = center_col - c0  # local col of centroid in crop

        local_free = free_mask[r0:r1, c0:c1]
        lh, lw = local_free.shape

        # --- Determine seed origins ---
        if local_free[lr, lc] >= 0.5:
            # Centroid is in free space — single seed
            origins = [(lr, lc)]
        else:
            # Centroid is in non-free space — use morphological dilation
            # to find ALL boundary free cells around the occupied region
            seed_radius_px = int(self._fmm_seed_radius_m * self.pixels_per_meter)
            rows_idx = np.arange(lh)
            cols_idx = np.arange(lw)
            rr, cc = np.meshgrid(rows_idx, cols_idx, indexing="ij")
            l1_dist = np.abs(rr - lr) + np.abs(cc - lc)
            nearby_occupied = (l1_dist <= seed_radius_px) & (local_free < 0.5)

            if nearby_occupied.any():
                occupied_u8 = nearby_occupied.astype(np.uint8)
                kernel = np.ones((3, 3), dtype=np.uint8)
                dilated = cv2.dilate(occupied_u8, kernel, iterations=1)
                boundary = (dilated > 0) & ~nearby_occupied & (local_free >= 0.5)
                seed_coords = np.argwhere(boundary)  # (N, 2) → [row, col]
                origins = [(int(s[0]), int(s[1])) for s in seed_coords]
            else:
                origins = []

            if not origins:
                # No free space nearby — signal is latent (will be
                # recomputed when free space appears via dirty flag)
                return None

        # --- Cast rays from every seed ---
        n_rays = max(720, int(2 * np.pi * radius_px * 1.5))
        angles = np.linspace(0, 2 * np.pi, n_rays, endpoint=False)
        cos_a = np.cos(angles)
        sin_a = np.sin(angles)

        signal = np.zeros((lh, lw), dtype=np.float32)

        for origin_r, origin_c in origins:
            signal[origin_r, origin_c] = peak_value

            active = np.ones(n_rays, dtype=bool)
            for step in range(1, radius_px + 1):
                rs = np.round(origin_r + cos_a * step).astype(np.intp)
                cs = np.round(origin_c + sin_a * step).astype(np.intp)

                # Deactivate rays that left the crop
                oob = (rs < 0) | (rs >= lh) | (cs < 0) | (cs >= lw)
                active &= ~oob
                if not active.any():
                    break

                # Deactivate rays that hit an obstacle / unexplored cell
                hit = np.zeros(n_rays, dtype=bool)
                valid_idx = np.where(active)[0]
                hit[valid_idx] = local_free[rs[valid_idx], cs[valid_idx]] < 0.5
                active &= ~hit

                # Paint surviving rays
                paint_idx = np.where(active)[0]
                if paint_idx.size > 0:
                    signal[rs[paint_idx], cs[paint_idx]] = peak_value

        # --- Fill sampling holes -------------------------------------------
        # Ray positions are rounded to integer pixels, so at large radii two
        # adjacent rays land on the SAME pixel and leave their neighbour
        # unpainted -- isolated unpainted specks inside an otherwise visible
        # region (measured: ~60 px, all at 5.1-6.0 m from the seed, none nearer).
        # Raising the ray count would fix it by brute force but raycast is
        # already the slowest mode; closing the holes costs ~1 ms instead.
        # The fill is masked to free cells, so it can never paint through a wall
        # or extend the signal past its line-of-sight boundary.
        painted = (signal > 0).astype(np.uint8)
        closed = cv2.morphologyEx(painted, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        holes = (closed > 0) & (painted == 0) & (local_free >= 0.5)
        if holes.any():
            signal[holes] = peak_value

        return (r0, r1, c0, c1, signal)


    def recompute_value_map(self) -> None:
        """
        Rebuild ``_value_map`` using cached per-object signal crops.

        Only objects marked "dirty" (new, score-improved, or whose
        propagation radius overlaps newly-freed cells) have their FMM / gaussian
        recomputed.  Clean objects reuse their cached crop, making the per-step
        cost proportional to the number of changed objects rather than all of
        them.

        Call this once per step after ``add_object_signal``.
        """
        # Handling case of no detected objects
        if not self._tracked_objects:
            self._value_map.fill(0)
            self._prev_free_mask = None
            return

        # Compute the free-space mask once for the whole step
        free_mask = self._get_free_space_mask()

        # Mark objects whose propagation area overlaps newly-freed cells
        self._mark_dirty_objects(free_mask)
        self._prev_free_mask = free_mask.copy()

        # Recompute signal crops only for dirty objects
        radius_px = int(self._max_propagation_m * self.pixels_per_meter)
        for obj in self._tracked_objects:
            if not obj.get("dirty", True):
                continue  # clean — reuse cached crop

            row, col = self._world_to_map_pixel(obj["world_xy"])
            if row < 0 or row >= self.size or col < 0 or col >= self.size:
                obj["signal_cache"] = None
                obj["dirty"] = False
                continue

            # Adaptive sigma: higher-scoring objects influence a larger area
            sigma_min_m = max(0.5, self._decay_sigma_m * 0.5)
            sigma_max_m = max(sigma_min_m + 0.1, self._decay_sigma_m * 2.5)
            weight = np.sqrt(np.clip(obj["score"], 0.0, 1.0))
            sigma_m = sigma_min_m + (sigma_max_m - sigma_min_m) * weight
            sigma_px = sigma_m * self.pixels_per_meter

            # Crop cost scales with radius^2 (FMM runs over every cell in it), so
            # PROPAGATION_SIGMA_CUTOFF can stop paying for cells the gaussian has
            # already decayed to nothing. Off by default: _max_propagation_m is
            # then the only ceiling, as upstream. Flat modes have no decay, so
            # they always keep the full radius.
            flat = self._fmm_no_decay or self._signal_mode == "raycast"
            if flat or self._propagation_sigma_cutoff <= 0:
                eff_radius_px = radius_px
            else:
                cutoff_px = np.ceil(self._propagation_sigma_cutoff * sigma_px)
                eff_radius_px = int(min(radius_px, max(1.0, cutoff_px)))

            if self._signal_mode == "fmm":
                crop = self._compute_fmm_crop(
                    row, col, obj["score"], sigma_px, eff_radius_px, free_mask
                )
            elif self._signal_mode == "raycast":
                crop = self._compute_raycast_crop(
                    row, col, obj["score"], eff_radius_px, free_mask
                )
            else:
                crop = self._compute_gaussian_crop(
                    row, col, obj["score"], sigma_px, eff_radius_px
                )

            obj["signal_cache"] = crop
            obj["dirty"] = False

        # ---- Aggregate cached crops into per-label maps ----
        self._label_signal_maps = {}
        for obj in self._tracked_objects:
            cache = obj.get("signal_cache")
            if cache is None:
                continue
            label = obj["label"]
            if label not in self._label_signal_maps:
                self._label_signal_maps[label] = np.zeros(
                    (self.size, self.size), dtype=np.float32
                )
            r0, r1, c0, c1, crop = cache
            patch = self._label_signal_maps[label][r0:r1, c0:c1]
            np.maximum(patch, crop, out=patch)

        # ---- Aggregate per-label maps into the multi-channel value map ----
        total = np.zeros((self.size, self.size), dtype=np.float32)
        count = np.zeros((self.size, self.size), dtype=np.float32)

        for label_map in self._label_signal_maps.values():
            nonzero = label_map > 0
            total[nonzero] += label_map[nonzero]
            count[nonzero] += 1

        # Zero out unexplored regions
        if self._obstacle_map is not None:
            explored = self._obstacle_map.explored_area
            total[explored == 0] = 0
            count[explored == 0] = 0

        with np.errstate(divide="ignore", invalid="ignore"):
            avg = np.where(count > 0, total / count, 0.0)
        avg = np.clip(avg, 0.0, 1.0).astype(np.float32)

        # Broadcast to all value channels
        for c in range(self._value_channels):
            self._value_map[..., c] = avg


    def _get_free_space_mask(self) -> np.ndarray:
        """Return a binary mask where 1 = navigable free space, 0 = obstacle or unexplored.

        Falls back to all-ones if no obstacle map is attached.
        """
        # If no obstacle map loaded, define as same size as occupancy grid where free space is 1 and obstacles/unexplored are 0 (initialise as all free)
        if self._obstacle_map is None:
            return np.ones((self.size, self.size), dtype=np.float32)
        # explored_area is True where explored; _map (obstacle map) is True where obstacles exist
        # Therefore, free space is where explored area is true and obstacle map is false
        free = self._obstacle_map.explored_area & ~self._obstacle_map._map
        return free.astype(np.float32)


    def apply_free_space_mask(self) -> None:
        """Zero out value-map cells that sit inside obstacles or unexplored space.

        Call this after ``recompute_value_map()`` to prevent signals from
        bleeding through walls.  This is the cheap O(1)-per-cell approach:
        generate gaussians normally, then multiply by the inverse occupancy grid.
        """
        free_mask = self._get_free_space_mask()
        # Mutliply the value map by the free space, so that unknown/occupied space is zeroed out
        # This doen't prevent signals from permeating through walls for frontier detection, but may be important if
        # following the field using gradients
        for c in range(self._value_channels):
            self._value_map[..., c] *= free_mask
        self._drop_small_blobs()

    def _drop_small_blobs(self) -> None:
        """Discard signal islands too small to be real.

        Masking by free space can cut a blob in two: when a wall is discovered
        where signal had already been propagated, everything beyond that wall is
        left stranded with no path back to the object that produced it. Small
        strays also come from ray/flood-fill quantisation at the propagation
        boundary. Either way an island of a few cells carries no usable
        information but still scores a frontier that happens to sit on it.

        Only 8-connected components smaller than ``MIN_BLOB_AREA_M2`` are
        removed, so a genuine narrow sliver seen through a doorway survives.

        Embodied-RPV-NOTE: added for TurtleBot4 deployment. This post-process has
        no upstream equivalent, so it is DISABLED by default
        (``MIN_BLOB_AREA_M2=0``); the TurtleBot4 launcher sets 0.1. Note this is a cleanup, not a
        fix for the underlying staleness -- a LARGE stranded region is left
        alone by design (see ``_mark_dirty_objects``, which only reacts to cells
        becoming free, never to cells becoming occupied).
        """
        min_area_m2 = float(os.environ.get("MIN_BLOB_AREA_M2", 0.0))
        if min_area_m2 <= 0:
            return
        min_area_px = max(1, int(round(min_area_m2 * self.pixels_per_meter**2)))

        occupied = (self._value_map[..., 0] > 0).astype(np.uint8)
        if not occupied.any():
            return
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(occupied, connectivity=8)
        if n_labels <= 1:
            return  # background only

        areas = stats[1:, cv2.CC_STAT_AREA]
        small = np.where(areas < min_area_px)[0] + 1  # +1: skip the background label
        if small.size == 0:
            return

        self._value_map[np.isin(labels, small)] = 0.0
        # Strays reappear at the propagation boundary on most steps, so only log
        # when the count changes -- otherwise this floods an episode's log.
        if small.size != self._last_blob_drop_count:
            dbg(
                f"[blob-gate] dropped {small.size} island(s) < {min_area_m2:.2f} m^2 "
                f"({int(areas[small - 1].sum())} px total)"
            )
            self._last_blob_drop_count = small.size


    def _mark_dirty_objects(self, new_free_mask: np.ndarray) -> None:
        """
        Mark tracked objects that need signal recomputation.

        An object is marked dirty when its propagation box overlaps cells that
        changed from non-free to free since the last step (i.e. frontier
        expansion into its neighbourhood).  New objects and score-improved
        re-detections are already marked dirty in ``add_object_signal``.
        """
        if self._prev_free_mask is None:
            # First step — everything is dirty by default
            return

        # Determine cells that are newly free since the last step
        newly_free = (self._prev_free_mask == 0) & (new_free_mask > 0)
        if not newly_free.any():
            return  # map unchanged — nothing new to propagate into

        newly_free_coords = np.argwhere(newly_free)  # (N, 2)
        radius_px = int(self._max_propagation_m * self.pixels_per_meter)

        for obj in self._tracked_objects:
            if obj.get("dirty", True):
                continue  # already scheduled for recompute
            row, col = self._world_to_map_pixel(obj["world_xy"])
            # Chebyshev box check (fast) — any newly-freed cell within the
            # propagation square triggers a recompute for this object
            in_box = (
                (np.abs(newly_free_coords[:, 0] - row) <= radius_px)
                & (np.abs(newly_free_coords[:, 1] - col) <= radius_px)
            )
            if in_box.any():
                obj["dirty"] = True


    #  Legacy FOV-cone API (kept for backward compatibility)
    def update_map(
        self,
        values: np.ndarray,
        depth: np.ndarray,
        tf_camera_to_episodic: np.ndarray,
        min_depth: float,
        max_depth: float,
        fov: float,
    ) -> None:
        """Updates the value map with the given depth image, pose, and value to use.

        Args:
            values: The value to use for updating the map.
            depth: The depth image to use for updating the map; expected to be already
                normalized to the range [0, 1].
            tf_camera_to_episodic: The transformation matrix from the episodic frame to
                the camera frame.
            min_depth: The minimum depth value in meters.
            max_depth: The maximum depth value in meters.
            fov: The field of view of the camera in RADIANS.
        """
        assert (
            len(values) == self._value_channels
        ), f"Incorrect number of values given ({len(values)}). Expected {self._value_channels}."

        curr_map = self._localize_new_data(depth, tf_camera_to_episodic, min_depth, max_depth, fov)

        # Fuse the new data with the existing data
        self._fuse_new_data(curr_map, values)

        if RECORDING:
            idx = len(glob.glob(osp.join(RECORDING_DIR, "*.png")))
            img_path = osp.join(RECORDING_DIR, f"{idx:04d}.png")
            cv2.imwrite(img_path, (depth * 255).astype(np.uint8))
            with open(JSON_PATH, "r") as f:
                data = json.load(f)
            data[img_path] = {
                "values": values.tolist(),
                "tf_camera_to_episodic": tf_camera_to_episodic.tolist(),
                "min_depth": min_depth,
                "max_depth": max_depth,
                "fov": fov,
            }
            with open(JSON_PATH, "w") as f:
                json.dump(data, f)

    def sort_waypoints(
        self,
        waypoints: np.ndarray,
        radius: float,
        reduce_fn: Optional[Callable] = None,
        reduction: str = "median",
        clamp_negative: bool = False,
    ) -> Tuple[np.ndarray, List[float]]:
        """Selects the best waypoint from the given list of waypoints.

        Args:
            waypoints (np.ndarray): An array of 2D waypoints to choose from.
            radius (float): The radius in meters to use for selecting the best waypoint.
            reduce_fn (Callable, optional): The function to use for reducing the values
                within the given radius. Defaults to np.max.
            reduction (str): How ``pixel_value_within_radius`` collapses the disc of
                pixels around a waypoint. ``"median"`` is the upstream default and
                what the published numbers used. ``"max"`` reports the strongest
                signal that reaches the waypoint instead -- sparse point-signal
                maps median to ~0 because most cells in the disc carry no signal.
                Embodied-RPV-NOTE: the TurtleBot4 policy selects ``"max"``.
            clamp_negative (bool): Treat the ``-1`` "no signal in the disc"
                sentinel as ``0.0``. Off by default (upstream behaviour).
                Embodied-RPV-NOTE: the TurtleBot4 policy turns this ON -- with
                sparse signals the sentinel makes every waypoint tie at -1, so
                ``np.argsort`` degenerates to contour-detection order and the
                "value is not much worse than last" check in ``BaseITMPolicy``
                (``curr_value + 0.01 > self._last_value``) latches onto the first
                frontier forever.

        Returns:
            Tuple[np.ndarray, List[float]]: A tuple of the sorted waypoints and
                their corresponding values.
        """
        radius_px = int(radius * self.pixels_per_meter)

        def get_value(point: np.ndarray) -> Union[float, Tuple[float, ...]]:
            x, y = point
            px = int(-x * self.pixels_per_meter) + self._episode_pixel_origin[0]
            py = int(-y * self.pixels_per_meter) + self._episode_pixel_origin[1]
            point_px = (self._value_map.shape[0] - px, py)
            # pixel_value_within_radius returns -1 when no cell in the disc carries
            # signal; clamp_negative reads that as "no signal here" = 0.0 (see the
            # docstring for why that matters on sparse maps).
            floor = 0.0 if clamp_negative else -np.inf
            all_values = [
                max(
                    floor,
                    float(
                        pixel_value_within_radius(
                            self._value_map[..., c], point_px, radius_px, reduction=reduction
                        )
                    ),
                )
                for c in range(self._value_channels)
            ]
            if len(all_values) == 1:
                return all_values[0]
            return tuple(all_values)

        values = [get_value(point) for point in waypoints]

        if self._value_channels > 1:
            assert reduce_fn is not None, "Must provide a reduction function when using multiple value channels."
            values = reduce_fn(values)

        # Use np.argsort to get the indices of the sorted values
        sorted_inds = np.argsort([-v for v in values])  # type: ignore
        sorted_values = [values[i] for i in sorted_inds]
        sorted_frontiers = np.array([waypoints[i] for i in sorted_inds])

        return sorted_frontiers, sorted_values

    def peak_world_xy_within_radius(
        self, center_world_xy: np.ndarray, radius_m: float
    ) -> Optional[np.ndarray]:
        """Episodic-world (x, y) of the highest-value cell within ``radius_m``.

        Returns ``None`` when there is no positive signal in that neighbourhood
        (e.g. early exploration before any detection). Used to aim a narrow-FOV
        camera at the most promising region near a goal instead of straight at
        the goal point, so off-axis targets are not missed.
        """
        reduced = np.max(self._value_map, axis=-1)
        center_row, center_col = self._world_to_map_pixel(
            np.asarray(center_world_xy, dtype=float)
        )
        radius_px = max(1, int(radius_m * self.pixels_per_meter))

        r0 = max(0, center_row - radius_px)
        r1 = min(self.size, center_row + radius_px + 1)
        c0 = max(0, center_col - radius_px)
        c1 = min(self.size, center_col + radius_px + 1)
        if r0 >= r1 or c0 >= c1:
            return None

        rows = np.arange(r0, r1)[:, None]
        cols = np.arange(c0, c1)[None, :]
        dist_sq = (rows - center_row) ** 2 + (cols - center_col) ** 2
        masked = np.where(dist_sq <= radius_px**2, reduced[r0:r1, c0:c1], 0.0)

        if masked.max() <= 0:
            return None

        local_row, local_col = np.unravel_index(int(np.argmax(masked)), masked.shape)
        peak_row = r0 + int(local_row)
        peak_col = c0 + int(local_col)

        # Inverse of _world_to_map_pixel.
        px = self.size - 1 - peak_row
        x = -(px - self._episode_pixel_origin[0]) / self.pixels_per_meter
        y = -(peak_col - self._episode_pixel_origin[1]) / self.pixels_per_meter
        return np.array([x, y], dtype=float)

    def _world_to_vis_rc(self, pt: np.ndarray) -> Tuple[float, float]:
        """World (x, y) in metres -> (row, col) in *flipped* full-canvas pixels.

        Float-precision equivalent of ``TrajectoryVisualizer._metric_to_pixel``
        (which rounds to int32). Sub-pixel precision matters here because these
        coordinates are subsequently multiplied by the output scale factor,
        which magnifies any rounding done at canvas resolution.
        """
        row = float(self._episode_pixel_origin[0]) - float(pt[0]) * self.pixels_per_meter
        col = float(self._episode_pixel_origin[1]) - float(pt[1]) * self.pixels_per_meter
        return row, col

    def visualize(
        self,
        markers: Optional[List[Tuple[np.ndarray, Dict[str, Any]]]] = None,
        reduce_fn: Callable = lambda i: np.max(i, axis=-1),
        obstacle_map: Optional["ObstacleMap"] = None,  # type: ignore # noqa: F821
    ) -> np.ndarray:
        """Return an image representation of the map.

        Occupied / unexplored cells are drawn in dark grey so that it is
        visually obvious the signal does not permeate through obstacles.

        Rendering runs in three stages, in this order for specific reasons:

        1. Crop the *scalar* field (not the RGB) to the explored bounding box.
        2. Resample it to a FIXED output canvas (``VALUE_MAP_OUTPUT_PX``). The
           scalar field is resampled bilinearly and the occupancy mask with
           NEAREST, then colour-mapped and composited -- interpolating the RGB
           instead would blend dark-grey walls into the inferno ramp and produce
           colours that correspond to no value at all.
        3. Draw every overlay (trajectory, agent, frontier rings, object dots)
           in output-pixel space, sized as a FRACTION of the output canvas.

        Because the canvas is fixed, a marker occupying a constant fraction of
        the frame corresponds to a world-space radius that grows linearly with
        the explored extent: markers stay the same size on screen no matter how
        far the auto-crop has zoomed out.
        """
        reduced_map = reduce_fn(self._value_map).copy()

        # Build a mask of cells that are not free navigable space
        # (obstacles + unexplored).  These will be rendered distinctly.
        crop_src = self._obstacle_map if obstacle_map is None else obstacle_map
        occupied_mask_raw: Optional[np.ndarray] = None
        if crop_src is not None:
            free = crop_src.explored_area & ~crop_src._map
            occupied_mask_raw = ~free  # True where obstacle OR unexplored
            reduced_map[occupied_mask_raw] = 0

        val = np.flipud(reduced_map).astype(np.float32)
        occ = (
            np.flipud(occupied_mask_raw)
            if occupied_mask_raw is not None
            else np.zeros(val.shape, dtype=bool)
        )

        # ---- 1. crop the scalar field to the explored region --------------
        # Without this the signal is a tiny speck in a sea of grey. r0/c0 are the
        # crop origin and are needed below to map world coords into the crop.
        H, W = val.shape[:2]
        r0, c0, r1, c1 = 0, 0, H, W
        if crop_src is not None:
            explored = np.flipud(crop_src.explored_area)  # match the flipped vis
            rows_any = np.any(explored, axis=1)
            cols_any = np.any(explored, axis=0)
            if rows_any.any() and cols_any.any():
                r_idx = np.where(rows_any)[0]
                c_idx = np.where(cols_any)[0]
                # Padding around the explored area, in METRES so the framing is
                # the same at any map resolution.
                pad = int(round(float(os.environ.get("VALUE_MAP_CROP_PAD_M", 0.8)) * self.pixels_per_meter))
                r0, r1 = max(0, r_idx[0] - pad), min(H, r_idx[-1] + pad + 1)
                c0, c1 = max(0, c_idx[0] - pad), min(W, c_idx[-1] + pad + 1)

                # Floor the crop at a minimum extent: on the first few steps the
                # explored area is a couple of metres across, and blowing that up
                # to the full canvas makes the early frames unreadably zoomed in
                # (and the framing lurch wildly between steps).
                min_px = int(round(float(os.environ.get("VALUE_MAP_MIN_EXTENT_M", 6.0)) * self.pixels_per_meter))
                # Square the crop so the aspect ratio is 1:1, then apply the
                # minimum. Both grow symmetrically about the crop centre.
                side = max(r1 - r0, c1 - c0, min_px)
                rc, cc = (r0 + r1) / 2.0, (c0 + c1) / 2.0
                r0 = int(round(rc - side / 2.0))
                c0 = int(round(cc - side / 2.0))
                r1, c1 = r0 + side, c0 + side

        # Slice with clamping, then pad back out so the crop is exactly
        # `side` x `side` even when it runs off the canvas edge. Padding rather
        # than clamping keeps the aspect ratio at 1:1, so the resize below never
        # stretches the map.
        sr0, sc0 = max(0, r0), max(0, c0)
        sr1, sc1 = min(H, r1), min(W, c1)
        val = val[sr0:sr1, sc0:sc1]
        occ = occ[sr0:sr1, sc0:sc1]
        pad_top, pad_left = sr0 - r0, sc0 - c0
        pad_bot, pad_right = r1 - sr1, c1 - sc1
        if any(p > 0 for p in (pad_top, pad_left, pad_bot, pad_right)):
            pad_spec = ((pad_top, pad_bot), (pad_left, pad_right))
            val = np.pad(val, pad_spec, constant_values=0.0)
            occ = np.pad(occ, pad_spec, constant_values=True)  # off-canvas = unknown

        # ---- 2. resample to a fixed output canvas -------------------------
        out_px = max(64, int(os.environ.get("VALUE_MAP_OUTPUT_PX", 1024)))
        src_px = max(1, val.shape[0])
        scale = out_px / float(src_px)
        # Bilinear when magnifying the (already continuous) propagated field;
        # NEAREST when minifying, and always for the mask, so wall edges stay
        # crisp and no cell is invented between two real ones.
        smooth = os.environ.get("VALUE_MAP_SMOOTH", "1").lower() in ("1", "true", "yes")
        val_interp = cv2.INTER_LINEAR if (smooth and scale > 1.0) else cv2.INTER_AREA if scale < 1.0 else cv2.INTER_NEAREST
        val = cv2.resize(val, (out_px, out_px), interpolation=val_interp)
        occ = cv2.resize(occ.astype(np.uint8), (out_px, out_px), interpolation=cv2.INTER_NEAREST).astype(bool)

        # ---- colour-map ---------------------------------------------------
        zero_mask = val <= 0.0
        if os.environ.get("VALUE_MAP_FIXED_SCALE", "0").lower() in ("1", "true", "yes"):
            # Absolute [0, 1] ramp: a given colour means the same score in every
            # frame and every run, at the cost of contrast when scores are low.
            map_img = monochannel_to_inferno_rgb(np.clip(val, 0.0, 1.0))
        else:
            # Per-frame auto-stretch (historical behaviour): maximum contrast,
            # but colours are NOT comparable across frames or runs.
            max_val = float(np.max(val))
            if max_val > 0:
                val = np.where(zero_mask, max_val, val)  # temp: avoid skewing colourmap
            map_img = monochannel_to_inferno_rgb(val)
        # White for zero-signal free space
        map_img[zero_mask] = (255, 255, 255)
        # Dark grey for occupied / unexplored — makes wall-blocking visible
        map_img[occ] = (60, 60, 60)

        # ---- 3. overlays, in output-pixel space ---------------------------
        def to_out(pt: np.ndarray) -> Tuple[int, int]:
            """World (x, y) metres -> (x, y) pixel in the output canvas."""
            row, col = self._world_to_vis_rc(pt)
            return int(round((col - c0) * scale)), int(round((row - r0) * scale))

        def frac_px(frac: float, min_px: int = 1) -> int:
            return max(min_px, int(round(frac * out_px)))

        # Sizes as a fraction of the output canvas -> constant on-screen size.
        path_th = frac_px(float(os.environ.get("VALUE_MAP_PATH_FRAC", 0.004)), 2)
        agent_r = frac_px(float(os.environ.get("VALUE_MAP_AGENT_FRAC", 0.012)), 3)
        marker_r = frac_px(float(os.environ.get("VALUE_MAP_MARKER_FRAC", 0.016)), 4)
        marker_th = max(2, marker_r // 4)
        dot_r = frac_px(float(os.environ.get("VALUE_MAP_DOT_FRAC", 0.009)), 3)
        # The selected frontier gets a bigger ring plus a crosshair; colour alone
        # is not legible once the ring is only a few pixels across.
        sel_r = int(round(marker_r * 1.9))

        if len(self._camera_positions) > 0:
            # Trajectory. Drawn fresh each frame in output space rather than
            # through TrajectoryVisualizer's incremental mask cache -- that cache
            # is keyed to canvas resolution and would be invalidated every time
            # the crop changes. A few hundred polyline points is negligible.
            if len(self._camera_positions) > 1:
                pts = np.array([to_out(p) for p in self._camera_positions], dtype=np.int32)
                cv2.polylines(map_img, [pts], False, self._traj_vis.path_color, path_th, cv2.LINE_AA)

            # Agent: filled circle + heading tick.
            ax, ay = to_out(self._camera_positions[-1])
            cv2.circle(map_img, (ax, ay), agent_r, (255, 192, 15), -1, cv2.LINE_AA)
            head_len = agent_r * 2.5
            hx = int(round(ax - head_len * np.sin(self._last_camera_yaw)))
            hy = int(round(ay - head_len * np.cos(self._last_camera_yaw)))
            cv2.line(map_img, (ax, ay), (hx, hy), (0, 0, 0), max(2, agent_r // 2), cv2.LINE_AA)

        # Frontier / goal markers. `radius` / `thickness` in marker_kwargs are
        # ignored (they were metric sizes for the old native-resolution draw);
        # a truthy "selected" key promotes a marker to the emphasised style.
        if markers is not None:
            for pos, marker_kwargs in markers:
                color = marker_kwargs.get("color", (0, 0, 255))
                selected = bool(marker_kwargs.get("selected", False))
                mx, my = to_out(np.asarray(pos, dtype=float))
                if selected:
                    cv2.circle(map_img, (mx, my), sel_r, (255, 255, 255), marker_th + 2, cv2.LINE_AA)
                    cv2.circle(map_img, (mx, my), sel_r, color, marker_th, cv2.LINE_AA)
                    cv2.line(map_img, (mx - sel_r, my), (mx + sel_r, my), color, marker_th, cv2.LINE_AA)
                    cv2.line(map_img, (mx, my - sel_r), (mx, my + sel_r), color, marker_th, cv2.LINE_AA)
                else:
                    cv2.circle(map_img, (mx, my), marker_r, color, marker_th, cv2.LINE_AA)

        # Pin a dot on each detected object where it was seen on the map.
        for obj in self._tracked_objects:
            wxy = obj.get("world_xy")
            if wxy is None:
                continue
            ox, oy = to_out(np.asarray(wxy, dtype=float))
            if 0 <= ox < out_px and 0 <= oy < out_px:
                cv2.circle(map_img, (ox, oy), dot_r, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(map_img, (ox, oy), dot_r, (0, 0, 255), max(1, dot_r // 3), cv2.LINE_AA)

        return map_img

    # helper: draw a numbered marker at a world-space position on a *flipped* vis image
    def _draw_numbered_marker(
        self,
        img: np.ndarray,
        world_xy: np.ndarray,
        index: int,
        color: Tuple[int, int, int] = (255, 0, 0),
    ) -> np.ndarray:
        """Draw a circled number at *world_xy* on the (already flipped) vis image."""
        px = self._traj_vis._metric_to_pixel(world_xy)
        # _metric_to_pixel returns (row, col) but cv2 wants (x, y) = (col, row)
        cx, cy = int(px[1]), int(px[0])
        h, w = img.shape[:2]
        if 0 <= cx < w and 0 <= cy < h:
            num_str = str(index)
            r = 12 if len(num_str) < 2 else 16
            cv2.circle(img, (cx, cy), r, (255, 255, 255), -1)
            cv2.circle(img, (cx, cy), r, color, 2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            fs = 0.50 if len(num_str) < 2 else 0.42
            tsz = cv2.getTextSize(num_str, font, fs, 1)[0]
            cv2.putText(img, num_str, (cx - tsz[0] // 2, cy + tsz[1] // 2),
                        font, fs, color, 1, cv2.LINE_AA)
        return img

    def _process_local_data(self, depth: np.ndarray, fov: float, min_depth: float, max_depth: float) -> np.ndarray:
        """Using the FOV and depth, return the visible portion of the FOV.

        Args:
            depth: The depth image to use for determining the visible portion of the
                FOV.
        Returns:
            A mask of the visible portion of the FOV.
        """
        # Squeeze out the channel dimension if depth is a 3D array
        if len(depth.shape) == 3:
            depth = depth.squeeze(2)
        # Squash depth image into one row with the max depth value for each column
        depth_row = np.max(depth, axis=0) * (max_depth - min_depth) + min_depth

        # Create a linspace of the same length as the depth row from -fov/2 to fov/2
        angles = np.linspace(-fov / 2, fov / 2, len(depth_row))

        # Assign each value in the row with an x, y coordinate depending on 'angles'
        # and the max depth value for that column
        x = depth_row
        y = depth_row * np.tan(angles)

        # Get blank cone mask
        cone_mask = self._get_confidence_mask(fov, max_depth)

        # Convert the x, y coordinates to pixel coordinates
        x = (x * self.pixels_per_meter + cone_mask.shape[0] / 2).astype(int)
        y = (y * self.pixels_per_meter + cone_mask.shape[1] / 2).astype(int)

        # Create a contour from the x, y coordinates, with the top left and right
        # corners of the image as the first two points
        last_row = cone_mask.shape[0] - 1
        last_col = cone_mask.shape[1] - 1
        start = np.array([[0, last_col]])
        end = np.array([[last_row, last_col]])
        contour = np.concatenate((start, np.stack((y, x), axis=1), end), axis=0)

        # Draw the contour onto the cone mask, in filled-in black
        visible_mask = cv2.drawContours(cone_mask, [contour], -1, 0, -1)  # type: ignore

        if DEBUG:
            vis = cv2.cvtColor((cone_mask * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
            cv2.drawContours(vis, [contour], -1, (0, 0, 255), -1)
            for point in contour:
                vis[point[1], point[0]] = (0, 255, 0)
            if SAVE_VISUALIZATIONS:
                # Create visualizations directory if it doesn't exist
                if not os.path.exists("visualizations"):
                    os.makedirs("visualizations")
                # Expand the depth_row back into a full image
                depth_row_full = np.repeat(depth_row.reshape(1, -1), depth.shape[0], axis=0)
                # Stack the depth images with the visible mask
                depth_rgb = cv2.cvtColor((depth * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
                depth_row_full = cv2.cvtColor((depth_row_full * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
                vis = np.flipud(vis)
                new_width = int(vis.shape[1] * (depth_rgb.shape[0] / vis.shape[0]))
                vis_resized = cv2.resize(vis, (new_width, depth_rgb.shape[0]))
                vis = np.hstack((depth_rgb, depth_row_full, vis_resized))
                time_id = int(time.time() * 1000)
                cv2.imwrite(f"visualizations/{time_id}.png", vis)
            else:
                cv2.imshow("obstacle mask", vis)
                cv2.waitKey(0)

        return visible_mask

    def _localize_new_data(
        self,
        depth: np.ndarray,
        tf_camera_to_episodic: np.ndarray,
        min_depth: float,
        max_depth: float,
        fov: float,
    ) -> np.ndarray:
        # Get new portion of the map
        curr_data = self._process_local_data(depth, fov, min_depth, max_depth)

        # Rotate this new data to match the camera's orientation
        yaw = extract_yaw(tf_camera_to_episodic)
        if PLAYING:
            if yaw > 0:
                yaw = 0
            else:
                yaw = np.deg2rad(30)
        curr_data = rotate_image(curr_data, -yaw)

        # Determine where this mask should be overlaid
        cam_x, cam_y = tf_camera_to_episodic[:2, 3] / tf_camera_to_episodic[3, 3]

        # Convert to pixel units
        px = int(cam_x * self.pixels_per_meter) + self._episode_pixel_origin[0]
        py = int(-cam_y * self.pixels_per_meter) + self._episode_pixel_origin[1]

        # Overlay the new data onto the map
        curr_map = np.zeros_like(self._map)
        curr_map = place_img_in_img(curr_map, curr_data, px, py)

        return curr_map

    def _get_blank_cone_mask(self, fov: float, max_depth: float) -> np.ndarray:
        """Generate a FOV cone without any obstacles considered"""
        size = int(max_depth * self.pixels_per_meter)
        cone_mask = np.zeros((size * 2 + 1, size * 2 + 1))
        cone_mask = cv2.ellipse(  # type: ignore
            cone_mask,
            (size, size),  # center_pixel
            (size, size),  # axes lengths
            0,  # angle circle is rotated
            -np.rad2deg(fov) / 2 + 90,  # start_angle
            np.rad2deg(fov) / 2 + 90,  # end_angle
            1,  # color
            -1,  # thickness
        )
        return cone_mask

    def _get_confidence_mask(self, fov: float, max_depth: float) -> np.ndarray:
        """Generate a FOV cone with central values weighted more heavily"""
        if (fov, max_depth) in self._confidence_masks:
            return self._confidence_masks[(fov, max_depth)].copy()
        cone_mask = self._get_blank_cone_mask(fov, max_depth)
        adjusted_mask = np.zeros_like(cone_mask).astype(np.float32)
        for row in range(adjusted_mask.shape[0]):
            for col in range(adjusted_mask.shape[1]):
                horizontal = abs(row - adjusted_mask.shape[0] // 2)
                vertical = abs(col - adjusted_mask.shape[1] // 2)
                angle = np.arctan2(vertical, horizontal)
                angle = remap(angle, 0, fov / 2, 0, np.pi / 2)
                confidence = np.cos(angle) ** 2
                confidence = remap(confidence, 0, 1, self._min_confidence, 1)
                adjusted_mask[row, col] = confidence
        adjusted_mask = adjusted_mask * cone_mask
        self._confidence_masks[(fov, max_depth)] = adjusted_mask.copy()

        return adjusted_mask

    def _fuse_new_data(self, new_map: np.ndarray, values: np.ndarray) -> None:
        """Fuse the new data with the existing value and confidence maps.

        Args:
            new_map: The new new_map map data to fuse. Confidences are between
                0 and 1, with 1 being the most confident.
            values: The values attributed to the new portion of the map.
        """
        assert (
            len(values) == self._value_channels
        ), f"Incorrect number of values given ({len(values)}). Expected {self._value_channels}."

        if self._obstacle_map is not None:
            # If an obstacle map is provided, we will use it to mask out the
            # new map
            explored_area = self._obstacle_map.explored_area
            new_map[explored_area == 0] = 0
            self._map[explored_area == 0] = 0
            self._value_map[explored_area == 0] *= 0

        if self._fusion_type == "replace":
            # Ablation. The values from the current observation will overwrite any
            # existing values
            print("VALUE MAP ABLATION:", self._fusion_type)
            new_value_map = np.zeros_like(self._value_map)
            new_value_map[new_map > 0] = values
            self._map[new_map > 0] = new_map[new_map > 0]
            self._value_map[new_map > 0] = new_value_map[new_map > 0]
            return
        elif self._fusion_type == "equal_weighting":
            # Ablation. Updated values will always be the mean of the current and
            # new values, meaning that confidence scores are forced to be the same.
            print("VALUE MAP ABLATION:", self._fusion_type)
            self._map[self._map > 0] = 1
            new_map[new_map > 0] = 1
        else:
            assert self._fusion_type == "default", f"Unknown fusion type {self._fusion_type}"

        # Any values in the given map that are less confident than
        # self._decision_threshold AND less than the new_map in the existing map
        # will be silenced into 0s
        new_map_mask = np.logical_and(new_map < self._decision_threshold, new_map < self._map)
        new_map[new_map_mask] = 0

        if self._use_max_confidence:
            # For every pixel that has a higher new_map in the new map than the
            # existing value map, replace the value in the existing value map with
            # the new value
            higher_new_map_mask = new_map > self._map
            self._value_map[higher_new_map_mask] = values
            # Update the new_map map with the new new_map values
            self._map[higher_new_map_mask] = new_map[higher_new_map_mask]
        else:
            # Each pixel in the existing value map will be updated with a weighted
            # average of the existing value and the new value. The weight of each value
            # is determined by the current and new new_map values. The new_map map
            # will also be updated with using a weighted average in a similar manner.
            confidence_denominator = self._map + new_map
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                weight_1 = self._map / confidence_denominator
                weight_2 = new_map / confidence_denominator

            weight_1_channeled = np.repeat(np.expand_dims(weight_1, axis=2), self._value_channels, axis=2)
            weight_2_channeled = np.repeat(np.expand_dims(weight_2, axis=2), self._value_channels, axis=2)

            self._value_map = self._value_map * weight_1_channeled + values * weight_2_channeled
            self._map = self._map * weight_1 + new_map * weight_2

            # Because confidence_denominator can have 0 values, any nans in either the
            # value or confidence maps will be replaced with 0
            self._value_map = np.nan_to_num(self._value_map)
            self._map = np.nan_to_num(self._map)


def remap(value: float, from_low: float, from_high: float, to_low: float, to_high: float) -> float:
    """Maps a value from one range to another.

    Args:
        value (float): The value to be mapped.
        from_low (float): The lower bound of the input range.
        from_high (float): The upper bound of the input range.
        to_low (float): The lower bound of the output range.
        to_high (float): The upper bound of the output range.

    Returns:
        float: The mapped value.
    """
    return (value - from_low) * (to_high - to_low) / (from_high - from_low) + to_low


def replay_from_dir() -> None:
    with open(KWARGS_JSON, "r") as f:
        kwargs = json.load(f)
    with open(JSON_PATH, "r") as f:
        data = json.load(f)

    v = ValueMap(**kwargs)

    sorted_keys = sorted(list(data.keys()))

    for img_path in sorted_keys:
        tf_camera_to_episodic = np.array(data[img_path]["tf_camera_to_episodic"])
        values = np.array(data[img_path]["values"])
        depth = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        v.update_map(
            values,
            depth,
            tf_camera_to_episodic,
            float(data[img_path]["min_depth"]),
            float(data[img_path]["max_depth"]),
            float(data[img_path]["fov"]),
        )

        img = v.visualize()
        cv2.imshow("img", img)
        key = cv2.waitKey(0)
        if key == ord("q"):
            break


if __name__ == "__main__":
    if PLAYING:
        replay_from_dir()
        quit()

    v = ValueMap(value_channels=1)
    depth = cv2.imread("depth.png", cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
    img = v._process_local_data(
        depth=depth,
        fov=np.deg2rad(79),
        min_depth=0.5,
        max_depth=5.0,
    )
    cv2.imshow("img", (img * 255).astype(np.uint8))
    cv2.waitKey(0)

    num_points = 20

    x = [0, 10, 10, 0]
    y = [0, 0, 10, 10]
    angles = [0, np.pi / 2, np.pi, 3 * np.pi / 2]

    points = np.stack((x, y), axis=1)

    for pt, angle in zip(points, angles):
        tf = np.eye(4)
        tf[:2, 3] = pt
        tf[:2, :2] = get_rotation_matrix(angle)
        v.update_map(
            np.array([1]),
            depth,
            tf,
            min_depth=0.5,
            max_depth=5.0,
            fov=np.deg2rad(79),
        )
        img = v.visualize()
        cv2.imshow("img", img)
        key = cv2.waitKey(0)
        if key == ord("q"):
            break

