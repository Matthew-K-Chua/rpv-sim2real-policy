# rpv-sim2real-policy

Room-Mediated Co-occurrence (RPV), a zero-shot Object-Goal Navigation
(ObjectNav) policy, packaged for deployment on physical robots. RPV extends
Vision-Language Frontier Maps (VLFM) by assigning each detected object spatial
context through the medium of room labels: an open-vocabulary perception stack
segments and localises objects from RGB-D observations, maps each detection to a
Room Probability Vector, compares that vector against the target, and writes the
resulting semantic signal into a value map over which frontiers are ranked.

This repository is the policy half of the deployment reported in *Sim-to-Real
Room-Mediated Object-Goal Navigation with ROS 2 and Nav2*. It presents the policy
as an HTTP server, such that the perception and frontier scoring process is
detached from agent control. The ROS 2 half, which supplies the observations and
executes the chosen goal through Nav2, is a separate repository,
[rpv-ros2-bridge](https://github.com/Matthew-K-Chua/rpv-ros2-bridge).

We separate the two because `rclpy` is available only for Python 3.10 while the
perception stack requires Python 3.12. On our rig both halves run on the same
off-board GPU host and communicate over loopback, although nothing prevents the
ROS half from running elsewhere. 

## Contents

- [What the policy does](#what-the-policy-does)
- [Repository layout](#repository-layout)
- [Installation](#installation)
- [Running](#running)
- [Configuration](#configuration)
- [The policy interface](#the-policy-interface)
- [Citation and attribution](#citation-and-attribution)

## What the policy does

```
  rpv-ros2-bridge ── POST /rpv_policy ──> rpv_policy_server      :13000
                                             ├─ CLIP             :12182
                                             ├─ SAM 3            :12183
                                             ├─ YOLOE            :12184
                                             └─ ZoeDepth         :12185  (optional)
  operator console ─ GET /status ─────────> value map, obstacle map, annotated RGB, state
```

At each tick the server receives an RGB snapshot, a depth raster, an occupancy
grid, and an egocentric pose, and returns a single navigation goal. The sequence
is as follows.

1. **Detection.** The RGB frame is passed through the perception funnel. YOLOE
   produces lightweight coarse proposals from its configured vocabulary, and
   SAM 3 refines those proposals into instance masks such that poor detections
   are removed. The funnel can also be collapsed to SAM 3 alone, prompted
   directly from a short label list; see [the vocabularies](#the-vocabularies).
2. **Localisation.** Each surviving mask is back-projected through the depth
   raster and the camera extrinsics into the episodic frame. A detection closer
   to the base origin than `min_object_distance` is discarded, as that geometry
   is level with or behind the robot's leading edge and the back-projection
   cannot be trusted for it.
3. **Scoring.** CLIP scores each masked crop against the room lexicon, which
   yields a Room Probability Vector per detection, and that vector is compared
   against the target's own. The resulting scalar becomes the magnitude of the
   signal written into the value map.
4. **Propagation.** Each signal is propagated over the free space of the
   occupancy grid by geodesic flood fill, decaying with distance, such that the
   map carries a gradient rather than a set of isolated peaks.
5. **Selection.** Frontiers are extracted from the occupancy grid and ranked on
   the value map. The highest-valued frontier becomes the goal while the target
   is undetected; once the target has been detected and localised the goal
   becomes a point in front of it, and the policy calls stop on arriving within
   `pointnav_stop_radius`.

The occupancy grid is supplied by the ROS half rather than projected from the
policy's own depth, which is one of our deployment deviations and is discussed
below. A request carrying `is_first=true` clears the maps and pins the episodic
origin at the robot's current pose. That flag is the only path by which a target
reaches the policy, so a target that differs from the current one without it
forces a reset rather than being pursued silently under the previous maps.

The ZoeDepth server and the ROS node under `ros_nodes/` exist for a platform
whose camera publishes colour only, and turn that stream into a dense metric
raster registered to the colour frame by construction. They are not started when
the camera supplies real aligned depth, which is the default configuration.

## Repository layout

```
vlfm/
  policy/
    rpv_policy_server.py     the Flask server: POST /rpv_policy, GET /status, and the robot profiles
    rpv_policies.py          the RPV policy: the initial sweep, frontier selection, arrival
    itm_policy.py            value-map construction and frontier scoring
    base_objectnav_policy.py the perception funnel and the object map
    rpv_timing.py            per-stage step timing, behind RPV_TIMING=1
    utils/                   the point-navigation network the policy carries
  mapping/
    obstacle_map.py          the obstacle map and frontier extraction
    value_map.py             signal propagation, decay, and frontier ranking
    object_point_cloud_map.py   3D centroid tracking for detected objects
    base_map.py, traj_visualizer.py
  vlm/
    clip.py, sam3.py, yoloe.py, zoedepth.py   the model servers and their HTTP clients
    mask2former.py           client only; the stage is disabled and its server is not shipped
    depth_codec.py           the 16-bit PNG depth codec, whose encoder the ROS bridge imports
    room_types.py, lvis_classes.py, coco_classes.py, detections.py
    server_wrapper.py        the Flask plumbing shared by every model server
  utils/                     geometry, image helpers, and the debug trace
frontier_exploration_src/    vendored frontier detection, carrying our edits; install editable
data/
  shortvis.yaml              the SAM 3 prompt list used when YOLOE is bypassed
  room_types_lab.yaml        the room lexicon CLIP scores detections against
  lvis.yaml                  the 1203-class LVIS vocabulary, which conditions YOLOE
  pointnav_weights.pth       weights for the point-navigation network
  dummy_policy.pth           a placeholder policy, used only on the simulation path
scripts/
  launch_offboard_ai.sh      CLIP, SAM 3 and the policy server in one tmux session
  launch_zoedepth_server.sh  the monocular depth server, for a colour-only camera
ros_nodes/                   the ROS 2 side of ZoeDepth: a node and a launch file
environment.yml              the conda environment, rpv-emb
requirements.txt             its pip half
```

## Installation

A CUDA GPU is required. Our off-board host is a Dell Precision 5820 with an
RTX 3090 and 24 GB of VRAM; SAM 3's weights alone are approximately 3.4 GB, and
the stack has also run on an 8 GB card.

```bash
git clone https://github.com/Matthew-K-Chua/rpv-sim2real-policy.git ~/rpv-sim2real-policy
cd ~/rpv-sim2real-policy

conda env create -f environment.yml              # creates rpv-emb
conda activate rpv-emb
pip install -r requirements.txt
pip install -e frontier_exploration_src/         # the vendored copy, which carries our edits

git clone https://github.com/facebookresearch/sam3.git   # vendored by clone, and git-ignored
pip install -e sam3
```

Model weights are not committed. We create `checkpoints/`, which is git-ignored,
and populate it:

| File | Required for |
|---|---|
| `checkpoints/sam3.pt` | Every configuration |
| `checkpoints/yoloe-26x-seg.pt` | `SKIP_YOLOE=0`, which is the configuration we report |

CLIP (`openai/clip-vit-base-patch32`) and, on the monocular path,
`Intel/zoedepth-nyu` are retrieved by `transformers` on first use into
`~/.cache/huggingface`. `data/pointnav_weights.pth` is committed. We do not
install `vlfm` as a package, as the launcher scripts run it from the repository
root, which is why they `cd` there first.

We also install `earlyoom` to prevent memory lockup, though it is optional in stack:

```bash
sudo apt install earlyoom tmux && sudo systemctl enable --now earlyoom
```

The ROS half locates this checkout at `~/rpv-sim2real-policy` by default. Export
`RPV_POLICY_REPO=/path/to/it` otherwise.

## Running

In normal operation the ROS half's `scripts/launch_offboard.sh` calls this
repository's launcher with the robot's depth band and camera tilt already derived
from the robot profile, and places its tmux windows in the same session as SLAM,
Nav2 and the bridge, such that the whole host side starts and stops as a unit. To
start this half alone:

```bash
./scripts/launch_offboard_ai.sh            # tmux session offboard_ai: vlm-clip, vlm-sam3, policy
tmux attach -t offboard_ai                 # Ctrl-B w selects a window, Ctrl-B d detaches
tmux kill-session -t offboard_ai
```

To reproduce the configuration of the reported trials, which runs the full
two-stage funnel:

```bash
SKIP_YOLOE=0 YOLOE_CLASSES=data/lvis.yaml ./scripts/launch_offboard_ai.sh
```

**Run the launcher from a shell with no ROS sourced.** Sourcing ROS places
`/opt/ros/humble/lib/python3.10/site-packages` on `PYTHONPATH`, and a conda
Python 3.12 that inherits it imports ROS's `numpy` and `cv2` and fails in a
manner indistinguishable from a broken installation. This is the most likely
reason for a first bring-up failing on a machine where the identical stack has
worked previously, so the launcher scrubs `PYTHONPATH`, `LD_LIBRARY_PATH`,
`AMENT_PREFIX_PATH` and the remainder before activating conda. We never set
`SCRUB_ROS_ENV=0`.

The launcher also does three things that are worth knowing about:

1. It sets `NO_PROXY` for loopback. A configured `http_proxy` will otherwise
   attempt to proxy `127.0.0.1:13000`, and the resulting error names the proxy
   rather than the policy.
2. It pre-flights the ports and refuses to start if any one of them is already
   held. A stale server on `:13000` is invisible until the bridge POSTs into it
   and receives answers from the wrong policy.
3. It waits on each port in turn, as a port only begins listening once that
   model's weights have loaded, and it writes the policy's output to `logs/`.

For a colour-only camera, `scripts/launch_zoedepth_server.sh` starts the
monocular depth model on `:12185` and prints the ROS command for the node that
feeds it. The ROS half then runs with `DEPTH_SOURCE=zoedepth`.

## Configuration

The policy is configured entirely through environment variables, of which the
ROS half's launcher sets the robot-dependent ones. Deviations from the published
RPV and VLFM defaults are tagged `Embodied-RPV-NOTE` in the source
(`grep -rn "Embodied-RPV-NOTE" vlfm/ data/`), and the code's own defaults remain
the benchmark's such that simulation stays comparable.

### Geometry, which the ROS half derives per robot

| Variable | Default | Effect |
|---|---|---|
| `RPV_ROBOT` | `turtlebot4` | Selects the robot profile. An unset value is logged loudly rather than guessed, and must be set to `spot` on Spot |
| `RPV_MIN_DEPTH_M`, `RPV_MAX_DEPTH_M` | per profile | The depth band. **This band is authoritative**, and the values carried in the request are read once only, to warn on a disagreement |
| `RPV_CAMERA_PITCH_RAD`, `RPV_CAMERA_ROLL_RAD` | per profile | The camera tilt used when the request carries none, where positive pitch places the lens down |
| `RPV_SCAN_ARRIVAL_M`, `RPV_HEADING_RADIUS_M` | per profile | The distance at which the heading is re-aimed at the value-map peak, and the radius searched for that peak |
| `RPV_POLICY_HOST`, `RPV_POLICY_PORT` | per profile, `13000` | The bind address. Spot's profile binds loopback, as the bridge shares the host; the TurtleBot 4's binds `0.0.0.0` |

The profiles themselves are the only per-robot values in this repository:

| | Spot | TurtleBot 4 |
|---|---|---|
| `agent_radius` | 0.25 m | 0.19 m |
| `pointnav_stop_radius` | 1.4 m | 0.9 m |
| `min_object_distance` | 0.5 m | 0.3 m |
| depth band | 0.3 to 2.0 m, raised to 3.0 m by the ROS launcher | 0.5 to 12.0 m |
| camera pitch | 0.12 rad, overridden by the measured mount | 0.0 rad |

`agent_radius` dilates the obstacle map by a square of side twice its value, and
is therefore the narrowest gap the frontier search will offer. It must match the
inscribed radius in the ROS half's Nav2 parameters and the `agent_radius_m` in
its robot profile, or the policy will propose goals through gaps that Nav2 will
not plan through. Spot's value is the half-*width* of its roughly 1.1 by 0.5 m
footprint; we previously used the half-length of 0.55 m, which closed every gap
narrower than 1.15 m and therefore every doorway.

`pointnav_stop_radius` is measured from the base origin. Spot's 1.4 m is the
success distance we report, and it parks the nose approximately 0.85 m from the
object, as the body origin sits roughly 0.55 m behind the nose.

### Perception

| Variable | Default | Effect |
|---|---|---|
| `CLIP_PORT`, `SAM3_PORT`, `YOLOE_PORT` | 12182, 12183, 12184 | The model server ports, which must match the clients in `vlfm/policy` |
| `SKIP_YOLOE` | `1` in the launcher | `1` bypasses YOLOE entirely and prompts SAM 3 directly, which frees the detector's VRAM at the cost of recall |
| `YOLOE_CLASSES` | `data/shortvis.yaml` in the launcher, `data/lvis.yaml` in the code | YOLOE's vocabulary when `SKIP_YOLOE=0`. We report `data/lvis.yaml` |
| `SKIP_YOLOE_CLASSES` | `data/shortvis.yaml` | SAM 3's prompt list when `SKIP_YOLOE=1` |
| `SKIP_MASK2FORMER` | `1` | `1` disables the panoptic masking stage, which our deployment removes |
| `SAM3_CONF` | `0.35` in the launcher, `0.6` in the server's own default | SAM 3's confidence threshold |
| `SAM3_IMGSZ` | `644` | SAM 3's input size, which we leave at the benchmark's value |
| `ROOM_TYPES_FILE` | `data/room_types_lab.yaml` | The room lexicon from which Room Probability Vectors are formed |
| `STEPS_BETWEEN_DETECTIONS` | `1` | The perception cadence, against the benchmark's every third explore step |
| `ANNOTATED_RGB_BASE` | `raw` | What the status image is drawn on: `raw` outlines SAM 3's masks on the camera frame, `yoloe` uses the detector's own box plot |

YOLOE's confidence threshold is not an environment variable. It is the
`conf_threshold` default of 0.3 in `vlfm/vlm/yoloe.py`, which is the value we
report, and the server takes no flag for it.

### The vocabularies

Which file conditions the perception stack depends on whether YOLOE runs, and
the two paths are not interchangeable.

| `SKIP_YOLOE` | File | Behaviour |
|---|---|---|
| `0`, which we report | `YOLOE_CLASSES`, set to `data/lvis.yaml` (1203 classes) | The full two-stage funnel: YOLOE proposes from this vocabulary and SAM 3 refines the proposals into masks. Requires `checkpoints/yoloe-26x-seg.pt` |
| `1`, the launcher's default | `SKIP_YOLOE_CLASSES`, defaulting to `data/shortvis.yaml` (14 labels) | YOLOE is not started and SAM 3 is prompted directly with these labels |

Note that the launcher defaults `YOLOE_CLASSES` to `data/shortvis.yaml` rather
than to the code's `data/lvis.yaml`, such that setting `SKIP_YOLOE=0` on its own
yields a detector conditioned on 14 labels, which is neither configuration. Both
variables must be set together.

The target is always prompted in either case, listed or not, so adding a target
to a vocabulary changes nothing about whether it is detected. What a vocabulary
determines is the *other* objects that are found, and those detections are what
the value map is built from. 

`data/room_types_lab.yaml` is the lexicon from which
the Room Probability Vectors themselves are formed. We replaced the policy's
published household rooms with nine labels specific to our university
environment: meeting area, kitchen, office, elevator room, printing station,
laboratory, exit, tooling, and lecture hall. The file should be changed
according to the discretion of the researcher.

### Value map

| Variable | Default | Effect |
|---|---|---|
| `DECAY_SIGMA_M` | `2.0` | The decay of each detection's signal over free space. The gradient is the point, so a sigma that is large relative to the arena leaves every frontier scoring alike |
| `MAX_PROPAGATION_M` | `6.0` | A hard geodesic cutoff, beyond which the signal is exactly zero. We keep it larger than the arena and allow sigma to do the shaping |
| `PROPAGATION_SIGMA_CUTOFF` | `3.0` | Truncates propagation at this many sigmas |
| `MIN_BLOB_AREA_M2` | `0.1` | Zeroes signal islands smaller than this; `0` disables the pruning |
| `MAP_DEBUG` | `0` | Writes the frontier masks to `map_debug/` |

### Throughput and logging

| Variable | Default | Effect |
|---|---|---|
| `RPV_STATUS_IMAGE_EVERY` | `3` | Encodes the console's three JPEGs every Nth step, each costing roughly 60 ms |
| `VALUE_MAP_IMG_EVERY` | `5` | Writes value-map frames to `value_map_imgs/<run>/` every Nth step; `0` disables them |
| `TIMED` | `0` | Reduces both of the above, for latency measurement |
| `VLM_REQUEST_RETRIES`, `VLM_REQUEST_BACKOFF`, `VLM_BUSY_TIMEOUT` | `2`, `0.5`, `15.0` | Bound one model round trip to approximately 30 s, which is necessary as the stock bound exceeds the bridge's own timeout |
| `RPV_TIMING` | `0` | One per-stage timing line per step |
| `RPV_DEBUG` | `0` | Per-detection traces on stdout |

## The policy interface

`POST /rpv_policy`, JSON:

| Field | Meaning |
|---|---|
| `rgb`, `rgb_order` | A base64 JPEG and its channel order, `"bgr"` or `"rgb"` |
| `depth_png16` | A base64 16-bit PNG of millimetres, where 0 denotes invalid, carrying its own shape, and taking precedence over the two fields below. Encoded by `vlfm/vlm/depth_codec.py`, whose encoder the ROS bridge imports from here such that the two cannot drift |
| `depth_row`, `depth_shape` | One row of a laser-scan curtain, which is tiled to the image here |
| `depth_m` | A full float raster, retained for compatibility |
| `occ_hash`; `occupancy`, `occ_shape`, `origin_x/y`, `resolution` | A hash of the occupancy grid, and the grid itself only when that hash is new. `{occ_cache_miss: true}` is returned when the full grid is required |
| `robot_map_x/y/yaw`, `start_x/y/yaw` | The live pose and the pinned episodic origin, in the map frame |
| `fx`, `fy`, `img_w`, `camera_height`, `camera_pitch`, `camera_roll` | Intrinsics and extrinsics, where positive pitch places the lens down |
| `min_depth`, `max_depth` | Advisory only; the server's own band is authoritative and a disagreement is logged once |
| `objectgoal` | The target class as free text. Synonyms may be supplied `\|`-separated, in which case all are prompted but only the first is matched against a returned mask's label |
| `is_first` | The episode reset: clears the maps, pins the origin, and adopts the target. A target that differs from the current one without this flag forces a reset |
| `finish_init` | Whether the initial sweep has completed |

The response is `{mode, goal_map_xy, goal_yaw, called_stop, done_initializing}`.
`GET /status` returns the policy's state together with the three images the
operator console displays.

## Citation and attribution
The policy this package deploys is Room-Mediated Co-occurrence (Scicluna et al.,
2026), which extends Vision-Language Frontier Maps (Yokoyama et al., 2024).


```bibtex
@misc{scicluna2026roommediatedcooccurrencezeroshotobjectcentric,
      title={Room-Mediated Co-occurrence for Zero-Shot Object-Centric Semantic Navigation via Frontier Scoring}, 
      author={Adam Scicluna and Gavin Paul and Alen Alempijevic},
      year={2026},
      eprint={2607.25448},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2607.25448}, 
}
```

```bibtex
@inproceedings{yokoyama2024vlfm,
  title={VLFM: Vision-Language Frontier Maps for Zero-Shot Semantic Navigation},
  author={Naoki Yokoyama and Sehoon Ha and Dhruv Batra and Jiuguang Wang and Bernadette Bucher},
  booktitle={International Conference on Robotics and Automation (ICRA)},
  year={2024},
}
```


