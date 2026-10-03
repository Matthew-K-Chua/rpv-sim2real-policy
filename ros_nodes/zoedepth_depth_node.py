#!/usr/bin/env python3
"""ROS 2 receiver: colour in, ZoeDepth metric depth out on the depth topics.

    /camera/camera/color/image_raw/compressed  ──┐
    /camera/camera/color/camera_info  ───────────┤
                                                 ├─► HTTP POST ─► zoedepth server
    /scan   (optional, for scale alignment)  ────┘                (conda, GPU)
                                                 ◄── 16-bit PNG, millimetres
    ──► /zoedepth/image_raw    sensor_msgs/Image  16UC1, mm, 0 = invalid
    ──► /zoedepth/camera_info  the COLOUR CameraInfo, restamped
    ──► /zoedepth/diagnostics  Float32MultiArray [scale, residual_m, n_pairs, infer_ms]

Everything downstream (depth_image_proc, pointcloud_to_laserscan, Nav2's voxel
layer, RTAB-Map, RViz) sees an ordinary registered depth camera. Which of them
should actually be wired to it is a separate question with a real answer -- read
the module docstring of vlfm/vlm/zoedepth.py first.

WHY A SEPARATE PROCESS FROM THE MODEL. Same reason the policy is a Flask server:
rclpy is built for Python 3.10 and the model stack wants 3.12 plus a GPU torch.
This node imports nothing heavier than numpy, cv2 and requests.

THREE THINGS THIS NODE IS CAREFUL ABOUT, all of them things that silently
produce a plausible-looking but wrong depth image:

  1. STAMP AND FRAME ARE THE COLOUR FRAME'S, COPIED. Not "now". A depth image
     stamped on arrival is stamped ~1 request-latency after the photons, and
     every TF lookup downstream then places the points where the robot no longer
     is. ZoeDepth's output is registered to the colour image by construction, so
     the colour header is the correct header, verbatim.
  2. ONE REQUEST IN FLIGHT, NEWEST FRAME WINS. Without this the node builds an
     unbounded queue the moment inference is slower than the camera and every
     depth image it publishes is progressively more stale -- the failure looks
     like drift, not like a backlog.
  3. THE CAMERA_INFO IS REPUBLISHED, NOT REBUILT. depth_image_proc reads K from
     it, and a K that does not match the raster's dimensions produces a cloud
     that is subtly the wrong shape -- and never an error.

Run (ROS environment, NOT conda):
    python3 ros_nodes/zoedepth_depth_node.py --ros-args \
        -p zoedepth_url:=http://127.0.0.1:12185/zoedepth \
        -p scan_topic:=/scan -p scale_align:=true
"""
import base64
import copy
import os
import sys
import threading
from collections import deque

import numpy as np
import rclpy
import requests
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import CameraInfo, CompressedImage, Image, LaserScan
from std_msgs.msg import Float32MultiArray

import tf2_ros

# The codec is shared with the server rather than reimplemented here, so the two
# ends cannot drift apart. vlfm/vlm/depth_codec.py imports numpy and cv2 only --
# nothing that would drag flask or torch into a ROS process.
_REPO = os.environ.get("RPV_POLICY_REPO", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
from vlfm.vlm.depth_codec import decode_png16_mm  # noqa: E402


def _stamp_to_sec(stamp) -> float:
    return stamp.sec + stamp.nanosec * 1e-9


def _quat_to_rotmat(q):
    x, y, z, w = q.x, q.y, q.z, q.w
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


class ZoeDepthNode(Node):
    def __init__(self) -> None:
        super().__init__("zoedepth_depth")

        p = self.declare_parameter
        p("zoedepth_url", "http://127.0.0.1:12185/zoedepth")
        p("rgb_topic", "/camera/camera/color/image_raw/compressed")
        p("camera_info_topic", "/camera/camera/color/camera_info")
        p("out_namespace", "/zoedepth")
        p("request_timeout_s", 5.0)
        # Publish nothing beyond this range. NOT a cosmetic clamp: a monocular
        # model returns a confident number for the far wall it is guessing at,
        # and 0 (= invalid) is how ROS says "no reading", which depth_image_proc
        # and the costmaps both honour. The NYU-trained head is credible to
        # roughly 8 m indoors and much less than that on texture-poor surfaces.
        p("trust_max_m", 8.0)
        p("trust_min_m", 0.3)
        # Hard rate cap independent of the camera. The Pi's colour stream is
        # 6 fps, so this is normally inert; it stops a future faster stream from
        # saturating the GPU the policy is sharing.
        p("max_rate_hz", 6.0)

        # ---- scale alignment against a real ranging source --------------------
        # The single most useful thing available here: /scan IS metric (on Spot it
        # is five real depth cameras), so the pixels it projects onto give a
        # per-frame ground truth to fit ZoeDepth's scale against. Without it the
        # depth is only as good as the model's absolute calibration on this scene;
        # with it, the model supplies SHAPE and the scan supplies SCALE, which is
        # the division each is actually good at.
        p("scale_align", False)
        p("scan_topic", "/scan")
        p("lidar_frame", "body")
        p("camera_optical_frame", "camera_color_optical_frame")
        p("scan_max_dt", 0.3)
        p("scan_buffer_len", 100)
        # Fitted scale is smoothed across frames: a single frame staring at a
        # blank wall can produce a wild fit, and a depth image that jumps scale
        # frame to frame is worse for a costmap than a consistently wrong one.
        p("scale_ema_alpha", 0.3)
        p("scale_min", 0.5)
        p("scale_max", 2.0)
        p("min_scale_pairs", 30)

        self.url = self.get_parameter("zoedepth_url").value
        self.timeout = float(self.get_parameter("request_timeout_s").value)
        self.trust_max_m = float(self.get_parameter("trust_max_m").value)
        self.trust_min_m = float(self.get_parameter("trust_min_m").value)
        self.min_period = 1.0 / max(1e-6, float(self.get_parameter("max_rate_hz").value))
        self.scale_align = bool(self.get_parameter("scale_align").value)
        self.lidar_frame = self.get_parameter("lidar_frame").value
        self.camera_optical_frame = self.get_parameter("camera_optical_frame").value
        self.scan_max_dt = float(self.get_parameter("scan_max_dt").value)
        self.ema_alpha = float(self.get_parameter("scale_ema_alpha").value)
        self.scale_min = float(self.get_parameter("scale_min").value)
        self.scale_max = float(self.get_parameter("scale_max").value)
        self.min_scale_pairs = int(self.get_parameter("min_scale_pairs").value)

        ns = self.get_parameter("out_namespace").value.rstrip("/")
        self.pub_depth = self.create_publisher(Image, f"{ns}/image_raw", 5)
        self.pub_info = self.create_publisher(CameraInfo, f"{ns}/camera_info", 5)
        self.pub_diag = self.create_publisher(Float32MultiArray, f"{ns}/diagnostics", 5)

        # Sensor data QoS (best-effort): the camera publishes with it, and a
        # RELIABLE subscriber simply never connects to a BEST_EFFORT publisher --
        # which presents as "no frames" with no error anywhere.
        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            durability=QoSDurabilityPolicy.VOLATILE,
        )

        rgb_topic = self.get_parameter("rgb_topic").value
        if rgb_topic.endswith("/compressed"):
            # Forward the camera's own JPEG bytes untouched: one less lossy
            # generation than decode+re-encode, and less CPU on this node.
            self.create_subscription(CompressedImage, rgb_topic, self._on_compressed, sensor_qos)
        else:
            self.create_subscription(Image, rgb_topic, self._on_raw, sensor_qos)
        self.create_subscription(
            CameraInfo, self.get_parameter("camera_info_topic").value, self._on_info, sensor_qos
        )

        self.info = None
        self._pending = None          # (jpeg_b64, header) newest frame not yet sent
        self._pending_lock = threading.Lock()
        self._inflight = False
        self._last_pub_t = 0.0
        self.scale = 1.0
        self._scan_buf = deque(maxlen=int(self.get_parameter("scan_buffer_len").value))
        self._scan_lock = threading.Lock()
        self._T_cam_lidar = None

        if self.scale_align:
            self.create_subscription(LaserScan, self.get_parameter("scan_topic").value, self._on_scan, sensor_qos)
            self.tf_buffer = tf2_ros.Buffer()
            self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # A worker thread, not a timer: the POST blocks for tens of milliseconds
        # and must not sit inside an executor callback.
        self._stop = threading.Event()
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

        self.get_logger().info(
            f"zoedepth_depth: {rgb_topic} -> {ns}/image_raw via {self.url} "
            f"(scale_align={self.scale_align}, trust {self.trust_min_m}-{self.trust_max_m} m)"
        )

    # ------------------------------------------------------------------ #
    # subscriptions
    # ------------------------------------------------------------------ #
    def _on_info(self, msg: CameraInfo) -> None:
        self.info = msg

    def _on_compressed(self, msg: CompressedImage) -> None:
        with self._pending_lock:
            self._pending = (base64.b64encode(bytes(msg.data)).decode("ascii"), msg.header, "bgr")

    def _on_raw(self, msg: Image) -> None:
        import cv2

        if msg.encoding not in ("rgb8", "bgr8"):
            self.get_logger().warn(f"unsupported encoding {msg.encoding!r}", throttle_duration_sec=5.0)
            return
        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        ok, buf = cv2.imencode(".jpg", arr if msg.encoding == "bgr8" else arr[:, :, ::-1],
                               [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not ok:
            return
        with self._pending_lock:
            self._pending = (base64.b64encode(buf).decode("ascii"), msg.header, "bgr")

    def _on_scan(self, msg: LaserScan) -> None:
        with self._scan_lock:
            self._scan_buf.append((_stamp_to_sec(msg.header.stamp), msg))

    # ------------------------------------------------------------------ #
    # worker
    # ------------------------------------------------------------------ #
    def _run(self) -> None:
        import time

        while not self._stop.is_set():
            if self.info is None:
                time.sleep(0.05)
                continue
            now = time.monotonic()
            if now - self._last_pub_t < self.min_period:
                time.sleep(0.01)
                continue
            with self._pending_lock:
                item, self._pending = self._pending, None
            if item is None:
                time.sleep(0.01)
                continue

            jpeg_b64, header, order = item
            try:
                self._process(jpeg_b64, header, order)
            except Exception as exc:  # noqa: BLE001 - a bad frame must not kill the node
                self.get_logger().warn(f"depth request failed: {exc}", throttle_duration_sec=2.0)
            self._last_pub_t = time.monotonic()

    def _process(self, jpeg_b64: str, header, order: str) -> None:
        # The scale in force for THIS frame, captured before the POST. _fit_scale
        # needs it: the raster that comes back is already multiplied by it, so the
        # ratio it measures is a correction TO that scale, not the scale itself.
        scale_used = float(self.scale)
        payload = {
            "image": jpeg_b64,
            "rgb_order": order,
            "trust_max_m": self.trust_max_m,
            # The scale correction is applied SERVER-SIDE (see zoedepth.py) so the
            # bytes on the wire are already corrected and the trust clamp is
            # applied to the corrected metres rather than the raw ones.
            "scale": scale_used,
        }
        resp = requests.post(self.url, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        body = resp.json()
        mm = decode_png16_mm(body["depth_png16"])

        # Near clamp here rather than server-side: trust_min is about THIS
        # camera's geometry (the D435's own minimum useful range and the robot's
        # own body in frame), not about the model.
        mm[mm < int(self.trust_min_m * 1000)] = 0

        info = self.info
        if (mm.shape[0], mm.shape[1]) != (info.height, info.width):
            self.get_logger().warn(
                f"depth {mm.shape} != camera_info {info.height}x{info.width}; "
                "K would not match -- dropping the frame rather than publishing a wrong cloud",
                throttle_duration_sec=5.0,
            )
            return

        scale, residual, n_pairs = self._fit_scale(mm, header, scale_used)

        img = Image()
        img.header = header                    # stamp AND frame_id, verbatim
        img.height, img.width = mm.shape
        img.encoding = "16UC1"
        img.is_bigendian = 0
        img.step = img.width * 2
        img.data = mm.astype("<u2").tobytes()
        self.pub_depth.publish(img)

        # Copy, not mutate: `info` is the object the subscription handed us and
        # is read again by the next frame's scale fit. Restamping it in place
        # would also mean publishing a message another thread can be reading.
        out_info = copy.deepcopy(info)
        out_info.header = header
        self.pub_info.publish(out_info)

        diag = Float32MultiArray()
        diag.data = [float(scale), float(residual), float(n_pairs), float(body.get("infer_ms", 0.0))]
        self.pub_diag.publish(diag)

    # ------------------------------------------------------------------ #
    # scale alignment
    # ------------------------------------------------------------------ #
    def _fit_scale(self, mm: np.ndarray, header, scale_used: float):
        """Robust per-frame scale of mono depth against the contemporary scan.

        Returns (scale_in_use, residual_m, n_pairs). Fits the single multiplier
        that best maps the model's depth onto the scan's ranges, then smooths it.

        NO SHIFT TERM: ZoeDepth's error is multiplicative (it is a metric-scale
        ambiguity, not an offset), and a two-parameter fit on a few dozen
        near-coplanar points is happy to trade a large shift against a large
        scale and produce a worse depth image with a smaller residual.

        ``mm`` has ALREADY been multiplied by ``scale_used`` server-side, so the
        measured ratio is a correction to that scale and the new absolute scale is
        the product of the two. Treating the ratio as the absolute scale would
        make the estimate converge to 1.0 no matter what the truth is.
        """
        if not self.scale_align:
            return self.scale, 0.0, 0

        scan, dt = self._scan_nearest(_stamp_to_sec(header.stamp))
        if scan is None or (self.scan_max_dt > 0 and dt > self.scan_max_dt):
            return self.scale, 0.0, 0
        if not self._ensure_tf(header.stamp):
            return self.scale, 0.0, 0

        ranges = np.asarray(scan.ranges, dtype=np.float32)
        angles = scan.angle_min + np.arange(ranges.size) * scan.angle_increment
        good = np.isfinite(ranges) & (ranges > scan.range_min) & (ranges < scan.range_max)
        if not good.any():
            return self.scale, 0.0, 0
        r, a = ranges[good], angles[good]

        pts = np.column_stack([r * np.cos(a), r * np.sin(a), np.zeros_like(r), np.ones_like(r)])
        cam = pts @ self._T_cam_lidar.T
        z = cam[:, 2]
        front = z > 1e-3
        cam, z = cam[front], z[front]
        if z.size == 0:
            return self.scale, 0.0, 0

        k = self.info.k
        fx, fy, cx, cy = k[0], k[4], k[2], k[5]
        u = np.round(cx + fx * cam[:, 0] / z).astype(int)
        v = np.round(cy + fy * cam[:, 1] / z).astype(int)
        h, w = mm.shape
        ok = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        u, v, z = u[ok], v[ok], z[ok]
        if z.size == 0:
            return self.scale, 0.0, 0

        mono = mm[v, u].astype(np.float32) / 1000.0
        ok = mono > 0.05
        mono, z = mono[ok], z[ok]
        if mono.size < self.min_scale_pairs:
            return self.scale, 0.0, int(mono.size)

        ratios = z / mono
        # Median then MAD rejection: the scan plane grazes object edges, where a
        # one-pixel projection error puts a foreground range against a background
        # depth and produces a ratio off by a factor of several.
        med = float(np.median(ratios))
        mad = float(np.median(np.abs(ratios - med))) + 1e-6
        keep = np.abs(ratios - med) < 3.0 * mad
        if keep.sum() < self.min_scale_pairs:
            return self.scale, 0.0, int(keep.sum())
        correction = float(np.median(ratios[keep]))
        absolute = min(max(correction * scale_used, self.scale_min), self.scale_max)

        self.scale = (1.0 - self.ema_alpha) * self.scale + self.ema_alpha * absolute
        # Residual of the fit as applied to THIS frame's raster (which carries
        # scale_used), so it reads as "how far off were we", not "how far off
        # would the next frame be".
        residual = float(np.median(np.abs(correction * mono[keep] - z[keep])))
        return self.scale, residual, int(keep.sum())

    def _scan_nearest(self, t: float):
        with self._scan_lock:
            snapshot = list(self._scan_buf)
        if not snapshot:
            return None, float("inf")
        stamp, scan = min(snapshot, key=lambda item: abs(item[0] - t))
        return scan, abs(stamp - t)

    def _ensure_tf(self, stamp) -> bool:
        """camera_optical <- lidar, looked up once and cached (it is static)."""
        if self._T_cam_lidar is not None:
            return True
        try:
            tf = self.tf_buffer.lookup_transform(
                self.camera_optical_frame, self.lidar_frame, rclpy.time.Time()
            )
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(
                f"no {self.lidar_frame} -> {self.camera_optical_frame} transform ({exc}); "
                "scale alignment idle. On Spot this is the six MEASURED d435_* launch arguments.",
                throttle_duration_sec=10.0,
            )
            return False
        T = np.eye(4)
        T[:3, :3] = _quat_to_rotmat(tf.transform.rotation)
        T[:3, 3] = [tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z]
        self._T_cam_lidar = T
        self.get_logger().info(f"cached {self.lidar_frame} -> {self.camera_optical_frame} transform")
        return True

    def destroy_node(self) -> bool:
        self._stop.set()
        self._worker.join(timeout=2.0)
        return super().destroy_node()


def main() -> None:
    rclpy.init()
    node = ZoeDepthNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
