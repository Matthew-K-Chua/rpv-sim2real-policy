# Copyright (c) 2026. Embodied-RPV.
"""Wire codec for metric depth rasters. numpy + cv2 only, on purpose.

THREE processes have to agree on these bytes and they live in three different
Python environments:

    * the ZoeDepth server        (its own conda env, Python 3.12)
    * the ROS receiver node      (ROS Humble's Python 3.10, no conda)
    * the policy server          (rpv-emb conda env, Python 3.12)

so the encoder and the decoder cannot be allowed to drift apart. This module is
the single copy, and it deliberately imports nothing beyond numpy and cv2 --
both of which every one of those three environments already has -- so the ROS
node can `sys.path.insert` this repo and import it without dragging flask,
requests or torch into a ROS process.

WHY 16-BIT PNG AND NOT JPEG. ``server_wrapper.image_to_str`` JPEG-encodes, which
is lossy; on a metric depth raster that is not "slightly blurry", it is metres of
error at every depth discontinuity, and DC-block ringing turns a doorway into a
ramp. PNG is lossless.

WHY MILLIMETRES IN uint16 AND NOT float32. It is exactly ``sensor_msgs/Image``
encoding ``16UC1``, which is what depth_image_proc, RTAB-Map, Nav2's voxel layer
and RViz all expect from a depth camera, so the ROS node republishes the decoded
array with no conversion at all. It is also ~8x smaller on the wire than raw
float32 (640x480: ~1.2 MB -> ~150 kB typical). The cost is 1 mm quantisation and
a 65.535 m ceiling, neither of which is close to being the dominant error in a
MONOCULAR depth estimate.

ZERO MEANS INVALID. That is the ROS convention (depth_image_proc drops zeros
rather than projecting them to the camera origin), and it is how this pipeline
expresses "the model produced a number here but we do not believe it" -- see
``trust_max_m`` below. Anything reading the float side must treat 0.0 as
no-reading and never as "0 m away".
"""

import base64
from typing import Optional

import cv2
import numpy as np

# uint16 millimetres: 65535 mm. Also the value that means "saturated", which is
# why encode() clamps to MAX_MM - 1 rather than letting a large value wrap.
MAX_MM = 65535
INVALID_MM = 0

# PNG compression level. 1, not the cv2 default of 3: this sits in a per-frame
# latency budget, and on a smooth depth raster level 1 is ~3x faster for ~10%
# more bytes over a loopback/LAN hop that is not the bottleneck.
_PNG_LEVEL = int(1)


def encode_depth_png16(depth_m: np.ndarray, trust_max_m: Optional[float] = None) -> str:
    """float32 metres -> base64 16-bit PNG of millimetres.

    trust_max_m: anything beyond this range is written as INVALID (0) rather
    than as a large number. This is the honest way to publish monocular depth:
    the model always returns *a* value for every pixel, including for the far
    wall it is guessing at, and a costmap that marks those guesses as obstacles
    is worse than one that sees nothing there. Pass None to keep every value.
    """
    d = np.asarray(depth_m, dtype=np.float32)
    mm = np.rint(d * 1000.0)
    # Order matters: build the validity mask from the ORIGINAL metres, before
    # the clip below silently pulls out-of-range values back into range.
    invalid = ~np.isfinite(d) | (d <= 0.0)
    if trust_max_m is not None:
        invalid |= d > float(trust_max_m)
    mm = np.clip(mm, 1.0, float(MAX_MM - 1))
    mm = mm.astype(np.uint16)
    mm[invalid] = INVALID_MM
    ok, buf = cv2.imencode(".png", mm, [int(cv2.IMWRITE_PNG_COMPRESSION), _PNG_LEVEL])
    if not ok:
        raise RuntimeError("cv2.imencode failed on the depth raster")
    return base64.b64encode(buf).decode("ascii")


def decode_png16_mm(payload: str) -> np.ndarray:
    """base64 16-bit PNG -> (H, W) uint16 millimetres, 0 = invalid.

    This is the form the ROS node wants: it goes straight into a
    ``sensor_msgs/Image`` with encoding "16UC1" and no arithmetic in between.

    IMREAD_UNCHANGED is load-bearing. Without it OpenCV helpfully converts the
    16-bit single-channel PNG to 8-bit BGR and every depth becomes garbage that
    still looks like a plausible image.
    """
    raw = np.frombuffer(base64.b64decode(payload), dtype=np.uint8)
    mm = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)
    if mm is None:
        raise ValueError("cv2.imdecode returned None -- payload is not a PNG")
    if mm.dtype != np.uint16 or mm.ndim != 2:
        raise ValueError(f"expected a 2-D uint16 PNG, got {mm.ndim}-D {mm.dtype}")
    return mm


def decode_depth_png16(payload: str) -> np.ndarray:
    """base64 16-bit PNG -> (H, W) float32 metres, 0.0 = invalid."""
    return decode_png16_mm(payload).astype(np.float32) / 1000.0


def encode_png16_mm(mm: np.ndarray) -> str:
    """(H, W) uint16 millimetres -> base64 16-bit PNG. The inverse of decode_png16_mm.

    Separate from ``encode_depth_png16`` so a caller that already holds a ROS
    ``16UC1`` raster -- the ceai bridge, forwarding /zoedepth/image_raw to the
    policy -- does not have to round-trip through float metres to re-encode it.
    That round trip is lossless in principle and pointless in practice, and it is
    one more place for a unit error to hide.

    Whatever validity convention the array already carries is preserved: this
    function applies no clamp and no trust range, because the node that produced
    the raster already did.
    """
    mm = np.ascontiguousarray(mm, dtype=np.uint16)
    ok, buf = cv2.imencode(".png", mm, [int(cv2.IMWRITE_PNG_COMPRESSION), _PNG_LEVEL])
    if not ok:
        raise RuntimeError("cv2.imencode failed on the depth raster")
    return base64.b64encode(buf).decode("ascii")
