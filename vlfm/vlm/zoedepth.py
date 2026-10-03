# Copyright (c) 2026. Embodied-RPV.
"""ZoeDepth metric monocular depth, hosted like every other model in vlfm/vlm/.

WHY THIS EXISTS. For a robot whose camera publishes colour only, there is no
depth raster in the camera's frame, and the policy would have to range its
detections off a "curtain" tiled from a single row of /scan -- every pixel in a
column gets the same distance. ZoeDepth turns the colour stream into a dense
metric raster that is registered to that colour frame BY CONSTRUCTION. With the
D435 streaming real aligned depth (the default, depth_source=realsense) none of
this is started; it is the DEPTH_SOURCE=zoedepth fallback.

WHAT IT IS NOT. It is not a depth sensor. Read the rest of this docstring
before pointing SLAM at it; the short version is that a monocular metric model
returns a confident number for every pixel including the ones it is guessing,
its scale drifts with scene content, and the failure that matters is an
UNDER-estimate (a phantom obstacle that stops the robot) rather than the noise.

TWO BACKENDS, and the default is not the upstream repo:

  "hf" (default)  transformers' own ZoeDepth port -- same weights, same metric
                  output, no torch.hub, no timm pin. Loads Intel/zoedepth-nyu.
  "torchhub"      isl-org/ZoeDepth itself, via torch.hub. Faithful to the paper
                  release, and brittle: it pins timm ~0.6 (newer timm breaks the
                  BEiT backbone import) and its checkpoint load predates torch
                  2.6's weights_only=True default. Use it to reproduce upstream
                  numbers, not to fly the robot.

Both are wrapped so ``predict()`` returns the same thing: float32 METRES, same
height and width as the input image.

Run:  python -m vlfm.vlm.zoedepth --port 12185
"""

import os
import time
from typing import Any, Optional

import numpy as np

from .depth_codec import encode_depth_png16
from .server_wrapper import ServerMixin, host_model, send_request, str_to_image

# Indoor NYU weights by default: the arena is a lab. Intel/zoedepth-nyu-kitti is
# the generalist head (use it the moment the robot sees daylight or a corridor
# longer than ~10 m); Intel/zoedepth-kitti is outdoor-only and wrong here.
DEFAULT_HF_MODEL = os.environ.get("ZOEDEPTH_HF_MODEL", "Intel/zoedepth-nyu")
# ZoeD_N / ZoeD_K / ZoeD_NK -- the torch.hub entrypoint names.
DEFAULT_HUB_MODEL = os.environ.get("ZOEDEPTH_HUB_MODEL", "ZoeD_N")

# Longest input side fed to the network. The D435 colour stream is 424x240 or
# 640x480, both already below ZoeDepth's 512x384 working resolution, so this is
# normally inert -- it exists so that raising the Pi's resolution later cannot
# silently multiply inference time. 0 disables the cap.
INPUT_MAX_SIDE = int(os.environ.get("ZOEDEPTH_INPUT_MAX_SIDE", "640"))

# ZoeDepth's paper-standard test-time augmentation averages the prediction with
# its horizontal mirror. It costs exactly one extra forward pass for a small
# accuracy gain, which is a bad trade inside a 6 Hz perception loop -- off here,
# on for offline evaluation.
FLIP_AUG = os.environ.get("ZOEDEPTH_FLIP_AUG", "0") == "1"


class ZoeDepth:
    def __init__(
        self,
        backend: str = os.environ.get("ZOEDEPTH_BACKEND", "hf"),
        model_name: Optional[str] = None,
        device: Optional[Any] = None,
        half: bool = os.environ.get("ZOEDEPTH_HALF", "1") == "1",
    ) -> None:
        import torch

        self.torch = torch
        if device is None:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.device = device
        self.backend = backend
        # fp16 on CPU is slower than fp32 and on some ops unimplemented, so the
        # flag only means anything on CUDA.
        self.half = bool(half) and device.type == "cuda"

        if backend == "hf":
            from transformers import AutoImageProcessor, ZoeDepthForDepthEstimation

            name = model_name or DEFAULT_HF_MODEL
            self.processor = AutoImageProcessor.from_pretrained(name)
            self.model = ZoeDepthForDepthEstimation.from_pretrained(name).to(device).eval()
            if self.half:
                self.model = self.model.half()
        elif backend == "torchhub":
            name = model_name or DEFAULT_HUB_MODEL
            # trust_repo: torch.hub otherwise blocks on an interactive y/n prompt,
            # which in a tmux window under a launcher looks like a hang.
            self.model = torch.hub.load("isl-org/ZoeDepth", name, pretrained=True, trust_repo=True)
            self.model = self.model.to(device).eval()
            self.processor = None
        else:
            raise ValueError(f"unknown ZOEDEPTH_BACKEND {backend!r} (expected 'hf' or 'torchhub')")

        self.model_name = name
        print(f"[zoedepth] backend={backend} model={name} device={device} half={self.half}", flush=True)

    # ------------------------------------------------------------------ #
    def predict(self, rgb: np.ndarray, flip_aug: bool = FLIP_AUG) -> np.ndarray:
        """(H, W, 3) uint8 RGB -> (H, W) float32 metres, same H and W."""
        import cv2

        h, w = rgb.shape[:2]
        net_in = rgb
        if INPUT_MAX_SIDE and max(h, w) > INPUT_MAX_SIDE:
            scale = INPUT_MAX_SIDE / float(max(h, w))
            net_in = cv2.resize(rgb, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)

        depth = self._forward(net_in, flip_aug)

        if depth.shape[:2] != (h, w):
            # Bilinear, not nearest: depth is a smooth field except at object
            # boundaries, and the boundaries are already soft in the prediction.
            depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)
        return np.ascontiguousarray(depth, dtype=np.float32)

    def _forward(self, rgb: np.ndarray, flip_aug: bool) -> np.ndarray:
        torch = self.torch
        if self.backend == "hf":
            inputs = self.processor(images=rgb, return_tensors="pt").to(self.device)
            if self.half:
                inputs["pixel_values"] = inputs["pixel_values"].half()
            with torch.no_grad():
                outputs = self.model(**inputs)
                outputs_flipped = None
                if flip_aug:
                    outputs_flipped = self.model(pixel_values=torch.flip(inputs["pixel_values"], dims=[3]))
            # post_process_depth_estimation undoes ZoeDepth's internal padding and
            # resizing and returns METRES at source_sizes. Doing it by hand (a
            # plain interpolate of outputs.predicted_depth) gets the scale right
            # but the crop wrong, which shows up as a few percent of range error
            # that grows towards the image edges.
            post = self.processor.post_process_depth_estimation(
                outputs, source_sizes=[rgb.shape[:2]], outputs_flipped=outputs_flipped
            )
            return post[0]["predicted_depth"].float().cpu().numpy()

        # torchhub: infer_pil already returns metres at the input resolution.
        from PIL import Image

        pil = Image.fromarray(rgb)
        with torch.no_grad():
            return np.asarray(
                self.model.infer_pil(pil, pad_input=True, with_flip_aug=flip_aug), dtype=np.float32
            )


class ZoeDepthClient:
    """Convenience client for anything already inside a conda env with this repo.

    The ROS receiver node deliberately does NOT use this -- it lives in ROS's
    Python 3.10 and posts with plain ``requests`` so it never imports flask or
    torch. See ros_nodes/zoedepth_depth_node.py.
    """

    def __init__(self, port: int = 12185, host: str = "localhost"):
        self.url = f"http://{host}:{port}/zoedepth"

    def predict(self, rgb: np.ndarray, trust_max_m: Optional[float] = None) -> np.ndarray:
        from .depth_codec import decode_depth_png16

        kwargs = {"image": rgb, "rgb_order": "rgb"}
        if trust_max_m is not None:
            kwargs["trust_max_m"] = float(trust_max_m)
        response = send_request(self.url, **kwargs)
        return decode_depth_png16(response["depth_png16"])


class ZoeDepthServer(ServerMixin, ZoeDepth):
    """The hosted form. Defined at module level rather than inside __main__ so it
    can be exercised with a stubbed ``predict`` -- the request/response path is
    worth testing without a GPU or a 1.5 GB download."""

    def process_payload(self, payload: dict) -> dict:
        import cv2

        t0 = time.perf_counter()
        rgb = str_to_image(payload["image"])
        # cv2.imdecode always yields BGR. The ROS node forwards the Pi's own
        # JPEG bytes unmodified (one less lossy generation than re-encoding)
        # and therefore declares "bgr"; same convention as the policy server.
        if str(payload.get("rgb_order", "rgb")).lower() == "bgr":
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        depth = self.predict(rgb, flip_aug=bool(payload.get("flip_aug", FLIP_AUG)))

        # Optional affine correction fitted OUTSIDE this server (the ROS node
        # fits it against /scan). Applied here so the number on the wire is
        # already the corrected one and nothing downstream can forget to.
        scale = float(payload.get("scale", 1.0))
        shift = float(payload.get("shift", 0.0))
        if scale != 1.0 or shift != 0.0:
            depth = depth * scale + shift

        trust_max_m = payload.get("trust_max_m")
        infer_ms = (time.perf_counter() - t0) * 1e3

        valid = np.isfinite(depth) & (depth > 0.0)
        return {
            "depth_png16": encode_depth_png16(depth, None if trust_max_m is None else float(trust_max_m)),
            "shape": list(depth.shape),
            "model": getattr(self, "model_name", "unknown"),
            "infer_ms": round(infer_ms, 1),
            # Stats over the prediction BEFORE the trust clamp -- this is how you
            # tell "the model saw nothing" from "we threw it away".
            "min_m": float(depth[valid].min()) if valid.any() else 0.0,
            "max_m": float(depth[valid].max()) if valid.any() else 0.0,
            "median_m": float(np.median(depth[valid])) if valid.any() else 0.0,
        }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=int(os.environ.get("ZOEDEPTH_PORT", "12185")))
    parser.add_argument("--host", type=str, default=os.environ.get("ZOEDEPTH_HOST", "127.0.0.1"))
    parser.add_argument("--backend", type=str, default=os.environ.get("ZOEDEPTH_BACKEND", "hf"))
    parser.add_argument("--model", type=str, default=None)
    args = parser.parse_args()

    print("Loading model...")
    zoe = ZoeDepthServer(backend=args.backend, model_name=args.model)
    print("Model loaded!")
    print(f"Hosting on {args.host}:{args.port} ...")
    host_model(zoe, name="zoedepth", port=args.port, host=args.host)
