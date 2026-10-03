# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

import base64
import os
import random
import socket
import time
import traceback
from typing import Any, Callable, Dict, Optional

import cv2
import numpy as np
import requests
from flask import Flask, jsonify, request


class ServerMixin:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def process_payload(self, payload: dict) -> dict:
        raise NotImplementedError


def host_model(
    model: Any,
    name: str,
    port: int = 5000,
    host: str = "localhost",
    status_provider: Optional[Callable[[], dict]] = None,
    aliases: Optional[list] = None,
) -> None:
    """
    Hosts a model as a REST API using Flask.

    host defaults to "localhost" (VLM servers are internal to one machine). Pass
    "0.0.0.0" to expose the server on all interfaces — e.g. the TurtleBot4 policy
    server, so the ceai bridge can POST to it from another machine.

    status_provider (optional): a zero-arg callable returning a JSON-serialisable
    dict. When given, a ``GET /status`` route is registered that returns it. The
    GUI console polls this to surface the latest policy reasoning (mode,
    detections, value/obstacle map images) without disturbing the POST path. VLM
    servers leave this None and so expose no /status route.

    aliases (optional): extra route names that POST to the same handler, e.g. to
    keep an old endpoint alive during a rename so a bridge still pointing at it
    reaches the server rather than 404-ing. The routes are indistinguishable to
    the caller; ``name`` is only what appears in log lines.
    """
    app = Flask(__name__)

    @app.route(f"/{name}", methods=["POST"])
    def process_request() -> Any:
        try:
            payload = request.json
            return jsonify(model.process_payload(payload))
        except Exception as e:
            # Surface the real cause instead of a bare 500: log the full traceback
            # to this server's console AND return it in the JSON body so the caller
            # (e.g. the ceai bridge) sees *why* the step failed, not just "500".
            tb = traceback.format_exc()
            print(f"[{name}] process_payload failed:\n{tb}", flush=True)
            return jsonify({"error": str(e), "type": type(e).__name__, "traceback": tb}), 500

    for alias in aliases or []:
        # endpoint= is required: Flask keys view functions by name, and every
        # alias reuses the same function object, so they would collide.
        app.add_url_rule(
            f"/{alias}", endpoint=f"process_request_{alias}", view_func=process_request, methods=["POST"]
        )

    if status_provider is not None:

        @app.route("/status", methods=["GET"])
        def status_request() -> Any:
            try:
                return jsonify(status_provider())
            except Exception as e:
                tb = traceback.format_exc()
                print(f"[{name}] status_provider failed:\n{tb}", flush=True)
                return jsonify({"error": str(e), "type": type(e).__name__}), 500

    app.run(host=host, port=port, threaded=True)


def bool_arr_to_str(arr: np.ndarray) -> str:
    """Converts a boolean array to a string."""
    packed_str = base64.b64encode(arr.tobytes()).decode()
    return packed_str


def str_to_bool_arr(s: str, shape: tuple) -> np.ndarray:
    """Converts a string to a boolean array."""
    # Convert the string back into bytes using base64 decoding
    bytes_ = base64.b64decode(s)

    # Convert bytes to np.uint8 array
    bytes_array = np.frombuffer(bytes_, dtype=np.uint8)

    # Reshape the data back into a boolean array
    unpacked = bytes_array.reshape(shape)
    return unpacked


def mask_to_str(mask: np.ndarray) -> Dict[str, Any]:
    """Compactly encode a boolean mask as bit-packed base64 + shape.

    ~30x smaller than ``mask.tolist()`` over JSON, and avoids materialising a
    giant nested Python list (a real RAM spike on full-resolution masks).
    """
    m = np.ascontiguousarray(mask, dtype=bool)
    return {"shape": list(m.shape), "data": base64.b64encode(np.packbits(m)).decode("ascii")}


def str_to_mask(entry: Any) -> np.ndarray:
    """Inverse of ``mask_to_str``. Also accepts a raw nested list / array so a
    policy can still talk to an older SAM3 server (back-compat)."""
    if isinstance(entry, dict):
        shape = tuple(entry["shape"])
        n = int(np.prod(shape)) if shape else 0
        bits = np.unpackbits(np.frombuffer(base64.b64decode(entry["data"]), dtype=np.uint8))
        return bits[:n].reshape(shape).astype(bool)
    return np.asarray(entry, dtype=bool)


def image_to_str(img_np: np.ndarray, quality: float = 90.0) -> str:
    """
    Convert a NumPy image array to a base64-encoded JPEG string for transport.

    Array is explicitly cast to uint8 (OpenCV expects this for JPEG encoding, 
    passing float arrays can cause silent failures or garbage output).
    
    "ascontiguousarray" ensures the memory layout is C-contiguous, which OpenCV requires.
    Non-contiguous arrays (e.g. slices or transposed arrays) can cause cv2.imencode 
    to fail or produce incorrect results.
    """
    img_u8 = np.ascontiguousarray(img_np.astype(np.uint8))
    # Rounding rather thna truncating for floats
    q = int(round(quality))
    # Clamps quality to the valid JPEG range of 0–100, prevening out of range values to IMWRITE_JPEG_QUALITY
    q = max(0, min(100, q))
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), q]
    retval, buffer = cv2.imencode(".jpg", img_u8, encode_param)
    img_str = base64.b64encode(buffer).decode("utf-8")
    return img_str


def str_to_image(img_str: str) -> np.ndarray:
    img_bytes = base64.b64decode(img_str)
    img_arr = np.frombuffer(img_bytes, dtype=np.uint8)
    img_np = cv2.imdecode(img_arr, cv2.IMREAD_ANYCOLOR)
    return img_np


# Request retry/timeout behaviour. A missing or unreachable VLM server should
# fail fast rather than block the caller (e.g. the policy Flask handler) for
# minutes. Override via env vars if a server legitimately needs longer.
_REQUEST_RETRIES = int(os.environ.get("VLM_REQUEST_RETRIES", "3"))
_REQUEST_BACKOFF_S = float(os.environ.get("VLM_REQUEST_BACKOFF", "1.0"))
# Per-request give-up windows: a refused connection (server down) gives up
# quickly, while a reachable-but-slow server (e.g. still loading weights) is
# allowed the longer window.
_CONNECT_GIVEUP_S = float(os.environ.get("VLM_CONNECT_TIMEOUT", "3.0"))
_BUSY_GIVEUP_S = float(os.environ.get("VLM_BUSY_TIMEOUT", "20.0"))


def send_request(url: str, **kwargs: Any) -> dict:
    last_exc = None
    for attempt in range(_REQUEST_RETRIES):
        try:
            return _send_request(url, **kwargs)
        except Exception as e:
            last_exc = e
            if attempt < _REQUEST_RETRIES - 1:
                print(
                    f"Request to {url} failed: {e}. Retrying in {_REQUEST_BACKOFF_S:.1f}s "
                    f"({attempt + 1}/{_REQUEST_RETRIES})..."
                )
                time.sleep(_REQUEST_BACKOFF_S)

    # Raise instead of exit(): a down VLM server must not take the hosting
    # process (e.g. the policy server) down with it. The exception propagates to
    # the caller (Flask returns 500), so clients fail fast instead of the 30 s
    # read timeout previously seen on the bridge.
    raise RuntimeError(f"Request to {url} failed after {_REQUEST_RETRIES} attempts") from last_exc


def _send_request(url: str, **kwargs: Any) -> dict:
    lockfiles_dir = "lockfiles"
    if not os.path.exists(lockfiles_dir):
        os.makedirs(lockfiles_dir)
    filename = url.replace("/", "_").replace(":", "_") + ".lock"
    filename = filename.replace("localhost", socket.gethostname())
    filename = os.path.join(lockfiles_dir, filename)
    try:
        while True:
            # Use a while loop to wait until this filename does not exist
            while os.path.exists(filename):
                # If the file exists, wait 50ms and try again
                time.sleep(0.05)

                try:
                    # If the file was last modified more than 120 seconds ago, delete it
                    if time.time() - os.path.getmtime(filename) > 120:
                        os.remove(filename)
                except FileNotFoundError:
                    pass

            rand_str = str(random.randint(0, 1000000))

            with open(filename, "w") as f:
                f.write(rand_str)
            time.sleep(0.05)
            try:
                with open(filename, "r") as f:
                    if f.read() == rand_str:
                        break
            except FileNotFoundError:
                pass

        # Create a payload dict which is a clone of kwargs but all np.array values are
        # converted to base64-encoded JPEG strings (or other encodings as appropriate)
        payload = {}
        for k, v in kwargs.items():
            if isinstance(v, np.ndarray):
                payload[k] = image_to_str(v, quality=kwargs.get("quality", 90))
            else:
                payload[k] = v

        # Set the headers
        headers = {"Content-Type": "application/json"}

        start_time = time.time()
        start_iteration = True     # On the first iteration, have a full 10s timeout due to model initialisation
        while True:
            try:
                if start_iteration:
                    resp = requests.post(url, headers=headers, json=payload, timeout=10)
                    start_iteration = False
                # After the first iteration, use a shorter timeout for retries, since the model should be loaded
                else:
                    resp = requests.post(url, headers=headers, json=payload, timeout=1)
                if resp.status_code == 200:
                    result = resp.json()
                    break
                else:
                    raise Exception("Request failed")
            except requests.exceptions.ConnectionError as e:
                # Server is down/unreachable: give up quickly instead of
                # busy-looping (which floods the log with refused-connection
                # errors), backing off briefly between attempts.
                if time.time() - start_time > _CONNECT_GIVEUP_S:
                    raise
                time.sleep(0.5)
            except (
                requests.exceptions.Timeout,
                requests.exceptions.RequestException,
            ) as e:
                # Server reachable but slow (e.g. still loading weights): keep
                # waiting up to the longer window.
                print(e)
                if time.time() - start_time > _BUSY_GIVEUP_S:
                    raise Exception(f"Request to {url} timed out after {_BUSY_GIVEUP_S:.0f}s")
                time.sleep(0.5)

        try:
            # Delete the lock file
            os.remove(filename)
        except FileNotFoundError:
            pass

    except Exception as e:
        try:
            # Delete the lock file
            os.remove(filename)
        except FileNotFoundError:
            pass
        raise e

    return result
