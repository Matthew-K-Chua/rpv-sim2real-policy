"""Per-stage step timing for the RPV policy server, behind ``RPV_TIMING=1``.

Why this exists: you cannot attribute a saving to a change without a per-stage
number, and this path had none. Every earlier latency
figure was measured by hand, once — which is why they could not be re-checked
after a change.

Why it is written as *shims* rather than ``perf_counter`` scopes inside the policy:
the stages worth timing (``_update_object_map``, ``recompute_value_map``,
``apply_free_space_mask``, ``_get_policy_info``) live in RPV-lineage files, and this
repo's whole Tier-A/Tier-B split (§6 vs §7 of that doc) exists to keep those files
byte-identical. ``install_step_timers`` wraps the bound methods on the policy
*instance*, so instrumentation adds exactly zero lines of diff against RPV. When
``RPV_TIMING`` is unset nothing is installed and nothing is wrapped.

Output — one line per step::

    [timing] step=12 mode=explore total=2431ms | decode=12 obstacle=48 policy=2310
             (objmap=980 [detect=910] signals=210 recompute=920 freemask=44 info=140)
             status=58 | dirty=37/37

``dirty=n_dirty/n_tracked`` is the number §7.1 B-1 and B-3 have to move: it is
counted *after* ``_mark_dirty_objects`` runs, so it is exactly the set
``recompute_value_map`` is about to rebuild, not an estimate.
"""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List

TIMING_ENABLED: bool = os.environ.get("RPV_TIMING", "") == "1"

# Stages that run *inside* another stage. Reported in brackets and excluded from any
# "unaccounted" arithmetic, so the line cannot be read as double-counting.
_NESTED = {"detect"}

# Print order. Anything a shim records that is not listed here still prints, after
# these, so adding a wrap does not silently produce an invisible number.
_ORDER = [
    "decode",
    "obstacle",
    "policy",
    "objmap",
    "detect",
    "signals",
    "redetect",
    "recompute",
    "freemask",
    "info",
    "status",
]


class StepTimer:
    """Stage durations for ONE step. Create a fresh one per step."""

    def __init__(self) -> None:
        self._ms: Dict[str, float] = {}
        self.counters: Dict[str, Any] = {}

    def add(self, name: str, ms: float) -> None:
        # Accumulate: a stage can legitimately run more than once per step (one
        # _update_object_map per rgbd tuple), and summing is the honest total.
        self._ms[name] = self._ms.get(name, 0.0) + ms

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.add(name, (time.perf_counter() - t0) * 1e3)

    def line(self, step: int, mode: str, total_ms: float) -> str:
        seen = set(self._ms)
        ordered: List[str] = [n for n in _ORDER if n in seen]
        ordered += sorted(seen - set(_ORDER))

        parts: List[str] = []
        for name in ordered:
            ms = self._ms[name]
            parts.append(f"[{name}={ms:.0f}]" if name in _NESTED else f"{name}={ms:.0f}")

        counters = " ".join(f"{k}={v}" for k, v in self.counters.items())
        tail = f" | {counters}" if counters else ""
        return (
            f"[timing] step={step} mode={mode} total={total_ms:.0f}ms | "
            + " ".join(parts)
            + tail
        )


def _wrap(timer_of: Any, owner: Any, attr: str, stage: str) -> bool:
    """Shadow ``owner.attr`` with a timed version. Returns whether it was installed.

    ``timer_of`` is a zero-arg callable returning the StepTimer for the step in
    flight (or None between steps) — the timer object changes every step, so the
    wrapper must look it up per call rather than close over one.
    """
    original = getattr(owner, attr, None)
    if original is None or not callable(original):
        print(f"[timing] no {attr} on {type(owner).__name__}; not timed", flush=True)
        return False

    def timed(*args: Any, **kwargs: Any) -> Any:
        timer = timer_of()
        if timer is None:
            return original(*args, **kwargs)
        t0 = time.perf_counter()
        try:
            return original(*args, **kwargs)
        finally:
            timer.add(stage, (time.perf_counter() - t0) * 1e3)

    setattr(owner, attr, timed)  # instance attribute shadows the class method
    return True


def _wrap_dirty_counter(timer_of: Any, value_map: Any) -> bool:
    """Record ``dirty=n_dirty/n_tracked`` after ``_mark_dirty_objects`` runs.

    Counted here rather than before ``recompute_value_map`` because
    ``_mark_dirty_objects`` is what promotes clean objects to dirty; counting any
    earlier reports the pre-promotion set, which is the wrong number.
    """
    original = getattr(value_map, "_mark_dirty_objects", None)
    if original is None:
        print("[timing] no _mark_dirty_objects on the value map; dirty count off", flush=True)
        return False

    def counted(*args: Any, **kwargs: Any) -> Any:
        result = original(*args, **kwargs)
        timer = timer_of()
        if timer is not None:
            tracked = getattr(value_map, "_tracked_objects", [])
            n_dirty = sum(1 for o in tracked if o.get("dirty", True))
            timer.counters["dirty"] = f"{n_dirty}/{len(tracked)}"
        return result

    setattr(value_map, "_mark_dirty_objects", counted)
    return True


def install_step_timers(policy: Any, timer_of: Any) -> None:
    """Install the instance-level shims. No-op unless ``RPV_TIMING=1``.

    Safe to call once at server start; wrapping twice would double-count, so it
    guards itself with a marker attribute.
    """
    if not TIMING_ENABLED:
        return
    if getattr(policy, "_rpv_timers_installed", False):
        return

    installed: List[str] = []
    for attr, stage in (
        ("_update_object_map", "objmap"),
        ("_get_object_detections", "detect"),
        ("_get_object_signals", "signals"),
        ("_redetect_target_object", "redetect"),
        ("_get_policy_info", "info"),
    ):
        if _wrap(timer_of, policy, attr, stage):
            installed.append(stage)

    value_map = getattr(policy, "_value_map", None)
    if value_map is None:
        print("[timing] policy has no _value_map; map stages not timed", flush=True)
    else:
        for attr, stage in (
            ("recompute_value_map", "recompute"),
            ("apply_free_space_mask", "freemask"),
        ):
            if _wrap(timer_of, value_map, attr, stage):
                installed.append(stage)
        if _wrap_dirty_counter(timer_of, value_map):
            installed.append("dirty")

    policy._rpv_timers_installed = True
    print(f"[timing] RPV_TIMING=1 — timing {', '.join(installed)}", flush=True)
