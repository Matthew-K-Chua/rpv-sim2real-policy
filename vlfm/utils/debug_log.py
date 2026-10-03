"""Opt-in verbose tracing for the perception / value-map hot loop.

Embodied-RPV-NOTE: added for TurtleBot4 deployment. These traces were written to
debug signal placement on the real robot and run per-detection, per-step, so
they are gated behind ``RPV_DEBUG=1`` rather than printed unconditionally --
synchronous stdout in the step loop is a measurable latency cost, and a
benchmark run should not pay it.
"""

import os

DEBUG: bool = os.environ.get("RPV_DEBUG", "") == "1"


def dbg(msg: str) -> None:
    """Print ``msg`` only when RPV_DEBUG=1."""
    if DEBUG:
        print(msg, flush=True)
