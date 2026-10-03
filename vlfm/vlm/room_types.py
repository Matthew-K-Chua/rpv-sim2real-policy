import os
from typing import List, Optional

import yaml

# The RPV room prior. Every detected label is scored as a softmax distribution
# over these candidates, and the target's distribution is dot-producted against
# it -- so this list *defines* what "relational prior" means for a run.
#
# Embodied-RPV-NOTE: this default is the upstream RPV/VLFM home-environment set
# and MUST NOT be edited to suit a deployment site. To run in a different kind
# of space (e.g. the lab), point ROOM_TYPES_FILE at a YAML file instead:
#     ROOM_TYPES_FILE=data/room_types_lab.yaml
# Benchmark runs leave it unset and get the list below.
ROOM_TYPES = room_types = [
    "bathroom",
    "bedroom",
    "dining room",
    "garage",
    "kitchen",
    "hallway",
    "laundry",
    "living room",
    "home office",
]


def load_room_types(path: Optional[str] = None) -> List[str]:
    """Room-type candidates for the RPV prior.

    Reads ``path`` (or ``$ROOM_TYPES_FILE``) when given; otherwise returns the
    upstream ``ROOM_TYPES`` above. The YAML may be a bare list or a mapping with
    a ``room_types`` key.
    """
    path = path or os.environ.get("ROOM_TYPES_FILE", "")
    if not path:
        return list(ROOM_TYPES)

    with open(path, "r") as f:
        data = yaml.safe_load(f)
    rooms = data.get("room_types", []) if isinstance(data, dict) else data
    if not rooms:
        raise ValueError(f"ROOM_TYPES_FILE={path} contained no room types")
    print(f"[room-types] loaded {len(rooms)} room types from {path}")
    return [str(r) for r in rooms]
