import sys

import lerobot_patches.custom_patches

from .data import shared_memory_utils as _shared_memory_utils

sys.modules.setdefault("shared_memory_utils", _shared_memory_utils)
