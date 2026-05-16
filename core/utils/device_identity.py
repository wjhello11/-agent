from __future__ import annotations

import re
from typing import Any


def normalize_device_user_id(device_id: Any) -> str:
    """
    Use one XiaoZhi device as one user.

    Device IDs may arrive as MAC-like strings with colons, dashes or uppercase
    letters. We normalize them into a compact lowercase ID so memory and health
    profile rows remain stable across connection paths.
    """
    text = str(device_id or "").strip()
    if not text:
        return ""
    compact = re.sub(r"[^0-9a-zA-Z]+", "", text).lower()
    return compact or text.lower()
