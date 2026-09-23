"""Shared JSON helpers.

``json_safe`` replaces non-finite floats (NaN/Infinity, which a degenerate audio
crop can produce) with ``null`` so every payload stays valid JSON. It used to be
duplicated in ``project.py`` and ``server.py``; a single copy cannot drift.
"""

from __future__ import annotations

import math


def json_safe(value):
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value
