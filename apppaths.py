"""Where the app reads bundled resources and writes user data.

Running from source, both live in the repository root. A PyInstaller build is
different: read-only resources (``static/``) are unpacked into ``sys._MEIPASS``,
while ``data/`` and any locally installed ``engines/`` must sit next to the exe
so they survive and stay writable.
"""

from __future__ import annotations

import os
import sys


def bundle_dir() -> str:
    """Read-only resources shipped with the app (``static/``)."""
    return getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))


def writable_dir() -> str:
    """Writable home: the exe's folder when frozen, else the repo root."""
    if getattr(sys, "frozen", False):
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.dirname(os.path.abspath(__file__))


def data_dir() -> str:
    """The runtime ``data/`` directory, honouring the ``SSP_DATA_DIR`` override."""
    return os.environ.get("SSP_DATA_DIR") or os.path.join(writable_dir(), "data")
