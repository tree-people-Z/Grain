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


# --- bundled resources (the "整合包" layout) ---------------------------------
#
# A portable package ships its heavy assets next to the exe (or, for a onefile
# build, inside ``_internal``). Both are searched, writable_dir() first so a
# package can update its own copy:
#
#     <root>/runtime/ffmpeg/bin/ffmpeg.exe|ffprobe.exe
#     <root>/models/hf/hub/models--...            (HuggingFace hub cache)
#     <root>/models/modelscope/models/...         (ModelScope cache)
#     <root>/engines/<key>/                        (isolated external engines)

def _first_dir(candidates) -> str | None:
    for candidate in candidates:
        if candidate and os.path.isdir(candidate):
            return candidate
    return None


def bundled_ffmpeg_dir() -> str | None:
    """Folder holding a bundled ``ffmpeg``/``ffprobe``, or None."""
    return _first_dir([
        os.path.join(writable_dir(), "runtime", "ffmpeg", "bin"),
        os.path.join(bundle_dir(), "runtime", "ffmpeg", "bin"),
    ])


def bundled_hf_hub_cache() -> str | None:
    """Bundled HuggingFace ``hub`` cache directory, or None."""
    return _first_dir([
        os.path.join(writable_dir(), "models", "hf", "hub"),
        os.path.join(bundle_dir(), "models", "hf", "hub"),
    ])


def bundled_modelscope_cache() -> str | None:
    """Bundled ModelScope cache root (its ``models/`` lives inside), or None."""
    return _first_dir([
        os.path.join(writable_dir(), "models", "modelscope"),
        os.path.join(bundle_dir(), "models", "modelscope"),
    ])


def configure_model_caches() -> None:
    """Point HF / ModelScope at the bundled caches so engines load offline.

    Only sets a variable when the user has not already overridden it and the
    bundled directory actually exists, so a normal source checkout (no bundled
    models) keeps using the default per-user cache untouched.
    """
    hf_hub = bundled_hf_hub_cache()
    if hf_hub and not (os.environ.get("HF_HUB_CACHE")
                       or os.environ.get("HUGGINGFACE_HUB_CACHE")):
        os.environ["HF_HUB_CACHE"] = hf_hub
        os.environ.setdefault("HF_HOME", os.path.dirname(hf_hub))
    ms = bundled_modelscope_cache()
    if ms and not os.environ.get("MODELSCOPE_CACHE"):
        os.environ["MODELSCOPE_CACHE"] = ms
