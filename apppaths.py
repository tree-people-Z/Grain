"""Where the app reads bundled resources and writes user data.

Everything lives in the repository root: ``static/`` is read from here and the
runtime ``data/`` directory is created next to it. ``SSP_DATA_DIR`` overrides
the data location (used by tests so they never touch real user data).
"""

from __future__ import annotations

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def writable_dir() -> str:
    """Writable home: the repository root."""
    return BASE_DIR


def data_dir() -> str:
    """The runtime ``data/`` directory, honouring the ``SSP_DATA_DIR`` override."""
    return os.environ.get("SSP_DATA_DIR") or os.path.join(BASE_DIR, "data")


# --- bundled resources -------------------------------------------------------
#
# A portable package ships its heavy assets next to the code:
#
#     <root>/runtime/ffmpeg/bin/ffmpeg.exe|ffprobe.exe
#     <root>/models/hf/hub/models--...            (HuggingFace hub cache)

def bundled_ffmpeg_dir() -> str | None:
    """Folder holding a bundled ``ffmpeg``/``ffprobe``, or None."""
    directory = os.path.join(writable_dir(), "runtime", "ffmpeg", "bin")
    return directory if os.path.isdir(directory) else None


def bundled_hf_hub_cache() -> str | None:
    """Bundled HuggingFace ``hub`` cache directory, or None."""
    directory = os.path.join(writable_dir(), "models", "hf", "hub")
    return directory if os.path.isdir(directory) else None


def configure_model_caches() -> None:
    """Point HuggingFace at the bundled cache so pyannote loads offline.

    Only sets the variable when the bundled directory exists and the user has
    not already overridden it, so a source checkout without bundled models keeps
    using the default per-user cache. When weights are bundled and no token is
    configured, HF is forced offline: the gated community-1 weights then load
    straight from the local cache with no token and no network access.
    """
    hf_hub = bundled_hf_hub_cache()
    if hf_hub and not (os.environ.get("HF_HUB_CACHE")
                       or os.environ.get("HUGGINGFACE_HUB_CACHE")):
        os.environ["HF_HUB_CACHE"] = hf_hub
        os.environ.setdefault("HF_HOME", os.path.dirname(hf_hub))
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if hf_hub and not token:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
