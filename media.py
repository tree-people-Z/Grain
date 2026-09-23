"""ffmpeg/ffprobe helpers."""

from __future__ import annotations

import glob
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile

import apppaths


def _find_tool(name: str) -> str:
    """Prefer a bundled ``runtime/ffmpeg/bin`` copy, then PATH."""
    directory = apppaths.bundled_ffmpeg_dir()
    if directory:
        exe = os.path.join(directory, name + (".exe" if os.name == "nt" else ""))
        if os.path.isfile(exe):
            return exe
    return shutil.which(name) or name


FFMPEG = _find_tool("ffmpeg")
FFPROBE = _find_tool("ffprobe")

AUDIO_SAMPLE_RATE = 16000
_AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".wma"}
_VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".flv", ".ts", ".m4v", ".wmv"}


def kind_of(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext in _AUDIO_EXTENSIONS:
        return "audio"
    if ext in _VIDEO_EXTENSIONS:
        return "video"
    return "unknown"


def probe_duration(path: str) -> float:
    """Duration in seconds via ffprobe; 0.0 when it cannot be determined."""
    try:
        result = subprocess.run(
            [FFPROBE, "-v", "error", "-show_entries", "format=duration",
             "-of", "json", path],
            capture_output=True, text=True, timeout=60,
        )
        payload = json.loads(result.stdout or "{}")
        return float(payload.get("format", {}).get("duration") or 0.0)
    except Exception:
        return 0.0


def to_wav16k(source: str, destination: str | None = None) -> str:
    """Decode any media file to mono 16 kHz 16-bit PCM WAV."""
    if destination is None:
        handle, destination = tempfile.mkstemp(suffix=".wav")
        os.close(handle)
    command = [
        FFMPEG, "-y", "-v", "error", "-i", source,
        "-vn", "-ac", "1", "-ar", str(AUDIO_SAMPLE_RATE),
        "-acodec", "pcm_s16le", destination,
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=3600)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr.strip()[:400]}")
    return destination


def demucs_available() -> bool:
    """True when the optional Demucs vocal separator is importable."""
    try:
        return importlib.util.find_spec("demucs") is not None
    except Exception:
        return False


def separate_vocals(source: str, destination: str) -> str:
    """Isolate the vocal stem of ``source`` into a mono 16 kHz WAV.

    Runs Demucs (htdemucs, two-stem mode) in a subprocess of the current
    interpreter so the model never lingers in the app process. Background music
    and the OP/ED accompaniment are the biggest cause of speaker confusion in
    anime audio, so separating the vocals first usually beats swapping engines.
    """
    if not demucs_available():
        raise RuntimeError("未安装 Demucs。安装：pip install demucs")
    outdir = tempfile.mkdtemp(prefix="demucs_")
    command = [
        sys.executable, "-m", "demucs", "--two-stems=vocals",
        "-o", outdir, source,
    ]
    # Separation of a full episode can take many minutes on CPU.
    result = subprocess.run(command, capture_output=True, text=True, timeout=7200)
    if result.returncode != 0:
        raise RuntimeError(f"Demucs failed: {(result.stderr or result.stdout).strip()[:400]}")
    matches = glob.glob(os.path.join(outdir, "*", "*", "vocals.wav"))
    if not matches:
        raise RuntimeError("Demucs 未生成 vocals 音轨。")
    try:
        return to_wav16k(matches[0], destination)
    finally:
        shutil.rmtree(outdir, ignore_errors=True)


def cut_wav(source_wav: str, start: float, end: float, destination: str) -> str:
    """Slice a segment out of an existing 16 kHz mono WAV."""
    command = [
        FFMPEG, "-y", "-v", "error", "-i", source_wav,
        "-ss", f"{max(0.0, start):.3f}", "-to", f"{max(0.0, end):.3f}",
        "-acodec", "pcm_s16le", destination,
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr.strip()[:400]}")
    return destination
