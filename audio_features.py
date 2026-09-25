"""WAV loading + waveform peaks (stdlib only; NumPy accelerates when present)."""

from __future__ import annotations

import array
import math
import wave

try:  # NumPy is present whenever a neural engine (torch) is installed.
    import numpy as _np
except Exception:  # pragma: no cover - base install stays dependency-free
    _np = None

HAS_NUMPY = _np is not None


def read_wav(path: str, target_rate: int = 16000) -> tuple["array.array[float]", int]:
    """Read a mono 16-bit PCM WAV into floats in [-1, 1].

    Returns an ``array('f')`` (4 bytes/sample) rather than a Python ``list`` of
    floats: an hour of 16 kHz audio drops from >1.5 GB to ~230 MB. It still
    behaves like a sequence (``len``, slicing, truthiness).
    """
    with wave.open(path, "rb") as handle:
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        rate = handle.getframerate()
        frames = handle.readframes(handle.getnframes())
    if width != 2:
        raise ValueError(f"expected 16-bit PCM wav, got {width * 8}-bit")
    samples = array.array("h")
    samples.frombytes(frames)
    if channels > 1:
        mono = array.array("f")
        append = mono.append
        for i in range(0, len(samples) - channels + 1, channels):
            append(sum(samples[i:i + channels]) / (channels * 32768.0))
    else:
        mono = array.array("f", (v / 32768.0 for v in samples))
    signal = mono
    if rate != target_rate:
        signal = _resample(signal, rate, target_rate)
        rate = target_rate
    return signal, rate


def waveform_peaks(signal, buckets: int = 1200) -> list[float]:
    """Downsample a signal to ``buckets`` normalised peak values for drawing."""
    if not signal or buckets <= 0:
        return []
    size = max(1, math.ceil(len(signal) / buckets))
    if _np is not None:
        data = _np.asarray(signal, dtype=_np.float32)
        count = math.ceil(len(data) / size)
        padding = count * size - len(data)
        if padding:
            data = _np.pad(data, (0, padding))
        peaks = _np.abs(data.reshape(count, size)).max(axis=1)
        ceiling = float(peaks.max()) if peaks.size else 0.0
        if ceiling > 0:
            peaks = _np.minimum(1.0, peaks / ceiling)
        return [round(float(value), 4) for value in peaks]
    peaks = []
    for index in range(0, len(signal), size):
        window = signal[index:index + size]
        peaks.append(max(abs(value) for value in window))
    ceiling = max(peaks) if peaks else 1.0
    if ceiling > 0:
        peaks = [min(1.0, value / ceiling) for value in peaks]
    return [round(value, 4) for value in peaks]


def _resample(signal, src_rate: int, dst_rate: int) -> "array.array[float]":
    """Linear-interpolation resample; adequate for waveform display."""
    if not signal:
        return array.array("f")
    ratio = dst_rate / src_rate
    out_len = int(len(signal) * ratio)
    output = array.array("f")
    append = output.append
    last = len(signal) - 1
    for i in range(out_len):
        position = i / ratio
        left = int(position)
        right = left + 1 if left + 1 <= last else last
        frac = position - left
        append(signal[left] * (1 - frac) + signal[right] * frac)
    return output
