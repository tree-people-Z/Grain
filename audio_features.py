"""Lightweight acoustic feature extraction and clustering (stdlib only).

This is deliberately dependency-free so the tool runs on a clean machine.
When ``pyannote.audio`` is installed the caller uses that engine instead and
this module is only used for voiceprint enrolment / prior matching.

Features per frame: 20 log-spaced band energies + zero-crossing rate +
normalised energy + autocorrelation pitch. Vectors are mean-normalised before
clustering so channel/gain differences matter less.
"""

from __future__ import annotations

import array
import math
import wave

try:  # NumPy is present whenever a neural engine (torch) is installed.
    import numpy as _np
except Exception:  # pragma: no cover - base install stays dependency-free
    _np = None

HAS_NUMPY = _np is not None

FRAME_MS = 25.0
HOP_MS = 10.0
FFT_SIZE = 512
N_BANDS = 20
MIN_PITCH_HZ = 70.0
MAX_PITCH_HZ = 320.0


# --- FFT --------------------------------------------------------------------

# Bit-reversal permutation and per-stage twiddle factors, built once per size
# and reused for every frame. The old implementation formatted/reversed a binary
# string for every sample of every frame, which dominated feature extraction.
_FFT_PLANS: dict[int, tuple[list[int], list[tuple[int, list[complex]]]]] = {}


def _fft_plan(size: int):
    plan = _FFT_PLANS.get(size)
    if plan is not None:
        return plan
    bits = size.bit_length() - 1
    reversal = [0] * size
    for index in range(size):
        value, reversed_index = index, 0
        for _ in range(bits):
            reversed_index = (reversed_index << 1) | (value & 1)
            value >>= 1
        reversal[index] = reversed_index
    stages: list[tuple[int, list[complex]]] = []
    length = 2
    while length <= size:
        angle = -2.0 * math.pi / length
        step = complex(math.cos(angle), math.sin(angle))
        roots = [complex(1.0, 0.0)] * (length // 2)
        factor = complex(1.0, 0.0)
        for offset in range(length // 2):
            roots[offset] = factor
            factor *= step
        stages.append((length, roots))
        length *= 2
    plan = (reversal, stages)
    _FFT_PLANS[size] = plan
    return plan


def _fft(values) -> list[complex]:
    """Iterative radix-2 Cooley-Tukey FFT. ``len(values)`` must be a power of 2."""
    size = len(values)
    if size & (size - 1):
        raise ValueError("FFT size must be a power of two")
    reversal, stages = _fft_plan(size)
    data = [complex(v, 0.0) for v in values]
    for index in range(size):
        reversed_index = reversal[index]
        if reversed_index > index:
            data[index], data[reversed_index] = data[reversed_index], data[index]
    for length, roots in stages:
        half = length // 2
        for start in range(0, size, length):
            for offset in range(half):
                even = data[start + offset]
                odd = data[start + offset + half] * roots[offset]
                data[start + offset] = even + odd
                data[start + offset + half] = even - odd
    return data


def _power_spectrum(values) -> list[float]:
    """Power (|X|^2) for FFT bins ``1 .. N/2-1``. NumPy when available."""
    size = len(values)
    if _np is not None:
        spectrum = _np.fft.rfft(_np.asarray(values, dtype=_np.float64))
        return (_np.abs(spectrum[1:size // 2]) ** 2).tolist()
    data = _fft(values)
    return [abs(data[i]) ** 2 for i in range(1, size // 2)]


def _hann(size: int) -> list[float]:
    return [0.5 - 0.5 * math.cos(2.0 * math.pi * n / (size - 1)) for n in range(size)]


_WINDOW = _hann(FFT_SIZE)


def _band_edges(sample_rate: int, count: int) -> list[tuple[int, int]]:
    """Log-spaced mel-like band edges mapped onto FFT bins (80 Hz .. 7.6 kHz)."""
    low, high = 80.0, min(7600.0, sample_rate / 2.0 - 100.0)
    edges = []
    for i in range(count + 1):
        ratio = i / count
        freq = low * (high / low) ** ratio
        bin_index = int(round(freq / (sample_rate / FFT_SIZE)))
        edges.append(bin_index)
    return [
        (max(1, edges[i]), max(max(1, edges[i]) + 1, edges[i + 1]))
        for i in range(count)
    ]


# --- WAV loading ------------------------------------------------------------

def read_wav(path: str, target_rate: int = 16000) -> tuple["array.array[float]", int]:
    """Read a mono 16-bit PCM WAV into floats in [-1, 1].

    Returns an ``array('f')`` (4 bytes/sample) rather than a Python ``list`` of
    floats (~32 bytes/sample): an hour of 16 kHz audio drops from >1.5 GB to
    ~230 MB. It still behaves like a sequence (``len``, slicing, truthiness), so
    callers are unaffected. NumPy, when installed, converts it with no copy.
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
    """Downsample a signal to ``buckets`` normalised peak values for drawing.

    Vectorised with NumPy when present; the old per-bucket ``max(abs(...))``
    generator decoded the whole file into Python floats just to take maxima.
    """
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
    """Linear-interpolation resample; adequate for feature extraction."""
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


# --- framing / features -----------------------------------------------------

def _frame_features(signal: list[float], rate: int, start: int, end: int,
                    bands: list[tuple[int, int]]) -> list[list[float]]:
    frame_len = int(rate * FRAME_MS / 1000)
    hop = int(rate * HOP_MS / 1000)
    if frame_len > FFT_SIZE:
        frame_len = FFT_SIZE
    vectors = []
    position = start
    while position + frame_len <= end:
        frame = signal[position:position + frame_len]
        windowed = [frame[i] * _WINDOW[i] for i in range(frame_len)]
        if frame_len < FFT_SIZE:
            windowed += [0.0] * (FFT_SIZE - frame_len)
        power = _power_spectrum(windowed)
        total = sum(power) + 1e-12

        vector = []
        for low, high in bands:
            energy = sum(power[low:high]) / max(1, high - low)
            vector.append(math.log(energy + 1e-10))
        # Normalise the spectral envelope to remove loudness dependence.
        mean = sum(vector) / len(vector)
        vector = [v - mean for v in vector]

        rms = math.sqrt(sum(v * v for v in frame) / max(1, len(frame)))
        zero_crossings = sum(
            1 for i in range(1, len(frame))
            if (frame[i - 1] < 0) != (frame[i] < 0)
        ) / max(1, len(frame) - 1)

        vector.append(math.log(rms + 1e-6))
        vector.append(zero_crossings)
        vector.append(_pitch(frame, rate))
        vectors.append(vector)
        position += hop
    return vectors


def _pitch(frame, rate: int) -> float:
    """Normalised autocorrelation pitch in Hz, or 0 when unvoiced."""
    if not frame:
        return 0.0
    energy = sum(v * v for v in frame)
    if energy < 1e-6:
        return 0.0
    length = len(frame)
    min_lag = int(rate / MAX_PITCH_HZ)
    max_lag = min(int(rate / MIN_PITCH_HZ), length - 1)
    if max_lag <= min_lag:
        return 0.0

    if _np is not None:
        x = _np.asarray(frame, dtype=_np.float64)
        cumulative = _np.concatenate(([0.0], _np.cumsum(x * x)))
        # ac[k] = sum_i x[i] * x[i+k]
        autocorr = _np.correlate(x, x, mode="full")[length - 1:]
        lags = _np.arange(min_lag, max_lag)
        tail_energy = cumulative[length - lags]
        scores = autocorr[min_lag:max_lag] / _np.sqrt(energy * tail_energy + 1e-12)
        best = int(_np.argmax(scores))
        best_score = float(scores[best])
        best_lag = min_lag + best
    else:
        # Prefix sums make the per-lag normaliser O(1) instead of O(L).
        cumulative = [0.0] * (length + 1)
        for i, v in enumerate(frame):
            cumulative[i + 1] = cumulative[i] + v * v
        best_lag, best_score = 0, 0.0
        for lag in range(min_lag, max_lag):
            count = length - lag
            score = sum(frame[i] * frame[i + lag] for i in range(count))
            score /= math.sqrt(energy * cumulative[count] + 1e-12)
            if score > best_score:
                best_score, best_lag = score, lag

    if best_score < 0.35 or best_lag == 0:
        return 0.0
    return rate / best_lag


def segment_embedding(signal: list[float], rate: int, start: float, end: float) -> list[float]:
    """Speaker embedding for ``[start, end]`` seconds; [] when too short.

    Mean frame features (band energies, loudness, ZCR, pitch) plus three
    segment-level pitch statistics (mean log-f0 of voiced frames, its spread,
    and voicing rate) — the strongest per-speaker cues in this feature space.
    """
    bands = _band_edges(rate, N_BANDS)
    frames = _frame_features(
        signal, rate,
        max(0, int(start * rate)),
        min(len(signal), int(end * rate)),
        bands,
    )
    if not frames:
        return []
    dimension = len(frames[0])
    mean = [sum(f[i] for f in frames) / len(frames) for i in range(dimension)]

    pitches = [frame[-1] for frame in frames]
    voiced = [p for p in pitches if p > 0]
    if voiced:
        logs = [math.log(p) for p in voiced]
        mean_log = sum(logs) / len(logs)
        spread = math.sqrt(sum((v - mean_log) ** 2 for v in logs) / len(logs))
        voicing = len(voiced) / len(pitches)
    else:
        mean_log, spread, voicing = 0.0, 0.0, 0.0
    return mean + [mean_log, spread, voicing]


def speech_trim(signal: list[float], rate: int, start: float, end: float,
                low: float = 0.05, high: float = 0.98,
                min_keep: float = 0.10) -> tuple[float, float]:
    """Shrink ``[start, end]`` to the speech-active core by cumulative energy.

    Subtitle and diarization boundaries usually include a little silence or a
    breath; embedding the whole crop dilutes the speaker's voiceprint. This
    trims the low-energy head and tail to the interval carrying ``low``–``high``
    of the total energy. Scale-invariant, so quiet and loud cues behave alike;
    the original interval is returned when it is too short, silent, or would
    shrink below ``min_keep`` seconds.
    """
    begin = max(0, int(start * rate))
    finish = min(len(signal), int(end * rate))
    if finish - begin < int(0.20 * rate):
        return start, end
    hop = max(1, int(rate * 0.01))
    energies = []
    position = begin
    while position + hop <= finish:
        frame = signal[position:position + hop]
        energies.append(sum(v * v for v in frame))
        position += hop
    total = sum(energies)
    if not energies or total <= 1e-12:
        return start, end
    low_hit, high_hit = total * low, total * high
    running = 0.0
    first = 0
    for index, energy in enumerate(energies):
        running += energy
        if running >= low_hit:
            first = index
            break
    running = 0.0
    last = len(energies) - 1
    for index, energy in enumerate(energies):
        running += energy
        if running >= high_hit:
            last = index
            break
    trimmed_start = start + first * hop / rate
    trimmed_end = start + (last + 1) * hop / rate
    if trimmed_end - trimmed_start < min_keep:
        return start, end
    return trimmed_start, trimmed_end


# --- vector maths -----------------------------------------------------------

def is_finite_vector(vector: list[float] | None) -> bool:
    """True when ``vector`` is a non-empty list of finite numbers.

    Neural embedders can return NaN/Inf for degenerate crops (e.g. a ~0.1s
    segment). A single such vector would otherwise poison a cluster centroid and
    silently disable all prior matching, so every embedding is screened here.
    """
    if not vector:
        return False
    for value in vector:
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            return False
    return True


def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    # NaN/Inf inputs (a bad embedder crop) would otherwise propagate a NaN
    # similarity all the way into the saved project / JSON.
    if not (math.isfinite(norm_a) and math.isfinite(norm_b)):
        return 0.0
    if norm_a < 1e-9 or norm_b < 1e-9:
        return 0.0
    result = dot / (norm_a * norm_b)
    return result if math.isfinite(result) else 0.0


def mean_vector(vectors: list[list[float]]) -> list[float]:
    # Drop empty *and* non-finite vectors: one NaN would turn the whole mean NaN.
    vectors = [v for v in vectors if is_finite_vector(v)]
    if not vectors:
        return []
    dimension = len(vectors[0])
    same = [v for v in vectors if len(v) == dimension]
    if not same:
        return []
    return [sum(v[i] for v in same) / len(same) for i in range(dimension)]


def distance(a: list[float], b: list[float]) -> float:
    return 1.0 - cosine(a, b)


# --- clustering -------------------------------------------------------------

def kmeans(vectors: list[list[float]], k: int, iterations: int = 40,
           seed: int = 7) -> tuple[list[int], float]:
    """Deterministic k-means++ over the given vectors. Returns (labels, inertia)."""
    if not vectors:
        return [], 0.0
    k = max(1, min(k, len(vectors)))
    # Deterministic k-means++ seeding.
    state = seed
    def rand() -> float:
        nonlocal state
        state = (1103515245 * state + 12345) % (2 ** 31)
        return state / (2 ** 31)

    centroids = [list(vectors[0])]
    while len(centroids) < k:
        best_index, best_score = 0, -1.0
        for index, vector in enumerate(vectors):
            nearest = min(distance(vector, c) for c in centroids)
            score = nearest * (0.75 + 0.5 * rand())
            if score > best_score:
                best_score, best_index = score, index
        centroids.append(list(vectors[best_index]))

    labels = [0] * len(vectors)
    for _ in range(iterations):
        changed = False
        for index, vector in enumerate(vectors):
            distances = [distance(vector, c) for c in centroids]
            label = distances.index(min(distances))
            if label != labels[index]:
                labels[index] = label
                changed = True
        for cluster in range(k):
            members = [vectors[i] for i in range(len(vectors)) if labels[i] == cluster]
            if members:
                dimension = len(members[0])
                centroids[cluster] = [
                    sum(m[i] for m in members) / len(members) for i in range(dimension)
                ]
        if not changed:
            break

    inertia = sum(
        distance(vectors[i], centroids[labels[i]]) ** 2 for i in range(len(vectors))
    )
    return labels, inertia


def choose_k(vectors: list[list[float]], min_k: int, max_k: int) -> int:
    """Pick k by the largest relative inertia drop (elbow), bounded by the caller."""
    if not vectors:
        return max(1, min_k)
    max_k = max(min_k, min(max_k, len(vectors)))
    if max_k <= min_k:
        return max(1, min_k)
    inertias = {}
    for k in range(min_k, max_k + 1):
        _, inertia = kmeans(vectors, k)
        inertias[k] = inertia
    best_k, best_gain = min_k, 0.0
    for k in range(min_k, max_k):
        previous, current = inertias[k], inertias[k + 1]
        if previous <= 1e-9:
            continue
        gain = (previous - current) / previous
        if gain > best_gain + 0.02:
            best_gain, best_k = gain, k + 1
    return best_k


def silhouette(vectors: list[list[float]], labels: list[int],
               sample: int = 400) -> float:
    """Mean silhouette of ``vectors`` under ``labels`` (cosine distance).

    Used by the speaker-count sweep to score a clustering objectively: values
    near 1 mean well-separated speakers, near 0 overlapping ones. Subsampled to
    ``sample`` points so it stays O(n^2) with a bounded n.
    """
    points = [(v, int(l)) for v, l in zip(vectors, labels) if v]
    if len(points) < 2:
        return 0.0
    if len(points) > sample:
        stride = len(points) / sample
        points = [points[int(i * stride)] for i in range(sample)]

    groups: dict[int, list[int]] = {}
    for index, (_, label) in enumerate(points):
        groups.setdefault(label, []).append(index)
    if len(groups) < 2:
        return 0.0

    total = 0.0
    for index, (vector, label) in enumerate(points):
        same = groups[label]
        if len(same) > 1:
            a = sum(1.0 - cosine(vector, points[j][0]) for j in same if j != index) / (len(same) - 1)
        else:
            a = 0.0
        b = None
        for other, indexes in groups.items():
            if other == label:
                continue
            mean = sum(1.0 - cosine(vector, points[j][0]) for j in indexes) / len(indexes)
            b = mean if b is None else min(b, mean)
        if not b:
            continue
        total += (b - a) / max(a, b)
    return total / len(points)
