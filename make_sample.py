"""Generate a synthetic two-speaker sample (WAV + MP4 + SRT) for testing.

Produces sample/interview.wav, sample/interview.mp4 and sample/interview.srt.
Two "voices" are synthesised with different fundamentals and formants so a
speaker-embedding diarizer has something separable to cluster.
"""

from __future__ import annotations

import math
import os
import struct
import subprocess
import wave

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "sample")
RATE = 16000

# (speaker, text, duration) — alternating speakers with small silent gaps.
SCRIPT = [
    ("A", "你好，欢迎来到这次的访谈节目。", 3.2),
    ("B", "谢谢，很高兴能在这里聊聊。", 2.8),
    ("A", "那我们直接进入第一个问题。", 2.6),
    ("B", "好的，这个问题我准备了很久。", 3.0),
    ("A", "你觉得数据集标注最难的是什么？", 3.1),
    ("B", "我认为是一致性和说话人边界的判断。", 3.4),
]

# Voice profiles: fundamental (Hz) and two formant centres (Hz).
VOICES = {
    "A": {"f0": 115.0, "formants": (620.0, 1180.0), "gain": 0.5},
    "B": {"f0": 205.0, "formants": (880.0, 1900.0), "gain": 0.45},
}
GAP = 0.35


def synth_voice(voice: dict, duration: float) -> list[float]:
    """Sum harmonics shaped by two formant resonances, plus vibrato."""
    count = int(duration * RATE)
    samples = []
    for index in range(count):
        t = index / RATE
        vibrato = 1.0 + 0.012 * math.sin(2 * math.pi * 4.6 * t)
        f0 = voice["f0"] * vibrato
        value = 0.0
        for harmonic in range(1, 41):
            freq = f0 * harmonic
            if freq > RATE / 2 - 200:
                break
            amplitude = 0.0
            for formant in voice["formants"]:
                bandwidth = 130.0
                amplitude += 1.0 / (1.0 + ((freq - formant) / bandwidth) ** 2)
            if harmonic % 2 == 0:
                amplitude *= 0.55
            value += amplitude * math.sin(2 * math.pi * freq * t + harmonic * 0.3)
        # Amplitude envelope and word-like amplitude modulation.
        envelope = min(1.0, t / 0.06, max(0.0, (duration - t) / 0.08))
        modulation = 0.72 + 0.28 * math.sin(2 * math.pi * 3.1 * t)
        peak = 1.0 / (1.0 + abs(value) * 0.2)
        samples.append(voice["gain"] * envelope * modulation * value * peak)
    return samples


def build_audio() -> tuple[list[float], list[tuple[str, float, float, str]]]:
    silence = [0.0] * int(GAP * RATE)
    signal: list[float] = []
    timeline = []
    for speaker, text, duration in SCRIPT:
        start = len(signal) / RATE
        signal.extend(synth_voice(VOICES[speaker], duration))
        end = len(signal) / RATE
        timeline.append((speaker, start, end, text))
        signal.extend(silence)
    ceiling = max((abs(v) for v in signal), default=1.0) or 1.0
    signal = [v / ceiling * 0.9 for v in signal]
    return signal, timeline


def write_wav(path: str, signal: list[float]) -> None:
    with wave.open(path, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(RATE)
        handle.writeframes(b"".join(
            struct.pack("<h", int(max(-1.0, min(1.0, v)) * 32767)) for v in signal
        ))


def fmt(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, milli = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{milli:03d}"


def write_srt(path: str, timeline) -> None:
    lines = []
    for index, (_, start, end, text) in enumerate(timeline, start=1):
        lines += [str(index), f"{fmt(start)} --> {fmt(end)}", text, ""]
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def make_mp4(wav_path: str, mp4_path: str, duration: float) -> bool:
    """Mux a plain colour video with the audio track."""
    command = [
        "ffmpeg", "-y", "-v", "error",
        "-f", "lavfi", "-i", f"color=c=0x1b2233:s=640x360:d={duration:.3f}:r=25",
        "-i", wav_path,
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
        "-shortest", mp4_path,
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=300)
        return result.returncode == 0
    except Exception:
        return False


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    signal, timeline = build_audio()
    duration = len(signal) / RATE

    wav_path = os.path.join(OUT, "interview.wav")
    srt_path = os.path.join(OUT, "interview.srt")
    mp4_path = os.path.join(OUT, "interview.mp4")

    write_wav(wav_path, signal)
    write_srt(srt_path, timeline)
    made_video = make_mp4(wav_path, mp4_path, duration)

    print(f"WAV : {wav_path}  ({duration:.2f}s)")
    print(f"SRT : {srt_path}  ({len(timeline)} 条)")
    print(f"MP4 : {mp4_path if made_video else '（生成失败，可用 WAV 测试）'}")
    print(f"媒体路径（供导入使用）：{wav_path}")
    print(f"字幕路径：{srt_path}")
    print("真实说话人顺序：", " ".join(speaker for speaker, *_ in timeline))


if __name__ == "__main__":
    main()
