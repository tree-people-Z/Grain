"""Optional ASR (transcription) — used only when the user has no subtitle file.

The main workflow is subtitle-first by design; this module exists because the
project plan allows "若无声曲，可先跑 ASR 生成字幕再进入主流程". The adapter is
faster-whisper (local, no cloud, no API key). Nothing here is imported unless
the user requests transcription, so the tool stays fully usable without it.
"""

from __future__ import annotations


def availability() -> dict:
    try:
        import faster_whisper  # noqa: F401

        return {
            "available": True,
            "label": "faster-whisper（本地转录）",
            "detail": "已安装。可选模型：tiny / base / small / medium / large-v3，越大越准越慢。",
            "models": ["tiny", "base", "small", "medium", "large-v3"],
        }
    except Exception as exc:
        short = str(exc).split("\n")[0][:120]
        return {
            "available": False,
            "label": "faster-whisper（本地转录）",
            "detail": f"未安装：{short}。可选接入：pip install faster-whisper（首次运行自动下载模型权重）。",
            "models": [],
        }


def transcribe(media_path: str, model_size: str = "small",
               language: str | None = None) -> tuple[list[dict], str]:
    """Run ASR on a media file. Returns (segments, detected_language)."""
    try:
        from faster_whisper import WhisperModel
    except Exception as exc:
        raise RuntimeError(
            "faster-whisper 未安装。可选接入：pip install faster-whisper。"
            f"原因：{str(exc)[:160]}"
        ) from exc

    model = WhisperModel(model_size, device="auto", compute_type="auto")
    raw_iter, info = model.transcribe(
        media_path,
        language=language or None,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 300},
    )
    segments = []
    for item in raw_iter:
        text = (item.text or "").strip()
        if not text:
            continue
        segments.append({
            "start": round(float(item.start), 3),
            "end": round(float(item.end), 3),
            "text": text,
        })
    if not segments:
        raise RuntimeError("ASR 没有识别到任何语音内容。")
    return segments, getattr(info, "language", "") or ""
