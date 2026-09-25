"""Speaker detection engines.

Two engines:
* ``pyannote`` — pyannote.audio 4 ``community-1`` (falls back to 3.1). Weights
  ship in ``models/`` so it loads offline; without bundled weights a HuggingFace
  token is needed for the gated models.
* ``manual``   — no auto detection; everything pending.

Every engine returns a common shape::

    {
      "turns":    [{"start": float, "end": float, "cluster": int}],
      "clusters": {int: {"duration": float, "frames": int}},
      "engine":   str,
      "notes":    [str],
    }
"""

from __future__ import annotations

import os
import threading

import audio_features as af
import apppaths
import media

# Guards the module-level caches below (signal / pipeline). The HTTP server is
# threaded and the engine warm-up thread probes concurrently, so an unguarded
# cache could run a model load twice or hand out a half-built entry.
_CACHE_LOCK = threading.RLock()

# Point HuggingFace at the package's bundled model cache (if any) before torch /
# pyannote are imported, so a portable build loads the weights offline.
apppaths.configure_model_caches()


# --- helpers ----------------------------------------------------------------

def _try_import(module: str) -> tuple[bool, str]:
    try:
        __import__(module)
        return True, ""
    except Exception as exc:
        short = str(exc).split("\n")[0][:160]
        return False, short


# --- availability -----------------------------------------------------------

_AVAILABILITY_CACHE: dict | None = None


def engine_availability(refresh: bool = False) -> dict[str, dict]:
    """Probe which engines can run here. Cached: the first probe imports torch
    and can take several seconds, and callers hit this on every page load."""
    global _AVAILABILITY_CACHE
    with _CACHE_LOCK:
        if _AVAILABILITY_CACHE is not None and not refresh:
            return _AVAILABILITY_CACHE
        # Probing imports torch (seconds); hold the lock so the warm-up thread
        # and a request thread cannot both run it.
        _AVAILABILITY_CACHE = _probe_engines()
        return _AVAILABILITY_CACHE


def warm_engines() -> None:
    """Probe engines ahead of the first request (called on a background thread)."""
    engine_availability()


def _pyannote_detail(ok: bool, err: str) -> str:
    if not ok:
        return f"未安装：{err}。安装：pip install torch pyannote.audio"
    if apppaths.bundled_hf_hub_cache():
        token_note = "已内置模型，无需 HuggingFace token，离线可用"
    elif _pyannote_token():
        token_note = "已检测到 HuggingFace token"
    else:
        token_note = "未检测到 token（设置环境变量 HF_TOKEN）"
    try:
        import torch

        device_note = (f"GPU：{torch.cuda.get_device_name(0)}"
                       if torch.cuda.is_available() else "CPU")
    except Exception:
        device_note = "设备状态未知"
    return (
        "默认用 pyannote.audio 4 的 community-1（失败回退 3.1），"
        "并采用更适合字幕归属的 exclusive 分段。"
        f"{token_note}。当前设备：{device_note}。"
    )


def _probe_engines() -> dict[str, dict]:
    ok_pyannote, pyannote_err = _try_import("pyannote.audio")
    return {
        "pyannote": {
            "available": ok_pyannote,
            "label": f"pyannote.audio {_pyannote_version()}" if ok_pyannote else "pyannote.audio",
            "detail": _pyannote_detail(ok_pyannote, pyannote_err),
        },
        "manual": {
            "available": True,
            "label": "纯手动（不自动检测）",
            "detail": "全部标为待定，完全由人工归属",
        },
    }


def engine_available(engine: str) -> bool:
    return bool(engine_availability().get(engine, {}).get("available"))


# --- shared audio cache -----------------------------------------------------

_SIGNAL_CACHE: dict[str, tuple[float, object, int]] = {}
_SIGNAL_CACHE_MAX = 1  # a mono 16 kHz hour is ~230MB; keep only the current file


def _load_signal(wav_path: str):
    """Decode a WAV once per (path, mtime); reused across a detection run."""
    try:
        mtime = os.path.getmtime(wav_path)
    except OSError:
        mtime = 0.0
    with _CACHE_LOCK:
        hit = _SIGNAL_CACHE.get(wav_path)
        if hit is not None and hit[0] == mtime:
            return hit[1], hit[2]
    signal, rate = af.read_wav(wav_path, media.AUDIO_SAMPLE_RATE)
    with _CACHE_LOCK:
        _SIGNAL_CACHE[wav_path] = (mtime, signal, rate)
        while len(_SIGNAL_CACHE) > _SIGNAL_CACHE_MAX:
            _SIGNAL_CACHE.pop(next(iter(_SIGNAL_CACHE)))
    return signal, rate


# --- manual -----------------------------------------------------------------

def run_manual() -> dict:
    return {"turns": [], "clusters": {}, "engine": "manual",
            "notes": ["未运行自动检测，所有字幕保持“待定”。"]}


# --- HuggingFace network / token --------------------------------------------

def _pyannote_token() -> str | None:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    return token.strip() if token else None


def _system_proxy() -> str | None:
    """Read the Windows system proxy (HKCU Internet Settings)."""
    if os.name != "nt":
        return None
    try:
        import winreg

        with winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r"Software\Microsoft\Windows\CurrentVersion\Internet Settings",
        ) as key:
            enabled = winreg.QueryValueEx(key, "ProxyEnable")[0]
            server = winreg.QueryValueEx(key, "ProxyServer")[0]
        if not enabled or not server:
            return None
        if "=" in server:  # "http=h:p;https=h:p" form
            parts = dict(piece.split("=", 1) for piece in server.split(";") if "=" in piece)
            return parts.get("https") or parts.get("http")
        return server
    except Exception:
        return None


def _ensure_hf_network() -> None:
    """Configure proxy + HuggingFace endpoint once, at import.

    A user-set HF_ENDPOINT/HTTP(S)_PROXY always wins.
    """
    proxy = (os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
             or _system_proxy())
    if proxy:
        url = proxy if "://" in proxy else "http://" + proxy
        os.environ.setdefault("HTTP_PROXY", url)
        os.environ.setdefault("HTTPS_PROXY", url)
        os.environ.setdefault("no_proxy", "127.0.0.1,localhost,::1")
        os.environ.setdefault("NO_PROXY", "127.0.0.1,localhost,::1")
    if not os.environ.get("HF_ENDPOINT"):
        os.environ["HF_ENDPOINT"] = (
            "https://huggingface.co" if proxy else "https://hf-mirror.com"
        )


_ensure_hf_network()


# --- pyannote engine --------------------------------------------------------

PYANNOTE_PIPELINE_PRIMARY = "pyannote/speaker-diarization-community-1"
PYANNOTE_PIPELINE_FALLBACK = "pyannote/speaker-diarization-3.1"
PYANNOTE_GATED_REPOS = ("pyannote/speaker-diarization-community-1",
                        "pyannote/segmentation-3.0", "pyannote/embedding")
PYANNOTE_TOKEN_URL = "https://hf.co/settings/tokens"
_PYANNOTE_CACHE: dict = {}
_EMBEDDING_CACHE: dict = {}


def _pyannote_version() -> str:
    try:
        import pyannote.audio as pa

        return getattr(pa, "__version__", "?")
    except Exception:
        return "?"


def _missing_gate(error: str) -> str | None:
    """Which gated repo the error names, if any."""
    if "403" not in error and "gated" not in error.lower():
        return None
    for repo in PYANNOTE_GATED_REPOS:
        if repo in error:
            return repo
    return None


def _pyannote_gate_report() -> list[str]:
    """Probe which of the pipeline's model repos this token may read."""
    try:
        from huggingface_hub import hf_hub_download

        from huggingface_hub.utils import GatedRepoError
    except Exception:
        return []
    token = _pyannote_token()
    lines: list[str] = []
    for repo in PYANNOTE_GATED_REPOS:
        try:
            hf_hub_download(repo, "config.yaml", token=token)
        except GatedRepoError:
            lines.append(repo)
        except Exception:
            continue
    return lines


def _pyannote_load_pipeline():
    """Load community-1 (preferred) or 3.1, cached across detections."""
    with _CACHE_LOCK:
        if _PYANNOTE_CACHE.get("pipeline") is not None:
            return _PYANNOTE_CACHE["pipeline"], _PYANNOTE_CACHE["name"]

    from pyannote.audio import Pipeline

    token = _pyannote_token()
    errors = []
    for name in (PYANNOTE_PIPELINE_PRIMARY, PYANNOTE_PIPELINE_FALLBACK):
        try:
            # pyannote >= 4 renamed use_auth_token -> token.
            try:
                pipeline = Pipeline.from_pretrained(name, token=token)
            except TypeError:
                pipeline = Pipeline.from_pretrained(name, use_auth_token=token)
        except Exception as exc:
            errors.append(f"{name}: {exc}")
            continue
        import torch

        device = "CPU"
        if torch.cuda.is_available():
            try:
                pipeline.to(torch.device("cuda"))
            except Exception as exc:
                # CUDA can be visible while the driver, VRAM, or a model
                # operator still fails during initialization. Keep the app
                # usable by moving the already loaded pipeline back to CPU.
                try:
                    torch.cuda.empty_cache()
                    pipeline.to(torch.device("cpu"))
                except Exception as cpu_exc:
                    raise RuntimeError(
                        f"GPU 初始化失败且无法回退 CPU：{exc}; {cpu_exc}"
                    ) from exc
                device = f"CPU（GPU 初始化失败，已回退）"
            else:
                device = f"GPU：{torch.cuda.get_device_name(0)}"
        with _CACHE_LOCK:
            _PYANNOTE_CACHE.update(pipeline=pipeline, name=name, device=device)
        return pipeline, name
    raise RuntimeError(" | ".join(errors) or "pyannote pipeline 加载失败")


def _pyannote_pick_annotation(output):
    """Return ``(annotation, used_exclusive)`` for a pyannote 4 DiarizeOutput."""
    exclusive = getattr(output, "exclusive_speaker_diarization", None)
    regular = getattr(output, "speaker_diarization", None)
    if exclusive is not None:
        return exclusive, True
    if regular is not None:
        return regular, False
    return output, False


def _pyannote_audio_input(wav_path: str, signal=None):
    """Return pyannote's in-memory audio Mapping for a WAV.

    pyannote >= 4 reads files through torchcodec, which needs shared FFmpeg
    libraries; a static ffmpeg.exe build has none, so loading by path fails.
    Handing the pipeline a ``{"waveform": tensor, "sample_rate": int}`` mapping
    skips that decoder — we already decode with ffmpeg ourselves.
    """
    import numpy as np
    import torch

    if signal is None:
        signal, rate = _load_signal(wav_path)
    else:
        rate = media.AUDIO_SAMPLE_RATE
    waveform = torch.from_numpy(np.asarray(signal, dtype="float32")).unsqueeze(0)
    return {"waveform": waveform, "sample_rate": rate}, waveform, rate


def run_pyannote(wav_path: str, min_speakers: int, max_speakers: int) -> dict:
    try:
        pipeline, pipeline_name = _pyannote_load_pipeline()
    except Exception as exc:
        raise RuntimeError(_pyannote_failure_message(str(exc))) from exc

    signal, rate = _load_signal(wav_path)
    audio, waveform, rate = _pyannote_audio_input(wav_path, signal)
    try:
        output = pipeline(
            audio,
            min_speakers=min_speakers or None,
            max_speakers=max_speakers or None,
        )
    except TypeError:
        # Older pyannote builds only accept the min/max kwargs positionally.
        output = pipeline(audio)
    diarization, used_exclusive = _pyannote_pick_annotation(output)
    label_map: dict[str, int] = {}
    turns = []
    for segment, _, label in diarization.itertracks(yield_label=True):
        cluster = label_map.setdefault(label, len(label_map))
        turns.append({"start": round(segment.start, 3),
                      "end": round(segment.end, 3), "cluster": cluster})
    notes = [f"使用 {pipeline_name}（pyannote.audio {_pyannote_version()}）。"]
    notes.append(f"检测设备：{_PYANNOTE_CACHE.get('device', 'CPU')}。")
    if used_exclusive:
        notes.append("已采用 exclusive 分段（同一时刻只保留最可能的一个说话人），"
                     "更适合逐条字幕归属。")
    clusters: dict[int, dict] = {}
    for turn in turns:
        entry = clusters.setdefault(int(turn["cluster"]),
                                   {"duration": 0.0, "frames": 0})
        entry["duration"] += turn["end"] - turn["start"]
        entry["frames"] += 1
    for entry in clusters.values():
        entry["duration"] = round(entry["duration"], 3)
    return {"turns": turns, "clusters": clusters, "engine": "pyannote", "notes": notes}


def extract_voice_embedding(wav_path: str, start: float, end: float) -> list[float]:
    """Extract one normalized speaker embedding from a clean speech crop.

    The bundled CAM++ model is used instead of the gated pyannote embedding
    checkpoint; it is present in the portable package and runs on CUDA.
    """
    import numpy as np

    if end - start < 0.8:
        raise ValueError("声纹样本至少需要 0.8 秒")
    with _CACHE_LOCK:
        inference = _EMBEDDING_CACHE.get("inference")
    if inference is None:
        from funasr import AutoModel
        model_dir = os.path.join(
            apppaths.writable_dir(), "models", "modelscope", "models",
            "damo--speech_campplus_sv_zh-cn_16k-common", "snapshots", "master",
        )
        if not os.path.isdir(model_dir):
            raise RuntimeError("未找到内置 CAM++ 声纹模型")
        import torch
        inference = AutoModel(model=model_dir,
                              device="cuda" if torch.cuda.is_available() else "cpu",
                              disable_update=True)
        with _CACHE_LOCK:
            _EMBEDDING_CACHE["inference"] = inference
    signal, _rate = _load_signal(wav_path)
    sample_start = max(0, int(start * _rate))
    sample_end = min(len(signal), int(end * _rate))
    audio = np.asarray(signal[sample_start:sample_end], dtype="float32")
    result = inference.inference(audio, input_len=[len(audio)], key=["voice"])
    vector = np.asarray(result[0]["spk_embedding"].detach().cpu(), dtype="float32")
    vector = vector.reshape(-1)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-8:
        raise ValueError("声纹样本没有有效声音")
    return (vector / norm).round(7).tolist()


def _pyannote_failure_message(error: str) -> str:
    """Turn a pyannote load failure into the one action that actually unblocks it."""
    endpoint = os.environ.get("HF_ENDPOINT") or ""
    using_mirror = "hf-mirror" in endpoint
    missing = _missing_gate(error)
    gated = _pyannote_gate_report()
    if missing and missing not in gated:
        gated.insert(0, missing)
    if gated:
        links = "\n".join(f"  https://hf.co/{repo}" for repo in gated)
        mirror_note = (
            "\n注意：当前用的是 hf-mirror.com 镜像，而镜像无法下载门控模型。"
            "pyannote 门控模型必须直连 huggingface.co——请开代理/VPN。"
            if using_mirror else ""
        )
        return (
            "pyannote 检测管线加载失败：有模型是门控（gated）的，当前 HuggingFace 账号尚未接受条款。\n"
            "请用浏览器逐个打开下面的页面，点一次「Agree / 接受」，然后用同一个 token 重试：\n"
            f"{links}\n"
            f"（token：{PYANNOTE_TOKEN_URL}；本机已配置{'✓' if _pyannote_token() else '✗'}。）"
            f"{mirror_note}\n"
            f"原始错误：{error[:300]}"
        )
    return (
        "加载 pyannote 管线失败（网络或依赖问题）。请确认已 pip install pyannote.audio，"
        "并配置 HF token（环境变量 HF_TOKEN）。"
        f"当前 HF_ENDPOINT={endpoint or 'huggingface.co'}；"
        "原始错误：" + error[:300]
    )


# --- dispatch ---------------------------------------------------------------

def run(engine: str, wav_path: str, min_speakers: int, max_speakers: int) -> dict:
    if engine == "manual":
        return run_manual()
    if engine == "pyannote":
        return run_pyannote(wav_path, min_speakers, max_speakers)
    raise RuntimeError(f"未知的检测引擎：{engine}")
