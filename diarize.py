"""Speaker detection engines (pluggable, switchable).

Every engine returns a common shape::

    {
      "turns":    [{"start": float, "end": float, "cluster": int, "embedding": [...]}],
      "clusters": {int: {"embedding": [...], "duration": float, "frames": int}},
      "engine":   str,
      "notes":    [str],
    }

Engines
-------
* ``campp``         — 3D-Speaker / FunASR CAM++ diarization via ModelScope (no HF token)
* ``pyannote``      — pyannote.audio 4 community-1 (falls back to 3.1; needs a token)
* ``voiceprint-cue``— per-subtitle-cue wespeaker embeddings + clustering (no gate)
* ``sortformer``    — NVIDIA Sortformer v2.1 via an isolated env (external engine)
* ``manual``        — no auto detection; everything pending

Embedder binding (hard constraint): the voiceprint extractor is bound to the
detection engine — voiceprints enrolled with one model must never be cosine-
matched against clusters from another model. :func:`embedder_for` maps each
engine to an embedder id, and the role library filters by that id.
"""

from __future__ import annotations

import os
import threading

import audio_features as af
import apppaths
import external_engines
import media

# Guards the module-level caches below (signal / pipeline / embedder). The HTTP
# server is threaded and the engine warm-up thread probes concurrently, so an
# unguarded cache could run a model load twice or hand out a half-built entry.
_CACHE_LOCK = threading.RLock()

# Point HuggingFace / ModelScope at the package's bundled model cache (if any)
# before torch / pyannote / funasr are imported, so a portable build loads the
# weights offline instead of reaching for the user's ~/.cache or the network.
apppaths.configure_model_caches()

MIN_TURN_SECONDS = 0.30

ENGINE_ALIASES = {"3dspeaker": "campp", "funasr": "campp", "nemo": "nemo",
                  "wespeaker": "voiceprint-cue"}

# Engines that have no embedder of their own and are enriched with the active
# voiceprint model (below): pyannote's turns, the cue/voiceprint engine, and the
# external engines all share one feature space so their voiceprints interchange.
_VOICEPRINT_ENGINES = {"pyannote", "voiceprint-cue", "sortformer", "diarizen"}

# Selectable speaker-embedding models for the pyannote family. Each profile is a
# *distinct feature space*: vectors from different models are never compared, so
# a role enrolled under one shows "需重录" under another. "wespeaker" keeps the
# historical space id "pyannote" so existing projects/roles stay valid.
VOICEPRINT_PROFILES: dict[str, dict] = {
    "wespeaker": {
        "label": "pyannote wespeaker（默认）",
        "space": "pyannote",
        "kind": "pyannote",
        "repos": ("pyannote/wespeaker-voxceleb-resnet34-LM", "pyannote/embedding"),
        "detail": "pyannote 官方管线的声纹模型，通用稳妥；与 community-1 聚类同源。",
    },
    "eres2netv2": {
        "label": "3D-Speaker ERes2NetV2（短句更强）",
        "space": "eres2netv2",
        "kind": "funasr",
        "model": "iic/speech_eres2netv2_sv_zh-cn_16k-common",
        "detail": ("阿里 3D-Speaker 的 ERes2NetV2，短字幕、噪声下比 CAM++/ECAPA 更稳；"
                   "模型从 ModelScope 下载，无需 token。与默认 wespeaker 特征空间不通用，"
                   "切换后需用新模型重新录入声纹。"),
    },
}

_VOICEPRINT_MODEL = "wespeaker"


def set_voiceprint_model(key: str) -> None:
    """Select the active speaker-embedding profile (affects feature-space tags)."""
    global _VOICEPRINT_MODEL
    key = key if key in VOICEPRINT_PROFILES else "wespeaker"
    if key == _VOICEPRINT_MODEL:
        return
    _VOICEPRINT_MODEL = key
    # Cached embedders and the client-facing space map are now stale.
    global _AVAILABILITY_CACHE
    _AVAILABILITY_CACHE = None
    with _CACHE_LOCK:
        _PYANNOTE_EMBEDDER_CACHE.clear()


def active_voiceprint_model() -> str:
    return _VOICEPRINT_MODEL


def voiceprint_profiles() -> list[dict]:
    """Profiles for the settings UI, each tagged with availability."""
    out = []
    for key, meta in VOICEPRINT_PROFILES.items():
        if meta["kind"] == "funasr":
            available = _try_import("funasr")[0] and _try_import("modelscope")[0]
        elif meta["kind"] == "pyannote":
            available = _try_import("pyannote.audio")[0]
        else:
            available = True
        out.append({"key": key, "label": meta["label"], "space": meta["space"],
                    "detail": meta.get("detail", ""), "available": available})
    return out


def embedder_for(engine: str) -> str:
    """Which voiceprint feature space an engine's clusters live in."""
    engine = ENGINE_ALIASES.get(engine, engine)
    if engine in _VOICEPRINT_ENGINES:
        return VOICEPRINT_PROFILES[_VOICEPRINT_MODEL]["space"]
    if engine == "campp":
        return "campp"
    return "builtin"


# --- availability -----------------------------------------------------------

def _try_import(module: str) -> tuple[bool, str]:
    try:
        __import__(module)
        return True, ""
    except Exception as exc:
        short = str(exc).split("\n")[0][:160]
        return False, short


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
        # The gated weights ship with the package, so no token is required.
        token_note = "已内置模型，无需 HuggingFace token"
    elif _pyannote_token():
        token_note = "已检测到 HuggingFace token"
    else:
        token_note = "未检测到 token：放到环境变量 HF_TOKEN，或写入 data/hf_token.txt"
    gated = "、".join(PYANNOTE_GATED_REPOS)
    return (
        "精度最高的通用方案，默认用 pyannote.audio 4 的 community-1（失败回退 3.1），"
        "并采用更适合字幕归属的 exclusive 分段。"
        f"{token_note}；模型是门控（gated）的，必须先用该账号在 HF 网站接受条款：{gated}。"
        "若加载报 403，运行检测时会直接列出还缺哪个仓库的授权链接。"
    )


def _probe_engines() -> dict[str, dict]:
    ok_pyannote, pyannote_err = _try_import("pyannote.audio")
    ms_ok, ms_err = _try_import("modelscope")
    fa_ok, fa_err = _try_import("funasr")
    campp_ok = ms_ok and fa_ok and _try_import("torch")[0]
    available: dict[str, dict] = {}

    # Professional engines first (recommended); built-in ones are the fallback.
    available["campp"] = {
        "available": campp_ok,
        "label": "3D-Speaker / FunASR CAM++",
        "detail": ("中文场景表现好，模型从 ModelScope 自动下载，无需 token。" if campp_ok
                   else f"未安装：modelscope({ms_err or '缺失'}) / funasr({fa_err or '缺失'}) / torch。"
                        "安装：pip install torch torchaudio funasr modelscope addict simplejson datasets hdbscan"),
        "family": "campp",
    }
    available["pyannote"] = {
        "available": ok_pyannote,
        "label": f"pyannote.audio {_pyannote_version()}" if ok_pyannote else "pyannote.audio",
        "detail": _pyannote_detail(ok_pyannote, pyannote_err),
        "family": "pyannote",
    }
    available["voiceprint-cue"] = {
        "available": ok_pyannote,
        "label": "字幕级声纹聚类（免门控）",
        "detail": ("对每条字幕的音频单独提取 pyannote(wespeaker) 声纹再聚类，无需门控授权。"
                   "字幕时间轴准确时，通常比“VAD 时间段 + 重叠对齐”更贴合“一句一个说话人”，"
                   "对同音色、串场、重叠语音更稳；与官方管线共用同一特征空间，声纹可复用。"
                   if ok_pyannote else
                   f"未安装：{pyannote_err}。安装：pip install torch pyannote.audio"),
        "family": "pyannote",
    }
    available["manual"] = {
        "available": True, "label": "纯手动（不自动检测）",
        "detail": "全部标为待定，完全由人工归属", "family": "builtin",
    }
    for key in external_engines.ENGINE_SPECS:
        meta = external_engines.spec(key)
        installed = external_engines.available(key)
        available[key] = {
            "available": installed,
            "label": meta.get("label", key),
            "detail": (meta.get("detail", "") if installed
                       else f"未安装。安装：{meta.get('install', '见 engines/README.md')}"),
            "family": "external",
        }
    # Voiceprints only match within a feature space, so expose each engine's
    # space to the client for the "this role needs re-enrolment" check.
    for key, info in available.items():
        info["space"] = embedder_for(key)
    return available


# --- shared helpers ---------------------------------------------------------

_SIGNAL_CACHE: dict[str, tuple[float, object, int]] = {}
_SIGNAL_CACHE_MAX = 2


def _load_signal(wav_path: str):
    """Decode a WAV once per (path, mtime); reused across a detection run so
    pyannote, the voiceprint extractor and the CAM++ slicer never re-decode it.

    Bounded to a couple of entries: each is ~230 MB/hour, so an unbounded cache
    would leak memory as the user switches projects.
    """
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


def _write_wav_slice(signal, rate: int, start: float, end: float, path: str) -> None:
    """Write ``signal[start:end]`` as a mono 16-bit PCM WAV using stdlib only.

    Replaces a per-turn ``ffmpeg`` subprocess: CAM++ embeds one span at a time,
    and process spawn + full-file decode per subtitle cue dominated its wall
    clock. Slicing the already-decoded signal in memory is effectively free.
    """
    import array
    import wave

    begin = max(0, int(start * rate))
    finish = min(len(signal), int(end * rate))
    frames = array.array("h")
    append = frames.append
    for value in signal[begin:finish]:
        sample = value * 32767.0
        if sample > 32767.0:
            sample = 32767.0
        elif sample < -32768.0:
            sample = -32768.0
        append(int(sample))
    with wave.open(path, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(frames.tobytes())


def _cluster_turns(turns: list[dict], min_speakers: int, max_speakers: int,
                   notes: list[str]) -> list[dict]:
    """Attach ``cluster`` labels to turns, reusing turn embeddings."""
    candidates = [t for t in turns if t.get("embedding")]
    if not candidates:
        for turn in turns:
            turn["cluster"] = 0
        notes.append("没有提取到有效的声学特征，所有片段归为同一说话人。")
        return turns

    vectors = [t["embedding"] for t in candidates]
    if max_speakers <= min_speakers:
        k = max(1, min_speakers)
        notes.append(f"说话人数固定为 {k}。")
    else:
        k = af.choose_k(vectors, max(1, min_speakers), max_speakers)
        notes.append(f"自动估计说话人数为 {k}（搜索范围 {min_speakers}–{max_speakers}）。")

    labels, _ = af.kmeans(vectors, k)
    for turn, label in zip(candidates, labels):
        turn["cluster"] = int(label)
    for turn in turns:
        turn.setdefault("cluster", 0)
    return turns


def _summarise(turns: list[dict], engine: str, notes: list[str]) -> dict:
    clusters: dict[int, dict] = {}
    for turn in turns:
        cluster = int(turn["cluster"])
        entry = clusters.setdefault(
            cluster, {"embedding": [], "duration": 0.0, "frames": 0, "_vectors": []}
        )
        entry["duration"] += turn["end"] - turn["start"]
        entry["frames"] += 1
        if turn.get("embedding"):
            entry["_vectors"].append(turn["embedding"])
    for entry in clusters.values():
        entry["embedding"] = af.mean_vector(entry.pop("_vectors"))
        entry["duration"] = round(entry["duration"], 3)
    return {"turns": turns, "clusters": clusters, "engine": engine, "notes": notes}


# --- manual ----------------------------------------------------------------

def run_manual() -> dict:
    return {"turns": [], "clusters": {}, "engine": "manual",
            "notes": ["未运行自动检测，所有字幕保持“待定”。"]}


def _voiceprint_cue_turns(wav_path: str, spans: list[tuple[float, float]],
                          notes: list[str]) -> list[dict]:
    """Embed every cue's own audio (unclustered). The expensive half of the
    voiceprint-cue engine, kept separate so a speaker-count sweep can cluster
    these vectors repeatedly without re-extracting them."""
    signal, rate = _load_signal(wav_path)
    if not signal:
        raise RuntimeError("音频为空，无法进行说话人检测。")
    # Use the active voiceprint profile so the cue engine's space matches
    # embedder_for("voiceprint-cue") — otherwise clusters and enrolled voiceprints
    # would live in different models but be tagged as the same space.
    embedder = make_embedder("voiceprint-cue", wav_path, signal)
    if embedder is None:
        raise RuntimeError(
            "声纹模型加载失败（检查已安装 torch + pyannote.audio / funasr，"
            "或改用默认 wespeaker 声纹模型）。"
        )
    turns = []
    short = 0
    for start, end in spans or []:
        if end - start < 0.20:
            short += 1
            continue
        vector = embedder(start, end)
        if vector:
            turns.append({"start": round(start, 3), "end": round(end, 3),
                          "embedding": vector})
    if short:
        notes.append(f"{short} 条字幕过短（<0.2s），已跳过声纹提取。")
    return turns


def run_voiceprint_cue(wav_path: str, spans: list[tuple[float, float]],
                       min_speakers: int, max_speakers: int) -> dict:
    """Cluster each subtitle cue's own audio with a neural voiceprint.

    Subtitles already give accurate boundaries; embedding each cue (instead of
    VAD regions that merely overlap the cue) removes most of the boundary and
    overlap error that dominates diarization. Uses the ungated pyannote
    wespeaker embedder, so it runs without accepting any gated licence.
    """
    notes: list[str] = []
    turns = _voiceprint_cue_turns(wav_path, spans, notes)
    turns = _cluster_turns(turns, min_speakers, max_speakers, notes)
    notes.append("对每条字幕提 pyannote(wespeaker) 声纹后聚类（字幕级、免门控）。")
    return _summarise(turns, "voiceprint-cue", notes)


# --- pyannote engine --------------------------------------------------------

def device_info() -> dict:
    """Report the best available compute device for the neural engines."""
    info = {"cuda": False, "device": "cpu", "name": "", "torch": None}
    try:
        import torch

        info["torch"] = torch.__version__
        if torch.cuda.is_available():
            info["cuda"] = True
            info["device"] = "cuda"
            try:
                info["name"] = torch.cuda.get_device_name(0)
                info["vram_gb"] = round(
                    torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 1
                )
            except Exception:
                pass
    except Exception:
        pass
    return info


def _funasr_device() -> str:
    """funasr picks 'cuda' when available; be explicit so we fail loudly if not."""
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


# NOTE: the hf-mirror.com default is applied in _ensure_hf_endpoint() below,
# *after* the token helper is defined — the mirror cannot serve gated repos, so
# we only fall back to it when there is no token to authenticate with.


def _pyannote_token() -> str | None:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if token:
        return token.strip()
    # Desktop/double-click launches have no env vars: read the local token file.
    # Honour SSP_DATA_DIR so a test run reads its own isolated directory.
    data_dir = apppaths.data_dir()
    token_file = os.path.join(data_dir, "hf_token.txt")
    try:
        with open(token_file, "r", encoding="utf-8-sig") as handle:
            token = handle.read().strip()
        return token or None
    except OSError:
        return None


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

    - A system (IE/registry) proxy is exported as HTTP(S)_PROXY so Python and the
      external-engine subprocesses use it; huggingface.co is then reachable and
      the gated pyannote models work.
    - Without a proxy, default to hf-mirror.com (the mirror cannot serve gated
      repos, so gated models still need a proxy/VPN).
    - A user-set HF_ENDPOINT/HTTP(S)_PROXY always wins.
    """
    proxy = (os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
             or _system_proxy())
    if proxy:
        url = proxy if "://" in proxy else "http://" + proxy
        os.environ.setdefault("HTTP_PROXY", url)
        os.environ.setdefault("HTTPS_PROXY", url)
        os.environ.setdefault("no_proxy", "127.0.0.1,localhost,::1,modelscope.cn,aliyuncs.com")
        os.environ.setdefault("NO_PROXY", "127.0.0.1,localhost,::1,modelscope.cn,aliyuncs.com")
    if not os.environ.get("HF_ENDPOINT"):
        os.environ["HF_ENDPOINT"] = (
            "https://huggingface.co" if proxy else "https://hf-mirror.com"
        )


_ensure_hf_network()


# pyannote.audio 4 ships "community-1", which fixes the speaker counting and
# assignment problems of the 3.1 recipe and adds an *exclusive* diarization
# (single speaker per instant) meant for reconciling with transcription /
# subtitle timestamps. We prefer it and fall back to 3.1 when it is unavailable.
PYANNOTE_PIPELINE_PRIMARY = "pyannote/speaker-diarization-community-1"
PYANNOTE_PIPELINE_FALLBACK = "pyannote/speaker-diarization-3.1"
PYANNOTE_PIPELINE = PYANNOTE_PIPELINE_PRIMARY  # kept for messages/labels
# community-1 is a single gated repo; the 3.1 recipe pulls the two below.
PYANNOTE_GATED_REPOS = ("pyannote/speaker-diarization-community-1",
                        "pyannote/segmentation-3.0", "pyannote/embedding")
PYANNOTE_TOKEN_URL = "https://hf.co/settings/tokens"
_PYANNOTE_CACHE: dict = {}
# Voiceprint extractors are cheap to reuse within a run (the model is the heavy
# part), so cache them per audio file.
_PYANNOTE_EMBEDDER_CACHE: dict = {}


def _pyannote_version() -> str:
    try:
        import pyannote.audio as pa

        return getattr(pa, "__version__", "?")
    except Exception:
        return "?"


def _missing_gate(error: str) -> str | None:
    """Which gated repo the error names, if any. pyannote/huggingface_hub wrap
    the 403 in prose, so match on the repo path rather than on the exception
    type (which moved between hub releases)."""
    if "403" not in error and "gated" not in error.lower():
        return None
    for repo in PYANNOTE_GATED_REPOS:
        if repo in error:
            return repo
    return None


def _pyannote_gate_report() -> list[str]:
    """Probe which of the pipeline's model repos this token may read.

    Only used to enrich a failure message, so it stays offline-safe: an
    unreachable hub just yields no extra detail rather than a second error.
    """
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
    """Load community-1 (preferred) or 3.1, cached across detections.

    Returns ``(pipeline, repo_name)``.
    """
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
        # Move to GPU when available; pyannote defaults to CPU otherwise.
        try:
            import torch

            if torch.cuda.is_available():
                pipeline.to(torch.device("cuda"))
        except Exception:
            pass
        with _CACHE_LOCK:
            _PYANNOTE_CACHE.update(pipeline=pipeline, name=name)
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


def _pyannote_audio_input(wav_path: str, signal: list[float] | None = None):
    """Return pyannote's in-memory audio Mapping for a WAV.

    pyannote >= 4 reads files through torchcodec, which needs shared FFmpeg
    libraries; a static ffmpeg.exe build (the usual Windows install) has none,
    so loading by path fails with "Could not load libtorchcodec". Handing the
    pipeline an ``{"waveform": tensor, "sample_rate": int}`` mapping skips that
    decoder entirely — we already decode with ffmpeg ourselves.
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
    if used_exclusive:
        notes.append("已采用 exclusive 分段（同一时刻只保留最可能的一个说话人），"
                     "更适合逐条字幕归属。")
    embedder = make_embedder("pyannote", wav_path, signal)
    if embedder is None:
        notes.append("声纹提取模型不可用，聚类不做声纹先验匹配（可直接人工归属）。")
    for turn in turns:
        turn["embedding"] = embedder(turn["start"], turn["end"]) if embedder else []
    return _summarise(turns, "pyannote", notes)


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
            "\n注意：当前用的是 hf-mirror.com 镜像，而**镜像无法下载门控模型**（即使已接受条款也报 403）。"
            "pyannote 门控模型必须直连 huggingface.co——请开代理/VPN，或改用「3D-Speaker / CAM++」"
            "（走 ModelScope，国内可直连）。"
            if using_mirror else ""
        )
        return (
            "pyannote 检测管线加载失败：有模型是门控（gated）的，当前 HuggingFace 账号尚未接受条款。\n"
            "请用浏览器逐个打开下面的页面，点一次「Agree / 接受」，然后用同一个 token 重试：\n"
            f"{links}\n"
            f"（token：{PYANNOTE_TOKEN_URL}；本机已配置{'✓' if _pyannote_token() else '✗'}。"
            "本工具默认用 community-1（pyannote.audio 4），失败会自动回退 3.1。）"
            f"{mirror_note}\n"
            f"原始错误：{error[:300]}"
        )
    return (
        "加载 pyannote 管线失败（网络或依赖问题）。请确认已 pip install pyannote.audio，"
        "并配置 HF token（HF_TOKEN 环境变量或 data/hf_token.txt）。"
        f"当前 HF_ENDPOINT={endpoint or 'huggingface.co'}；"
        "若无法直连 huggingface.co，可设置 HF_ENDPOINT=https://hf-mirror.com（但镜像不支持门控模型）。"
        "原始错误：" + error[:300]
    )



# --- 3D-Speaker / FunASR (CAM++) engine -------------------------------------

def _run_campp_modelscope(wav_path: str) -> list[dict]:
    from modelscope.pipelines import pipeline
    from modelscope.utils.constant import Tasks

    pipe = pipeline(
        task=Tasks.speaker_diarization,
        model="iic/speech_campplus_speaker-diarization_common",
        device=_funasr_device(),
    )
    result = pipe(wav_path)
    rows = []
    if isinstance(result, dict):
        result = result.get("text") or result.get("result") or result.get("segments") or []
    for row in result or []:
        # Accepted shapes: [start, end, label] or [label, start, end]
        try:
            if isinstance(row, dict):
                start, end = float(row["start"]), float(row["end"])
                label = row.get("spk", row.get("speaker", row.get("label", 0)))
            elif isinstance(row, (list, tuple)) and len(row) >= 3:
                a, b, c = row[0], row[1], row[2]
                if isinstance(a, str) or (
                    isinstance(a, int) and not isinstance(a, bool) and float(b) >= 0
                ):
                    label, start, end = a, float(b), float(c)
                else:
                    start, end, label = float(a), float(b), c
            else:
                continue
        except (TypeError, ValueError, IndexError, KeyError):
            continue
        rows.append({"start": round(float(start), 3), "end": round(float(end), 3),
                     "label": str(label)})
    return rows


def _run_campp_funasr(wav_path: str) -> list[dict]:
    from funasr import AutoModel

    model = AutoModel(model="iic/speech_campplus_speaker-diarization_common",
                      device=_funasr_device())
    result = model.generate(input=wav_path)
    rows = []
    for item in result or []:
        for row in item.get("spk", []) or []:
            rows.append({"start": round(float(row[0]), 3),
                         "end": round(float(row[1]), 3),
                         "label": str(row[2]) if len(row) > 2 else "0"})
    return rows


def run_campp(wav_path: str, min_speakers: int, max_speakers: int) -> dict:
    notes: list[str] = ["使用 3D-Speaker CAM++ 说话人日志（ModelScope）。"]
    try:
        rows = _run_campp_modelscope(wav_path)
        notes.append("引擎：modelscope pipeline。")
    except ImportError as exc:
        try:
            rows = _run_campp_funasr(wav_path)
            notes.append("引擎：funasr AutoModel。")
        except Exception:
            raise RuntimeError(
                "CAM++ 说话人日志不可用，请安装：pip install modelscope funasr "
                f"（模型自动从 ModelScope 下载）。原因：{exc}"
            ) from exc

    label_map: dict[str, int] = {}
    turns = []
    for row in rows:
        if row["end"] - row["start"] < MIN_TURN_SECONDS:
            continue
        cluster = label_map.setdefault(row["label"], len(label_map))
        turns.append({"start": row["start"], "end": row["end"], "cluster": cluster})

    signal, rate = _load_signal(wav_path)
    embedder = make_embedder("campp", wav_path, signal)
    if embedder is None:
        notes.append("未找到 CAM++ 声纹提取模型，聚类将不做声纹先验匹配（可直接人工归属）。")
    for turn in turns:
        turn["embedding"] = embedder(turn["start"], turn["end"]) if embedder else []

    # CAM++ decides its own speaker count; only honour an explicit cap.
    if max_speakers and len({t["cluster"] for t in turns}) > max_speakers:
        notes.append(f"模型输出说话人数超过上限 {max_speakers}，保留全部聚类（可在复核时合并）。")
    return _summarise(turns, "campp", notes)


# --- embedders (voiceprint extraction bound to engine family) ---------------

PYANNOTE_EMBEDDING_REPOS = (
    # The 3.1 pipeline itself embeds with this one, so using it keeps our
    # enrolled voiceprints in exactly the space the detector clusters in — and
    # unlike pyannote/embedding it is not gated.
    "pyannote/wespeaker-voxceleb-resnet34-LM",
    "pyannote/embedding",
)


def _make_pyannote_embedder(wav_path: str, signal: list[float] | None = None,
                            repos: tuple[str, ...] | None = None):
    """``f(start, end) -> vector`` in the pyannote feature space, or None.

    pyannote >= 4 removed the ``Inference("repo/id")`` shortcut: the first
    argument is now a loaded ``Model``, and passing a string fails deep inside
    the wrapper with a confusing ``'str' object has no attribute 'device'``.
    Build the model explicitly, then crop it in memory so the decoder
    (torchcodec) is never involved.

    Two things happen before each crop: the span is trimmed of its silent
    head/tail (see ``af.speech_trim``) so breaths do not dilute the voiceprint,
    and the result is memoised per span. Enrolment and matching share this
    extractor, so both sides stay in the same distribution, and a speaker-count
    sweep over the same spans only pays for the crops once.
    """
    repos = repos or PYANNOTE_EMBEDDING_REPOS
    cache_key = (wav_path, repos[0])
    with _CACHE_LOCK:
        cached = _PYANNOTE_EMBEDDER_CACHE.get(cache_key)
    if cached is not None:
        return cached["extract"]
    try:
        import numpy as np
        import torch
        from pyannote.audio import Inference, Model
        from pyannote.core import Segment
    except Exception:
        return None

    token = _pyannote_token()
    model = None
    for repo in repos:
        try:
            model = Model.from_pretrained(repo, token=token)
            break
        except TypeError:
            try:
                model = Model.from_pretrained(repo, use_auth_token=token)
                break
            except Exception:
                continue
        except Exception:
            continue
    if model is None:
        return None

    try:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        inference = Inference(model, window="whole", device=device)
        if signal is None:
            signal, _rate = _load_signal(wav_path)
        _, waveform, rate = _pyannote_audio_input(wav_path, signal)
        audio = {"waveform": waveform, "sample_rate": rate}
        vectors: dict[tuple[float, float], list[float]] = {}

        def extract(start: float, end: float) -> list[float]:
            # Segments shorter than ~0.2s make the embedder emit NaN, and carry
            # almost no speaker information — skip them up front.
            if end - start < 0.20:
                return []
            start, end = af.speech_trim(signal, rate, start, end)
            if end - start < 0.10:
                return []
            key = (round(start, 3), round(end, 3))
            hit = vectors.get(key)
            if hit is not None:
                return hit
            try:
                output = inference.crop(audio, Segment(start, end))
            except Exception:
                return []
            if hasattr(output, "data"):
                output = output.data
            array = np.asarray(output, dtype="float32")
            if array.ndim > 1:
                array = array.mean(axis=0)
            # A degenerate crop can still come back NaN/Inf; treat that as a
            # failure so it can never poison a cluster centroid.
            if array.size == 0 or not np.isfinite(array).all():
                return []
            vector = [float(v) for v in array]
            if not af.is_finite_vector(vector):
                return []
            vectors[key] = vector
            return vector

        with _CACHE_LOCK:
            _PYANNOTE_EMBEDDER_CACHE[cache_key] = {"extract": extract, "vectors": vectors}
            # Each cached entry pins an Inference model (hundreds of MB) plus
            # every crop vector; keep only the most recent couple of files.
            while len(_PYANNOTE_EMBEDDER_CACHE) > 2:
                _PYANNOTE_EMBEDDER_CACHE.pop(next(iter(_PYANNOTE_EMBEDDER_CACHE)))
        return extract
    except Exception:
        return None


def _make_funasr_embedder(model_id: str, wav_path: str, signal=None):
    """``f(start, end) -> vector`` from a FunASR/ModelScope speaker model.

    Used for CAM++ and ERes2NetV2. Slices the already-decoded signal in memory
    (one reused temp WAV) so a per-cue ``ffmpeg`` spawn never dominates.
    """
    try:
        from funasr import AutoModel

        model = AutoModel(model=model_id, device=_funasr_device())
        if signal is None:
            signal, rate = _load_signal(wav_path)
        else:
            rate = media.AUDIO_SAMPLE_RATE

        def extract(start: float, end: float) -> list[float]:
            import tempfile

            temp_path = getattr(extract, "_temp_path", None)
            if temp_path is None:
                handle, temp_path = tempfile.mkstemp(suffix=".wav")
                import os as _os
                _os.close(handle)
                extract._temp_path = temp_path
            _write_wav_slice(signal, rate, start, end, temp_path)
            result = model.generate(input=temp_path)
            for item in result or []:
                vector = item.get("spk_embedding")
                if vector is None:
                    vector = item.get("embedding")
                if vector is None:
                    continue
                # funasr returns a tensor; may carry a batch dim.
                if hasattr(vector, "detach"):
                    vector = vector.detach().cpu()
                if hasattr(vector, "numpy"):
                    import numpy as _np
                    vector = _np.asarray(vector).reshape(-1)
                values = [float(v) for v in list(vector)]
                if af.is_finite_vector(values):
                    return values
            return []

        return extract
    except Exception:
        return None


def make_embedder(engine: str, wav_path: str, signal=None):
    """Return ``f(start, end) -> vector`` for the engine's feature space, or None.

    ``signal`` (the already-decoded mono audio) is optional; callers inside a
    detection run pass it so the file is not decoded again.
    """
    engine = ENGINE_ALIASES.get(engine, engine)

    if engine in _VOICEPRINT_ENGINES:
        # pyannote's own turns, the cue engine and the external engines all share
        # the active voiceprint profile's feature space (see embedder_for).
        profile = VOICEPRINT_PROFILES[_VOICEPRINT_MODEL]
        if profile["kind"] == "pyannote":
            return _make_pyannote_embedder(wav_path, signal, repos=profile["repos"])
        if profile["kind"] == "funasr":
            return _make_funasr_embedder(profile["model"], wav_path, signal)
        return None

    if engine == "campp":
        return _make_funasr_embedder("iic/speech_campplus_sv_zh-cn_16k-common",
                                     wav_path, signal)

    # builtin feature space
    if signal is None:
        signal, rate = _load_signal(wav_path)
    else:
        rate = media.AUDIO_SAMPLE_RATE

    def extract_builtin(start: float, end: float) -> list[float]:
        return af.segment_embedding(signal, rate, start, end)

    return extract_builtin


# --- dispatch ---------------------------------------------------------------

def run_external(key: str, wav_path: str, min_speakers: int, max_speakers: int,
                 embedder_key: str = "pyannote") -> dict:
    """Run an engine that lives in its own venv, then enrich with voiceprints."""
    payload = external_engines.run(key, wav_path, min_speakers, max_speakers)
    turns = payload.get("turns") or []
    notes = list(payload.get("notes") or [])
    embedder = make_embedder(embedder_key, wav_path)
    if embedder is None:
        notes.append("声纹提取模型不可用，聚类不做声纹先验匹配（可直接人工归属）。")
    for turn in turns:
        turn["embedding"] = embedder(turn["start"], turn["end"]) if embedder else []
    return _summarise(turns, key, notes)


def engine_available(engine: str) -> bool:
    engine = ENGINE_ALIASES.get(engine, engine)
    return bool(engine_availability().get(engine, {}).get("available"))


def consensus_engine_for(engine: str) -> str:
    """Complementary engine for cross-checking (kept in the same voiceprint space)."""
    engine = ENGINE_ALIASES.get(engine, engine)
    return "pyannote" if engine == "voiceprint-cue" else "voiceprint-cue"


def _pick_sweep(candidates: list[tuple]) -> tuple | None:
    """Choose a speaker count from ``[(k, turns, score), ...]`` robustly.

    Plain argmax silhouette over-segments: each extra split (down to singleton
    clusters) raises the score, so the largest k always "wins". Skip candidate
    counts that produce a cluster of fewer than two cues, then — among the
    remaining — prefer the *smallest* k whose score is within a small tolerance
    of the best. That keeps a genuine extra speaker (a clear score jump) while
    rejecting a marginal, over-split one.
    """
    valid = []
    for k, turns, vectors, labels in candidates:
        sizes: dict[int, int] = {}
        for label in labels:
            sizes[int(label)] = sizes.get(int(label), 0) + 1
        if len(sizes) < 2 or min(sizes.values()) < 2:
            continue
        valid.append((k, turns, af.silhouette(vectors, labels)))
    if not valid:
        return None
    best = max(score for _, _, score in valid)
    tolerance = max(0.02, 0.03 * abs(best))
    return min((item for item in valid if item[2] >= best - tolerance),
               key=lambda item: item[0])


def run_sweep(engine: str, wav_path: str, spans: list[tuple[float, float]] | None,
              min_speakers: int, max_speakers: int) -> dict:
    """Try every speaker count in ``[min, max]`` and keep the best-separated one.

    The winner is picked by mean silhouette (cosine) of the turn embeddings —
    the "are these really distinct voices" question a fixed N cannot answer —
    with a bias toward the smallest count that scores within tolerance (see
    ``_pick_sweep``). Returns the chosen run with ``sweep_k``.
    """
    lo = max(1, int(min_speakers))
    hi = max(lo, int(max_speakers))
    if hi <= lo:
        result = run(engine, wav_path, spans, lo, lo)
        result.setdefault("notes", []).append(f"人数固定为 {lo}。")
        result["sweep_k"] = lo
        return result

    # Per-cue voiceprints do not depend on k, so extract once and only re-cluster
    # for each candidate count — a full voiceprint-cue sweep costs about one run.
    if ENGINE_ALIASES.get(engine, engine) == "voiceprint-cue":
        notes: list[str] = []
        base_turns = _voiceprint_cue_turns(wav_path, spans or [], notes)
        candidates = []
        for k in range(lo, hi + 1):
            trial = [dict(turn) for turn in base_turns]
            _cluster_turns(trial, k, k, [])
            vectors = [t["embedding"] for t in trial if t.get("embedding")]
            labels = [int(t["cluster"]) for t in trial if t.get("embedding")]
            candidates.append((k, trial, vectors, labels))
        chosen = _pick_sweep(candidates)
        if chosen is None:
            raise RuntimeError("人数自动扫描失败：所有候选人数都未能形成有效聚类。")
        best_k, best_turns, best_score = chosen
        notes.append("对每条字幕提 pyannote(wespeaker) 声纹后聚类（字幕级、免门控）。")
        notes.append(
            f"人数自动扫描：在 {lo}–{hi} 内选定 {best_k} 人（轮廓系数 {best_score:.3f}）。"
        )
        result = _summarise(best_turns, "voiceprint-cue", notes)
        result["sweep_k"] = best_k
        return result

    candidates = []
    skipped: list[str] = []
    for k in range(lo, hi + 1):
        try:
            candidate = run(engine, wav_path, spans, k, k)
        except Exception as exc:  # one bad count must not abort the sweep
            skipped.append(f"{k}({str(exc)[:40]})")
            continue
        vectors = [t["embedding"] for t in candidate.get("turns", []) if t.get("embedding")]
        labels = [int(t["cluster"]) for t in candidate.get("turns", []) if t.get("embedding")]
        candidates.append((k, candidate, vectors, labels))
    chosen = _pick_sweep(candidates)
    if chosen is None:
        raise RuntimeError("人数自动扫描失败：所有候选人数都未能完成。")
    best_k, best, best_score = chosen
    best.setdefault("notes", []).append(
        f"人数自动扫描：在 {lo}–{hi} 内选定 {best_k} 人（轮廓系数 {best_score:.3f}）。"
    )
    if skipped:
        best["notes"].append("扫描跳过：" + "、".join(skipped))
    best["sweep_k"] = best_k
    return best


def run(engine: str, wav_path: str, spans: list[tuple[float, float]] | None,
        min_speakers: int, max_speakers: int) -> dict:
    engine = ENGINE_ALIASES.get(engine, engine)
    if engine == "manual":
        return run_manual()
    if engine == "pyannote":
        return run_pyannote(wav_path, min_speakers, max_speakers)
    if engine == "voiceprint-cue":
        return run_voiceprint_cue(wav_path, spans or [], min_speakers, max_speakers)
    if engine == "campp":
        return run_campp(wav_path, min_speakers, max_speakers)
    if engine in external_engines.ENGINE_SPECS:
        return run_external(engine, wav_path, min_speakers, max_speakers)
    if engine == "nemo":
        raise RuntimeError(
            "NeMo 请改用内置的「NVIDIA Sortformer v2.1」引擎（已隔离环境），"
            "或使用 pyannote / CAM++。"
        )
    raise RuntimeError(f"未知的检测引擎：{engine}")
