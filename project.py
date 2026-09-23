"""Project state and orchestration: import -> detect -> align -> review -> export."""

from __future__ import annotations

import copy
import json
import os
import re
import shutil
import threading
import time
import uuid
from datetime import datetime

import align as aligner
import apppaths
import dataset_export
import diarize
import jsonutil
import media
import roles as rolelib
import subtitle_io as sio

BASE_DIR = apppaths.writable_dir()
# Tests must never touch the real data directory: an explicit override lets them
# point the whole store at a throwaway directory instead of deleting `data/`.
DATA_DIR = os.environ.get("SSP_DATA_DIR") or os.path.join(BASE_DIR, "data")
PROJECT_DIR = os.path.join(DATA_DIR, "projects")
WORK_DIR = os.path.join(DATA_DIR, "work")
EXPORT_DIR = os.path.join(DATA_DIR, "exports")
LIBRARY_PATH = os.path.join(DATA_DIR, "role_library.json")
SETTINGS_PATH = os.path.join(DATA_DIR, "settings.json")

DEFAULT_SETTINGS = {
    "default_engine": "campp",      # honoured when the engine is available
    "min_speakers": 1,
    "max_speakers": 6,
    # None = auto: use the feature space's calibrated threshold (roles.py).
    "threshold": None,
    "asr_model": "small",
    "asr_language": "",
    # Run detection / enrolment on a Demucs vocal stem (strips BGM + OP/ED).
    "separate_vocals": False,
    # Speaker-embedding profile for the pyannote family (see diarize.py).
    "voiceprint_model": "wespeaker",
}


def load_settings() -> dict:
    _ensure_dirs()
    settings = dict(DEFAULT_SETTINGS)
    if os.path.exists(SETTINGS_PATH):
        try:
            stored = _read_json(SETTINGS_PATH, {})
            for key in DEFAULT_SETTINGS:
                if key in stored and stored[key] is not None:
                    settings[key] = stored[key]
        except Exception:
            pass
    # 0.80 was the old fixed default; it now means "auto, per feature space".
    if settings.get("threshold") == 0.80:
        settings["threshold"] = None
    try:
        settings["min_speakers"] = max(1, int(settings["min_speakers"]))
        settings["max_speakers"] = max(1, int(settings["max_speakers"]))
        if settings["threshold"] is not None:
            settings["threshold"] = min(1.0, max(0.0, float(settings["threshold"])))
    except Exception:
        settings["min_speakers"], settings["max_speakers"] = 1, 6
        settings["threshold"] = None
    if settings["min_speakers"] > settings["max_speakers"]:
        settings["min_speakers"], settings["max_speakers"] = (
            settings["max_speakers"], settings["min_speakers"],
        )
    # Keep the engine layer's active voiceprint profile in sync with settings so
    # feature-space tags (role.embedder, engine.space) always reflect reality.
    try:
        diarize.set_voiceprint_model(settings.get("voiceprint_model") or "wespeaker")
    except Exception:
        pass
    return settings


def save_settings(updates: dict) -> dict:
    """Merge updates into data/settings.json; persist an HF token if provided."""
    _ensure_dirs()
    settings = load_settings()
    token = updates.pop("hf_token", None)
    for key in DEFAULT_SETTINGS:
        # threshold may be explicitly cleared back to None (auto); others keep
        # the previous value when the update omits them.
        if key in updates and (updates[key] is not None or key == "threshold"):
            settings[key] = updates[key]
    _write_json(SETTINGS_PATH, settings, indent=2)
    if token:
        token = str(token).strip()
        if token:
            with open(os.path.join(DATA_DIR, "hf_token.txt"), "w", encoding="utf-8") as handle:
                handle.write(token)
    return public_settings()


def public_settings() -> dict:
    settings = load_settings()
    token_set = bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN"))
    if not token_set:
        try:
            with open(os.path.join(DATA_DIR, "hf_token.txt"), "r", encoding="utf-8-sig") as handle:
                token_set = bool(handle.read().strip())
        except OSError:
            pass
    settings["hf_token_set"] = token_set
    settings["data_dir"] = DATA_DIR
    return settings

DEFAULT_THRESHOLD = None  # auto: per-feature-space calibration (roles.py)


def _ensure_dirs() -> None:
    for path in (DATA_DIR, PROJECT_DIR, WORK_DIR, EXPORT_DIR):
        os.makedirs(path, exist_ok=True)


# Parsed-JSON cache keyed by path and (mtime_ns, size). `state_payload` reads the
# library and settings on nearly every request; without this the same files were
# re-read and re-parsed several times per page load. Writes invalidate eagerly
# (mtime alone can collide on a same-tick rewrite of equal length).
_CACHE_LOCK = threading.RLock()
_JSON_CACHE: dict[str, tuple[tuple[int, int], object]] = {}


def _invalidate(path: str) -> None:
    with _CACHE_LOCK:
        _JSON_CACHE.pop(path, None)


def _write_json(path: str, data, indent: int | None = None) -> None:
    """Write JSON atomically: a crash mid-write must not truncate the target.

    The old direct write left a half-file behind, which `list_projects` then
    swallowed silently — the whole project vanished from the UI. Writing to a
    sibling temp file and ``os.replace`` makes the swap atomic.
    """
    tmp = f"{path}.{uuid.uuid4().hex[:8]}.tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=indent)
    os.replace(tmp, path)
    _invalidate(path)


def _read_json(path: str, default):
    try:
        stat = os.stat(path)
    except OSError:
        return copy.deepcopy(default)
    key = (stat.st_mtime_ns, stat.st_size)
    with _CACHE_LOCK:
        hit = _JSON_CACHE.get(path)
        if hit is not None and hit[0] == key:
            return copy.deepcopy(hit[1])
    with open(path, "r", encoding="utf-8-sig") as handle:
        data = json.load(handle)
    with _CACHE_LOCK:
        _JSON_CACHE[path] = (key, data)
    return copy.deepcopy(data)


def _slug(text: str) -> str:
    cleaned = re.sub(r"[^\w\u4e00-\u9fff-]+", "_", text or "").strip("_")
    return cleaned[:40] or "project"


# Public alias: server.py names ASR output with it (was reaching into `_slug`).
slug = _slug


def normalize_path(path: str) -> str:
    """Accept Windows / Git-Bash / quoted paths and return a native absolute path."""
    path = (path or "").strip().strip('"').strip("'")
    if not path:
        return path
    # Git Bash style /c/Users/... -> C:\Users\...
    match = re.fullmatch(r"/([a-zA-Z])/(.*)", path)
    if match:
        drive, rest = match.groups()
        path = f"{drive.upper()}:/{rest}"
    path = path.replace("/", os.sep)
    return os.path.abspath(path)


# --- persistence ------------------------------------------------------------

def project_path(project_id: str) -> str:
    return os.path.join(PROJECT_DIR, f"{project_id}.json")


def save(project: dict) -> dict:
    _ensure_dirs()
    project["updated"] = datetime.now().isoformat(timespec="seconds")
    path = project_path(project["id"])
    _write_json(path, jsonutil.json_safe(project))
    # Same-tick rewrites of equal length can share (mtime, size); drop the
    # summary explicitly so a review save is never shown stale in the list.
    with _CACHE_LOCK:
        _SUMMARY_CACHE.pop(os.path.basename(path), None)
    return project


def load(project_id: str) -> dict:
    with open(project_path(project_id), "r", encoding="utf-8-sig") as handle:
        return jsonutil.json_safe(json.load(handle))


def exists(project_id: str) -> bool:
    return os.path.exists(project_path(project_id))


_SUMMARY_CACHE: dict[str, tuple[tuple[int, int], dict]] = {}


def list_projects() -> list[dict]:
    """Summarise every project, parsing only files whose (mtime, size) changed.

    ``state_payload`` calls this on almost every request; the full-project cache
    is deliberately not kept (projects can be large), so this per-file summary
    cache is what keeps the project list cheap as the user accumulates files.
    """
    _ensure_dirs()
    items: list[dict] = []
    seen: set[str] = set()
    for filename in os.listdir(PROJECT_DIR):
        if not filename.endswith(".json"):
            continue
        path = os.path.join(PROJECT_DIR, filename)
        try:
            stat = os.stat(path)
        except OSError:
            continue
        seen.add(filename)
        key = (stat.st_mtime_ns, stat.st_size)
        with _CACHE_LOCK:
            hit = _SUMMARY_CACHE.get(filename)
        if hit is not None and hit[0] == key:
            items.append(copy.deepcopy(hit[1]))
            continue
        try:
            project = _read_json(path, {})
            summary = {
                "id": project["id"],
                "name": project.get("name"),
                "media_path": project.get("media_path"),
                "subtitle_path": project.get("subtitle_path"),
                "created": project.get("created"),
                "updated": project.get("updated"),
                "segments": len(project.get("segments", [])),
                "stats": stats(project),
            }
        except Exception:
            continue
        with _CACHE_LOCK:
            _SUMMARY_CACHE[filename] = (key, summary)
        items.append(copy.deepcopy(summary))
    with _CACHE_LOCK:
        for name in list(_SUMMARY_CACHE):
            if name not in seen:
                _SUMMARY_CACHE.pop(name, None)
    items.sort(key=lambda item: item.get("updated") or "", reverse=True)
    return items


# --- role library (cross-project voiceprints) ------------------------------

def load_library() -> list[dict]:
    _ensure_dirs()
    try:
        return _read_json(LIBRARY_PATH, {"roles": []}).get("roles", [])
    except Exception:
        return []


def save_library(library_roles: list[dict]) -> list[dict]:
    _ensure_dirs()
    _write_json(LIBRARY_PATH, {"roles": library_roles})
    return library_roles


def _library_key(name: str, embedder: str) -> tuple[str, str]:
    """Identity of a library role: same person, same feature space."""
    return (str(name or "").strip(), embedder or "builtin")


def _merge_samples(a: list[dict], b: list[dict]) -> list[dict]:
    seen: set[tuple] = set()
    merged: list[dict] = []
    for sample in list(a or []) + list(b or []):
        key = (round(float(sample.get("start", 0.0)), 3),
               round(float(sample.get("end", 0.0)), 3),
               sample.get("source", ""))
        if key in seen:
            continue
        seen.add(key)
        merged.append(sample)
    return merged


def add_library_role(name: str, color: str | None = None,
                     embedding: list[float] | None = None,
                     samples: list[dict] | None = None,
                     embedder: str = "builtin") -> tuple[dict, str]:
    """Insert or update a global-library voiceprint, keyed by (name, space).

    Recording the same role again must not pile up duplicate rows, so an entry
    with the same trimmed name *and* feature space is updated in place (its
    ``library_id`` and colour are kept); anything else appends. The stored
    embedding is only replaced when a non-empty one is supplied, so an
    embed-less update cannot wipe a good voiceprint.
    Returns ``(entry, "created" | "updated")``.
    """
    library_roles = load_library()
    key = _library_key(name, embedder)
    existing = next(
        (r for r in library_roles
         if _library_key(r.get("name", ""), r.get("embedder", "builtin")) == key),
        None,
    )
    now = datetime.now().isoformat(timespec="seconds")
    if existing is not None:
        existing["name"] = str(name or "").strip() or existing.get("name", "")
        if embedding:
            existing["embedding"] = list(embedding)
        if samples:
            existing["samples"] = _merge_samples(existing.get("samples") or [], samples)
        existing.setdefault("color", color or rolelib.next_color(
            [r["color"] for r in library_roles]))
        existing["updated"] = now
        save_library(library_roles)
        return existing, "updated"

    existing_colors = [r["color"] for r in library_roles]
    role = {
        "library_id": f"lib_{uuid.uuid4().hex[:8]}",
        "name": str(name or "").strip() or "未命名",
        "color": color or rolelib.next_color(existing_colors),
        "embedder": embedder or "builtin",
        "embedding": list(embedding or []),
        "samples": list(samples or []),
        "created": now,
        "updated": now,
    }
    library_roles.append(role)
    save_library(library_roles)
    return role, "created"


def dedupe_library() -> dict:
    """Collapse same (name, feature space) library rows into one.

    Keeps the first entry's ``library_id``/colour, merges the sample lists, and
    recomputes the embedding as the sample-count-weighted mean of the merged
    entries' embeddings. Returns ``{"removed": N, "kept": M}``.
    """
    library_roles = load_library()
    groups: dict[tuple[str, str], dict] = {}
    order: list[tuple[str, str]] = []
    weight: dict[tuple[str, str], float] = {}
    sums: dict[tuple[str, str], list[float]] = {}
    for entry in library_roles:
        key = _library_key(entry.get("name", ""), entry.get("embedder", "builtin"))
        if key not in groups:
            groups[key] = {
                "library_id": entry.get("library_id"),
                "name": entry.get("name"),
                "color": entry.get("color"),
                "embedder": entry.get("embedder", "builtin"),
                "samples": list(entry.get("samples") or []),
                "created": entry.get("created"),
                "updated": entry.get("updated"),
                "embedding": [],
            }
            order.append(key)
            weight[key] = 0.0
            sums[key] = []
        else:
            groups[key]["samples"] = _merge_samples(
                groups[key]["samples"], entry.get("samples") or [])
        vector = entry.get("embedding") or []
        if vector:
            if not sums[key]:
                sums[key] = [0.0] * len(vector)
            if len(vector) == len(sums[key]):
                scale = max(1.0, float(len(entry.get("samples") or [])))
                for index, value in enumerate(vector):
                    sums[key][index] += value * scale
                weight[key] += scale
    for key in order:
        if sums[key] and weight[key] > 0:
            groups[key]["embedding"] = [value / weight[key] for value in sums[key]]
    result = [groups[key] for key in order]
    removed = len(library_roles) - len(result)
    if removed:
        save_library(result)
    return {"removed": removed, "kept": len(result)}


# --- project construction ---------------------------------------------------

def create(media_path: str, subtitle_path: str, name: str | None = None,
           second_subtitle_path: str | None = None,
           line_mode: str = "auto") -> dict:
    """Import a project. ``second_subtitle_path`` is an optional translation track."""
    _ensure_dirs()
    media_path = normalize_path(media_path)
    subtitle_path = normalize_path(subtitle_path)
    if not os.path.exists(media_path):
        raise FileNotFoundError(f"找不到媒体文件：{media_path}")
    if not os.path.exists(subtitle_path):
        raise FileNotFoundError(f"找不到字幕文件：{subtitle_path}")

    segments, meta = sio.parse_subtitle(subtitle_path)
    if not segments:
        raise ValueError("字幕文件解析结果为空，请检查文件格式。")

    notes: list[str] = []
    skipped = int(meta.get("skipped_effects") or 0)
    if skipped:
        notes.append(f"已跳过 {skipped} 条特效 / 标题 / 字幕组 / 歌词卡拉OK事件（非对白，未计入清单）。")
    recovered = int(meta.get("recovered_lyrics") or 0)
    if recovered:
        notes.append(f"已从注释恢复 {recovered} 条歌词文本。")
    line_info = sio.detect_line_mode(segments)
    if second_subtitle_path:
        second_path = normalize_path(second_subtitle_path)
        if not os.path.exists(second_path):
            raise FileNotFoundError(f"找不到翻译字幕文件：{second_path}")
        others, _ = sio.parse_subtitle(second_path)
        matched = sio.merge_translations(segments, others)
        notes.append(f"已按时间轴合并翻译字幕，{matched}/{len(segments)} 条匹配成功。")
        if line_info["suggested"] == "bilingual":
            # A dedicated translation file wins over inline second lines.
            segments, line_info = sio.apply_line_mode(segments, "join")
    else:
        second_path = None
        segments, line_info = sio.apply_line_mode(segments, line_mode)
        if line_info.get("applied") == "bilingual":
            notes.append(f"检测到双语字幕：{line_info['multiline']} 条多行字幕已拆分为原文 + 译文。")
        elif line_info.get("applied") == "join" and line_info["multiline"]:
            notes.append(f"{line_info['multiline']} 条多行字幕已合并为单句（疑似硬换行）。")

    if not second_path and any((s.get("translation") or "").strip() for s in segments):
        notes.append("字幕内含译文，导出时可选择同时输出或只输出单语。")

    project_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    kind = media.kind_of(media_path)
    project = {
        "id": project_id,
        "name": _slug(name) if name else _slug(os.path.splitext(os.path.basename(media_path))[0]),
        "media_path": os.path.abspath(media_path),
        "media_kind": kind if kind != "unknown" else "video",
        "subtitle_path": os.path.abspath(subtitle_path),
        "subtitle_format": os.path.splitext(subtitle_path)[1].lstrip(".").lower(),
        "second_subtitle_path": os.path.abspath(second_path) if second_path else None,
        "line_mode": line_info.get("applied", line_mode),
        "line_info": line_info,
        "import_notes": notes,
        "bilingual": any((s.get("translation") or "").strip() for s in segments),
        "duration": round(max(media.probe_duration(media_path),
                              segments[-1]["end"] if segments else 0.0), 3),
        "segments": segments,
        "roles": [],
        "diarization": {"turns": [], "clusters": {}, "engine": None, "notes": []},
        "engine": None,
        "detection_notes": [],
        "subtitle_meta": {k: v for k, v in meta.items() if k != "styles"},
        "exports": [],
        "created": datetime.now().isoformat(timespec="seconds"),
    }
    # Pre-create roles from names already present in the subtitle file.
    hints = []
    for segment in segments:
        hint = segment.get("speaker_name_hint")
        if hint and hint not in hints:
            hints.append(hint)
    for hint in hints:
        project["roles"].append(
            rolelib.new_role(_next_role_id(project), hint,
                             rolelib.next_color([r["color"] for r in project["roles"]]),
                             role_type="named")
        )
    return save(project)


def _next_role_id(project: dict) -> int:
    used = [role["id"] for role in project["roles"]]
    return (max(used) + 1) if used else 0


def work_wav(project: dict) -> str:
    """Decode (and cache) the project audio as mono 16 kHz WAV."""
    _ensure_dirs()
    target = os.path.join(WORK_DIR, f"{project['id']}.wav")
    source = project["media_path"]
    if os.path.exists(target) and os.path.getmtime(target) >= os.path.getmtime(source):
        return target
    return media.to_wav16k(source, target)


def vocals_wav(project: dict) -> str:
    """Decode + Demucs-separate the project audio (cached mono 16 kHz vocals WAV)."""
    _ensure_dirs()
    target = os.path.join(WORK_DIR, f"{project['id']}.vocals.wav")
    source = project["media_path"]
    if os.path.exists(target) and os.path.getmtime(target) >= os.path.getmtime(source):
        return target
    return media.separate_vocals(source, target)


def analysis_wav(project: dict, separate_vocals: bool) -> tuple[str, str | None]:
    """WAV used for detection and voiceprint work.

    With ``separate_vocals`` on, both detection *and* enrolment run on the Demucs
    vocal stem so their voiceprints stay comparable; otherwise the plain decode.
    Falls back to the plain WAV (with a note) when Demucs is missing or fails.
    """
    if not separate_vocals:
        return work_wav(project), None
    try:
        return vocals_wav(project), None
    except Exception as exc:
        return work_wav(project), f"人声分离失败，改用原始音轨：{str(exc)[:140]}"


# --- detection + alignment --------------------------------------------------

def _refine_segments(project: dict, result: dict, cluster_to_role: dict,
                     wanted_embedder: str, wav_path: str,
                     overwrite_manual: bool, notes: list[str]) -> None:
    """Re-decide cues by their own voiceprint where the engine used overlap only.

    Neural feature spaces only (pyannote wespeaker / ERes2NetV2). The CAM++
    engine slices audio per call and would be far too slow here; the builtin
    space carries too little speaker information to override a timing vote.
    Failures are recorded as a note, never fatal.
    """
    if wanted_embedder not in ("pyannote", "eres2netv2") \
            or result.get("engine") == "voiceprint-cue":
        return
    embedder = diarize.make_embedder(result["engine"], wav_path)
    if embedder is None:
        notes.append("声纹精修跳过：声纹模型不可用。")
        return
    try:
        _reassigned, _filled, refine_notes = aligner.refine_with_voiceprints(
            project["segments"], result.get("clusters", {}), cluster_to_role,
            embedder, overwrite_manual=overwrite_manual,
        )
        notes.extend(refine_notes)
    except Exception as exc:
        notes.append(f"声纹精修跳过：{str(exc)[:120]}")
    # Refinement can only move a cue between clusters the matcher accepted. A cue
    # that overlaps no turn, or lives in a cluster no role claimed, still has no
    # candidate; match its own voiceprint against every enrolled role so minor
    # cast and singing cues get attributed instead of sitting 待定 forever.
    role_vectors = [role for role in project["roles"]
                    if role.get("embedding") and role.get("type") != "ignored"
                    and role.get("embedder", "builtin") == wanted_embedder]
    if role_vectors:
        try:
            _filled, fill_notes = aligner.fill_with_role_voiceprints(
                project["segments"], role_vectors, embedder,
                overwrite_manual=overwrite_manual,
            )
            notes.extend(fill_notes)
        except Exception as exc:
            notes.append(f"声纹判定跳过：{str(exc)[:120]}")


def _apply_consensus(project: dict, result: dict, engine: str, sweep: bool,
                     min_speakers: int, max_speakers: int, wav_path: str,
                     spans: list, notes: list[str]) -> None:
    """Cross-check the primary decisions with a complementary engine.

    Cues the two engines disagree on go back to "pending" for review; consensus
    only ever adds pending, never corrects. Any failure is a note, not an error.
    """
    primary = result.get("engine", engine)
    secondary = diarize.consensus_engine_for(primary)
    # Consensus needs both engines' clusters in the same voiceprint space.
    if diarize.embedder_for(primary) != diarize.embedder_for(secondary):
        notes.append("双引擎共识跳过：两引擎声纹特征空间不同，无法比对。")
        return
    if not diarize.engine_available(secondary):
        notes.append(f"双引擎共识跳过：{secondary} 不可用。")
        return
    if sweep and result.get("sweep_k"):
        c_min = c_max = int(result["sweep_k"])
    else:
        c_min, c_max = max(1, min_speakers), max(1, max_speakers)
    try:
        secondary_result = diarize.run(secondary, wav_path, spans, c_min, c_max)
        secondary_segments = [dict(s) for s in project["segments"]]
        secondary_segments, _ = aligner.align(
            secondary_segments, secondary_result.get("turns", []),
            {c: c for c in secondary_result.get("clusters", {})},
        )
        secondary_by_id = {s["id"]: s.get("cluster") for s in secondary_segments}
        b_to_a = aligner.match_clusters(result.get("clusters", {}),
                                        secondary_result.get("clusters", {}))
        disagreed = 0
        for segment in project["segments"]:
            if segment.get("speaker_id") is None or segment.get("status") == "manual":
                continue
            # The cue's own voiceprint clearly backs the role it got: a second,
            # differently-shaped clustering disagreeing is not enough to undo it.
            if (segment.get("vp_score") or 0.0) >= aligner.VOICEPRINT_TRUST:
                continue
            other = secondary_by_id.get(segment["id"])
            mapped = None if other is None else b_to_a.get(int(other))
            if mapped is None or segment.get("cluster") is None:
                continue  # secondary had nothing to say about this cue
            if int(segment["cluster"]) != mapped:
                segment["speaker_id"] = None
                segment["status"] = "pending"
                segment["confidence"] = None
                segment["note"] = "双引擎不一致，待复核"
                disagreed += 1
        if disagreed:
            notes.append(
                f"双引擎共识（{primary} × {secondary}）：{disagreed} 条不一致，已标为待定。"
            )
        else:
            notes.append(f"双引擎共识（{primary} × {secondary}）：结果一致。")
    except Exception as exc:
        notes.append(f"双引擎共识跳过：{str(exc)[:120]}")


def run_detection(project: dict, engine: str = "manual", min_speakers: int = 1,
                  max_speakers: int = 6, threshold: float | None = DEFAULT_THRESHOLD,
                  overwrite_manual: bool = False, sweep: bool = False,
                  consensus: bool = False, separate_vocals: bool = False) -> dict:
    wav_path, separation_note = analysis_wav(project, separate_vocals)
    spans = [(segment["start"], segment["end"]) for segment in project["segments"]]
    if sweep:
        result = diarize.run_sweep(engine, wav_path, spans, max(1, min_speakers),
                                   max(1, max_speakers))
    else:
        result = diarize.run(engine, wav_path, spans, max(1, min_speakers),
                             max(1, max_speakers))
    # Turn-level embeddings are only needed while summarising clusters; keeping
    # them would bloat the project JSON (and every review save) with thousands of
    # 256-d floats. Cluster means stay for voiceprint matching.
    for turn in result.get("turns", []):
        turn.pop("embedding", None)
    project["diarization"] = result
    project["engine"] = result["engine"]
    # Drop last run's per-cue voiceprint confidence so a cue this run does not
    # re-score cannot inherit a stale "trusted" flag.
    for segment in project["segments"]:
        segment.pop("vp_score", None)

    notes = list(result.get("notes", []))
    if separate_vocals and separation_note is None:
        notes.append("已使用 Demucs 人声分离后的音轨做检测（去除 BGM / 伴奏）。")
    elif separation_note:
        notes.append(separation_note)

    # Human review is the cleanest enrolment data: fold each role's manually
    # assigned cues into its voiceprint before matching so the next run knows
    # them. Also collapse duplicate global-library rows left by older builds.
    _, learn_notes = learn_voiceprints(project, engine=result["engine"], wav_path=wav_path)
    notes.extend(learn_notes)
    library_cleanup = dedupe_library()
    if library_cleanup["removed"]:
        notes.append(f"整理全局角色库：合并了 {library_cleanup['removed']} 条重复记录。")

    # Voiceprints only match clusters from the same feature space (hard rule).
    wanted_embedder = diarize.embedder_for(result["engine"])
    candidates = [dict(role) for role in project["roles"]
                  if role.get("embedding") and role.get("embedder", "builtin") == wanted_embedder]
    skipped = sum(
        1 for role in project["roles"]
        if role.get("embedding") and role.get("embedder", "builtin") != wanted_embedder
    )
    if skipped:
        notes.append(
            f"{skipped} 个角色的声纹属于其他特征空间，本次检测不参与先验匹配"
            "（用当前引擎重新录入即可）。"
        )

    library_roles = load_library()
    for entry in library_roles:
        if not entry.get("embedding"):
            continue
        if entry.get("embedder", "builtin") != wanted_embedder:
            continue
        candidates.append({
            "id": f"library:{entry['library_id']}",
            "name": entry["name"],
            "color": entry["color"],
            "type": "registered",
            "embedder": entry.get("embedder", "builtin"),
            "embedding": entry.get("embedding") or [],
            "samples": entry.get("samples") or [],
        })

    mapping, map_notes = rolelib.map_clusters(result.get("clusters", {}), candidates,
                                              threshold, embedder=wanted_embedder)
    notes.extend(map_notes)

    name_to_role = {role["name"]: role["id"] for role in project["roles"]}
    cluster_to_role: dict[int, int] = {}
    unmatched_clusters = 0

    for cluster_id in sorted(result.get("clusters", {})):
        entry = mapping.get(int(cluster_id), {})
        matched_id = entry.get("role_id")
        if matched_id is not None and isinstance(matched_id, int):
            cluster_to_role[int(cluster_id)] = matched_id
        elif isinstance(matched_id, str) and matched_id.startswith("library:"):
            library_id = matched_id.split(":", 1)[1]
            source = next((r for r in library_roles if r["library_id"] == library_id), None)
            if source is not None:
                new_role = rolelib.new_role(
                    _next_role_id(project), source["name"], source["color"],
                    role_type="registered", embedder=source.get("embedder", "builtin"),
                )
                new_role["embedding"] = list(source.get("embedding") or [])
                new_role["samples"] = list(source.get("samples") or [])
                new_role["from_library"] = library_id
                project["roles"].append(new_role)
                cluster_to_role[int(cluster_id)] = new_role["id"]
                notes.append(f"角色「{source['name']}」按先验声纹自动匹配成功。")
            else:
                unmatched_clusters += 1
        else:
            # No confident match to a user-created role. Leave the cluster out of
            # the map entirely so its cues fall through to 待定: a miss costs one
            # manual assignment, a wrong auto-assignment silently corrupts the set.
            unmatched_clusters += 1

    # Rebuild name→role map because library roles may have just been created.
    name_to_role = {role["name"]: role["id"] for role in project["roles"]}

    project["segments"], align_notes = aligner.align(
        project["segments"], result.get("turns", []), cluster_to_role,
        name_to_role=name_to_role, overwrite_manual=overwrite_manual,
    )
    notes.extend(align_notes)

    _refine_segments(project, result, cluster_to_role, wanted_embedder, wav_path,
                     overwrite_manual, notes)

    if consensus:
        _apply_consensus(project, result, engine, sweep, min_speakers,
                         max_speakers, wav_path, spans, notes)

    project["cluster_to_role"] = cluster_to_role
    if unmatched_clusters:
        notes.append(
            f"{unmatched_clusters} 个聚类未匹配到已创建的角色，相关字幕保持待定"
            "（宁可漏、不误判，可人工归属）。"
        )
    # Detection no longer invents 待定角色: clear any left by older runs that the
    # latest mapping does not use (manual-referenced ones are kept).
    pruned = _prune_pending_roles(project, set(cluster_to_role.values()))
    if pruned:
        notes.append(f"清理了 {pruned} 个历史待定角色（未匹配的聚类不再自动生成）。")
    project["detection_notes"] = notes
    return save(project)


def _ensure_pending_role(project: dict, embedder: str = "builtin",
                         assigned: dict | None = None) -> int:
    """Return the id of an unused pending role, creating one when needed.

    Legacy helper: detection no longer creates "待定角色" (unmatched clusters now
    keep their cues 待定), but older projects may still hold them, so this stays
    for reuse/cleanup. ``assigned`` is the in-progress cluster→role map; only
    roles already handed out *this* run are considered used.
    """
    used = set((assigned or {}).values())
    for role in project["roles"]:
        if role["type"] == "pending" and role["id"] not in used:
            return role["id"]
    pending_count = sum(1 for role in project["roles"] if role["type"] == "pending") + 1
    role = rolelib.new_role(
        _next_role_id(project), f"{rolelib.UNKNOWN_NAME}角色{pending_count}",
        rolelib.next_color([r["color"] for r in project["roles"]]),
        role_type="pending", embedder=embedder,
    )
    project["roles"].append(role)
    return role["id"]


def _prune_pending_roles(project: dict, keep_ids: set[int]) -> int:
    """Drop auto "待定" roles not referenced by the latest cluster map.

    Detection no longer creates pending roles, so this also clears the ones left
    by older runs. Pending roles still referenced (by the new cluster map or by
    a manual assignment) are kept. Returns how many roles were removed.
    """
    keep = {int(i) for i in keep_ids}
    for segment in project["segments"]:
        if segment.get("status") == "manual" and segment.get("speaker_id") is not None:
            keep.add(int(segment["speaker_id"]))
    dropped = {role["id"] for role in project["roles"]
               if role.get("type") == "pending" and role["id"] not in keep}
    if dropped:
        _remove_roles(project, dropped)
    return len(dropped)


# --- review operations ------------------------------------------------------

def set_segment(project: dict, segment_id: int, speaker_id: int | None,
                reviewed: bool = True) -> dict:
    for segment in project["segments"]:
        if segment["id"] != segment_id:
            continue
        segment["speaker_id"] = speaker_id
        segment["status"] = "manual" if reviewed else "pending"
        segment["confidence"] = None
        return project
    raise KeyError(f"字幕 {segment_id} 不存在")


def bulk_set(project: dict, segment_ids: list[int], speaker_id: int | None,
             status: str = "manual", confidence: float | None = None) -> int:
    target = set(int(i) for i in segment_ids)
    changed = 0
    for segment in project["segments"]:
        if segment["id"] in target:
            segment["speaker_id"] = speaker_id
            segment["status"] = "manual" if speaker_id is not None and status == "manual" else status
            # None (the default, and every hand-assignment path) clears the
            # detector's score; undo/redo passes the snapshot value back so an
            # auto assignment keeps its confidence badge.
            segment["confidence"] = confidence
            changed += 1
    return changed


def reset_auto(project: dict) -> int:
    """Clear automatic assignments, keeping human-reviewed ones."""
    cleared = 0
    for segment in project["segments"]:
        if segment.get("status") == "auto":
            segment["speaker_id"] = None
            segment["status"] = "pending"
            segment["confidence"] = None
            cleared += 1
    return cleared


UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")

_MEDIA_EXT = set(media._VIDEO_EXTENSIONS) | set(media._AUDIO_EXTENSIONS)
_SUB_EXT = {".srt", ".vtt", ".ass", ".ssa"}


def _upload_target(name: str) -> str:
    """Reserve a safe, unique path under data/uploads/ (no bytes written yet)."""
    _ensure_dirs()
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    base = os.path.basename(name or "file")
    safe = re.sub(r'[\\/:*?"<>|\r\n]+', "_", base).strip(" ._") or "file"
    target = os.path.join(UPLOAD_DIR, safe)
    if os.path.exists(target):
        stem, ext = os.path.splitext(safe)
        target = os.path.join(UPLOAD_DIR, f"{stem}_{uuid.uuid4().hex[:6]}{ext}")
    return target


def save_upload(name: str, payload: bytes) -> str:
    """Persist an uploaded file under data/uploads/ and return its path."""
    target = _upload_target(name)
    with open(target, "wb") as handle:
        handle.write(payload)
    return target


def classify_file(name: str) -> str:
    """media | subtitle | other, by extension."""
    ext = os.path.splitext(name)[1].lower()
    if ext in _MEDIA_EXT:
        return "media"
    if ext in _SUB_EXT:
        return "subtitle"
    return "other"


def set_translation(project: dict, path: str) -> dict:
    """Attach a subtitle file as the translation track (merged by overlap)."""
    path = normalize_path(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到翻译字幕：{path}")
    others, _ = sio.parse_subtitle(path)
    if not others:
        raise ValueError("翻译字幕解析结果为空。")
    matched = sio.merge_translations(project["segments"], others)
    project["second_subtitle_path"] = path
    project["bilingual"] = matched > 0
    notes = project.setdefault("import_notes", [])
    notes.append(f"已挂载翻译字幕 {os.path.basename(path)}：{matched}/{len(project['segments'])} 条匹配。")
    return project


def set_line_mode(project: dict, mode: str) -> dict:
    """Re-interpret cue text (bilingual / interleaved / join) after import.

    The original multi-line body is kept in ``raw_text`` the first time so the
    switch is reversible without re-importing the file. Modes that change the
    cue count (interleaved) splice the segment list, transferring any speaker
    assignment from folded cues onto their partner.
    """
    for segment in project["segments"]:
        if "raw_text" not in segment:
            translation = (segment.get("translation") or "").strip()
            segment["raw_text"] = (
                f"{segment.get('text', '')}\n{translation}" if translation
                else segment.get("text", "")
            )
        segment["text"] = segment["raw_text"]
        segment["translation"] = ""
    segments, info = sio.apply_line_mode(project["segments"], mode)
    project["segments"] = segments
    project["line_mode"] = info.get("applied", mode)
    project["line_info"] = info
    project["bilingual"] = any((s.get("translation") or "").strip()
                               for s in project["segments"])
    return project


def stats(project: dict) -> dict:
    segments = project.get("segments", [])
    assigned = sum(1 for s in segments if s.get("speaker_id") is not None)
    manual = sum(1 for s in segments if s.get("status") == "manual")
    auto = sum(1 for s in segments if s.get("status") == "auto")
    pending = len(segments) - assigned
    return {
        "total": len(segments),
        "assigned": assigned,
        "pending": pending,
        "auto": auto,
        "manual": manual,
        "roles": len(project.get("roles", [])),
        "speakers_used": len({s.get("speaker_id") for s in segments
                              if s.get("speaker_id") is not None}),
    }


# --- role operations --------------------------------------------------------

def add_role(project: dict, name: str, color: str | None = None) -> dict:
    if any(role["name"] == name for role in project["roles"]):
        raise ValueError(f"角色「{name}」已存在")
    role = rolelib.new_role(
        _next_role_id(project), name,
        color or rolelib.next_color([r["color"] for r in project["roles"]]),
        role_type="named",
    )
    project["roles"].append(role)
    return role


def update_role(project: dict, role_id: int, name: str | None = None,
                color: str | None = None, role_type: str | None = None) -> dict:
    role = _find_role(project, role_id)
    if name is not None and name != role["name"]:
        role["name"] = name
        if role["type"] == "pending":
            role["type"] = "registered" if role.get("embedding") else "named"
            # Legacy projects kept grey pending roles; give them a palette
            # colour on promotion so they don't render identically.
            if role["color"] == rolelib.PENDING_COLOR:
                role["color"] = rolelib.next_color([r["color"] for r in project["roles"]
                                                    if r["id"] != role["id"]])
    if color is not None:
        role["color"] = color
    if role_type is not None:
        role["type"] = role_type
    if role["type"] == "registered" and not role.get("embedding"):
        role["type"] = "named"
    # Naming a role is an explicit "this is a real person" act: register its
    # voiceprint in the global library automatically (upsert) so other projects
    # can reuse it — no separate "入库" button needed.
    if name is not None and role.get("embedding") and role.get("type") in ("named", "registered"):
        add_library_role(role["name"], role["color"], role.get("embedding"),
                         role.get("samples"), embedder=role.get("embedder", "builtin"))
        role["in_library"] = True
    return role


def _find_role(project: dict, role_id: int) -> dict:
    for role in project["roles"]:
        if role["id"] == int(role_id):
            return role
    raise KeyError(f"角色 {role_id} 不存在")


def enroll_engine(project: dict, available=None) -> str:
    """Engine whose feature space a newly enrolled voiceprint should live in.

    A voiceprint only compares against clusters from the same model, so it must
    be stored in the space detection will use. A project that has detected
    before carries that engine; before any detection we fall back to the
    configured default engine, so the common "create roles and enrol first,
    detect later" flow still lands in the right space. ``manual``/builtin is
    the last resort.
    """
    is_available = available or diarize.engine_available
    engine = project.get("engine") or ""
    if engine and engine != "manual" and is_available(engine):
        return engine
    default = load_settings().get("default_engine") or ""
    if default and default != "manual" and is_available(default):
        return default
    return "manual"


def enroll_role(project: dict, role_id: int, start: float, end: float,
                also_library: bool = False) -> tuple[dict, list[str]]:
    """Enrol a voiceprint into the feature space detection will use."""
    role = _find_role(project, role_id)
    wav_path, separation_note = analysis_wav(
        project, bool(load_settings().get("separate_vocals")))
    engine = enroll_engine(project)
    embedder = diarize.embedder_for(engine)
    notes = []
    if separation_note:
        notes.append(separation_note)
    if engine != (project.get("engine") or "manual"):
        notes.append(f"按默认引擎「{engine}」录入声纹（特征空间 {embedder}）。")
    if role.get("embedding") and role.get("embedder", "builtin") != embedder:
        notes.append(
            f"该角色已有「{role['embedder']}」特征空间的声纹，将用当前引擎（{embedder}）"
            "的声纹覆盖。"
        )
        role["embedding"] = []
    extractor = diarize.make_embedder(engine, wav_path)
    if extractor is None and embedder != "builtin":
        notes.append(f"当前引擎（{engine}）的声纹提取模型不可用，本次使用内置特征空间。")
        embedder = "builtin"
    note = rolelib.enroll(role, wav_path, float(start), float(end),
                          source=project["id"], embedder=embedder, extractor=extractor)
    if note:
        notes.append(note)
    if also_library:
        _, action = add_library_role(role["name"], role["color"], role.get("embedding"),
                                     role.get("samples"), embedder=role.get("embedder", "builtin"))
        notes.append("已写入全局角色库。" if action == "created" else "已更新全局角色库记录。")
        role["type"] = "registered"
        role["in_library"] = True
    return role, notes


def _manual_samples_by_role(project: dict) -> dict[int, list[dict]]:
    """Cues a human assigned to each role, in time order (min length 0.2s)."""
    by_role: dict[int, list[dict]] = {}
    for segment in project["segments"]:
        if segment.get("status") != "manual" or segment.get("speaker_id") is None:
            continue
        if segment["end"] - segment["start"] < 0.20:
            continue
        by_role.setdefault(int(segment["speaker_id"]), []).append(segment)
    for segs in by_role.values():
        segs.sort(key=lambda s: s["start"])
    return by_role


def learn_voiceprints(project: dict, engine: str | None = None,
                      wav_path: str | None = None, max_new: int = 30) -> tuple[int, list[str]]:
    """Fold manually-assigned cues into each role's voiceprint.

    Human review is the cleanest enrolment data, so the next detection matches
    better without the user re-enrolling by hand. Only manual cues in the run's
    feature space are used, keyed by (start, end) so repeats are skipped and the
    process converges; at most ``max_new`` fresh samples per role per call.
    Returns ``(added, notes)``.
    """
    by_role = _manual_samples_by_role(project)
    if not by_role:
        return 0, []
    engine = engine or enroll_engine(project)
    space = diarize.embedder_for(engine)
    wav_path = wav_path or analysis_wav(
        project, bool(load_settings().get("separate_vocals")))[0]
    extractor = diarize.make_embedder(engine, wav_path)
    if extractor is None:
        return 0, []
    added = 0
    notes: list[str] = []
    for role in project["roles"]:
        segs = by_role.get(role["id"])
        if not segs or role.get("type") in ("ignored", "pending"):
            continue
        if role.get("embedding") and role.get("embedder", "builtin") != space:
            notes.append(f"「{role['name']}」的声纹在其它特征空间，未用复核字幕学习。")
            continue
        seen = {(round(s.get("start", 0), 3), round(s.get("end", 0), 3))
                for s in (role.get("samples") or [])}
        fresh = 0
        for segment in segs:
            key = (round(segment["start"], 3), round(segment["end"], 3))
            if key in seen or fresh >= max_new:
                continue
            try:
                rolelib.enroll(role, wav_path, segment["start"], segment["end"],
                               source="manual", embedder=space, extractor=extractor)
            except Exception:
                continue  # too short / no features: skip this cue
            seen.add(key)
            fresh += 1
        added += fresh
        if fresh and role.get("in_library"):
            add_library_role(role["name"], role["color"], role.get("embedding"),
                             role.get("samples"), embedder=role.get("embedder", "builtin"))
    if added:
        notes.append(f"已用 {added} 条人工归属字幕补充声纹。")
    return added, notes


def merge_roles(project: dict, target_id: int, source_id: int) -> dict:
    target = _find_role(project, target_id)
    source = _find_role(project, source_id)
    if target["id"] == source["id"]:
        raise ValueError("不能把角色合并到自身")
    rolelib.merge_roles(target, source)
    for segment in project["segments"]:
        if segment.get("speaker_id") == source["id"]:
            segment["speaker_id"] = target["id"]
    project["roles"] = [r for r in project["roles"] if r["id"] != source["id"]]
    return target


def _remove_roles(project: dict, dropped: set[int]) -> None:
    """Drop roles, release their cues to 未归属, and prune cluster→role mappings."""
    if not dropped:
        return
    project["roles"] = [r for r in project["roles"] if r["id"] not in dropped]
    for segment in project["segments"]:
        if segment.get("speaker_id") in dropped:
            segment["speaker_id"] = None
            segment["status"] = "pending"
            segment["confidence"] = None
    mapping = project.get("cluster_to_role") or {}
    project["cluster_to_role"] = {c: r for c, r in mapping.items() if r not in dropped}


def delete_role(project: dict, role_id: int) -> None:
    role = _find_role(project, role_id)
    _remove_roles(project, {role["id"]})


def delete_pending_roles(project: dict) -> int:
    """Delete every auto-created "待定" role; its cues return to 未归属.

    Older builds parked each unmatched cluster in a "待定角色N" role; current
    detection leaves unmatched cues 待定 instead, so this is a cleanup path for
    legacy projects (and for roles created via the roles API). Named/registered
    roles are never touched.
    """
    pending = {r["id"] for r in project["roles"] if r.get("type") == "pending"}
    _remove_roles(project, pending)
    return len(pending)


# --- export -----------------------------------------------------------------

def export(project: dict, formats: list[str], options: dict | None = None) -> list[dict]:
    options = dict(options or {})
    options.setdefault("include_pending", True)
    options.setdefault("include_confidence", True)
    options.setdefault("include_status", True)
    options.setdefault("include_role_library", True)
    options.setdefault("include_voiceprints", False)
    options.setdefault("filename_stem", project["name"])
    options["filename_stem"] = _slug(options["filename_stem"]) or "export"
    options.setdefault("text_field", "both")
    options.setdefault("include_translation", True)

    wav_path = os.path.join(WORK_DIR, f"{project['id']}.wav")
    context = {
        "video_id": project["name"],
        "media_path": project["media_path"],
        "audio_path": wav_path if os.path.exists(wav_path) else project["media_path"],
        "subtitle_path": project["subtitle_path"],
        "second_subtitle_path": project.get("second_subtitle_path"),
        "line_mode": project.get("line_mode"),
        "duration": project.get("duration"),
        "engine": project.get("engine"),
        "detection_notes": project.get("detection_notes", []),
        "stats": stats(project),
        "uri": project["name"],
    }
    output_dir = os.path.join(EXPORT_DIR, project["id"])
    written = dataset_export.write_all(
        project["segments"], project["roles"], context, options, output_dir, formats
    )
    project["exports"] = written
    save(project)
    return written


def clean_work_files(project_id: str) -> None:
    for suffix in (".wav", ".peaks.json"):
        path = os.path.join(WORK_DIR, f"{project_id}{suffix}")
        if os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass


def _owned_upload(path: str | None) -> bool:
    """True when ``path`` is a file the app copied into ``data/uploads/``.

    Only app-owned copies are ever deleted: a path the user typed (their own
    video/subtitle elsewhere on disk) must never be removed by a project action.
    """
    if not path:
        return False
    try:
        return os.path.commonpath(
            [os.path.abspath(path), os.path.abspath(UPLOAD_DIR)]
        ) == os.path.abspath(UPLOAD_DIR)
    except ValueError:
        return False


def _remove_owned_uploads(project: dict) -> int:
    """Delete this project's imported media/subtitle copies; returns the count."""
    removed = 0
    for key in ("media_path", "subtitle_path", "second_subtitle_path"):
        path = project.get(key)
        if path and _owned_upload(path) and os.path.isfile(path):
            try:
                os.remove(path)
                removed += 1
            except OSError:
                pass
    return removed


def delete_project(project_id: str) -> int:
    """Delete a project and everything it owns (uploads, exports, work files).

    Returns how many uploaded media/subtitle copies were removed. Files the user
    referenced from elsewhere on disk are left alone.
    """
    removed_uploads = 0
    try:
        removed_uploads = _remove_owned_uploads(load(project_id))
    except Exception:
        pass
    path = project_path(project_id)
    if os.path.exists(path):
        os.remove(path)
    shutil.rmtree(os.path.join(EXPORT_DIR, project_id), ignore_errors=True)
    clean_work_files(project_id)
    return removed_uploads


def clean_orphan_uploads() -> int:
    """Delete uploaded files no project references any more; returns the count.

    Uploads survive a project that was removed without cleaning up (older builds)
    or an import that failed after the files were copied in; this reclaims them.
    """
    _ensure_dirs()
    if not os.path.isdir(UPLOAD_DIR):
        return 0
    referenced: set[str] = set()
    for filename in os.listdir(PROJECT_DIR):
        if not filename.endswith(".json"):
            continue
        try:
            project = load(filename[:-5])
        except Exception:
            continue
        for key in ("media_path", "subtitle_path", "second_subtitle_path"):
            value = project.get(key)
            if value:
                referenced.add(os.path.abspath(value))
    removed = 0
    for name in os.listdir(UPLOAD_DIR):
        full = os.path.join(UPLOAD_DIR, name)
        if os.path.isfile(full) and os.path.abspath(full) not in referenced:
            try:
                os.remove(full)
                removed += 1
            except OSError:
                pass
    return removed
