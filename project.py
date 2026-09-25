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
from typing import Callable

import align as aligner
import apppaths
import diarize
import jsonutil
import media
import subtitle_io as sio

BASE_DIR = apppaths.writable_dir()
# Tests must never touch the real data directory: an explicit override lets them
# point the whole store at a throwaway directory instead of deleting `data/`.
DATA_DIR = os.environ.get("SSP_DATA_DIR") or os.path.join(BASE_DIR, "data")
# Every project is a self-contained folder so its media / work / exports can be
# deleted together and never leak into another project:
#
#     data/projects/<id>/project.json
#     data/projects/<id>/media/     app-owned copies of uploaded media + subtitles
#     data/projects/<id>/work/      decoded wav / peaks caches
#     data/projects/<id>/exports/   exported subtitles
PROJECT_DIR = os.path.join(DATA_DIR, "projects")
UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")

# Export formats this minimal build supports.
FORMAT_LABELS = {
    "srt": "SRT（带 [说话人] 前缀）",
    "ass": "ASS（按说话人上色）",
}

# --- role helpers -----------------------------------------------------------

PALETTE = [
    "#FF6B6B", "#4ECDC4", "#FFD166", "#8E7CFF", "#06D6A0",
    "#F78C6B", "#5DA9E9", "#E56B9F", "#B5E48C", "#C792EA",
]
PENDING_COLOR = "#B9BEC7"


def next_color(existing: list[str]) -> str:
    for color in PALETTE:
        if color not in existing:
            return color
    return PALETTE[len(existing) % len(PALETTE)]


def new_role(role_id: int, name: str, color: str, role_type: str = "named") -> dict:
    return {"id": role_id, "name": name, "color": color, "type": role_type}


def role_view(role: dict) -> dict:
    return {"id": role["id"], "name": role["name"], "color": role["color"],
            "type": role.get("type", "named"),
            "has_voiceprint": bool(role.get("voiceprint"))}


# --- persistence ------------------------------------------------------------

def _ensure_dirs() -> None:
    for path in (DATA_DIR, PROJECT_DIR, UPLOAD_DIR):
        os.makedirs(path, exist_ok=True)


# Parsed-JSON cache keyed by path and (mtime_ns, size). `state_payload` reads the
# project list on nearly every request; without this the same files were re-read
# and re-parsed several times per page load. Writes invalidate eagerly (mtime
# alone can collide on a same-tick rewrite of equal length).
_CACHE_LOCK = threading.RLock()
_JSON_CACHE: dict[str, tuple[tuple[int, int], object]] = {}


def _invalidate(path: str) -> None:
    with _CACHE_LOCK:
        _JSON_CACHE.pop(path, None)


def _write_json(path: str, data, indent: int | None = None) -> None:
    """Write JSON atomically: a crash mid-write must not truncate the target."""
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


def normalize_path(path: str) -> str:
    """Accept Windows / Git-Bash / quoted paths and return a native absolute path."""
    path = (path or "").strip().strip('"').strip("'")
    if not path:
        return path
    match = re.fullmatch(r"/([a-zA-Z])/(.*)", path)  # Git Bash /c/Users/... -> C:\Users\...
    if match:
        drive, rest = match.groups()
        path = f"{drive.upper()}:/{rest}"
    path = path.replace("/", os.sep)
    return os.path.abspath(path)


def _within(path: str, root: str) -> bool:
    """True when ``path`` is inside ``root`` (not merely a name-prefix match)."""
    try:
        return os.path.commonpath(
            [os.path.abspath(path), os.path.abspath(root)]
        ) == os.path.abspath(root)
    except ValueError:
        return False


def project_dir(project_id: str) -> str:
    """The self-contained folder that owns every file this project creates."""
    return os.path.join(PROJECT_DIR, project_id)


def project_path(project_id: str) -> str:
    return os.path.join(project_dir(project_id), "project.json")


def project_media_dir(project_id: str) -> str:
    return os.path.join(project_dir(project_id), "media")


def project_work_dir(project_id: str) -> str:
    return os.path.join(project_dir(project_id), "work")


def project_export_dir(project_id: str) -> str:
    return os.path.join(project_dir(project_id), "exports")


def _unique_in(directory: str, name: str) -> str:
    """A path under ``directory`` that does not collide with an existing file."""
    target = os.path.join(directory, name)
    if os.path.exists(target):
        stem, ext = os.path.splitext(name)
        target = os.path.join(directory, f"{stem}_{uuid.uuid4().hex[:6]}{ext}")
    return target


def _is_staged(path: str) -> bool:
    """True when ``path`` is an app-owned file still in the upload staging dir."""
    return _within(path, UPLOAD_DIR)


def _adopt_staged_file(project_id: str, path: str | None) -> str | None:
    """Move an uploaded file out of staging and into the project folder.

    Uploads are written before the project id exists; adopting them on import is
    what makes each project self-contained.
    """
    if not path or not _is_staged(path) or not os.path.isfile(path):
        return path
    destination = project_media_dir(project_id)
    os.makedirs(destination, exist_ok=True)
    target = _unique_in(destination, os.path.basename(path))
    try:
        shutil.move(path, target)
        _invalidate(path)
        return target
    except OSError:
        return path


_SUMMARY_CACHE: dict[str, tuple[tuple[int, int], dict]] = {}


def save(project: dict) -> dict:
    _ensure_dirs()
    project["updated"] = datetime.now().isoformat(timespec="seconds")
    os.makedirs(project_dir(project["id"]), exist_ok=True)
    _write_json(project_path(project["id"]), jsonutil.json_safe(project))
    with _CACHE_LOCK:
        _SUMMARY_CACHE.pop(project["id"], None)
    return project


def load(project_id: str) -> dict:
    with open(project_path(project_id), "r", encoding="utf-8-sig") as handle:
        return jsonutil.json_safe(json.load(handle))


def _project_json_paths() -> list[tuple[str, str]]:
    """``(project_id, project.json path)`` for every project on disk."""
    _ensure_dirs()
    found: list[tuple[str, str]] = []
    for name in os.listdir(PROJECT_DIR):
        directory = os.path.join(PROJECT_DIR, name)
        if os.path.isdir(directory):
            path = os.path.join(directory, "project.json")
            if os.path.isfile(path):
                found.append((name, path))
    return found


def stats(project: dict) -> dict:
    segments = project.get("segments", [])
    assigned = sum(1 for s in segments if s.get("speaker_id") is not None)
    manual = sum(1 for s in segments if s.get("status") == "manual")
    auto = sum(1 for s in segments if s.get("status") == "auto")
    return {
        "total": len(segments),
        "assigned": assigned,
        "pending": len(segments) - assigned,
        "auto": auto,
        "manual": manual,
        "roles": len(project.get("roles", [])),
        "speakers_used": len({s.get("speaker_id") for s in segments
                              if s.get("speaker_id") is not None}),
    }


def list_projects() -> list[dict]:
    """Summarise every project, parsing only files whose (mtime, size) changed."""
    _ensure_dirs()
    items: list[dict] = []
    seen: set[str] = set()
    for project_id, path in _project_json_paths():
        try:
            stat = os.stat(path)
        except OSError:
            continue
        seen.add(project_id)
        key = (stat.st_mtime_ns, stat.st_size)
        with _CACHE_LOCK:
            hit = _SUMMARY_CACHE.get(project_id)
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
            _SUMMARY_CACHE[project_id] = (key, summary)
        items.append(copy.deepcopy(summary))
    with _CACHE_LOCK:
        for name in list(_SUMMARY_CACHE):
            if name not in seen:
                _SUMMARY_CACHE.pop(name, None)
    items.sort(key=lambda item: item.get("updated") or "", reverse=True)
    return items


# --- project construction ---------------------------------------------------


def preview_import(media_path: str, subtitle_path: str) -> dict:
    media_path = normalize_path(media_path)
    subtitle_path = normalize_path(subtitle_path)
    if not os.path.isfile(media_path):
        raise FileNotFoundError(f"找不到媒体文件：{media_path}")
    if not os.path.isfile(subtitle_path):
        raise FileNotFoundError(f"找不到字幕文件：{subtitle_path}")
    segments, meta = sio.parse_subtitle(subtitle_path)
    if not segments:
        raise ValueError("字幕文件解析结果为空，请检查文件格式。")
    duration = media.probe_duration(media_path)
    subtitle_end = max(segment["end"] for segment in segments)
    notes = []
    if duration and subtitle_end > duration + 3:
        notes.append(f"字幕结束时间超过媒体时长 {round(subtitle_end - duration, 1)} 秒，请检查是否配对正确。")
    skipped = int(meta.get("skipped_effects") or 0)
    if skipped:
        notes.append(f"已过滤 {skipped} 条非对白事件。")
    duplicates = sum(1 for left, right in zip(segments, segments[1:])
                     if left["start"] == right["start"] and left["end"] == right["end"]
                     and left["text"] == right["text"])
    if duplicates:
        notes.append(f"发现 {duplicates} 条时间与内容完全相同的字幕。")
    return {"media_name": os.path.basename(media_path),
            "subtitle_name": os.path.basename(subtitle_path),
            "media_duration": round(duration, 2), "subtitle_end": round(subtitle_end, 2),
            "segments": len(segments), "notes": notes}

def _next_role_id(project: dict) -> int:
    used = [role["id"] for role in project["roles"]]
    return (max(used) + 1) if used else 0


def create(media_path: str, subtitle_path: str, name: str | None = None) -> dict:
    """Import a project from a media file + an existing subtitle file."""
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

    project_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    # Pull browser uploads out of staging into this project's folder so the files
    # belong to it alone (delete-project then removes them too).
    media_path = _adopt_staged_file(project_id, media_path)
    subtitle_path = _adopt_staged_file(project_id, subtitle_path)
    kind = media.kind_of(media_path)
    project = {
        "id": project_id,
        "name": _slug(name) if name else _slug(os.path.splitext(os.path.basename(media_path))[0]),
        "media_path": os.path.abspath(media_path),
        "media_kind": kind if kind != "unknown" else "video",
        "subtitle_path": os.path.abspath(subtitle_path),
        "subtitle_format": os.path.splitext(subtitle_path)[1].lstrip(".").lower(),
        "import_notes": notes,
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
            new_role(_next_role_id(project), hint,
                     next_color([r["color"] for r in project["roles"]]), "named")
        )
    return save(project)


def work_wav(project: dict) -> str:
    """Decode (and cache) the project audio as mono 16 kHz WAV."""
    _ensure_dirs()
    directory = project_work_dir(project["id"])
    os.makedirs(directory, exist_ok=True)
    target = os.path.join(directory, f"{project['id']}.wav")
    source = project["media_path"]
    if os.path.exists(target) and os.path.getmtime(target) >= os.path.getmtime(source):
        return target
    return media.to_wav16k(source, target)


# --- detection + alignment --------------------------------------------------

def _assign_roles(project: dict, cluster_ids: list[int]) -> tuple[dict[int, int], int]:
    """First detection: map clusters onto roles positionally (cluster 0 -> role 0).

    A role per cluster, in order; there is nothing to anchor on yet, so this is
    the unavoidable guess. Existing roles are reused positionally and any extra
    cluster gets a fresh grey deferred role the user can rename.
    """
    mapping: dict[int, int] = {}
    created = 0
    for cluster_id in cluster_ids:
        index = len(mapping)
        if index < len(project["roles"]):
            mapping[cluster_id] = project["roles"][index]["id"]
            continue
        existing = {r["name"] for r in project["roles"]}
        number = len(project["roles"]) + 1
        name = f"Character{number}"
        while name in existing:
            number += 1
            name = f"Character{number}"
        role = new_role(_next_role_id(project), name,
                        next_color([r["color"] for r in project["roles"]]), "pending")
        project["roles"].append(role)
        mapping[cluster_id] = role["id"]
        created += 1
    return mapping, created


def _previous_cluster_map(project: dict) -> dict | None:
    """The last detection's ``turns`` + cluster→role map (still-valid roles only).

    With it, a re-run can line new clusters up with the old ones by how their
    speech overlaps in time — far more stable than cluster *order*, which
    pyannote does not guarantee between runs.
    """
    diarization = project.get("diarization") or {}
    turns = [t for t in (diarization.get("turns") or []) if t.get("cluster") is not None]
    role_ids = {r["id"] for r in project["roles"]}
    old_map = {str(k): v for k, v in (project.get("cluster_to_role") or {}).items()}
    old_map = {k: v for k, v in old_map.items() if v in role_ids}
    if not turns or not old_map:
        return None
    return {"turns": turns, "map": old_map}


def _match_clusters_by_time(new_turns: list[dict], old_turns: list[dict],
                            old_map: dict[str, int]) -> dict[int, int]:
    """Greedy one-to-one match of new clusters to old clusters by overlap seconds.

    For every (new, old) cluster pair, accumulate the total speaker-turn overlap
    (a returned mapping covers old keys stored in ``cluster_to_role`` — string
    keys — and both are handled here). Maps it. One new cluster takes at most one
    old cluster and vice versa, always taking the largest overlap first.
    """
    old_by_cluster: dict[str, list[dict]] = {}
    for turn in old_turns:
        old_by_cluster.setdefault(str(int(turn["cluster"])), []).append(turn)
    new_by_cluster: dict[int, list[dict]] = {}
    for turn in new_turns:
        new_by_cluster.setdefault(int(turn["cluster"]), []).append(turn)

    overlap: dict[tuple[int, str], float] = {}
    for new_id, nturns in new_by_cluster.items():
        for nturn in nturns:
            for key, oturns in old_by_cluster.items():
                for oturn in oturns:
                    seconds = (min(nturn["end"], oturn["end"])
                               - max(nturn["start"], oturn["start"]))
                    if seconds > 0:
                        overlap[(new_id, key)] = overlap.get((new_id, key), 0.0) + seconds

    mapping: dict[int, int] = {}
    used_new: set[int] = set()
    used_old: set[str] = set()
    for (new_id, key), _seconds in sorted(overlap.items(), key=lambda kv: -kv[1]):
        if new_id in used_new or key in used_old:
            continue
        role_id = old_map.get(key)
        if role_id is None:
            continue
        mapping[new_id] = role_id
        used_new.add(new_id)
        used_old.add(key)
    return mapping


def _match_clusters_by_voiceprint(project: dict, wav_path: str,
                                  turns: list[dict]) -> tuple[dict[int, int], list[str]]:
    """Map detected clusters to enrolled roles using cosine similarity."""
    import math
    profiles = [(role, role.get("voiceprint", {}).get("vector"))
                for role in project.get("roles", [])]
    profiles = [(role, vector) for role, vector in profiles if vector]
    if not profiles:
        return {}, []
    by_cluster: dict[int, list[dict]] = {}
    for turn in turns:
        if turn.get("cluster") is not None and turn["end"] - turn["start"] >= 1.0:
            by_cluster.setdefault(int(turn["cluster"]), []).append(turn)
    notes: list[str] = []
    cluster_vectors: dict[int, list[list[float]]] = {}
    for cluster, candidates in by_cluster.items():
        vectors = []
        for turn in sorted(candidates, key=lambda item: item["end"] - item["start"], reverse=True)[:4]:
            try:
                vectors.append(diarize.extract_voice_embedding(wav_path, turn["start"], turn["end"]))
            except Exception:
                continue
        if vectors:
            cluster_vectors[cluster] = vectors

    # Resolve all cluster/role scores together. Greedy global assignment keeps
    # one noisy cluster from consuming the same role that another cluster fits
    # better, which was a major source of unmapped subtitles.
    candidates: list[tuple[float, float, int, int]] = []
    for cluster, vectors in cluster_vectors.items():
        scores = []
        for role, profile in profiles:
            score = max(sum(a * b for a, b in zip(vector, profile))
                        for vector in vectors)
            scores.append((score, role["id"]))
        scores.sort(reverse=True)
        if not scores:
            continue
        best, best_role = scores[0]
        second = scores[1][0] if len(scores) > 1 else 0.0
        candidates.append((best, best - second, cluster, best_role))

    mapping: dict[int, int] = {}
    used_roles: set[int] = set()
    for best, margin, cluster, role_id in sorted(candidates, reverse=True):
        if best < 0.62 or margin < 0.06:
            notes.append(f"聚类 {cluster + 1} 的声纹匹配不确定，保持待定。")
            continue
        if role_id in used_roles:
            notes.append(f"聚类 {cluster + 1} 与已分配角色冲突，保持待定。")
            continue
        mapping[cluster] = role_id
        used_roles.add(role_id)
    if mapping:
        notes.insert(0, f"已用角色声纹校正 {len(mapping)} 个聚类映射。")
    return mapping, notes


def _conflicted_clusters(project: dict, turns: list[dict]) -> set[int]:
    """Find clusters that human corrections show contain several speakers."""
    from collections import Counter

    votes: dict[int, Counter] = {}
    for segment in project["segments"]:
        if segment.get("status") != "manual":
            continue
        role_id = segment.get("speaker_id")
        if role_id is None:
            continue
        overlaps = aligner.overlap_by_cluster(segment, turns)
        if overlaps:
            cluster = max(overlaps, key=overlaps.get)
            votes.setdefault(int(cluster), Counter())[int(role_id)] += 1
    return {cluster for cluster, counts in votes.items()
            if len(counts) >= 2 and sum(counts.values()) >= 4
            and counts.most_common(2)[1][1] >= 2}


def realign_detection(project: dict) -> dict:
    """Reapply the current detection while preserving manual decisions."""
    turns = (project.get("diarization") or {}).get("turns") or []
    mapping = {int(cluster): int(role) for cluster, role in
               (project.get("cluster_to_role") or {}).items()}
    conflicts = _conflicted_clusters(project, turns)
    names = {role["name"]: role["id"] for role in project["roles"]}
    project["segments"], notes = aligner.align(
        project["segments"], turns, mapping, name_to_role=names,
        overwrite_manual=False,
        auto_min_confidence=0.85 if project.get("mapping_conservative", False) else 0.70,
        reject_ambiguous=True, conflicted_clusters=conflicts,
    )
    if conflicts:
        labels = "、".join(str(cluster + 1) for cluster in sorted(conflicts))
        notes.insert(0, f"聚类 {labels} 含多位人工确认的说话人，已暂停整簇自动归属。")
    project["detection_notes"] = notes
    return save(project)


def run_detection(project: dict, engine: str = "pyannote", min_speakers: int = 1,
                  max_speakers: int = 6, overwrite_manual: bool = False,
                  conservative: bool = False,
                  progress: Callable[[str], None] | None = None) -> dict:
    if engine not in ("pyannote", "manual"):
        engine = "pyannote"

    previous = _previous_cluster_map(project)
    notes: list[str] = []
    if engine == "manual":
        result = diarize.run_manual()
        cluster_ids: list[int] = []
    else:
        if progress:
            progress("正在解码音频")
        wav_path = work_wav(project)
        if progress:
            progress("正在检测说话人")
        result = diarize.run(engine, wav_path, max(1, int(min_speakers)),
                             max(1, int(max_speakers)))
        cluster_ids = sorted(int(c) for c in result.get("clusters", {}))

    if progress:
        progress("正在对齐字幕")
    project["diarization"] = result
    project["engine"] = result["engine"]
    for segment in project["segments"]:
        segment.pop("vp_score", None)
    notes.extend(result.get("notes", []))

    mapping: dict[int, int] = {}
    if cluster_ids:
        if previous:
            mapping = _match_clusters_by_time(result.get("turns", []), previous["turns"],
                                              previous["map"])
            if mapping:
                notes.append(
                    f"{len(mapping)} clusters wrapped onto existing roles via time overlap from the previous detection."
                )
            unmatched = len(cluster_ids) - len(mapping)
            if unmatched:
                notes.append(
                    f"{unmatched} clusters did not match the previous detection, their lines remain pending "
                    "(renaming or merging roles then re-detecting can correct them)."
                )
        else:
            notes.append("检测完成，请先试听各聚类并确认角色映射。")

    if engine == "pyannote" and project.get("roles"):
        try:
            voice_mapping, voice_notes = _match_clusters_by_voiceprint(
                project, work_wav(project), result.get("turns", []))
            if voice_mapping:
                mapping.update(voice_mapping)
            notes.extend(voice_notes)
        except Exception as exc:
            notes.append(f"声纹校正未启用：{type(exc).__name__}。")

    # A voiceprint-backed mapping is already a reviewed suggestion. Only open
    # the confirmation panel when at least one cluster still needs a decision.
    project["mapping_pending"] = bool(cluster_ids and len(mapping) < len(cluster_ids))
    if cluster_ids and mapping and not project["mapping_pending"]:
        notes.append("所有聚类均已通过声纹映射，无需再次确认。")

    name_to_role = {role["name"]: role["id"] for role in project["roles"]}
    project["segments"], align_notes = aligner.align(
        project["segments"], result.get("turns", []), mapping,
        name_to_role=name_to_role, overwrite_manual=overwrite_manual,
        auto_min_confidence=0.85 if conservative else 0.70,
        reject_ambiguous=True,
        conflicted_clusters=_conflicted_clusters(project, result.get("turns", [])),
    )
    notes.extend(align_notes)
    project["cluster_to_role"] = mapping
    project["mapping_overwrite_manual"] = bool(overwrite_manual)
    project["mapping_conservative"] = bool(conservative)
    project["detection_notes"] = notes
    return save(project)


def confirm_cluster_mapping(project: dict, choices: dict) -> dict:
    """Apply a reviewed cluster-to-role map to the last detection."""
    if not project.get("mapping_pending"):
        raise ValueError("当前项目没有待确认的聚类映射")
    turns = (project.get("diarization") or {}).get("turns") or []
    cluster_ids = {int(turn["cluster"]) for turn in turns if turn.get("cluster") is not None}
    if set(choices) != {str(cluster) for cluster in cluster_ids}:
        raise ValueError("请为每个聚类选择角色或保持待定")
    role_ids = {role["id"] for role in project["roles"]}
    mapping: dict[int, int] = {}
    for key, choice in choices.items():
        cluster = int(key)
        if choice is None:
            continue
        if choice == "new":
            number = len(project["roles"]) + 1
            names = {role["name"] for role in project["roles"]}
            while f"角色{number}" in names:
                number += 1
            role = new_role(_next_role_id(project), f"角色{number}",
                            next_color([r["color"] for r in project["roles"]]), "pending")
            project["roles"].append(role)
            mapping[cluster] = role["id"]
        else:
            role_id = int(choice)
            if role_id not in role_ids:
                raise ValueError(f"角色 {role_id} 不存在")
            mapping[cluster] = role_id
    names = {role["name"]: role["id"] for role in project["roles"]}
    project["segments"], notes = aligner.align(
        project["segments"], turns, mapping, name_to_role=names,
        overwrite_manual=project.get("mapping_overwrite_manual", False),
        auto_min_confidence=0.85 if project.get("mapping_conservative", False) else 0.70,
        reject_ambiguous=True,
        conflicted_clusters=_conflicted_clusters(project, turns),
    )
    project["cluster_to_role"] = mapping
    project["mapping_pending"] = False
    project.pop("mapping_overwrite_manual", None)
    project.pop("mapping_conservative", None)
    project["detection_notes"] = ["聚类角色映射已确认。", *notes]
    return save(project)


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
            # None (every hand-assignment path) clears the detector's score;
            # undo/redo passes the snapshot value back so an auto assignment
            # keeps its confidence badge.
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


# --- uploads -----------------------------------------------------------------

_MEDIA_EXT = set(media._VIDEO_EXTENSIONS) | set(media._AUDIO_EXTENSIONS)
_SUB_EXT = {".srt", ".ass", ".ssa"}


def _upload_target(name: str) -> str:
    """Reserve a safe, unique path under data/uploads/ (no bytes written yet)."""
    _ensure_dirs()
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


# --- role operations --------------------------------------------------------

def _find_role(project: dict, role_id: int) -> dict:
    for role in project["roles"]:
        if role["id"] == int(role_id):
            return role
    raise KeyError(f"角色 {role_id} 不存在")


def add_role(project: dict, name: str, color: str | None = None) -> dict:
    if any(role["name"] == name for role in project["roles"]):
        raise ValueError(f"角色「{name}」已存在")
    role = new_role(_next_role_id(project), name,
                    color or next_color([r["color"] for r in project["roles"]]), "named")
    project["roles"].append(role)
    return role


def update_role(project: dict, role_id: int, name: str | None = None,
                color: str | None = None) -> dict:
    role = _find_role(project, role_id)
    if name is not None and name != role["name"]:
        role["name"] = name
        if role.get("type") == "pending":
            role["type"] = "named"
            if role["color"] == PENDING_COLOR:
                role["color"] = next_color([r["color"] for r in project["roles"]
                                            if r["id"] != role["id"]])
    if color is not None:
        role["color"] = color
    return role


def enroll_role_voiceprint(project: dict, role_id: int,
                           segment_ids: list[int] | None = None) -> dict:
    """Build a role profile from manually confirmed subtitle segments."""
    role = _find_role(project, role_id)
    chosen = set(int(value) for value in (segment_ids or []))
    segments = [segment for segment in project["segments"]
                if segment.get("speaker_id") == role_id and segment.get("status") == "manual"
                and (not chosen or segment["id"] in chosen)
                and segment["end"] - segment["start"] >= 0.8]
    if len(segments) < 2:
        raise ValueError("请先人工确认至少 2 条、每条超过 0.8 秒的字幕")
    wav_path = work_wav(project)
    vectors = [diarize.extract_voice_embedding(wav_path, segment["start"], segment["end"])
               for segment in segments[:20]]
    import math
    width = len(vectors[0])
    centroid = [sum(vector[index] for vector in vectors) / len(vectors) for index in range(width)]
    norm = math.sqrt(sum(value * value for value in centroid)) or 1.0
    role["voiceprint"] = {
        "vector": [round(value / norm, 7) for value in centroid],
        "samples": len(vectors),
        "updated": datetime.now().isoformat(timespec="seconds"),
    }
    return role


def merge_roles(project: dict, target_id: int, source_id: int) -> dict:
    target = _find_role(project, target_id)
    source = _find_role(project, source_id)
    if target["id"] == source["id"]:
        raise ValueError("不能把角色合并到自身")
    for segment in project["segments"]:
        if segment.get("speaker_id") == source["id"]:
            segment["speaker_id"] = target["id"]
    project["roles"] = [r for r in project["roles"] if r["id"] != source["id"]]
    mapping = project.get("cluster_to_role") or {}
    project["cluster_to_role"] = {c: r for c, r in mapping.items() if r != source["id"]}
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


# --- export -----------------------------------------------------------------

def export(project: dict, formats: list[str], options: dict | None = None) -> list[dict]:
    options = dict(options or {})
    include_pending = bool(options.get("include_pending", True))
    stem = _slug(options.get("filename_stem") or project["name"]) or "export"
    names = {role["id"]: role["name"] for role in project["roles"]}
    colors = {role["id"]: role["color"] for role in project["roles"]}
    quality = {
        "pending": sum(1 for seg in project["segments"] if seg.get("speaker_id") is None),
        "review": sum(1 for seg in project["segments"] if seg.get("status") == "auto"
                      and (seg.get("ambiguous") or (seg.get("confidence") is not None
                          and seg["confidence"] < 0.85))),
        "unnamed_roles": [role["name"] for role in project["roles"]
                          if role.get("type") == "pending"],
        "mapping_pending": bool(project.get("mapping_pending")),
    }

    output_dir = project_export_dir(project["id"])
    os.makedirs(output_dir, exist_ok=True)
    written: list[dict] = []
    for name in formats:
        if name == "srt":
            content = sio.write_srt(project["segments"], names, include_pending)
            extension, mime = ".srt", "text/plain"
        elif name == "ass":
            content = sio.write_ass(project["segments"], names, colors, include_pending)
            extension, mime = ".ass", "text/plain"
        else:
            continue
        filename = f"{stem}_{name}{extension}"
        path = os.path.join(output_dir, filename)
        with open(path, "w", encoding="utf-8", newline="") as handle:
            handle.write(content)
        written.append({
            "format": name,
            "label": FORMAT_LABELS.get(name, name),
            "filename": filename,
            "path": path,
            "mime": mime,
            "bytes": len(content.encode("utf-8")),
            "quality": quality,
        })
    project["exports"] = written
    save(project)
    return written


# --- deletion ---------------------------------------------------------------

def _owned_upload(project: dict, path: str | None) -> bool:
    """True when ``path`` is an app-owned copy (project folder or staging)."""
    if not path:
        return False
    roots = (project_media_dir(project["id"]), UPLOAD_DIR)
    return any(_within(path, root) for root in roots)


def _remove_owned_uploads(project: dict) -> int:
    """Delete this project's staged media/subtitle copies; returns the count."""
    removed = 0
    for key in ("media_path", "subtitle_path"):
        path = project.get(key)
        if path and _is_staged(path) and os.path.isfile(path):
            try:
                os.remove(path)
                removed += 1
            except OSError:
                pass
    return removed


def delete_project(project_id: str) -> int:
    """Delete a project and everything it owns (folder, media, work, exports)."""
    removed_uploads = 0
    try:
        removed_uploads = _remove_owned_uploads(load(project_id))
    except Exception:
        pass
    directory = project_dir(project_id)
    if os.path.isdir(directory):
        shutil.rmtree(directory, ignore_errors=True)
    with _CACHE_LOCK:
        _SUMMARY_CACHE.pop(project_id, None)
    return removed_uploads


def clean_orphan_uploads() -> int:
    """Delete staged uploads no project references; returns the count."""
    _ensure_dirs()
    referenced: set[str] = set()
    for project_id, _path in _project_json_paths():
        try:
            project = load(project_id)
        except Exception:
            continue
        for key in ("media_path", "subtitle_path"):
            value = project.get(key)
            if value:
                referenced.add(os.path.abspath(value))
    removed = 0
    if os.path.isdir(UPLOAD_DIR):
        for name in os.listdir(UPLOAD_DIR):
            full = os.path.join(UPLOAD_DIR, name)
            if os.path.isfile(full) and os.path.abspath(full) not in referenced:
                try:
                    os.remove(full)
                    removed += 1
                except OSError:
                    pass
    return removed


def reset_all() -> dict:
    """Delete every project (with its files) and clear staging."""
    _ensure_dirs()
    projects = 0
    for project_id, _path in _project_json_paths():
        delete_project(project_id)
        projects += 1
    uploads = clean_orphan_uploads()
    return {"projects": projects, "uploads": uploads}
