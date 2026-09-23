"""Exporters: classic subtitle formats plus dataset formats for training use.

All writers share one signature so the HTTP layer can loop over them::

    writer(segments, roles, context, options) -> str

``context`` carries media paths and ids; ``options`` carries the export panel
toggles (include pending cues, confidence, review status, ...).
"""

from __future__ import annotations

import csv
import io
import json
import os
import re

import subtitle_io as sio


def _role_maps(roles: list[dict]) -> tuple[dict[int, str], dict[int, str], dict[int, dict]]:
    names = {role["id"]: role["name"] for role in roles}
    colors = {role["id"]: role["color"] for role in roles}
    by_id = {role["id"]: role for role in roles}
    return names, colors, by_id


def _visible(segments: list[dict], options: dict) -> list[dict]:
    """Apply the export panel toggles."""
    include_pending = options.get("include_pending", True)
    only_assigned = options.get("only_assigned", False)
    output = []
    for segment in segments:
        assigned = segment.get("speaker_id") is not None
        if only_assigned and not assigned:
            continue
        if not assigned and not include_pending:
            continue
        output.append(segment)
    return output


def _text_fields(segment: dict, options: dict) -> dict:
    """text/translation payload honouring the export panel's language toggle."""
    field = options.get("text_field", "both")
    primary = segment.get("text", "")
    translation = (segment.get("translation") or "").strip()
    payload = {}
    if field == "translation":
        payload["text"] = translation
        payload["source_text"] = primary
    else:
        payload["text"] = primary
        if translation and (field == "both" or options.get("include_translation", True)):
            payload["translation"] = translation
    return payload


def _seconds(value) -> float:
    return round(float(value), 3)


# --- classic subtitle formats ----------------------------------------------

def export_srt(segments, roles, context, options):
    names, _, _ = _role_maps(roles)
    return sio.write_srt(
        _visible(segments, options), names,
        include_pending=options.get("include_pending", True),
        text_field=options.get("text_field", "both"),
    )


def export_vtt(segments, roles, context, options):
    names, _, _ = _role_maps(roles)
    return sio.write_vtt(
        _visible(segments, options), names,
        include_pending=options.get("include_pending", True),
        text_field=options.get("text_field", "both"),
    )


def export_ass(segments, roles, context, options):
    names, colors, _ = _role_maps(roles)
    return sio.write_ass(
        _visible(segments, options), names, colors,
        include_pending=options.get("include_pending", True),
        text_field=options.get("text_field", "both"),
    )


def export_script(segments, roles, context, options):
    """Screenplay style: one line per cue as ``说话人：台词``.

    Unassigned cues fall back to a neutral label; a bilingual translation
    stays on an indented follow-up line under the speaker tag.
    """
    names, _, _ = _role_maps(roles)
    text_field = options.get("text_field", "both")
    unknown = options.get("unknown_speaker_label") or "（待定）"
    lines = []
    for segment in _visible(segments, options):
        name = names.get(segment.get("speaker_id")) or unknown
        body = sio.compose_text(segment, text_field)
        indent = " " * (len(name) + 1)
        lines.append(name + "：" + body.replace("\n", "\n" + indent))
    return "\n".join(lines) + ("\n" if lines else "")


# --- dataset formats --------------------------------------------------------

def export_jsonl(segments, roles, context, options):
    names, _, _ = _role_maps(roles)
    with_confidence = options.get("include_confidence", True)
    with_status = options.get("include_status", True)
    with_speaker_id = options.get("include_speaker_id", True)
    lines = []
    for index, segment in enumerate(_visible(segments, options)):
        record = {
            "id": index,
            "start": _seconds(segment["start"]),
            "end": _seconds(segment["end"]),
            "duration": _seconds(segment["end"] - segment["start"]),
            "speaker": names.get(segment.get("speaker_id")),
        }
        record.update(_text_fields(segment, options))
        if with_speaker_id:
            record["speaker_id"] = segment.get("speaker_id")
        if with_confidence:
            record["confidence"] = segment.get("confidence")
        if with_status:
            record["status"] = segment.get("status")
        lines.append(json.dumps(record, ensure_ascii=False))
    return "\n".join(lines) + ("\n" if lines else "")


def export_rttm(segments, roles, context, options):
    """NIST RTTM. Speaker name goes in the 8th column."""
    names, _, _ = _role_maps(roles)
    uri = (context.get("uri") or "media").replace(" ", "_").replace("\t", "_")
    lines = []
    for segment in _visible(segments, options):
        if segment.get("speaker_id") is None:
            continue  # RTTM has no representation for an unlabelled region
        name = names.get(segment["speaker_id"], "unknown").replace(" ", "_")
        duration = segment["end"] - segment["start"]
        lines.append(
            f"SPEAKER {uri} 1 {segment['start']:.3f} {duration:.3f} "
            f"<NA> <NA> {name} <NA> <NA>"
        )
    return "\n".join(lines) + ("\n" if lines else "")


_CSV_FIELDS = ["id", "start", "end", "duration", "text", "translation", "source_text",
               "speaker", "speaker_id", "confidence", "status"]


def export_csv(segments, roles, context, options):
    names, _, _ = _role_maps(roles)
    with_confidence = options.get("include_confidence", True)
    with_status = options.get("include_status", True)
    text_field = options.get("text_field", "both")

    def columns():
        fields = []
        for field in _CSV_FIELDS:
            if field == "confidence" and not with_confidence:
                continue
            if field == "status" and not with_status:
                continue
            if field == "translation" and text_field == "primary":
                continue
            if field == "source_text" and text_field != "translation":
                continue
            fields.append(field)
        return fields

    fields = columns()
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(fields)
    for index, segment in enumerate(_visible(segments, options)):
        payload = _text_fields(segment, options)
        row = {
            "id": index,
            "start": f"{segment['start']:.3f}",
            "end": f"{segment['end']:.3f}",
            "duration": f"{segment['end'] - segment['start']:.3f}",
            "text": payload.get("text", ""),
            "translation": payload.get("translation", ""),
            "source_text": payload.get("source_text", ""),
            "speaker": names.get(segment.get("speaker_id")) or "",
            "speaker_id": segment.get("speaker_id") if segment.get("speaker_id") is not None else "",
            "confidence": segment.get("confidence") if segment.get("confidence") is not None else "",
            "status": segment.get("status", ""),
        }
        writer.writerow([row[f] for f in fields])
    return buffer.getvalue()


def export_hf(segments, roles, context, options):
    """HuggingFace-datasets-friendly JSON: one record per media file."""
    names, _, _ = _role_maps(roles)
    with_confidence = options.get("include_confidence", True)
    with_status = options.get("include_status", True)
    items = []
    for index, segment in enumerate(_visible(segments, options)):
        item = {
            "id": index,
            "start": _seconds(segment["start"]),
            "end": _seconds(segment["end"]),
            "speaker": names.get(segment.get("speaker_id")),
        }
        item.update(_text_fields(segment, options))
        if with_confidence:
            item["confidence"] = segment.get("confidence")
        if with_status:
            item["status"] = segment.get("status")
        items.append(item)
    record = {
        "audio": context.get("audio_path") or context.get("media_path"),
        "video": context.get("media_path"),
        "subtitle": context.get("subtitle_path"),
        "num_segments": len(items),
        "speakers": sorted({i["speaker"] for i in items if i["speaker"]}),
        "segments": items,
    }
    return json.dumps(record, ensure_ascii=False, indent=2)


def export_dataset(segments, roles, context, options):
    """Custom dataset format: annotations plus the full role/speaker library."""
    names, _, by_id = _role_maps(roles)
    include_role_library = options.get("include_role_library", True)
    include_voiceprints = options.get("include_voiceprints", False)

    annotations = []
    for index, segment in enumerate(_visible(segments, options)):
        role = by_id.get(segment.get("speaker_id"))
        record = {
            "id": index,
            "start": _seconds(segment["start"]),
            "end": _seconds(segment["end"]),
            "speaker_id": segment.get("speaker_id"),
            "speaker": names.get(segment.get("speaker_id")),
        }
        record.update(_text_fields(segment, options))
        if options.get("include_confidence", True):
            record["confidence"] = segment.get("confidence")
        if options.get("include_status", True):
            record["status"] = segment.get("status")
            record["reviewed"] = segment.get("status") == "manual"
        if role and role["type"] == "pending":
            record["speaker_type"] = "pending"
        annotations.append(record)

    payload = {
        "video_id": context.get("video_id"),
        "media_path": context.get("media_path"),
        "audio_path": context.get("audio_path"),
        "subtitle_path": context.get("subtitle_path"),
        "duration": context.get("duration"),
        "detection_engine": context.get("engine"),
        "detection_notes": context.get("detection_notes", []),
        "stats": context.get("stats", {}),
        "annotations": annotations,
    }
    if include_role_library:
        speakers = []
        for role in roles:
            entry = {
                "speaker_id": role["id"],
                "name": role["name"],
                "color": role["color"],
                "type": role["type"],
                "sample_count": len(role.get("samples", [])),
            }
            if include_voiceprints and role.get("embedding"):
                entry["embedding"] = [round(v, 6) for v in role["embedding"]]
            if role.get("samples"):
                entry["samples"] = role["samples"]
            speakers.append(entry)
        payload["speakers"] = speakers
    return json.dumps(payload, ensure_ascii=False, indent=2)


def export_report(segments, roles, context, options):
    """Human-readable review report: what was auto-assigned vs changed by hand."""
    names, _, by_id = _role_maps(roles)
    stats = context.get("stats", {})
    lines = [
        "# 说话人归属复核报告",
        "",
        f"- 媒体文件：`{context.get('media_path')}`",
        f"- 字幕文件：`{context.get('subtitle_path')}`",
    ]
    if context.get("second_subtitle_path"):
        lines.append(f"- 翻译字幕：`{context['second_subtitle_path']}`")
    if context.get("line_mode"):
        lines.append(f"- 多行文本处理：`{context['line_mode']}`")
    lines += [
        f"- 检测引擎：`{context.get('engine')}`",
        f"- 字幕总条数：{stats.get('total', len(segments))}",
        f"- 已归属：{stats.get('assigned', 0)}",
        f"- 待定：{stats.get('pending', 0)}",
        f"- 自动归属：{stats.get('auto', 0)}",
        f"- 人工确认/修改：{stats.get('manual', 0)}",
        "",
        "## 说话人统计",
        "",
        "| 说话人 | 类型 | 声纹样本 | 字幕条数 | 总时长(s) |",
        "| --- | --- | --- | --- | --- |",
    ]
    counts: dict[int | None, list[float]] = {}
    for segment in segments:
        bucket = counts.setdefault(segment.get("speaker_id"), [0, 0.0])
        bucket[0] += 1
        bucket[1] += segment["end"] - segment["start"]
    for role in roles:
        hit = counts.get(role["id"], [0, 0.0])
        lines.append(
            f"| {role['name']} | {role['type']} | {len(role.get('samples', []))} | "
            f"{hit[0]} | {hit[1]:.2f} |"
        )
    unknown = counts.get(None)
    if unknown:
        lines.append(f"| （待定/无归属） | pending | 0 | {unknown[0]} | {unknown[1]:.2f} |")

    if context.get("detection_notes"):
        lines += ["", "## 检测与对齐说明", ""]
        lines += [f"- {note}" for note in context["detection_notes"]]

    low_confidence = [
        s for s in segments
        if s.get("confidence") is not None and s["confidence"] < 0.5
        and s.get("status") != "manual"
    ]
    if low_confidence:
        lines += ["", f"## 低置信度待复核（{len(low_confidence)} 条）", ""]
        for segment in low_confidence[:200]:
            lines.append(
                f"- `{sio.format_timestamp(segment['start'])}` "
                f"置信度 {segment['confidence']}：{segment['text'][:60]}"
            )
    return "\n".join(lines) + "\n"


WRITERS = {
    "srt": (export_srt, ".srt", "text/plain"),
    "vtt": (export_vtt, ".vtt", "text/vtt"),
    "ass": (export_ass, ".ass", "text/plain"),
    "script": (export_script, ".txt", "text/plain"),
    "jsonl": (export_jsonl, ".jsonl", "application/jsonl"),
    "rttm": (export_rttm, ".rttm", "text/plain"),
    "csv": (export_csv, ".csv", "text/csv"),
    "hf": (export_hf, ".json", "application/json"),
    "dataset": (export_dataset, ".json", "application/json"),
    "report": (export_report, ".md", "text/markdown"),
}

FORMAT_LABELS = {
    "srt": "SRT（带 [说话人] 前缀）",
    "vtt": "WebVTT（<v 说话人> 标签）",
    "ass": "ASS（按说话人上色）",
    "script": "剧本（人物：内容）",
    "jsonl": "JSONL（通用）",
    "rttm": "RTTM（说话人日志标准）",
    "csv": "CSV",
    "hf": "HuggingFace Dataset 风格",
    "dataset": "自定义数据集格式（含角色库）",
    "report": "复核报告（Markdown）",
}


def _safe_stem(stem: str) -> str:
    """Keep the name readable but strip characters that break paths/URLs."""
    cleaned = re.sub(r'[\\/:*?"<>|\r\n\t]+', "_", stem or "").strip(" ._")
    return cleaned or "export"


def write_all(segments, roles, context, options, output_dir: str,
              formats: list[str]) -> list[dict]:
    """Write every requested format into ``output_dir``; returns file metadata."""
    os.makedirs(output_dir, exist_ok=True)
    stem = _safe_stem(options.get("filename_stem") or context.get("video_id") or "export")
    written = []
    for name in formats:
        entry = WRITERS.get(name)
        if not entry:
            continue
        writer, extension, mime = entry
        content = writer(segments, roles, context, options)
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
        })
    return written
