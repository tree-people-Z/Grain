"""Subtitle parsing and writing for SRT / WebVTT / ASS.

Stdlib only. The internal representation is a list of segment dicts:

    {"id": int, "start": float, "end": float, "text": str,
     "speaker_id": int | None, "confidence": float | None, "status": str}

`status` is one of "auto" (assigned by the detector), "manual" (assigned or
corrected by a human) or "pending" (not assigned yet).
"""

from __future__ import annotations

import bisect
import re

# --- timecode helpers -------------------------------------------------------

_TS = re.compile(
    r"(?:(?P<h>\d+):)?(?P<m>\d{1,2}):(?P<s>\d{1,2})[.,](?P<ms>\d{1,3})"
)


def parse_timestamp(value: str) -> float:
    """Parse ``00:01:02,500`` (SRT), ``00:01:02.500`` (VTT) or ``0:01:02.50``."""
    match = _TS.search(value.strip())
    if not match:
        raise ValueError(f"unrecognised timestamp: {value!r}")
    hours = int(match.group("h") or 0)
    minutes = int(match.group("m"))
    seconds = int(match.group("s"))
    fraction = match.group("ms").ljust(3, "0")
    return hours * 3600 + minutes * 60 + seconds + int(fraction) / 1000.0


def format_timestamp(seconds: float, sep: str = ",") -> str:
    """Format seconds as ``HH:MM:SS,mmm``; ``sep`` selects ``.`` for VTT."""
    if seconds < 0:
        seconds = 0.0
    total_ms = int(round(seconds * 1000))
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, millis = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{sep}{millis:03d}"


def ass_timestamp(seconds: float) -> str:
    """ASS uses ``H:MM:SS.cc`` (centiseconds)."""
    if seconds < 0:
        seconds = 0.0
    total_cs = int(round(seconds * 100))
    hours, rem = divmod(total_cs, 360_000)
    minutes, rem = divmod(rem, 6_000)
    secs, centis = divmod(rem, 100)
    return f"{hours:d}:{minutes:02d}:{secs:02d}.{centis:02d}"


def ass_color(hex_color: str) -> str:
    """Convert ``#RRGGBB`` to ASS ``&H00BBGGRR`` (BGR order, as ASS expects)."""
    value = (hex_color or "#FFFFFF").lstrip("#")
    if len(value) != 6:
        value = "FFFFFF"
    r, g, b = value[0:2], value[2:4], value[4:6]
    return f"&H00{b.upper()}{g.upper()}{r.upper()}"


def _new_segment(start: float, end: float, text: str) -> dict:
    return {
        "id": 0,
        "start": round(start, 3),
        "end": round(end, 3),
        "text": text.strip(),
        "translation": "",
        "speaker_id": None,
        "confidence": None,
        "status": "pending",
    }


def _reindex(segments: list[dict]) -> list[dict]:
    for index, segment in enumerate(segments):
        segment["id"] = index
    return segments


# --- line handling: bilingual subtitles vs hard-wrapped text ----------------

def _script_class(text: str) -> str:
    """Rough script class of one line: ja / ko / zh / latin / other.

    Kana marks Japanese even when the line is kanji-heavy; hangul marks Korean.
    """
    kana = sum(1 for char in text if "\u3040" <= char <= "\u30ff")
    if kana > 0:
        return "ja"
    hangul = sum(1 for char in text if "\uac00" <= char <= "\ud7af")
    if hangul > 0:
        return "ko"
    cjk = sum(1 for char in text if "\u4e00" <= char <= "\u9fff")
    latin = sum(1 for char in text if char.isascii() and char.isalpha())
    if cjk == 0 and latin == 0:
        return "other"
    return "zh" if cjk >= latin else "latin"


def split_lines(text: str) -> list[str]:
    return [line.strip() for line in (text or "").split("\n") if line.strip()]


def detect_line_mode(segments: list[dict]) -> dict:
    """Guess whether multi-line cues are bilingual subtitles or wrapped text.

    Bilingual cues pair two different scripts (e.g. 中文 + English); hard-wrapped
    cues repeat the same script. The distinction decides whether line 2 is a
    translation or a continuation of the sentence.
    """
    multiline = 0
    different_script = 0
    for segment in segments:
        lines = split_lines(segment.get("text", ""))
        if len(lines) < 2:
            continue
        multiline += 1
        classes = {_script_class(line) for line in lines[:2]}
        if len(classes) == 2 and "other" not in classes:
            different_script += 1
    total = len(segments) or 1
    ratio = multiline / total
    bilingual = multiline >= 2 and different_script / max(1, multiline) >= 0.6
    suggested = "bilingual" if bilingual else ("join" if ratio >= 0.3 else "keep")
    if not bilingual and _detect_interleaved(segments):
        suggested = "interleaved"
    return {
        "suggested": suggested,
        "mode": suggested,
        "multiline": multiline,
        "different_script": different_script,
        "multiline_ratio": round(ratio, 3),
    }


def _cue_classes(segments: list[dict]) -> list[str]:
    return [
        _script_class(split_lines(s.get("text", ""))[0]) if split_lines(s.get("text", "")) else "other"
        for s in segments
    ]


def _detect_interleaved(segments: list[dict]) -> bool:
    """True when two languages alternate cue-by-cue on one timeline (e.g. a
    JPSC ASS where 日文 and 简体中文 cues share identical timings)."""
    if len(segments) < 6:
        return False
    classes = _cue_classes(segments)
    counts: dict[str, int] = {}
    for c in classes:
        counts[c] = counts.get(c, 0) + 1
    top = sorted(counts.items(), key=lambda kv: -kv[1])[:2]
    if len(top) < 2:
        return False
    total = len(classes)
    (c1, n1), (c2, n2) = top
    if "other" in (c1, c2):
        return False
    if n1 / total < 0.2 or n2 / total < 0.2:
        return False
    if (n1 + n2) / total < 0.8:
        return False
    switches = sum(1 for i in range(1, total)
                   if classes[i] != classes[i - 1] and classes[i] in (c1, c2) and classes[i - 1] in (c1, c2))
    return switches / (total - 1) >= 0.3


def apply_interleaved(segments: list[dict]) -> tuple[list[dict], dict]:
    """Fold alternating-language cues into one bilingual track.

    The majority script class becomes the primary text; each cue of the other
    language is folded (by maximum time overlap) into the primary cue it
    belongs to, and the folded cue is dropped. Cues with no partner stay.
    """
    classes = _cue_classes(segments)
    counts: dict[str, int] = {}
    for c in classes:
        counts[c] = counts.get(c, 0) + 1
    # Primary language preference: 中文 > latin > other > ja (UI 语言优先)。
    order = ["zh", "latin", "ko", "other", "ja"]
    primary = next((c for c in order if counts.get(c)), None)
    if primary is None:
        return segments, {"applied": "interleaved", "folded": 0, "primary": None}
    foreign = max((c for c in counts if c != primary), key=lambda c: counts[c], default=None)
    if foreign is None:
        return segments, {"applied": "interleaved", "folded": 0, "primary": primary}

    prim_indices = sorted(
        (i for i, c in enumerate(classes) if c != foreign),
        key=lambda i: segments[i]["start"],
    )
    prim_starts = [segments[i]["start"] for i in prim_indices]

    import bisect

    # Pairing is not one-to-one: one Japanese line is often split across two
    # Chinese cues (and vice versa), so a foreign cue may attach its text to
    # several heavily-overlapping primary cues.
    folded = 0
    remove = set()
    for i, segment in enumerate(segments):
        if classes[i] != foreign:
            continue
        duration = max(0.01, segment["end"] - segment["start"])
        pos = bisect.bisect_left(prim_starts, segment["start"])
        partners = []
        for j in range(max(0, pos - 6), min(len(prim_indices), pos + 6)):
            k = prim_indices[j]
            overlap = (min(segment["end"], segments[k]["end"])
                       - max(segment["start"], segments[k]["start"]))
            if overlap >= 0.25 * duration:
                partners.append((overlap, k))
        if not partners:
            continue  # no believable partner: keep the cue as its own line
        partners.sort(key=lambda kv: -kv[0])
        strongest = partners[0][0]
        for overlap, k in partners:
            if overlap < 0.6 * strongest:
                continue
            target = segments[k]
            existing = (target.get("translation") or "").strip()
            if segment["text"] in existing:
                continue
            target["translation"] = (existing + "\n" + segment["text"]).strip()
            # A human/previous assignment on the folded cue transfers over.
            if target.get("speaker_id") is None and segment.get("speaker_id") is not None:
                target["speaker_id"] = segment["speaker_id"]
                target["status"] = segment.get("status", "pending")
            folded += 1
        remove.add(i)

    kept = [s for i, s in enumerate(segments) if i not in remove]
    _reindex(kept)
    return kept, {"applied": "interleaved", "folded": folded,
                  "primary": primary, "foreign": foreign,
                  "kept": len(kept), "removed": len(remove)}


def apply_line_mode(segments: list[dict], mode: str = "auto") -> tuple[list[dict], dict]:
    """Normalise multi-line cue bodies. Returns (segments, info).

    mode: auto | bilingual | join | keep
      bilingual — line 1 becomes ``text``, the rest become ``translation``
      join      — every line merges into one sentence (SRT hard-wrapping)
      keep      — leave the raw multi-line text untouched
    """
    info = detect_line_mode(segments)
    if mode == "auto":
        mode = info["suggested"]
    if mode == "keep":
        info["applied"] = mode
        return segments, info
    if mode == "interleaved":
        kept, applied = apply_interleaved(segments)
        info["applied"] = "interleaved"
        info.update(applied)
        return kept, info
    for segment in segments:
        lines = split_lines(segment.get("text", ""))
        if not lines:
            continue
        if mode == "bilingual":
            segment["text"] = lines[0]
            segment["translation"] = "\n".join(lines[1:])
        else:  # join
            separator = " " if any(_script_class(l) == "latin" for l in lines) else ""
            segment["text"] = separator.join(lines)
            segment["translation"] = ""
    info["applied"] = mode
    return segments, info


def merge_translations(segments: list[dict], others: list[dict]) -> int:
    """Attach ``others`` as translations of ``segments`` by maximum time overlap.

    Used for the two-file workflow (e.g. ``video.zh.srt`` + ``video.en.srt``).
    Returns the number of cues that received a translation.
    """
    if not others:
        return 0
    # Interval-indexed scan instead of the old O(n×m) double loop. Sort the
    # translation track once, advance a pointer past cues that end before the
    # segment, and binary-search the upper bound: everything outside [j, right)
    # has zero overlap, so the best match is unchanged.
    ordered = sorted(others, key=lambda other: other["start"])
    starts = [other["start"] for other in ordered]
    ends = [other["end"] for other in ordered]
    total = len(ordered)
    matched = 0
    j = 0
    for segment in sorted(segments, key=lambda item: item["start"]):
        seg_start, seg_end = segment["start"], segment["end"]
        while j < total and ends[j] <= seg_start:
            j += 1
        right = bisect.bisect_left(starts, seg_end)
        best, best_overlap = None, 0.0
        for k in range(j, right):
            other = ordered[k]
            overlap = min(seg_end, other["end"]) - max(seg_start, other["start"])
            if overlap > best_overlap:
                best_overlap, best = overlap, other
        if best is not None and best_overlap > 0.05:
            translation = (best.get("text") or "").strip()
            if translation and translation != segment.get("text"):
                segment["translation"] = translation
                matched += 1
    return matched


def compose_text(segment: dict, field: str = "both") -> str:
    """Assemble a cue body honouring bilingual content.

    field: both | primary | translation
    """
    primary = segment.get("text", "")
    translation = (segment.get("translation") or "").strip()
    if not translation or field == "primary":
        return primary
    if field == "translation":
        return translation
    return f"{primary}\n{translation}" if primary else translation


# --- parsing ----------------------------------------------------------------

_SRT_BLOCK = re.compile(
    r"(?P<idx>\d+)\s*\n"
    r"(?P<start>[^\n]+?)\s*-->\s*(?P<end>[^\n]+?)(?:[ \t]+[^\n]*)?\n"
    r"(?P<text>(?:.|\n)*?)(?=\n\s*\n|\n\d+\s*\n|\Z)",
    re.MULTILINE,
)

_VTT_CUE = re.compile(
    r"(?:(?P<idx>[\w.-]+)[^\n]*\n)?"
    r"(?P<start>[^\n]+?)\s*-->\s*(?P<end>[^\n]+?)(?:[ \t]+[^\n]*)?\n"
    r"(?P<text>(?:.|\n)*?)(?=\n\s*\n|\Z)",
    re.MULTILINE,
)

_VOICE_TAG = re.compile(r"^<\s*v[\s.]([^>]+)>", re.IGNORECASE)
_TAG = re.compile(r"<[^>]+>")


def _strip_vtt_voice(text: str) -> tuple[str, str | None]:
    """Return (clean_text, inline_speaker_name) for a VTT cue body."""
    lines = []
    inline_speaker = None
    for raw in text.splitlines():
        voice = _VOICE_TAG.match(raw.strip())
        if voice:
            inline_speaker = voice.group(1).strip()
            raw = _VOICE_TAG.sub("", raw, count=1)
        lines.append(_TAG.sub("", raw))
    return "\n".join(lines).strip(), inline_speaker


def parse_srt(content: str) -> tuple[list[dict], dict]:
    content = content.replace("\r\n", "\n").replace("\r", "\n").lstrip("\ufeff")
    segments = []
    for match in _SRT_BLOCK.finditer(content):
        start = parse_timestamp(match.group("start"))
        end = parse_timestamp(match.group("end"))
        text = match.group("text").strip()
        segments.append(_new_segment(start, end, text))
    return _reindex(segments), {}


def parse_vtt(content: str) -> tuple[list[dict], dict]:
    content = content.replace("\r\n", "\n").replace("\r", "\n").lstrip("\ufeff")
    body = re.sub(r"^WEBVTT[^\n]*\n", "", content)
    # Drop NOTE/STYLE/REGION blocks, which are not cues.
    body = re.sub(r"^(?:NOTE|STYLE|REGION)\b.*?(?=\n\s*\n|\Z)", "", body,
                  flags=re.MULTILINE | re.DOTALL)
    segments = []
    for match in _VTT_CUE.finditer(body):
        text, inline_speaker = _strip_vtt_voice(match.group("text"))
        if not text:
            continue
        segment = _new_segment(
            parse_timestamp(match.group("start")),
            parse_timestamp(match.group("end")),
            text,
        )
        if inline_speaker:
            # Preserve the name so review can pre-fill known roles.
            segment["speaker_name_hint"] = inline_speaker
        segments.append(segment)
    return _reindex(segments), {}


_ASS_OVERRIDE = re.compile(r"\{[^}]*\}")

# ASS override tags that only typesetting needs: movement, clipping, transforms,
# karaoke timing and vector drawing. A cue carrying any of these is on-screen
# decoration (a sign, a title, or one karaoke glyph), never spoken dialogue, so
# the review track must not import it. Plain dialogue rarely uses them.
_ASS_EFFECT_TAG = re.compile(
    r"\\(?:pos|move|org|clip|iclip|fad|fade|blur|frx|fry|frz|fax|fay|"
    r"fscx|fscy|xshad|yshad|bord|shad|kf|ko|kt|k|t|p)(?![a-z])"
)

# Style names that mark signs / titles / staff credits / translator notes rather
# than speech. Bounded so "Dialogue" or "Dial_CH" never match.
_ASS_EFFECT_STYLE = re.compile(
    r"(?:^|[_\s-])(?:sign|screen|title|staff|note|comment|cmt|info|logo|"
    r"typeset|shadow|预告|标题|注释|说明|特效|字幕组|制作)(?:$|[_\s-])",
    re.IGNORECASE,
)

# Fansub karaoke parks the readable song line in a Comment event while the
# Dialogue events are the per-syllable typesetting. Recover those lyric lines
# (positive duration + a song-style name); comments are otherwise ignored.
_ASS_LYRIC_STYLE = re.compile(
    r"(?:^|[_\s-])(?:op|ed|lyric|song|karaoke|主题曲|片头|片尾|插曲|歌词)"
    r"(?:$|[_\s0-9-])",
    re.IGNORECASE,
)

_ASS_DEFAULT_STYLE = {
    "Fontname": "Microsoft YaHei",
    "Fontsize": "48",
    "PrimaryColour": "&H00FFFFFF",
    "SecondaryColour": "&H000000FF",
    "OutlineColour": "&H00000000",
    "BackColour": "&H80000000",
    "Bold": "0",
    "Italic": "0",
    "Underline": "0",
    "StrikeOut": "0",
    "ScaleX": "100",
    "ScaleY": "100",
    "Spacing": "0",
    "Angle": "0",
    "BorderStyle": "1",
    "Outline": "2",
    "Shadow": "1",
    "Alignment": "2",
    "MarginL": "20",
    "MarginR": "20",
    "MarginV": "30",
    "Encoding": "1",
}

# Column order of an ASS "Style:" line when no "Format:" line preceded it.
_ASS_STYLE_FORMAT_DEFAULT = list(_ASS_DEFAULT_STYLE.keys())


def parse_ass(content: str) -> tuple[list[dict], dict]:
    content = content.replace("\r\n", "\n").replace("\r", "\n").lstrip("\ufeff")
    styles: dict[str, dict] = {}
    records: list[tuple[str, dict]] = []
    section = None
    dialogue_format: list[str] = []
    style_format: list[str] = []

    for raw in content.splitlines():
        line = raw.strip()
        if line.startswith("[") and line.endswith("]"):
            section = line.strip("[]").lower()
            continue
        if not line or line.startswith(";"):
            continue

        head, _, tail = line.partition(":")
        head_lower = head.strip().lower()

        if section == "v4+ styles" or section == "v4 styles":
            if head_lower == "format":
                style_format = [f.strip() for f in tail.split(",")]
            elif head_lower == "style":
                if not style_format:
                    # Some files place Style: before Format:; assume ASS defaults.
                    style_format = _ASS_STYLE_FORMAT_DEFAULT
                values = tail.split(",", len(style_format) - 1)
                record = dict(zip(style_format, (v.strip() for v in values)))
                name = record.get("Name", "Default")
                styles[name] = record
        elif section == "events":
            if head_lower == "format":
                dialogue_format = [f.strip() for f in tail.split(",")]
            elif head_lower in ("dialogue", "comment"):
                if not dialogue_format:
                    dialogue_format = [
                        "Layer", "Start", "End", "Style", "Name",
                        "MarginL", "MarginR", "MarginV", "Effect", "Text",
                    ]
                values = tail.split(",", len(dialogue_format) - 1)
                record = dict(zip(dialogue_format, (v.strip() for v in values)))
                records.append((head_lower, record))

    segments: list[dict] = []
    skipped_effects = 0
    recovered_lyrics = 0
    for kind, record in records:
        raw_text = record.get("Text", "")
        text = _ASS_OVERRIDE.sub("", raw_text).replace("\\N", "\n").strip()
        if not text:
            continue
        style_name = record.get("Style", "").strip()
        start = parse_timestamp(record.get("Start", "0:00:00.00"))
        end = parse_timestamp(record.get("End", "0:00:00.00"))
        if kind == "comment":
            # Comments are not rendered. The one useful exception is a fansub
            # karaoke that parks the readable song line here; keep positive-
            # duration lyric comments and drop markers / editor notes.
            if not (end - start > 0.05 and _ASS_LYRIC_STYLE.search(style_name)):
                continue
            recovered_lyrics += 1
        elif _ASS_EFFECT_STYLE.search(style_name) or _ASS_EFFECT_TAG.search(raw_text):
            # A sign / title / staff credit / karaoke glyph: not dialogue.
            skipped_effects += 1
            continue
        segment = _new_segment(start, end, text)
        actor = record.get("Name", "").strip()
        if actor and actor.upper() not in ("", "N/A"):
            segment["speaker_name_hint"] = actor
        segment["_ass_style"] = style_name
        segments.append(segment)

    segments.sort(key=lambda s: s["start"])
    return _reindex(segments), {
        "styles": styles,
        "style_format": style_format,
        "skipped_effects": skipped_effects,
        "recovered_lyrics": recovered_lyrics,
    }


def _read_text(path: str) -> str:
    """Decode a subtitle file, tolerating the common UTF-8 / GBK encodings.

    Chinese subtitles are frequently GB18030/GBK; decoding those as UTF-8 with
    ``errors="replace"`` silently turns every character into U+FFFD, so try the
    likely encodings and only fall back to lossy decoding as a last resort.
    """
    with open(path, "rb") as handle:
        raw = handle.read()
    if raw.startswith(b"\xef\xbb\xbf"):
        return raw.decode("utf-8-sig")
    for encoding in ("utf-8", "gb18030"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def parse_subtitle(path: str) -> tuple[list[dict], dict]:
    """Dispatch on file extension and return ``(segments, meta)``."""
    content = _read_text(path)
    suffix = path.lower().rsplit(".", 1)[-1] if "." in path else ""
    if suffix == "srt":
        return parse_srt(content)
    if suffix in ("vtt", "webvtt"):
        return parse_vtt(content)
    if suffix in ("ass", "ssa"):
        return parse_ass(content)
    # Unknown extension: sniff by content shape.
    if content.lstrip().startswith("WEBVTT"):
        return parse_vtt(content)
    if "[Events]" in content or "[Script Info]" in content:
        return parse_ass(content)
    return parse_srt(content)


# --- writing ----------------------------------------------------------------

def _speaker_prefix(segment: dict, name: str | None) -> str:
    return f"[{name}] " if name else ""


def _prefix_body(segment: dict, name: str | None, text_field: str) -> str:
    """Compose the cue body and indent a translation line under the speaker tag."""
    body = compose_text(segment, text_field)
    prefix = _speaker_prefix(segment, name)
    if not prefix:
        return body
    indent = " " * len(prefix)
    return prefix + body.replace("\n", "\n" + indent)


def write_srt(segments: list[dict], names: dict[int, str], include_pending: bool = True,
              text_field: str = "both") -> str:
    lines = []
    index = 0
    for segment in segments:
        name = names.get(segment.get("speaker_id"))
        if name is None and not include_pending:
            continue
        index += 1
        lines.append(str(index))
        lines.append(
            f"{format_timestamp(segment['start'])} --> {format_timestamp(segment['end'])}"
        )
        lines.append(_prefix_body(segment, name, text_field))
        lines.append("")
    return "\n".join(lines)


def write_vtt(segments: list[dict], names: dict[int, str], include_pending: bool = True,
              text_field: str = "both") -> str:
    lines = ["WEBVTT", ""]
    for segment in segments:
        name = names.get(segment.get("speaker_id"))
        if name is None and not include_pending:
            continue
        lines.append(
            f"{format_timestamp(segment['start'], '.')} --> "
            f"{format_timestamp(segment['end'], '.')}"
        )
        body = compose_text(segment, text_field)
        lines.append(f"<v {name}>{body}" if name else body)
        lines.append("")
    return "\n".join(lines)


def write_ass(
    segments: list[dict],
    names: dict[int, str],
    colors: dict[int, str],
    include_pending: bool = True,
    text_field: str = "both",
) -> str:
    """ASS with one style per speaker so players show the speaker colour.

    A bilingual translation line is rendered with the smaller ``Translation``
    style via an inline ``{\\rTranslation}`` reset.
    """
    header = [
        "[Script Info]",
        "; Generated by Grain",
        "ScriptType: v4.00+",
        "WrapStyle: 0",
        "ScaledBorderAndShadow: yes",
        "YCbCr Matrix: TV.601",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, "
        "ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, "
        "MarginR, MarginV, Encoding",
    ]

    def style_line(name: str, color: str, scale: float = 1.0) -> str:
        style = dict(_ASS_DEFAULT_STYLE)
        style["PrimaryColour"] = ass_color(color)
        if scale != 1.0:
            base = float(_ASS_DEFAULT_STYLE["Fontsize"])
            style["Fontsize"] = str(int(base * scale))
        order = [
            "Fontname", "Fontsize", "PrimaryColour", "SecondaryColour",
            "OutlineColour", "BackColour", "Bold", "Italic", "Underline",
            "StrikeOut", "ScaleX", "ScaleY", "Spacing", "Angle", "BorderStyle",
            "Outline", "Shadow", "Alignment", "MarginL", "MarginR", "MarginV",
            "Encoding",
        ]
        values = ",".join(style[key] for key in order)
        return f"Style: {name},{values}"

    header.append(style_line("Default", "#CCCCCC"))
    header.append(style_line("Translation", "#D8D8DE", 0.82))
    for speaker_id in sorted(names):
        safe = names[speaker_id].replace(",", " ").strip()
        header.append(style_line(f"S{speaker_id}_{safe}"[:48], colors.get(speaker_id, "#FFFFFF")))

    lines = header + [
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    for segment in segments:
        speaker_id = segment.get("speaker_id")
        name = names.get(speaker_id)
        if name is None and not include_pending:
            continue
        style_name = f"S{speaker_id}_{name.replace(',', ' ').strip()}"[:48] if name else "Default"
        actor = name or ""
        primary = segment.get("text", "")
        translation = (segment.get("translation") or "").strip()
        if text_field == "translation":
            primary, translation = translation, ""
        elif text_field == "primary":
            translation = ""
        text = primary.replace("\n", "\\N")
        if translation:
            text += "\\N{\\rTranslation}" + translation.replace("\n", "\\N")
        lines.append(
            f"Dialogue: 0,{ass_timestamp(segment['start'])},"
            f"{ass_timestamp(segment['end'])},{style_name},{actor},"
            f"0,0,0,,{text}"
        )
    return "\n".join(lines) + "\n"
