"""Subtitle parsing and writing for SRT / ASS.

Stdlib only. The internal representation is a list of segment dicts:

    {"id": int, "start": float, "end": float, "text": str,
     "speaker_id": int | None, "confidence": float | None, "status": str}

`status` is one of "auto" (assigned by the detector), "manual" (assigned or
corrected by a human) or "pending" (not assigned yet).
"""

from __future__ import annotations

import re

# --- timecode helpers -------------------------------------------------------

_TS = re.compile(
    r"(?:(?P<h>\d+):)?(?P<m>\d{1,2}):(?P<s>\d{1,2})[.,](?P<ms>\d{1,3})"
)


def parse_timestamp(value: str) -> float:
    """Parse ``00:01:02,500`` (SRT) or ``0:01:02.50`` (ASS)."""
    match = _TS.search(value.strip())
    if not match:
        raise ValueError(f"unrecognised timestamp: {value!r}")
    hours = int(match.group("h") or 0)
    minutes = int(match.group("m"))
    seconds = int(match.group("s"))
    fraction = match.group("ms").ljust(3, "0")
    return hours * 3600 + minutes * 60 + seconds + int(fraction) / 1000.0


def format_timestamp(seconds: float, sep: str = ",") -> str:
    """Format seconds as ``HH:MM:SS,mmm`` (``sep`` selects ``.`` when needed)."""
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
        "speaker_id": None,
        "confidence": None,
        "status": "pending",
    }


def _reindex(segments: list[dict]) -> list[dict]:
    for index, segment in enumerate(segments):
        segment["id"] = index
    return segments


# --- parsing ----------------------------------------------------------------

_SRT_BLOCK = re.compile(
    r"(?P<idx>\d+)\s*\n"
    r"(?P<start>[^\n]+?)\s*-->\s*(?P<end>[^\n]+?)(?:[ \t]+[^\n]*)?\n"
    r"(?P<text>(?:.|\n)*?)(?=\n\s*\n|\n\d+\s*\n|\Z)",
    re.MULTILINE,
)


def parse_srt(content: str) -> tuple[list[dict], dict]:
    content = content.replace("\r\n", "\n").replace("\r", "\n").lstrip("\ufeff")
    segments = []
    for match in _SRT_BLOCK.finditer(content):
        start = parse_timestamp(match.group("start"))
        end = parse_timestamp(match.group("end"))
        text = match.group("text").strip()
        segments.append(_new_segment(start, end, text))
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

# Common fansub style suffixes. A JP and CH event with the same time range is
# one bilingual cue, not two separate lines to classify independently.
_ASS_LANG = re.compile(r"(?:^|_)(JP|CH)(?:\d+)?(?:_|$)", re.IGNORECASE)


def _bilingual_key(style: str) -> tuple[str, str] | None:
    match = _ASS_LANG.search(style or "")
    if not match:
        return None
    family = _ASS_LANG.sub("_LANG", style, count=1)
    return family.lower(), match.group(1).upper()

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

        if section in ("v4+ styles", "v4 styles"):
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

    # Some ASS files duplicate every event and place Japanese/Chinese on
    # separate style lines. Remove exact duplicates, then combine a matching
    # JP/CH pair into one cue so speaker detection runs once per spoken line.
    unique = []
    seen = set()
    for segment in segments:
        key = (segment["start"], segment["end"], segment.get("_ass_style"),
               segment["text"], segment.get("speaker_name_hint"))
        if key in seen:
            continue
        seen.add(key)
        unique.append(segment)
    segments = unique
    bilingual: dict[tuple[float, float, str], dict[str, list[dict]]] = {}
    for segment in segments:
        lang_key = _bilingual_key(segment.get("_ass_style", ""))
        if lang_key:
            family, language = lang_key
            bilingual.setdefault((segment["start"], segment["end"], family), {}) \
                .setdefault(language, []).append(segment)
    removed = set()
    for (start, end, _family), languages in bilingual.items():
        pairs = min(len(languages.get("JP", [])), len(languages.get("CH", [])))
        for index in range(pairs):
            jp, ch = languages["JP"][index], languages["CH"][index]
            jp["text"] = f"{jp['text']}\n{ch['text']}"
            removed.add(id(ch))
    segments = [segment for segment in segments if id(segment) not in removed]
    segments.sort(key=lambda s: (s["start"], s["end"]))
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
    if suffix in ("ass", "ssa"):
        return parse_ass(content)
    if suffix == "srt":
        return parse_srt(content)
    # Unknown extension: sniff by content shape.
    if "[Events]" in content or "[Script Info]" in content:
        return parse_ass(content)
    return parse_srt(content)


# --- writing ----------------------------------------------------------------

def _speaker_prefix(name: str | None) -> str:
    return f"[{name}] " if name else ""


def write_srt(segments: list[dict], names: dict[int, str],
              include_pending: bool = True) -> str:
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
        lines.append(_speaker_prefix(name) + segment.get("text", ""))
        lines.append("")
    return "\n".join(lines)


def write_ass(
    segments: list[dict],
    names: dict[int, str],
    colors: dict[int, str],
    include_pending: bool = True,
) -> str:
    """ASS with one style per speaker so players show the speaker colour."""
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

    def style_line(name: str, color: str) -> str:
        style = dict(_ASS_DEFAULT_STYLE)
        style["PrimaryColour"] = ass_color(color)
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
    for speaker_id in sorted(names):
        safe = names[speaker_id].replace(",", " ").strip()
        header.append(style_line(f"S{speaker_id}_{safe}"[:48],
                                 colors.get(speaker_id, "#FFFFFF")))

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
        style_name = (f"S{speaker_id}_{name.replace(',', ' ').strip()}"[:48]
                      if name else "Default")
        actor = name or ""
        text = segment.get("text", "").replace("\n", "\\N")
        lines.append(
            f"Dialogue: 0,{ass_timestamp(segment['start'])},"
            f"{ass_timestamp(segment['end'])},{style_name},{actor},"
            f"0,0,0,,{text}"
        )
    return "\n".join(lines) + "\n"
