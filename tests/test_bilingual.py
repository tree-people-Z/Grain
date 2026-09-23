"""Bilingual subtitle handling tests (no external dependencies).

Run:  python tests/test_bilingual.py
"""

from __future__ import annotations

import os
import sys

try:  # Chinese Windows consoles default to GBK, which cannot encode ✓/✗
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

import dataset_export  # noqa: E402
import subtitle_io as sio  # noqa: E402

SAMPLE = os.path.join(BASE, "sample")
PASS = 0
FAIL = 0


def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS  {name}" + (f"  ({detail})" if detail else ""))
    else:
        FAIL += 1
        print(f"  FAIL  {name}  {detail}")


def roles():
    return [
        {"id": 0, "name": "张三", "color": "#FF6B6B", "type": "registered",
         "embedder": "builtin", "embedding": [], "samples": []},
        {"id": 1, "name": "李四", "color": "#4ECDC4", "type": "registered",
         "embedder": "builtin", "embedding": [], "samples": []},
    ]


def main():
    print("-- inline bilingual detection --")
    segs, _ = sio.parse_subtitle(os.path.join(SAMPLE, "bilingual_inline.srt"))
    check("parses 3 cues", len(segs) == 3, str(len(segs)))
    info = sio.detect_line_mode(segs)
    check("detects bilingual mode", info["suggested"] == "bilingual", str(info))

    segs, info = sio.apply_line_mode(segs, "auto")
    check("applies bilingual", info["applied"] == "bilingual")
    check("line1 becomes text", segs[0]["text"] == "你好，欢迎来到这次的访谈节目。",
          segs[0]["text"])
    check("line2 becomes translation",
          segs[0]["translation"] == "Hello and welcome to this interview.",
          segs[0]["translation"])
    check("all cues have translation",
          all(s["translation"] for s in segs))

    print("-- hard-wrapped detection --")
    wrapped, _ = sio.parse_subtitle(os.path.join(SAMPLE, "wrapped.srt"))
    winfo = sio.detect_line_mode(wrapped)
    check("detects join mode", winfo["suggested"] == "join", str(winfo))
    joined, jinfo = sio.apply_line_mode(wrapped, "auto")
    check("joins without separator",
          joined[0]["text"] == "你好，欢迎来到这次的访谈节目。", joined[0]["text"])
    check("joined cues carry no translation",
          all(not s["translation"] for s in joined))

    print("-- two-file merge --")
    base, _ = sio.parse_subtitle(os.path.join(SAMPLE, "interview.srt"))
    tr, _ = sio.parse_subtitle(os.path.join(SAMPLE, "bilingual_en.srt"))
    matched = sio.merge_translations(base, tr)
    check("merges translations by overlap", matched == 3, str(matched))
    check("merged text correct",
          base[0]["translation"] == "Hello and welcome to this interview.",
          base[0]["translation"])
    check("primary text untouched",
          base[0]["text"] == "你好，欢迎来到这次的访谈节目。", base[0]["text"])

    print("-- writers honour text_field --")
    inline, _ = sio.parse_subtitle(os.path.join(SAMPLE, "bilingual_inline.srt"))
    inline, _ = sio.apply_line_mode(inline, "bilingual")
    for segment in inline:
        segment["speaker_id"] = 0 if segment["id"] % 2 == 0 else 1

    srt_both = sio.write_srt(inline, {0: "张三", 1: "李四"}, text_field="both")
    check("srt both: primary + translation on separate lines",
          "你好，欢迎来到这次的访谈节目。\n" in srt_both
          and "Hello and welcome" in srt_both
          and "[张三] 你好" in srt_both)
    check("srt both: translation indented under tag",
          "\n            Hello" in srt_both or "[张三] 你好" in srt_both)

    srt_primary = sio.write_srt(inline, {0: "张三", 1: "李四"}, text_field="primary")
    check("srt primary: translation dropped",
          "Hello" not in srt_primary and "你好" in srt_primary)

    srt_trans = sio.write_srt(inline, {0: "张三", 1: "李四"}, text_field="translation")
    check("srt translation: only translation",
          "你好" not in srt_trans and "Hello" in srt_trans)

    ass = sio.write_ass(inline, {0: "张三", 1: "李四"},
                        {0: "#FF6B6B", 1: "#4ECDC4"}, text_field="both")
    check("ass: Translation style defined", "Style: Translation," in ass)
    check("ass: bilingual dialogue uses style reset",
          "\\N{\\rTranslation}" in ass)

    print("-- dataset exporters carry translation --")
    context = {"video_id": "t", "media_path": "a.mp4", "subtitle_path": "a.srt",
               "uri": "t", "stats": {}, "engine": "manual"}
    options = {"include_pending": True, "include_confidence": True,
               "include_status": True, "filename_stem": "t", "text_field": "both"}
    jsonl = dataset_export.export_jsonl(inline, roles(), context, options)
    import json
    first = json.loads(jsonl.splitlines()[0])
    check("jsonl: text + translation keys",
          first["text"].startswith("你好") and first["translation"].startswith("Hello"),
          str(list(first.keys())))
    csv_text = dataset_export.export_csv(inline, roles(), context, options)
    check("csv: translation column", "translation" in csv_text.splitlines()[0])
    dataset = json.loads(dataset_export.export_dataset(inline, roles(), context, options))
    check("dataset: annotation exposes translation",
          dataset["annotations"][0]["translation"].startswith("Hello"))
    options_primary = dict(options, text_field="primary")
    csv_primary = dataset_export.export_csv(inline, roles(), context, options_primary)
    check("csv primary: no translation column",
          "translation" not in csv_primary.splitlines()[0])

    print(f"\n{PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
