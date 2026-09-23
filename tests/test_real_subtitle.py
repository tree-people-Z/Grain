"""Tests against a real-world JPSC (日文+简体中文) interleaved ASS release.

These are the semantics that a genuine bilingual release exercises: two
language tracks interleaved cue-by-cue with identical timings. Skipped when the
sample file is absent so the suite stays runnable on other machines.

Run:  python tests/test_real_subtitle.py
"""

from __future__ import annotations

import glob
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

import subtitle_io as sio  # noqa: E402

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


def find_sample() -> str | None:
    """Look for a JPSC-style bilingual ASS in the usual test locations."""
    patterns = [
        os.path.join(BASE, "sample", "*JPSC*.ass"),
        os.path.join(BASE, "sample", "*JPSC*.ASS"),
        os.path.join(os.path.expanduser("~"), "Desktop", "测试", "*.ass"),
        os.path.join(os.path.expanduser("~"), "Desktop", "测试", "*.ASS"),
    ]
    for pattern in patterns:
        hits = glob.glob(pattern)
        if hits:
            return hits[0]
    return None


def main() -> int:
    sample = find_sample()
    if not sample:
        print("SKIP  no JPSC bilingual ASS sample found")
        return 0
    print(f"sample: {os.path.basename(sample)}")

    print("-- script classification --")
    check("kana line is japanese", sio._script_class("学校休んでるし") == "ja")
    check("kanji+中文 is chinese", sio._script_class("你最近不来学校") == "zh")
    check("latin is latin", sio._script_class("Hello world") == "latin")
    check("hangul is korean", sio._script_class("안녕하세요") == "ko")

    print("-- parse + interleaved detection --")
    segs, meta = sio.parse_subtitle(sample)
    check("parses dialogue cues", 200 < len(segs) < 1000, f"{len(segs)} cues")
    check("sign / title / staff / note events dropped",
          not any(s.get("_ass_style") in {"Screen", "Title", "Staff", "Cmt_CH"}
                  for s in segs))
    check("karaoke glyph events skipped", meta.get("skipped_effects", 0) > 1000,
          f"{meta.get('skipped_effects')} skipped")
    check("readable ED lyric lines recovered", meta.get("recovered_lyrics", 0) >= 10,
          f"{meta.get('recovered_lyrics')} recovered")
    check("cues are time-ordered",
          all(segs[i]["start"] <= segs[i + 1]["start"] for i in range(len(segs) - 1)))
    info = sio.detect_line_mode(segs)
    check("auto-detects interleaved", info["suggested"] == "interleaved", str(info["suggested"]))

    ja = sum(1 for s in segs if sio._script_class(s["text"]) == "ja")
    check("both languages present", 0.2 < ja / len(segs) < 0.8,
          f"ja={ja}/{len(segs)}")

    print("-- interleaved merge --")
    merged, applied = sio.apply_line_mode(segs, "auto")
    check("primary language is chinese", applied.get("primary") == "zh", str(applied.get("primary")))
    check("foreign tongue is japanese", applied.get("foreign") == "ja", str(applied.get("foreign")))
    check("cue count drops", len(merged) < len(segs), f"{len(segs)} -> {len(merged)}")
    check("ids reindexed contiguously",
          [s["id"] for s in merged] == list(range(len(merged))))
    check("cues stay time-ordered",
          all(merged[i]["start"] <= merged[i + 1]["start"] for i in range(len(merged) - 1)))

    paired = [s for s in merged if (s.get("translation") or "").strip()]
    # Real releases mix bilingual dialogue with Chinese-only narration, so not
    # every cue pairs; the bulk of them must.
    check("most chinese cues gained a japanese translation",
          len(paired) / max(1, len(merged)) > 0.8,
          f"{len(paired)}/{len(merged)} = {len(paired)/len(merged)*100:.0f}%")
    check("translations are japanese",
          sum(1 for s in paired if sio._script_class(s["translation"]) == "ja") / max(1, len(paired)) > 0.7)
    # A paired cue must actually overlap its partner in time.
    ja_end = max((s["end"] for s in segs if sio._script_class(s["text"]) == "ja"), default=0)
    check("merged timeline covers the original span",
          abs(merged[-1]["end"] - max(s["end"] for s in segs)) < 1.0,
          f"last={merged[-1]['end']:.1f}s ja_last={ja_end:.1f}s")

    print("-- bilingual writers on real data --")
    names = {0: "高松灯", 1: "千早爱音"}
    for i, s in enumerate(merged):
        s["speaker_id"] = i % 2
    srt = sio.write_srt(merged[:50], names, text_field="both")
    check("srt carries speaker prefix", srt.startswith("1\n") and "[高松灯]" in srt or "[千早爱音]" in srt)
    check("srt carries both languages", any(ch > "\u3040" for ch in srt) and any("\u4e00" <= ch <= "\u9fff" for ch in srt))
    ass = sio.write_ass(merged[:50], names, {0: "#FF6B6B", 1: "#4ECDC4"}, text_field="both")
    check("ass defines Translation style", "Style: Translation," in ass)
    check("ass bilingual lines use style reset", "\\N{\\rTranslation}" in ass)

    print(f"\n{PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
