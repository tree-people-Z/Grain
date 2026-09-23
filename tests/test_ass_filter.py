"""ASS import hygiene: only dialogue (plus readable song lyrics) is imported.

Real BDRip releases pack their ASS with typesetting: signs, title cards, staff
credits, translator notes and per-syllable karaoke glyphs. Those are display
decoration, not speech, so they must never reach the review track — otherwise
the cue list is flooded with single glyphs and the detector has nothing to
attribute. This locks that filter in with a self-contained fixture.

Run:  python tests/test_ass_filter.py
"""

from __future__ import annotations

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


STYLE_FMT = ("Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
             "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, "
             "ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, "
             "MarginL, MarginR, MarginV, Encoding")
STYLE = ("Style: {name},Arial,48,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,"
         "0,0,0,0,100,100,0,0,1,2,0,2,10,10,10,1")

FIXTURE = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080

[V4+ Styles]
{STYLE_FMT}
{STYLE.format(name="Default")}
{STYLE.format(name="Sub")}
{STYLE.format(name="Sign")}
{STYLE.format(name="ED_JP")}
{STYLE.format(name="ED_CH")}
{STYLE.format(name="Cmt_CH")}

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Comment: 0,0:00:00.00,0:00:00.00,Default,,0,0,0,,Dialogue
Dialogue: 0,0:00:01.00,0:00:02.00,Sub,,0,0,0,,你好
Dialogue: 0,0:00:02.50,0:00:03.50,Sub,,0,0,0,,早上好
Dialogue: 0,0:00:04.00,0:00:05.00,Sign,,0,0,0,,{{\\pos(100,100)}}路牌
Dialogue: 0,0:00:05.50,0:00:06.00,Sub,,0,0,0,,{{\\move(1,2,3,4)}}普
Dialogue: 0,0:00:06.50,0:00:07.50,Cmt_CH,,0,0,0,,注：译者注
Dialogue: 0,0:00:09.00,0:00:09.50,ED_JP,,0,0,0,,{{\\move(1,2,3,4)\\t(0,200,\\blur0.5)}}普
Comment: 0,0:00:09.00,0:00:12.00,ED_JP,,0,0,0,,{{\\fad(480,480)}}普通ってなんだろう
Comment: 0,0:00:09.00,0:00:12.00,ED_CH,,0,0,0,,{{\\fad(480,480)}}普通是什么呢
"""


def main() -> int:
    print("-- tag / style classifiers --")
    check("position tag detected", bool(sio._ASS_EFFECT_TAG.search(r"{\pos(1,2)}")))
    check("move tag detected", bool(sio._ASS_EFFECT_TAG.search(r"{\move(1,2,3,4)}普")))
    check("transform tag detected", bool(sio._ASS_EFFECT_TAG.search(r"{\t(0,200,\blur0.5)}")))
    check("plain text is not an effect", not sio._ASS_EFFECT_TAG.search("你好世界"))
    check("line break \\N is not an effect", not sio._ASS_EFFECT_TAG.search("上\\N下"))
    check("Dial_CH is a dialogue style", not sio._ASS_EFFECT_STYLE.search("Dial_CH"))
    check("Screen is an effect style", bool(sio._ASS_EFFECT_STYLE.search("Screen")))
    check("Cmt_CH is an effect style", bool(sio._ASS_EFFECT_STYLE.search("Cmt_CH")))
    check("ED_JP is a lyric style", bool(sio._ASS_LYRIC_STYLE.search("ED_JP")))
    check("Sub is not a lyric style", not sio._ASS_LYRIC_STYLE.search("Sub"))

    print("-- fixture parse --")
    segs, meta = sio.parse_ass(FIXTURE)
    styles = [s["_ass_style"] for s in segs]
    texts = [s["text"] for s in segs]
    check("only dialogue + lyrics kept", len(segs) == 4, f"{len(segs)}: {styles}")
    check("sign / note / karaoke dropped", "Sign" not in styles and "Cmt_CH" not in styles,
          str(styles))
    check("karaoke glyphs skipped", meta["skipped_effects"] == 4,
          f"{meta['skipped_effects']} skipped")
    check("readable lyric comments recovered", meta["recovered_lyrics"] == 2,
          f"{meta['recovered_lyrics']} recovered")
    check("comment markers ignored", "Dialogue" not in texts)
    check("override tags stripped from text", all("\\" not in t for t in texts))
    check("cues are time-ordered",
          all(segs[i]["start"] <= segs[i + 1]["start"] for i in range(len(segs) - 1)))
    check("ids reindexed contiguously", [s["id"] for s in segs] == list(range(len(segs))))

    print(f"\n{PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
