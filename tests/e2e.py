"""End-to-end test: starts the real server, exercises the full workflow.

Run:  python tests/e2e.py
Uses only the stdlib. Exits non-zero on the first failed assertion.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request

try:  # Chinese Windows consoles default to GBK, which cannot encode ✓/✗
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PORT = 8799
ROOT = f"http://127.0.0.1:{PORT}"
MEDIA = os.path.join(BASE, "sample", "interview.wav")
SUBTITLE = os.path.join(BASE, "sample", "interview.srt")
# Git-Bash style path to exercise normalization
MEDIA_BASH = "/c/" + MEDIA.replace("\\", "/").split(":/")[-1] if ":/" in MEDIA.replace("\\", "/") else MEDIA

PASS: list[str] = []
EXPECTED_ORDER = ["A", "B", "A", "B", "A", "B"]  # ground truth from make_sample


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        PASS.append(name)
        print(f"  PASS  {name}" + (f"  ({detail})" if detail else ""))
    else:
        print(f"  FAIL  {name}  {detail}")
        raise SystemExit(f"FAILED: {name} {detail}")


def call(method: str, path: str, payload: dict | None = None):
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(
        ROOT + path, data=data, method=method,
        headers={"Content-Type": "application/json"} if data else {},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> None:
    # Always run against a throwaway data dir (SSP_DATA_DIR). This test used to
    # delete the real `data/` tree, which destroyed the user's saved projects.
    data_dir = os.path.join(BASE, "data", "__e2e__")
    shutil.rmtree(data_dir, ignore_errors=True)
    os.makedirs(data_dir, exist_ok=True)
    # Keep enrolment hermetic: with a manual default engine the test's
    # voiceprints stay in the builtin space instead of probing a neural model.
    with open(os.path.join(data_dir, "settings.json"), "w", encoding="utf-8") as handle:
        json.dump({"default_engine": "manual"}, handle)
    # The token lives with the real data; copy it in so ASR/engine paths can auth.
    token_file = os.path.join(BASE, "data", "hf_token.txt")
    if os.path.exists(token_file):
        shutil.copyfile(token_file, os.path.join(data_dir, "hf_token.txt"))
    shutil.rmtree(os.path.join(BASE, "sample", "__e2e__"), ignore_errors=True)

    env = dict(os.environ, SSP_DATA_DIR=data_dir)
    server = subprocess.Popen(
        [sys.executable, os.path.join(BASE, "server.py"), "--port", str(PORT)],
        # Drain nothing: piping the server output can deadlock on chatty engines.
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env,
    )
    try:
        run_tests()
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()
        shutil.rmtree(data_dir, ignore_errors=True)

    print(f"\nALL {len(PASS)} CHECKS PASSED")


def run_tests() -> None:
    # -- server health + state ------------------------------------------
    for _ in range(60):
        try:
            state = call("GET", "/api/state")
            break
        except Exception:
            time.sleep(0.5)
    else:
        raise SystemExit("server did not become healthy")

    check("state: engines listed",
          {"manual", "pyannote", "campp", "voiceprint-cue"}
          <= set(state["engines"]))
    check("state: manual engine available",
          state["engines"]["manual"]["available"])
    check("state: professional engines listed first",
          list(state["engines"])[0] in ("campp", "pyannote"))
    check("state: ASR optional entry present", "detail" in state["asr"])
    check("state: 10 export formats", len(state["formats"]) == 10)
    check("state: ffmpeg found", state["ffmpeg"] is True)

    # -- create project (exercises path normalization) -------------------
    bash_path = "/c/" + MEDIA.replace("\\", "/").replace("C:/", "")
    created = call("POST", "/api/projects", {
        "media_path": bash_path, "subtitle_path": SUBTITLE, "name": "e2e",
    })
    project = created["project"]
    pid = project["id"]
    check("create: 6 cues parsed", len(project["segments"]) == 6)
    check("create: media path normalized to Windows form",
          project["media_path"].lower() == MEDIA.lower(),
          project["media_path"])
    check("create: duration ~20s", 19 < project["duration"] < 22,
          str(project["duration"]))

    # -- waveform peaks ---------------------------------------------------
    peaks = call("GET", f"/api/projects/{pid}/peaks?buckets=400")
    check("peaks: 400 buckets normalised", len(peaks["peaks"]) == 400
          and max(peaks["peaks"]) <= 1.0)

    # -- manual detection + explicit assignment ----------------------------
    result = call("POST", f"/api/projects/{pid}/detect", {"engine": "manual"})
    project = result["project"]
    check("manual: detection leaves cues pending",
          all(s["speaker_id"] is None for s in project["segments"]))

    call("POST", f"/api/projects/{pid}/roles", {"name": "张三"})
    call("POST", f"/api/projects/{pid}/roles", {"name": "李四"})
    project = call("GET", f"/api/projects/{pid}")["project"]
    check("roles: two named roles created", len(project["roles"]) == 2,
          str([r["name"] for r in project["roles"]]))
    role_a, role_b = project["roles"][0]["id"], project["roles"][1]["id"]
    for index, seg in enumerate(project["segments"]):
        call("POST", f"/api/projects/{pid}/segments",
             {"segment_id": seg["id"],
              "speaker_id": role_a if index % 2 == 0 else role_b})
    project = call("GET", f"/api/projects/{pid}")["project"]
    order = [next(r["name"] for r in project["roles"] if r["id"] == s["speaker_id"])
             for s in project["segments"]]
    check("assignment: all cues assigned",
          all(s["speaker_id"] is not None for s in project["segments"]))
    check("assignment: two roles alternate A/B/A/B/A/B",
          len(set(order)) == 2 and
          all(order[i] != order[i + 1] for i in range(len(order) - 1)),
          str(order))

    # -- manual override + bulk + undo-friendly status param --------------
    seg0 = project["segments"][0]["id"]
    call("POST", f"/api/projects/{pid}/segments",
         {"segment_id": seg0, "speaker_id": 1})
    project = call("GET", f"/api/projects/{pid}")["project"]
    check("manual override works", project["segments"][0]["speaker_id"] == 1
          and project["segments"][0]["status"] == "manual")
    result = call("POST", f"/api/projects/{pid}/bulk",
                  {"ids": [seg0], "speaker_id": 0, "status": "auto"})
    project = result["project"]
    check("bulk with explicit status", project["segments"][0]["speaker_id"] == 0
          and project["segments"][0]["status"] == "auto")

    # -- detection folds manually-reviewed cues into role voiceprints ------
    result = call("POST", f"/api/projects/{pid}/detect", {"engine": "manual"})
    project = result["project"]
    named = {r["name"]: r for r in project["roles"]}
    check("learn: manual cues feed role voiceprints",
          named["张三"]["sample_count"] >= 1 and named["李四"]["sample_count"] >= 1,
          f"张三={named['张三']['sample_count']} 李四={named['李四']['sample_count']}")
    check("learn: reported in detection notes",
          any("人工归属字幕补充声纹" in n for n in project["detection_notes"]))

    # -- pending-role cleanup ---------------------------------------------
    pending = call("POST", f"/api/projects/{pid}/roles", {"name": "待定测试"})
    pending_id = pending["role"]["id"]
    call("POST", f"/api/projects/{pid}/roles_update",
         {"role_id": pending_id, "type": "pending"})
    call("POST", f"/api/projects/{pid}/segments",
         {"segment_id": seg0, "speaker_id": pending_id})
    result = call("POST", f"/api/projects/{pid}/roles_delete_pending", {})
    project = result["project"]
    check("pending-role purge removes the role",
          result["removed"] == 1 and all(r["id"] != pending_id for r in project["roles"]),
          f"removed={result['removed']}")
    purged_seg = next(s for s in project["segments"] if s["id"] == seg0)
    check("pending-role purge releases its cues",
          purged_seg["speaker_id"] is None and purged_seg["status"] == "pending")
    check("pending-role purge keeps named roles",
          {r["name"] for r in project["roles"]} == {"张三", "李四"})
    # restore seg0 so the export checks below see the earlier steady state
    call("POST", f"/api/projects/{pid}/bulk", {"ids": [seg0], "speaker_id": 0, "status": "auto"})

    # -- voiceprint enrolment (builtin space) ------------------------------
    result = call("POST", f"/api/projects/{pid}/enroll",
                  {"role_id": 0, "start": project["segments"][0]["start"],
                   "end": project["segments"][0]["end"], "also_library": True})
    check("enroll: voiceprint stored",
          result["role"]["has_voiceprint"] and result["role"]["sample_count"] >= 1,
          f"embedder={result['role']['embedder']}")
    check("enroll: lands in the builtin space (manual default engine)",
          result["role"]["embedder"] == "builtin", result["role"]["embedder"])
    check("enroll: role promoted to registered", result["role"]["type"] == "registered")
    state = call("GET", "/api/state")
    check("library: entry written with embedder tag",
          len(state["library"]) == 1 and state["library"][0]["name"] == "张三"
          and state["library"][0]["embedder"] == "builtin")
    # Re-enrolling the same role must UPDATE the library row, not duplicate it.
    call("POST", f"/api/projects/{pid}/enroll",
         {"role_id": 0, "start": project["segments"][1]["start"],
          "end": project["segments"][1]["end"], "also_library": True})
    state = call("GET", "/api/state")
    check("library: re-enrol updates instead of duplicating",
          len(state["library"]) == 1, f"{len(state['library'])} rows")

    # -- second project (manual detection) ---------------------------------
    created2 = call("POST", "/api/projects", {
        "media_path": MEDIA, "subtitle_path": SUBTITLE, "name": "e2e-prior",
    })
    pid2 = created2["project"]["id"]
    result = call("POST", f"/api/projects/{pid2}/detect", {"engine": "manual"})
    check("second project: manual detection runs",
          result["project"]["engine"] == "manual"
          and all(s["speaker_id"] is None for s in result["project"]["segments"]))

    # -- exports ------------------------------------------------------------
    formats = ["srt", "vtt", "ass", "script", "jsonl", "rttm", "csv", "hf", "dataset", "report"]
    result = call("POST", f"/api/projects/{pid}/export", {"formats": formats})
    files = {f["format"]: f for f in result["files"]}
    check("export: all 10 files written", len(files) == 10)

    srt = open(files["srt"]["path"], encoding="utf-8").read()
    check("export srt: speaker prefixes", "[张三]" in srt and "[李四]" in srt)
    script = open(files["script"]["path"], encoding="utf-8").read()
    check("export script: speaker name + content lines",
          "张三：" in script and "李四：" in script)
    vtt = open(files["vtt"]["path"], encoding="utf-8").read()
    check("export vtt: voice tags", vtt.startswith("WEBVTT") and "<v 张三>" in vtt)
    ass = open(files["ass"]["path"], encoding="utf-8").read()
    check("export ass: per-speaker styles", "Style: S0_张三" in ass
          and "Style: S1_李四" in ass)
    jsonl = open(files["jsonl"]["path"], encoding="utf-8").read().strip().splitlines()
    records = [json.loads(line) for line in jsonl]
    check("export jsonl: 6 records with required keys",
          len(records) == 6 and all(
              {"start", "end", "text", "speaker", "confidence", "status"} <= set(r)
              for r in records))
    rttm = open(files["rttm"]["path"], encoding="utf-8").read().strip().splitlines()
    check("export rttm: SPEAKER rows with 10 columns and names",
          len(rttm) == 6 and all(len(r.split()) == 10 for r in rttm)
          and any("张三" in r for r in rttm))
    csv_text = open(files["csv"]["path"], encoding="utf-8").read()
    check("export csv: header and rows", csv_text.startswith("id,start,end")
          and "张三" in csv_text)
    hf = json.loads(open(files["hf"]["path"], encoding="utf-8").read())
    check("export hf: segments + speaker list",
          len(hf["segments"]) == 6 and set(hf["speakers"]) == {"张三", "李四"})
    ds = json.loads(open(files["dataset"]["path"], encoding="utf-8").read())
    check("export dataset: role library + annotations",
          len(ds["speakers"]) == 2 and len(ds["annotations"]) == 6
          and ds["speakers"][0]["name"] == "张三")
    report = open(files["report"]["path"], encoding="utf-8").read()
    check("export report: stats and low-confidence sections",
          "说话人统计" in report and "复核报告" in report)

    # -- bilingual import + export ------------------------------------------
    bilingual_sub = os.path.join(BASE, "sample", "bilingual_inline.srt")
    created3 = call("POST", "/api/projects", {
        "media_path": MEDIA, "subtitle_path": bilingual_sub, "name": "e2e-bilingual",
    })
    p3 = created3["project"]
    pid3 = p3["id"]
    check("bilingual: detected", p3["bilingual"] is True, str(p3["bilingual"]))
    check("bilingual: line_mode applied", p3["line_mode"] == "bilingual", str(p3["line_mode"]))
    check("bilingual: original kept in text",
          p3["segments"][0]["text"] == "你好，欢迎来到这次的访谈节目。", p3["segments"][0]["text"])
    check("bilingual: translation separated",
          p3["segments"][0]["translation"].startswith("Hello"), p3["segments"][0]["translation"])

    call("POST", f"/api/projects/{pid3}/detect", {"engine": "manual"})
    # switch to single-sentence interpretation and back (raw lines are preserved)
    r = call("POST", f"/api/projects/{pid3}/line_mode", {"mode": "join"})
    joined_text = r["project"]["segments"][0]["text"]
    check("line_mode join: lines merged into one sentence",
          "你好" in joined_text and "Hello" in joined_text
          and "\n" not in joined_text, joined_text[:50])
    check("line_mode join: no separate translation field",
          not r["project"]["segments"][0]["translation"])
    r = call("POST", f"/api/projects/{pid3}/line_mode", {"mode": "bilingual"})
    check("line_mode bilingual: reversible back to split",
          r["project"]["segments"][0]["translation"].startswith("Hello")
          and r["project"]["segments"][0]["text"] == "你好，欢迎来到这次的访谈节目。")

    r = call("POST", f"/api/projects/{pid3}/export",
             {"formats": ["srt", "jsonl", "csv"],
              "options": {"text_field": "both", "filename_stem": "bi"}})
    files3 = {f["format"]: f for f in r["files"]}
    srt_bi = open(files3["srt"]["path"], encoding="utf-8").read()
    check("export bilingual srt: both languages present",
          "你好，欢迎来到" in srt_bi and "Hello and welcome" in srt_bi)
    r = call("POST", f"/api/projects/{pid3}/export",
             {"formats": ["srt", "jsonl"],
              "options": {"text_field": "translation", "filename_stem": "bi-tr"}})
    files3b = {f["format"]: f for f in r["files"]}
    srt_tr = open(files3b["srt"]["path"], encoding="utf-8").read()
    check("export translation-only srt: no source text",
          "Hello and welcome" in srt_tr and "你好" not in srt_tr)
    jsonl_tr = [json.loads(l) for l in
                open(files3b["jsonl"]["path"], encoding="utf-8").read().splitlines()]
    check("export translation-only jsonl: text is translation, source_text kept",
          jsonl_tr[0]["text"].startswith("Hello")
          and jsonl_tr[0]["source_text"].startswith("你好"))

    # -- two-file bilingual import -----------------------------------------
    second = os.path.join(BASE, "sample", "bilingual_en.srt")
    created4 = call("POST", "/api/projects", {
        "media_path": MEDIA, "subtitle_path": SUBTITLE,
        "second_subtitle_path": second, "name": "e2e-twofile",
    })
    p4 = created4["project"]
    check("two-file: translation merged by overlap",
          p4["segments"][0]["translation"].startswith("Hello"),
          str(p4["segments"][0].get("translation"))[:40])
    call("DELETE", f"/api/projects/{p4['id']}")
    call("DELETE", f"/api/projects/{pid3}")

    # downloads endpoint
    with urllib.request.urlopen(
        f"{ROOT}/api/projects/{pid}/export/{files['jsonl']['filename']}"
    ) as response:
        check("download endpoint serves files", response.status == 200)

    # -- cleanup -------------------------------------------------------------
    call("DELETE", f"/api/projects/{pid}")
    call("DELETE", f"/api/projects/{pid2}")
    remaining = call("GET", "/api/projects")["projects"]
    check("cleanup: projects deleted", len(remaining) == 0)


if __name__ == "__main__":
    main()
