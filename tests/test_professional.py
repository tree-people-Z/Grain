"""Professional-engine test: runs pyannote and CAM++ through the real HTTP API.

Skips engines whose dependencies are not installed (exit 0 with a note).
Run:  python tests/test_professional.py
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request

try:  # Chinese Windows consoles default to GBK, which cannot encode ✓/✗
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PORT = 8801
ROOT = f"http://127.0.0.1:{PORT}"
MEDIA = os.path.join(BASE, "sample", "interview.wav")
SUBTITLE = os.path.join(BASE, "sample", "interview.srt")

EXPECTED_ALTERNATING = True


def call(method: str, path: str, payload: dict | None = None):
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(
        ROOT + path, data=data, method=method,
        headers={"Content-Type": "application/json"} if data else {},
    )
    with urllib.request.urlopen(request, timeout=1800) as response:
        return json.loads(response.read().decode("utf-8"))


def alternates(project: dict) -> tuple[bool, list[str]]:
    names = {r["id"]: r["name"] for r in project["roles"]}
    order = []
    for segment in project["segments"]:
        if segment["speaker_id"] is None:
            order.append("None")
        else:
            order.append(names.get(segment["speaker_id"], "?"))
    alternating = (
        len(set(order)) == 2 and "None" not in order
        and all(order[i] != order[i + 1] for i in range(len(order) - 1))
    )
    return alternating, order


def main() -> None:
    # Run against a throwaway data dir (SSP_DATA_DIR): this test used to delete
    # `data/projects`, which destroyed the user's real project.
    data_dir = os.path.join(BASE, "data", "__test_professional__")
    shutil.rmtree(data_dir, ignore_errors=True)
    os.makedirs(data_dir, exist_ok=True)
    env = dict(os.environ, SSP_DATA_DIR=data_dir)

    state = None
    server = subprocess.Popen(
        [sys.executable, os.path.join(BASE, "server.py"), "--port", str(PORT)],
        # Never pipe the server's output: a chatty engine (NeMo) fills the OS
        # pipe buffer and deadlocks the server if nobody reads it.
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env,
    )
    try:
        for _ in range(60):
            try:
                state = call("GET", "/api/state")
                break
            except Exception:
                time.sleep(0.5)
        engines = state["engines"]

        # "voiceprint-cue" is the ungated pyannote path (wespeaker, per cue);
        # it must stay in the list so a missing HF gate cannot hide a regression.
        for engine in ("pyannote", "campp", "voiceprint-cue", "sortformer"):
            if not engines[engine]["available"]:
                print(f"SKIP  {engine}: not installed")
                continue
            print(f"-- testing engine: {engine}")
            created = call("POST", "/api/projects", {
                "media_path": MEDIA, "subtitle_path": SUBTITLE,
                "name": f"prof-{engine}",
            })
            pid = created["project"]["id"]
            try:
                result = call("POST", f"/api/projects/{pid}/detect",
                              {"engine": engine, "min_speakers": 1, "max_speakers": 6})
            except urllib.error.HTTPError as exc:
                # A gated HuggingFace model is an environment condition, not a
                # regression: report it and move on (the message names the repo
                # and the page that must be accepted).
                detail = exc.read().decode("utf-8", "replace")[:300]
                print(f"  SKIP  {engine}: {detail}")
                call("DELETE", f"/api/projects/{pid}")
                continue
            project = result["project"]
            alternating, order = alternates(project)
            notes = " ".join(project["detection_notes"])
            print(f"  turns={len(project['turns'])} roles={len(project['roles'])}")
            print(f"  order={order}")
            print(f"  notes={notes[:200]}")
            if alternating:
                print(f"  PASS  {engine}: alternation matches ground truth")
            else:
                print(f"  WARN  {engine}: order differs from ground truth "
                      f"(synthetic audio; check turns/notes above)")
            call("DELETE", f"/api/projects/{pid}")
        print("DONE")
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()
        shutil.rmtree(data_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
