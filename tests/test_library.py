"""Global role library: upsert keyed by (name, feature space) and de-duplication.

Pure Python (no torch). The library is shared across projects, so a regression
here silently pollutes every future detection's prior matching.

Run:  python tests/test_library.py
"""

from __future__ import annotations

import os
import sys
import tempfile

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
os.environ["SSP_DATA_DIR"] = tempfile.mkdtemp(prefix="ssp_lib_")

import project as store  # noqa: E402

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


def main() -> int:
    print("-- upsert keyed by (name, feature space) --")
    e1, a1 = store.add_library_role(
        "高松灯", "#ff0000", embedding=[1.0, 0.0],
        samples=[{"start": 1.0, "end": 2.0, "source": "p1"}], embedder="pyannote")
    check("first add creates", a1 == "created" and len(store.load_library()) == 1)
    e2, a2 = store.add_library_role(
        " 高松灯 ", "#00ff00", embedding=[0.0, 1.0],
        samples=[{"start": 3.0, "end": 4.0, "source": "p2"}], embedder="pyannote")
    check("same name+space updates in place",
          a2 == "updated" and len(store.load_library()) == 1)
    check("library_id kept", e2["library_id"] == e1["library_id"])
    check("embedding replaced", e2["embedding"] == [0.0, 1.0], str(e2["embedding"]))
    check("samples merged (deduped)", len(e2["samples"]) == 2, str(len(e2["samples"])))
    check("colour kept from the first entry", e2["color"] == "#ff0000", e2["color"])

    e3, _ = store.add_library_role("高松灯", None, embedding=[], samples=[],
                                   embedder="pyannote")
    check("empty embedding does not wipe the stored one",
          e3["embedding"] == [0.0, 1.0], str(e3["embedding"]))

    _, a4 = store.add_library_role("高松灯", None, embedding=[1.0, 2.0, 3.0],
                                   samples=[], embedder="campp")
    check("a different feature space is its own entry",
          a4 == "created" and len(store.load_library()) == 2)

    print("-- de-duplicate --")
    lib = store.load_library()
    lib.append({
        "library_id": "lib_dup01", "name": "高松灯", "color": "#abcdef",
        "embedder": "pyannote", "embedding": [1.0, 1.0],
        "samples": [{"start": 5.0, "end": 6.0, "source": "p3"}],
        "created": "2020-01-01T00:00:00", "updated": "2020-01-01T00:00:00",
    })
    store.save_library(lib)
    result = store.dedupe_library()
    check("dedupe collapses the duplicate", result == {"removed": 1, "kept": 2}, str(result))
    merged = next(r for r in store.load_library()
                  if r["name"] == "高松灯" and r["embedder"] == "pyannote")
    check("keeps the first library_id", merged["library_id"] == e1["library_id"])
    check("unions samples", len(merged["samples"]) == 3, str(len(merged["samples"])))
    # [0,1] weighted by 2 samples, [1,1] weighted by 1 -> [1/3, 1].
    check("embedding is sample-count weighted",
          abs(merged["embedding"][0] - 1 / 3) < 1e-9
          and abs(merged["embedding"][1] - 1.0) < 1e-9, str(merged["embedding"]))
    check("second call is a no-op", store.dedupe_library()["removed"] == 0)

    print(f"\n{PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
