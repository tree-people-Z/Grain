"""Voiceprint enrolment maths: running mean, bounding and robustness.

Pure Python (no torch / no audio): the extractor is stubbed, so these exercise
only ``roles.enroll`` and ``roles.merge_roles``. A regression here is silent and
serious — a wrong blend leaves every role's voiceprint either dominated by its
first sample or, past a few hundred enrollments, mapped to ``null`` and dead.

Run:  python tests/test_enroll.py
"""

from __future__ import annotations

import math
import os
import sys
import tempfile

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
os.environ["SSP_DATA_DIR"] = tempfile.mkdtemp(prefix="ssp_enroll_")

import audio_features as af  # noqa: E402
import roles as rolelib  # noqa: E402

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


def _role():
    return rolelib.new_role(0, "测试", "#ffffff", role_type="registered")


def _enroll(role, vector, start=0.0):
    return rolelib.enroll(role, "unused.wav", start, start + 1.0,
                          embedder="pyannote", extractor=lambda s, e: list(vector))


def _close(a, b, tol=1e-9):
    return len(a) == len(b) and all(abs(x - y) <= tol for x, y in zip(a, b))


def main() -> int:
    print("-- running mean, not an exponential sum --")
    role = _role()
    _enroll(role, [1.0, 0.0, 0.0])
    _enroll(role, [0.0, 1.0, 0.0])
    _enroll(role, [0.0, 0.0, 1.0])
    check("three samples average to the mean direction",
          _close(role["embedding"], [1 / 3, 1 / 3, 1 / 3]), str(role["embedding"]))

    # The old bug multiplied the running vector by the sample count, so later
    # samples barely moved the direction: v3=[2,0] used to give [2,1] not [3,1].
    role = _role()
    _enroll(role, [1.0, 0.0])
    _enroll(role, [0.0, 1.0])
    _enroll(role, [2.0, 0.0])
    check("a later sample still pulls the mean",
          _close(role["embedding"], [1.0, 1 / 3]), str(role["embedding"]))
    check("samples list records every enrol",
          len(role["samples"]) == 3, str(len(role["samples"])))

    print("-- bounded: hundreds of enrolments stay finite --")
    role = _role()
    for index in range(400):
        _enroll(role, [1.0, 0.0], start=float(index))
    check("embedding is finite after 400 enrols",
          af.is_finite_vector(role["embedding"]), str(role["embedding"][:3]))
    check("magnitude stays bounded",
          math.sqrt(sum(x * x for x in role["embedding"])) <= 2.0,
          str(role["embedding"]))

    print("-- fixed window: recent samples take over past the cap --")
    role = _role()
    for index in range(60):
        _enroll(role, [1.0, 0.0], start=float(index))
    for index in range(200):
        _enroll(role, [0.0, 1.0], start=float(100 + index))
    check("latest two hundred samples dominate the window",
          role["embedding"][1] > 0.9, str(role["embedding"]))
    check("cap is enforced in the averaging divisor",
          rolelib.MAX_ROLE_SAMPLES == 50, str(rolelib.MAX_ROLE_SAMPLES))

    print("-- non-finite safety --")
    role = _role()
    role["embedding"] = [float("inf"), 0.0]
    _enroll(role, [1.0, 1.0])
    check("a poisoned voiceprint is replaced with the fresh finite sample",
          _close(role["embedding"], [1.0, 1.0]), str(role["embedding"]))

    role = _role()
    _enroll(role, [3.0, 4.0])
    _enroll(role, [3.0, 4.0])
    check("finite enrols never become non-finite",
          af.is_finite_vector(role["embedding"]), str(role["embedding"]))

    print("-- merge_roles blends like a weighted mean --")
    target = _role()
    target["embedding"] = [1.0, 0.0]
    target["samples"] = [{"start": 0.0, "end": 1.0, "source": "a"}] * 3
    source = _role()
    source["embedding"] = [0.0, 1.0]
    source["samples"] = [{"start": 2.0, "end": 3.0, "source": "b"}]
    rolelib.merge_roles(target, source)
    check("weighted by sample count, not summed",
          _close(target["embedding"], [0.75, 0.25]), str(target["embedding"]))
    check("samples are unioned", len(target["samples"]) == 4, str(len(target["samples"])))

    # Repeated merges must not grow the magnitude towards overflow.
    for _ in range(300):
        other = _role()
        other["embedding"] = [1.0, 1.0]
        rolelib.merge_roles(target, other)
    check("repeated merges stay finite and bounded",
          af.is_finite_vector(target["embedding"])
          and max(abs(x) for x in target["embedding"]) <= 2.0,
          str(target["embedding"]))

    print(f"\n{PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
