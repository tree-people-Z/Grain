"""Unit tests for the detection helpers that keep accuracy stable.

Pure Python (no torch): the sweep selector, the silence-trimmer and the
per-feature-space prior matching. These are the pieces that decide *which*
speaker a cue gets, so a regression here silently degrades every detection.

Run:  python tests/test_detection_utils.py
"""

from __future__ import annotations

import math
import os
import sys
import tempfile

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
# Isolate any project-level storage this test touches.
os.environ["SSP_DATA_DIR"] = tempfile.mkdtemp(prefix="ssp_detutils_")

import audio_features as af  # noqa: E402
import diarize  # noqa: E402
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


def main() -> int:
    print("-- sweep selection --")
    # Six points in two clear groups. k>=3 forces singleton clusters, which a
    # plain argmax-silhouette sweep would happily prefer; the selector must not.
    group_a = [[0.98, 0.02, 0.0], [0.99, 0.01, 0.0], [0.97, 0.03, 0.0]]
    group_b = [[0.02, 0.98, 0.0], [0.01, 0.99, 0.0], [0.03, 0.97, 0.0]]
    vectors = group_a + group_b
    candidates = []
    for k in range(1, 7):
        labels = [i % k for i in range(6)]
        candidates.append((k, {"k": k}, vectors, labels))
    chosen = diarize._pick_sweep(candidates)
    check("sweep rejects singleton over-segmentation", chosen is not None and chosen[0] == 2,
          f"picked k={chosen[0] if chosen else None}")

    # A genuine third speaker (clearly separated) must still be found.
    extra = [[0.0, 0.0, 1.0], [0.0, 0.0, 0.99]]
    vectors3 = vectors + extra
    candidates3 = []
    labels3 = {1: [0] * 8, 2: [0, 0, 0, 0, 0, 0, 1, 1], 3: [0, 0, 0, 1, 1, 1, 2, 2]}
    for k in (1, 2, 3):
        candidates3.append((k, {"k": k}, vectors3, labels3[k]))
    chosen3 = diarize._pick_sweep(candidates3)
    check("sweep keeps a clearly distinct speaker", chosen3 is not None and chosen3[0] == 3,
          f"picked k={chosen3[0] if chosen3 else None}")

    print("-- silence trim --")
    rate = 16000
    tone = [0.5 * math.sin(2 * math.pi * 220 * t / rate) for t in range(int(1.5 * rate))]
    signal = [0.0] * int(0.4 * rate) + tone + [0.0] * int(0.4 * rate)
    start, end = af.speech_trim(signal, rate, 0.0, len(signal) / rate)
    check("trim removes silent head/tail", start > 0.3 and end < 2.0,
          f"{start:.2f}..{end:.2f}")
    check("silent span is left untouched",
          af.speech_trim([0.0] * rate, rate, 0.0, 1.0) == (0.0, 1.0))
    check("very short span is left untouched",
          af.speech_trim(signal, rate, 0.0, 0.1) == (0.0, 0.1))

    print("-- prior matching (per feature space) --")
    clusters = {0: {"embedding": [1.0, 0.0, 0.0]}, 1: {"embedding": [0.0, 1.0, 0.0]}}
    role_a = {"id": 0, "name": "A", "type": "named", "embedding": [1.0, 0.0, 0.0]}
    role_b = {"id": 1, "name": "B", "type": "named", "embedding": [0.0, 1.0, 0.0]}
    mapping, notes = rolelib.map_clusters(clusters, [role_a, role_b],
                                          threshold=None, embedder="pyannote")
    check("auto threshold matches both clusters",
          mapping[0]["role_id"] == 0 and mapping[1]["role_id"] == 1, str(mapping))
    check("auto threshold is the pyannote default",
          any("0.62" in n and "自动校准" in n for n in notes), str(notes))
    check("pyannote default differs from builtin",
          rolelib.match_defaults("pyannote")[0] < rolelib.match_defaults("builtin")[0],
          f"{rolelib.match_defaults('pyannote')} vs {rolelib.match_defaults('builtin')}")

    # A runner-up that is nearly as close as the winner must not produce a
    # confident assignment: the calibrated margin sends it to review instead.
    near = {"id": 2, "name": "near", "type": "named", "embedding": [0.97, 0.24, 0.0]}
    mapping2, _ = rolelib.map_clusters(clusters, [role_a, near],
                                       threshold=None, embedder="pyannote")
    check("close runner-up defers to review (margin)",
          mapping2[0]["role_id"] is None and mapping2[1]["role_id"] is None, str(mapping2))

    print("-- enrolment feature space --")
    # A voiceprint must be stored in the space detection will use: the project's
    # engine when known, otherwise the configured default engine (never builtin
    # just because no detection has run yet).
    import project as store
    store.save_settings({"default_engine": "pyannote"})
    have = lambda engine, _refresh=False: engine in {"pyannote", "campp"}
    check("engine known & available wins",
          store.enroll_engine({"engine": "campp"}, available=have) == "campp")
    check("no engine yet -> default engine",
          store.enroll_engine({"engine": None}, available=have) == "pyannote")
    check("manual engine -> default engine",
          store.enroll_engine({"engine": "manual"}, available=have) == "pyannote")
    only_py = lambda engine, _refresh=False: engine == "pyannote"
    check("unavailable project engine -> default engine",
          store.enroll_engine({"engine": "campp"}, available=only_py) == "pyannote")
    none = lambda engine, _refresh=False: False
    check("nothing available -> builtin (manual)",
          store.enroll_engine({"engine": None}, available=none) == "manual")

    print("-- non-finite embeddings --")
    # A degenerate crop makes the neural embedder return NaN; that must never
    # poison a centroid or be matched on.
    check("finite vector accepted", af.is_finite_vector([1.0, 2.0]) is True)
    check("empty vector rejected", af.is_finite_vector([]) is False)
    check("NaN rejected", af.is_finite_vector([float("nan"), 1.0]) is False)
    check("Inf rejected", af.is_finite_vector([float("inf")]) is False)
    check("non-number rejected", af.is_finite_vector([1.0, "x"]) is False)
    mean = af.mean_vector([[1.0, 2.0], [float("nan"), 0.0], [3.0, 4.0]])
    check("mean_vector ignores NaN vectors", mean == [2.0, 3.0], str(mean))
    check("cosine with NaN is 0", af.cosine([float("nan")] * 3, [1.0, 2.0, 3.0]) == 0.0)
    bad_clusters = {0: {"embedding": [float("nan")] * 3}, 1: {"embedding": [1.0, 0.0, 0.0]}}
    bad_mapping, bad_notes = rolelib.map_clusters(bad_clusters, [role_a],
                                                  threshold=None, embedder="pyannote")
    check("non-finite cluster is not matched", bad_mapping[0]["matched"] is False
          and bad_mapping[1]["matched"] is True, str(bad_mapping))

    # Same person listed twice (a project role plus its global-library twin, or
    # several accumulated library copies) must not blow the runner-up margin.
    twin_clusters = {0: {"embedding": [1.0, 0.0, 0.0]}}
    proj_role = {"id": 7, "name": "灯", "type": "named", "embedding": [1.0, 0.0, 0.0]}
    lib_twin = {"id": "library:x", "name": " 灯 ", "type": "registered",
                "embedding": [0.99, 0.01, 0.0]}
    twin_map, _ = rolelib.map_clusters(twin_clusters, [proj_role, lib_twin],
                                       threshold=None, embedder="pyannote")
    check("duplicate same-name candidate does not block the match",
          twin_map[0]["matched"] and twin_map[0]["role_id"] == 7, str(twin_map))

    print("-- pending-role reuse across detections --")
    # Re-running detection must reuse pending roles, not pile up 待定角色1,2,3….
    proj = {"roles": [], "segments": [], "cluster_to_role": {}}
    a = store._ensure_pending_role(proj, "pyannote", {})
    proj["cluster_to_role"] = {0: a}          # previous run's mapping

    def pending_ids(p):
        return [r["id"] for r in p["roles"] if r.get("type") == "pending"]

    b = store._ensure_pending_role(proj, "pyannote", {})
    check("re-detection reuses the pending role",
          a == b and len(pending_ids(proj)) == 1, f"a={a} b={b} pending={pending_ids(proj)}")
    # Two clusters in one run still get distinct pending roles.
    c = store._ensure_pending_role(proj, "pyannote", {0: a})
    check("clusters in one run get distinct pending roles",
          c != a and len(pending_ids(proj)) == 2, f"a={a} c={c}")
    # A later run with fewer clusters must not leave the surplus behind.
    dropped = store._prune_pending_roles(proj, keep_ids={a})
    check("surplus pending role is pruned",
          dropped == 1 and pending_ids(proj) == [a], f"dropped={dropped} pending={pending_ids(proj)}")
    # A pending role still referenced by a manual assignment is kept.
    proj2 = {"roles": [], "segments": [], "cluster_to_role": {}}
    m = store._ensure_pending_role(proj2, "pyannote", {})
    proj2["segments"].append({"id": 0, "speaker_id": m, "status": "manual"})
    check("manual-referenced pending role is kept",
          store._prune_pending_roles(proj2, keep_ids=set()) == 0
          and pending_ids(proj2) == [m])

    print("-- naming auto-registers the voiceprint --")
    named = {"roles": [rolelib.new_role(0, "灯", "#ffffff", role_type="named")],
             "segments": [], "cluster_to_role": {}}
    store.update_role(named, 0, name="高松灯")            # no embedding yet
    check("no voiceprint -> nothing registered", len(store.load_library()) == 0)
    named["roles"][0]["embedding"] = [1.0, 2.0]
    store.update_role(named, 0, name="高松灯")
    lib = store.load_library()
    check("naming a role registers its voiceprint",
          len(lib) == 1 and lib[0]["name"] == "高松灯", str([e["name"] for e in lib]))
    # A run that matches every cluster to a real role uses no pending roles:
    # all leftovers must be removed automatically.
    proj3 = {"roles": [], "segments": [], "cluster_to_role": {}}
    store._ensure_pending_role(proj3, "pyannote", {})
    check("no pending used -> all old pending roles pruned",
          store._prune_pending_roles(proj3, keep_ids={99}) == 1 and pending_ids(proj3) == [])

    print(f"\n{PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
