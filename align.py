"""Align subtitle cues with detected speaker turns (time-interval overlap).

Rule set (matching the project plan):
  * single speaker overlap        -> assign directly
  * several speakers overlap      -> take the largest total overlap, keep the
                                     overlap ratio as confidence
  * no overlap at all             -> leave ``pending`` with confidence 0
"""

from __future__ import annotations

import audio_features as af


def _intersect(start_a: float, end_a: float, start_b: float, end_b: float) -> float:
    return max(0.0, min(end_a, end_b) - max(start_a, start_b))


def overlap_by_cluster(segment: dict, turns: list[dict]) -> dict[int, float]:
    """Total overlap seconds per cluster for one subtitle cue."""
    totals: dict[int, float] = {}
    for turn in turns:
        cluster = turn.get("cluster")
        if cluster is None:
            continue
        overlap = _intersect(segment["start"], segment["end"], turn["start"], turn["end"])
        if overlap > 0:
            cluster = int(cluster)
            totals[cluster] = totals.get(cluster, 0.0) + overlap
    return totals


def align(
    segments: list[dict],
    turns: list[dict],
    cluster_to_role: dict[int, int],
    name_to_role: dict[str, int] | None = None,
    overwrite_manual: bool = False,
) -> tuple[list[dict], list[str]]:
    """Write ``speaker_id`` / ``confidence`` / ``status`` / ``cluster`` on each cue.

    ``cluster_to_role`` maps a diarization cluster id to a role id (already
    resolved, including auto-created "待定" roles).
    """
    notes: list[str] = []
    unknown = 0
    ambiguous = 0
    already_manual = 0
    hints_applied = 0
    name_to_role = name_to_role or {}

    for segment in segments:
        # Never silently discard a human decision unless explicitly asked.
        if segment.get("status") == "manual" and not overwrite_manual:
            already_manual += 1
            continue

        hint = segment.get("speaker_name_hint")
        if hint and hint in name_to_role:
            segment["speaker_id"] = name_to_role[hint]
            segment["status"] = "manual"
            segment["confidence"] = None
            segment["cluster"] = None
            segment["note"] = "来自字幕文件自带的说话人标记"
            hints_applied += 1
            continue

        duration = max(1e-6, segment["end"] - segment["start"])
        totals = overlap_by_cluster(segment, turns)

        if not totals:
            segment["speaker_id"] = None
            segment["confidence"] = 0.0
            segment["status"] = "pending"
            segment.pop("cluster", None)
            unknown += 1
            continue

        ranked = sorted(totals.items(), key=lambda item: -item[1])
        best_cluster, best_overlap = ranked[0]
        if len(ranked) > 1 and ranked[1][1] > 0:
            ratio = ranked[1][1] / best_overlap
            if ratio >= 0.75:
                ambiguous += 1

        segment["cluster"] = int(best_cluster)
        segment["confidence"] = round(min(1.0, best_overlap / duration), 3)
        role_id = cluster_to_role.get(int(best_cluster))
        if role_id is None:
            segment["speaker_id"] = None
            segment["status"] = "pending"
            unknown += 1
        else:
            segment["speaker_id"] = role_id
            segment["status"] = "auto"

    if hints_applied:
        notes.append(f"{hints_applied} 条字幕使用了文件自带的说话人标记。")
    if unknown:
        notes.append(f"{unknown} 条字幕与说话人时间段无重叠，已标记为待定。")
    if ambiguous:
        notes.append(f"{ambiguous} 条字幕与多个说话人时间接近（重叠接近），建议人工复核。")
    if already_manual:
        notes.append(f"保留了 {already_manual} 条已人工归属的字幕。")
    return segments, notes


# Score floor / margin for the voiceprint refinement. A same-speaker cue vs its
# cluster centroid is usually well above 0.5 cosine in the pyannote space; the
# margin keeps a confident temporal decision unless the voice clearly disagrees.
VOICEPRINT_FLOOR = 0.50
VOICEPRINT_MARGIN = 0.06


def refine_with_voiceprints(
    segments: list[dict],
    clusters: dict,
    cluster_to_role: dict[int, int],
    embedder,
    floor: float = VOICEPRINT_FLOOR,
    margin: float = VOICEPRINT_MARGIN,
    min_duration: float = 0.20,
    overwrite_manual: bool = False,
) -> tuple[int, int, list[str]]:
    """Second pass: re-assign cues by their own voiceprint vs cluster centroids.

    The overlap pass is robust to subtitle timing but blind to *which* voice a
    cue sounds like, so a cue that barely clips a turn (or clips two) gets the
    wrong speaker. Scoring each cue's voiceprint against every cluster centroid
    and demanding that the winner beat the current choice by ``margin`` fixes
    most of those while leaving confident overlap decisions alone.

    Returns ``(reassigned, filled, notes)``.
    """
    centroids = []
    for cluster_id, cluster in (clusters or {}).items():
        vector = (cluster or {}).get("embedding") or []
        role_id = cluster_to_role.get(int(cluster_id))
        if vector and role_id is not None:
            centroids.append((int(cluster_id), role_id, vector))
    if not centroids:
        return 0, 0, []

    by_cluster = {cid: vec for cid, _, vec in centroids}
    reassigned = filled = 0
    for segment in segments:
        if segment.get("speaker_name_hint"):
            continue  # a human/name hint already decided this one
        if segment.get("status") == "manual" and not overwrite_manual:
            continue
        if segment["end"] - segment["start"] < min_duration:
            continue
        vector = embedder(segment["start"], segment["end"])
        if not vector:
            continue
        scored = sorted(
            ((af.cosine(vector, vec), cid) for cid, _, vec in centroids),
            key=lambda item: -item[0],
        )
        best_score, best_cluster = scored[0]
        if best_score < floor:
            continue
        current_cluster = segment.get("cluster")
        if current_cluster is not None and int(current_cluster) == best_cluster:
            continue
        current_score = None
        if current_cluster is not None and int(current_cluster) in by_cluster:
            current_score = af.cosine(vector, by_cluster[int(current_cluster)])
        if current_score is not None and best_score - current_score < margin:
            continue  # the voice does not clearly contradict the timing vote

        was_assigned = segment.get("speaker_id") is not None
        segment["cluster"] = best_cluster
        segment["speaker_id"] = cluster_to_role[best_cluster]
        segment["status"] = "auto"
        segment["confidence"] = round(best_score, 3)
        segment["note"] = "声纹精修"
        if was_assigned:
            reassigned += 1
        else:
            filled += 1

    notes = []
    if reassigned:
        notes.append(f"声纹精修修正了 {reassigned} 条时间段重叠判断有误的字幕。")
    if filled:
        notes.append(f"声纹精修为 {filled} 条无重叠字幕补全了归属。")
    return reassigned, filled, notes


def match_clusters(a_clusters: dict, b_clusters: dict,
                   threshold: float = 0.50) -> dict[int, int]:
    """Greedy one-to-one map of B clusters onto A clusters by centroid cosine.

    Used by two-engine consensus to line up the two engines' speaker sets before
    comparing their per-cue decisions.
    """
    a_vectors = {int(cid): ((info or {}).get("embedding") or [])
                 for cid, info in (a_clusters or {}).items()}
    pairs = []
    for bcid, info in (b_clusters or {}).items():
        vector = (info or {}).get("embedding") or []
        if not vector:
            continue
        for acid, avec in a_vectors.items():
            if not avec:
                continue
            score = af.cosine(vector, avec)
            if score >= threshold:
                pairs.append((score, int(bcid), acid))
    pairs.sort(key=lambda item: -item[0])
    used_a: set[int] = set()
    used_b: set[int] = set()
    mapping: dict[int, int] = {}
    for _, bcid, acid in pairs:
        if bcid in used_b or acid in used_a:
            continue
        mapping[bcid] = acid
        used_b.add(bcid)
        used_a.add(acid)
    return mapping
