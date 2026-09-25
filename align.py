"""Align subtitle cues with detected speaker turns (time-interval overlap).

Rule set:
  * single speaker overlap   -> assign directly
  * several speakers overlap -> take the largest total overlap, keep the
                                overlap ratio as confidence
  * no overlap at all        -> leave ``pending`` with confidence 0
"""

from __future__ import annotations


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
    auto_min_confidence: float = 0.85,
    reject_ambiguous: bool = True,
    conflicted_clusters: set[int] | None = None,
) -> tuple[list[dict], list[str]]:
    """Write ``speaker_id`` / ``confidence`` / ``status`` / ``cluster`` on each cue.

    ``cluster_to_role`` maps a diarization cluster id to a role id. Clusters left
    out of the map have no role, so their cues stay 待定 rather than being
    assigned to a guess.
    """
    notes: list[str] = []
    no_overlap = 0
    unmapped = 0
    ambiguous = 0
    already_manual = 0
    hints_applied = 0
    name_to_role = name_to_role or {}
    conflicted_clusters = conflicted_clusters or set()

    for segment in segments:
        # Never silently discard a human decision unless explicitly asked.
        if segment.get("status") == "manual" and not overwrite_manual:
            already_manual += 1
            continue
        segment["ambiguous"] = False
        segment.pop("note", None)

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
            no_overlap += 1
            continue

        ranked = sorted(totals.items(), key=lambda item: -item[1])
        best_cluster, best_overlap = ranked[0]
        if len(ranked) > 1 and ranked[1][1] > 0:
            ratio = ranked[1][1] / best_overlap
            if ratio >= 0.75:
                segment["ambiguous"] = True
                ambiguous += 1

        segment["cluster"] = int(best_cluster)
        segment["confidence"] = round(min(1.0, best_overlap / duration), 3)
        role_id = cluster_to_role.get(int(best_cluster))
        if int(best_cluster) in conflicted_clusters:
            segment["speaker_id"] = None
            segment["status"] = "pending"
            segment["note"] = "该聚类包含多位已人工确认的说话人"
        elif role_id is None or (reject_ambiguous and segment["ambiguous"]) \
                or segment["confidence"] < auto_min_confidence:
            segment["speaker_id"] = None
            segment["status"] = "pending"
            if role_id is None:
                unmapped += 1
        else:
            segment["speaker_id"] = role_id
            segment["status"] = "auto"

    if hints_applied:
        notes.append(f"{hints_applied} 条字幕使用了文件自带的说话人标记。")
    if no_overlap:
        notes.append(f"{no_overlap} 条字幕与说话人时间段无重叠，已标记为待定。")
    if unmapped:
        notes.append(f"{unmapped} 条字幕对应的聚类尚未映射到角色，已标记为待定。")
    if ambiguous:
        notes.append(f"{ambiguous} 条字幕与多个说话人时间接近（重叠接近），建议人工复核。")
    if already_manual:
        notes.append(f"保留了 {already_manual} 条已人工归属的字幕。")
    return segments, notes
