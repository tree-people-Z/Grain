"""Role (speaker) library: colours, voiceprint enrolment, cluster matching."""

from __future__ import annotations

import audio_features as af

PALETTE = [
    "#FF6B6B", "#4ECDC4", "#FFD166", "#8E7CFF", "#06D6A0",
    "#F78C6B", "#5DA9E9", "#E56B9F", "#B5E48C", "#C792EA",
]
PENDING_COLOR = "#B9BEC7"
UNKNOWN_NAME = "待定"

# How many samples a single voiceprint averages over. The running mean is
# incremental (no per-sample vectors are stored), so without a cap a long run of
# enrollments would let the earliest samples keep most of the weight and the
# magnitude grow without bound. Past this count the update becomes a fixed-window
# moving average (weight 1/MAX_ROLE_SAMPLES per new sample), which bounds both.
MAX_ROLE_SAMPLES = 50


def next_color(existing: list[str]) -> str:
    for color in PALETTE:
        if color not in existing:
            return color
    return PALETTE[len(existing) % len(PALETTE)]


def new_role(role_id: int, name: str, color: str, role_type: str = "registered",
             embedder: str = "builtin") -> dict:
    return {
        "id": role_id,
        "name": name,
        "color": color,
        "type": role_type,           # registered | pending | named | ignored
        "embedder": embedder,        # voiceprint feature space: builtin | pyannote | campp
        "embedding": [],
        "samples": [],
        "cluster": None,
    }


def role_summary(role: dict) -> dict:
    """JSON-safe view for the client (drops the raw embedding vector)."""
    return {
        "id": role["id"],
        "name": role["name"],
        "color": role["color"],
        "type": role["type"],
        "embedder": role.get("embedder", "builtin"),
        "has_voiceprint": bool(role.get("embedding")),
        "sample_count": len(role.get("samples") or []),
        "cluster": role.get("cluster"),
    }


# --- voiceprint enrolment ---------------------------------------------------

def enroll(role: dict, wav_path: str, start: float, end: float, source: str = "",
           embedder: str = "builtin", extractor=None) -> str:
    """Extract an acoustic embedding for ``[start, end]`` and add it to the role.

    ``extractor`` is the engine-bound voiceprint extractor when available;
    otherwise the built-in feature space is used and tagged as such. The role's
    ``embedder`` tag always records the space the stored vector actually lives in.
    Returns a short note for the caller's message list.
    """
    if end - start < 0.20:
        raise ValueError("录入片段过短（至少 0.2 秒）")
    note = ""
    vector: list[float] = []
    if extractor is not None:
        try:
            vector = extractor(start, end)
        except Exception as exc:
            note = f"引擎声纹提取失败，改用内置特征：{str(exc)[:120]}"
    if not af.is_finite_vector(vector):
        signal, rate = af.read_wav(wav_path, 16000)
        vector = af.segment_embedding(signal, rate, start, end)
        embedder = "builtin"
    if not af.is_finite_vector(vector):
        raise ValueError("无法从该片段提取声学特征，请换一段更长的清晰语音")
    role["embedder"] = embedder
    samples = list(role.get("samples") or [])
    role["samples"] = samples
    samples.append({"start": round(start, 3), "end": round(end, 3), "source": source})

    # Running mean of every enrolled sample, weighted equally: a voiceprint is
    # the average direction, so clean samples only average out noise. The update
    # is done in place from the previous mean, so the stored vector is always a
    # bounded combination of the samples rather than their exponential sum. Past
    # MAX_ROLE_SAMPLES the divisor stops growing, giving a fixed-window average.
    existing = role.get("embedding") or []
    if af.is_finite_vector(existing) and len(existing) == len(vector):
        count = min(len(samples), MAX_ROLE_SAMPLES)
        blended = [existing[i] + (vector[i] - existing[i]) / count
                   for i in range(len(vector))]
    else:
        blended = list(vector)
    if not af.is_finite_vector(blended):
        # Overflow/degenerate blend: never store a non-finite vector, because the
        # project saver turns it into null and the voiceprint dies silently.
        blended = list(existing) if af.is_finite_vector(existing) else list(vector)
    role["embedding"] = blended
    if role["type"] == "pending":
        role["type"] = "named"
    return note


# Per-feature-space prior matching, calibrated separately: a cluster matches a
# role when the cosine similarity clears the absolute threshold AND leads the
# runner-up role by the margin. Neural speaker spaces (pyannote wespeaker /
# CAM++) separate speakers well but same-speaker cosines drop in noisy audio, so
# their bar sits lower than the uncalibrated builtin envelope space — where all
# vectors are near-parallel and only a tiny lead means anything.
#
# The bars are deliberately strict: a cluster that does not clearly match a
# user-created role is left unmatched and its cues stay 待定 for review. Missing
# a real match costs one manual assignment; a wrong match silently poisons the
# dataset, so we prefer the miss.
# (threshold, margin); threshold None on the caller side selects the default.
EMBEDDER_MATCH = {
    "pyannote": (0.66, 0.07),
    "campp": (0.76, 0.06),
    "builtin": (0.84, 0.006),
}


def match_defaults(embedder: str) -> tuple[float, float]:
    """(threshold, margin) for a feature space; the builtin one is the fallback."""
    return EMBEDDER_MATCH.get(embedder, EMBEDDER_MATCH["builtin"])


def map_clusters(clusters: dict, roles: list[dict], threshold: float | None = None,
                 embedder: str = "builtin") -> tuple[dict, list[str]]:
    """Map detected clusters onto roles via greedy one-to-one assignment.

    A cluster matches a role when the cosine similarity clears ``threshold``
    (``None`` = the feature space's calibrated default), leads any runner-up role
    for that cluster by the feature-space margin, and neither side is already
    claimed — one role owns one cluster. Remaining clusters are reported
    anonymous; the caller leaves their cues 待定 rather than inventing a role
    (prefer a miss over a mis-assignment).
    Returns ({cluster: {role_id, score, matched}}, notes).
    """
    default_threshold, margin = match_defaults(embedder)
    auto = threshold is None
    if auto:
        threshold = default_threshold
    # One candidate per person. A project role and its global-library twin (or
    # several accumulated library copies) are the *same* speaker; if they are all
    # kept, the runner-up for a cluster is a near-identical copy, the margin is
    # always blown, and nothing ever matches. Callers pass project roles first,
    # so the first occurrence wins.
    matchable_roles: list[dict] = []
    seen_names: set[str] = set()
    for role in roles:
        if role["type"] == "ignored" or not role.get("embedding"):
            continue
        name_key = str(role.get("name") or "").strip()
        if name_key and name_key in seen_names:
            continue
        seen_names.add(name_key)
        matchable_roles.append(role)
    notes: list[str] = []
    if matchable_roles:
        notes.append(
            f"声纹先验匹配阈值 {threshold:.2f}（{embedder} 空间"
            f"{'自动校准' if auto else '手动'}）。"
        )
    mapping = {
        int(cluster_id): {"role_id": None, "score": 0.0, "matched": False}
        for cluster_id in clusters
    }

    scored_roles: dict[int, list[tuple[float, dict]]] = {}
    pairs: list[tuple[float, int, dict]] = []
    for cluster_id, cluster in clusters.items():
        embedding = cluster.get("embedding") or []
        # A NaN/Inf centroid means the embedding step failed for this cluster;
        # treat it as "no voiceprint" instead of matching on garbage.
        if not af.is_finite_vector(embedding):
            continue
        scores = []
        for role in matchable_roles:
            score = af.cosine(embedding, role["embedding"])
            if score > 0:
                scores.append((score, role))
        scores.sort(key=lambda item: -item[0])
        scored_roles[int(cluster_id)] = scores
        pairs.extend((score, int(cluster_id), role) for score, role in scores)

    pairs.sort(key=lambda item: -item[0])
    taken_roles: set[int] = set()
    for score, cluster_id, role in pairs:
        if mapping[cluster_id]["matched"] or role["id"] in taken_roles:
            continue
        runner_up = next(
            (s for s, other in scored_roles.get(cluster_id, [])
             if other["id"] != role["id"]),
            None,
        )
        leads = runner_up is None or (score - runner_up) >= margin
        if score >= threshold and leads:
            mapping[cluster_id] = {"role_id": role["id"], "score": round(score, 4),
                                   "matched": True}
            taken_roles.add(role["id"])

    matched_names = []
    for cluster_id in sorted(mapping):
        entry = mapping[cluster_id]
        if entry["matched"]:
            role = next(r for r in roles if r["id"] == entry["role_id"])
            matched_names.append(f"{role['name']}({entry['score']:.2f})")
    if matched_names:
        notes.append("先验声纹匹配成功：" + "、".join(matched_names) + "。")
    unmatched = sum(1 for v in mapping.values() if not v["matched"])
    if unmatched:
        notes.append(f"{unmatched} 个聚类未匹配到已创建的角色，相关字幕保持待定（可人工归属）。")
    if matchable_roles:
        # Show each cluster's best candidate score so a near miss is visible
        # (a low best score means the voices genuinely differ, not a bug).
        ranked = []
        for cluster_id in sorted(scored_roles):
            scores = scored_roles[cluster_id]
            if scores:
                score, role = scores[0]
                ranked.append(f"c{cluster_id}→{role['name']} {score:.2f}")
        if ranked:
            if len(ranked) > 12:
                ranked = ranked[:12] + ["…"]
            notes.append("聚类最佳匹配：" + "、".join(ranked) + "。")
    return mapping, notes


def merge_roles(target: dict, source: dict) -> dict:
    """Merge ``source`` into ``target`` (same person split into two clusters).

    Voiceprints only blend when they live in the same feature space and have the
    same dimension; otherwise mixing them would corrupt the target vector (and
    its ``embedder`` tag) into an unusable hybrid.
    """
    samples = target.setdefault("samples", [])
    target_count = max(1, len(samples))
    source_count = max(1, len(source.get("samples") or []))
    samples.extend(source.get("samples") or [])
    target_embedding = target.get("embedding") or []
    source_embedding = source.get("embedding") or []
    same_space = (target.get("embedder", "builtin") == source.get("embedder", "builtin")
                  and len(target_embedding) == len(source_embedding))
    if source_embedding and same_space:
        if target_embedding:
            # Sample-count weighted mean, not a sum: adding the vectors would
            # keep growing the magnitude on every merge for no matching benefit.
            total = target_count + source_count
            blended = [(a * target_count + b * source_count) / total
                       for a, b in zip(target_embedding, source_embedding)]
            target["embedding"] = (blended if af.is_finite_vector(blended)
                                   else list(target_embedding))
        else:
            target["embedding"] = list(source_embedding)
            target["embedder"] = source.get("embedder", target.get("embedder", "builtin"))
    elif source_embedding and not target_embedding:
        # Nothing to blend against: adopt the source voiceprint as-is.
        target["embedding"] = list(source_embedding)
        target["embedder"] = source.get("embedder", target.get("embedder", "builtin"))
    if target["type"] == "named" and target.get("embedding"):
        target["type"] = "registered"
    return target
