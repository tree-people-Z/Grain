"""Sortformer v2.1 runner (isolated env). Reads a WAV, prints one JSON line.

Called by the app via `external_engines.run`. Keep stdout clean: only the final
line is parsed, so anything else must go to stderr or be brief.

Model: nvidia/diar_streaming_sortformer_4spk-v2.1 (CC-BY-4.0, no HF gating).
"""

from __future__ import annotations

import argparse
import json
import sys


def _normalise(segments) -> list[dict]:
    """Normalise Sortformer rows to turns.

    NeMo's streaming ``diarize()`` returns rows as strings like
    ``"1.360 1.440 speaker_0"`` (offline may use lists/dicts), so accept all.
    """
    turns = []
    label_map: dict[str, int] = {}
    for row in segments or []:
        if isinstance(row, str):
            parts = row.split()
            if len(parts) < 3:
                continue
            start, end, speaker = parts[0], parts[1], " ".join(parts[2:])
        elif isinstance(row, dict):
            start, end = row.get("start"), row.get("end")
            speaker = row.get("speaker", row.get("label", 0))
        elif isinstance(row, (list, tuple)) and len(row) >= 3:
            start, end, speaker = row[0], row[1], row[2]
        else:
            continue
        try:
            start, end = float(start), float(end)
        except (TypeError, ValueError):
            continue
        cluster = label_map.setdefault(str(speaker), len(label_map))
        turns.append({"start": round(start, 3), "end": round(end, 3), "cluster": cluster})
    return turns


def main() -> int:
    parser = argparse.ArgumentParser(description="Sortformer v2.1 diarization runner")
    parser.add_argument("--wav", required=True)
    parser.add_argument("--min", type=int, default=1)
    parser.add_argument("--max", type=int, default=4)
    parser.add_argument("--model", default="nvidia/diar_streaming_sortformer_4spk-v2.1")
    args = parser.parse_args()

    notes: list[str] = []
    try:
        import torch
        from nemo.collections.asr.models import SortformerEncLabelModel
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"error": f"NeMo 未就绪：{exc}"}, ensure_ascii=False))
        return 2

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = SortformerEncLabelModel.from_pretrained(args.model)
        model.eval()
        try:
            model.to(torch.device(device))
        except Exception as exc:  # noqa: BLE001
            # Do not pretend we are on GPU when the move failed: the note below
            # reports the model's *actual* device, and the reason is surfaced.
            notes.append(f"移到 {device} 失败，已在 CPU 上运行：{str(exc)[:120]}")
        segments = model.diarize(audio=[args.wav], batch_size=1)
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"error": f"Sortformer 推理失败：{exc}"}, ensure_ascii=False))
        return 3

    rows = segments[0] if segments else []
    turns = _normalise(rows)
    # Report the device the model actually sits on (NeMo's from_pretrained may
    # already have placed it on cuda:0, independently of the `device` we picked).
    actual = str(getattr(model, "device", device))
    if actual.startswith("cuda"):
        try:
            actual = f"{actual} · {torch.cuda.get_device_name(0)}"
        except Exception:
            pass
    notes.append(f"Sortformer v2.1（{actual}），最多同时 4 人。")
    if args.max > 4:
        notes.append("Sortformer 上限 4 人；5 人以上请改用 pyannote community-1 或字幕级声纹聚类。")
    print(json.dumps({"turns": turns, "notes": notes}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
