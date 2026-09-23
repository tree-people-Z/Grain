"""DiariZen runner (isolated env). Reads a WAV, prints one JSON line.

Called by the app via `external_engines.run`. Keep stdout clean: only the final
line is parsed, so anything else must go to stderr.

Model: BUT-FIT/diarizen-wavlm-large-s80-md-v2 (WavLM-Large + Conformer + VBx).
Upstream: https://github.com/BUTSpeechFIT/DiariZen
License: code MIT, weights CC BY-NC 4.0 (non-commercial). See engines/README.md.
"""

from __future__ import annotations

import argparse
import json
import sys


def _normalise(annotation) -> list[dict]:
    """Convert a pyannote ``Annotation`` to the app's turn shape."""
    turns = []
    label_map: dict[str, int] = {}
    for segment, _, label in annotation.itertracks(yield_label=True):
        cluster = label_map.setdefault(str(label), len(label_map))
        turns.append({"start": round(float(segment.start), 3),
                      "end": round(float(segment.end), 3), "cluster": cluster})
    return turns


def main() -> int:
    parser = argparse.ArgumentParser(description="DiariZen diarization runner")
    parser.add_argument("--wav", required=True)
    parser.add_argument("--min", type=int, default=1)
    parser.add_argument("--max", type=int, default=6)
    parser.add_argument("--model", default="BUT-FIT/diarizen-wavlm-large-s80-md-v2")
    args = parser.parse_args()

    notes: list[str] = []
    try:
        import torch
        from diarizen.pipelines.inference import DiariZenPipeline
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"error": f"DiariZen 未就绪：{exc}"}, ensure_ascii=False))
        return 2

    try:
        pipeline = DiariZenPipeline.from_pretrained(args.model)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # Not every DiariZen build exposes .to(); a failure is not fatal.
        try:
            pipeline.to(torch.device(device))
        except Exception:
            pass
        annotation = pipeline(args.wav)
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"error": f"DiariZen 推理失败：{exc}"}, ensure_ascii=False))
        return 3

    turns = _normalise(annotation)
    notes.append(f"DiariZen（{args.model}，WavLM-Large + EEND + VBx）。")
    notes.append("权重为 CC BY-NC 4.0，仅限非商用；人数由模型自行估计。")
    print(json.dumps({"turns": turns, "notes": notes}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
