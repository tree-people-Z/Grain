"""Bridge to heavy diarization engines that run in their own Python environment.

Each engine lives in ``engines/<key>/`` with a ``run.py`` and an isolated
interpreter under ``.venv/``. The app calls it as a subprocess and reads one
JSON object from stdout. This keeps a NeMo or DiariZen stack (which pins its own
torch/Python) from ever colliding with the app's environment — and it is what we
ship when packaging.

Protocol (the runner must honour it)::

    <venv-python> run.py --wav <16k mono wav> --min <n> --max <n>
    stdout: one JSON object on the last non-empty line:
        {"turns": [{"start": float, "end": float, "cluster": int}, ...],
         "notes": [str, ...]}
"""

from __future__ import annotations

import json
import os
import subprocess

import apppaths

BASE_DIR = apppaths.writable_dir()
ENGINES_DIR = os.path.join(BASE_DIR, "engines")

# key -> metadata. "dir" defaults to the key.
ENGINE_SPECS: dict[str, dict] = {
    "sortformer": {
        "label": "NVIDIA Sortformer v2.1（NeMo，≤4 人）",
        "install": "python scripts/setup_engine.py sortformer",
        "detail": ("端到端 Transformer diarization，中/日语表现最强、无需 HF 门控。"
                   "最多同时 4 人；5 人以上请用 pyannote community-1 或字幕级声纹聚类。"),
    },
}

# Candidate interpreter locations inside an engine dir (Windows then POSIX).
_PY_CANDIDATES = (
    os.path.join(".venv", "Scripts", "python.exe"),
    os.path.join(".venv", "bin", "python"),
)


def engine_dir(key: str) -> str:
    return os.path.join(ENGINES_DIR, ENGINE_SPECS.get(key, {}).get("dir", key))


def python_path(key: str) -> str | None:
    directory = engine_dir(key)
    for relative in _PY_CANDIDATES:
        candidate = os.path.join(directory, relative)
        if os.path.isfile(candidate):
            return candidate
    return None


def available(key: str) -> bool:
    return (python_path(key) is not None
            and os.path.isfile(os.path.join(engine_dir(key), "run.py")))


def spec(key: str) -> dict:
    return ENGINE_SPECS.get(key, {})


def _child_env() -> dict:
    """Give the runner the HF token/endpoint the app already has."""
    env = dict(os.environ)
    if not env.get("HF_TOKEN"):
        # Same resolution as the rest of the app (honours SSP_DATA_DIR), so a
        # test run never reads the real user's token file.
        token_file = os.path.join(apppaths.data_dir(), "hf_token.txt")
        try:
            with open(token_file, "r", encoding="utf-8-sig") as handle:
                token = handle.read().strip()
            if token:
                env["HF_TOKEN"] = token
        except OSError:
            pass
    return env


def run(key: str, wav_path: str, min_speakers: int, max_speakers: int,
        timeout: int = 3600) -> dict:
    """Invoke the engine and return its parsed payload (raises on failure)."""
    interpreter = python_path(key)
    runner = os.path.join(engine_dir(key), "run.py")
    if not interpreter or not os.path.isfile(runner):
        raise RuntimeError(
            f"引擎「{key}」未安装。安装：{spec(key).get('install', '(见 engines/README.md)')}"
        )
    command = [interpreter, runner, "--wav", wav_path,
               "--min", str(max(1, int(min_speakers))),
               "--max", str(max(1, int(max_speakers)))]
    try:
        proc = subprocess.run(command, capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=timeout,
                              env=_child_env())
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"引擎「{key}」运行超时（>{timeout}s）。") from exc

    stdout = proc.stdout or ""
    payload = None
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            break
    if proc.returncode != 0:
        detail = (payload or {}).get("error") if isinstance(payload, dict) else None
        raise RuntimeError(
            f"引擎「{key}」运行失败：{detail or (proc.stderr or stdout or '')[-400:]}"
        )
    if payload is None:
        raise RuntimeError(f"引擎「{key}」未返回结果：{(proc.stderr or stdout)[-400:]}")
    if payload.get("error"):
        raise RuntimeError(f"引擎「{key}」错误：{payload['error']}")
    return payload
