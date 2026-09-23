"""Create / refresh the isolated environment for an external engine.

Usage:
    python scripts/setup_engine.py sortformer
    python scripts/setup_engine.py diarizen

Uses `uv` (fast) when available, else falls back to `python -m venv` + pip.

A recipe may describe a plain PyPI install (``packages``) or a source checkout
(``git`` + ``requirements`` + ``editable``). DiariZen, for example, is not on
PyPI and needs its own repo plus the pyannote fork it vendors as a submodule.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENGINES_DIR = os.path.join(BASE_DIR, "engines")

RECIPES: dict[str, dict] = {
    "sortformer": {
        "python": "3.12",
        "packages": [
            "Cython",
            "packaging",
            "nemo_toolkit[asr]",
        ],
        "torch_index": "https://download.pytorch.org/whl/cu128",
    },
    # Experimental: DiariZen lives on GitHub (with submodules), not PyPI. Its
    # requirements pull a matching torch; the weights are CC BY-NC 4.0.
    "diarizen": {
        "python": "3.10",
        "torch_index": "https://download.pytorch.org/whl/cu121",
        "git": {
            "url": "https://github.com/BUTSpeechFIT/DiariZen",
            "dir": "_src",
            "submodules": True,
        },
        "requirements": "requirements.txt",
        "editable": True,
    },
}


def which(name: str) -> str | None:
    return shutil.which(name)


def venv_python(engine_dir: str) -> str:
    win = os.path.join(engine_dir, ".venv", "Scripts", "python.exe")
    posix = os.path.join(engine_dir, ".venv", "bin", "python")
    return win if os.name == "nt" else posix


def run(cmd: list[str], **kw) -> None:
    print("  $", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, **kw)


def checkout_source(recipe: dict, engine_dir: str) -> str | None:
    """Clone/update a git source tree; returns its path (or None)."""
    spec = recipe.get("git")
    if not spec:
        return None
    target = os.path.join(engine_dir, spec.get("dir", "_src"))
    if os.path.isdir(os.path.join(target, ".git")):
        run(["git", "-C", target, "pull", "--ff-only"])
    else:
        run(["git", "clone", "--recursive", spec["url"], target])
    if spec.get("submodules"):
        run(["git", "-C", target, "submodule", "update", "--init", "--recursive"])
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description="安装外挂检测引擎的隔离环境")
    parser.add_argument("engine", choices=sorted(RECIPES))
    args = parser.parse_args()

    recipe = RECIPES[args.engine]
    engine_dir = os.path.join(ENGINES_DIR, args.engine)
    if not os.path.isfile(os.path.join(engine_dir, "run.py")):
        print(f"找不到 {engine_dir}\\run.py", file=sys.stderr)
        return 1
    if recipe.get("git") and not which("git"):
        print("该引擎需要 git（用于拉取源码）。请先安装 git。", file=sys.stderr)
        return 1
    venv_dir = os.path.join(engine_dir, ".venv")

    source_dir = checkout_source(recipe, engine_dir)

    if which("uv"):
        run(["uv", "venv", "--python", recipe["python"], venv_dir])
        python = venv_python(engine_dir)
        base = ["uv", "pip", "install", "--python", python]
    else:
        print("未找到 uv，改用 python -m venv（较慢）。建议 pip install uv。")
        run([sys.executable, "-m", "venv", venv_dir])
        python = venv_python(engine_dir)
        base = [python, "-m", "pip", "install"]

    # torch first, from the CUDA wheel index, so the engine does not pull a CPU
    # build (and, for a source checkout, so its requirements see torch present).
    if recipe.get("torch_index"):
        run(base + ["--index-url", recipe["torch_index"], "torch"])
    if recipe.get("packages"):
        run(base + recipe["packages"])
    if recipe.get("requirements") and source_dir:
        run(base + ["-r", os.path.join(source_dir, recipe["requirements"])])
    if recipe.get("editable") and source_dir:
        run(base + ["-e", source_dir])

    print(f"\n完成。{args.engine} 现在应显示为可用。")
    print("首次运行检测会下载模型权重，请耐心等待。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
