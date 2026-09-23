"""Create / refresh the isolated environment for an external engine.

Usage:
    python scripts/setup_engine.py sortformer

Uses `uv` (fast) when available, else falls back to `python -m venv` + pip.
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


def main() -> int:
    parser = argparse.ArgumentParser(description="安装外挂检测引擎的隔离环境")
    parser.add_argument("engine", choices=sorted(RECIPES))
    args = parser.parse_args()

    recipe = RECIPES[args.engine]
    engine_dir = os.path.join(ENGINES_DIR, args.engine)
    if not os.path.isfile(os.path.join(engine_dir, "run.py")):
        print(f"找不到 {engine_dir}\\run.py", file=sys.stderr)
        return 1
    venv_dir = os.path.join(engine_dir, ".venv")

    if which("uv"):
        run(["uv", "venv", "--python", recipe["python"], venv_dir])
        python = venv_python(engine_dir)
        base = ["uv", "pip", "install", "--python", python]
    else:
        print("未找到 uv，改用 python -m venv（较慢）。建议 pip install uv。")
        run([sys.executable, "-m", "venv", venv_dir])
        python = venv_python(engine_dir)
        base = [python, "-m", "pip", "install"]

    # torch first, from the CUDA wheel index, so NeMo does not pull a CPU build.
    run(base + ["--index-url", recipe["torch_index"], "torch"])
    run(base + recipe["packages"])

    print(f"\n完成。{args.engine} 现在应显示为可用。")
    print("首次运行检测会下载模型权重，请耐心等待。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
