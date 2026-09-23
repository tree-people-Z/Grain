"""Download the bundled models + ffmpeg for the Grain 整合包.

The heavy assets are not in git (GitHub rejects files >100 MB). They live as
Release assets and are fetched on first launch:

    python scripts/fetch_assets.py            # fetch what is missing
    python scripts/fetch_assets.py --force    # re-download everything

The .bat launchers call this automatically, so a fresh clone becomes usable on
first open (needs network that one time). If a download fails the script exits
non-zero but the app still starts with the built-in / manual engines.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import urllib.request
import zipfile

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RELEASE = "https://github.com/tree-people-Z/Grain/releases/download/assets-v1"

# (marker directory that must exist once unpacked, archive name)
ASSETS = [
    (os.path.join("models", "hf", "hub"), "grain-models.zip"),
    (os.path.join("runtime", "ffmpeg", "bin"), "grain-ffmpeg.zip"),
]


def _present(marker: str) -> bool:
    path = os.path.join(BASE, marker)
    return os.path.isdir(path) and any(os.scandir(path))


def _download(url: str, destination: str) -> None:
    with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310
        total = int(response.headers.get("Content-Length") or 0)
        done = 0
        with open(destination, "wb") as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
                done += len(chunk)
                if total:
                    pct = done * 100 // total
                    sys.stdout.write(f"\r  {pct:3d}%  {done // (1024 * 1024)}/"
                                     f"{total // (1024 * 1024)} MB")
                    sys.stdout.flush()
    sys.stdout.write("\n")


def _extract(archive: str) -> None:
    root = os.path.abspath(BASE)
    with zipfile.ZipFile(archive) as zf:
        for member in zf.namelist():
            target = os.path.abspath(os.path.join(root, member))
            if target != root and not target.startswith(root + os.sep):
                raise RuntimeError(f"压缩包内路径越界，已中止：{member}")
        zf.extractall(root)


def main() -> int:
    parser = argparse.ArgumentParser(description="下载整合包资源（模型 + ffmpeg）")
    parser.add_argument("--force", action="store_true", help="已存在也重新下载")
    args = parser.parse_args()

    missing = [(marker, name) for marker, name in ASSETS
               if args.force or not _present(marker)]
    if not missing:
        print("[assets] 已内置，跳过下载。")
        return 0

    print("[assets] 首次运行：下载内置模型与 ffmpeg（约 1 GB，仅这一次）…")
    failures = 0
    for marker, name in missing:
        url = f"{RELEASE}/{name}"
        print(f"[assets] {name}  <-  {url}")
        try:
            with tempfile.TemporaryDirectory(prefix="grain_assets_") as tmp:
                archive = os.path.join(tmp, name)
                _download(url, archive)
                print(f"[assets] 解压 {name} …")
                _extract(archive)
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"[assets] 失败：{name} — {type(exc).__name__}: {exc}")
            print("[assets] 可稍后重试，或手动把压缩包解压到项目根目录。")
    if failures:
        print("[assets] 部分资源未就绪：自动检测引擎可能显示“不可用”，基础功能不受影响。")
        return 1
    print("[assets] 完成。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
