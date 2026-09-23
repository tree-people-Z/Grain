# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the Grain desktop app.

Builds a *base* desktop app: import / review / export plus the built-in and
manual engines. The detection stack (torch & friends, ~4 GB installed) is
deliberately excluded — the heavy engines run in their own environment via
``engines/`` (see README), and bundling torch would bloat the build without
bundling the model weights anyway.

    python -m PyInstaller --noconfirm --clean Grain.spec
"""

# Everything the detection engines need, but the base app does not. Listed so
# PyInstaller does not follow the (function-level) imports in diarize/transcribe
# and pull gigabytes of torch into the build.
excludes = [
    "torch", "torchaudio", "torchvision", "torchgen",
    "pyannote", "funasr", "modelscope", "datasets", "transformers",
    "huggingface_hub", "tokenizers", "safetensors", "onnxruntime",
    "numpy", "scipy", "sklearn", "hdbscan", "numba", "llvmlite",
    "librosa", "soundfile", "audioread", "faster_whisper",
    "pandas", "matplotlib", "PIL", "cv2", "tensorflow", "jax",
]

a = Analysis(
    ["desktop_pywebview.py"],
    pathex=[],
    binaries=[],
    datas=[("static", "static")],
    hiddenimports=[
        # Top-level app modules (server pulls most of them in anyway; explicit
        # so a refactor cannot silently drop one from the bundle).
        "server", "align", "apppaths", "audio_features", "dataset_export",
        "diarize", "external_engines", "media", "project", "roles",
        "subtitle_io", "transcribe",
        # pywebview picks its backend dynamically.
        "webview.platforms.winforms",
        "webview.platforms.edgechromium",
        "webview.platforms.mshtml",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Grain",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="Grain",
)
