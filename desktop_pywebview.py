"""Grain —— pywebview 桌面壳（纯 Python，无需 Rust）。

启动流程：选一个空闲端口 → 拉起 server.py 后端 → 轮询 /api/state 健康检查
→ 用系统 WebView 打开原生窗口 → 退出时回收后端子进程。

后端仍然是 server.py：它实现了视频播放所需的 HTTP Range 分片，
因此这里只负责开窗口，不复用 pywebview 自带的静态服务器。

运行：
    python desktop_pywebview.py            # 正常启动
    python desktop_pywebview.py --dev      # 开发模式：改 .py 自动重启后端，改 static/ 自动刷新窗口
依赖：  pip install pywebview
打包：  pyinstaller --noconsole --add-data "static;static" desktop_pywebview.py
        （同时把 server.py / *.py / static 一并收集，详见 README「桌面版打包」）
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.request

import apppaths

BASE_DIR = apppaths.writable_dir()
WINDOW_TITLE = "Grain"

# 开发模式：只监视后端会用到的顶层 .py 与 static/，跳过这些目录与启动器自身。
WATCH_SKIP_FILES = {"desktop_pywebview.py", "make_sample.py"}
WATCH_SKIP_DIRS = {"data", "__pycache__", "sample", ".git", "node_modules"}
POLL_SECONDS = 1.0

# 拖放聚合：WebView2 会把一次拖放的文件拆成多个事件，静默这么久就一起处理。
AGGREGATE_SECONDS = 0.6


def free_port() -> int:
    """让操作系统分配一个空闲端口。"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def spawn_server(port: int) -> subprocess.Popen:
    """用当前解释器启动后端，保证和本进程同一套 Python 环境。

    Frozen (PyInstaller) builds have no ``server.py`` on disk and ``sys.executable``
    is the GUI exe itself, so the backend is the same exe re-entered with
    ``--backend`` (see ``__main__``).
    """
    if getattr(sys, "frozen", False):
        cmd = [sys.executable, "--backend", "--port", str(port)]
    else:
        cmd = [sys.executable, "server.py", "--port", str(port)]
    return subprocess.Popen(cmd, cwd=BASE_DIR)


def healthy(port: int, timeout: float = 2.0) -> bool:
    """GET /api/state 返回 200 即视为后端就绪。"""
    url = f"http://127.0.0.1:{port}/api/state"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            resp.read()  # 读干响应体，避免后端记录无谓的连接中断
            return 200 <= resp.status < 300
    except Exception:
        return False


def wait_for_server(child: subprocess.Popen, port: int, deadline_seconds: int = 60) -> bool:
    deadline = time.time() + deadline_seconds
    while time.time() < deadline:
        if healthy(port):
            return True
        if child.poll() is not None:
            print(f"Python 后端提前退出（code={child.returncode}）。")
            return False
        time.sleep(0.4)
    print("后端启动超时。")
    return False


class Backend:
    """在固定端口上持有 server.py 子进程，支持原地重启（开发模式用）。"""

    def __init__(self, port: int) -> None:
        self.port = port
        self.proc: subprocess.Popen | None = spawn_server(port)

    def stop(self) -> None:
        proc = self.proc
        self.proc = None
        if proc is None:
            return
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    def restart(self) -> bool:
        self.stop()
        self.proc = spawn_server(self.port)
        return wait_for_server(self.proc, self.port, deadline_seconds=30)


class Api:
    """暴露给前端的原生能力：JS 侧通过 ``window.pywebview.api.*`` 调用。

    注意：pywebview 会递归遍历本对象的所有公开属性来生成 JS 桥。这里把窗口
    引用命名为 ``_window``（下划线开头会被跳过），否则它会一路爬进
    ``window.native.browser.webview`` 的 COM 对象并在后台疯狂报错。
    """

    def __init__(self) -> None:
        self._window = None
        self._maximized = False

    def pick_file(self, kind: str = "media"):
        """弹出系统文件选择框，返回绝对路径（取消则返回 None）。"""
        import webview

        file_types = {
            "media": ("媒体文件 (*.mp4;*.mkv;*.mov;*.webm;*.avi;*.m4v;"
                      "*.wav;*.mp3;*.m4a;*.flac;*.ogg;*.aac)",),
            "subtitle": ("字幕文件 (*.srt;*.vtt;*.ass;*.ssa)",),
        }.get(kind, ("所有文件 (*.*)",))
        result = self._window.create_file_dialog(
            webview.OPEN_DIALOG, allow_multiple=False, file_types=file_types
        )
        return result[0] if result else None

    # 无边框窗口（frameless）没有系统标题栏，最小化/最大化/关闭由前端按钮调用。

    def window_minimize(self) -> None:
        if self._window is not None:
            self._window.minimize()

    def window_maximize(self) -> None:
        if self._window is None:
            return
        self._maximized = not self._maximized
        if self._maximized:
            self._window.maximize()
        else:
            self._window.restore()

    def window_close(self) -> None:
        if self._window is not None:
            self._window.destroy()


# --- 拖放桥接：pywebview 拦下了原生文件拖放 --------------------------------

def bind_drop(window) -> None:
    """把系统拖入的文件路径转交给前端的 ``handleDroppedFiles``。

    WebView2 会拦截外部文件拖放，HTML5 的 ``drop`` 事件里拿不到真实路径；
    pywebview 在 Python 侧的 DOM 事件里提供了 ``pywebviewFullPath``，用它回填。
    """
    try:
        from webview.dom import DOMEventHandler
    except Exception as exc:
        print(f"[drop] 无法启用拖放桥接：{exc}")
        return

    def show(event):
        try:
            window.evaluate_js("window.showDropOverlay && showDropOverlay()")
        except Exception:
            pass

    def leave(event):
        try:
            window.evaluate_js("window.hideDropOverlay && hideDropOverlay()")
        except Exception:
            pass

    def drop(event):
        try:
            files = (event or {}).get("dataTransfer", {}).get("files", []) or []
        except Exception:
            files = []
        paths = [f.get("pywebviewFullPath") for f in files
                 if isinstance(f, dict) and f.get("pywebviewFullPath")]
        try:
            window.evaluate_js("window.hideDropOverlay && hideDropOverlay()")
        except Exception:
            pass
        if not paths:
            return
        # pywebview/WebView2 会把同一次拖放的多个文件拆成多个 drop 事件送达，
        # 这里在很短的静默窗口内聚合，凑齐「视频 + 字幕」再一次性交给前端。
        buffer.extend(paths)
        if state["timer"] is not None:
            state["timer"].cancel()
        timer = threading.Timer(AGGREGATE_SECONDS, flush)
        timer.daemon = True
        timer.start()
        state["timer"] = timer

    buffer: list[str] = []
    state = {"timer": None}

    def flush():
        state["timer"] = None
        paths = list(dict.fromkeys(buffer))
        buffer.clear()
        if not paths:
            return
        print(f"[drop] 拖入 {len(paths)} 个文件")
        payload = json.dumps(paths, ensure_ascii=False)
        try:
            window.evaluate_js(f"handleDroppedFiles({payload})")
        except Exception as exc:
            print(f"[drop] 导入失败：{exc}")

    def register():
        document = window.dom.document
        document.events.dragenter += DOMEventHandler(show, True, True)
        document.events.dragover += DOMEventHandler(show, True, True, debounce=200)
        document.events.dragleave += DOMEventHandler(leave, True, True)
        document.events.drop += DOMEventHandler(drop, True, True)

    try:
        register()
    except Exception:
        # DOM 尚未就绪时，等页面加载完成再挂。
        try:
            window.events.loaded += lambda *a, **k: register()
        except Exception as exc:
            print(f"[drop] 无法启用拖放桥接：{exc}")


# --- 开发模式：文件监视 + 自动重载 -----------------------------------------

def iter_watched_files(base_dir: str):
    """顶层后端 .py + static/ 下的全部资源（跳过缓存/构建目录）。"""
    for name in os.listdir(base_dir):
        path = os.path.join(base_dir, name)
        if os.path.isfile(path) and name.endswith(".py") and name not in WATCH_SKIP_FILES:
            yield path
    static_dir = os.path.join(base_dir, "static")
    for root, dirs, files in os.walk(static_dir):
        dirs[:] = [d for d in dirs if d not in WATCH_SKIP_DIRS]
        for name in files:
            yield os.path.join(root, name)


def snapshot(base_dir: str) -> dict:
    state = {}
    for path in iter_watched_files(base_dir):
        try:
            state[path] = os.path.getmtime(path)
        except OSError:
            pass
    return state


def watch(base_dir: str, window, backend: Backend) -> None:
    """后台线程：代码变了就重启后端 / 刷新窗口。作为 webview.start 的 func 传入。"""
    state = snapshot(base_dir)
    while True:
        time.sleep(POLL_SECONDS)
        time.sleep(0.2)  # 给编辑器把文件写完的时间，避免读到半截
        current = snapshot(base_dir)
        if current == state:
            continue
        changed = {p for p in set(state) | set(current) if state.get(p) != current.get(p)}
        state = current

        py_changed = [p for p in changed if p.endswith(".py")]
        if py_changed:
            names = "、".join(os.path.basename(p) for p in sorted(py_changed))
            print(f"[dev] 后端改动（{names}），重启后端…")
            if not backend.restart():
                print("[dev] 后端重启失败，请检查上面的报错。")
        else:
            names = "、".join(os.path.basename(p) for p in sorted(changed))
            print(f"[dev] 前端改动（{names}），刷新窗口。")

        try:
            window.evaluate_js("location.reload()")
        except Exception as exc:  # 窗口已关闭等
            print(f"[dev] 刷新窗口失败：{exc}")
            return


def main() -> int:
    parser = argparse.ArgumentParser(description="Grain 桌面版")
    parser.add_argument("--dev", action="store_true",
                        help="开发模式：改 .py 自动重启后端、改 static/ 自动刷新窗口")
    parser.add_argument("--port", type=int, default=0,
                        help="后端端口（默认自动选择空闲端口）")
    args = parser.parse_args()

    try:
        import webview
    except ImportError:
        print("未安装 pywebview。请运行： pip install pywebview")
        print("也可以改用 启动网页版.bat（浏览器版，无需此依赖）。")
        return 1

    try:  # 中文 Windows 控制台默认 GBK，日志里的 ✓/✗ 等会编码失败
        sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    port = args.port or free_port()
    backend = Backend(port)
    if not wait_for_server(backend.proc, port):
        backend.stop()
        return 1

    api = Api()
    window = webview.create_window(
        WINDOW_TITLE,
        f"http://127.0.0.1:{port}/",
        width=1500,
        height=940,
        min_size=(1180, 700),
        js_api=api,
        frameless=True,          # 去掉系统白色标题栏，改用底部工具栏里的自绘按钮
        easy_drag=False,         # 只允许 .pywebview-drag-region 拖动，避免和轨道拖拽打架
        background_color="#151517",
    )
    api._window = window
    window.events.maximized += lambda *_: setattr(api, "_maximized", True)
    window.events.restored += lambda *_: setattr(api, "_maximized", False)

    def on_start(win):
        bind_drop(win)  # 桌面版拖放桥接（必须在 GUI 起来后绑定）
        if args.dev:
            print(f"[dev] 开发模式已开启（端口 {port}）："
                  "改 .py 自动重启后端，改 static/ 自动刷新窗口。")
            watch(BASE_DIR, win, backend)

    try:
        webview.start(on_start, window)
    finally:
        backend.stop()
    return 0


def run_backend(argv: list[str]) -> int:
    """Backend entry for a frozen build: same exe, no window, just the HTTP server."""
    import server

    def value(flag: str, default: str) -> str:
        return argv[argv.index(flag) + 1] if flag in argv else default

    server.serve(value("--host", "127.0.0.1"), int(value("--port", "8770")))
    return 0


if __name__ == "__main__":
    if "--backend" in sys.argv:
        raise SystemExit(run_backend(sys.argv))
    raise SystemExit(main())
