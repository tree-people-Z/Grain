"""HTTP server + REST API for the subtitle speaker-attribution tool.

Run:  python server.py [--port 8770] [--host 127.0.0.1]
Stdlib only; the frontend is plain HTML/CSS/JS served from ./static.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import re
import sys
import threading
import traceback
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import apppaths
import audio_features as af
import dataset_export
import diarize
import jsonutil
import media
import project as store
import roles as rolelib
import subtitle_io as sio
import transcribe

BASE_DIR = apppaths.bundle_dir()
STATIC_DIR = os.path.join(BASE_DIR, "static")

MEDIA_MIME = {
    ".mp4": "video/mp4", ".mkv": "video/x-matroska", ".mov": "video/quicktime",
    ".webm": "video/webm", ".avi": "video/x-msvideo", ".m4v": "video/mp4",
    ".wav": "audio/wav", ".mp3": "audio/mpeg", ".m4a": "audio/mp4",
    ".flac": "audio/flac", ".ogg": "audio/ogg", ".aac": "audio/aac",
}


MAX_UPLOAD_BYTES = int(os.environ.get("SSP_MAX_UPLOAD_MB", "4096")) * 1024 * 1024


def _within(path: str, root: str) -> bool:
    """True when ``path`` is inside ``root`` (not merely sharing a name prefix)."""
    try:
        return os.path.commonpath(
            [os.path.abspath(path), os.path.abspath(root)]
        ) == os.path.abspath(root)
    except ValueError:
        return False


# --- client-facing views ----------------------------------------------------

def role_view(role: dict) -> dict:
    return rolelib.role_summary(role)


def project_view(project: dict, include_segments: bool = True) -> dict:
    view = {
        "id": project["id"],
        "name": project["name"],
        "media_path": project["media_path"],
        "media_kind": project.get("media_kind"),
        "subtitle_path": project["subtitle_path"],
        "subtitle_format": project.get("subtitle_format"),
        "duration": project.get("duration"),
        "engine": project.get("engine"),
        "detection_notes": project.get("detection_notes", []),
        "import_notes": project.get("import_notes", []),
        "cluster_notes": (project.get("diarization") or {}).get("notes", []),
        "bilingual": project.get("bilingual", False),
        "line_mode": project.get("line_mode"),
        "line_info": project.get("line_info", {}),
        "second_subtitle_path": project.get("second_subtitle_path"),
        "stats": store.stats(project),
        "roles": [role_view(role) for role in project["roles"]],
        "exports": project.get("exports", []),
        "created": project.get("created"),
        "updated": project.get("updated"),
        "media_url": f"/api/projects/{project['id']}/media",
        "peaks_url": f"/api/projects/{project['id']}/peaks",
    }
    if include_segments:
        names = {role["id"]: role["name"] for role in project["roles"]}
        colors = {role["id"]: role["color"] for role in project["roles"]}
        view["segments"] = [
            {
                "id": segment["id"],
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"],
                "translation": segment.get("translation", ""),
                "speaker_id": segment.get("speaker_id"),
                "speaker_name": names.get(segment.get("speaker_id")),
                "color": colors.get(segment.get("speaker_id")),
                "confidence": segment.get("confidence"),
                "status": segment.get("status", "pending"),
                "note": segment.get("note"),
            }
            for segment in project["segments"]
        ]
    turns = (project.get("diarization") or {}).get("turns") or []
    if turns:
        # The client draws its own timeline; turns are diagnostic. Cap the payload.
        view["turns"] = turns[:5000]
        view["cluster_to_role"] = {
            str(k): v for k, v in (project.get("cluster_to_role") or {}).items()
        }
    return view


def state_payload() -> dict:
    return {
        "engines": diarize.engine_availability(),
        "formats": [{"id": key, "label": label}
                    for key, label in dataset_export.FORMAT_LABELS.items()],
        "projects": store.list_projects(),
        "library": [
            {
                "library_id": entry["library_id"],
                "name": entry["name"],
                "color": entry["color"],
                "embedder": entry.get("embedder", "builtin"),
                "has_voiceprint": bool(entry.get("embedding")),
                "sample_count": len(entry.get("samples") or []),
            }
            for entry in store.load_library()
        ],
        "asr": transcribe.availability(),
        "device": diarize.device_info(),
        "settings": store.public_settings(),
        "default_threshold": store.load_settings()["threshold"],
        "ffmpeg": bool(media.FFMPEG),
    }


# --- waveform peaks (for the timeline track) --------------------------------

def peaks_for(project: dict, buckets: int = 1200) -> list[float]:
    cache = os.path.join(store.WORK_DIR, f"{project['id']}.peaks.json")
    try:
        payload = store._read_json(cache, {})
        if payload.get("buckets") == buckets:
            return payload["peaks"]
    except Exception:
        pass
    wav_path = store.work_wav(project)
    signal, _rate = af.read_wav(wav_path, media.AUDIO_SAMPLE_RATE)
    peaks = af.waveform_peaks(signal, buckets)
    try:
        # Atomic: a concurrent GET must never read a half-written peaks file.
        store._write_json(cache, {"buckets": buckets, "peaks": peaks})
    except OSError:
        pass
    return peaks


# --- request handler --------------------------------------------------------

class QuietHTTPServer(ThreadingHTTPServer):
    """ThreadingHTTPServer that ignores client-side connection aborts.

    The webview aborts in-flight requests all the time (page reloads, video
    seeking, range requests), which otherwise prints a full traceback per
    connection reset. Those are not server errors.
    """

    daemon_threads = True

    def handle_error(self, request, client_address):
        if isinstance(sys.exc_info()[1], (ConnectionError,)):
            return
        super().handle_error(request, client_address)


class Handler(BaseHTTPRequestHandler):
    server_version = "SpeakerAttribution/1.0"
    protocol_version = "HTTP/1.1"
    _mutation_lock = threading.RLock()

    # -- helpers ------------------------------------------------------------
    def log_message(self, fmt, *args):
        pass  # keep the console clean; errors are reported explicitly

    def _send(self, status: int, body: bytes, content_type: str,
              extra: dict | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, payload, status: int = 200) -> None:
        body = json.dumps(jsonutil.json_safe(payload), ensure_ascii=False).encode("utf-8")
        self._send(status, body, "application/json; charset=utf-8")

    def _error(self, message: str, status: int = 400) -> None:
        self._json({"ok": False, "error": message}, status)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        raw = self.rfile.read(length)
        try:
            return json.loads(raw.decode("utf-8-sig"))
        except Exception as exc:
            raise ValueError(f"请求体不是合法 JSON：{exc}") from exc

    def _static(self, relative: str) -> None:
        relative = relative.lstrip("/") or "index.html"
        path = os.path.normpath(os.path.join(STATIC_DIR, relative))
        if not _within(path, STATIC_DIR) or not os.path.isfile(path):
            self._error("not found", 404)
            return
        mime = mimetypes.guess_type(path)[0] or "application/octet-stream"
        if path.endswith((".html", ".css", ".js")):
            mime += "; charset=utf-8"
        with open(path, "rb") as handle:
            self._send(200, handle.read(), mime)

    # -- routing ------------------------------------------------------------
    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = urllib.parse.unquote(parsed.path)
        query = urllib.parse.parse_qs(parsed.query)
        try:
            if path in ("/", "/index.html"):
                return self._static("index.html")
            if path == "/api/state":
                return self._json({"ok": True, **state_payload()})
            if path == "/api/projects":
                return self._json({"ok": True, "projects": store.list_projects()})
            if path == "/api/library":
                return self._json({"ok": True,
                                   "library": state_payload()["library"]})

            if path == "/api/settings":
                return self._json({"ok": True, "settings": store.public_settings()})

            match = re.fullmatch(r"/api/projects/([\w.-]+)", path)
            if match:
                project = store.load(match.group(1))
                return self._json({"ok": True, "project": project_view(project)})

            match = re.fullmatch(r"/api/projects/([\w.-]+)/media", path)
            if match:
                return self._stream_media(store.load(match.group(1)))

            match = re.fullmatch(r"/api/projects/([\w.-]+)/peaks", path)
            if match:
                try:
                    buckets = int((query.get("buckets") or ["1200"])[0])
                except ValueError:
                    buckets = 1200
                buckets = max(50, min(buckets, 4000))
                peaks = peaks_for(store.load(match.group(1)), buckets)
                return self._json({"ok": True, "peaks": peaks})

            match = re.fullmatch(r"/api/projects/([\w.-]+)/export/([^/]+)", path)
            if match:
                return self._download(match.group(1), match.group(2))

            return self._static(path)
        except ConnectionError:
            return  # 客户端（webview）主动断开：刷新/拖动/取消，非服务端错误
        except FileNotFoundError:
            return self._error("项目或文件不存在", 404)
        except Exception as exc:
            traceback.print_exc()
            return self._error(f"{type(exc).__name__}: {exc}", 500)

    do_HEAD = do_GET

    def _receive_upload(self, name: str, length: int) -> str:
        """Stream the request body to an upload file in 1 MB chunks.

        The old code did ``rfile.read(length)`` into memory first, so dropping a
        multi-GB video on the window could OOM the server. Chunking keeps peak
        memory at the buffer size regardless of file size.
        """
        target = store._upload_target(name)
        remaining = length
        try:
            with open(target, "wb") as handle:
                while remaining > 0:
                    chunk = self.rfile.read(min(1024 * 1024, remaining))
                    if not chunk:
                        raise ConnectionError("upload interrupted")
                    handle.write(chunk)
                    remaining -= len(chunk)
        except BaseException:
            try:
                os.remove(target)
            except OSError:
                pass
            raise
        return target

    def do_PUT(self):
        """Raw file upload: PUT /api/upload?name=<urlencoded filename>."""
        parsed = urllib.parse.urlparse(self.path)
        if urllib.parse.unquote(parsed.path) != "/api/upload":
            return self._error("not found", 404)
        try:
            query = urllib.parse.parse_qs(parsed.query)
            name = query.get("name", ["file"])[0]
            length = int(self.headers.get("Content-Length") or 0)
            if not length:
                return self._error("空的上传内容", 400)
            if length > MAX_UPLOAD_BYTES:
                return self._error(
                    f"文件超过上限 {MAX_UPLOAD_BYTES // (1024 * 1024)} MB", 413)
            path = self._receive_upload(name, length)
            return self._json({"ok": True, "path": path,
                               "kind": store.classify_file(path)})
        except ConnectionError:
            return
        except Exception as exc:
            traceback.print_exc()
            return self._error(f"{type(exc).__name__}: {exc}", 500)

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = urllib.parse.unquote(parsed.path)
        # Serialize all state mutations: handlers do load -> mutate -> save, and
        # ThreadingHTTPServer would otherwise let two edits clobber each other.
        with self._mutation_lock:
            try:
                payload = self._read_json()
                return self._route_post(path, payload)
            except ConnectionError:
                return
            except FileNotFoundError as exc:
                return self._error(str(exc), 404)
            except (KeyError, ValueError) as exc:
                return self._error(str(exc), 400)
            except Exception as exc:
                traceback.print_exc()
                return self._error(f"{type(exc).__name__}: {exc}", 500)

    def do_PATCH(self):
        self.do_POST()

    def do_DELETE(self):
        parsed = urllib.parse.urlparse(self.path)
        path = urllib.parse.unquote(parsed.path)
        with self._mutation_lock:
            return self._do_delete(path)

    def _do_delete(self, path: str):
        try:
            match = re.fullmatch(r"/api/projects/([\w.-]+)", path)
            if match:
                store.delete_project(match.group(1))
                return self._json({"ok": True})
            match = re.fullmatch(r"/api/projects/([\w.-]+)/roles/(\d+)", path)
            if match:
                project = store.load(match.group(1))
                store.delete_role(project, int(match.group(2)))
                store.save(project)
                return self._json({"ok": True, "project": project_view(project)})
            match = re.fullmatch(r"/api/library/([\w.-]+)", path)
            if match:
                library = [r for r in store.load_library()
                           if r["library_id"] != match.group(1)]
                store.save_library(library)
                return self._json({"ok": True})
            return self._error("not found", 404)
        except ConnectionError:
            return
        except FileNotFoundError:
            return self._error("项目不存在", 404)
        except Exception as exc:
            traceback.print_exc()
            return self._error(f"{type(exc).__name__}: {exc}", 500)

    def _route_post(self, path: str, payload: dict):
        if path == "/api/projects":
            project = store.create(
                payload.get("media_path", ""),
                payload.get("subtitle_path", ""),
                payload.get("name"),
                second_subtitle_path=payload.get("second_subtitle_path") or None,
                line_mode=payload.get("line_mode") or "auto",
            )
            return self._json({"ok": True, "project": project_view(project)})

        if path == "/api/library":
            entry, action = store.add_library_role(
                payload.get("name") or "未命名",
                payload.get("color"),
                payload.get("embedding"),
            )
            return self._json({"ok": True, "action": action, "library_role": entry,
                               "library": state_payload()["library"]})

        if path == "/api/library_dedupe":
            result = store.dedupe_library()
            return self._json({"ok": True, "removed": result["removed"],
                               "kept": result["kept"],
                               "library": state_payload()["library"]})

        if path == "/api/transcribe":
            media_path = store.normalize_path(payload.get("media_path", ""))
            if not os.path.isfile(media_path):
                return self._error(f"找不到媒体文件：{media_path}", 404)
            cfg = store.load_settings()
            segments, language = transcribe.transcribe(
                media_path,
                model_size=payload.get("model_size") or cfg["asr_model"],
                language=payload.get("language") or payload.get("asr_language")
                or cfg["asr_language"] or None,
            )
            stem = payload.get("name") or os.path.splitext(os.path.basename(media_path))[0]
            target_dir = os.path.join(store.DATA_DIR, "asr")
            os.makedirs(target_dir, exist_ok=True)
            srt_path = os.path.join(target_dir, f"{store.slug(stem)}.srt")
            with open(srt_path, "w", encoding="utf-8") as handle:
                handle.write(sio.write_srt(segments, {}))
            return self._json({
                "ok": True,
                "subtitle_path": srt_path,
                "count": len(segments),
                "language": language,
            })

        if path == "/api/settings":
            return self._json({"ok": True, "settings": store.save_settings(payload)})

        if path == "/api/maintenance":
            if payload.get("action") == "clean_work":
                removed = 0
                for name in os.listdir(store.WORK_DIR):
                    try:
                        os.remove(os.path.join(store.WORK_DIR, name))
                        removed += 1
                    except OSError:
                        pass
                return self._json({"ok": True, "removed": removed})
            return self._error("未知维护操作", 400)

        match = re.fullmatch(r"/api/projects/([\w.-]+)/(\w[\w-]*)", path)
        if not match:
            return self._error("not found", 404)
        project_id, action = match.group(1), match.group(2)
        project = store.load(project_id)

        if action == "detect":
            cfg = store.load_settings()
            threshold = payload.get("threshold")
            project = store.run_detection(
                project,
                engine=payload.get("engine") or cfg["default_engine"],
                min_speakers=int(payload.get("min_speakers") or cfg["min_speakers"]),
                max_speakers=int(payload.get("max_speakers") or cfg["max_speakers"]),
                threshold=cfg["threshold"] if threshold is None else float(threshold),
                overwrite_manual=bool(payload.get("overwrite_manual", False)),
                sweep=bool(payload.get("sweep", False)),
                consensus=bool(payload.get("consensus", False)),
            )
            return self._json({"ok": True, "project": project_view(project)})

        if action == "translation":
            store.set_translation(project, store.normalize_path(payload.get("path", "")))
            store.save(project)
            return self._json({"ok": True, "project": project_view(project)})

        if action == "line_mode":
            mode = payload.get("mode") or "auto"
            if mode == "auto":
                mode = store.sio.detect_line_mode(project["segments"])["suggested"]
            store.set_line_mode(project, mode)
            store.save(project)
            return self._json({"ok": True, "project": project_view(project)})

        if action == "reset_auto":
            cleared = store.reset_auto(project)
            store.save(project)
            return self._json({"ok": True, "cleared": cleared,
                               "project": project_view(project)})

        if action == "bulk":
            changed = store.bulk_set(project, payload.get("ids") or [],
                                     payload.get("speaker_id"),
                                     status=payload.get("status", "manual"),
                                     confidence=payload.get("confidence"))
            store.save(project)
            return self._json({"ok": True, "changed": changed,
                               "project": project_view(project)})

        if action == "roles":
            role = store.add_role(project, payload.get("name") or "新角色",
                                  payload.get("color"))
            store.save(project)
            return self._json({"ok": True, "role": role_view(role),
                               "project": project_view(project)})

        if action == "roles_merge":
            store.merge_roles(project, int(payload["target_id"]),
                              int(payload["source_id"]))
            store.save(project)
            return self._json({"ok": True, "project": project_view(project)})

        if action == "roles_update":
            role = store.update_role(
                project, int(payload["role_id"]), payload.get("name"),
                payload.get("color"), payload.get("type"),
            )
            store.save(project)
            return self._json({"ok": True, "role": role_view(role),
                               "project": project_view(project)})

        if action == "roles_delete_pending":
            removed = store.delete_pending_roles(project)
            store.save(project)
            return self._json({"ok": True, "removed": removed,
                               "project": project_view(project)})

        if action == "enroll":
            role, notes = store.enroll_role(
                project, int(payload["role_id"]), float(payload["start"]),
                float(payload["end"]), also_library=bool(payload.get("also_library")),
            )
            store.save(project)
            return self._json({"ok": True, "role": role_view(role),
                               "notes": notes, "project": project_view(project)})

        if action == "segments":
            store.set_segment(project, int(payload["segment_id"]),
                              payload.get("speaker_id"))
            store.save(project)
            return self._json({"ok": True, "project": project_view(project)})

        if action == "export":
            written = store.export(project, payload.get("formats") or ["srt"],
                                   payload.get("options") or {})
            return self._json({"ok": True, "files": written,
                               "project": project_view(project)})

        if action == "promote":
            role = next((r for r in project["roles"]
                         if r["id"] == int(payload["role_id"])), None)
            if role is None:
                return self._error("角色不存在", 404)
            _, lib_action = store.add_library_role(
                role["name"], role["color"],
                role.get("embedding"), role.get("samples"),
                embedder=role.get("embedder", "builtin"))
            role["in_library"] = True
            store.save(project)
            return self._json({"ok": True, "action": lib_action,
                               "project": project_view(project),
                               "library": state_payload()["library"]})

        return self._error(f"未知操作：{action}", 404)

    # -- media streaming + downloads ---------------------------------------
    def _stream_media(self, project: dict) -> None:
        path = project["media_path"]
        if not os.path.isfile(path):
            return self._error("媒体文件已不存在", 404)
        size = os.path.getsize(path)
        mime = MEDIA_MIME.get(os.path.splitext(path)[1].lower()) or "application/octet-stream"
        range_header = self.headers.get("Range") or self.headers.get("range")
        start, end = 0, size - 1
        status = 200
        if range_header:
            match = re.match(r"bytes=(\d*)-(\d*)", range_header.strip())
            if match and (match.group(1) or match.group(2)):
                first, last = match.group(1), match.group(2)
                if first and last:
                    start, end = int(first), int(last)
                elif first:            # bytes=start-
                    start, end = int(first), size - 1
                else:                  # bytes=-N  -> last N bytes
                    start, end = max(0, size - int(last)), size - 1
                if size == 0 or start >= size or start > end:
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                end = min(end, size - 1)
                status = 206
        length = end - start + 1 if size > 0 else 0
        self.send_response(status)
        self.send_header("Content-Type", mime)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        if self.command == "HEAD":
            return
        with open(path, "rb") as handle:
            handle.seek(start)
            remaining = length
            while remaining > 0:
                chunk = handle.read(min(256 * 1024, remaining))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    return
                remaining -= len(chunk)

    def _download(self, project_id: str, filename: str) -> None:
        folder = os.path.join(store.EXPORT_DIR, project_id)
        path = os.path.normpath(os.path.join(folder, filename))
        if not _within(path, folder) or not os.path.isfile(path):
            return self._error("导出文件不存在", 404)
        mime = mimetypes.guess_type(path)[0] or "application/octet-stream"
        if path.endswith((".md", ".csv", ".jsonl", ".rttm", ".srt", ".vtt", ".ass")):
            mime = "text/plain; charset=utf-8"
        with open(path, "rb") as handle:
            body = handle.read()
        quoted = urllib.parse.quote(filename)
        self._send(200, body, mime,
                   {"Content-Disposition": f"attachment; filename*=UTF-8''{quoted}"})


def serve(host: str = "127.0.0.1", port: int = 8770) -> None:
    """Run the blocking HTTP server. Used by ``main`` and by a frozen shell."""
    store._ensure_dirs()

    # Probing engines imports torch (several seconds). Do it off the request path
    # so the first page load is not stuck behind it.
    threading.Thread(target=diarize.warm_engines, daemon=True).start()

    server = QuietHTTPServer((host, port), Handler)
    print(f"Grain 已启动： http://{host}:{port}")
    print("按 Ctrl+C 停止。")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止。")
    finally:
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Grain")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8770)
    args = parser.parse_args()
    serve(args.host, args.port)


if __name__ == "__main__":
    main()
