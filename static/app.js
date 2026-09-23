/* Grain —— 前端逻辑（原生 JS，无依赖） */

const API = {
  async _parse(res, path) {
    let data = null;
    try { data = await res.json(); } catch (err) { data = null; }
    if (data === null) throw new Error(`服务响应不是合法 JSON（HTTP ${res.status}）：${path}`);
    if (!res.ok || data.ok === false) throw new Error(data.error || `请求失败: ${path}`);
    return data;
  },
  async get(path) {
    return this._parse(await fetch(path), path);
  },
  async post(path, body) {
    const res = await fetch(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body || {}),
    });
    return this._parse(res, path);
  },
  async del(path) {
    return this._parse(await fetch(path, { method: "DELETE" }), path);
  },
};

const state = {
  meta: null,
  project: null,
  projectId: null,
  selectedId: null,
  selection: new Set(),
  filter: "all",
  search: "",
  zoom: 1,
  pxPerSecond: 60,
  viewStart: 0,
  following: true,
  dragging: false,
  dragStartX: 0,
  dragStartView: 0,
  undo: [],
  redo: [],
  peaks: [],
  busy: false,
  media: null,
  tool: "select",       // select | hand (PR: V / H)
  drag: null,           // active track gesture descriptor
  scrubbing: false,     // true while the ruler is being dragged
  seekTarget: null,     // coalesced scrub target (applied once per frame)
  laneCache: null,      // reused lane DOM, rebuilt only when the lanes change
  waveCache: null,      // offscreen waveform strip, re-sliced instead of resampled
  blockNodes: new Map(),// id -> reused clip node, so a gesture never rebuilds the DOM
  roleMap: new Map(),   // id -> role, so hot render loops never linear-search roles
  segIndex: null,       // {sorted, starts, edges}: binary-searchable cue index
  segById: new Map(),   // id -> segment hash map for O(1) lookup
  laneGroupsCache: null,// {project, groups}: rebuilt only when the project changes
  speakerUi: null,      // cached speaker-button DOM, only active classes update
  cssVars: null,        // cached :root custom properties (theme changes rarely)
};

const el = (id) => document.getElementById(id);
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

/* Timeline zoom model. One source of truth: `state.zoom` is the multiplier over
   the base window (30s of content across the strip), and `state.pxPerSecond`
   is always derived from it, so the × label, the buttons and the view agree.
   The floor is dynamic because "适配" a feature-length recording needs a zoom
   well below the interactive 0.25× floor. */
const BASE_WINDOW = 30;      // seconds visible across the strip at zoom 1
const ZOOM_MIN = 0.25;       // interactive floor (unless fit needs to go lower)
const ZOOM_MAX = 24;
const ZOOM_IN = 1.35, ZOOM_OUT = 0.74;

const ppsForZoom = (zoom, width) => (width / BASE_WINDOW) * zoom;
const zoomForPps = (pps, width) => pps / (width / BASE_WINDOW);
const fitPps = (width, duration) => (width - 24) / Math.max(0.001, duration);
/* Lowest zoom this clip is allowed: never above the interactive floor, but low
   enough that fitting the whole duration is always reachable. */
const minZoomFor = (width, duration) => Math.min(ZOOM_MIN, zoomForPps(fitPps(width, duration), width));


/* ----------------------------------------------------------- ui prefs */

const PREFS_KEY = "ssp.prefs.v1";
const prefs = Object.assign({
  theme: "dark",        // dark | light | auto
  accent: "violet",     // violet | blue | rose | green
  autonext: true,
  loop: false,
  follow: true,
  zoom: 1,
  rate: 1,
  showKbd: true,
  lang: "both",         // primary | both | translation
  snap: true,           // snap the playhead to cue edges
  mediaRows: null,      // user-dragged media pane height (null = automatic)
  overlay: true,        // burn subtitles onto the video
}, JSON.parse(localStorage.getItem(PREFS_KEY) || "{}"));

function savePrefs() {
  localStorage.setItem(PREFS_KEY, JSON.stringify(prefs));
}

function applyTheme() {
  const root = document.documentElement;
  const theme = prefs.theme === "auto"
    ? (window.matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark")
    : prefs.theme;
  root.dataset.theme = theme;
  root.dataset.accent = prefs.accent;
  state.cssVars = null;  // theme/accent changed: the cached custom props are stale
}

/* Toggle straight to the opposite of what is currently on screen. */
function toggleTheme() {
  const current = document.documentElement.dataset.theme;
  prefs.theme = current === "light" ? "dark" : "light";
  savePrefs();
  applyTheme();
  setMessage(prefs.theme === "dark" ? "已切换到深色" : "已切换到浅色");
}

/* --------------------------------------------------------- window controls */
/* The desktop shell is frameless (no OS title bar), so min / max / close are
   our own buttons in the top toolbar. They only work under pywebview, which
   injects `window.pywebview` and fires `pywebviewready` once the bridge is up;
   in a plain browser the buttons stay hidden. */
function wireWindowControls() {
  const apply = () => {
    if (!window.pywebview?.api) return;
    document.body.classList.add("desktop-shell");
    el("win-min")?.addEventListener("click", () => window.pywebview.api.window_minimize());
    el("win-max")?.addEventListener("click", () => window.pywebview.api.window_maximize());
    el("win-close")?.addEventListener("click", () => window.pywebview.api.window_close());
  };
  if (window.pywebview?.api) apply();
  else window.addEventListener("pywebviewready", apply, { once: true });
}

/* The whole toolbar is a pywebview drag region, so the shell must not treat a
   press on one of its controls as a window drag. Stopping the mousedown before
   it bubbles to the shell's body listener keeps buttons / selects clickable. */
function wireToolbarDrag() {
  const toolbar = document.querySelector(".toolbar");
  if (!toolbar) return;
  toolbar.querySelectorAll("button, select, input, a").forEach((node) => {
    node.addEventListener("mousedown", (event) => event.stopPropagation());
  });
}

/* Choose the initial zoom and scroll position for the timeline. Short media fits
   the whole span; long media opens at the user's default zoom, scrolled to where
   the dialogue actually starts, so a silent cold-open is not shown. Only called
   when a project is mounted — tweaking a preference must never reset the view. */
function initTimelineView() {
  const outer = document.querySelector(".track-outer");
  if (!outer || !state.project) return;
  const duration = state.project.duration || 0;
  const geom = trackGeom();
  const width = geom.width;
  if (duration && duration < 120) {
    applyFitToWidth();
    renderTrack();
    return;
  }
  state.zoom = clamp(prefs.zoom || 1, minZoomFor(width, duration), 4);
  state.pxPerSecond = ppsForZoom(state.zoom, width);
  // Open on the content, not on a silent leader: park the playhead at the first
  // cue (Premiere keeps the playhead visible when a sequence opens, and this
  // project's speech starts well after t=0). The viewport follows the playhead.
  const cues = state.project.segments || [];
  const firstStart = cues.length ? cues[0].start : 0;
  const visible = width / state.pxPerSecond;
  if (state.media && firstStart > 0.5) {
    state.media.currentTime = firstStart;
  }
  // Park the playhead at the same ratio follow-play will hold it at, so starting
  // playback does not immediately slide the view.
  const anchorRatio = 0.35;
  state.viewStart = clamp(firstStart - visible * anchorRatio, 0, Math.max(0, duration - visible));
  renderTrack();
}

/* Apply the "默认缩放" preference to the current view without touching playback
   position (the settings slider calls this instead of a full timeline reset). */
function applyDefaultZoom() {
  if (!state.project) return;
  const width = trackGeom().width;
  const duration = state.project.duration || 1;
  state.zoom = clamp(prefs.zoom || 1, minZoomFor(width, duration), ZOOM_MAX);
  state.pxPerSecond = ppsForZoom(state.zoom, width);
  state.viewStart = clamp(state.viewStart, 0, Math.max(0, duration - width / state.pxPerSecond));
  renderTrack();
}

function applyPrefsToWorkspace() {
  const autonext = el("chk-autonext"), loop = el("chk-loop"), follow = el("chk-follow");
  if (autonext) autonext.checked = prefs.autonext;
  if (loop) loop.checked = prefs.loop;
  if (follow) { follow.checked = prefs.follow; state.following = prefs.follow; }
  if (state.media) state.media.playbackRate = prefs.rate;
  document.body.classList.toggle("hide-kbd", !prefs.showKbd);
}

function fmtTime(seconds, withMs = false) {
  if (!isFinite(seconds)) seconds = 0;
  const total = Math.max(0, seconds);
  const m = Math.floor(total / 60);
  const s = Math.floor(total % 60);
  const base = `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
  if (!withMs) return base;
  return `${base}.${String(Math.floor((total % 1) * 1000)).padStart(3, "0")}`;
}

function setMessage(text, kind = "") {
  const node = el("stat-message");
  if (node) { node.textContent = text || ""; node.className = "hint " + kind; }
}

function setBusy(text) {
  state.busy = Boolean(text);
  const node = el("detect-status");
  if (node) { node.textContent = text || ""; node.className = "status " + (text ? "busy" : ""); }
}

/* ---------------------------------------------------------------- helpers */

function roleById(id) {
  if (id === null || id === undefined) return null;
  return state.roleMap.get(id)
    || (state.project?.roles || []).find((r) => r.id === id) || null;
}

function segmentById(id) {
  const index = state.segIndex;
  if (index && state.segById) return state.segById.get(id) || null;
  return (state.project?.segments || []).find((s) => s.id === id) || null;
}

/* Cue lookups by id and by time must not scan the whole list: `segmentById`
   runs on every selection change and `segmentAtTime` runs every animation frame
   during playback. Build sorted arrays + hash maps once per project instead. */
function buildSegmentIndex() {
  const segments = state.project?.segments || [];
  const sorted = [...segments].sort((a, b) => a.start - b.start);
  const starts = sorted.map((s) => s.start);
  const edges = [];
  const byId = new Map();
  for (const seg of segments) byId.set(seg.id, seg);
  for (const seg of sorted) { edges.push(seg.start, seg.end); }
  edges.sort((a, b) => a - b);
  state.segIndex = { sorted, starts, edges };
  state.segById = byId;
}

/* Cached :root custom properties. `getComputedStyle` forces a style recalc;
   calling it on every render frame was measurable, and the theme changes only
   on an explicit toggle (which clears this cache via `applyTheme`). */
function cssVars() {
  if (state.cssVars) return state.cssVars;
  const root = getComputedStyle(document.documentElement);
  state.cssVars = {
    mono: root.getPropertyValue("--mono").trim(),
    accent: root.getPropertyValue("--accent").trim(),
    grid: root.getPropertyValue("--line-2").trim(),
    muted: root.getPropertyValue("--muted").trim(),
  };
  return state.cssVars;
}

function currentSegments() {
  if (!state.project) return [];
  let items = state.project.segments;
  if (state.filter === "pending") items = items.filter((s) => s.speaker_id === null);
  else if (state.filter === "manual") items = items.filter((s) => s.status === "manual");
  else if (state.filter === "low") items = items.filter((s) => s.confidence !== null && s.confidence < 0.6);
  if (state.search) {
    const needle = state.search.toLowerCase();
    items = items.filter((s) => (s.text || "").toLowerCase().includes(needle));
  }
  return items;
}

function speakerColor(seg) {
  if (seg.speaker_id === null) return "#b9bec7";
  const role = roleById(seg.speaker_id);
  return role ? role.color : "#b9bec7";
}

function speakerName(seg) {
  if (seg.speaker_id === null) return "待定";
  const role = roleById(seg.speaker_id);
  return role ? role.name : `#${seg.speaker_id}`;
}

/* --------------------------------------------------------------- snapshot */

function snapshotSegments(ids) {
  const snap = {};
  for (const id of ids) {
    const seg = segmentById(id);
    if (seg) snap[id] = { speaker_id: seg.speaker_id, status: seg.status, confidence: seg.confidence };
  }
  return snap;
}

function pushUndo(snap) {
  state.undo.push(snap);
  if (state.undo.length > 100) state.undo.shift();
  state.redo = [];
}

async function restore(snap) {
  const ids = Object.keys(snap).map(Number);
  const groups = new Map();
  for (const id of ids) {
    const entry = snap[id];
    // speaker_id, status and confidence all round-trip, so undoing an auto
    // assignment restores its 85% badge instead of degrading it to "自动".
    const key = JSON.stringify([entry.speaker_id, entry.status, entry.confidence]);
    if (!groups.has(key)) groups.set(key, { ids: [], entry });
    groups.get(key).ids.push(id);
  }
  for (const { ids: groupIds, entry } of groups.values()) {
    await API.post(`/api/projects/${state.projectId}/bulk`, {
      ids: groupIds,
      speaker_id: entry.speaker_id,
      status: entry.status,
      confidence: entry.confidence,
    });
  }
  await refreshProject();
}

/* ------------------------------------------------------------ assignment */

async function assign(segmentId, speakerId) {
  const seg = segmentById(segmentId);
  if (!seg) return;
  const snap = snapshotSegments([segmentId]);
  try {
    const data = await API.post(`/api/projects/${state.projectId}/segments`, {
      segment_id: segmentId, speaker_id: speakerId,
    });
    pushUndo(snap);
    applyProject(data.project);
    if (prefs.autonext) goToSegment(segmentId, +1);
  } catch (err) {
    setMessage(err.message);
  }
}

async function assignMany(ids, speakerId) {
  if (!ids.length) return;
  const snap = snapshotSegments(ids);
  try {
    const data = await API.post(`/api/projects/${state.projectId}/bulk`, {
      ids, speaker_id: speakerId,
    });
    pushUndo(snap);
    applyProject(data.project);
    setMessage(`已批量归属 ${data.changed} 条字幕`, "ok");
  } catch (err) {
    setMessage(err.message);
  }
}

function onSpeakerClick(speakerId) {
  if (state.selection.size) {
    const ids = [...state.selection];
    state.selection.clear();
    assignMany(ids, speakerId);
  } else if (state.selectedId !== null) {
    assign(state.selectedId, speakerId);
  }
}

/* -------------------------------------------------------------- rendering */

function applyProject(project) {
  if (!project) { setMessage("项目数据加载失败"); return; }
  const prevMediaPath = state.project?.media_path;
  const changedProject = state.projectId !== project.id;
  // renderAll() mounts the media element itself when it creates the workspace,
  // so remember that here and skip the remount below (it used to load the file
  // and fetch peaks twice on the first project).
  const stageWasEmpty = el("stage")?.classList.contains("empty");
  state.project = project;
  state.projectId = project.id;
  // Reused-node caches are keyed on this project's cue ids, turns and roles: a
  // different project (or a re-run detection) must start them from scratch.
  if (changedProject) {
    state.blockNodes = new Map();
    state.laneCache = null;
    state.seekTarget = null;
    state.speakerUi = null;
  }
  // Indexes and the lane-group cache depend on this exact project object. Every
  // mutation replaces it, so an identity check is enough to invalidate them —
  // this is what keeps per-frame rendering O(visible) instead of O(all cues).
  buildSegmentIndex();
  state.roleMap = new Map((project.roles || []).map((role) => [role.id, role]));
  state.laneGroupsCache = null;
  if (state.selectedId === null || !segmentById(state.selectedId)) {
    state.selectedId = project.segments.length ? project.segments[0].id : null;
  }
  renderAll();
  // Switching projects must swap the media element too — otherwise the old
  // file keeps playing under the new project's timeline. Zoom is also reset
  // per project (fit for short media), so stale zoom never squashes the view.
  if (!stageWasEmpty && prevMediaPath !== project.media_path && document.querySelector(".workspace")) {
    mountMedia();
    applyPrefsToWorkspace();
  }
}

function renderAll() {
  renderWorkspace();
  fitTrackPane();
  renderStats();
  renderSide();
  renderTrack();
  renderCurrent();
  renderQueue();
  renderProjectSelect();
}

/* Side panel: roles, detection status, notes (replaces the old tall sidebar). */
function renderSide() {
  const project = state.project;
  if (!project) return;

  const roleHost = el("role-summary");
  if (roleHost) {
    roleHost.innerHTML = "";
    const roles = project.roles || [];
    if (!roles.length) {
      roleHost.innerHTML = '<div class="hint">还没有角色。点「+ 新增」创建角色，检测匹配上后会自动归属。</div>';
    }
    // One tally pass instead of a full scan per role (1904 cues x N roles).
    const counts = new Map();
    for (const segment of project.segments) {
      const id = segment.speaker_id;
      if (id !== null && id !== undefined) counts.set(id, (counts.get(id) || 0) + 1);
    }
    for (const role of roles) {
      const chip = document.createElement("div");
      chip.className = "role-chip";
      chip.title = `${role.name} — ${counts.get(role.id) || 0} 条${role.has_voiceprint ? "（有声纹）" : ""}`;
      chip.innerHTML = `<span class="swatch" style="background:${role.color}"></span>
        <b>${escapeHtml(role.name)}</b>
        <span class="pill">${counts.get(role.id) || 0}</span>`;
      chip.addEventListener("click", () => renameRole(role.id));
      roleHost.appendChild(chip);
    }
  }

  const detectHost = el("detect-summary");
  if (detectHost) {
    const notes = project.detection_notes || [];
    const engineLabels = {};
    for (const [key, info] of Object.entries(state.meta?.engines || {})) engineLabels[key] = info.label;
    // Only the two numbers that drive the review loop stay expanded; engine,
    // cluster count and the detector's notes live one click away.
    detectHost.innerHTML = `
      <div class="row"><span>已归属</span><b>${project.stats.assigned} / ${project.stats.total}</b></div>
      <div class="row"><span>待定</span><b>${project.stats.pending}</b></div>
      <details class="detect-notes"><summary>检测详情</summary>
        <div class="row"><span>引擎</span><b>${escapeHtml(project.engine ? (engineLabels[project.engine] || project.engine) : "未运行")}</b></div>
        <div class="row"><span>聚类 / 说话人</span><b>${(project.turns || []).length} / ${project.stats.speakers_used}</b></div>
        ${notes.length ? `<ul class="note-list">${notes.map((n) => `<li>${escapeHtml(n)}</li>`).join("")}</ul>` : ""}
      </details>
    `;
  }
}

function renderProjectSelect() {
  const select = el("project-select");
  if (!select || !state.meta) return;
  const options = state.meta.projects || [];
  select.innerHTML = "";
  if (!options.length) {
    const opt = document.createElement("option");
    opt.textContent = "（无项目）";
    opt.value = "";
    select.appendChild(opt);
    return;
  }
  for (const item of options) {
    const opt = document.createElement("option");
    opt.value = item.id;
    const pct = item.stats?.total ? Math.round((item.stats.assigned / item.stats.total) * 100) : 0;
    opt.textContent = `${item.name} · ${pct}%`;
    if (item.id === state.projectId) opt.selected = true;
    select.appendChild(opt);
  }
}

function renderStats() {
  const stats = state.project?.stats;
  if (!stats) return;
  el("stat-total").textContent = String(stats.total);
  el("stat-assigned").textContent = String(stats.assigned);
  el("stat-pending").textContent = String(stats.pending);
  el("stat-roles").textContent = String(stats.roles);
  el("stat-progress").textContent =
    `${stats.total ? Math.round((stats.assigned / stats.total) * 100) : 0}%`;
}

/* Mount the workspace shell the first time it is shown. resetWorkspace() may
   have mounted it already (no project); later applyProject() calls reuse it. */
function renderWorkspace() {
  if (!state.project) return;
  const stage = el("stage");
  if (stage.querySelector(".workspace")) return;
  stage.classList.remove("empty");
  stage.replaceChildren(el("workspace-template").content.cloneNode(true));
  bindWorkspace();
  mountMedia();
}

function mountMedia() {
  const host = el("media-host");
  if (!host || !state.project) return;
  host.innerHTML = "";
  const isAudio = state.project.media_kind === "audio";
  const media = document.createElement(isAudio ? "audio" : "video");
  media.controls = true;
  media.preload = "metadata";
  media.src = state.project.media_url;
  media.addEventListener("timeupdate", onTimeUpdate);
  media.addEventListener("play", () => {
    el("btn-play").textContent = "⏸";
    // Resuming playback re-enables follow-play per the user's preference.
    state.following = prefs.follow;
    const box = el("chk-follow");
    if (box) box.checked = prefs.follow;
    ensurePlaybackLoop();
  });
  media.addEventListener("playing", ensurePlaybackLoop);
  media.addEventListener("pause", () => {
    el("btn-play").textContent = "▶";
    stopPlaybackLoop();
  });
  media.addEventListener("seeked", () => {
    renderPlaybackFrame();
    ensurePlaybackLoop();
  });
  media.addEventListener("loadedmetadata", () => {
    if (!state.project.duration) state.project.duration = media.duration;
    el("time-display").textContent =
      `${fmtTime(media.currentTime, true)} / ${fmtTime(state.project.duration || media.duration)}`;
    fitMediaPane(media, isAudio);
    renderTrack();
  });
  media.addEventListener("ended", () => {
    el("btn-play").textContent = "▶";
    stopPlaybackLoop();
  });
  host.appendChild(media);
  state.media = media;
  loadPeaks();
  fitMediaPane(media, isAudio);
  initTimelineView();
}

/* Reserve exactly the height the timeline needs for its ruler, lanes and clip
   band. Without this the row is an `auto` track that the grid compresses first,
   squashing six speaker lanes into ~100px. */
function fitTrackPane() {
  // Lanes are the detection tracks actually drawn (one per mapped cluster), not
  // one per role: a hand-added role with no turns must not reserve a blank row.
  const lanes = state.project ? laneGroups().length : 0;
  const CHROME = 42 + 44;           // panel head + meta bar
  const BODY = 34 + 20 + 84 + 14;   // padding + ruler + clip band + padding
  const LANE = 24, GAP = 3;
  const laneBlock = lanes ? lanes * LANE + (lanes - 1) * GAP + 6 : 0;
  const height = CHROME + BODY + laneBlock;
  document.documentElement.style.setProperty("--track-rows", `${Math.round(height)}px`);
}

/* Size the media row from the picture's aspect ratio, but never at the expense
   of the timeline and the cue editor: media is the part that can afford to
   shrink, so it absorbs whatever space is left over. */
function fitMediaPane(media, isAudio) {
  const host = el("media-host");
  if (!host) return;
  fitTrackPane();
  const paneWidth = host.clientWidth || 800;
  const ratio = (media && media.videoWidth && media.videoHeight)
    ? media.videoWidth / media.videoHeight
    : (isAudio ? 0 : 16 / 9);

  const CHROME = 42;                 // panel header
  const OUTER = 52 + 34;             // toolbar + status bar
  const GAPS = 10 * 3 + 10 * 2;      // workspace gaps + padding
  const CUE = 122;                   // cue bar: its own natural height
  const QUEUE_FLOOR = 110;           // keep the clip list usable
  const reserved = readPixelVar("--track-rows", 230) + CUE + QUEUE_FLOOR;

  if (!ratio) {
    // Audio has no picture: keep a compact strip for the native controls.
    document.documentElement.style.setProperty("--media-rows", "150px");
    return;
  }
  const roomForMedia = window.innerHeight - OUTER - GAPS - reserved;
  // The top band is a fixed height; derive the picture's natural width from it
  // so the media column is never wider than the 16:9 frame needs (surplus width
  // would otherwise show up as black bars either side of the video).
  const rows = prefs.mediaRows
    ? prefs.mediaRows
    : Math.max(276, Math.min(paneWidth / ratio + CHROME, Math.max(276, roomForMedia)));
  document.documentElement.style.setProperty("--media-rows", `${Math.round(rows)}px`);
  const naturalWidth = Math.round((rows - CHROME) * ratio);
  const colWidth = clamp(naturalWidth, 330, Math.min(620, window.innerWidth * 0.42));
  document.documentElement.style.setProperty("--media-col", `${colWidth}px`);
}

/* Draggable divider between the media pane and the timeline. The two compete
   for the same vertical space, and how to split it depends on the screen and on
   whether the user is watching the picture or working the track. */
function wireMediaResizer() {
  const handle = el("media-resizer");
  if (!handle) return;
  const startDrag = (event) => {
    event.preventDefault();
    const startY = event.clientY;
    const startRows = readPixelVar("--media-rows", 300);
    const minRows = 150;
    const maxRows = window.innerHeight - 260;
    handle.classList.add("active");
    document.body.classList.add("resizing-media");
    const onMove = (moveEvent) => {
      const next = clamp(startRows + (moveEvent.clientY - startY), minRows, maxRows);
      document.documentElement.style.setProperty("--media-rows", `${Math.round(next)}px`);
      renderTrack();
    };
    const onUp = () => {
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
      handle.classList.remove("active");
      document.body.classList.remove("resizing-media");
      prefs.mediaRows = Math.round(readPixelVar("--media-rows", 300));
      savePrefs();
      fitMediaPane(state.media, state.project?.media_kind === "audio");
      renderTrack();
    };
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
  };
  handle.addEventListener("mousedown", startDrag);
  // double-click resets to the automatic height
  handle.addEventListener("dblclick", () => {
    prefs.mediaRows = null;
    savePrefs();
    fitMediaPane(state.media, state.project?.media_kind === "audio");
    renderTrack();
  });
}

function readPixelVar(name, fallback) {
  const raw = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  const value = parseFloat(raw);
  return Number.isFinite(value) ? value : fallback;
}

async function loadPeaks() {
  try {
    const data = await API.get(`/api/projects/${state.projectId}/peaks?buckets=1400`);
    state.peaks = data.peaks || [];
    state.waveCache = null; // peak set changed: rebuild the strip
    renderTrack();
  } catch (err) { /* waveform is optional */ }
}

function visibleSeconds() {
  return trackGeom().width / state.pxPerSecond;
}

/* Burn the current cue onto the picture. Reads the same cueText/cueTranslation
   helpers as the panels, so the language toggle drives it too. */
function renderVideoOverlay() {
  const box = el("video-overlay");
  const textNode = el("overlay-text");
  const translationNode = el("overlay-translation");
  if (!box || !textNode || !translationNode) return;
  const seg = state.media ? segmentAtTime(state.media.currentTime) : null;
  if (!seg || prefs.overlay === false) {
    box.classList.add("hidden");
    return;
  }
  box.classList.remove("hidden");
  textNode.textContent = cueText(seg);
  translationNode.textContent = cueTranslation(seg);
}

/* The cue covering `time`, or null in a gap between cues. Binary search over
   the start-sorted index: this runs on every animation frame while playing, and
   a linear scan over thousands of cues dominated the frame budget. */
function segmentAtTime(time) {
  if (!state.project) return null;
  const index = state.segIndex;
  if (!index || !index.sorted.length) {
    return state.project.segments.find(
      (s) => time >= s.start - 0.02 && time < s.end
    ) || null;
  }
  const threshold = time + 0.02;
  const { sorted, starts } = index;
  let lo = 0, hi = starts.length - 1, pos = -1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (starts[mid] <= threshold) { pos = mid; lo = mid + 1; }
    else hi = mid - 1;
  }
  if (pos < 0) return null;
  const seg = sorted[pos];
  if (time < seg.end) return seg;
  // Overlapping cues are rare; check a small neighbourhood before giving up.
  for (let i = pos - 1; i >= 0 && i >= pos - 3; i -= 1) {
    const candidate = sorted[i];
    if (candidate.end > time && candidate.start <= threshold) return candidate;
  }
  return null;
}

/* Which text to show for a cue, honouring the language toggle. */
function cueText(seg) {
  const primary = seg.text || "";
  const translation = (seg.translation || "").trim();
  if (prefs.lang === "translation") return translation || primary;
  return primary;
}

function cueTranslation(seg) {
  const translation = (seg.translation || "").trim();
  if (!translation || prefs.lang !== "both") return "";
  return translation;
}

function renderRuler(canvas, ctx, width, viewStart, pps, duration, dpr, mono, accentColor,
                     gridColor, labelColor) {
  const height = 20;
  const bw = Math.max(1, Math.floor(width * dpr));
  const bh = Math.floor(height * dpr);
  if (canvas.width !== bw) canvas.width = bw;
  if (canvas.height !== bh) canvas.height = bh;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, width, height);
  // Pick a "nice" tick step so labels stay readable at any zoom.
  const targets = [0.1, 0.25, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600];
  const minPx = 58;
  let step = targets[targets.length - 1];
  for (const candidate of targets) {
    if (candidate * pps >= minPx) { step = candidate; break; }
  }
  ctx.font = "10px " + (mono || "monospace");
  ctx.textBaseline = "top";
  const first = Math.floor(viewStart / step) * step;
  for (let t = first; t <= viewStart + width / pps + step; t += step) {
    const x = Math.round((t - viewStart) * pps) + 0.5;
    if (x < -20 || x > width + 20) continue;
    ctx.strokeStyle = gridColor || "rgba(255,255,255,.14)";
    ctx.beginPath();
    ctx.moveTo(x, 8);
    ctx.lineTo(x, height);
    ctx.stroke();
    ctx.fillStyle = labelColor || "rgba(200,205,220,.55)";
    const label = step < 1
      ? `${t.toFixed(2)}s`
      : t >= 60 ? `${Math.floor(t / 60)}:${String(Math.round(t % 60)).padStart(2, "0")}` : `${Math.round(t)}s`;
    ctx.fillText(label, x + 4, 0);
  }
  // Playhead marker on the ruler: the grab handle for scrubbing.
  if (state.media) {
    const px = (state.media.currentTime - viewStart) * pps;
    if (px >= -8 && px <= width + 8) {
      ctx.fillStyle = accentColor || "#8b7cff";
      ctx.beginPath();
      ctx.moveTo(px - 6, 0);
      ctx.lineTo(px + 6, 0);
      ctx.lineTo(px, 10);
      ctx.closePath();
      ctx.fill();
      ctx.fillRect(px - 1, 0, 2, height);
    }
  }
}

/* Group detected turns by the role they were mapped to (or by raw cluster). */
function laneGroups() {
  const mapping = state.project.cluster_to_role || {};
  const roleById = new Map((state.project.roles || []).map((r) => [r.id, r]));
  const groups = new Map();
  for (const turn of state.project.turns || []) {
    const roleId = mapping[String(turn.cluster)];
    const key = roleId == null ? `c${turn.cluster}` : `r${roleId}`;
    let group = groups.get(key);
    if (!group) {
      const role = roleId == null ? null : roleById.get(roleId);
      group = {
        key,
        label: role ? role.name : `聚类 ${turn.cluster}`,
        color: role ? role.color : "#98a0af",
        pending: !role,
        spans: [],
      };
      groups.set(key, group);
    }
    group.spans.push(turn);
  }
  // Cues assigned by hand have no detected turn, so their role would otherwise
  // get no lane at all. Give every such role its own row, built from its cue
  // spans (one bar per cue) so the track shows where that speaker talks.
  const represented = new Set(
    [...groups.keys()].filter((k) => k.startsWith("r")).map((k) => Number(k.slice(1))));
  const manualSpans = new Map();
  for (const seg of state.project.segments || []) {
    const id = seg.speaker_id;
    if (id === null || id === undefined || represented.has(id)) continue;
    if (!manualSpans.has(id)) manualSpans.set(id, []);
    manualSpans.get(id).push({ start: seg.start, end: seg.end });
  }
  for (const [roleId, spans] of manualSpans) {
    const role = roleById.get(roleId);
    if (!role || role.type === "ignored") continue;
    groups.set(`r${roleId}`, {
      key: `r${roleId}`,
      label: role.name,
      color: role.color,
      pending: role.type === "pending",
      manual: true,
      spans,
    });
  }
  return [...groups.values()];
}

function positionLaneBars(groupBars, viewStart, pps, width) {
  const from = viewStart - 1;
  const to = viewStart + width / pps + 1;
  for (const { bar, span } of groupBars) {
    const visible = span.end >= from && span.start <= to;
    if (!visible) {
      if (bar.hidden !== true) bar.hidden = true;
      continue;
    }
    if (bar.hidden) bar.hidden = false;
    bar.style.left = `${(span.start - viewStart) * pps}px`;
    bar.style.width = `${Math.max(2, (span.end - span.start) * pps)}px`;
  }
}

/* Lanes are structural (one row per speaker) and change only when detection
   runs or roles are remapped, so the DOM is built once and then re-positioned.
   Rebuilding it on every pan frame was the bulk of the drag stutter. */
function cachedLaneGroups() {
  const cache = state.laneGroupsCache;
  if (cache && cache.project === state.project) return cache.groups;
  const groups = laneGroups();
  state.laneGroupsCache = { project: state.project, groups };
  return groups;
}

function renderLanes(container, viewStart, pps, width) {
  const groups = cachedLaneGroups();
  const signature = groups
    .map((g) => `${g.key}\u0001${g.label}\u0001${g.color}\u0001${g.spans.length}`)
    .join("\u0002");
  if (!state.laneCache || state.laneCache.signature !== signature) {
    container.innerHTML = "";
    const bars = [];
    for (const group of groups) {
      const lane = document.createElement("div");
      lane.className = "lane" + (group.pending ? " pending" : "");
      const label = document.createElement("div");
      label.className = "lane-label";
      label.textContent = group.label;
      lane.appendChild(label);
      bars.push(group.spans.map((span) => {
        const bar = document.createElement("div");
        bar.className = "lane-bar";
        bar.style.setProperty("--lane-color", group.color);
        bar.title = `${group.label}  ${span.start.toFixed(2)}s–${span.end.toFixed(2)}s`;
        bar.addEventListener("click", (event) => {
          event.stopPropagation();
          if (state.media) state.media.currentTime = span.start;
        });
        lane.appendChild(bar);
        return { bar, span };
      }));
      container.appendChild(lane);
    }
    state.laneCache = { signature, bars };
  }
  for (const groupBars of state.laneCache.bars) {
    positionLaneBars(groupBars, viewStart, pps, width);
  }
}
function laneGutterWidth() {
  const roles = (state.project?.roles || []).filter((r) => r.type !== "ignored");
  if (!roles.length) return 0;
  const longest = Math.max(...roles.map((r) => (r.name || "").length));
  // ~11px per CJK glyph plus padding, clamped to a sensible band.
  return clamp(longest * 12 + 14, 56, 132);
}

/* The single geometry every timeline layer must agree on.
 *
 * The speaker-label gutter is carved out of the left edge by CSS, so the ruler,
 * lanes and clip band start `gutter` px inside `.track-outer`. The playhead and
 * every pointer->time conversion are expressed in `.track-outer` coordinates,
 * which is why they must add `left` back: skipping it offsets the playhead from
 * the ruler/waveform/blocks by exactly the gutter width. */
function trackGeom() {
  const outer = document.querySelector(".track-outer");
  const total = outer ? outer.clientWidth : 800;
  const gutter = laneGutterWidth();
  return { total, gutter, left: gutter, width: Math.max(80, total - gutter) };
}

/* Pre-render the waveform for a ~3-screen window to an offscreen canvas, so the
   per-frame draw is a whole-pixel blit (no re-sampling => no shimmer). Rebuilt
   only when the zoom, theme, peak set or covered time range changes. */
function ensureWaveStrip(dpr, pps, viewStart, width, duration, waveHeight, accentColor) {
  const screenSec = width / pps;
  const cache = state.waveCache;
  const needEnd = viewStart + screenSec;
  if (cache && cache.pps === pps && cache.dpr === dpr
      && cache.waveHeight === waveHeight && cache.accent === accentColor
      && viewStart >= cache.start && needEnd <= cache.start + cache.span) {
    return cache;
  }
  const start = Math.max(0, viewStart - screenSec);
  const span = Math.min(Math.max(0.001, duration - start), screenSec * 3);
  const cw = Math.max(1, Math.ceil(span * pps * dpr));
  const ch = Math.max(1, Math.round(waveHeight * dpr));
  const off = document.createElement("canvas");
  off.width = cw;
  off.height = ch;
  const octx = off.getContext("2d");
  octx.scale(dpr, dpr);

  const mid = waveHeight / 2;
  const grad = octx.createLinearGradient(0, 0, 0, waveHeight);
  grad.addColorStop(0, "rgba(139,124,255,.22)");
  grad.addColorStop(0.5, "rgba(94,162,239,.6)");
  grad.addColorStop(1, "rgba(139,124,255,.22)");
  octx.fillStyle = grad;

  const peaks = state.peaks;
  const last = peaks.length - 1;
  const cssW = cw / dpr;
  const step = 2;
  const xs = [];
  const hs = [];
  for (let x = 0; x <= cssW; x += step) {
    const pos = ((start + x / pps) / duration) * last;
    const i0 = clamp(Math.floor(pos), 0, last);
    const i1 = Math.min(last, i0 + 1);
    const amp = (peaks[i0] || 0) * (1 - (pos - i0)) + (peaks[i1] || 0) * (pos - i0);
    xs.push(x);
    hs.push(Math.max(0.8, amp * 25));
  }
  if (xs.length) {
    octx.beginPath();
    octx.moveTo(xs[0], mid - hs[0]);
    for (let i = 1; i < xs.length; i += 1) octx.lineTo(xs[i], mid - hs[i]);
    for (let i = xs.length - 1; i >= 0; i -= 1) octx.lineTo(xs[i], mid + hs[i]);
    octx.closePath();
    octx.fill();
  }
  state.waveCache = { pps, dpr, waveHeight, accent: accentColor, start, span, canvas: off };
  return state.waveCache;
}

function renderTrack() {
  const track = el("track");
  if (!track || !state.project) return;
  const geom = trackGeom();
  const width = geom.width;
  const duration = state.project.duration || 1;
  const pps = state.pxPerSecond;
  // The playhead position is what the picture follows. While scrubbing it is the
  // coalesced seek target, so the marker tracks the pointer at frame rate even
  // though the <video> only gets a handful of real seeks.
  const playheadTime = state.seekTarget != null && state.scrubbing
    ? state.seekTarget
    : (state.media ? state.media.currentTime : 0);
  // followPlayhead() is the single owner of `viewStart` while follow-play is on;
  // renderTrack only clamps it, so a resize or a stale value can never show a
  // blank strip. Re-deriving it here used to fight followPlayhead ("page" mode
  // was dead code) and made the view creep on every frame.
  state.viewStart = clamp(state.viewStart, 0, Math.max(0, duration - width / pps));
  const viewStart = state.viewStart;

  const dpr = window.devicePixelRatio || 1;
  if (state.laneGutterPx !== geom.gutter) {
    document.documentElement.style.setProperty("--lane-gutter", `${geom.gutter}px`);
    state.laneGutterPx = geom.gutter;
  }
  const vars = cssVars();
  const monoFont = vars.mono;
  const accentColor = vars.accent;
  const gridColor = vars.grid;
  const labelColor = vars.muted;
  renderRuler(el("ruler"), el("ruler").getContext("2d"), width, viewStart, pps, duration, dpr,
              monoFont, accentColor, gridColor, labelColor);
  renderLanes(el("lanes"), viewStart, pps, width);

  // Waveform for the visible window (accent gradient). The canvas is inset by the
  // gutter, so both its bitmap and its time->x mapping must use the content width.
  const canvas = el("waveform");
  const waveHeight = 64;
  const wbw = Math.max(1, Math.round(width * dpr));
  const wbh = Math.round(waveHeight * dpr);
  if (canvas.width !== wbw) canvas.width = wbw;
  if (canvas.height !== wbh) canvas.height = wbh;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, wbw, wbh);
  if (state.peaks.length) {
    // Blit a pre-rendered strip and let it *translate* by whole device pixels.
    // Re-sampling the low-res peak array every frame made the waveform shimmer;
    // a fixed strip can only shift, never change shape.
    const strip = ensureWaveStrip(dpr, pps, viewStart, width, duration, waveHeight, accentColor);
    const srcX = clamp(Math.round((viewStart - strip.start) * pps * dpr),
                       0, Math.max(0, strip.canvas.width - wbw));
    ctx.drawImage(strip.canvas, srcX, 0, wbw, wbh, 0, 0, wbw, wbh);
    ctx.fillStyle = gridColor || "rgba(255,255,255,.08)";
    ctx.fillRect(0, Math.round((waveHeight / 2) * dpr), wbw, Math.max(1, Math.round(dpr)));
  }

  renderBlocks(el("track-blocks"), viewStart, pps, width);

  // Playhead. `left` puts it in the same frame as the ruler/lanes/blocks.
  const playhead = el("playhead");
  if (playhead && state.media) {
    const x = geom.left + (playheadTime - viewStart) * pps;
    // GPU-composited: transform avoids a layout pass every animation frame.
    playhead.style.transform = `translateX(${x - 1}px)`;
    playhead.style.display = x >= geom.left - 2 && x <= geom.left + width + 2 ? "block" : "none";
  }
  const label = el("track-position");
  if (label) {
    label.textContent = `${viewStart.toFixed(2)}s → ${(viewStart + width / pps).toFixed(2)}s`;
    label.title = width / pps >= duration ? "已显示全部时长，滚轮放大后可平移" : "";
  }
  const zoomLabel = el("zoom-label");
  if (zoomLabel) zoomLabel.textContent = `${state.zoom.toFixed(1)}×`;
}

/* Cue structure, filled in once when a clip node is first created. */
function buildBlockNode() {
  const node = document.createElement("div");
  node.className = "track-block";
  const name = document.createElement("div");
  name.className = "tb-name";
  const text = document.createElement("div");
  text.className = "tb-text";
  const translation = document.createElement("div");
  translation.className = "tb-translation";
  const conf = document.createElement("span");
  conf.className = "tb-conf";
  node.append(name, text, translation, conf);
  node._parts = { name, text, translation, conf };
  return node;
}

/* Virtualised subtitle blocks, reusing nodes across frames: during a drag the
   visible window slides but the cue set barely changes, so creating thousands of
   nodes per frame was pure waste (and the main source of the stutter). */
function renderBlocks(container, viewStart, pps, width) {
  const cache = state.blockNodes;
  const windowStart = viewStart - 2, windowEnd = viewStart + width / pps + 2;
  const seen = new Set();
  const fragment = document.createDocumentFragment();
  for (const seg of state.project.segments) {
    if (seg.end < windowStart || seg.start > windowEnd) continue;
    seen.add(seg.id);
    let node = cache.get(seg.id);
    if (!node) {
      node = buildBlockNode();
      node.dataset.id = seg.id;
      cache.set(seg.id, node);
      fragment.appendChild(node);
    }
    const blockWidth = Math.max(34, (seg.end - seg.start) * pps);
    node.style.left = `${(seg.start - viewStart) * pps}px`;
    node.style.width = `${blockWidth}px`;
    const tip = [speakerName(seg), cueText(seg)];
    if (cueTranslation(seg)) tip.push(cueTranslation(seg));
    if (seg.confidence != null && seg.speaker_id !== null) {
      tip.push(`置信度 ${(seg.confidence * 100).toFixed(0)}%`);
    }
    node.title = tip.join("\n");
    const color = speakerColor(seg);
    node.style.setProperty("--cur-color", color);
    node.style.borderLeftColor = color;
    if (seg.speaker_id !== null) {
      node.style.background = hexA(color, 0.18);
      node.style.borderColor = hexA(color, 0.45);
      node.style.borderLeftWidth = "3px";
      node.style.borderLeftColor = color;
    } else {
      node.style.background = "";
      node.style.borderColor = "";
    }
    node.classList.toggle("pending", seg.speaker_id === null);
    node.classList.toggle("reviewed", seg.status === "manual");
    node.classList.toggle("current", seg.id === state.selectedId);
    node.classList.toggle("selected", state.selection.has(seg.id));
    // Detail degrades with the block's width: a 0.8s cue can only carry a name
    // and a truncated line, while a long cue has room for the translation too.
    node.dataset.detail = blockWidth < 90 ? "min" : (blockWidth < 190 ? "mid" : "full");
    const parts = node._parts;
    const name = speakerName(seg);
    if (parts.name.textContent !== name) parts.name.textContent = name;
    parts.name.style.color = color;
    const body = cueText(seg);
    if (parts.text.textContent !== body) parts.text.textContent = body;
    const translationText = blockWidth >= 190 ? cueTranslation(seg) : "";
    if (parts.translation.textContent !== translationText) {
      parts.translation.textContent = translationText;
    }
    parts.translation.hidden = !translationText;
    // Confidence is a corner badge, never a text row competing with the cue.
    const showConf = seg.confidence !== null && seg.confidence !== undefined
      && seg.speaker_id !== null && seg.confidence < 0.9 && blockWidth >= 120;
    if (showConf) {
      const label = `${(seg.confidence * 100).toFixed(0)}%`;
      if (parts.conf.textContent !== label) parts.conf.textContent = label;
    }
    parts.conf.hidden = !showConf;
  }
  if (fragment.childNodes.length) container.appendChild(fragment);
  // Drop nodes that scrolled out of the window so the DOM stays bounded.
  for (const [id, node] of cache) {
    if (!seen.has(id)) { node.remove(); cache.delete(id); }
  }
}

/* Darken (amount < 0) or lighten (amount > 0) a hex colour. */
function shade(hex, amount) {
  const value = (hex || "#888888").replace("#", "");
  const full = value.length === 3 ? value.split("").map((c) => c + c).join("") : value;
  const parts = [0, 2, 4].map((i) => {
    const channel = parseInt(full.slice(i, i + 2), 16) || 0;
    const shifted = amount < 0
      ? channel * (1 + amount)
      : channel + (255 - channel) * amount;
    return Math.max(0, Math.min(255, Math.round(shifted)));
  });
  return `rgb(${parts.join(",")})`;
}

function hexA(hex, alpha) {
  const value = (hex || "#888888").replace("#", "");
  const full = value.length === 3 ? value.split("").map((c) => c + c).join("") : value;
  const r = parseInt(full.slice(0, 2), 16) || 0;
  const g = parseInt(full.slice(2, 4), 16) || 0;
  const b = parseInt(full.slice(4, 6), 16) || 0;
  return `rgba(${r},${g},${b},${alpha})`;
}

function renderCurrent() {
  const seg = segmentById(state.selectedId);
  const textNode = el("current-text");
  if (!textNode) return;
  if (!seg) { textNode.textContent = "—"; return; }
  textNode.textContent = cueText(seg);
  const translationNode = el("current-translation");
  const translation = cueTranslation(seg);
  if (translationNode) {
    translationNode.textContent = translation;
    translationNode.hidden = !translation;
  }
  el("current-range").textContent = `${fmtTime(seg.start, true)} → ${fmtTime(seg.end, true)}`;
  const badge = el("current-badge");
  const role = roleById(seg.speaker_id);
  const statusText = seg.status === "manual" ? "已人工复核"
    : seg.status === "auto" ? `自动归属${seg.confidence != null ? ` · ${(seg.confidence * 100).toFixed(0)}%` : ""}`
      : "待定";
  badge.textContent = `${role ? role.name : "待定"} · ${statusText}`;
  badge.className = "badge " + (seg.status === "manual" ? "manual" : seg.status === "auto" ? "auto" : "");
  // Name the current speaker right here: the review flow is
  // "hear the voice -> recognise who -> name them".
  badge.title = role ? "点击重命名这个说话人" : "";
  badge.style.cursor = role ? "pointer" : "";
  badge.onclick = role ? () => renameRole(role.id) : null;

  // Speaker buttons (pills driven by --role-color). The DOM is built once per
  // role set and only the `.active` class flips afterwards: rebuilding these on
  // every cue change churned nodes and re-bound listeners during playback.
  const host = el("speaker-buttons");
  const roles = (state.project.roles || []).filter((r) => r.type !== "ignored");
  const signature = roles
    .map((r) => `${r.id}\u0001${r.name}\u0001${r.color}\u0001${r.type}`).join("\u0002");
  if (!state.speakerUi || state.speakerUi.host !== host
      || state.speakerUi.signature !== signature) {
    host.innerHTML = "";
    const buttons = new Map();
    roles.forEach((role, index) => {
      const btn = document.createElement("button");
      btn.className = "speaker-btn";
      btn.style.setProperty("--role-color", role.color);
      // The palette is tuned for dark backgrounds; on white an active pill needs
      // a deeper fill so white text keeps enough contrast.
      btn.style.setProperty("--role-color-ink", shade(role.color, -0.34));
      btn.innerHTML = `${escapeHtml(role.name)}<span class="key">${index < 9 ? index + 1 : ""}</span>`;
      btn.title = `${role.name}${role.type === "pending" ? "（待定）" : ""} — 点击归属，双击改名`;
      btn.addEventListener("click", () => onSpeakerClick(role.id));
      btn.addEventListener("dblclick", (event) => { event.preventDefault(); renameRole(role.id); });
      host.appendChild(btn);
      buttons.set(role.id, btn);
    });
    const pendingBtn = document.createElement("button");
    pendingBtn.className = "speaker-btn pending-btn";
    pendingBtn.innerHTML = `待定<span class="key">0</span>`;
    pendingBtn.addEventListener("click", () => onSpeakerClick(null));
    host.appendChild(pendingBtn);

    const addBtn = document.createElement("button");
    addBtn.className = "speaker-btn";
    addBtn.style.setProperty("--role-color", "var(--accent)");
    addBtn.textContent = "+ 新增";
    addBtn.addEventListener("click", () => promptNewRole());
    host.appendChild(addBtn);
    state.speakerUi = { host, signature, buttons, pendingBtn };
  }
  for (const [id, btn] of state.speakerUi.buttons) {
    btn.classList.toggle("active", seg.speaker_id === id);
  }
  state.speakerUi.pendingBtn.classList.toggle("active", seg.speaker_id === null);
}

/* Name a diarised speaker without leaving the review flow. */
function renameRole(roleId) {
  const role = roleById(roleId);
  if (!role) return;
  const cueCount = state.project.segments.filter((s) => s.speaker_id === roleId).length;
  modal("命名说话人 · 该声纹有 " + cueCount + " 条字幕", (body) => {
    body.innerHTML = `
      <p class="hint">给这个声纹起个名字。保存后，它的声纹会自动进入全局角色库，下次检测就能认出这个人。</p>
      <div class="field">
        <label>说话人名称</label>
        <input id="rename-input" type="text" value="${escapeHtml(role.type === "pending" ? "" : role.name)}"
               placeholder="例如：高松灯" autocomplete="off">
      </div>
      <div class="field-row">
        <div class="field"><label>颜色</label>
          <input id="rename-color" type="color" value="${role.color}" style="height:34px;padding:2px">
        </div>
        <div class="field"><label>声纹</label>
          <div class="pill">${role.has_voiceprint ? `已录入（${role.embedder || "builtin"}）` : "尚未录入"}</div>
        </div>
      </div>
    `;
    setTimeout(() => body.querySelector("#rename-input")?.focus(), 30);
  }, [
    { label: "取消" },
    { label: "保存", primary: true, onClick: (close) => applyRename(roleId, close) },
  ]);
}

async function applyRename(roleId, close) {
  const name = (document.getElementById("rename-input").value || "").trim();
  if (!name) { setMessage("请输入名称"); return; }
  const color = document.getElementById("rename-color").value;
  const role = roleById(roleId);
  const cueCount = state.project.segments.filter((s) => s.speaker_id === roleId).length;
  try {
    // The backend registers a named role's voiceprint in the global library
    // automatically, so no separate "入库" step is needed here.
    const data = await API.post(`/api/projects/${state.projectId}/roles_update`,
                                { role_id: roleId, name, color });
    applyProject(data.project);
    await refreshMeta();
    const library = role && role.has_voiceprint ? "，声纹已入库" : "（暂无可用声纹）";
    setMessage(`已命名为「${name}」${library}，影响 ${cueCount} 条字幕`, "ok");
    close();
  } catch (err) {
    setMessage(err.message);
  }
}

function escapeHtml(text) {
  return String(text || "").replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));
}

function renderQueue() {
  const host = el("queue");
  if (!host) return;
  const items = currentSegments();
  if (!items.length) {
    host.innerHTML = "";
    const empty = document.createElement("div");
    empty.className = "queue-empty";
    empty.textContent = "没有符合条件的字幕。";
    host.appendChild(empty);
    return;
  }
  // Rows are keyed by cue id and refilled in place. Rebuilding the whole list on
  // every highlight change is what made selection feel sluggish, and it also
  // threw away the scroll position under the pointer.
  const existing = new Map();
  for (const row of host.querySelectorAll(".queue-item")) existing.set(Number(row.dataset.id), row);
  const reused = new Set();
  const nodes = items.map((seg) => {
    let row = existing.get(seg.id);
    if (!row) row = buildQueueRow(seg);
    else reused.add(seg.id);
    updateQueueRow(row, seg);
    return row;
  });
  const sameSet = reused.size === existing.size && nodes.length === existing.size
    && nodes.every((row, i) => host.children[i] === row);
  if (sameSet) return; // same rows, same order: the in-place update was enough
  const scrollTop = host.scrollTop;
  host.replaceChildren(...nodes);
  host.scrollTop = scrollTop;
}

/* Highlight-only refresh for gestures that change the selection on every frame
   (rubber-band drag), where refilling even the visible rows would be wasteful. */
function syncQueueSelection() {
  const host = el("queue");
  if (!host) return;
  for (const row of host.querySelectorAll(".queue-item")) {
    const id = Number(row.dataset.id);
    row.classList.toggle("current", id === state.selectedId);
    row.classList.toggle("selected", state.selection.has(id));
  }
}

/* Fill a clip-list row from its cue. Called on creation and on every refill, so
   a reused row never shows a stale speaker name or status after an assignment. */
function updateQueueRow(row, seg) {
  const color = speakerColor(seg);
  row.classList.toggle("current", seg.id === state.selectedId);
  row.classList.toggle("selected", state.selection.has(seg.id));
  row._time.textContent = `${fmtTime(seg.start)} → ${fmtTime(seg.end)}`;
  row._speaker.style.color = color;
  row._speaker.innerHTML =
    `<span class="dot" style="background:${color}"></span>${escapeHtml(speakerName(seg))}`;
  row._text.replaceChildren();
  const primaryLine = document.createElement("span");
  primaryLine.className = "q-line";
  primaryLine.textContent = cueText(seg);
  row._text.appendChild(primaryLine);
  const translation = cueTranslation(seg);
  if (translation) {
    const translationLine = document.createElement("span");
    translationLine.className = "q-line translation";
    translationLine.textContent = translation;
    row._text.appendChild(translationLine);
  }
  row._status.textContent = seg.status === "manual" ? "已复核"
    : seg.status === "auto"
      ? (seg.confidence != null && seg.confidence < 0.9 ? `${(seg.confidence * 100).toFixed(0)}%` : "自动")
      : "待定";
}

/* Build one clip-list row, wiring its click once; content is filled by
   updateQueueRow so a refill never re-attaches listeners. */
function buildQueueRow(seg) {
  const row = document.createElement("div");
  row.className = "queue-item";
  row.dataset.id = seg.id;
  const time = document.createElement("div");
  time.className = "q-time";
  const speaker = document.createElement("div");
  speaker.className = "q-speaker";
  const text = document.createElement("div");
  text.className = "q-text";
  const status = document.createElement("div");
  status.className = "q-status";
  row._time = time;
  row._speaker = speaker;
  row._text = text;
  row._status = status;
  row.append(time, speaker, text, status);
  row.addEventListener("click", (event) => {
    const fresh = segmentById(seg.id);
    if (!fresh) return;
    if (event.shiftKey) {
      state.selection.add(fresh.id);
      renderQueue(); renderTrack();
    } else if (event.ctrlKey) {
      state.selection.has(fresh.id) ? state.selection.delete(fresh.id) : state.selection.add(fresh.id);
      renderQueue(); renderTrack();
    } else {
      selectSegment(fresh.id, true);
    }
  });
  return row;
}

/* ------------------------------------------------------------ navigation */

function selectSegment(id, seek) {
  state.selectedId = id;
  if (!arguments[2]) state.selection.clear(); // plain navigation drops box-selection
  const seg = segmentById(id);
  if (seek && seg) {
    if (state.media) state.media.currentTime = seg.start;
    // Following: let follow-play place the view (smooth mode parks the playhead
    // left-of-centre). Free-browsing: centre this cue manually.
    if (state.following) {
      followPlayhead();
    } else {
      const width = trackGeom().width;
      const duration = state.project?.duration || 1;
      state.viewStart = clamp(seg.start - width / 2 / state.pxPerSecond, 0,
        Math.max(0, duration - width / state.pxPerSecond));
    }
  }
  renderCurrent(); renderTrack(); renderQueue(); renderVideoOverlay();
  const row = document.querySelector(".queue-item.current");
  if (row) row.scrollIntoView({ block: "nearest" });
}

function goToSegment(id, offset) {
  const segments = state.project.segments;
  const index = segments.findIndex((s) => s.id === id);
  const next = segments[clamp(index + offset, 0, segments.length - 1)];
  if (next) selectSegment(next.id, true);
}

function nextPending() {
  const segments = state.project.segments;
  const start = segments.findIndex((s) => s.id === state.selectedId);
  const pending = (s) => s.speaker_id === null;
  for (let step = 1; step <= segments.length; step += 1) {
    const seg = segments[(start + step) % segments.length];
    if (pending(seg)) { selectSegment(seg.id, true); return; }
  }
  setMessage("没有待定字幕了 🎉", "ok");
}

let trackRenderQueued = false;

function queueTrackRender() {
  if (trackRenderQueued) return;
  trackRenderQueued = true;
  requestAnimationFrame(() => {
    trackRenderQueued = false;
    if (state.seekTarget != null) applySeekTarget();
    if (state.project) renderTrack();
  });
}

/* Seeking a <video> is expensive, so a scrub drag only records the target here
   and the next animation frame performs at most one real seek. The playhead is
   drawn from `state.seekTarget` meanwhile, so the marker never lags the pointer. */
function applySeekTarget() {
  const target = state.seekTarget;
  state.seekTarget = null;
  const media = state.media;
  if (target == null || !media) return;
  try {
    if (typeof media.fastSeek === "function") media.fastSeek(target);
    else media.currentTime = target;
  } catch (err) {
    media.currentTime = target;
  }
}

/* Continuous playback visuals run on requestAnimationFrame. The <video>
   `timeupdate` event only fires ~4x/s, so driving the playhead / follow-scroll
   from it made the timeline step instead of glide. */
let playbackRaf = null;

function renderPlaybackFrame() {
  const media = state.media;
  if (!media || !state.project) return;
  const total = state.project.duration || media.duration || 0;
  el("time-display").textContent =
    `${fmtTime(media.currentTime, true)} / ${fmtTime(total)}`;
  if (!state.scrubbing) followPlayhead();
  renderVideoOverlay();
  queueTrackRender();
}

function playbackFrame() {
  playbackRaf = null;
  const media = state.media;
  if (!media || !state.project || media.paused || media.ended) return;
  renderPlaybackFrame();
  playbackRaf = requestAnimationFrame(playbackFrame);
}

function ensurePlaybackLoop() {
  if (playbackRaf == null && state.media && !state.media.paused && !state.media.ended) {
    playbackRaf = requestAnimationFrame(playbackFrame);
  }
}

function stopPlaybackLoop() {
  if (playbackRaf != null) {
    cancelAnimationFrame(playbackRaf);
    playbackRaf = null;
  }
}

function onTimeUpdate() {
  const media = state.media;
  if (!media || !state.project) return;

  const seg = segmentAtTime(media.currentTime);
  if (seg && seg.id !== state.selectedId) {
    state.selectedId = seg.id;
    renderCurrent(); syncQueueSelection();
  }
  if (prefs.loop && !state.busy) {
    const active = seg ? seg : state.project.segments.find((s) => s.id === state.selectedId);
    if (active && media.currentTime >= active.end - 0.03) {
      media.currentTime = active.start;
    }
  }
  // Render once here (covers paused seeks) and let the rAF loop own the smooth
  // per-frame updates during playback.
  renderPlaybackFrame();
  ensurePlaybackLoop();
}

/* ------------------------------------------------------------------ track */

/* Premiere-style track interaction.
 *
 *   ruler drag        -> scrub the playhead (the picture follows the playhead)
 *   empty area drag   -> rubber-band select cues
 *   wheel             -> pan horizontally
 *   alt+wheel         -> zoom anchored at the pointer
 *   middle / Alt drag -> grab-pan (never moves the playhead)
 *
 * Panning and zooming never change the playhead, and the view scrolls
 * independently of the video: the picture is a function of the playhead alone.
 */
function bindTrack() {
  const outer = document.querySelector(".track-outer");
  if (!outer) return;

  outer.addEventListener("mousedown", (event) => {
    const onRuler = !!event.target.closest("#ruler");
    const onBlock = event.target.closest(".track-block");
    const wantPan = event.button === 1 || state.tool === "hand";

    if (onRuler || wantPan) {
      const scrub = onRuler && !wantPan;
      state.drag = {
        kind: scrub ? "scrub" : "pan",
        startX: event.clientX,
        startView: state.viewStart,
        moved: false,
        wasPlaying: state.media ? !state.media.paused && !state.media.ended : false,
      };
      if (scrub) {
        // Hold the view still for the whole gesture (PR scrubs in place), and
        // pause playback so the picture reads as a still frame under the cursor.
        state.scrubbing = true;
        if (state.drag.wasPlaying && state.media) state.media.pause();
        seekToPointer(event);
      }
      outer.classList.add(scrub ? "scrubbing" : "panning");
      setMessage(scrub ? "拖动标尺定位播放头" : "");
      event.preventDefault();
      return;
    }

    if (event.button !== 0) return;

    if (onBlock) {
      state.drag = { kind: "block", startX: event.clientX, blockId: Number(onBlock.dataset.id),
                     moved: false, shift: event.shiftKey,
                     ctrl: event.ctrlKey || event.metaKey };
      event.preventDefault();
      return;
    }

    // Empty area: rubber-band selection.
    state.drag = {
      kind: "band",
      startX: event.clientX,
      startTime: pointerTime(event),
      moved: false,
      additive: event.shiftKey || event.ctrlKey,
      baseline: new Set(state.selection),
    };
    if (!state.drag.additive) state.selection.clear();
    outer.classList.add("banding");
    event.preventDefault();
  });

  outer.addEventListener("wheel", (event) => {
    event.preventDefault();
    if (event.altKey) {
      zoomAt(event.clientX, event.deltaY < 0 ? 1.12 : 0.89);
      return;
    }
    const duration = state.project?.duration || 1;
    const width = trackGeom().width;
    const delta = Math.abs(event.deltaX) > Math.abs(event.deltaY) ? event.deltaX : event.deltaY;
    state.viewStart = clamp(state.viewStart + delta / state.pxPerSecond, 0,
                            Math.max(0, duration - width / state.pxPerSecond));
    disableFollowing();
    renderTrack();
  }, { passive: false });

  outer.addEventListener("dblclick", (event) => {
    if (event.target.closest("#ruler")) centerOnPlayhead();
  });
}

/* Zoom by a factor, keeping the time under `clientX` pinned to that pixel. */
function zoomAt(clientX, factor) {
  const outer = document.querySelector(".track-outer");
  if (!outer) return;
  const duration = state.project?.duration || 1;
  const geom = trackGeom();
  const rect = outer.getBoundingClientRect();
  const mouseX = clamp(clientX - rect.left - geom.left, 0, geom.width);
  const atTime = state.viewStart + mouseX / state.pxPerSecond;
  const ratio = clamp(mouseX / Math.max(1, geom.width), 0, 1);
  state.zoom = clamp(state.zoom * factor, minZoomFor(geom.width, duration), ZOOM_MAX);
  state.pxPerSecond = ppsForZoom(state.zoom, geom.width);
  state.viewStart = clamp(atTime - ratio * geom.width / state.pxPerSecond, 0,
                          Math.max(0, duration - geom.width / state.pxPerSecond));
  disableFollowing();
  renderTrack();
}

/* Pointer x -> time. Content time starts at the gutter (`geom.left`), so the
   gutter offset must come out of the pixel distance before it is divided. */
function pointerTime(event) {
  const outer = document.querySelector(".track-outer");
  if (!outer) return 0;
  const rect = outer.getBoundingClientRect();
  const geom = trackGeom();
  const x = clamp(event.clientX - rect.left - geom.left, 0, geom.width);
  return clamp(state.viewStart + x / state.pxPerSecond, 0, state.project?.duration || 0);
}

/* Ruler drag: move the playhead under the pointer, snapped to cue edges. The
   seek itself is coalesced (one per animation frame) — an unthrottled seek per
   mousemove is what made scrubbing feel stuck. */
function seekToPointer(event) {
  if (!state.media) return;
  // PR-style edge auto-scroll: hold the pointer near an edge and the view keeps
  // travelling, so a long recording can be scrubbed without leaving the ruler.
  const outer = document.querySelector(".track-outer");
  if (outer && state.project) {
    const geom = trackGeom();
    const rect = outer.getBoundingClientRect();
    const x = event.clientX - rect.left - geom.left;
    const zone = Math.max(28, geom.width * 0.08);
    const visible = geom.width / state.pxPerSecond;
    const maxStart = Math.max(0, (state.project.duration || 1) - visible);
    const speed = Math.max(0.5, state.pxPerSecond * 0.9); // px/frame -> seconds/frame
    if (x > geom.width - zone) {
      state.viewStart = clamp(state.viewStart + speed / state.pxPerSecond, 0, maxStart);
      disableFollowing();
    } else if (x < zone) {
      state.viewStart = clamp(state.viewStart - speed / state.pxPerSecond, 0, maxStart);
      disableFollowing();
    }
  }
  state.seekTarget = snapTime(pointerTime(event));
  queueTrackRender();
}

/* Snap a time to the closest cue edge when snapping is on. The tolerance is
   ~8px, but capped: at a fitted (very low) zoom 8px would span tens of seconds,
   which would yank the playhead far away from the pointer. */
function snapTime(time) {
  if (!prefs.snap || !state.project) return time;
  const tolerance = Math.min(8 / state.pxPerSecond, 0.75);
  const edges = state.segIndex?.edges;
  if (!edges || !edges.length) return time;
  // Binary search the pre-sorted edge list, then test the two neighbours. This
  // runs on every pointermove while scrubbing, so scanning all cue edges was
  // the scrub's dominant cost on long recordings.
  let lo = 0, hi = edges.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (edges[mid] < time) lo = mid + 1;
    else hi = mid;
  }
  let best = time, bestDist = tolerance;
  for (const index of [lo - 1, lo]) {
    if (index < 0 || index >= edges.length) continue;
    const dist = Math.abs(edges[index] - time);
    if (dist < bestDist) { bestDist = dist; best = edges[index]; }
  }
  return best;
}

function centerOnPlayhead() {
  const outer = document.querySelector(".track-outer");
  if (!outer || !state.media) return;
  const duration = state.project?.duration || 1;
  const width = trackGeom().width;
  state.viewStart = clamp(state.media.currentTime - width / 2 / state.pxPerSecond, 0,
                          Math.max(0, duration - width / state.pxPerSecond));
  renderTrack();
}

/* Auto-scroll while playing (PR-style): the playhead stays parked left-of-centre
   and the ruler slides under it, one continuous motion. Manual panning/zooming
   turns it off. */
function followPlayhead() {
  if (!state.following || !state.media || state.scrubbing || !state.project) return;
  const outer = document.querySelector(".track-outer");
  if (!outer) return;
  const duration = state.project.duration || 1;
  const width = trackGeom().width;
  const visible = width / state.pxPerSecond;
  const t = state.media.currentTime;
  const maxStart = Math.max(0, duration - visible);

  const anchorTime = state.viewStart + visible * 0.35;
  if (t > anchorTime) {
    state.viewStart = clamp(t - visible * 0.35, 0, maxStart);
  } else if (t < state.viewStart) {
    state.viewStart = clamp(t - visible * 0.1, 0, maxStart);
  }
}

/* Manual panning/zooming suspends follow-play; the checkbox reflects it and
   hitting play (or re-checking) brings following back. */
function disableFollowing() {
  state.following = false;
  const box = el("chk-follow");
  if (box) box.checked = false;
}

/* Global listeners: bound exactly once in init(); they survive workspace remounts. */
function bindGlobalListeners() {
  window.addEventListener("mousemove", (event) => {
    const drag = state.drag;
    if (!drag || !state.project) return;
    const outer = document.querySelector(".track-outer");
    if (!outer) return;
    const width = trackGeom().width;
    const duration = state.project.duration || 1;
    const maxStart = Math.max(0, duration - width / state.pxPerSecond);

    if (!drag.moved) {
      if (Math.abs(event.clientX - drag.startX) < 4) return;
      drag.moved = true;
      outer.classList.add("dragging");
    }

    if (drag.kind === "pan") {
      state.viewStart = clamp(drag.startView - (event.clientX - drag.startX) / state.pxPerSecond,
                              0, maxStart);
      disableFollowing();
      queueTrackRender();
      return;
    }

    if (drag.kind === "scrub") {
      // The view is held still for the whole gesture, so this only moves the
      // playhead; queueTrackRender() also flushes the coalesced seek.
      seekToPointer(event);
      return;
    }

    if (drag.kind === "band") {
      const t1 = drag.startTime;
      const t2 = pointerTime(event);
      const lo = Math.min(t1, t2);
      const hi = Math.max(t1, t2);
      const picked = new Set(
        state.project.segments.filter((s) => s.end >= lo && s.start <= hi).map((s) => s.id)
      );
      if (drag.additive && drag.baseline) {
        for (const id of drag.baseline) picked.add(id);
      }
      state.selection = picked;
      queueTrackRender();
      // The queue is a long list; only touch it when the picked set actually moves.
      const stamp = picked.size;
      if (drag.lastStamp !== stamp) {
        drag.lastStamp = stamp;
        syncQueueSelection();
        setMessage(`已选中 ${state.selection.size} 条字幕，点人物按钮批量归属。`);
      }
    }
  });

  window.addEventListener("mouseup", () => {
    const drag = state.drag;
    const outer = document.querySelector(".track-outer");
    if (outer) outer.classList.remove("dragging", "panning", "scrubbing", "banding");
    state.drag = null;
    if (!drag || !state.project) return;

    if (drag.kind === "block" && !drag.moved) {
      if (drag.shift || drag.ctrl) {
        if (state.selection.has(drag.blockId)) state.selection.delete(drag.blockId);
        else state.selection.add(drag.blockId);
        renderTrack();
        syncQueueSelection();
      } else {
        selectSegment(drag.blockId, true);
      }
      return;
    }
    if (drag.kind === "scrub") {
      // Land the coalesced seek before handing the playhead back to the media
      // element, then resume precisely from where the picture stopped.
      if (state.seekTarget != null && state.media) state.media.currentTime = state.seekTarget;
      state.seekTarget = null;
      state.scrubbing = false;
      if (drag.wasPlaying && state.media) state.media.play().catch(() => {});
      setMessage("");
      return;
    }
    if (drag.kind === "band") {
      if (!drag.moved && !drag.additive) state.selection.clear();
      renderTrack();
      syncQueueSelection();
      if (!state.selection.size) setMessage("");
    }
  });

  // Resize fires in bursts; coalesce to one layout + redraw per frame so a
  // window drag does not run fitMediaPane (a forced layout) dozens of times.
  let resizeQueued = false;
  window.addEventListener("resize", () => {
    if (!state.project || resizeQueued) return;
    resizeQueued = true;
    requestAnimationFrame(() => {
      resizeQueued = false;
      if (!state.project) return;
      fitMediaPane(state.media, state.project.media_kind === "audio");
      renderTrack();
    });
  });

  document.addEventListener("keydown", onKeyDown);
}

function bindWorkspace() {
  bindTrack();
  wireMediaResizer();
  applyPrefsToWorkspace();
  el("btn-play").addEventListener("click", () => {
    if (!state.media) return;
    state.media.paused ? state.media.play() : state.media.pause();
  });
  el("btn-zoom-in").addEventListener("click", () => zoomAt(trackCenterX(), ZOOM_IN));
  el("btn-zoom-out").addEventListener("click", () => zoomAt(trackCenterX(), ZOOM_OUT));
  const fitBtn = el("btn-zoom-fit");
  if (fitBtn) fitBtn.addEventListener("click", fitToWidth);
  bindLanguageToggle();
  el("chk-follow").addEventListener("change", (e) => {
    state.following = e.target.checked;
    prefs.follow = e.target.checked; savePrefs();
    renderTrack();
  });
  if (state.media) state.media.playbackRate = prefs.rate;
  // Rubber-band selection is the default gesture on empty track space, so the
  // old modal "框选" toggle is gone. Snap / jump-to-pending / undo / redo have
  // keyboard shortcuts (S / P / Ctrl+Z / Ctrl+Y) and no buttons.

  document.querySelectorAll(".chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      document.querySelectorAll(".chip").forEach((c) => c.classList.remove("active"));
      chip.classList.add("active");
      state.filter = chip.dataset.filter;
      renderQueue();
    });
  });
  // Filtering re-scans every cue and re-builds the list, so debounce keystrokes.
  let searchTimer = null;
  el("search").addEventListener("input", (e) => {
    state.search = e.target.value;
    if (searchTimer) clearTimeout(searchTimer);
    searchTimer = setTimeout(() => { searchTimer = null; renderQueue(); }, 120);
  });

}

/* Fit the whole duration into the visible content width. The resulting zoom may
   sit below the interactive floor on long recordings, so the floor is derived
   from the same fit value (see minZoomFor) and the × label stays honest. */
function fitToWidth() {
  if (!state.project) return;
  applyFitToWidth();
  disableFollowing();
  renderTrack();
}

/* The fit view itself, without the follow-play side effect, so mounting a short
   project can fit it while still honouring the "跟随播放" preference. */
function applyFitToWidth() {
  const width = trackGeom().width;
  const duration = state.project.duration || 1;
  state.pxPerSecond = fitPps(width, duration);
  state.zoom = zoomForPps(state.pxPerSecond, width);
  state.viewStart = Math.max(0, (duration - width / state.pxPerSecond) / 2);
}

function bindLanguageToggle() {
  const toggle = el("lang-toggle");
  if (!toggle) return;
  toggle.hidden = !state.project?.bilingual;
  toggle.querySelectorAll("button").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.lang === prefs.lang);
    btn.addEventListener("click", () => {
      prefs.lang = btn.dataset.lang;
      savePrefs();
      toggle.querySelectorAll("button").forEach((b) => b.classList.toggle("active", b === btn));
      renderCurrent(); renderTrack(); renderQueue(); renderVideoOverlay();
    });
  });
}

function onKeyDown(event) {
  if (el("modal-root").childElementCount > 0) {
    if (event.key === "Escape") {
      const backdrop = el("modal-root").querySelector(".modal-backdrop");
      if (backdrop) backdrop.remove();
    }
    return; // a modal is open
  }
  // An active IME (Chinese/Japanese) uses the spacebar to confirm a candidate;
  // the keydown then carries isComposing / keyCode 229 and must not toggle play.
  if (event.defaultPrevented || event.isComposing || event.keyCode === 229) return;
  const target = event.target || {};
  const tag = (target.tagName || "").toLowerCase();
  if (tag === "input" || tag === "textarea" || tag === "select"
      || target.isContentEditable) return;
  // A focused <video controls> toggles playback itself on space; handling it
  // here too would double-toggle (play then pause -> "space does nothing").
  if (target === state.media) return;
  if (!state.project) return;

  if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "z") {
    event.preventDefault(); undo(); return;
  }
  if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "y") {
    event.preventDefault(); redo(); return;
  }
  if (event.code === "Space") {
    event.preventDefault();
    // Key repeat (holding space) would otherwise flip play/pause many times, and
    // a focused <video> would also toggle natively — preventDefault covers that.
    if (event.repeat) return;
    if (state.media) {
      if (state.media.paused) state.media.play().catch(() => {});
      else state.media.pause();
    }
    return;
  }
  const key = event.key;
  const lower = key.toLowerCase();

  // --- Premiere-style navigation ------------------------------------
  if (key === "ArrowDown" || key === "ArrowUp") {
    // previous / next edit point (cue boundary)
    event.preventDefault();
    goToSegment(state.selectedId, key === "ArrowDown" ? 1 : -1);
    return;
  }
  if (key === "ArrowRight" || key === "ArrowLeft") {
    event.preventDefault();
    nudgePlayhead((key === "ArrowRight" ? 1 : -1) * (event.shiftKey ? 5 : 1));
    return;
  }
  if (key === "Home" || key === "End") {
    event.preventDefault();
    if (state.media) {
      state.media.currentTime = key === "Home" ? 0 : (state.project.duration || 0);
      centerOnPlayhead();
    }
    return;
  }
  if (key === "\\" || key === "|") { event.preventDefault(); fitToWidth(); return; }
  if (key === "-" || key === "_") { event.preventDefault(); zoomAt(trackCenterX(), ZOOM_OUT); return; }
  if (key === "=" || key === "+") { event.preventDefault(); zoomAt(trackCenterX(), ZOOM_IN); return; }
  if (lower === "s") {
    prefs.snap = !prefs.snap; savePrefs();
    setMessage(prefs.snap ? "吸附：开" : "吸附：关");
    return;
  }
  if (lower === "v") { setTool("select"); return; }
  if (lower === "h") { setTool("hand"); return; }

  // --- application shortcuts ---------------------------------------
  if (lower === "p") { nextPending(); return; }
  if (key === "0") { onSpeakerClick(null); return; }
  if (/^[1-9]$/.test(key)) {
    const roles = (state.project.roles || []).filter((r) => r.type !== "ignored");
    const role = roles[Number(key) - 1];
    if (role) onSpeakerClick(role.id);
  }
}

function trackCenterX() {
  const outer = document.querySelector(".track-outer");
  if (!outer) return 0;
  const rect = outer.getBoundingClientRect();
  return rect.left + rect.width / 2;
}

/* Frame-accurate nudge. The video element has no frame step API, so assume
   the project's nominal rate and move by that much. */
function nudgePlayhead(frames) {
  if (!state.media) return;
  const fps = state.project?.fps || 23.976;
  state.media.currentTime = clamp(state.media.currentTime + frames / fps, 0,
                                  state.project.duration || 0);
  followPlayhead();
  renderTrack();
}

function setTool(tool) {
  state.tool = tool;
  const outer = document.querySelector(".track-outer");
  if (outer) outer.classList.toggle("hand-tool", tool === "hand");
}

async function undo() {
  if (!state.undo.length) { setMessage("没有可撤销的操作"); return; }
  const snap = state.undo[state.undo.length - 1];
  try {
    await restore(snap);
    state.undo.pop();
    state.redo.push(snap);
    setMessage("已撤销", "ok");
  } catch (err) {
    setMessage(err.message);
  }
}

async function redo() {
  if (!state.redo.length) { setMessage("没有可重做的操作"); return; }
  const snap = state.redo[state.redo.length - 1];
  try {
    await restore(snap);
    state.redo.pop();
    state.undo.push(snap);
    setMessage("已重做", "ok");
  } catch (err) {
    setMessage(err.message);
  }
}

/* ------------------------------------------------------------ role/admin */

function modal(title, bodyBuilder, footerButtons) {
  const root = el("modal-root");
  const backdrop = document.createElement("div");
  backdrop.className = "modal-backdrop";
  const box = document.createElement("div");
  box.className = "modal";
  const heading = document.createElement("h2");
  heading.textContent = title;
  const body = document.createElement("div");
  body.className = "modal-body";
  const foot = document.createElement("div");
  foot.className = "modal-foot";
  backdrop.append(box);
  box.append(heading, body, foot);
  const close = () => { root.innerHTML = ""; };
  (footerButtons || [{ label: "关闭", onClick: close }]).forEach((spec) => {
    const btn = document.createElement("button");
    btn.textContent = spec.label;
    btn.className = "btn" + (spec.primary ? " primary" : "");
    btn.addEventListener("click", () => spec.onClick ? spec.onClick(close) : close());
    foot.appendChild(btn);
  });
  backdrop.addEventListener("mousedown", (event) => { if (event.target === backdrop) close(); });
  root.appendChild(backdrop);
  bodyBuilder(body, close);
  return close;
}

async function refreshMeta() {
  state.meta = await API.get("/api/state");
  if (el("project-select")) renderProjectSelect();
}

async function refreshProject() {
  const data = await API.get(`/api/projects/${state.projectId}`);
  applyProject(data.project);
}

function promptNewRole() {
  const name = window.prompt("新角色名称：");
  if (!name) return;
  API.post(`/api/projects/${state.projectId}/roles`, { name })
    .then((data) => { applyProject(data.project); setMessage(`已新增角色「${name}」`, "ok"); })
    .catch((err) => setMessage(err.message));
}

function openRoleManager() {
  modal("角色管理", (body, close) => {
    const render = () => {
      const roles = state.project.roles || [];
      const pending = roles.filter((r) => r.type === "pending");
      const targetEngine = state.project.engine || state.meta?.settings?.default_engine || "";
      const targetSpace = state.meta?.engines?.[targetEngine]?.space || "builtin";
      const library = state.meta?.library || [];
      body.innerHTML = `
        <p class="hint">重命名、改色、合并同一人的不同聚类，或把声纹写入全局角色库（跨项目复用）。</p>
        <div class="role-actions">
          <span class="hint">「待定角色」是旧版本检测遗留的聚类占位；命名后入库，或在此一键清理。</span>
          <button id="purge-pending-btn" class="btn ghost sm danger"${pending.length ? "" : " disabled"}>清理全部待定角色${pending.length ? `（${pending.length}）` : ""}</button>
        </div>
        <div class="file-list" id="role-list"></div>
        <div class="field-row">
          <div class="field"><label>新增角色</label><input id="new-role-name" type="text" placeholder="例如：张三"></div>
          <div class="field"><label>&nbsp;</label><button id="new-role-btn">添加</button></div>
        </div>
        <div class="field-row">
          <div class="field"><label>合并：把</label><select id="merge-source">${roles
            .map((r) => `<option value="${r.id}">${escapeHtml(r.name)}</option>`).join("")}</select></div>
          <div class="field"><label>并入</label><select id="merge-target">${roles
            .map((r) => `<option value="${r.id}">${escapeHtml(r.name)}</option>`).join("")}</select></div>
          <div class="field"><label>&nbsp;</label><button id="merge-btn">合并</button></div>
        </div>
        <h2 style="all:unset;font-weight:600;font-size:13px;margin-top:6px">全局角色库</h2>
        <div class="file-list" id="library-list"></div>
      `;
      const list = body.querySelector("#role-list");
      for (const role of roles) {
        const row = document.createElement("div");
        row.className = "role-row";
        const reenroll = role.has_voiceprint && role.embedder && role.embedder !== targetSpace;
        row.innerHTML = `
          <div class="swatch" style="background:${role.color}" title="点击改色"></div>
          <div><input type="text" value="${escapeHtml(role.name)}"><div class="meta">${role.type} · 字幕 ${state.project.segments.filter((s) => s.speaker_id === role.id).length} 条 · 样本 ${role.sample_count}${role.has_voiceprint ? ` · 有声纹（${role.embedder || "builtin"}）` : ""}${reenroll ? ` · <b class="warn-text">需重录</b>` : ""}</div></div>
          <div class="meta">${role.id}</div>
          <button class="btn ghost sm danger del-btn">删除</button>
        `;
        const [swatch, nameInput] = [row.querySelector(".swatch"), row.querySelector("input")];
        swatch.addEventListener("click", () => {
          const next = prompt("颜色（#RRGGBB）：", role.color);
          if (next) API.post(`/api/projects/${state.projectId}/roles_update`, { role_id: role.id, color: next })
            .then((d) => { applyProject(d.project); render(); });
        });
        nameInput.addEventListener("change", () => {
          API.post(`/api/projects/${state.projectId}/roles_update`, { role_id: role.id, name: nameInput.value })
            .then(async (d) => { applyProject(d.project); await refreshMeta(); render(); })
            .catch((err) => setMessage(err.message));
        });
        row.querySelector(".del-btn").addEventListener("click", () => {
          if (!window.confirm(`删除角色「${role.name}」？其字幕将变为待定。`)) return;
          API.del(`/api/projects/${state.projectId}/roles/${role.id}`)
            .then((d) => { applyProject(d.project); render(); });
        });
        list.appendChild(row);
      }
      if (!roles.length) list.innerHTML = '<div class="hint">还没有角色。先在“当前字幕”区点“+ 新增”。</div>';

      body.querySelector("#new-role-btn").addEventListener("click", () => {
        const name = body.querySelector("#new-role-name").value.trim();
        if (!name) return;
        API.post(`/api/projects/${state.projectId}/roles`, { name })
          .then((d) => { applyProject(d.project); render(); });
      });
      body.querySelector("#merge-btn").addEventListener("click", () => {
        const source = Number(body.querySelector("#merge-source").value);
        const target = Number(body.querySelector("#merge-target").value);
        if (source === target) { setMessage("不能合并到自身"); return; }
        API.post(`/api/projects/${state.projectId}/roles_merge`, { source_id: source, target_id: target })
          .then((d) => { applyProject(d.project); render(); setMessage("已合并", "ok"); });
      });

      const purgeBtn = body.querySelector("#purge-pending-btn");
      if (purgeBtn) {
        purgeBtn.addEventListener("click", () => {
          if (!pending.length) return;
          if (!window.confirm(`删除全部 ${pending.length} 个待定角色？其字幕将回到未归属。`)) return;
          API.post(`/api/projects/${state.projectId}/roles_delete_pending`, {})
            .then((d) => {
              applyProject(d.project); render();
              setMessage(`已清理 ${d.removed} 个待定角色`, "ok");
            })
            .catch((err) => setMessage(err.message));
        });
      }

      const libraryList = body.querySelector("#library-list");
      libraryList.innerHTML = library.length ? "" : '<div class="hint">角色库为空。在角色上点“入库”即可注册声纹。</div>';
      for (const entry of library) {
        const row = document.createElement("div");
        row.className = "file-item";
        row.innerHTML = `<span class="dot" style="display:inline-block;width:10px;height:10px;border-radius:50%;background:${entry.color}"></span>
          <strong>${escapeHtml(entry.name)}</strong>
          <span class="pill">${entry.has_voiceprint ? "有声纹" : "无声纹"} · 样本 ${entry.sample_count}</span>
          <span class="spacer"></span><button class="btn ghost sm danger">删除</button>`;
        row.querySelector("button").addEventListener("click", () => {
          API.del(`/api/library/${entry.library_id}`).then(async () => { await refreshMeta(); render(); });
        });
        libraryList.appendChild(row);
      }
    };
    render();
  });
}

/* -------------------------------------------------------------- detection */

function openDetectModal() {
  const engines = state.meta.engines || {};
  const cfg = state.meta.settings || {};
  // Honour the configured default engine; fall back to the first available one.
  let defaultEngine = cfg.default_engine || "campp";
  if (!engines[defaultEngine]?.available) {
    defaultEngine = Object.keys(engines).find((k) => engines[k].available) || "manual";
  }
  const options = Object.entries(engines)
    .map(([key, info]) => `<option value="${key}" ${info.available ? "" : "disabled"} ${key === defaultEngine ? "selected" : ""}>${escapeHtml(info.label)}${info.available ? "" : "（不可用）"}</option>`)
    .join("");
  modal("运行说话人检测", (body) => {
    body.innerHTML = `
      <p class="hint">检测结果会自动与字幕时间轴对齐并按置信度归属；匹配不上你已创建角色的聚类，其字幕会保持「待定」，由你人工归属（宁可漏、不误判）。</p>
      <div class="field"><label>检测引擎</label><select id="d-engine">${options}</select><div class="hint" id="d-engine-note"></div></div>
      <div class="device-row" id="d-device"></div>
      <div class="field-row">
        <div class="field"><label>最少说话人数</label><input id="d-min" type="number" min="1" value="${cfg.min_speakers ?? 1}"></div>
        <div class="field"><label>最多说话人数</label><input id="d-max" type="number" min="1" value="${cfg.max_speakers ?? 6}"></div>
      </div>
      <p class="hint warn" id="d-space-warn" hidden></p>
      <details class="adv">
        <summary>高级</summary>
        <div class="adv-body">
          <label class="check"><input type="checkbox" id="d-sweep"${defaultEngine === "voiceprint-cue" ? " checked" : ""}> 人数自动扫描（在上、下限之间逐个试，取分离度最优）</label>
          <label class="check"><input type="checkbox" id="d-consensus"> 双引擎共识（<b>只增加待定、不纠错</b>；仅两引擎都强时做质检）</label>
          <label class="check"><input type="checkbox" id="d-overwrite"> 覆盖已人工复核的归属</label>
          <p class="hint" id="d-extra-note"></p>
          <p class="hint">声纹匹配阈值默认按引擎自动校准，可在「设置 → 检测」里调整。</p>
        </div>
      </details>
    `;
    const select = body.querySelector("#d-engine");
    const note = body.querySelector("#d-engine-note");
    const updateNote = () => { note.textContent = engines[select.value]?.detail || ""; };
    select.addEventListener("change", updateNote);
    updateNote();

    // Warn before running when enrolled voiceprints live in another feature
    // space than the selected engine: they would silently be skipped.
    const spaceWarn = body.querySelector("#d-space-warn");
    const updateSpaceWarn = () => {
      const space = engines[select.value]?.space;
      const stale = (state.project?.roles || []).filter(
        (r) => r.has_voiceprint && r.embedder && space && r.embedder !== space);
      if (stale.length) {
        spaceWarn.hidden = false;
        spaceWarn.textContent =
          `⚠ ${stale.length} 个角色的声纹在「${stale[0].embedder}」空间，与所选引擎的「${space}」`
          + "空间不同，本次检测不会参与先验匹配——请用当前引擎重新录入这些角色。";
      } else {
        spaceWarn.hidden = true;
      }
    };

    const dev = state.meta.device || {};
    const devNode = body.querySelector("#d-device");
    const gpuEngines = new Set(["campp", "pyannote", "voiceprint-cue", "sortformer"]);
    const renderDevice = () => {
      const usesGpu = gpuEngines.has(select.value);
      if (!usesGpu) {
        devNode.className = "device-row";
        devNode.innerHTML = "<span class=\"pill\">该引擎为内置算法，使用 CPU</span>";
      } else if (dev.cuda) {
        devNode.className = "device-row ok";
        devNode.innerHTML = `<span class="card"><b>GPU 加速已启用</b><span class="hint">${escapeHtml(dev.name || "CUDA")}${dev.vram_gb ? " · " + dev.vram_gb + "GB" : ""} · torch ${escapeHtml(dev.torch || "")}</span></span>`;
      } else {
        devNode.className = "device-row warn";
        devNode.innerHTML = `<span class="card"><b>当前在 CPU 上运行</b><span class="hint">未检测到可用的 CUDA（装了 CPU 版 torch？）。长视频会明显更慢。</span></span>`;
      }
      devNode.hidden = false;
    };
    select.addEventListener("change", renderDevice);
    renderDevice();

    const extra = body.querySelector("#d-extra-note");
    const updateExtra = () => {
      const parts = [];
      if (body.querySelector("#d-sweep").checked) {
        parts.push(select.value === "voiceprint-cue"
          ? "人数扫描复用逐句声纹、只重聚类，耗时≈单次。"
          : "人数扫描会按每个候选人数各跑一次（耗时≈单次 × 候选个数）。");
      }
      if (body.querySelector("#d-consensus").checked) {
        parts.push("共识只把两引擎不一致的句子标为待定（不纠错），耗时≈两次。");
      }
      extra.textContent = parts.join(" ");
    };
    // Sweep defaults on only for the voiceprint-cue engine (its clustering is the
    // weak link and a sweep is nearly free); follow the engine picker until the
    // user touches the checkbox themselves.
    const sweepBox = body.querySelector("#d-sweep");
    let sweepTouched = false;
    sweepBox.addEventListener("change", () => { sweepTouched = true; updateExtra(); });
    body.querySelector("#d-consensus").addEventListener("change", updateExtra);
    select.addEventListener("change", () => {
      if (!sweepTouched) sweepBox.checked = select.value === "voiceprint-cue";
      updateExtra();
      updateSpaceWarn();
    });
    updateExtra();
    updateSpaceWarn();
  }, [
    { label: "取消" },
    {
      label: "开始检测", primary: true, onClick: async (close) => {
        const payload = {
          engine: document.getElementById("d-engine").value,
          min_speakers: Number(document.getElementById("d-min").value),
          max_speakers: Number(document.getElementById("d-max").value),
          overwrite_manual: document.getElementById("d-overwrite").checked,
          sweep: document.getElementById("d-sweep").checked,
          consensus: document.getElementById("d-consensus").checked,
        };
        try {
          setBusy("检测中，请稍候…（长视频首次会先解码音频）");
          const data = await API.post(`/api/projects/${state.projectId}/detect`, payload);
          applyProject(data.project);
          setMessage("检测完成：" + (data.project.detection_notes || []).join(" "), "ok");
          close();
        } catch (err) {
          setMessage(err.message);
        } finally {
          setBusy("");
        }
      },
    },
  ]);
}

/* ------------------------------------------------------ 拖放导入 */

function classifyPaths(paths) {
  const media = [], subs = [], other = [];
  for (const path of paths) {
    // classify client-side by extension (mirrors backend rules)
    const ext = (path.split(".").pop() || "").toLowerCase();
    const mediaExt = ["mp4","mkv","mov","avi","webm","flv","ts","m4v","wmv","wav","mp3","m4a","aac","flac","ogg","opus","wma"];
    const subExt = ["srt","vtt","ass","ssa"];
    if (mediaExt.includes(ext)) media.push(path);
    else if (subExt.includes(ext)) subs.push(path);
    else other.push(path);
  }
  return { media, subs, other };
}

async function uploadDroppedFile(file) {
  if (file.size > 512 * 1024 * 1024) {
    throw new Error(`「${file.name}」超过 512MB，浏览器上传受限——请使用桌面版拖放（免上传）或直接填路径`);
  }
  setBusy(`正在上传 ${file.name}…`);
  try {
    const res = await fetch(`/api/upload?name=${encodeURIComponent(file.name)}`, {
      method: "PUT",
      body: file,
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok || data.ok === false) throw new Error(data.error || `上传失败: ${file.name}`);
    return data.path;
  } finally {
    setBusy("");
  }
}

function showDropOverlay(text) {
  let node = document.getElementById("drop-overlay");
  if (!node) {
    node = document.createElement("div");
    node.id = "drop-overlay";
    document.body.appendChild(node);
  }
  node.textContent = text || "松开导入 · 视频/音频 + 字幕（可带译文）";
  node.classList.add("visible");
}

function hideDropOverlay() {
  document.getElementById("drop-overlay")?.classList.remove("visible");
}

async function handleDroppedFiles(names) {
  try {
    const { media, subs, other } = classifyPaths(names);
    if (other.length) setMessage(`已忽略不认识的文件：${other.map((n) => n.split(/[\\/]/).pop()).join("、")}`);

    // 视频/音频 + 字幕 → 直接新建项目（无论当前是否已打开项目）
    if (media.length && subs.length) {
      setBusy("正在导入…");
      const data = await API.post("/api/projects", {
        media_path: media[0],
        subtitle_path: subs[0],
        second_subtitle_path: subs[1] || null,
        line_mode: "auto",
      });
      await refreshMeta();
      applyProject(data.project);
      setMessage(`已导入「${data.project.name}」，共 ${data.project.segments.length} 条字幕。`, "ok");
      return;
    }

    // 只拖字幕且已打开项目 → 作为译文挂载
    if (state.project && !media.length && subs.length) {
      setBusy("正在挂载翻译字幕…");
      const data = await API.post(`/api/projects/${state.projectId}/translation`, { path: subs[0] });
      applyProject(data.project);
      setMessage(`已挂载译文「${subs[0].split(/[\\/]/).pop()}」，匹配 ${data.project.segments.filter(s => s.translation).length} 条`, "ok");
      return;
    }

    // Otherwise prefill the import modal and let the user confirm.
    openImportModal();
    if (media.length) el("imp-media").value = media[0];
    if (subs.length) el("imp-sub").value = subs[0];
    if (subs.length > 1) el("imp-sub2").value = subs[1];
    setMessage("已填入拖入的文件路径，确认后点「导入」。");
  } catch (err) {
    setMessage(err.message);
  } finally {
    setBusy("");
  }
}

function wireDragDrop() {
  // Browser: HTML5 drag-drop. Dropped files are uploaded to the local backend.
  window.addEventListener("dragover", (event) => {
    event.preventDefault();
    showDropOverlay();
  });
  window.addEventListener("dragenter", (event) => {
    event.preventDefault();
    showDropOverlay();
  });
  window.addEventListener("dragleave", (event) => {
    if (!event.relatedTarget) hideDropOverlay();
  });
  window.addEventListener("drop", async (event) => {
    event.preventDefault();
    hideDropOverlay();
    // 桌面版（pywebview）：由 Python 侧的 DOM 事件提供真实路径，这里不重复处理。
    if (window.pywebview) return;
    const files = [...(event.dataTransfer?.files || [])];
    if (!files.length) return;
    try {
      const paths = [];
      for (const file of files) paths.push(await uploadDroppedFile(file));
      await handleDroppedFiles(paths);
    } catch (err) {
      setMessage(err.message);
    }
  });
}

/* --------------------------------------------------------- 设置中心 */

async function saveServerSetting(patch) {
  try {
    const data = await API.post("/api/settings", patch);
    if (state.meta) state.meta.settings = data.settings;
    setMessage("设置已保存", "ok");
    return data.settings;
  } catch (err) {
    setMessage(err.message);
    return null;
  }
}

function openSettings() {
  const cfg = state.meta?.settings || {};
  const engines = state.meta?.engines || {};
  const asr = state.meta?.asr || {};
  const sections = [
    { id: "appearance", label: "外观" },
    { id: "playback", label: "播放与复核" },
    { id: "detect", label: "检测" },
    { id: "asr", label: "转录 ASR" },
    { id: "advanced", label: "高级" },
  ];

  const root = el("modal-root");
  const backdrop = document.createElement("div");
  backdrop.className = "modal-backdrop";
  backdrop.innerHTML = `
    <div class="modal modal-wide settings">
      <h2>设置</h2>
      <div class="settings-body">
        <aside class="set-nav">${sections.map((s, i) =>
          `<button class="set-item${i === 0 ? " active" : ""}" data-sec="${s.id}">${s.label}</button>`).join("")}
        </aside>
        <div class="set-content"></div>
      </div>
    </div>`;
  root.appendChild(backdrop);
  const content = backdrop.querySelector(".set-content");
  const close = () => { root.innerHTML = ""; };
  backdrop.addEventListener("mousedown", (e) => { if (e.target === backdrop) close(); });

  const row = (label, hint, control) => `
    <div class="set-row">
      <div class="set-label"><b>${label}</b>${hint ? `<span>${hint}</span>` : ""}</div>
      <div class="set-control">${control}</div>
    </div>`;
  const group = (title, inner) => `<div class="set-group"><div class="set-group-title">${title}</div><div class="set-group-box">${inner}</div></div>`;
  const sw = (id, checked) => `<label class="toggle"><input type="checkbox" id="${id}" ${checked ? "checked" : ""}><i></i></label>`;

  const renderers = {
    appearance() {
      content.innerHTML = `
        ${group("主题", row("外观", "工具栏右上角可一键切换深色 / 浅色", `
          <div class="segmented" id="seg-theme">
            ${[["dark", "深色"], ["light", "浅色"], ["auto", "跟随系统"]].map(([v, t]) =>
              `<button data-v="${v}" class="${prefs.theme === v ? "active" : ""}">${t}</button>`).join("")}
          </div>`))}
        ${group("强调色", row("主色调", "按钮、播放头与选中态", `
          <div class="accent-dots" id="accent-dots">
            ${[["violet", "#8b7cff"], ["blue", "#4da3ff"], ["rose", "#ff6b8a"], ["green", "#2ec27e"]].map(([v, c]) =>
              `<button data-v="${v}" style="--c:${c}" class="${prefs.accent === v ? "active" : ""}"></button>`).join("")}
          </div>`))}
        ${group("辅助",
          row("显示快捷键角标", "说话人按钮上的 1-9 / 0", sw("set-kbd", prefs.showKbd)) +
          row("视频内嵌字幕", "把当前字幕叠在画面上", sw("set-overlay", prefs.overlay !== false)))}
      `;
      content.querySelector("#seg-theme").addEventListener("click", (e) => {
        const btn = e.target.closest("button[data-v]"); if (!btn) return;
        prefs.theme = btn.dataset.v; savePrefs(); applyTheme();
        content.querySelectorAll("#seg-theme button").forEach((b) => b.classList.toggle("active", b === btn));
      });
      content.querySelector("#accent-dots").addEventListener("click", (e) => {
        const btn = e.target.closest("button[data-v]"); if (!btn) return;
        prefs.accent = btn.dataset.v; savePrefs(); applyTheme();
        content.querySelectorAll("#accent-dots button").forEach((b) => b.classList.toggle("active", b === btn));
      });
      content.querySelector("#set-kbd").addEventListener("change", (e) => {
        prefs.showKbd = e.target.checked; savePrefs(); applyPrefsToWorkspace();
      });
      content.querySelector("#set-overlay").addEventListener("change", (e) => {
        prefs.overlay = e.target.checked; savePrefs(); renderVideoOverlay();
      });
    },

    playback() {
      content.innerHTML = `
        ${group("播放", 
          row("播放速度", "", `<div class="select-wrap"><select id="set-rate">${[0.75, 1, 1.25, 1.5, 2].map(r =>
            `<option value="${r}" ${prefs.rate === r ? "selected" : ""}>${r}×</option>`).join("")}</select></div>`) +
          row("单句循环", "播到当前句结尾自动回退", sw("set-loop", prefs.loop)))}
        ${group("复核",
          row("归属后跳下一句", "连续归属更顺手", sw("set-autonext", prefs.autonext)) +
          row("播放条跟随播放", "播放时标尺平滑跟随；关闭后可手动定位", sw("set-follow", prefs.follow)) +
          row("默认缩放", "播放条时间密度", `<input type="range" id="set-zoom" min="0.25" max="4" step="0.05" value="${prefs.zoom}">`))}
      `;
      content.querySelector("#set-rate").addEventListener("change", (e) => {
        prefs.rate = Number(e.target.value); savePrefs();
        if (state.media) state.media.playbackRate = prefs.rate;
      });
      content.querySelector("#set-loop").addEventListener("change", (e) => {
        prefs.loop = e.target.checked; savePrefs();
        const c = el("chk-loop"); if (c) c.checked = prefs.loop;
      });
      content.querySelector("#set-autonext").addEventListener("change", (e) => {
        prefs.autonext = e.target.checked; savePrefs();
        const c = el("chk-autonext"); if (c) c.checked = prefs.autonext;
      });
      content.querySelector("#set-follow").addEventListener("change", (e) => {
        prefs.follow = e.target.checked; savePrefs(); applyPrefsToWorkspace();
        const c = el("chk-follow"); if (c) c.checked = prefs.follow;
      });
      content.querySelector("#set-zoom").addEventListener("input", (e) => {
        prefs.zoom = Number(e.target.value); savePrefs(); applyDefaultZoom();
      });
    },

    detect() {
      const engineOptions = Object.entries(engines).map(([key, info]) =>
        `<option value="${key}" ${info.available ? "" : "disabled"} ${cfg.default_engine === key ? "selected" : ""}>${escapeHtml(info.label)}${info.available ? "" : "（不可用）"}</option>`).join("");
      content.innerHTML = `
        ${group("引擎",
          row("默认检测引擎", "检测弹窗将预选此项", `<div class="select-wrap"><select id="set-engine">${engineOptions}</select></div>`) +
          row("最少 / 最多说话人数", "", `
            <div class="range-inputs">
              <input type="number" id="set-min" min="1" value="${cfg.min_speakers ?? 1}" style="width:70px">
              <span class="pill">至</span>
              <input type="number" id="set-max" min="1" value="${cfg.max_speakers ?? 6}" style="width:70px">
            </div>`))}
        ${group("声纹先验",
          row("匹配阈值", "自动 = 按引擎特征空间校准（pyannote≈0.66 / CAM++≈0.76 / 内置≈0.84）；匹配不上的聚类保持待定", `
            <div class="range-inputs">
              <label class="check inline"><input type="checkbox" id="set-threshold-auto" ${cfg.threshold == null ? "checked" : ""}> 自动</label>
              <input type="range" id="set-threshold" min="0.5" max="1" step="0.01" value="${cfg.threshold ?? 0.8}" ${cfg.threshold == null ? "disabled" : ""}>
              <b class="mono" id="set-threshold-val">${(cfg.threshold ?? 0.8).toFixed(2)}</b>
            </div>`))}
        <p class="hint">声纹只在同一特征空间内匹配；切换引擎后需用当前引擎重新录入声纹。</p>
      `;
      content.querySelector("#set-engine").addEventListener("change", (e) => saveServerSetting({ default_engine: e.target.value }));
      const saveCount = () => saveServerSetting({
        min_speakers: Number(content.querySelector("#set-min").value) || 1,
        max_speakers: Number(content.querySelector("#set-max").value) || 6,
      });
      content.querySelector("#set-min").addEventListener("change", saveCount);
      content.querySelector("#set-max").addEventListener("change", saveCount);
      const threshold = content.querySelector("#set-threshold");
      const thresholdAuto = content.querySelector("#set-threshold-auto");
      threshold.addEventListener("input", () => {
        content.querySelector("#set-threshold-val").textContent = Number(threshold.value).toFixed(2);
      });
      threshold.addEventListener("change", () => saveServerSetting({ threshold: Number(threshold.value) }));
      thresholdAuto.addEventListener("change", () => {
        threshold.disabled = thresholdAuto.checked;
        saveServerSetting({ threshold: thresholdAuto.checked ? null : Number(threshold.value) });
      });
    },

    asr() {
      const models = asr.models || [];
      content.innerHTML = `
        ${group("转录",
          row("默认模型", "越大越准越慢", models.length
            ? `<div class="select-wrap"><select id="set-asr-model">${models.map((m) =>
                `<option value="${m}" ${cfg.asr_model === m ? "selected" : ""}>${m}</option>`).join("")}</select></div>`
            : `<span class="pill">faster-whisper 未安装</span>`) +
          row("默认语言", "留空自动检测", `<input type="text" id="set-asr-lang" value="${escapeHtml(cfg.asr_language || "")}" placeholder="zh" style="width:110px">`))}
        ${asr.available ? "" : `<p class="hint">${escapeHtml(asr.detail || "安装 faster-whisper 后可用。")}</p>`}
      `;
      const modelSel = content.querySelector("#set-asr-model");
      if (modelSel) modelSel.addEventListener("change", (e) => saveServerSetting({ asr_model: e.target.value }));
      content.querySelector("#set-asr-lang").addEventListener("change", (e) => saveServerSetting({ asr_language: e.target.value.trim() }));
    },

    advanced() {
      const engineRows = Object.entries(engines).map(([key, info]) =>
        `<div class="set-row"><div class="set-label"><b>${escapeHtml(info.label)}</b><span>${escapeHtml(info.available ? "可用" : "不可用 · 需要安装依赖")}</span></div><span class="status-dot ${info.available ? "ok" : "off"}"></span></div>`).join("");
      const dev = state.meta?.device || {};
      content.innerHTML = `
        ${group("计算设备", `
          <div class="set-row"><div class="set-label"><b>${dev.cuda ? "GPU 加速" : "CPU 模式"}</b>
            <span>${dev.cuda ? escapeHtml((dev.name || "CUDA") + (dev.vram_gb ? " · " + dev.vram_gb + "GB 显存" : "")) : "未启用 CUDA：神经引擎（CAM++ / pyannote）将用 CPU，长视频较慢"}</span></div>
            <span class="status-dot ${dev.cuda ? "ok" : "off"}"></span></div>
          <div class="set-row"><div class="set-label"><b>torch</b><span class="mono">${escapeHtml(dev.torch || "未安装")}</span></div></div>`)}
        ${group("HuggingFace Token", `
          ${row("Token 状态", "pyannote 需要模型使用授权", `<span class="pill">${cfg.hf_token_set ? "已配置 ✓" : "未配置"}</span>`)}
          <div class="set-row"><div class="set-label"><b>写入 Token</b><span>保存到本地 data/hf_token.txt（仅本机）</span></div>
            <div class="range-inputs"><input type="password" id="set-token" placeholder="hf_..." style="width:190px">
            <button class="btn sm" id="set-token-save">保存</button></div></div>`)}
        ${group("引擎可用性", engineRows)}
        ${group("维护",
          row("数据目录", `<span class="mono">${escapeHtml(cfg.data_dir || "")}</span>`, "") +
          row("临时文件", "清理解码缓存 WAV 与波形缓存", `<button class="btn sm" id="set-clean">清理</button>`))}
        <p class="hint">ffmpeg：${state.meta?.ffmpeg ? "已找到 ✓" : "未找到 ✗（影响解码与导出）"}</p>
      `;
      content.querySelector("#set-token-save").addEventListener("click", async () => {
        const value = content.querySelector("#set-token").value.trim();
        if (!value) { setMessage("请先粘贴 Token"); return; }
        const result = await saveServerSetting({ hf_token: value });
        if (result) setMessage("Token 已保存到本地文件", "ok");
      });
      content.querySelector("#set-clean").addEventListener("click", async () => {
        try {
          const data = await API.post("/api/maintenance", { action: "clean_work" });
          setMessage(`已清理 ${data.removed} 个临时文件`, "ok");
        } catch (err) { setMessage(err.message); }
      });
    },
  };

  const show = (secId) => {
    content.classList.add("fade");
    (renderers[secId] || renderers.appearance)();
    setTimeout(() => content.classList.remove("fade"), 180);
  };
  backdrop.querySelectorAll(".set-item").forEach((btn) => {
    btn.addEventListener("click", () => {
      backdrop.querySelectorAll(".set-item").forEach((b) => b.classList.toggle("active", b === btn));
      show(btn.dataset.sec);
    });
  });
  show("appearance");
}

/* --------------------------------------------------------- 设置中心 end */

function openAsrModal() {
  const asr = state.meta.asr || { available: false, detail: "" };
  const cfg = state.meta.settings || {};
  modal("ASR 转录生成字幕（可选）", (body) => {
    const models = asr.models || [];
    body.innerHTML = `
      <p class="hint">${escapeHtml(asr.detail || "")}</p>
      ${asr.available ? `
      <div class="field"><label>模型</label><select id="asr-model">${models
        .map((m) => `<option value="${m}" ${m === (cfg.asr_model || "small") ? "selected" : ""}>${m}</option>`).join("")}</select></div>
      <div class="field"><label>语言（留空自动检测，中文填 zh；默认可在设置里改）</label><input id="asr-lang" type="text" placeholder="${escapeHtml(cfg.asr_language || "zh")}" value="${escapeHtml(cfg.asr_language || "")}"></div>
      <p class="hint">转录完成后会生成一个 SRT 文件，路径自动填回导入表单。</p>` : ""}
    `;
  }, [
    { label: "关闭" },
    ...(asr.available ? [{
      label: "开始转录", primary: true, onClick: async (close) => {
        const mediaPath = document.getElementById("imp-media").value.trim();
        if (!mediaPath) { setMessage("请先在上面的导入表单里填写媒体路径"); return; }
        try {
          setBusy("转录中，首次会下载模型，请耐心等待…");
          const data = await API.post("/api/transcribe", {
            media_path: mediaPath,
            model_size: document.getElementById("asr-model").value,
            language: document.getElementById("asr-lang").value.trim() || null,
          });
          document.getElementById("imp-sub").value = data.subtitle_path;
          setMessage(`转录完成：${data.count} 条字幕（语言 ${data.language}），字幕路径已填入。`, "ok");
          close();
        } catch (err) {
          setMessage(err.message);
        } finally {
          setBusy("");
        }
      },
    }] : []),
  ]);
}

/* ----------------------------------------------------------------- export */

function openExportModal() {
  const formats = state.meta.formats || [];
  modal("导出 / 数据集", (body) => {
    body.innerHTML = `
      <div class="field"><label>导出格式</label>
        <div class="grid-checks" id="fmt-checks">${formats.map((f) => `
          <label class="check"><input type="checkbox" value="${f.id}" ${["srt", "ass", "jsonl", "dataset"].includes(f.id) ? "checked" : ""}>${escapeHtml(f.label)}</label>
        `).join("")}</div>
      </div>
      <div class="field"><label>导出内容</label>
        <div class="grid-checks">
          <label class="check"><input type="checkbox" id="opt-pending" checked> 包含待定字幕</label>
          <label class="check"><input type="checkbox" id="opt-assigned-only"> 仅导出已归属</label>
          <label class="check"><input type="checkbox" id="opt-confidence" checked> 包含置信度</label>
          <label class="check"><input type="checkbox" id="opt-status" checked> 包含复核状态</label>
          <label class="check"><input type="checkbox" id="opt-rolelib" checked> 数据集含角色库</label>
          <label class="check"><input type="checkbox" id="opt-voiceprints"> 数据集含声纹向量</label>
        </div>
      </div>
      <div class="field"><label>文件名前缀</label><input id="opt-stem" type="text" value="${escapeHtml(state.project.name)}"></div>
      ${state.project.bilingual ? `
      <div class="field"><label>语言输出</label>
        <div class="segmented" id="opt-lang">
          <button data-v="primary">仅原文</button>
          <button data-v="both" class="active">原文 + 译文</button>
          <button data-v="translation">仅译文</button>
        </div>
        <div class="hint">字幕：译文缩进在原文下方（ASS 用独立小字号样式）；数据集：同时输出 text / translation 字段。</div>
      </div>` : ""}
      <div id="export-result"></div>
    `;
    const langSeg = body.querySelector("#opt-lang");
    if (langSeg) {
      langSeg.querySelectorAll("button").forEach((btn) => {
        btn.classList.toggle("active", btn.dataset.v === prefs.lang);
        btn.addEventListener("click", () => {
          langSeg.querySelectorAll("button").forEach((b) => b.classList.toggle("active", b === btn));
        });
      });
    }
  }, [
    { label: "关闭" },
    {
      label: "导出", primary: true, onClick: async (close) => {
        const selected = [...document.querySelectorAll("#fmt-checks input:checked")].map((i) => i.value);
        if (!selected.length) { setMessage("请至少选择一种导出格式"); return; }
        const options = {
          include_pending: document.getElementById("opt-pending").checked,
          only_assigned: document.getElementById("opt-assigned-only").checked,
          include_confidence: document.getElementById("opt-confidence").checked,
          include_status: document.getElementById("opt-status").checked,
          include_role_library: document.getElementById("opt-rolelib").checked,
          include_voiceprints: document.getElementById("opt-voiceprints").checked,
          filename_stem: document.getElementById("opt-stem").value || state.project.name,
          text_field: (document.querySelector("#opt-lang button.active") || {}).dataset?.v || "both",
        };
        const target = document.getElementById("export-result");
        try {
          setBusy("导出中…");
          const data = await API.post(`/api/projects/${state.projectId}/export`, { formats: selected, options });
          applyProject(data.project);
          target.innerHTML = `<div class="field"><label>已生成 ${data.files.length} 个文件</label>
            <div class="file-list">${data.files.map((f) => `
              <div class="file-item"><a href="/api/projects/${state.projectId}/export/${encodeURIComponent(f.filename)}" download>${escapeHtml(f.filename)}</a>
              <span class="pill">${f.label} · ${(f.bytes / 1024).toFixed(1)} KB</span></div>`).join("")}</div></div>`;
          setMessage(`导出完成：${data.files.length} 个文件`, "ok");
        } catch (err) {
          setMessage(err.message);
        } finally {
          setBusy("");
        }
      },
    },
  ]);
}

/* ----------------------------------------------------------------- import */

function openImportModal() {
  modal("导入视频 + 字幕", (body) => {
    body.innerHTML = `
      <p class="hint">填写本机绝对路径（Windows 例：<code>C:\\videos\\a.mp4</code>，Git Bash 例：<code>/c/videos/a.mp4</code>）。字幕支持 SRT / VTT / ASS。</p>
      <div class="field"><label>视频或音频文件</label>
        <div class="path-input">
          <input id="imp-media" type="text" placeholder="C:\\videos\\interview.mp4">
          <button class="btn ghost small" data-pick="media" data-target="imp-media" hidden>浏览…</button>
        </div>
      </div>
      <div class="field"><label>字幕文件（原文）</label>
        <div class="path-input">
          <input id="imp-sub" type="text" placeholder="C:\\videos\\interview.srt">
          <button class="btn ghost small" data-pick="subtitle" data-target="imp-sub" hidden>浏览…</button>
        </div>
      </div>
      <div class="field"><label>翻译字幕文件（可选，用于双语）</label>
        <div class="path-input">
          <input id="imp-sub2" type="text" placeholder="C:\\videos\\interview.en.srt — 留空则自动识别单文件双语">
          <button class="btn ghost small" data-pick="subtitle" data-target="imp-sub2" hidden>浏览…</button>
        </div>
      </div>
      <div class="field"><label>多行字幕处理</label>
        <select id="imp-linemode">
          <option value="auto" selected>自动识别（推荐）</option>
          <option value="interleaved">交错双语：日/中轨交错，按时间配对</option>
          <option value="bilingual">双语：第 1 行原文，其余为译文</option>
          <option value="join">硬换行：合并为一句</option>
          <option value="keep">保持原样</option>
        </select>
      </div>
      <div class="field"><label>项目名称（可选）</label><input id="imp-name" type="text" placeholder="留空则用文件名"></div>
      <div class="assign-row" style="border-bottom:none;padding:4px 0 0">
        <span class="assign-label">没有字幕？</span>
        <button id="imp-asr" class="ghost small">用 ASR 转录生成（可选）</button>
      </div>
    `;
    body.querySelector("#imp-asr").addEventListener("click", openAsrModal);
    // 桌面版（pywebview）提供系统文件选择框，可直接选路径；浏览器版不显示浏览按钮。
    if (window.pywebview?.api?.pick_file) {
      body.querySelectorAll("[data-pick]").forEach((btn) => {
        btn.hidden = false;
        btn.addEventListener("click", async () => {
          try {
            const path = await window.pywebview.api.pick_file(btn.dataset.pick);
            if (path) document.getElementById(btn.dataset.target).value = path;
          } catch (err) {
            setMessage(String(err));
          }
        });
      });
    }
  }, [
    { label: "取消" },
    {
      label: "导入", primary: true, onClick: async (close) => {
        const payload = {
          media_path: document.getElementById("imp-media").value.trim().replace(/^"|"$/g, ""),
          subtitle_path: document.getElementById("imp-sub").value.trim().replace(/^"|"$/g, ""),
          second_subtitle_path: document.getElementById("imp-sub2").value.trim().replace(/^"|"$/g, "") || null,
          line_mode: document.getElementById("imp-linemode").value,
          name: document.getElementById("imp-name").value.trim() || null,
        };
        if (!payload.media_path || !payload.subtitle_path) { setMessage("请填写媒体与字幕路径"); return; }
        try {
          setBusy("正在导入…");
          const data = await API.post("/api/projects", payload);
          await refreshMeta();
          applyProject(data.project);
          close();
          const notes = (data.project.import_notes || []).join(" ");
          setMessage(`已导入「${data.project.name}」，共 ${data.project.segments.length} 条字幕。${notes}`, "ok");
        } catch (err) {
          setMessage(err.message);
        } finally {
          setBusy("");
        }
      },
    },
  ]);
}

function openProjectList() {
  modal("打开项目", (body, close) => {
    const projects = state.meta.projects || [];
    body.innerHTML = projects.length
      ? `<div class="file-list">${projects.map((p) => `
        <div class="file-item"><a href="#" data-id="${p.id}">${escapeHtml(p.name)}</a>
        <span class="pill">${p.segments} 条 · 已归属 ${p.stats.assigned} · 更新 ${p.updated || ""}</span>
        <span class="spacer"></span><button class="ghost small" data-del="${p.id}">删除</button></div>`).join("")}</div>`
      : '<div class="hint">还没有项目，先导入一个视频和字幕。</div>';
    body.querySelectorAll("a[data-id]").forEach((link) => {
      link.addEventListener("click", async (event) => {
        event.preventDefault();
        const data = await API.get(`/api/projects/${link.dataset.id}`);
        close();
        state.selection.clear();
        applyProject(data.project);
      });
    });
    body.querySelectorAll("button[data-del]").forEach((btn) => {
      btn.addEventListener("click", async () => {
        if (!window.confirm("删除该项目（含导出文件）？")) return;
        await API.del(`/api/projects/${btn.dataset.del}`);
        await refreshMeta();
        if (state.projectId === btn.dataset.del) resetWorkspace();
        close();
        openProjectList();
      });
    });
  });
}

/* ------------------------------------------------------------------- init */

/* Show the workspace shell with no project open (used on a cold start and after
   deleting the current project) so the app always lands on the work page. */
function resetWorkspace() {
  state.project = null; state.projectId = null; state.media = null;
  state.selection.clear(); state.undo = []; state.redo = [];
  const stage = el("stage");
  if (!stage.querySelector(".workspace")) {
    stage.classList.remove("empty");
    stage.replaceChildren(el("workspace-template").content.cloneNode(true));
    bindWorkspace();
  }
  const host = el("media-host");
  if (host) host.replaceChildren();
  renderAll();
  renderProjectSelect();
}

async function init() {
  bindGlobalListeners();
  applyTheme();
  // Bind every chrome control up front. /api/state probes the detection engines
  // (importing torch can take several seconds on a cold start), and the UI must
  // stay responsive during that window instead of silently ignoring clicks.
  el("btn-theme").addEventListener("click", toggleTheme);
  el("btn-settings").addEventListener("click", openSettings);
  wireDragDrop();
  wireWindowControls();
  wireToolbarDrag();
  el("btn-open-project").addEventListener("click", openProjectList);
  el("btn-detect").addEventListener("click", () => {
    if (!state.project) { setMessage("请先导入项目"); return; }
    openDetectModal();
  });
  el("btn-roles").addEventListener("click", () => {
    if (!state.project) { setMessage("请先导入项目"); return; }
    openRoleManager();
  });
  el("btn-export").addEventListener("click", () => {
    if (!state.project) { setMessage("请先导入项目"); return; }
    openExportModal();
  });
  el("project-select").addEventListener("change", async (event) => {
    if (!event.target.value) return;
    const data = await API.get(`/api/projects/${event.target.value}`);
    state.selection.clear();
    applyProject(data.project);
  });

  // Land on the work page immediately, before the (possibly slow) engine probe.
  resetWorkspace();

  setBusy("正在加载…");
  await refreshMeta();
  setBusy("");

  // Open the most recent project straight away.
  if (state.meta.projects?.length) {
    const data = await API.get(`/api/projects/${state.meta.projects[0].id}`);
    applyProject(data.project);
  }
  if (!state.meta.ffmpeg) {
    setMessage("未找到 ffmpeg，导入视频与声纹提取将不可用。");
  }
}

init().catch((err) => { console.error(err); setMessage(err.message); });
