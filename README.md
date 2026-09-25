# Grain

给**已有字幕**的视频/音频自动标注"每句字幕属于哪个说话人"，再用播放条逐句人工复核，最后导出字幕。

**不是转录工具**——只处理你已有的 SRT / ASS 字幕。

## 功能

- **字幕优先**：直接利用已有字幕的时间轴，不重新转录；支持 SRT 与 ASS（SSA）。
  - **双语字幕**：ASS 中同时间轴的 `JP/CH`、`日/中` 风格行会合并为一条双语字幕，日文与中文分行显示，避免同一句被重复检测。
  - ASS 自动**只留对白**：屏幕文字、标题、字幕组信息、译注、逐字卡拉OK特效都会被跳过；若成品歌词只存在于 `Comment:`（卡拉OK常见做法），会自动还原为歌词行。
- **说话人检测**：pyannote.audio `community-1`（失败回退 3.1），可设人数上下限；也可选「纯手动」不做自动检测。
- **角色声纹**：角色管理中可用人工确认字幕建立 CAM++ 声纹样本；后续检测会用声纹校正聚类到角色的映射。
- **逐句人工复核**：播放条 + 说话人轨道 + 快捷键 + 批量归属 + 撤销/重做。
- **待复核清单**：按待定、低时间匹配度或多人争议筛选；长清单只渲染可见字幕。
- **后台检测**：显示解码、检测、对齐阶段；刷新页面后可继续查看任务状态。
- **导出**：SRT（`[说话人]` 前缀）· ASS（按说话人上色）。

## 环境要求

- 源码运行需要 Python **3.10+**；分发整合包已内置 Python 3.12，无需对方另行安装 Python。
- 导入 / 复核 / 导出 / 纯手动引擎：**零第三方依赖**。
- `ffmpeg`：整合包已内置 `runtime\ffmpeg\bin\`；缺失时才回退 `PATH`。
- 自动检测需 `pip install -r requirements.txt`（torch + pyannote.audio）。
- `启动网页版.bat` 优先使用包内 `runtime\python\python.exe`，默认使用 CUDA GPU；没有 GPU、驱动不可用或模型初始化失败时自动回退 CPU。再回退到 `.venv` 或系统 Python。
- GPU 机器只需要安装兼容的 NVIDIA 显卡驱动；CUDA 运行库已随整合包内置。
- 模型：整合包已内置 `models\`，离线可用；缺失时才回退用户缓存 / 联网下载（门控模型需 HuggingFace token）。

## 快速开始

```bat
双击 启动网页版.bat
```

### 分发整合包

将整个 Grain 文件夹压缩后发给对方，解压到任意目录，双击 `启动网页版.bat` 即可。请保留 `runtime\python`、`runtime\ffmpeg`、`models`、`static` 和所有 `.py` 文件；这些目录共同组成离线运行环境。`data` 目录只在需要连同已有项目和媒体一起分享时保留。

或直接：

```bat
pip install -r requirements.txt
python server.py          REM 打开 http://127.0.0.1:8770
```

导入方式：点击顶部「＋ 导入」或把「视频/音频 + 字幕」拖进窗口。系统会先让你确认文件配对，再检查媒体时长、字幕条数和解析警告；确认后才创建项目。大文件也可以选择「按本机路径导入」，不经过浏览器上传。

## 使用流程

1. **导入**视频/音频 + 自带字幕（SRT/ASS）。完成一个项目后，点击顶部「＋ 导入」继续；已完成的项目会保留在项目库，也可以在导出完成后直接点击「＋ 导入下一组」。
2. **运行检测**：选引擎（默认 pyannote），设人数上下限 → 自动归属 + 置信度 + 待定标记。
   - **首次检测需要确认映射**：先试听每个聚类的代表片段，再把聚类选择为已有角色、新建角色或保持待定，确认后才批量归属字幕。重新检测时会根据时间重叠复用上次映射。
   - **宁可漏、不误判**：与说话人时间段无重叠、多人重叠或时间匹配低于 85% 的字幕保持「待定」，不猜；高级设置可关闭保守归属以提高自动覆盖率。
3. **播放条复核**：拖动标尺定位播放头；点击字幕块跳到该句；当前句下方点人物按钮归属。进入「需复核」或「待定」筛选后，归属会自动跳到下一条需要处理的字幕。
   - 「需复核」包含待定、时间匹配度低于 85% 的自动归属，以及多人时间重叠接近的字幕。时间匹配度是时间轴重叠比例，不代表声纹识别概率。
   - 快捷键：`1-9` 人物 · `0` 待定 · `空格` 播放/暂停 · `←/→` 逐句 · `P` 下一个待处理 · `V/H` 选择/抓手 · `S` 吸附 · `Ctrl+Z/Y` 撤销/重做
   - 框选多条 → 点人物按钮批量归属。
4. **角色管理**：新增 / 重命名 / 改色 / 合并 / 删除。
5. **导出**：选 SRT 或 ASS。导出前会显示待定、争议和未命名角色数量；文件在 `data/projects/<项目id>/exports/`，弹窗内可直接下载，并可点击「＋ 导入下一组」继续工作。

## 项目结构

```
server.py          HTTP 服务 + REST API（Range 流式播放、导出下载）
project.py         项目编排：导入→检测→对齐→复核→导出
diarize.py         pyannote 检测引擎 + 纯手动
align.py           字幕×说话人时间段重叠对齐（置信度=重叠比）
subtitle_io.py     SRT / ASS 解析导出（含 ASS 对白过滤、歌词还原）
audio_features.py  WAV 读取 + 波形峰值
media.py           ffmpeg/ffprobe 封装
jsonutil.py        共用的 JSON 工具（NaN/Inf → null）
apppaths.py        数据目录与内置模型缓存路径
static/            前端（原生 JS，播放条轨道；自带字体）
启动网页版.bat

data/              运行时数据（勿提交）
  projects/<id>/   每个项目一个自包含目录：project.json + media/ + work/ + exports/
  uploads/         导入暂存；导入时归入对应项目目录
runtime/python/     整合包内置 Python + 第三方依赖
runtime/ffmpeg/    内置 ffmpeg（整合包，未提交 git）
models/            内置 pyannote 权重（整合包，未提交 git）
```

## API

```
GET    /api/state
GET    /api/projects
GET    /api/projects/<id>
GET    /api/projects/<id>/media
GET    /api/projects/<id>/peaks
GET    /api/projects/<id>/export/<file>
POST   /api/projects
PUT    /api/upload?name=<文件名>
POST   /api/initialize
POST   /api/projects/<id>/detect
POST   /api/projects/<id>/detect/start
GET    /api/projects/<id>/detect/status
POST   /api/projects/<id>/segments
POST   /api/projects/<id>/bulk
POST   /api/projects/<id>/roles
POST   /api/projects/<id>/roles_update
POST   /api/projects/<id>/roles_merge
POST   /api/projects/<id>/reset_auto
POST   /api/projects/<id>/export
DELETE /api/projects/<id>
DELETE /api/projects/<id>/roles/<rid>
```

单句和批量归属接口返回变化后的 `segments` 与 `stats`，不再回传完整项目；客户端可按需重新读取 `GET /api/projects/<id>`。

## 说明

- 检测只在**同一特征空间**内本可匹配；本精简版**不含声纹库 / 跨项目角色复用**，因此聚类与已有角色的对应按顺序进行。
- 资源就位后由代码自动发现，无需环境变量：`media.py` 优先用 `runtime\ffmpeg\bin\`，`diarize.py` 把 HuggingFace 缓存指向 `models\`。
