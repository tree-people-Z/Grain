# Grain

给**已有字幕**的视频/音频自动标注"每句字幕属于哪个说话人"，再用播放条逐句人工复核，最后导出字幕文件与训练用数据集格式。

**不是转录工具**——转录（ASR）只是无字幕时的可选辅助。

## 功能特性

- **字幕优先**：直接利用已有的 SRT/VTT/ASS 时间轴，不重新转录；ASR 仅作无字幕时的兜底。
- **多引擎说话人检测**：pyannote community-1 / NVIDIA Sortformer / FunASR CAM++ / 字幕级声纹聚类 / 纯手动，可切换、可人数扫描、可双引擎共识质检。
- **声纹先验 + 自动学习**：人工复核的归属会回灌成角色声纹，越核越准；角色命名即登记进跨项目全局角色库。
- **逐句人工复核**：播放条 + 说话人轨道 + 快捷键 + 批量归属 + 撤销/重做。
- **双语字幕**：单文件双语 / 双文件翻译轨自动识别与合并，显示与导出可切原文 / 双语 / 译文。
- **10 种导出**：SRT · ASS · VTT · 剧本 · JSONL · CSV · RTTM · HF Dataset · 自定义数据集（可含声纹）· 复核报告。

## 环境要求

- Python **3.10+**（开发使用 3.13）。
- 基础功能（导入 / 复核 / 导出）**零第三方依赖**。
- `ffmpeg`：**整合包已内置** `runtime\ffmpeg\bin\`，无需装；缺失时才回退 `PATH`。
- 自动检测需 `torch` 及各引擎依赖（`pip install -r requirements.txt`）；桌面壳需 `pywebview`。
- 模型：**整合包已内置** `models\`，离线可用；缺失时才回退用户缓存 / 联网下载。

## 快速开始

### 桌面版（pywebview，推荐）

```bat
双击 启动桌面版.bat
```

纯 Python 桌面壳（`desktop_pywebview.py`）：自动选端口、拉起 `server.py` 后端、健康检查通过后用系统 WebView 开原生窗口，关闭窗口即回收后端。**无需 Rust**。

- 依赖：`pip install pywebview`
- 桌面版导入弹窗里，媒体/字幕输入框旁会多一个「浏览…」按钮，直接调系统文件对话框选路径。
- 直接把文件（视频/音频 + 字幕）拖进窗口即可导入：WebView2 拦下了原生拖放，改由 Python 侧 DOM 事件读取真实路径后回填给前端，全程无需上传。

**开发模式（改代码自动重载）**：

```bat
双击 启动桌面版-开发.bat        REM 等价于 python desktop_pywebview.py --dev
```

`--dev` 会监视顶层后端 `.py` 与 `static/`：改动 `.py` 自动重启后端、改动前端资源自动刷新窗口，无需关掉重开。都是后端进程内的轮询，**不引入任何新依赖**。

### 网页版（无需桌面依赖）

```bat
双击 启动网页版.bat
```

即 `python server.py` + 自动打开浏览器 `http://127.0.0.1:8770`。

### 依赖

```bat
pip install -r requirements.txt
```

- 基础功能（导入/复核/导出）**零第三方依赖**，Python 3.10+ 即可（不跑自动检测时）。
- 检测引擎需要 `torch`（PyPI 的 Windows 轮子即 CPU 版）+ `pyannote.audio` 或 `funasr`+`modelscope`。
- `ffmpeg` 必须在 PATH（解码音频、生成波形）。
- 可选 ASR：`pip install faster-whisper`。
- 可选人声分离：`pip install demucs`（设置 → 检测 → 人声分离，先剥离 BGM 再检测）。

## 检测引擎（可切换）

**首选 pyannote（community-1）**：日番/通用最稳，尤其在 BGM、重叠语音、字幕边界不十分贴的情况下；`voiceprint-cue` 只适合「字幕时间轴极准且音频干净」的场景。

| 引擎 | 依赖 | 说明 |
|---|---|---|
| **pyannote.audio（community-1）** | torch（**模型已内置，无需 token**） | 通用精度最高、人数不限；默认 community-1，失败回退 3.1。整合包内置权重后离线可用、免门控；仅当删掉 `models\` 改为联网下载时才需要 token 与门控授权（且门控模型必须直连 huggingface.co）。**推荐** |
| **NVIDIA Sortformer v2.1** | 隔离环境（`uv` 自动装） | 中/日语最强、端到端，**无门控**；最多同时 4 人。`python scripts/setup_engine.py sortformer` |
| **DiariZen**（实验性） | 隔离环境（`uv`+git 自动装） | BUT Speech@FIT 开源 SOTA（WavLM-Large + EEND + VBx），AMI/AliMeeting/DIHARD 优于 pyannote 3.1，人数不限。权重 **CC BY-NC 4.0（仅非商用）**。`python scripts/setup_engine.py diarizen` |
| **3D-Speaker / FunASR CAM++** | torch、modelscope | 模型从 ModelScope 下载，**无需 token**，中文场景好、国内可直连 |
| **字幕级声纹聚类（免门控）** | torch、pyannote.audio | 对**每条字幕**单独提 wespeaker 声纹再聚类；字幕时间轴极准时贴合“一句一个说话人”，**无门控**。注意：它把每条字幕都强行归入某个聚类，但**只有能匹配到你已创建角色的聚类才会被归属，匹配不上的仍保持「待定」** |
| 纯手动 | 无 | 不自动检测，全部待定，人工归属 |

引擎在「运行说话人检测」弹窗里切换；未安装的引擎会标"不可用"并给出安装命令。

> **录入空间 = 检测空间（硬约束）**：声纹是某个模型产出的，只能和同一模型的聚类比较。录入时会**自动存入"当前/默认检测引擎"的特征空间**（先建角色再检测也没问题）；若某角色声纹的空间与所选检测引擎不一致，检测弹窗会**提前警告**、角色管理里标注「需重录」。

> **声纹模型**（设置 → 检测 → 声纹先验）：pyannote 家族可换用更强的说话人嵌入。默认 `pyannote wespeaker`（通用稳妥、与 community-1 同源）；可切到 **3D-Speaker ERes2NetV2**（短字幕/噪声下更稳，走 ModelScope 免 token）。不同模型是**不同特征空间**，切换后需用新模型重新录入声纹（界面会提示「需重录」）。

> **人声分离**（设置 → 检测 → 音频预处理，或检测弹窗「高级」）：检测/录入前用 **Demucs** 先剥离 BGM 与 OP/ED 伴奏，再跑检测。日番、有背景音乐或歌曲时往往比换引擎更有效。需 `pip install demucs`；首次对整段音频分离较慢（CPU 可能数分钟），结果会缓存。切换该开关后建议重新录入声纹。

> **声纹精修**：pyannote / Sortformer 这类按「说话人时间段」聚类的引擎，检测后会自动用**每句字幕自身的声纹**与各说话人簇质心比对，修正纯时间重叠判断错误、补全无重叠的句子（要求明显优于原判断才改，不打扰高置信结果）。随后还会用**已录入的角色声纹**给「无重叠 / 未匹配聚类」的句子补全归属（阈值更严），把聚类没分出来的人（配角、演唱段）也认回来。`voiceprint-cue` 不走这一步。

> **人数自动扫描**（检测弹窗可选）：在「最少–最多人数」范围内逐个试，按**轮廓系数**自动选最优人数。对 `voiceprint-cue` 默认勾选（它复用已提声纹、只重聚类，耗时≈单次）；**对 pyannote 默认关闭**（它是整条管线重跑，且 pyannote 自己会估人数，强制固定 k 可能更差）。

> **双引擎共识**（检测弹窗可选，**默认关闭**）：再跑一个互补引擎交叉验证，两引擎归属**不一致的句子标为「待定」**。它**只增加待定、不纠错**，仅当两个引擎都强时用作质检；耗时 ≈ 两次。**自身声纹已明确支持当前归属的句子不会被退回待定**（避免互补引擎聚类粒度不同导致的误伤）。

> **人数建议**：≤4 人可用 Sortformer；多数情况直接 pyannote（community-1，人数不限）。

### 外挂引擎（隔离环境）

Sortformer 的 NeMo 依赖与主程序不同，放在 `engines/sortformer/` 独立 venv，由
`external_engines.py` 以子进程调用（协议见 `engines/README.md`）。安装：

```bat
python scripts/setup_engine.py sortformer
```

### HuggingFace token 与门控授权

**整合包已内置模型 → 不需要 token。** pyannote 的权重随 `models\` 一起分发，`configure_model_caches()` 在检测不到 token 时会自动开启离线模式，直接从内置缓存加载，**既不联网、也不需要授权**（已实测：清空 token 后 community-1 仍能加载）。token 只在你**想重新下载/更换模型**时才需要。

需要 token 的场景（例如删掉了 `models\` 让它联网下载），按下面配置（环境变量 `HF_TOKEN`，或写入本地文件）：

```
data/hf_token.txt        ← 仅本机使用，不要提交到任何仓库
```

首次联网下载模型权重约 100MB。

⚠ **只接受 `speaker-diarization-3.1` 是不够的**——它只是一个配方，实际加载时会去拉它引用的两个模型仓库，而这两个仓库各自是门控的：

```
https://hf.co/pyannote/segmentation-3.0     ← 真正卡住检测的通常是这个
https://hf.co/pyannote/embedding
```

用同一个账号逐个打开、点一次「Agree / 接受」即可。若缺授权，运行检测会直接报出**还缺哪个仓库**以及对应链接，不用自己猜。

**不想折腾授权？** 选「字幕级声纹聚类」或「3D-Speaker / CAM++」：前者用 pyannote 开源声纹模型 wespeaker 对每条字幕提特征后聚类，后者走 ModelScope，两者都不需要任何门控条款。声纹与官方管线**同属一个特征空间**（之后补上授权切回官方管线时，已录入的声纹仍然可用）。

## ⚠ 声纹与引擎绑定（硬约束）

声纹先验只在**同一特征空间**内匹配：

- 用 pyannote 检测 → 只有 pyannote 空间录入的声纹参与先验匹配；
- 切换引擎后请用当前引擎重新录入（录入时自动打标签，跨空间声纹会被跳过并提示）。

**声纹无需手动录入**：复核时把字幕**归属给正确角色**，下次检测会自动把这些「人工归属」的字幕补进对应角色的声纹（同特征空间、去重、限量），越核越准；**给角色命名时会自动把它的声纹登记进全局角色库**，跨项目复用。

**宁可漏、不误判**：检测只在你**已创建的角色**里找匹配，匹配不上的聚类**不再自动生成“待定角色”**，其字幕一律保持「待定」等你人工归属。少匹配一条只是多一次手工，错归属一条却会悄悄污染数据集。

## 使用流程

1. **导入**视频/音频 + 自带字幕（SRT/VTT/ASS；Git Bash 路径 `/c/...` 也能识别）。没有字幕？点「用 ASR 转录生成」。
   - ASS 会自动**只留对白**：字幕组常把屏幕文字、标题、字幕组信息、译注和**逐字卡拉OK特效**都塞进 `Dialogue:`，这些非对白事件会被跳过（导入提示会给出跳过条数）；若某文件的成品歌词只存在于 `Comment:`（卡拉OK 常见做法），会自动还原为歌词行。
2. **运行说话人检测**（选引擎，**首选 pyannote community-1**）→ 自动归属 + 置信度 + 待定标记。
   - **阈值默认「自动」**：按引擎特征空间校准（pyannote≈0.66 / CAM++≈0.76 / 内置≈0.84）；可在「设置 → 检测」里手动指定（检测弹窗里不再显示）。
   - **只认已创建的角色**：检测只在已创建/已录入声纹的角色里找匹配；匹配不上的聚类**不再自动生成“待定角色”**，其字幕保持「待定」，由你人工归属（宁可漏、不误判）。
   - **声纹去静音**：每条字幕/时间段提声纹前会先按能量裁掉首尾静音与换气，逐句声纹更纯。
   - **人数自动扫描**：对 `voiceprint-cue` 默认勾选（复用声纹、耗时≈单次）；对 pyannote 等默认关闭。选最优人数时会**抑制过度切分**（不会为凑高分把一个人拆成多个）。
   - **双引擎共识默认关闭**：它只把两引擎不一致的句子标待定（**只增待定、不纠错**），两个引擎都强时才当质检用。
   - **自动学习声纹**：检测时会自动把「人工归属」过的字幕补进对应角色的声纹（同特征空间、去重、限量），越核越准；同时自动合并全局角色库里历史遗留的重名重复记录。
3. **播放条复核**：
   - 拖动标尺定位播放头（拖动时画面停止、播放头跟手）→ 视频跟随 seek；点击字幕块 → 跳到该句
   - 滚轮平移 · `Alt+滚轮` 以鼠标为锚缩放 · 中键/抓手拖动平移（不移动播放头）
    - 时间轴上方是**说话人检测轨道**：每个说话人一行，一眼看出聚类是否正确、有无重叠语音（**人工归属但无检测结果的角色也会生成一行**，位置来自其字幕）
   - 当前句下方点人物按钮归属；归属后自动跳下一句（可关）
   - **跟随播放**：播放时标尺平滑跟随（播放头停在内容左侧约 1/3 处不动，标尺与字幕块连续滑过）；手动平移/缩放会暂停跟随，重新勾选或继续播放即恢复。
   - 快捷键：`1-9` 人物 · `0` 待定 · `空格` 播放/暂停 · `←/→` 逐句 · `P` 下一个待处理 · `V/H` 选择/抓手 · `S` 吸附 · `Ctrl+Z/Y` 撤销/重做
   - 框选多条 → 点人物按钮批量归属
4. **角色管理**：重命名（**命名即自动登记声纹到全局角色库**）/改色/合并误分聚类；旧版本检测遗留的「待定角色」可逐个删除，也可一键「清理全部待定角色」（其字幕回到未归属）。**全局角色库按「名字 + 特征空间」去重**：同名同空间再次入库为覆盖更新，历史遗留的重名重复会在下次检测时自动合并。
5. **导出**。

## 双语字幕支持

两种真实场景都支持，导入时自动识别：

**场景 A：单文件双语**（同一 cue 两行，如中文原文 + 英文译文）

```
1
00:00:00,000 --> 00:00:03,200
你好，欢迎来到这次的访谈节目。
Hello and welcome to this interview.
```

导入时按脚本类型自动判断第 2 行是**译文**还是**硬换行的续行**（中文+英文 → 译文；两行同语言 → 合并为一句）。也可在导入弹窗手动指定：
`自动识别` / `双语（第 1 行原文，其余译文）` / `硬换行（合并为一句）` / `保持原样`。

**场景 B：双语双文件**（`video.zh.srt` + `video.en.srt`）

导入时在「翻译字幕文件（可选）」填入第二份字幕，系统按**时间轴最大重叠**逐条对齐合并。

**复核界面**：清单上方有「原文 / 双语 / 译文」切换，时间轴块、当前句、清单三处同步切换显示。

**导出**：导出弹窗里选择「仅原文 / 原文+译文 / 仅译文」——
- SRT/VTT：译文缩进在原文下一行（对齐说话人标签）
- ASS：译文用独立的 `Translation` 样式（小一号字），通过 `{\rTranslation}` 切换
- JSONL/CSV/HF/自定义数据集：原文进 `text`，译文进 `translation`；只导译文时原文保留在 `source_text`
- 单文件双语的原始多行文本会被保留，可在导入后无损切换多行模式（`POST /api/projects/<id>/line_mode`）

## 导出格式（10 种）

| 类别 | 格式 |
|---|---|
| 字幕 | SRT（`[说话人]` 前缀）· ASS（按说话人上色样式）· WebVTT（`<v 说话人>`） |
| 剧本 | 纯文本 `说话人：台词`，未归属句标为「（待定）」 |
| 数据集 | JSONL · CSV · RTTM（说话人日志标准）· HuggingFace Dataset 风格 JSON · 自定义数据集格式（含角色库、可含声纹向量） |
| 报告 | Markdown 复核报告（自动/人工统计、低置信度清单） |

导出文件在 `data/exports/<项目id>/`，弹窗内可直接下载。

## 项目结构

```
# 后端（Python，同目录平铺，互相 import）
server.py            HTTP 服务 + REST API（Range 流式播放、导出下载）
project.py           项目编排：导入→检测→对齐→复核→导出
diarize.py           说话人引擎（community-1 / 字幕级声纹 / CAM++ / Sortformer / 手动）+ 引擎绑定声纹
external_engines.py  外挂引擎桥（子进程 + JSON 协议，隔离 venv）
align.py             字幕×说话人时间段重叠对齐（置信度=重叠比）
roles.py             角色库、声纹录入、一对一先验匹配
audio_features.py    内置特征空间（FFT 频带能量 + 基频统计）；装有 numpy 时自动走向量化快路径
jsonutil.py          共用的 JSON 工具（NaN/Inf → null）
subtitle_io.py       SRT/VTT/ASS 解析导出 + 双语识别/拆分/合并（可逆）
dataset_export.py    10 种导出格式
media.py             ffmpeg/ffprobe 封装
transcribe.py        可选 ASR（faster-whisper）
make_sample.py       合成双人样例（测试用）
desktop_pywebview.py 桌面壳（纯 Python：起后端 + 开原生窗口，无需 Rust）

# 资源与子目录
static/              前端（原生 JS，播放条轨道；自带字体）
engines/             外挂引擎（sortformer、diarizen），各自隔离环境；见 engines/README.md
scripts/             setup_engine.py（安装外挂引擎的隔离环境，支持 PyPI 或 git 源码）
tests/               e2e.py · test_bilingual.py · test_ass_filter.py · test_detection_utils.py · test_library.py · test_enroll.py · test_professional.py · test_real_subtitle.py
sample/              测试样例（interview.* 与双语样例）
data/                运行时数据：projects/ exports/ work/ uploads/、settings.json、role_library.json、hf_token.txt（勿提交）

# 内置资源（整合包，.bat 启动，可整目录拷贝；体积大，未提交 git）
runtime/ffmpeg/bin/  ffmpeg.exe · ffprobe.exe（media.py 优先用这里，找不到才用 PATH）
models/hf/hub/       HuggingFace 缓存（pyannote、Sortformer），离线加载
models/modelscope/   ModelScope 缓存（CAM++ / FunASR），离线加载

# 启动与配置
启动网页版.bat · 启动桌面版.bat · 启动桌面版-开发.bat（首次启动自动下载资源）
scripts/fetch_assets.py  从 GitHub Release 下载模型 + ffmpeg（也可手动运行）
scripts/setup_engine.py  安装 Sortformer 隔离环境（需要时）
requirements.txt · README.md · .gitignore
```

## 整合包（`.bat` 启动，克隆即用）

不做 exe 打包。模型与 ffmpeg 体积太大，**不放在 git 里**（GitHub 拒收 >100MB 的单文件），而是放在 GitHub Release，由 `.bat` **首次启动时自动下载解压**（约 1GB，仅这一次）：

```bat
启动桌面版.bat        REM 原生窗口（需 pip install pywebview）
启动网页版.bat        REM 浏览器打开 http://127.0.0.1:8770（无桌面依赖）
启动桌面版-开发.bat    REM 开发模式：改代码自动重载
```

也就是说，**从 git 克隆 → 双击 `.bat` → 自动拉取资源 → 直接可用**，无需手动找模型或 ffmpeg。

- 资源地址：`https://github.com/tree-people-Z/Grain/releases/download/assets-v1/`
  （`grain-models.zip` → `models\`，`grain-ffmpeg.zip` → `runtime\ffmpeg\bin\`）
- 手动方式：`python scripts\fetch_assets.py`（`--force` 强制重下）；也可自己把两个 zip 解压到项目根目录。
- 下载失败不影响基础功能：程序照常启动，只是自动检测引擎显示「不可用」。

资源就位后由代码自动发现，**不需要任何环境变量**：

- `media.py` 优先使用 `runtime\ffmpeg\bin\ffmpeg.exe` / `ffprobe.exe`，找不到才回退系统 `PATH`；
- `diarize.py` / `transcribe.py` 启动时调用 `apppaths.configure_model_caches()`，把 HuggingFace / ModelScope 的缓存指向 `models\`；**无 token 时自动离线**（已实测清空 token 后 pyannote community-1 仍能从内置缓存加载）。

把资源就绪后的整个文件夹（含 `runtime\`、`models\`、`engines\`、`data\`）拷到另一台机器，只要那台机器有 Python 3.10+ 与 `pip install -r requirements.txt`（torch 选 CPU 或 CUDA 版皆可），双击 `.bat` 即可完全离线运行。

- **重装依赖**：`pip install -r requirements.txt` + `pip install pywebview`（桌面版）。
- **Sortformer**：`engines\sortformer\` 是隔离环境（NeMo 与主程序 torch 不兼容），**未随包分发**（5GB）。首次要跑 Sortformer 时执行一次 `python scripts\setup_engine.py sortformer` 重建（模型权重已在 `models\`，不会再下载）。
- **新增/更换模型**：直接放进 `models\hf\hub\` 或 `models\modelscope\`，或删掉 `models\` 让它回退到用户默认缓存/联网下载。
- 无边框窗口（frameless）没有原生标题栏；最小化 / 最大化 / 关闭是导航栏右端的自绘按钮，拖动窗口用导航栏空白处。

## 测试

仓库为「只留运行要用的」已移除 `tests\`、`sample\`、`make_sample.py`。如需回归，从 git 历史取回：

```bat
git checkout HEAD~1 -- tests sample make_sample.py
```
