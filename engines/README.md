# 外挂检测引擎（隔离环境）

这些引擎依赖与主程序不同的 Python/torch 版本，因此各自放在独立目录里，由
`external_engines.py` 以**子进程**方式调用，互不污染。

## 约定

每个 `<引擎>/` 目录包含：

- `run.py` —— 入口。参数 `--wav <16k 单声道 wav> --min <n> --max <n>`；
  在 stdout 的**最后一行**输出一个 JSON：
  ```json
  {"turns": [{"start": 1.2, "end": 3.4, "cluster": 0}], "notes": ["..."]}
  ```
  其余日志写 stderr。
- `.venv/` —— 由安装脚本创建的隔离解释器（`Scripts/python.exe` 或 `bin/python`）。

主程序只有在 `.venv` 与 `run.py` 都存在时才把该引擎标为“可用”。

## 已接入

### sortformer —— NVIDIA Sortformer v2.1（NeMo，≤4 人）

端到端 Transformer diarization；中/日语表现最强、无需 HF 门控。**最多 4 人**。

安装：

```bat
python scripts/setup_engine.py sortformer
```

脚本会用 `uv` 创建 Python 3.12 环境，装 CUDA 版 torch 与 `nemo_toolkit[asr]`，
首次推理再从 HuggingFace 下载模型权重（约 470MB）。

> 5 人以上请改用内置「pyannote community-1」或「字幕级声纹聚类」。

### diarizen —— DiariZen（WavLM + EEND + VBx，实验性）

BUT Speech@FIT 的开源 SOTA 说话人日志，在 AMI / AliMeeting / DIHARD 等基准上优于
pyannote 3.1，人数不限、自带 EEND 分段与 VBx 聚类。

安装：

```bat
python scripts/setup_engine.py diarizen
```

脚本会用 `uv` 建 Python 3.10 环境，`git clone` 官方仓库（含子模块），再装
`requirements.txt` 与可编辑包。首次推理从 HuggingFace 下载权重
（`BUT-FIT/diarizen-wavlm-large-s80-md-v2`，约 280MB）。

> ⚠ 权重是 **CC BY-NC 4.0（仅限非商用）**；安装依赖 git 且耗时较长，属实验性接入。
> 若失败，该引擎会显示为「不可用」，不影响内置引擎。
