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
