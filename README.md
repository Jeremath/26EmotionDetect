# Emotion-Qwen

本项目用于多模态情感识别，输入为 `JSONL manifest`，每条样本包含文本、音频和视频路径；输出为结构化 `JSON` 结果，支持单模型推理和双模型辩论推理两种流程。

当前实现已经完成以下能力：

1. 使用 `Qwen/Qwen2-Audio-7B-Instruct` 提取音频情感线索。
2. 使用 `Qwen/Qwen2.5-VL-7B-Instruct` 提取视频情感线索。
3. 使用 `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` 进行最终情感推理。
4. 使用 `THUDM/glm-4-9b-chat` 与 DeepSeek 组成双模型辩论推理。
5. 支持通用数据集 manifest 构建，以及 MELD 数据集下载、整理、推理和指标统计。

## 项目结构

- `src/multimodal_emotion_pipeline.py`
  单模型主流程：音频线索提取 + 视频线索提取 + DeepSeek 最终推理。
- `src/debate.py`
  双模型辩论流程：DeepSeek 与 GLM 独立推理，不一致时进入辩论轮次，直到达成一致或超过阈值。
- `src/data_process.py`
  数据整理脚本，支持：
  - `generic`：将已有 `text/`、`audio/`、`video/` 目录整理为 manifest。
  - `meld`：将 MELD 原始数据和标注整理为系统输入 JSONL。
- `src/data_require.py`
  数据下载脚本，当前已内置 MELD。
- `data/`
  输入样本、MELD 原始数据和整理后的 manifest。
- `output/`
  推理结果和评测指标输出目录。

## 当前模型配置

### 单模型流程

- 音频模型：`Qwen/Qwen2-Audio-7B-Instruct`
- 视频模型：`Qwen/Qwen2.5-VL-7B-Instruct`
- 最终推理模型：`deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`

### 双模型辩论流程

- 音频模型：`Qwen/Qwen2-Audio-7B-Instruct`
- 视频模型：`Qwen/Qwen2.5-VL-7B-Instruct`
- 辩论模型 1：`deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`
- 辩论模型 2：`THUDM/glm-4-9b-chat`

默认情况下，辩论阶段最多进行 `3` 轮；如果超过阈值仍然未达成一致，则取 `GLM-4-9B-Chat` 的结果。

## 推理输出格式

最终推理模型必须返回严格的标签格式：

```xml
<think>推理过程</think><answer>最终情感</answer>
```

其中：

- `<think>`：输出模型的推理思考过程。
- `<answer>`：只输出最终情感标签。

代码内部已经对模型输出做了兼容处理，能够尽量修复如下情况：

- 返回中混入多余文本。
- 缺失 `<think>` 起始标签但存在 `</think>`。
- 返回 JSON 或代码块包裹的线索内容。

## 输入 manifest 格式

系统输入是 `JSONL`，每行一条样本。推荐字段如下：

```json
{"id":"sample_0001","text":"I am very upset.","audio_path":"data/audio/sample_0001.wav","video_path":"data/video/sample_0001.mp4","label":"anger"}
{"id":"sample_0002","text_path":"data/text/sample_0002.txt","audio_path":"data/audio/sample_0002.wav","video_path":"data/video/sample_0002.mp4","label":"joy"}
```

字段说明：

- `id`：样本编号，可选；不写时程序会自动生成。
- `text` 或 `text_path`：文本内容或文本文件路径。
- `audio_path`：音频路径。
- `video_path`：视频路径。
- `label`：真实标签，可选；如果提供，`debate.py` 可统计指标。
- `meta`：额外元信息，可选，会原样写入结果文件。

相对路径默认相对于 `manifest` 所在目录解析，也可以通过 `--data-root` 指定统一根目录。

## 输出结果格式

注意：输入是 `JSONL`，输出结果文件是一个 `JSON 数组`，不是 JSONL。

### 单模型输出

`src/multimodal_emotion_pipeline.py` 默认输出到 `output/result.json`，每条记录包含：

```json
{
  "id": "sample_0001",
  "text": "Why cant you understand my feel?",
  "audio_path": "/abs/path/audio.mp3",
  "video_path": "/abs/path/video.mp4",
  "emotion_cues": {
    "text": "Why cant you understand my feel?",
    "audio": ["angry mood", "disgusted mood"],
    "video": ["anger", "intensity"]
  },
  "audio_cues": {"audio": ["angry mood", "disgusted mood"]},
  "video_cues": {"video": ["anger", "intensity"]},
  "audio_raw_response": "...",
  "video_raw_response": "...",
  "model_output": "<think>...</think><answer>anger</answer>",
  "raw_model_output": "...",
  "emotion_prediction": "anger",
  "answer": "anger",
  "think": "...",
  "prompt": {},
  "label": "anger",
  "meta": {}
}
```

### 双模型辩论输出

`src/debate.py` 默认输出到 `output/result_debate.json`，在单模型字段基础上额外包含：

- `debate_max_rounds`
- `debate_history`
- `termination_reason`
- `consensus_reached`
- `consensus_round`
- `selected_model`

这样可以回溯每一轮辩论中两个模型的推理过程与最终裁决来源。

## 环境准备

建议环境：Ubuntu 22.04+，Python 3.10，NVIDIA GPU。

推荐安装方式：

```bash
conda create -n emotion-qwen python=3.10 -y
conda activate emotion-qwen
pip install -r requests.txt
pip install tiktoken
```

`requests.txt` 当前包含的核心依赖有：

- `torch==2.5.1`
- `transformers==4.50.0`
- `accelerate==1.2.1`
- `sentencepiece==0.2.0`
- `librosa==0.10.2.post1`
- `decord==0.6.0`
- `bitsandbytes==0.45.2`
- `av==12.0.0`

说明：

- `GLM-4-9B-Chat` 依赖 `tiktoken`，需要额外安装。
- MELD 数据整理与视频抽帧依赖 `av`。
- 如果要使用 4bit/8bit 量化，需保证 `bitsandbytes` 可用。

## 模型缓存路径

在运行前建议设置缓存目录：

```bash
export HF_HOME=/rczhang/rczhang/zhangrch/code2/hf_cache
export TRANSFORMERS_CACHE=/rczhang/rczhang/zhangrch/code2/hf_cache
export TORCH_HOME=/rczhang/rczhang/zhangrch/code2/torch_cache
```

## 运行方式

以下命令默认在项目根目录 `26EmotionDetect/` 下执行。

### 单模型推理

```bash
python src/multimodal_emotion_pipeline.py \
  --manifest data/samples.jsonl \
  --output output/result.json \
  --audio-device cuda:0 \
  --video-device cuda:1 \
  --reasoner-device cuda:0 \
  --qwen-dtype bfloat16 \
  --video-fps 0.25 \
  --video-max-pixels 200704
```

常用参数：

- `--manifest`：输入样本清单。
- `--output`：输出结果文件，默认 `output/result.json`。
- `--data-root`：相对路径统一解析根目录。
- `--audio-device` / `--video-device` / `--reasoner-device`：模型所在设备。
- `--audio-max-new-tokens`：音频线索生成长度。
- `--video-max-new-tokens`：视频线索生成长度。
- `--reasoner-max-new-tokens`：最终推理生成长度。
- `--video-fps`：视频抽帧频率，默认 `0.25`。
- `--video-min-pixels` / `--video-max-pixels`：控制视频视觉 token 数量。
- `--video-quantization`：`none`、`8bit`、`4bit`。
- `--video-cpu-offload`：8bit 模式下启用 CPU offload。
- `--append-output`：追加到已有输出结果。
- `--limit`：只处理前 N 条样本，方便调试。

### 双模型辩论推理

```bash
python src/debate.py \
  --manifest data/samples.jsonl \
  --output output/result_debate.json \
  --audio-device cuda:0 \
  --video-device cuda:1 \
  --reasoner-device cuda:2 \
  --debate-max-rounds 3
```

辩论流程如下：

1. 先用音频模型和视频模型分别抽取情感线索。
2. 将线索分别交给 DeepSeek 和 GLM 独立推理。
3. 如果两个模型 `<answer>` 一致，则直接终止。
4. 如果不一致，则把双方前一轮的 `<think>` 与 `<answer>` 一并作为上下文继续推理。
5. 如果达到最大轮次后仍不一致，则选择 `GLM-4-9B-Chat` 的结果。

额外参数：

- `--deepseek-model-id`：DeepSeek 模型 ID。
- `--glm-model-id`：GLM 模型 ID。
- `--debate-max-rounds`：最大辩论轮次，默认 `3`。
- `--metrics-output`：输出指标统计文本。

实现说明：

- 当前辩论模型按顺序在同一推理设备上加载，以降低显存压力。
- 输出中保留完整 `debate_history`，方便后续分析一致性与分歧来源。

## 数据整理

### 通用数据集整理

如果你已有如下目录结构：

```text
dataset_root/
  text/
  audio/
  video/
```

可运行：

```bash
python src/data_process.py generic \
  --dataset-root /path/to/dataset_root \
  --output-root /path/to/output_dir
```

生成：

- `output_dir/samples.jsonl`
- `output_dir/samples_data_process_warnings.log`

兼容说明：

- `--DESFOLDER` 和 `--OUTFOLDER` 仍然可用，便于兼容旧命令。
- `generic` 模式支持递归扫描子目录，并按相对路径匹配文本、音频、视频文件。

### MELD 数据下载

使用内置脚本下载 MELD：

```bash
python src/data_require.py --dataset meld --data-root data
```

当前下载内容包括：

- `data/MELD/annotations/train_sent_emo.csv`
- `data/MELD/annotations/dev_sent_emo.csv`
- `data/MELD/annotations/test_sent_emo.csv`
- `data/MELD/raw/original/MELD.Raw.tar.gz`
- `data/MELD/raw/unpacked/`
- `data/MELD/media/video/`

### MELD 数据整理

将 MELD 整理成系统输入 manifest：

```bash
python src/data_process.py meld \
  --dataset-root data/MELD \
  --output-jsonl data/meld.jsonl
```

如需提前提取 wav 音频，可增加：

```bash
python src/data_process.py meld \
  --dataset-root data/MELD \
  --output-jsonl data/meld.jsonl \
  --extract-audio
```

说明：

- 现在默认会将音频抽取为独立 `wav`，并写入 `data/MELD/media/audio/<split>/`。
- 标准化后的视频会放到 `data/MELD/media/video/<split>/`，文本会放到 `data/MELD/prepared/text/<split>/`。
- 如果你确实希望 `audio_path` 直接指向视频文件，可显式指定 `--audio-source video`。
- 脚本会自动查找以下任一原始压缩包位置：
  - `data/MELD/raw/original/MELD.Raw.tar.gz`
  - `data/MELD/raw/MELD.Raw.tar.gz`
  - `data/MELD/downloads/MELD.Raw.tar.gz`
  - `data/MELD/MELD.Raw.tar.gz`
- 如果部分样本缺少视频，会写入 warning 文件而不是直接中断。

## MELD 评测示例

先处理前 200 条样本：

```bash
python src/debate.py \
  --manifest data/meld.jsonl \
  --output output/meld_result.json \
  --metrics-output output/meld_metric.txt \
  --audio-device cuda:0 \
  --video-device cuda:1 \
  --reasoner-device cuda:2 \
  --limit 200
```

会生成：

- `output/meld_result.json`
- `output/meld_metric.txt`

当前实现支持以下指标：

- Accuracy
- Weighted Precision
- Weighted Recall
- Weighted F1
- Macro Precision
- Macro Recall
- Macro F1
- Per-label Precision / Recall / F1 / Support

MELD 相关工作中，通常以 `Accuracy` 和 `Weighted F1` 作为主指标。代码中的指标说明参考了以下文献：

- Frontiers 2023 GCF2-Net
  https://www.frontiersin.org/articles/10.3389/fnins.2023.1183132/full
- Adaptive weighting in a transformer framework for multimodal emotion recognition
  https://www.sciencedirect.com/science/article/pii/S0167639325001475

## 稳定性与兼容性说明

当前代码已经处理了几个常见问题：

- 短视频保护：对于非常短的视频，系统会确保至少采样 1 帧，避免 `video_fps` 较低时出现 0 帧导致的错误。
- 视频显存控制：支持降低 `video_max_pixels`，以及使用 4bit/8bit 量化和 CPU offload。
- 断点续跑：
  - `debate.py --append-output` 会跳过已经存在于输出文件中的样本。
  - 适合长时间跑大数据集时中断后继续。
- 多数据集兼容：
  - 主流程读取的是统一 manifest，不依赖具体数据集名称。
  - 只要能通过 `data_process.py` 或自定义脚本生成相同字段的 JSONL，就能复用现有推理与评测代码。

## 后续扩展建议

如果你后续需要接入其他数据集，通常只需做下面几件事：

1. 下载并放置数据到 `data/` 下。
2. 用 `src/data_process.py` 新增一个对应模式，或单独生成新的 manifest。
3. 继续复用 `src/multimodal_emotion_pipeline.py` 或 `src/debate.py`。
4. 如果新数据集标签空间不同，仍然可以直接统计 Accuracy 和 P/R/F1。

如果后续继续扩展更多数据集，建议始终把“数据整理”和“推理主流程”分离，保持 manifest 作为统一接口。
