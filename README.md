# Emotion-Qwen

本项目提供一个简洁的单文件多模态情感推理脚手架，满足以下流程：

1. 使用 `Qwen/Qwen2-Audio-7B-Instruct` 从音频中提取情感线索。
2. 使用 `Qwen/Qwen2.5-VL-7B-Instruct` 从视频中提取情感线索。
3. 将 `文本 + 音频JSON线索 + 视频JSON线索` 拼接成一个字符串。
4. 使用 `bhadresh-savani/bert-base-uncased-emotion` 对拼接后的文本做情感分类。
5. 将每条样本的完整输入与输出写入 `output.log`，方便后续继续做准确率等指标实验。

当前项目文件：

- `multimodal_emotion_pipeline.py`：主程序，包含数据读取、三模型调用、结果落盘。
- `README.md`：设计说明与运行方法。

## 设计说明

本实现默认采用 `JSONL` 作为输入清单格式，每行对应一个样本。这样后续你更换不同数据集时，只需要重新生成 manifest，而不需要重写主程序。

每条样本至少包含三个模态：

- 文本：`text` 或 `text_path`
- 音频：`audio_path`
- 视频：`video_path`

主程序的执行逻辑为：

1. 读取一条样本的文本、音频路径、视频路径。
2. 使用 AM 读取音频并生成如下格式的 JSON：

```json
{"audio":["cue1","cue2"]}
```

3. 使用 VM 读取视频并生成如下格式的 JSON：

```json
{"video":["cue1","cue2"]}
```

4. 将以下内容直接拼接为 BERT 输入：

```text
Prompt3

文本内容
{"audio":["cue1","cue2"]}
{"video":["cue1","cue2"]}
```

5. 输出 JSON 结果，并写入 `output.log`。

代码中已经预留了以下占位符，后续你可以直接修改：

- `PROMPT1 = "Prompt1"`
- `PROMPT2 = "Prompt2"`
- `PROMPT3 = "Prompt3"`

默认设备分配适配你的双卡环境：

- 音频模型：`cuda:0`
- 视频模型：`cuda:1`
- BERT：`cuda:0`

如果你后续想调整，只需要修改命令行参数：

- `--audio-device`
- `--video-device`
- `--bert-device`

## 输入数据格式

推荐在项目下准备一个 manifest 文件，例如 `data/samples.jsonl`。

示例 1：文本直接写在 manifest 中

```json
{"id":"sample_0001","text":"I am feeling very nervous today.","audio_path":"data/audio/sample_0001.wav","video_path":"data/video/sample_0001.mp4","label":"fear"}
{"id":"sample_0002","text":"I finally solved the problem.","audio_path":"data/audio/sample_0002.wav","video_path":"data/video/sample_0002.mp4","label":"joy"}
```

示例 2：文本单独存成文件

```json
{"id":"sample_0003","text_path":"data/text/sample_0003.txt","audio_path":"data/audio/sample_0003.wav","video_path":"data/video/sample_0003.mp4","label":"sadness"}
```

说明：

- `id` 可选，不写时程序会自动生成。
- `label` 可选，当前不会计算准确率，但会原样写入输出，方便你后续做评测。
- 相对路径默认相对于 `manifest` 所在目录解析，也可以通过 `--data-root` 指定新的根目录。

## 运行环境

你的目标环境是 Ubuntu 22.04.5 LTS + 2 x RTX 4090 24G。本项目代码就是按这个场景写的。

建议使用 Python 3.10 或 3.11。

一个可参考的安装流程如下：

```bash
conda create -n emotion-qwen python=3.10 -y
conda activate emotion-qwen
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install "git+https://github.com/huggingface/transformers" accelerate librosa sentencepiece decord bitsandbytes
```

说明：

- `Qwen2.5-VL` 的本地视频路径处理依赖较新的 `transformers` 实现，因此这里建议直接安装最新源码版。
- 如果你使用其他 CUDA 版本，请把 PyTorch 安装命令替换成与你服务器匹配的版本。
- 首次运行会自动从 Hugging Face 下载模型，确保服务器能访问相关模型仓库。

## 运行方法

最常用的运行命令：

```bash
python multimodal_emotion_pipeline.py \
  --manifest data/samples.jsonl \
  --output output.log \
  --audio-device cuda:0 \
  --video-device cuda:1 \
  --bert-device cuda:0 \
  --qwen-dtype bfloat16 \
  --video-fps 0.5
```

如果你希望进一步降低 `Qwen2.5-VL-7B-Instruct` 的显存占用，可以打开量化：

8-bit 量化：

```bash
python multimodal_emotion_pipeline.py \
  --manifest data/samples.jsonl \
  --output output.log \
  --audio-device cuda:0 \
  --video-device cuda:1 \
  --bert-device cuda:0 \
  --qwen-dtype float16 \
  --video-quantization 8bit \
  --video-fps 0.5
```

4-bit 量化：

```bash
python multimodal_emotion_pipeline.py \
  --manifest data/samples.jsonl \
  --output output.log \
  --audio-device cuda:0 \
  --video-device cuda:1 \
  --bert-device cuda:0 \
  --qwen-dtype float16 \
  --video-quantization 4bit \
  --video-fps 0.5
```

如果你的音频、视频、文本路径都希望相对于某个统一目录解析，可以增加：

```bash
python multimodal_emotion_pipeline.py \
  --manifest data/samples.jsonl \
  --data-root /path/to/dataset_root \
  --output output.log
```

如果你只想先抽几条样本调试：

```bash
python multimodal_emotion_pipeline.py \
  --manifest data/samples.jsonl \
  --output output.log \
  --limit 5
```

常用参数：

- `--manifest`：输入样本清单，JSONL 格式。
- `--output`：输出日志文件，默认是 `output.log`。
- `--data-root`：相对路径解析根目录。
- `--audio-device` / `--video-device` / `--bert-device`：三个模型的运行设备。
- `--audio-max-new-tokens`：音频线索提取最大生成长度。
- `--video-max-new-tokens`：视频线索提取最大生成长度。
- `--video-fps`：视频抽帧速率。当前默认是 `0.5`，用于降低 VL 显存占用。
- `--video-attn-implementation`：视频模型注意力实现，默认自动优先使用 `flash_attention_2`。
- `--video-quantization`：VL 量化模式，可选 `none`、`8bit`、`4bit`。
- `--video-min-pixels` / `--video-max-pixels`：视频视觉 token 范围，减小上限可以显著节省显存。
- `--video-use-cache`：开启视频生成 KV cache，默认关闭以节省显存。
- `--bert-max-length`：BERT 输入最大长度，默认 512。
- `--limit`：只处理前 N 条样本，便于调试。
- `--append-output`：追加写入 `output.log`，否则默认覆盖。

## 输出说明

程序会把每条样本的结果逐行写入 `output.log`。`output.log` 虽然扩展名是 `.log`，但内容实际是 JSON Lines，方便后续直接读取统计。

每行会包含如下信息：

```json
{
  "id": "sample_0001",
  "text": "I am feeling very nervous today.",
  "audio_path": "/abs/path/to/sample_0001.wav",
  "video_path": "/abs/path/to/sample_0001.mp4",
  "audio_cues": {"audio":["cue1","cue2"]},
  "video_cues": {"video":["cue1","cue2"]},
  "answer": "fear",
  "think": "文本提供了基础情绪语义信息，音频线索显示语速较慢且能量偏低，视频线索显示表情紧绷且目光回避，综合判断该样本最可能的情感标签为 fear。"
}
```

按当前要求，输出结果只保留以下字段：

- `id`
- `text`
- `audio_path`
- `video_path`
- `audio_cues`
- `video_cues`
- `answer`
- `think`

其中：

- `answer`：BERT 阶段输出的最终情感标签。
- `think`：围绕文本、音频线索、视频线索整理出的解释性文字。

如果某条样本处理失败，程序不会直接中断，而是仍然输出同样结构的 JSON，只是 `answer` 为空，`think` 中写入失败原因，便于你继续跑完整个数据集。

## 后续修改建议

如果你后续要适配不同数据集，通常只需要改下面几个位置：

1. 修改 `data/samples.jsonl` 的生成方式，不必改主流程。
2. 直接在 `multimodal_emotion_pipeline.py` 顶部替换 `PROMPT1`、`PROMPT2`、`PROMPT3`。
3. 如果你未来想把最终推理模型从 BERT 换成其他模型，只需要重点改 `classify_emotion()`。
4. 如果你后续要增加准确率、F1 等指标，可以在读取 `output.log` 后单独写评测脚本，也可以在当前脚本基础上追加统计逻辑。

## 一个需要特别注意的点

`bhadresh-savani/bert-base-uncased-emotion` 是英文 `uncased` 情感分类模型。如果你的数据集文本是中文，或者你让 AM / VM 输出中文线索，那么最终 BERT 分类效果很可能不理想。

也就是说，当前代码已经严格按你的模型要求搭好了流程，但从实验设计角度看：

- 如果你的文本主要是英文，这个组合更自然。
- 如果你的文本主要是中文，建议你后续重点关注 `PROMPT1`、`PROMPT2`、`PROMPT3` 的语言，以及是否需要更换最终分类模型。

## 说明

当前版本没有实现准确率、F1、召回率等评测指标，这部分按你的要求暂时留空，但输入输出结构已经整理好，后续可以很方便继续扩展。

补充说明：

- 当前脚本本身就是逐条样本处理，因此对 VL 来说有效 batch size 已经是 `1`，没有再额外增大。
- 如果显存仍然紧张，优先尝试 `--video-quantization 8bit`，其次再尝试 `--video-quantization 4bit`。
