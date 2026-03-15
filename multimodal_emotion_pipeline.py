#!/usr/bin/env python3
"""
Multimodal emotion pipeline:
1. Use Qwen2-Audio to extract audio emotion cues.
2. Use Qwen2.5-VL to extract video emotion cues.
3. Concatenate text + audio JSON + video JSON.
4. Use BERT emotion classifier for final label prediction.

Input format: JSONL manifest, one sample per line.
Output format: JSON lines written to output.log by default.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import librosa
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
    Qwen2AudioForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
)


PROMPT1 = "Prompt1"
PROMPT2 = "Prompt2"
PROMPT3 = "Prompt3"


@dataclass
class Sample:
    sample_id: str
    text: str
    audio_path: Path
    video_path: Path
    label: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None


@dataclass
class PipelineConfig:
    manifest_path: Path
    output_path: Path
    data_root: Path
    audio_model_id: str
    video_model_id: str
    bert_model_id: str
    audio_device: str
    video_device: str
    bert_device: str
    qwen_dtype: str
    audio_max_new_tokens: int
    video_max_new_tokens: int
    video_fps: float
    video_attn_implementation: str
    video_quantization: str
    video_use_cache: bool
    video_min_pixels: Optional[int]
    video_max_pixels: Optional[int]
    bert_max_length: int
    limit: Optional[int]
    append_output: bool


def parse_args() -> PipelineConfig:
    parser = argparse.ArgumentParser(
        description="Run multimodal emotion reasoning with text, audio, and video."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/samples.jsonl"),
        help="Path to the input JSONL manifest.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output.log"),
        help="Path to the output log file (JSON lines).",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Base directory for relative text/audio/video paths. Defaults to the manifest parent directory.",
    )
    parser.add_argument(
        "--audio-model-id",
        default="Qwen/Qwen2-Audio-7B-Instruct",
        help="Audio cue extraction model.",
    )
    parser.add_argument(
        "--video-model-id",
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        help="Video cue extraction model.",
    )
    parser.add_argument(
        "--bert-model-id",
        default="bhadresh-savani/bert-base-uncased-emotion",
        help="Emotion classification model.",
    )
    parser.add_argument("--audio-device", default="cuda:0", help="Device for audio model.")
    parser.add_argument("--video-device", default="cuda:1", help="Device for video model.")
    parser.add_argument("--bert-device", default="cuda:0", help="Device for BERT model.")
    parser.add_argument(
        "--qwen-dtype",
        choices=["auto", "bfloat16", "float16", "float32"],
        default="bfloat16",
        help="Torch dtype used when loading the two Qwen models.",
    )
    parser.add_argument(
        "--audio-max-new-tokens",
        type=int,
        default=256,
        help="Maximum generated tokens for the audio cue model.",
    )
    parser.add_argument(
        "--video-max-new-tokens",
        type=int,
        default=96,
        help="Maximum generated tokens for the video cue model.",
    )
    parser.add_argument(
        "--video-fps",
        type=float,
        default=0.5,
        help="Frame sampling rate passed to Qwen2.5-VL for video understanding.",
    )
    parser.add_argument(
        "--video-attn-implementation",
        choices=["auto", "sdpa", "flash_attention_2", "eager"],
        default="auto",
        help="Attention implementation for Qwen2.5-VL. auto prefers flash_attention_2 when available, else sdpa.",
    )
    parser.add_argument(
        "--video-quantization",
        choices=["none", "8bit", "4bit"],
        default="none",
        help="Quantization mode for Qwen2.5-VL. Requires bitsandbytes for 8bit/4bit.",
    )
    parser.add_argument(
        "--video-use-cache",
        action="store_true",
        help="Enable KV cache during video generation. Disabled by default to save GPU memory.",
    )
    parser.add_argument(
        "--video-min-pixels",
        type=int,
        default=256 * 28 * 28,
        help="Minimum visual tokens for the video processor.",
    )
    parser.add_argument(
        "--video-max-pixels",
        type=int,
        default=1024 * 28 * 28,
        help="Maximum visual tokens for the video processor. Lower values reduce GPU memory.",
    )
    parser.add_argument(
        "--bert-max-length",
        type=int,
        default=512,
        help="Maximum token length for BERT classification input.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N samples for quick debugging.",
    )
    parser.add_argument(
        "--append-output",
        action="store_true",
        help="Append to output.log instead of overwriting it.",
    )

    args = parser.parse_args()
    manifest_path = args.manifest.resolve()
    data_root = args.data_root.resolve() if args.data_root else manifest_path.parent

    return PipelineConfig(
        manifest_path=manifest_path,
        output_path=args.output.resolve(),
        data_root=data_root,
        audio_model_id=args.audio_model_id,
        video_model_id=args.video_model_id,
        bert_model_id=args.bert_model_id,
        audio_device=args.audio_device,
        video_device=args.video_device,
        bert_device=args.bert_device,
        qwen_dtype=args.qwen_dtype,
        audio_max_new_tokens=args.audio_max_new_tokens,
        video_max_new_tokens=args.video_max_new_tokens,
        video_fps=args.video_fps,
        video_attn_implementation=args.video_attn_implementation,
        video_quantization=args.video_quantization,
        video_use_cache=args.video_use_cache,
        video_min_pixels=args.video_min_pixels,
        video_max_pixels=args.video_max_pixels,
        bert_max_length=args.bert_max_length,
        limit=args.limit,
        append_output=args.append_output,
    )


def resolve_path(path_value: str, data_root: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (data_root / path).resolve()


def path_to_uri(path: Path) -> str:
    return path.resolve().as_uri()


def move_batch_to_device(batch: Any, device: str) -> Any:
    if hasattr(batch, "to"):
        return batch.to(device)
    if isinstance(batch, dict):
        return {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in batch.items()
        }
    return batch


def empty_cuda_cache(device: str) -> None:
    if not torch.cuda.is_available() or not device.startswith("cuda"):
        return
    with torch.cuda.device(device):
        torch.cuda.empty_cache()


def torch_dtype_from_name(name: str) -> Any:
    if name == "auto":
        return "auto"
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    return mapping[name]


def read_text_from_record(record: Dict[str, Any], data_root: Path) -> str:
    if "text" in record and record["text"] is not None:
        return str(record["text"])

    text_path = record.get("text_path")
    if text_path:
        path = resolve_path(text_path, data_root)
        return path.read_text(encoding="utf-8").strip()

    raise ValueError("Each sample must contain either 'text' or 'text_path'.")


def load_samples(manifest_path: Path, data_root: Path, limit: Optional[int]) -> List[Sample]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    samples: List[Sample] = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue

            record = json.loads(stripped)
            sample_id = str(record.get("id", f"sample_{line_number:06d}"))
            text = read_text_from_record(record, data_root)
            audio_path = resolve_path(record["audio_path"], data_root)
            video_path = resolve_path(record["video_path"], data_root)

            samples.append(
                Sample(
                    sample_id=sample_id,
                    text=text,
                    audio_path=audio_path,
                    video_path=video_path,
                    label=record.get("label"),
                    meta=record.get("meta"),
                )
            )

            if limit is not None and len(samples) >= limit:
                break

    return samples


def extract_first_json_object(text: str) -> Optional[str]:
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    for index in range(start, len(text)):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def normalize_cue_json(raw_text: str, key: str) -> Dict[str, List[str]]:
    candidate = raw_text.strip()

    if "```" in candidate:
        candidate = candidate.replace("```json", "```").replace("```JSON", "```")
        segments = [segment.strip() for segment in candidate.split("```") if segment.strip()]
        json_like = next((segment for segment in segments if segment.startswith("{")), None)
        if json_like:
            candidate = json_like

    parsed: Optional[Dict[str, Any]] = None
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        json_fragment = extract_first_json_object(candidate)
        if json_fragment:
            try:
                parsed = json.loads(json_fragment)
            except json.JSONDecodeError:
                parsed = None

    if isinstance(parsed, dict):
        cues = parsed.get(key, [])
        if isinstance(cues, str):
            cues = [cues]
        if isinstance(cues, list):
            cleaned = [str(item).strip() for item in cues if str(item).strip()]
            return {key: cleaned}

    fallback_lines = [
        line.strip(" -*0123456789.")
        for line in raw_text.splitlines()
        if line.strip() and not line.strip().startswith("{") and not line.strip().startswith("}")
    ]
    fallback_lines = [line for line in fallback_lines if line]
    if not fallback_lines:
        fallback_lines = [raw_text.strip()]
    return {key: fallback_lines}


def json_dumps(data: Dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


def resolve_video_attn_implementation(config: PipelineConfig) -> str:
    if config.video_attn_implementation != "auto":
        return config.video_attn_implementation
    if (
        config.qwen_dtype in {"bfloat16", "float16"}
        and importlib.util.find_spec("flash_attn") is not None
    ):
        return "flash_attention_2"
    return "sdpa"


def build_video_quantization_config(config: PipelineConfig) -> Optional[BitsAndBytesConfig]:
    if config.video_quantization == "none":
        return None

    if importlib.util.find_spec("bitsandbytes") is None:
        raise ImportError(
            "bitsandbytes is required for --video-quantization 8bit/4bit. "
            "Install it with: pip install bitsandbytes"
        )

    compute_dtype = torch_dtype_from_name(config.qwen_dtype)
    if compute_dtype == "auto":
        compute_dtype = torch.bfloat16

    if config.video_quantization == "8bit":
        return BitsAndBytesConfig(load_in_8bit=True)

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )


class MultiModalEmotionPipeline:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.qwen_dtype = torch_dtype_from_name(config.qwen_dtype)
        self.video_quantization_config = build_video_quantization_config(config)
        self.audio_processor = AutoProcessor.from_pretrained(config.audio_model_id)
        self.audio_model = Qwen2AudioForConditionalGeneration.from_pretrained(
            config.audio_model_id,
            torch_dtype=self.qwen_dtype,
            low_cpu_mem_usage=True,
        )
        self.audio_model.to(config.audio_device)
        self.audio_model.eval()

        self.video_processor = AutoProcessor.from_pretrained(
            config.video_model_id,
            min_pixels=config.video_min_pixels,
            max_pixels=config.video_max_pixels,
        )
        video_model_kwargs: Dict[str, Any] = {
            "low_cpu_mem_usage": True,
            "attn_implementation": resolve_video_attn_implementation(config),
        }
        if self.video_quantization_config is None:
            video_model_kwargs["torch_dtype"] = self.qwen_dtype
        else:
            video_model_kwargs["quantization_config"] = self.video_quantization_config
            video_model_kwargs["device_map"] = {"": config.video_device}
            if self.qwen_dtype != "auto":
                video_model_kwargs["torch_dtype"] = self.qwen_dtype

        self.video_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            config.video_model_id,
            **video_model_kwargs,
        )
        if self.video_quantization_config is None:
            self.video_model.to(config.video_device)
        self.video_model.eval()

        self.bert_tokenizer = AutoTokenizer.from_pretrained(config.bert_model_id)
        self.bert_model = AutoModelForSequenceClassification.from_pretrained(config.bert_model_id)
        self.bert_model.to(config.bert_device)
        self.bert_model.eval()

    def build_audio_instruction(self) -> str:
        return (
            f"{PROMPT1}\n"
            'Return JSON only with schema {"audio": ["cue1", "cue2"]}. '
            "Do not output Markdown."
        )

    def build_video_instruction(self) -> str:
        return (
            f"{PROMPT2}\n"
            'Return JSON only with schema {"video": ["cue1", "cue2"]}. '
            "Do not output Markdown."
        )

    def build_bert_input(self, sample: Sample, audio_json: Dict[str, Any], video_json: Dict[str, Any]) -> str:
        return (
            f"{PROMPT3}\n\n"
            f"{sample.text}\n"
            f"{json_dumps(audio_json)}\n"
            f"{json_dumps(video_json)}"
        )

    def extract_audio_cues(self, sample: Sample) -> Dict[str, Any]:
        if not sample.audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {sample.audio_path}")

        conversation = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a Sentiment Analysis Expert."}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio_url": path_to_uri(sample.audio_path)},
                    {"type": "text", "text": self.build_audio_instruction()},
                ],
            },
        ]

        prompt_text = self.audio_processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )
        audio_waveform, _ = librosa.load(
            sample.audio_path.as_posix(),
            sr=self.audio_processor.feature_extractor.sampling_rate,
        )
        try:
            inputs = self.audio_processor(
                text=[prompt_text],
                audios=[audio_waveform],
                return_tensors="pt",
                padding=True,
            )
        except TypeError:
            inputs = self.audio_processor(
                text=[prompt_text],
                audio=[audio_waveform],
                return_tensors="pt",
                padding=True,
            )
        inputs = move_batch_to_device(inputs, self.config.audio_device)

        with torch.inference_mode():
            generated_ids = self.audio_model.generate(
                **inputs,
                max_new_tokens=self.config.audio_max_new_tokens,
                do_sample=False,
            )

        trimmed_ids = generated_ids[:, inputs["input_ids"].shape[1] :]
        response = self.audio_processor.batch_decode(
            trimmed_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        audio_json = normalize_cue_json(response, "audio")
        return {"json": audio_json, "raw_response": response}

    def extract_video_cues(self, sample: Sample) -> Dict[str, Any]:
        if not sample.video_path.exists():
            raise FileNotFoundError(f"Video file not found: {sample.video_path}")
        empty_cuda_cache(self.config.video_device)

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "path": sample.video_path.as_posix()},
                    {"type": "text", "text": self.build_video_instruction()},
                ],
            },
        ]

        inputs = self.video_processor.apply_chat_template(
            conversation,
            fps=self.config.video_fps,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = move_batch_to_device(inputs, self.config.video_device)

        with torch.inference_mode():
            generated_ids = self.video_model.generate(
                **inputs,
                max_new_tokens=self.config.video_max_new_tokens,
                do_sample=False,
                use_cache=self.config.video_use_cache,
            )

        trimmed_ids = generated_ids[:, inputs["input_ids"].shape[1] :]
        response = self.video_processor.batch_decode(
            trimmed_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        del inputs
        del generated_ids
        del trimmed_ids
        empty_cuda_cache(self.config.video_device)
        video_json = normalize_cue_json(response, "video")
        return {"json": video_json, "raw_response": response}

    def build_think(
        self,
        sample: Sample,
        audio_json: Dict[str, Any],
        video_json: Dict[str, Any],
        answer: str,
    ) -> str:
        parts: List[str] = []
        if sample.text.strip():
            parts.append("文本提供了主要的情绪语义线索")

        audio_cues = audio_json.get("audio", [])
        if audio_cues:
            parts.append(f"语音线索显示{audio_cues[0]}")

        video_cues = video_json.get("video", [])
        if video_cues:
            parts.append(f"视频线索显示{video_cues[0]}")

        if not parts:
            parts.append("当前样本缺少足够的多模态情绪线索")

        return f"{'，'.join(parts)}，综合判断该样本的情感标签为 {answer}。"

    def classify_emotion(
        self,
        sample: Sample,
        audio_json: Dict[str, Any],
        video_json: Dict[str, Any],
        bert_input: str,
    ) -> Dict[str, Any]:
        model_inputs = self.bert_tokenizer(
            bert_input,
            truncation=True,
            max_length=self.config.bert_max_length,
            return_tensors="pt",
        )
        model_inputs = move_batch_to_device(model_inputs, self.config.bert_device)

        with torch.inference_mode():
            logits = self.bert_model(**model_inputs).logits

        probabilities = torch.softmax(logits, dim=-1)[0].detach().cpu()
        scores = []
        for index, score in enumerate(probabilities.tolist()):
            label = self.bert_model.config.id2label[index]
            scores.append({"label": label, "score": round(score, 6)})
        scores.sort(key=lambda item: item["score"], reverse=True)
        answer = scores[0]["label"]
        think = self.build_think(
            sample=sample,
            audio_json=audio_json,
            video_json=video_json,
            answer=answer,
        )

        return {"answer": answer, "think": think}

    def process_sample(self, sample: Sample) -> Dict[str, Any]:
        audio_result = self.extract_audio_cues(sample)
        video_result = self.extract_video_cues(sample)
        bert_input = self.build_bert_input(sample, audio_result["json"], video_result["json"])
        bert_result = self.classify_emotion(
            sample=sample,
            audio_json=audio_result["json"],
            video_json=video_result["json"],
            bert_input=bert_input,
        )

        return {
            "id": sample.sample_id,
            "text": sample.text,
            "audio_path": str(sample.audio_path),
            "video_path": str(sample.video_path),
            "audio_cues": audio_result["json"],
            "video_cues": video_result["json"],
            "answer": bert_result["answer"],
            "think": bert_result["think"],
        }


def iter_samples(samples: Iterable[Sample]) -> Iterable[Sample]:
    for sample in samples:
        yield sample


def write_result(handle: Any, result: Dict[str, Any]) -> None:
    handle.write(json.dumps(result, ensure_ascii=False) + "\n")
    handle.flush()


def main() -> None:
    config = parse_args()
    samples = load_samples(config.manifest_path, config.data_root, config.limit)
    if not samples:
        raise ValueError(f"No samples found in {config.manifest_path}")

    output_mode = "a" if config.append_output else "w"
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    pipeline = MultiModalEmotionPipeline(config)

    with config.output_path.open(output_mode, encoding="utf-8") as output_handle:
        for index, sample in enumerate(iter_samples(samples), start=1):
            print(f"[{index}/{len(samples)}] Processing sample: {sample.sample_id}")
            try:
                result = pipeline.process_sample(sample)
            except Exception as exc:  # noqa: BLE001
                traceback.print_exc()
                result = {
                    "id": sample.sample_id,
                    "text": sample.text,
                    "audio_path": str(sample.audio_path),
                    "video_path": str(sample.video_path),
                    "audio_cues": {"audio": []},
                    "video_cues": {"video": []},
                    "answer": "",
                    "think": f"处理失败，原因是：{exc}",
                }
            write_result(output_handle, result)

    print(f"Finished. Results written to {config.output_path}")


if __name__ == "__main__":
    main()
