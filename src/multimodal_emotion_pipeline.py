#!/usr/bin/env python3
"""
Multimodal emotion pipeline:
1. Use Qwen2-Audio to extract audio emotion cues.
2. Use Qwen2.5-VL to extract video emotion cues.
3. Concatenate text + audio JSON + video JSON.
4. Use DeepSeek-R1-Distill-Qwen for final emotion reasoning.

Input format: JSONL manifest, one sample per line.
Output format: JSON written to result.json by default.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import librosa
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2AudioForConditionalGeneration,
)


PROMPT1 = (
    "You are an emotion analysis expert for speech. "
    'Return JSON only with schema {"audio":["cue1","cue2"]}. '
    "Summarize the strongest vocal emotion cues in concise phrases."
    "Please summarize with rich descriptions, and avoid using emotional vocabulary to draw conclusions."
)
PROMPT2 = (
    "You are an emotion analysis expert for video. "
    'Return JSON only with schema {"video":["cue1","cue2"]}. '
    "Summarize the strongest visual emotion cues in concise phrases."
    "Please summarize with rich descriptions, and avoid using emotional vocabulary to draw conclusions."
)
PROMPT3 = (
    "You are a multimodal emotion analysis expert. "
    "Use the user text, audio cues, and video cues to reason about the speaker's emotion. "
    "Return exactly two XML-style tags with no extra text: "
    "<think>your reasoning</think><answer>final emotion label</answer>. "
    "The <answer> tag must contain only the final emotion label."
)


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
    reasoner_model_id: str
    audio_device: str
    video_device: str
    reasoner_device: str
    qwen_dtype: str
    audio_max_new_tokens: int
    video_max_new_tokens: int
    reasoner_max_new_tokens: int
    video_fps: float
    video_attn_implementation: str
    video_quantization: str
    video_cpu_offload: bool
    video_gpu_memory_limit_gib: int
    video_cpu_memory_limit_gib: int
    video_use_cache: bool
    video_min_pixels: Optional[int]
    video_max_pixels: Optional[int]
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
        default=Path("output/result.json"),
        help="Path to the output JSON file.",
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
        "--reasoner-model-id",
        "--bert-model-id",
        dest="reasoner_model_id",
        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        help="Final multimodal emotion reasoning model.",
    )
    parser.add_argument("--audio-device", default="cuda:0", help="Device for audio model.")
    parser.add_argument("--video-device", default="cuda:1", help="Device for video model.")
    parser.add_argument(
        "--reasoner-device",
        "--bert-device",
        dest="reasoner_device",
        default="cuda:0",
        help="Device for the final reasoning model.",
    )
    parser.add_argument(
        "--qwen-dtype",
        choices=["auto", "bfloat16", "float16", "float32"],
        default="bfloat16",
        help="Torch dtype used when loading the Qwen-family models.",
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
        "--reasoner-max-new-tokens",
        type=int,
        default=512,
        help="Maximum generated tokens for the final reasoning model.",
    )
    parser.add_argument(
        "--video-fps",
        type=float,
        default=0.25,
        help="Frame sampling rate passed to the video model for video understanding.",
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
        "--video-cpu-offload",
        action="store_true",
        help="Enable CPU offload for 8-bit VL loading to reduce GPU memory pressure.",
    )
    parser.add_argument(
        "--video-gpu-memory-limit-gib",
        type=int,
        default=20,
        help="GPU memory limit used with --video-cpu-offload.",
    )
    parser.add_argument(
        "--video-cpu-memory-limit-gib",
        type=int,
        default=64,
        help="CPU memory limit used with --video-cpu-offload.",
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
        default=256 * 28 * 28,
        help="Maximum visual tokens for the video processor. Lower values reduce GPU memory.",
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
        help="Append to the existing JSON result file instead of overwriting it.",
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
        reasoner_model_id=args.reasoner_model_id,
        audio_device=args.audio_device,
        video_device=args.video_device,
        reasoner_device=args.reasoner_device,
        qwen_dtype=args.qwen_dtype,
        audio_max_new_tokens=args.audio_max_new_tokens,
        video_max_new_tokens=args.video_max_new_tokens,
        reasoner_max_new_tokens=args.reasoner_max_new_tokens,
        video_fps=args.video_fps,
        video_attn_implementation=args.video_attn_implementation,
        video_quantization=args.video_quantization,
        video_cpu_offload=args.video_cpu_offload,
        video_gpu_memory_limit_gib=args.video_gpu_memory_limit_gib,
        video_cpu_memory_limit_gib=args.video_cpu_memory_limit_gib,
        video_use_cache=args.video_use_cache,
        video_min_pixels=args.video_min_pixels,
        video_max_pixels=args.video_max_pixels,
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


def device_map_value(device: str) -> Any:
    if device == "cpu":
        return "cpu"
    if device == "cuda":
        return 0
    if device.startswith("cuda:"):
        return int(device.split(":", maxsplit=1)[1])
    raise ValueError(f"Unsupported device for quantized loading: {device}")


def build_video_max_memory(config: PipelineConfig) -> Dict[Any, str]:
    return {
        device_map_value(config.video_device): f"{config.video_gpu_memory_limit_gib}GiB",
        "cpu": f"{config.video_cpu_memory_limit_gib}GiB",
    }


def torch_dtype_from_name(name: str) -> Any:
    if name == "auto":
        return "auto"
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    return mapping[name]


def load_audio_waveform(path: Path, target_sr: int) -> Any:
    try:
        return librosa.load(path.as_posix(), sr=target_sr)
    except Exception as librosa_exc:  # noqa: BLE001
        if importlib.util.find_spec("av") is None:
            raise RuntimeError(
                f"Failed to load audio from {path} with librosa and PyAV is not installed."
            ) from librosa_exc

        import numpy as np
        import av  # type: ignore

        container = av.open(str(path))
        audio_streams = [stream for stream in container.streams if stream.type == "audio"]
        if not audio_streams:
            container.close()
            raise RuntimeError(f"No audio stream found in media file: {path}") from librosa_exc

        resampler = av.audio.resampler.AudioResampler(
            format="fltp",
            layout="mono",
            rate=target_sr,
        )
        chunks: List[Any] = []
        for packet in container.demux(audio_streams[0]):
            for frame in packet.decode():
                resampled = resampler.resample(frame)
                if resampled is None:
                    continue
                frames = resampled if isinstance(resampled, list) else [resampled]
                for out_frame in frames:
                    array = out_frame.to_ndarray()
                    chunks.append(array.reshape(-1))
        container.close()

        if not chunks:
            raise RuntimeError(f"No decoded audio samples found in media file: {path}") from librosa_exc

        waveform = np.concatenate(chunks).astype("float32")
        return waveform, target_sr


def estimate_video_num_frames(path: Path, target_fps: float, minimum_frames: int = 1) -> int:
    if target_fps <= 0:
        return minimum_frames

    if importlib.util.find_spec("av") is None:
        return minimum_frames

    try:
        import av  # type: ignore

        container = av.open(str(path))
        stream = next((item for item in container.streams if item.type == "video"), None)
        if stream is None:
            container.close()
            return minimum_frames

        duration_seconds = None
        if stream.duration is not None and stream.time_base is not None:
            duration_seconds = float(stream.duration * stream.time_base)
        elif container.duration is not None:
            duration_seconds = float(container.duration / 1_000_000)
        container.close()

        if not duration_seconds or duration_seconds <= 0:
            return minimum_frames
        estimated = int(round(duration_seconds * target_fps))
        return max(minimum_frames, estimated)
    except Exception:
        return minimum_frames


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
        if parsed is None and '\\"' in candidate:
            unescaped_candidate = (
                candidate.replace('\\"', '"')
                .replace("\\n", "\n")
                .replace("\\t", "\t")
            )
            try:
                parsed = json.loads(unescaped_candidate)
            except json.JSONDecodeError:
                json_fragment = extract_first_json_object(unescaped_candidate)
                if json_fragment:
                    try:
                        parsed = json.loads(json_fragment)
                    except json.JSONDecodeError:
                        parsed = None

    if isinstance(parsed, str):
        nested_candidate = parsed.strip()
        try:
            parsed = json.loads(nested_candidate)
        except json.JSONDecodeError:
            json_fragment = extract_first_json_object(nested_candidate)
            if json_fragment:
                try:
                    parsed = json.loads(json_fragment)
                except json.JSONDecodeError:
                    parsed = {key: [nested_candidate]}
            else:
                parsed = {key: [nested_candidate]}

    if isinstance(parsed, dict):
        cues = parsed.get(key, [])
        if isinstance(cues, str):
            cues = [cues]
        if isinstance(cues, list):
            cleaned = []
            for item in cues:
                if isinstance(item, dict):
                    cue_name = str(item.get("cue1", "")).strip()
                    description = str(item.get("description", "")).strip()
                    if cue_name and description:
                        cleaned.append(f"{cue_name}: {description}")
                    elif description:
                        cleaned.append(description)
                    elif cue_name:
                        cleaned.append(cue_name)
                    else:
                        serialized = json.dumps(item, ensure_ascii=False)
                        if serialized.strip():
                            cleaned.append(serialized)
                else:
                    value = str(item).strip()
                    if value:
                        cleaned.append(value)
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


def extract_tag_content(text: str, tag: str) -> str:
    match = re.search(rf"<{tag}>(.*?)</{tag}>", text, flags=re.DOTALL | re.IGNORECASE)
    if not match:
        return ""
    return match.group(1).strip()


def normalize_reasoner_output(raw_text: str) -> Dict[str, str]:
    think = extract_tag_content(raw_text, "think")
    answer = extract_tag_content(raw_text, "answer")

    if not think and "</think>" in raw_text:
        think_prefix = raw_text.split("</think>", maxsplit=1)[0]
        if "<think>" in think_prefix:
            think_prefix = think_prefix.split("<think>", maxsplit=1)[1]
        think = think_prefix.strip()

    if not think and "<answer>" in raw_text:
        think_prefix = raw_text.split("<answer>", maxsplit=1)[0]
        think = think_prefix.strip()

    if not answer:
        stripped = raw_text.strip()
        if stripped:
            answer = stripped

    tagged_output = f"<think>{think}</think><answer>{answer}</answer>"
    return {
        "think": think,
        "answer": answer,
        "tagged_output": tagged_output,
    }


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
        if config.video_cpu_offload:
            raise ValueError("--video-cpu-offload requires --video-quantization 8bit.")
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
        return BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=config.video_cpu_offload,
        )

    if config.video_cpu_offload:
        raise ValueError("--video-cpu-offload currently supports 8bit quantization only.")

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )


def read_existing_results(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []

    raw_text = path.read_text(encoding="utf-8").strip()
    if not raw_text:
        return []

    parsed = json.loads(raw_text)
    if not isinstance(parsed, list):
        raise ValueError(f"Expected a JSON array in {path}")
    return parsed


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
            if config.video_cpu_offload:
                video_model_kwargs["device_map"] = "auto"
                video_model_kwargs["max_memory"] = build_video_max_memory(config)
            else:
                video_model_kwargs["device_map"] = {"": device_map_value(config.video_device)}
            if self.qwen_dtype != "auto":
                video_model_kwargs["torch_dtype"] = self.qwen_dtype

        self.video_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            config.video_model_id,
            **video_model_kwargs,
        )
        if self.video_quantization_config is None:
            self.video_model.to(config.video_device)
        self.video_model.eval()

        if hasattr(self.video_model, "get_memory_footprint"):
            footprint_gib = self.video_model.get_memory_footprint() / (1024**3)
            print(
                f"Video model loaded with quantization={config.video_quantization}, "
                f"footprint={footprint_gib:.2f} GiB"
            )

        self.reasoner_tokenizer = AutoTokenizer.from_pretrained(config.reasoner_model_id)
        if self.reasoner_tokenizer.pad_token is None:
            self.reasoner_tokenizer.pad_token = self.reasoner_tokenizer.eos_token
        self.reasoner_model = AutoModelForCausalLM.from_pretrained(
            config.reasoner_model_id,
            torch_dtype=self.qwen_dtype,
            low_cpu_mem_usage=True,
        )
        self.reasoner_model.to(config.reasoner_device)
        self.reasoner_model.eval()

    def build_audio_instruction(self) -> str:
        return PROMPT1

    def build_video_instruction(self) -> str:
        return PROMPT2

    def build_reasoner_input(
        self,
        sample: Sample,
        audio_json: Dict[str, Any],
        video_json: Dict[str, Any],
    ) -> str:
        return (
            "Text:\n"
            f"{sample.text}\n\n"
            "Audio emotion cues:\n"
            f"{json_dumps(audio_json)}\n\n"
            "Video emotion cues:\n"
            f"{json_dumps(video_json)}"
        )

    def build_reasoner_prompt_text(self, user_prompt: str) -> str:
        messages = [
            {"role": "system", "content": PROMPT3},
            {"role": "user", "content": user_prompt},
        ]
        try:
            return self.reasoner_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except ValueError:
            return f"{PROMPT3}\n\n{user_prompt}\n\n"

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
        audio_waveform, _ = load_audio_waveform(
            sample.audio_path,
            target_sr=self.audio_processor.feature_extractor.sampling_rate,
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
        video_num_frames = estimate_video_num_frames(
            sample.video_path,
            target_fps=self.config.video_fps,
            minimum_frames=1,
        )
        inputs = self.video_processor.apply_chat_template(
            conversation,
            fps=self.config.video_fps,
            num_frames=video_num_frames,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = move_batch_to_device(inputs, self.config.video_device)

        with torch.inference_mode():
            generated_ids = self.video_model.generate(
                **inputs,
                do_sample=False,
                use_cache=self.config.video_use_cache,
                max_new_tokens=self.config.video_max_new_tokens,
            )

        trimmed_ids = generated_ids[:, inputs["input_ids"].shape[1] :]
        response = self.video_processor.batch_decode(
            trimmed_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

        del inputs
        del generated_ids
        del trimmed_ids
        empty_cuda_cache(self.config.video_device)
        video_json = normalize_cue_json(response, "video")
        return {"json": video_json, "raw_response": response}

    def classify_emotion(
        self,
        sample: Sample,
        audio_json: Dict[str, Any],
        video_json: Dict[str, Any],
    ) -> Dict[str, Any]:
        empty_cuda_cache(self.config.reasoner_device)
        user_prompt = self.build_reasoner_input(sample, audio_json, video_json)
        prompt_text = self.build_reasoner_prompt_text(user_prompt)
        model_inputs = self.reasoner_tokenizer(
            prompt_text,
            return_tensors="pt",
            padding=True,
        )
        model_inputs = move_batch_to_device(model_inputs, self.config.reasoner_device)

        with torch.inference_mode():
            generated_ids = self.reasoner_model.generate(
                **model_inputs,
                max_new_tokens=self.config.reasoner_max_new_tokens,
                do_sample=False,
                pad_token_id=self.reasoner_tokenizer.pad_token_id,
                eos_token_id=self.reasoner_tokenizer.eos_token_id,
            )

        trimmed_ids = generated_ids[:, model_inputs["input_ids"].shape[1] :]
        response = self.reasoner_tokenizer.batch_decode(
            trimmed_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
        normalized_output = normalize_reasoner_output(response)
        return {
            "answer": normalized_output["answer"],
            "think": normalized_output["think"],
            "tagged_output": normalized_output["tagged_output"],
            "raw_response": response,
            "prompt": {
                "system": PROMPT3,
                "user": user_prompt,
            },
        }

    def process_sample(self, sample: Sample) -> Dict[str, Any]:
        audio_result = self.extract_audio_cues(sample)
        video_result = self.extract_video_cues(sample)
        reasoner_result = self.classify_emotion(
            sample=sample,
            audio_json=audio_result["json"],
            video_json=video_result["json"],
        )

        return {
            "id": sample.sample_id,
            "text": sample.text,
            "audio_path": str(sample.audio_path),
            "video_path": str(sample.video_path),
            "emotion_cues": {
                "text": sample.text,
                "audio": audio_result["json"].get("audio", []),
                "video": video_result["json"].get("video", []),
            },
            "audio_cues": audio_result["json"],
            "video_cues": video_result["json"],
            "audio_raw_response": audio_result["raw_response"],
            "video_raw_response": video_result["raw_response"],
            "model_output": reasoner_result["tagged_output"],
            "raw_model_output": reasoner_result["raw_response"],
            "emotion_prediction": reasoner_result["answer"],
            "answer": reasoner_result["answer"],
            "think": reasoner_result["think"],
            "prompt": reasoner_result["prompt"],
            "label": sample.label,
            "meta": sample.meta,
        }


def iter_samples(samples: Iterable[Sample]) -> Iterable[Sample]:
    for sample in samples:
        yield sample


def write_results(path: Path, results: List[Dict[str, Any]]) -> None:
    path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    config = parse_args()
    samples = load_samples(config.manifest_path, config.data_root, config.limit)
    if not samples:
        raise ValueError(f"No samples found in {config.manifest_path}")

    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    results = read_existing_results(config.output_path) if config.append_output else []
    pipeline = MultiModalEmotionPipeline(config)

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
                "emotion_cues": {
                    "text": sample.text,
                    "audio": [],
                    "video": [],
                },
                "audio_cues": {"audio": []},
                "video_cues": {"video": []},
                "audio_raw_response": "",
                "video_raw_response": "",
                "model_output": "",
                "raw_model_output": "",
                "emotion_prediction": "",
                "answer": "",
                "think": f"处理失败，原因是：{exc}",
                "prompt": {},
                "label": sample.label,
                "meta": sample.meta,
            }
        results.append(result)
        write_results(config.output_path, results)

    print(f"Finished. Results written to {config.output_path}")


if __name__ == "__main__":
    main()
