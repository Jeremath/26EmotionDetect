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
import ast
import importlib.util
import json
import re
import traceback
from collections import Counter
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
    'Return JSON only with schema {"audio":["cue1","cue2"],"quality":0.0,"confidence":0.0,"ambiguity":0.0,"signal_strength":0.0,"recommended_use":true,"reason":"..."} . '
    "Describe only observable vocal evidence such as speaking rate, loudness, pitch variation, hesitation, tremor, breathiness, pauses, or emphasis. "
    "Avoid direct emotion labels whenever possible. "
    "quality means signal cleanliness and reliability, confidence means confidence that the cues are useful, ambiguity means how mixed or unclear the signal is, and signal_strength means how strongly the audio suggests any emotion rather than neutral speech. "
    "recommended_use should be false when the audio evidence is weak, noisy, generic, or highly ambiguous."
)
PROMPT2 = (
    "You are an emotion analysis expert for video. "
    'Return JSON only with schema {"video":["cue1","cue2"],"quality":0.0,"confidence":0.0,"ambiguity":0.0,"signal_strength":0.0,"recommended_use":true,"reason":"..."} . '
    "Describe only observable visual evidence such as facial muscle tension, eye gaze, mouth shape, posture, gesture, head movement, or interaction context. "
    "Avoid direct emotion labels whenever possible. "
    "quality means visual reliability, confidence means confidence that the cues are useful, ambiguity means how mixed or unclear the visual signal is, and signal_strength means how strongly the visual evidence suggests any emotion rather than a neutral state. "
    "recommended_use should be false when the video evidence is weak, generic, occluded, or highly ambiguous."
)
TEXT_GATE_PROMPT = (
    "You are an emotion analysis expert. "
    "Assess the text alone before multimodal fusion. "
    'Return JSON only with schema {"label":"neutral|joy|sadness|anger|fear|disgust|surprise","confidence":0.0,"clarity":0.0,"reason":"..."} . '
    "confidence measures how likely your text-only label is correct. "
    "clarity measures how explicit and unambiguous the emotion is in the text itself. "
    "If the text is emotionally weak, mixed, or mostly factual, assign low clarity and prefer neutral."
)
PROMPT3 = (
    "You are a multimodal emotion analysis expert. "
    "Text is the primary evidence. Audio and video cues are auxiliary evidence and may be noisy, generic, or misleading. "
    "Follow the provided gate decisions strictly. If text emotion is already clear, do not let weak auxiliary cues override it. Ignore any modality marked as excluded. "
    "Use the user text, the text assessment, and any included auxiliary cues to reason about the speaker's emotion. "
    "Return exactly two XML-style tags with no extra text: "
    "<think>your reasoning</think><answer>final emotion label</answer>. "
    "The <answer> tag must contain only the final emotion label."
    "Please choose one of the following seven labels as your result: neutral, joy, sadness, anger, fear, disgust, and surprise. If there is no obvious sentiment bias, please use neutral as your final result."
)

EMOTION_WORD_HINTS = {
    "anger",
    "angry",
    "joy",
    "happy",
    "happiness",
    "sad",
    "sadness",
    "fear",
    "fearful",
    "disgust",
    "disgusted",
    "surprise",
    "surprised",
    "neutral",
}
AMBIGUOUS_WORD_HINTS = {
    "appears",
    "possibly",
    "maybe",
    "might",
    "could",
    "suggests",
    "seems",
    "unclear",
    "mixed",
    "ambiguous",
}


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
    text_gate_confidence_threshold: float
    text_gate_clarity_threshold: float
    modality_gate_threshold: float
    strong_modality_gate_threshold: float
    metrics_output: Optional[Path]
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
        "--text-gate-confidence-threshold",
        type=float,
        default=0.80,
        help="If text confidence is above this threshold, treat text as strong primary evidence.",
    )
    parser.add_argument(
        "--text-gate-clarity-threshold",
        type=float,
        default=0.72,
        help="If text clarity is above this threshold, treat text as strong primary evidence.",
    )
    parser.add_argument(
        "--modality-gate-threshold",
        type=float,
        default=0.22,
        help="Minimum modality gate score required to include an auxiliary modality when text is not clearly decisive.",
    )
    parser.add_argument(
        "--strong-modality-gate-threshold",
        type=float,
        default=0.45,
        help="Minimum modality gate score required to include an auxiliary modality when text is already clear.",
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=None,
        help="Optional metrics report path. If labels are available, accuracy and P/R/F1 metrics are written here.",
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
        text_gate_confidence_threshold=args.text_gate_confidence_threshold,
        text_gate_clarity_threshold=args.text_gate_clarity_threshold,
        modality_gate_threshold=args.modality_gate_threshold,
        strong_modality_gate_threshold=args.strong_modality_gate_threshold,
        metrics_output=args.metrics_output.resolve() if args.metrics_output else None,
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


def parse_structured_candidate(raw_text: str) -> Any:
    candidate = raw_text.strip()
    if not candidate:
        return None

    if "```" in candidate:
        candidate = candidate.replace("```json", "```").replace("```JSON", "```")
        segments = [segment.strip() for segment in candidate.split("```") if segment.strip()]
        json_like = next((segment for segment in segments if segment.startswith("{") or segment.startswith("[")), None)
        if json_like:
            candidate = json_like

    for attempt in (candidate, candidate.replace('\\"', '"').replace("\\n", "\n").replace("\\t", "\t")):
        try:
            return json.loads(attempt)
        except json.JSONDecodeError:
            json_fragment = extract_first_json_object(attempt)
            if json_fragment:
                try:
                    return json.loads(json_fragment)
                except json.JSONDecodeError:
                    pass
            try:
                return ast.literal_eval(attempt)
            except (ValueError, SyntaxError):
                pass
    return None


def clamp_score(value: Any, default: float = 0.0) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, score))


def normalize_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "y", "1"}:
            return True
        if lowered in {"false", "no", "n", "0"}:
            return False
    return default


def cue_item_to_text(item: Any, key: str) -> List[str]:
    if isinstance(item, dict):
        nested_cues = item.get(key)
        if isinstance(nested_cues, list):
            flattened: List[str] = []
            for nested in nested_cues:
                flattened.extend(cue_item_to_text(nested, key))
            return flattened

        cue_name = str(item.get("cue1", "")).strip()
        description = str(item.get("description", "")).strip()
        if cue_name and description:
            return [f"{cue_name}: {description}"]
        if description:
            return [description]
        if cue_name:
            return [cue_name]
        serialized = json.dumps(item, ensure_ascii=False)
        return [serialized] if serialized.strip() else []

    if isinstance(item, list):
        flattened = []
        for nested in item:
            flattened.extend(cue_item_to_text(nested, key))
        return flattened

    value = str(item).strip()
    if not value:
        return []

    parsed = parse_structured_candidate(value)
    if isinstance(parsed, dict) or isinstance(parsed, list):
        return cue_item_to_text(parsed, key)
    return [value]


def estimate_modality_metadata(cues: List[str]) -> Dict[str, Any]:
    if not cues:
        return {
            "quality": 0.10,
            "confidence": 0.10,
            "ambiguity": 0.95,
            "signal_strength": 0.05,
            "recommended_use": False,
            "reason": "No usable modality cues were extracted.",
        }

    joined = " ".join(cues).lower()
    direct_emotion_count = sum(
        1
        for cue in cues
        if any(
            token in cue.lower().split() or cue.lower().strip() == token or token in cue.lower()
            for token in EMOTION_WORD_HINTS
        )
    )
    rich_description_count = sum(1 for cue in cues if len(cue.split()) >= 4)
    ambiguous_hits = sum(1 for token in AMBIGUOUS_WORD_HINTS if token in joined)

    quality = 0.55 + min(0.20, rich_description_count * 0.06) - min(0.25, direct_emotion_count * 0.08)
    confidence = 0.58 + min(0.18, len(cues) * 0.05) - min(0.18, ambiguous_hits * 0.04)
    ambiguity = 0.30 + min(0.35, ambiguous_hits * 0.06) + min(0.20, direct_emotion_count * 0.05)
    signal_strength = 0.30 + min(0.25, direct_emotion_count * 0.08) + min(0.15, rich_description_count * 0.04)

    quality = max(0.05, min(0.95, quality))
    confidence = max(0.05, min(0.95, confidence))
    ambiguity = max(0.05, min(0.95, ambiguity))
    signal_strength = max(0.05, min(0.95, signal_strength))
    gate_score = quality * confidence * (1.0 - ambiguity) * (0.5 + 0.5 * signal_strength)
    return {
        "quality": quality,
        "confidence": confidence,
        "ambiguity": ambiguity,
        "signal_strength": signal_strength,
        "recommended_use": gate_score >= 0.22,
        "reason": "Metadata estimated heuristically because the extractor did not provide structured reliability scores.",
    }


def normalize_cue_json(raw_text: str, key: str) -> Dict[str, Any]:
    candidate = raw_text.strip()
    parsed = parse_structured_candidate(candidate)

    if isinstance(parsed, str):
        nested_candidate = parsed.strip()
        nested_parsed = parse_structured_candidate(nested_candidate)
        parsed = nested_parsed if nested_parsed is not None else {key: [nested_candidate]}

    result: Dict[str, Any] = {key: []}
    if isinstance(parsed, dict):
        cues = parsed.get(key, parsed.get("cues", []))
        cleaned: List[str] = []
        if isinstance(cues, str):
            cleaned.extend(cue_item_to_text(cues, key))
        elif isinstance(cues, list):
            for item in cues:
                cleaned.extend(cue_item_to_text(item, key))
        elif cues:
            cleaned.extend(cue_item_to_text(cues, key))

        result[key] = [item for item in cleaned if item]
        result["quality"] = clamp_score(parsed.get("quality"), default=-1.0)
        result["confidence"] = clamp_score(parsed.get("confidence"), default=-1.0)
        result["ambiguity"] = clamp_score(parsed.get("ambiguity"), default=-1.0)
        result["signal_strength"] = clamp_score(parsed.get("signal_strength"), default=-1.0)
        result["recommended_use"] = normalize_bool(parsed.get("recommended_use"), default=False)
        result["reason"] = str(parsed.get("reason", "")).strip()

    fallback_lines = [
        line.strip(" -*0123456789.")
        for line in raw_text.splitlines()
        if line.strip() and not line.strip().startswith("{") and not line.strip().startswith("}")
    ]
    fallback_lines = [line for line in fallback_lines if line]
    if not fallback_lines:
        fallback_lines = [raw_text.strip()]
    if not result.get(key):
        result[key] = fallback_lines

    estimated = estimate_modality_metadata(result[key])
    for metadata_key, estimated_value in estimated.items():
        if metadata_key not in result:
            result[metadata_key] = estimated_value
            continue
        current_value = result.get(metadata_key)
        if metadata_key == "reason":
            if not str(current_value or "").strip():
                result[metadata_key] = estimated_value
        elif metadata_key == "recommended_use":
            if current_value in {None, ""}:
                result[metadata_key] = estimated_value
        else:
            if isinstance(current_value, (int, float)) and current_value >= 0:
                continue
            result[metadata_key] = estimated_value

    return result


def normalize_text_assessment(raw_text: str) -> Dict[str, Any]:
    parsed = parse_structured_candidate(raw_text)
    if isinstance(parsed, str):
        nested_parsed = parse_structured_candidate(parsed)
        parsed = nested_parsed if nested_parsed is not None else {"label": parsed}

    if isinstance(parsed, dict):
        label = canonicalize_answer(str(parsed.get("label", "") or ""))
        if not label:
            label = canonicalize_answer(str(parsed.get("answer", "") or ""))
        if not label:
            label = "neutral"
        confidence = clamp_score(parsed.get("confidence"), default=0.50)
        clarity = clamp_score(parsed.get("clarity"), default=confidence)
        reason = str(parsed.get("reason", "")).strip()
        if not reason:
            reason = "Structured text assessment generated by the reasoner."
        return {
            "label": label,
            "confidence": confidence,
            "clarity": clarity,
            "reason": reason,
            "raw_response": raw_text,
        }

    normalized = normalize_reasoner_output(raw_text)
    fallback_label = canonicalize_answer(normalized["answer"]) or "neutral"
    return {
        "label": fallback_label,
        "confidence": 0.50,
        "clarity": 0.45,
        "reason": "Fallback text assessment was derived from an unstructured model response.",
        "raw_response": raw_text,
    }


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

    def build_text_assessment_prompt_text(self, text: str) -> str:
        messages = [
            {"role": "system", "content": TEXT_GATE_PROMPT},
            {"role": "user", "content": text},
        ]
        try:
            return self.reasoner_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except ValueError:
            return f"{TEXT_GATE_PROMPT}\n\n{text}\n\n"

    def assess_text_emotion(self, sample: Sample) -> Dict[str, Any]:
        text = sample.text.strip()
        if not text:
            return {
                "label": "",
                "confidence": 0.0,
                "clarity": 0.0,
                "reason": "No text was provided for text-first assessment.",
                "raw_response": "",
            }

        prompt_text = self.build_text_assessment_prompt_text(text)
        model_inputs = self.reasoner_tokenizer(
            prompt_text,
            return_tensors="pt",
            padding=True,
        )
        model_inputs = move_batch_to_device(model_inputs, self.config.reasoner_device)

        with torch.inference_mode():
            generated_ids = self.reasoner_model.generate(
                **model_inputs,
                max_new_tokens=128,
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
        return normalize_text_assessment(response)

    def compute_modality_gate(
        self,
        text_assessment: Dict[str, Any],
        audio_json: Dict[str, Any],
        video_json: Dict[str, Any],
    ) -> Dict[str, Any]:
        text_primary = (
            bool(text_assessment.get("label"))
            and float(text_assessment.get("confidence", 0.0)) >= self.config.text_gate_confidence_threshold
            and float(text_assessment.get("clarity", 0.0)) >= self.config.text_gate_clarity_threshold
        )

        def gate_one(modality_name: str, modality_json: Dict[str, Any]) -> Dict[str, Any]:
            quality = clamp_score(modality_json.get("quality"), default=0.10)
            confidence = clamp_score(modality_json.get("confidence"), default=0.10)
            ambiguity = clamp_score(modality_json.get("ambiguity"), default=0.95)
            signal_strength = clamp_score(modality_json.get("signal_strength"), default=0.05)
            recommended_use = normalize_bool(modality_json.get("recommended_use"), default=False)
            gate_score = quality * confidence * (1.0 - ambiguity) * (0.5 + 0.5 * signal_strength)
            threshold = (
                self.config.strong_modality_gate_threshold
                if text_primary
                else self.config.modality_gate_threshold
            )
            use_modality = bool(modality_json.get(modality_name)) and gate_score >= threshold and (
                recommended_use or gate_score >= threshold + 0.08
            )
            if not modality_json.get(modality_name):
                reason = f"{modality_name} excluded because no usable cues were extracted."
            elif text_primary and gate_score < threshold:
                reason = (
                    f"{modality_name} excluded because text evidence is already clear and the modality gate "
                    f"score {gate_score:.3f} is below the strong threshold {threshold:.2f}."
                )
            elif gate_score < threshold:
                reason = (
                    f"{modality_name} excluded because the modality gate score {gate_score:.3f} is below "
                    f"the threshold {threshold:.2f}."
                )
            elif not recommended_use and gate_score < threshold + 0.08:
                reason = f"{modality_name} excluded because the extractor marked it as weak or ambiguous."
            else:
                reason = f"{modality_name} included with gate score {gate_score:.3f}."
            return {
                "use": use_modality,
                "gate_score": gate_score,
                "threshold": threshold,
                "quality": quality,
                "confidence": confidence,
                "ambiguity": ambiguity,
                "signal_strength": signal_strength,
                "recommended_use": recommended_use,
                "reason": reason,
            }

        audio_gate = gate_one("audio", audio_json)
        video_gate = gate_one("video", video_json)
        return {
            "text_primary": text_primary,
            "text_confidence": float(text_assessment.get("confidence", 0.0)),
            "text_clarity": float(text_assessment.get("clarity", 0.0)),
            "audio": audio_gate,
            "video": video_gate,
        }

    def build_reasoner_input(
        self,
        sample: Sample,
        text_assessment: Dict[str, Any],
        audio_json: Dict[str, Any],
        video_json: Dict[str, Any],
        gate_report: Dict[str, Any],
    ) -> str:
        audio_section = (
            json_dumps(audio_json)
            if gate_report["audio"]["use"]
            else f"OMITTED. {gate_report['audio']['reason']}"
        )
        video_section = (
            json_dumps(video_json)
            if gate_report["video"]["use"]
            else f"OMITTED. {gate_report['video']['reason']}"
        )
        return (
            "Fusion policy:\n"
            f'{json_dumps({"text_primary": gate_report["text_primary"], "audio_gate": gate_report["audio"], "video_gate": gate_report["video"]})}\n\n'
            "Text-first assessment:\n"
            f"{json_dumps(text_assessment)}\n\n"
            "Text:\n"
            f"{sample.text}\n\n"
            "Audio auxiliary evidence:\n"
            f"{audio_section}\n\n"
            "Video auxiliary evidence:\n"
            f"{video_section}"
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
        text_assessment: Dict[str, Any],
        audio_json: Dict[str, Any],
        video_json: Dict[str, Any],
        gate_report: Dict[str, Any],
    ) -> Dict[str, Any]:
        empty_cuda_cache(self.config.reasoner_device)
        user_prompt = self.build_reasoner_input(sample, text_assessment, audio_json, video_json, gate_report)
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
            "text_assessment": text_assessment,
            "gate_report": gate_report,
        }

    def process_sample(self, sample: Sample) -> Dict[str, Any]:
        audio_result = self.extract_audio_cues(sample)
        video_result = self.extract_video_cues(sample)
        text_assessment = self.assess_text_emotion(sample)
        gate_report = self.compute_modality_gate(
            text_assessment=text_assessment,
            audio_json=audio_result["json"],
            video_json=video_result["json"],
        )
        reasoner_result = self.classify_emotion(
            sample=sample,
            text_assessment=text_assessment,
            audio_json=audio_result["json"],
            video_json=video_result["json"],
            gate_report=gate_report,
        )

        gated_audio = audio_result["json"] if gate_report["audio"]["use"] else {"audio": []}
        gated_video = video_result["json"] if gate_report["video"]["use"] else {"video": []}

        return {
            "id": sample.sample_id,
            "text": sample.text,
            "audio_path": str(sample.audio_path),
            "video_path": str(sample.video_path),
            "text_assessment": text_assessment,
            "modality_gate": gate_report,
            "emotion_cues": {
                "text": sample.text,
                "audio": audio_result["json"].get("audio", []),
                "video": video_result["json"].get("video", []),
            },
            "gated_emotion_cues": {
                "text": sample.text,
                "audio": gated_audio.get("audio", []),
                "video": gated_video.get("video", []),
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


def canonicalize_answer(answer: str) -> str:
    answer = answer.strip().lower()
    answer = answer.strip(" \t\r\n'\"`.,;:!?")
    return " ".join(answer.split())


def project_prediction_to_known_label(prediction: str, known_labels: List[str]) -> str:
    normalized_prediction = canonicalize_answer(prediction)
    if not normalized_prediction:
        return normalized_prediction

    normalized_labels = {canonicalize_answer(label) for label in known_labels if canonicalize_answer(label)}
    if normalized_prediction in normalized_labels:
        return normalized_prediction

    cleaned_prediction = normalized_prediction.replace(" emotion", "").replace(" mood", "")
    if cleaned_prediction in normalized_labels:
        return cleaned_prediction

    synonym_map = {
        "angry": "anger",
        "angry mood": "anger",
        "disgusted": "disgust",
        "fearful": "fear",
        "happy": "joy",
        "happiness": "joy",
        "sad": "sadness",
        "surprised": "surprise",
    }
    for source, target in synonym_map.items():
        if normalized_prediction == source or source in normalized_prediction:
            if target in normalized_labels:
                return target

    for label in normalized_labels:
        if label in normalized_prediction or normalized_prediction in label:
            return label

    return cleaned_prediction


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def compute_metrics(results: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    known_labels = sorted(
        {
            canonicalize_answer(str(result.get("label", "") or ""))
            for result in results
            if canonicalize_answer(str(result.get("label", "") or ""))
        }
    )
    pairs: List[tuple[str, str]] = []
    for result in results:
        label = canonicalize_answer(str(result.get("label", "") or ""))
        prediction = project_prediction_to_known_label(
            str(result.get("answer", "") or ""),
            known_labels=known_labels,
        )
        if label and prediction:
            pairs.append((label, prediction))

    if not pairs:
        return None

    labels = sorted({label for label, _ in pairs} | {prediction for _, prediction in pairs})
    label_support = Counter(label for label, _ in pairs)
    prediction_support = Counter(prediction for _, prediction in pairs)
    true_positive = Counter()
    for label, prediction in pairs:
        if label == prediction:
            true_positive[label] += 1

    per_label: List[Dict[str, Any]] = []
    for label in labels:
        tp = true_positive[label]
        fp = prediction_support[label] - tp
        fn = label_support[label] - tp
        precision = safe_divide(tp, tp + fp)
        recall = safe_divide(tp, tp + fn)
        f1 = safe_divide(2 * precision * recall, precision + recall)
        per_label.append(
            {
                "label": label,
                "support": label_support[label],
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

    total = len(pairs)
    accuracy = safe_divide(sum(1 for label, prediction in pairs if label == prediction), total)
    weighted_precision = sum(item["precision"] * item["support"] for item in per_label) / total
    weighted_recall = sum(item["recall"] * item["support"] for item in per_label) / total
    weighted_f1 = sum(item["f1"] * item["support"] for item in per_label) / total
    macro_precision = sum(item["precision"] for item in per_label) / len(per_label)
    macro_recall = sum(item["recall"] for item in per_label) / len(per_label)
    macro_f1 = sum(item["f1"] for item in per_label) / len(per_label)

    return {
        "evaluated_samples": total,
        "accuracy": accuracy,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "per_label": per_label,
    }


def write_metrics_report(metrics_path: Path, results: List[Dict[str, Any]]) -> None:
    metrics = compute_metrics(results)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    if metrics is None:
        metrics_path.write_text(
            "No labeled predictions were available, so no metrics were computed.\n",
            encoding="utf-8",
        )
        return

    lines = [
        "MELD / generic classification metrics",
        "",
        "Related-work note:",
        "MELD papers commonly report Accuracy and Weighted F1 as the primary metrics.",
        "Examples:",
        "- Frontiers 2023 GCF2-Net reports improvements on MELD in terms of accuracy and weighted average F1.",
        "  https://www.frontiersin.org/articles/10.3389/fnins.2023.1183132/full",
        "- Adaptive weighting in a transformer framework for multimodal emotion recognition reports MELD accuracy and weighted F1 in Table 3.",
        "  https://www.sciencedirect.com/science/article/pii/S0167639325001475",
        "",
        f"Evaluated samples: {metrics['evaluated_samples']}",
        f"Accuracy: {metrics['accuracy']:.6f}",
        f"Weighted Precision: {metrics['weighted_precision']:.6f}",
        f"Weighted Recall: {metrics['weighted_recall']:.6f}",
        f"Weighted F1: {metrics['weighted_f1']:.6f}",
        f"Macro Precision: {metrics['macro_precision']:.6f}",
        f"Macro Recall: {metrics['macro_recall']:.6f}",
        f"Macro F1: {metrics['macro_f1']:.6f}",
        "",
        "Per-label metrics:",
    ]

    for item in metrics["per_label"]:
        lines.append(
            f"- {item['label']}: support={item['support']}, "
            f"precision={item['precision']:.6f}, "
            f"recall={item['recall']:.6f}, "
            f"f1={item['f1']:.6f}"
        )

    metrics_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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
                "text_assessment": {
                    "label": "",
                    "confidence": 0.0,
                    "clarity": 0.0,
                    "reason": "Unavailable because sample processing failed.",
                    "raw_response": "",
                },
                "modality_gate": {
                    "text_primary": False,
                    "text_confidence": 0.0,
                    "text_clarity": 0.0,
                    "audio": {"use": False, "gate_score": 0.0, "threshold": 0.0, "quality": 0.0, "confidence": 0.0, "ambiguity": 1.0, "signal_strength": 0.0, "recommended_use": False, "reason": "Unavailable because sample processing failed."},
                    "video": {"use": False, "gate_score": 0.0, "threshold": 0.0, "quality": 0.0, "confidence": 0.0, "ambiguity": 1.0, "signal_strength": 0.0, "recommended_use": False, "reason": "Unavailable because sample processing failed."},
                },
                "emotion_cues": {
                    "text": sample.text,
                    "audio": [],
                    "video": [],
                },
                "gated_emotion_cues": {
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
        if config.metrics_output is not None:
            write_metrics_report(config.metrics_output, results)

    if config.metrics_output is not None:
        write_metrics_report(config.metrics_output, results)
    print(f"Finished. Results written to {config.output_path}")


if __name__ == "__main__":
    main()
