#!/usr/bin/env python3
"""
Multimodal emotion debate pipeline.

Pipeline:
1. Use Qwen2-Audio to extract audio emotion cues.
2. Use Qwen2.5-VL to extract video emotion cues.
3. Let DeepSeek-R1-Distill-Qwen-7B and GLM-4-9B-Chat reason independently.
4. If their <answer> tags disagree, run debate rounds until consensus or max rounds.
5. If max rounds are exceeded without consensus, choose the GLM result.
"""

from __future__ import annotations

import argparse
import gc
import json
import traceback
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2AudioForConditionalGeneration,
)

from multimodal_emotion_pipeline import (
    PROMPT1,
    PROMPT2,
    Sample,
    build_video_max_memory,
    build_video_quantization_config,
    clamp_score,
    device_map_value,
    empty_cuda_cache,
    estimate_video_num_frames,
    json_dumps,
    load_audio_waveform,
    load_samples,
    move_batch_to_device,
    normalize_cue_json,
    normalize_bool,
    normalize_reasoner_output,
    normalize_text_assessment,
    path_to_uri,
    read_existing_results,
    resolve_video_attn_implementation,
    TEXT_GATE_PROMPT,
    torch_dtype_from_name,
    write_results,
)


INITIAL_SYSTEM_PROMPT = (
    "You are a multimodal emotion analysis expert participating in a two-model debate. "
    "Text is the primary evidence. Audio and video cues are auxiliary evidence and may be noisy, generic, or misleading. "
    "Follow the provided gate decisions strictly. If text emotion is already clear, do not let weak auxiliary cues override it. Ignore any modality marked as excluded. "
    "Read the text, text-first assessment, and any included auxiliary cues carefully. "
    "Return exactly two XML-style tags with no extra text: "
    "<think>your reasoning</think><answer>final emotion label</answer>. "
    "The <answer> tag must contain only the final emotion label."
    "Please choose one of the following seven labels as your result: neutral, joy, sadness, anger, fear, disgust, and surprise. If there is no obvious sentiment bias, please use neutral as your final result."
)

DEBATE_SYSTEM_PROMPT = (
    "You are a multimodal emotion analysis expert participating in a two-model debate. "
    "Text is the primary evidence. Audio and video cues are auxiliary evidence and may be noisy, generic, or misleading. "
    "Follow the provided gate decisions strictly. If text emotion is already clear, do not let weak auxiliary cues override it. Ignore any modality marked as excluded. "
    "Review the evidence, your previous reasoning, and the other model's reasoning. "
    "If the other model is more convincing, revise your answer. If not, defend your answer. "
    "Return exactly two XML-style tags with no extra text: "
    "<think>your reasoning</think><answer>final emotion label</answer>. "
    "The <answer> tag must contain only the final emotion label."
    "Please choose one of the following seven labels as your result: neutral, joy, sadness, anger, fear, disgust, and surprise. If there is no obvious sentiment bias, please use neutral as your final result."
)


@dataclass
class DebateConfig:
    manifest_path: Path
    output_path: Path
    data_root: Path
    audio_model_id: str
    video_model_id: str
    deepseek_model_id: str
    glm_model_id: str
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
    debate_max_rounds: int
    metrics_output: Optional[Path]
    limit: Optional[int]
    append_output: bool


@dataclass
class ModelSpec:
    name: str
    model_id: str
    trust_remote_code: bool = False


def parse_args() -> DebateConfig:
    parser = argparse.ArgumentParser(
        description="Run multimodal emotion debate with DeepSeek and GLM."
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
        default=Path("output/result_debate.json"),
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
        "--deepseek-model-id",
        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        help="First debate model.",
    )
    parser.add_argument(
        "--glm-model-id",
        default="THUDM/glm-4-9b-chat",
        help="Second debate model.",
    )
    parser.add_argument("--audio-device", default="cuda:0", help="Device for audio model.")
    parser.add_argument("--video-device", default="cuda:1", help="Device for video model.")
    parser.add_argument(
        "--reasoner-device",
        default="cuda:2",
        help="Device for the sequential debate models.",
    )
    parser.add_argument(
        "--qwen-dtype",
        choices=["auto", "bfloat16", "float16", "float32"],
        default="bfloat16",
        help="Torch dtype used when loading the models.",
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
        help="Maximum generated tokens for the debate models.",
    )
    parser.add_argument(
        "--video-fps",
        type=float,
        default=0.25,
        help="Frame sampling rate passed to the video model.",
    )
    parser.add_argument(
        "--video-attn-implementation",
        choices=["auto", "sdpa", "flash_attention_2", "eager"],
        default="auto",
        help="Attention implementation for Qwen2.5-VL.",
    )
    parser.add_argument(
        "--video-quantization",
        choices=["none", "8bit", "4bit"],
        default="none",
        help="Quantization mode for Qwen2.5-VL.",
    )
    parser.add_argument(
        "--video-cpu-offload",
        action="store_true",
        help="Enable CPU offload for 8-bit VL loading.",
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
        help="Enable KV cache during video generation.",
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
        help="Maximum visual tokens for the video processor.",
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
        "--debate-max-rounds",
        type=int,
        default=3,
        help="Maximum number of debate rounds after the initial disagreement.",
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

    return DebateConfig(
        manifest_path=manifest_path,
        output_path=args.output.resolve(),
        data_root=data_root,
        audio_model_id=args.audio_model_id,
        video_model_id=args.video_model_id,
        deepseek_model_id=args.deepseek_model_id,
        glm_model_id=args.glm_model_id,
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
        debate_max_rounds=args.debate_max_rounds,
        metrics_output=args.metrics_output.resolve() if args.metrics_output else None,
        limit=args.limit,
        append_output=args.append_output,
    )


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


class MultiModalCueExtractor:
    def __init__(self, config: DebateConfig) -> None:
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
                    {"type": "text", "text": PROMPT1},
                ],
            },
        ]
        prompt_text = self.audio_processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False,
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
        return {
            "json": normalize_cue_json(response, "audio"),
            "raw_response": response,
        }

    def extract_video_cues(self, sample: Sample) -> Dict[str, Any]:
        if not sample.video_path.exists():
            raise FileNotFoundError(f"Video file not found: {sample.video_path}")

        empty_cuda_cache(self.config.video_device)
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "path": sample.video_path.as_posix()},
                    {"type": "text", "text": PROMPT2},
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

        return {
            "json": normalize_cue_json(response, "video"),
            "raw_response": response,
        }


class SequentialDebateReasoner:
    def __init__(self, config: DebateConfig) -> None:
        self.config = config
        self.qwen_dtype = torch_dtype_from_name(config.qwen_dtype)
        self.current_model_id: Optional[str] = None
        self.current_model_name: Optional[str] = None
        self.tokenizer: Any = None
        self.model: Any = None

    def unload_current_model(self) -> None:
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        self.current_model_id = None
        self.current_model_name = None
        gc.collect()
        empty_cuda_cache(self.config.reasoner_device)

    def ensure_model_loaded(self, spec: ModelSpec) -> None:
        if self.current_model_id == spec.model_id and self.model is not None and self.tokenizer is not None:
            return

        self.unload_current_model()
        self.tokenizer = AutoTokenizer.from_pretrained(
            spec.model_id,
            trust_remote_code=spec.trust_remote_code,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or self.tokenizer.unk_token
        self.model = AutoModelForCausalLM.from_pretrained(
            spec.model_id,
            torch_dtype=self.qwen_dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=spec.trust_remote_code,
        )
        self.model.to(self.config.reasoner_device)
        self.model.eval()
        self.current_model_id = spec.model_id
        self.current_model_name = spec.name

    def build_prompt_text(self, system_prompt: str, user_prompt: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except ValueError:
            return f"{system_prompt}\n\n{user_prompt}\n\n"

    def generate_text(
        self,
        spec: ModelSpec,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: Optional[int] = None,
    ) -> str:
        self.ensure_model_loaded(spec)
        prompt_text = self.build_prompt_text(system_prompt, user_prompt)
        model_inputs = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            padding=True,
        )
        model_inputs = move_batch_to_device(model_inputs, self.config.reasoner_device)

        with torch.inference_mode():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens or self.config.reasoner_max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        trimmed_ids = generated_ids[:, model_inputs["input_ids"].shape[1] :]
        return self.tokenizer.batch_decode(
            trimmed_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

    def infer(self, spec: ModelSpec, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        response = self.generate_text(spec, system_prompt, user_prompt)
        normalized = normalize_reasoner_output(response)
        return {
            "model_name": spec.name,
            "model_id": spec.model_id,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "raw_model_output": response,
            "model_output": normalized["tagged_output"],
            "think": normalized["think"],
            "answer": normalized["answer"],
            "normalized_answer": canonicalize_answer(normalized["answer"]),
        }

    def infer_text_assessment(self, spec: ModelSpec, text: str) -> Dict[str, Any]:
        response = self.generate_text(
            spec=spec,
            system_prompt=TEXT_GATE_PROMPT,
            user_prompt=text,
            max_new_tokens=128,
        )
        assessment = normalize_text_assessment(response)
        assessment["model_name"] = spec.name
        assessment["model_id"] = spec.model_id
        return assessment


class DebatePipeline:
    def __init__(self, config: DebateConfig) -> None:
        self.config = config
        self.cue_extractor = MultiModalCueExtractor(config)
        self.reasoner = SequentialDebateReasoner(config)
        self.deepseek_spec = ModelSpec(
            name="DeepSeek-R1-Distill-Qwen-7B",
            model_id=config.deepseek_model_id,
            trust_remote_code=False,
        )
        self.glm_spec = ModelSpec(
            name="GLM-4-9B-Chat",
            model_id=config.glm_model_id,
            trust_remote_code=True,
        )

    def build_initial_prompt(
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

    def assess_text_emotion(self, sample: Sample) -> Dict[str, Any]:
        text = sample.text.strip()
        if not text:
            return {
                "label": "",
                "confidence": 0.0,
                "clarity": 0.0,
                "reason": "No text was provided for text-first assessment.",
                "raw_response": "",
                "model_name": self.deepseek_spec.name,
                "model_id": self.deepseek_spec.model_id,
            }
        return self.reasoner.infer_text_assessment(self.deepseek_spec, text)

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

    def build_debate_prompt(
        self,
        base_prompt: str,
        own_result: Dict[str, Any],
        other_result: Dict[str, Any],
        debate_round: int,
    ) -> str:
        return (
            f"{base_prompt}\n\n"
            f"Debate round: {debate_round}\n\n"
            "Your previous reasoning and answer:\n"
            f"{own_result['model_output']}\n\n"
            "The other model's previous reasoning and answer:\n"
            f"{other_result['model_output']}\n\n"
            "Re-evaluate the evidence and debate context. "
            "You may keep or revise your answer, but you must output only the required tags."
        )

    def debate(
        self,
        sample: Sample,
        text_assessment: Dict[str, Any],
        audio_json: Dict[str, Any],
        video_json: Dict[str, Any],
        gate_report: Dict[str, Any],
    ) -> Dict[str, Any]:
        base_prompt = self.build_initial_prompt(sample, text_assessment, audio_json, video_json, gate_report)
        rounds: List[Dict[str, Any]] = []

        deepseek_result = self.reasoner.infer(self.deepseek_spec, INITIAL_SYSTEM_PROMPT, base_prompt)
        glm_result = self.reasoner.infer(self.glm_spec, INITIAL_SYSTEM_PROMPT, base_prompt)
        rounds.append(
            {
                "round_type": "initial",
                "round_index": 0,
                "deepseek": deepseek_result,
                "glm": glm_result,
                "agreement": deepseek_result["normalized_answer"] == glm_result["normalized_answer"],
            }
        )

        if deepseek_result["normalized_answer"] == glm_result["normalized_answer"]:
            return {
                "termination_reason": "initial_agreement",
                "consensus_reached": True,
                "consensus_round": 0,
                "final_model": "consensus",
                "final_result": glm_result,
                "rounds": rounds,
            }

        current_deepseek = deepseek_result
        current_glm = glm_result
        for debate_round in range(1, self.config.debate_max_rounds + 1):
            deepseek_prompt = self.build_debate_prompt(
                base_prompt=base_prompt,
                own_result=current_deepseek,
                other_result=current_glm,
                debate_round=debate_round,
            )
            glm_prompt = self.build_debate_prompt(
                base_prompt=base_prompt,
                own_result=current_glm,
                other_result=current_deepseek,
                debate_round=debate_round,
            )
            current_deepseek = self.reasoner.infer(
                self.deepseek_spec,
                DEBATE_SYSTEM_PROMPT,
                deepseek_prompt,
            )
            current_glm = self.reasoner.infer(
                self.glm_spec,
                DEBATE_SYSTEM_PROMPT,
                glm_prompt,
            )
            agreement = current_deepseek["normalized_answer"] == current_glm["normalized_answer"]
            rounds.append(
                {
                    "round_type": "debate",
                    "round_index": debate_round,
                    "deepseek": current_deepseek,
                    "glm": current_glm,
                    "agreement": agreement,
                }
            )
            if agreement:
                return {
                    "termination_reason": "debate_agreement",
                    "consensus_reached": True,
                    "consensus_round": debate_round,
                    "final_model": "consensus",
                    "final_result": current_glm,
                    "rounds": rounds,
                }

        return {
            "termination_reason": "max_rounds_glm_fallback",
            "consensus_reached": False,
            "consensus_round": None,
            "final_model": "GLM-4-9B-Chat",
            "final_result": current_glm,
            "rounds": rounds,
        }

    def process_sample(self, sample: Sample) -> Dict[str, Any]:
        audio_result = self.cue_extractor.extract_audio_cues(sample)
        video_result = self.cue_extractor.extract_video_cues(sample)
        text_assessment = self.assess_text_emotion(sample)
        gate_report = self.compute_modality_gate(
            text_assessment=text_assessment,
            audio_json=audio_result["json"],
            video_json=video_result["json"],
        )
        debate_result = self.debate(
            sample,
            text_assessment,
            audio_result["json"],
            video_result["json"],
            gate_report,
        )
        final_result = debate_result["final_result"]
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
            "debate_max_rounds": self.config.debate_max_rounds,
            "debate_history": debate_result["rounds"],
            "termination_reason": debate_result["termination_reason"],
            "consensus_reached": debate_result["consensus_reached"],
            "consensus_round": debate_result["consensus_round"],
            "selected_model": debate_result["final_model"],
            "model_output": final_result["model_output"],
            "raw_model_output": final_result["raw_model_output"],
            "emotion_prediction": final_result["answer"],
            "answer": final_result["answer"],
            "think": final_result["think"],
            "prompt": {
                "system": final_result["system_prompt"],
                "user": final_result["user_prompt"],
            },
            "label": sample.label,
            "meta": sample.meta,
        }


def iter_samples(samples: Iterable[Sample]) -> Iterable[Sample]:
    for sample in samples:
        yield sample


def main() -> None:
    config = parse_args()
    samples = load_samples(config.manifest_path, config.data_root, config.limit)
    if not samples:
        raise ValueError(f"No samples found in {config.manifest_path}")

    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    results = read_existing_results(config.output_path) if config.append_output else []
    processed_ids = {str(item.get("id", "")).strip() for item in results if str(item.get("id", "")).strip()}
    if processed_ids:
        samples = [sample for sample in samples if sample.sample_id not in processed_ids]
        print(f"Skipping {len(processed_ids)} already processed samples from existing output.")
    if not samples:
        print(f"No remaining samples to process. Existing results kept at {config.output_path}")
        return
    pipeline = DebatePipeline(config)

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
                "debate_max_rounds": config.debate_max_rounds,
                "debate_history": [],
                "termination_reason": f"failure: {exc}",
                "consensus_reached": False,
                "consensus_round": None,
                "selected_model": "",
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

    pipeline.reasoner.unload_current_model()
    if config.metrics_output is not None:
        write_metrics_report(config.metrics_output, results)
    print(f"Finished. Results written to {config.output_path}")


if __name__ == "__main__":
    main()
