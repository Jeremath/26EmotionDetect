#!/usr/bin/env python3
"""
Run modal ablation experiments on top of the single-model pipeline.

Ablation settings:
1. without_text: use only audio + video cues.
2. without_audio: use text + video cues.
3. without_video: use text + audio cues.
4. text_only: use text only.

The script reuses the existing multimodal_emotion_pipeline implementation so
that all conditions share the same cue extractor and final reasoner.
"""

from __future__ import annotations

import argparse
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from multimodal_emotion_pipeline import (
    MultiModalEmotionPipeline,
    PipelineConfig,
    Sample,
    compute_metrics,
    load_samples,
)


@dataclass(frozen=True)
class AblationSpec:
    key: str
    description: str
    use_text: bool
    use_audio: bool
    use_video: bool


ABLATION_SPECS: List[AblationSpec] = [
    AblationSpec(
        key="without_text",
        description="Remove text; keep audio and video.",
        use_text=False,
        use_audio=True,
        use_video=True,
    ),
    AblationSpec(
        key="without_audio",
        description="Remove audio; keep text and video.",
        use_text=True,
        use_audio=False,
        use_video=True,
    ),
    AblationSpec(
        key="without_video",
        description="Remove video; keep text and audio.",
        use_text=True,
        use_audio=True,
        use_video=False,
    ),
    AblationSpec(
        key="text_only",
        description="Use text only.",
        use_text=True,
        use_audio=False,
        use_video=False,
    ),
]


@dataclass
class ModalAblationConfig:
    pipeline: PipelineConfig
    summary_output: Path


def parse_args() -> ModalAblationConfig:
    parser = argparse.ArgumentParser(
        description="Run modal ablation experiments with the single-model pipeline."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/samples.jsonl"),
        help="Path to the input JSONL manifest.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("output/Modal-ablation.txt"),
        help="Path to the ablation summary report.",
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
        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        help="Final multimodal emotion reasoning model.",
    )
    parser.add_argument("--audio-device", default="cuda:0", help="Device for audio model.")
    parser.add_argument("--video-device", default="cuda:1", help="Device for video model.")
    parser.add_argument(
        "--reasoner-device",
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
        "--limit",
        type=int,
        default=None,
        help="Only process the first N samples.",
    )
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    data_root = args.data_root.resolve() if args.data_root else manifest_path.parent
    pipeline_config = PipelineConfig(
        manifest_path=manifest_path,
        output_path=Path("unused.json").resolve(),
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
        metrics_output=None,
        limit=args.limit,
        append_output=False,
    )
    return ModalAblationConfig(
        pipeline=pipeline_config,
        summary_output=args.summary_output.resolve(),
    )


def empty_audio_result() -> Dict[str, Any]:
    return {"json": {"audio": []}, "raw_response": ""}


def empty_video_result() -> Dict[str, Any]:
    return {"json": {"video": []}, "raw_response": ""}


def build_case_sample(sample: Sample, spec: AblationSpec) -> Sample:
    return Sample(
        sample_id=sample.sample_id,
        text=sample.text if spec.use_text else "",
        audio_path=sample.audio_path,
        video_path=sample.video_path,
        label=sample.label,
        meta=sample.meta,
    )


def build_case_result(
    sample: Sample,
    spec: AblationSpec,
    reasoner_result: Dict[str, Any],
    audio_result: Dict[str, Any],
    video_result: Dict[str, Any],
) -> Dict[str, Any]:
    used_text = sample.text if spec.use_text else ""
    used_audio_json = audio_result["json"] if spec.use_audio else {"audio": []}
    used_video_json = video_result["json"] if spec.use_video else {"video": []}
    return {
        "id": sample.sample_id,
        "ablation_case": spec.key,
        "description": spec.description,
        "modalities": {
            "text": spec.use_text,
            "audio": spec.use_audio,
            "video": spec.use_video,
        },
        "label": sample.label,
        "answer": reasoner_result["answer"],
        "emotion_prediction": reasoner_result["answer"],
        "think": reasoner_result["think"],
        "model_output": reasoner_result["tagged_output"],
        "raw_model_output": reasoner_result["raw_response"],
        "text_assessment": reasoner_result.get("text_assessment", {}),
        "modality_gate": reasoner_result.get("gate_report", {}),
        "emotion_cues": {
            "text": used_text,
            "audio": used_audio_json.get("audio", []),
            "video": used_video_json.get("video", []),
        },
        "audio_cues": used_audio_json,
        "video_cues": used_video_json,
        "audio_raw_response": audio_result["raw_response"] if spec.use_audio else "",
        "video_raw_response": video_result["raw_response"] if spec.use_video else "",
        "prompt": reasoner_result["prompt"],
        "meta": sample.meta,
    }


def build_failure_result(sample: Sample, spec: AblationSpec, error: str) -> Dict[str, Any]:
    return {
        "id": sample.sample_id,
        "ablation_case": spec.key,
        "description": spec.description,
        "modalities": {
            "text": spec.use_text,
            "audio": spec.use_audio,
            "video": spec.use_video,
        },
        "label": sample.label,
        "answer": "",
        "emotion_prediction": "",
        "think": f"处理失败，原因是：{error}",
        "model_output": "",
        "raw_model_output": "",
        "text_assessment": {},
        "modality_gate": {},
        "emotion_cues": {
            "text": sample.text if spec.use_text else "",
            "audio": [],
            "video": [],
        },
        "audio_cues": {"audio": []},
        "video_cues": {"video": []},
        "audio_raw_response": "",
        "video_raw_response": "",
        "prompt": {},
        "meta": sample.meta,
    }


def with_empty_modality(modality_name: str) -> Dict[str, Any]:
    return {
        modality_name: [],
        "quality": 0.0,
        "confidence": 0.0,
        "ambiguity": 1.0,
        "signal_strength": 0.0,
        "recommended_use": False,
        "reason": f"{modality_name} removed by modal ablation.",
    }


def render_metrics_block(title: str, metrics: Optional[Dict[str, Any]]) -> List[str]:
    lines = [title]
    if metrics is None:
        lines.append("No labeled predictions were available, so no metrics were computed.")
        return lines

    lines.extend(
        [
            f"Evaluated samples: {metrics['evaluated_samples']}",
            f"Accuracy: {metrics['accuracy']:.6f}",
            f"Weighted Precision: {metrics['weighted_precision']:.6f}",
            f"Weighted Recall: {metrics['weighted_recall']:.6f}",
            f"Weighted F1: {metrics['weighted_f1']:.6f}",
            f"Macro Precision: {metrics['macro_precision']:.6f}",
            f"Macro Recall: {metrics['macro_recall']:.6f}",
            f"Macro F1: {metrics['macro_f1']:.6f}",
            "Per-label metrics:",
        ]
    )
    for item in metrics["per_label"]:
        lines.append(
            f"- {item['label']}: support={item['support']}, "
            f"precision={item['precision']:.6f}, "
            f"recall={item['recall']:.6f}, "
            f"f1={item['f1']:.6f}"
        )
    return lines


def write_summary(
    summary_path: Path,
    results_by_case: Dict[str, List[Dict[str, Any]]],
    processed_samples: int,
    total_samples: int,
) -> None:
    lines = [
        "Modal Ablation Report",
        "",
        f"Processed samples: {processed_samples}/{total_samples}",
        "",
        "Related-work note:",
        "MELD papers commonly report Accuracy and Weighted F1 as the primary metrics.",
        "Examples:",
        "- Frontiers 2023 GCF2-Net reports improvements on MELD in terms of accuracy and weighted average F1.",
        "  https://www.frontiersin.org/articles/10.3389/fnins.2023.1183132/full",
        "- Adaptive weighting in a transformer framework for multimodal emotion recognition reports MELD accuracy and weighted F1 in Table 3.",
        "  https://www.sciencedirect.com/science/article/pii/S0167639325001475",
        "",
    ]
    for index, spec in enumerate(ABLATION_SPECS, start=1):
        metrics = compute_metrics(results_by_case[spec.key])
        lines.extend(
            render_metrics_block(
                title=f"[{index}] {spec.key}: {spec.description}",
                metrics=metrics,
            )
        )
        lines.append("")

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


class ModalAblationExperiment:
    def __init__(self, config: ModalAblationConfig) -> None:
        self.config = config
        self.pipeline = MultiModalEmotionPipeline(config.pipeline)

    def extract_modal_cues(self, sample: Sample) -> tuple[Dict[str, Any], Dict[str, Any]]:
        audio_result = empty_audio_result()
        video_result = empty_video_result()

        try:
            audio_result = self.pipeline.extract_audio_cues(sample)
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            audio_result = {
                "json": {"audio": []},
                "raw_response": f"Audio extraction failed: {exc}",
            }

        try:
            video_result = self.pipeline.extract_video_cues(sample)
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            video_result = {
                "json": {"video": []},
                "raw_response": f"Video extraction failed: {exc}",
            }

        return audio_result, video_result

    def run_case(
        self,
        sample: Sample,
        spec: AblationSpec,
        audio_result: Dict[str, Any],
        video_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        working_sample = build_case_sample(sample, spec)
        used_audio_json = audio_result["json"] if spec.use_audio else with_empty_modality("audio")
        used_video_json = video_result["json"] if spec.use_video else with_empty_modality("video")
        try:
            text_assessment = self.pipeline.assess_text_emotion(working_sample)
            gate_report = self.pipeline.compute_modality_gate(
                text_assessment=text_assessment,
                audio_json=used_audio_json,
                video_json=used_video_json,
            )
            reasoner_result = self.pipeline.classify_emotion(
                sample=working_sample,
                text_assessment=text_assessment,
                audio_json=used_audio_json,
                video_json=used_video_json,
                gate_report=gate_report,
            )
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            return build_failure_result(sample, spec, str(exc))

        return build_case_result(
            sample=sample,
            spec=spec,
            reasoner_result=reasoner_result,
            audio_result=audio_result,
            video_result=video_result,
        )


def iter_samples(samples: Iterable[Sample]) -> Iterable[Sample]:
    for sample in samples:
        yield sample


def main() -> None:
    config = parse_args()
    samples = load_samples(
        config.pipeline.manifest_path,
        config.pipeline.data_root,
        config.pipeline.limit,
    )
    if not samples:
        raise ValueError(f"No samples found in {config.pipeline.manifest_path}")

    results_by_case: Dict[str, List[Dict[str, Any]]] = {spec.key: [] for spec in ABLATION_SPECS}
    experiment = ModalAblationExperiment(config)

    for index, sample in enumerate(iter_samples(samples), start=1):
        print(f"[{index}/{len(samples)}] Processing sample: {sample.sample_id}")
        audio_result, video_result = experiment.extract_modal_cues(sample)

        for spec in ABLATION_SPECS:
            result = experiment.run_case(sample, spec, audio_result, video_result)
            results_by_case[spec.key].append(result)

        write_summary(
            summary_path=config.summary_output,
            results_by_case=results_by_case,
            processed_samples=index,
            total_samples=len(samples),
        )

    print(f"Finished. Summary written to {config.summary_output}")


if __name__ == "__main__":
    main()
