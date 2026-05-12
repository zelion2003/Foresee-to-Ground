#!/usr/bin/env python3
"""Run public evaluation for the plain Qwen3-VL SFT baseline."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import List, Optional, Tuple

import torch

try:
    from torch.utils import _pytree as _torch_pytree  # type: ignore

    if not hasattr(torch.utils, "register_pytree_node"):
        def register_pytree_node(node_type, flatten_fn, unflatten_fn, serialized_type_name=None, *args, **kwargs):
            return _torch_pytree._register_pytree_node(node_type, flatten_fn, unflatten_fn)

        torch.utils.register_pytree_node = register_pytree_node  # type: ignore[attr-defined]
        _torch_pytree.register_pytree_node = register_pytree_node  # type: ignore[attr-defined]
except Exception:
    pass

if not hasattr(torch, "compiler"):
    class _DummyCompiler:
        @staticmethod
        def is_compiling():
            return False

    torch.compiler = _DummyCompiler()  # type: ignore[attr-defined]
elif not hasattr(torch.compiler, "is_compiling"):
    torch.compiler.is_compiling = lambda: False  # type: ignore[attr-defined]

from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVAL_JSON = REPO_ROOT / "data" / "annotations" / "timelens" / "charades_timelens_test.json"
DEFAULT_VIDEO_ROOT = REPO_ROOT / "data" / "videos" / "timelens_bench_336"
DEFAULT_OUTPUT_JSON = REPO_ROOT / "outputs" / "eval" / "qwen3vl_sft_eval.json"


def load_model_and_processor(
    model_id: str,
    torch_dtype: str = "auto",
    device_map: str = "auto",
    attn_implementation: str = "flash_attention_2",
    local_files_only: bool = True,
):
    """Load the Qwen3-VL model and processor for inference."""
    if torch_dtype == "bf16":
        dtype = torch.bfloat16
    elif torch_dtype == "fp16":
        dtype = torch.float16
    else:
        dtype = "auto"

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map=device_map,
        attn_implementation=attn_implementation,
        local_files_only=local_files_only,
    )
    processor = AutoProcessor.from_pretrained(model_id, local_files_only=local_files_only)
    return model, processor


def extract_span_seconds(text: str, mode: str, sec_per_index: float = 1.0) -> Optional[Tuple[float, float]]:
    """Parse a predicted temporal span and always return seconds."""
    text = text.strip().lower()

    if mode == "frame":
        pattern = r"from\s*(?:frame\s*)?(\d+(?:\.\d+)?)\s*(?:to|-|~)\s*(?:frame\s*)?(\d+(?:\.\d+)?)"
        match = re.search(pattern, text)
        if not match:
            numbers = re.findall(r"\d+(?:\.\d+)?", text)
            if len(numbers) < 2:
                return None
            start_idx, end_idx = float(numbers[0]), float(numbers[1])
        else:
            start_idx, end_idx = float(match.group(1)), float(match.group(2))
        return min(start_idx, end_idx) * sec_per_index, max(start_idx, end_idx) * sec_per_index

    pattern = r"from\s*(\d+(?:\.\d+)?)\s*(?:s|sec|seconds)?\s*(?:to|-|~)\s*(\d+(?:\.\d+)?)"
    match = re.search(pattern, text)
    if not match:
        numbers = re.findall(r"\d+(?:\.\d+)?", text)
        if len(numbers) < 2:
            return None
        start_sec, end_sec = float(numbers[0]), float(numbers[1])
    else:
        start_sec, end_sec = float(match.group(1)), float(match.group(2))
    return min(start_sec, end_sec), max(start_sec, end_sec)


def calc_iou_seconds(pred: Tuple[float, float], gt: Tuple[float, float]) -> float:
    """Compute temporal IoU in seconds."""
    pred_start, pred_end = pred
    gt_start, gt_end = gt
    inter = max(0.0, min(pred_end, gt_end) - max(pred_start, gt_start))
    union = max(pred_end, gt_end) - min(pred_start, gt_start)
    if union <= 0:
        return 0.0
    return inter / union


def run_eval(
    model,
    processor,
    data_path: str,
    video_root: str,
    save_path: str,
    prompt_template: str,
    mode: str,
    sec_per_index: float = 1.0,
    max_new_tokens: int = 128,
    sample_limit: Optional[int] = None,
    system_prompt: Optional[str] = None,
) -> List[dict]:
    """Run evaluation on a flat JSON benchmark and save predictions incrementally."""
    with open(data_path, "r", encoding="utf-8") as handle:
        items = json.load(handle)

    results: List[dict] = []
    if os.path.exists(save_path):
        with open(save_path, "r", encoding="utf-8") as handle:
            try:
                results = json.load(handle)
            except Exception:
                results = []
    done_ids = {record["id"] for record in results}

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    for idx, sample in enumerate(items):
        if sample_limit is not None and idx >= sample_limit:
            break
        if sample["id"] in done_ids:
            continue

        video_path = os.path.join(video_root, sample["video"])
        if not os.path.exists(video_path):
            print(f"[warn][missing video] {video_path}")
            continue

        prompt = prompt_template.format(sample["query"])
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": video_path},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        if system_prompt:
            messages.insert(0, {"role": "system", "content": [{"type": "text", "text": system_prompt}]})

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], videos=[video_path], padding=True, return_tensors="pt")
        inputs = {key: value.to(model.device) for key, value in inputs.items()}

        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=max_new_tokens)
        trimmed = [out[len(inp):] for inp, out in zip(inputs["input_ids"], generated)]
        decoded = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        response = decoded[0].strip()

        span = extract_span_seconds(response, mode=mode, sec_per_index=sec_per_index)
        pred_start, pred_end = span if span else (0.0, 0.0)

        gt_start = sample.get("start_time")
        gt_end = sample.get("end_time")
        iou = 0.0
        if gt_start is not None and gt_end is not None and span is not None:
            iou = calc_iou_seconds((pred_start, pred_end), (float(gt_start), float(gt_end)))

        record = {
            "id": sample["id"],
            "video": sample.get("video"),
            "query": sample.get("query"),
            "response": response,
            "pred_start_sec": pred_start,
            "pred_end_sec": pred_end,
            "gt_start": gt_start,
            "gt_end": gt_end,
            "duration": sample.get("duration"),
            "iou": iou,
        }
        results.append(record)

        with open(save_path, "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)
        print(f"[{len(results)}/{len(items)}] {sample['id']} -> IoU={iou:.3f}")

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Evaluate the public Qwen3-VL SFT baseline.")
    parser.add_argument("--model_id", default="Qwen/Qwen3-VL-8B-Instruct", help="Model identifier or local checkpoint path.")
    parser.add_argument("--data_path", default=str(DEFAULT_EVAL_JSON), help="Path to the evaluation JSON file.")
    parser.add_argument("--video_root", default=str(DEFAULT_VIDEO_ROOT), help="Root directory that contains the evaluation videos.")
    parser.add_argument("--save_path", default=str(DEFAULT_OUTPUT_JSON), help="Path used to save predictions incrementally.")
    parser.add_argument(
        "--prompt_template",
        default="During which time can we see {}? Answer in seconds using the format: from X seconds to Y seconds.",
        help="Prompt template that receives the query via Python format syntax.",
    )
    parser.add_argument("--mode", choices=["seconds", "frame"], default="seconds", help="Expected answer format from the model.")
    parser.add_argument("--sec_per_index", type=float, default=1.0, help="Frame-to-seconds ratio when mode=frame.")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Maximum generated token count.")
    parser.add_argument("--sample_limit", type=int, default=None, help="Optional cap on the number of evaluated samples.")
    parser.add_argument("--system_prompt", default=None, help="Optional system prompt.")
    parser.add_argument("--torch_dtype", choices=["auto", "bf16", "fp16"], default="bf16", help="Torch dtype used for loading the model.")
    parser.add_argument("--device_map", default="auto", help='Device map, for example "auto" or "cuda:0".')
    parser.add_argument("--attn_implementation", default="flash_attention_2", help="Attention backend passed to the model loader.")
    parser.add_argument("--local_files_only", action="store_true", help="Only load local model files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model, processor = load_model_and_processor(
        model_id=args.model_id,
        torch_dtype=args.torch_dtype,
        device_map=args.device_map,
        attn_implementation=args.attn_implementation,
        local_files_only=args.local_files_only,
    )
    model.eval()
    run_eval(
        model=model,
        processor=processor,
        data_path=args.data_path,
        video_root=args.video_root,
        save_path=args.save_path,
        prompt_template=args.prompt_template,
        mode=args.mode,
        sec_per_index=args.sec_per_index,
        max_new_tokens=args.max_new_tokens,
        sample_limit=args.sample_limit,
        system_prompt=args.system_prompt,
    )


if __name__ == "__main__":
    main()
