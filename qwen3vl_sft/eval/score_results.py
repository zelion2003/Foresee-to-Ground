#!/usr/bin/env python3
"""Score temporal grounding predictions for the public Qwen3-VL SFT baseline."""

from __future__ import annotations

import argparse
import json
import re
from typing import List, Optional, Tuple


def parse_span_from_response(text: str, mode: str, sec_per_index: float) -> Optional[Tuple[float, float]]:
    """Parse a temporal span from raw model text and always return seconds."""
    if text is None:
        return None
    text = str(text).lower()
    numbers = re.findall(r"\d+(?:\.\d+)?", text)

    if mode == "frame":
        match = re.search(r"from\s*(?:frame\s*)?(\d+(?:\.\d+)?)\s*(?:to|-|~)\s*(?:frame\s*)?(\d+(?:\.\d+)?)", text)
        if match:
            start_idx, end_idx = float(match.group(1)), float(match.group(2))
        elif len(numbers) >= 2:
            start_idx, end_idx = float(numbers[0]), float(numbers[1])
        else:
            return None
        return min(start_idx, end_idx) * sec_per_index, max(start_idx, end_idx) * sec_per_index

    match = re.search(r"from\s*(\d+(?:\.\d+)?)\s*(?:s|sec|seconds)?\s*(?:to|-|~)\s*(\d+(?:\.\d+)?)", text)
    if match:
        start_sec, end_sec = float(match.group(1)), float(match.group(2))
    elif len(numbers) >= 2:
        start_sec, end_sec = float(numbers[0]), float(numbers[1])
    else:
        return None
    return min(start_sec, end_sec), max(start_sec, end_sec)


def iou_seconds(pred: Tuple[float, float], gt: Tuple[float, float]) -> float:
    """Compute temporal IoU in seconds."""
    pred_start, pred_end = pred
    gt_start, gt_end = gt
    inter = max(0.0, min(pred_end, gt_end) - max(pred_start, gt_start))
    union = max(pred_end, gt_end) - min(pred_start, gt_start)
    return inter / union if union > 0 else 0.0


def evaluate(predictions: List[dict], mode: str, sec_per_index: float, write_back: bool, path: str) -> None:
    """Evaluate prediction records and optionally write normalized fields back to disk."""
    ious: List[float] = []
    for sample in predictions:
        if sample.get("pred_start_sec") is not None and sample.get("pred_end_sec") is not None:
            pred_start = float(sample["pred_start_sec"])
            pred_end = float(sample["pred_end_sec"])
        else:
            span = parse_span_from_response(sample.get("response", ""), mode, sec_per_index)
            pred_start, pred_end = span if span else (0.0, 0.0)
            sample["pred_start_sec"] = pred_start
            sample["pred_end_sec"] = pred_end

        gt_start = sample.get("gt_start")
        gt_end = sample.get("gt_end")
        if gt_start is None or gt_end is None:
            sample["iou"] = 0.0
            ious.append(0.0)
            continue

        score = iou_seconds((pred_start, pred_end), (float(gt_start), float(gt_end)))
        sample["iou"] = score
        ious.append(score)

    denom = len(ious) if ious else 1
    r03 = sum(iou >= 0.3 for iou in ious) / denom * 100.0
    r05 = sum(iou >= 0.5 for iou in ious) / denom * 100.0
    r07 = sum(iou >= 0.7 for iou in ious) / denom * 100.0
    miou = sum(ious) / denom * 100.0

    print(f"Total samples: {len(predictions)}")
    print(f"R@0.3 = {r03:.2f}")
    print(f"R@0.5 = {r05:.2f}")
    print(f"R@0.7 = {r07:.2f}")
    print(f"mIoU  = {miou:.2f}")

    if write_back:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(predictions, handle, indent=2)
        print(f"[saved] wrote normalized predictions back to {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Score Qwen3-VL baseline temporal grounding predictions.")
    parser.add_argument("--pred_json", required=True, help="Path to the prediction JSON file.")
    parser.add_argument("--mode", choices=["frame", "seconds"], default="seconds", help="Expected answer format produced by the model.")
    parser.add_argument("--sec_per_index", type=float, default=1.0, help="Frame-to-seconds ratio when mode=frame.")
    parser.add_argument("--write_back", action="store_true", help="Write normalized prediction fields and IoU values back to the JSON file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.pred_json, "r", encoding="utf-8") as handle:
        predictions = json.load(handle)
    evaluate(
        predictions=predictions,
        mode=args.mode,
        sec_per_index=args.sec_per_index,
        write_back=args.write_back,
        path=args.pred_json,
    )


if __name__ == "__main__":
    main()
