#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import json
import re
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = str(REPO_ROOT / "data" / "annotations" / "training_sft.json")
OUTPUT_PATH = str(REPO_ROOT / "data" / "annotations" / "qwen_training_sft.json")
FRAME_TO_SEC = 2.0

FRAME_RANGE_RE = re.compile(
    r"\bfrom\s+([0-9]+(?:\.[0-9]+)?)\s+to\s+([0-9]+(?:\.[0-9]+)?)\b",
    re.IGNORECASE,
)
META_TOKEN_RE = re.compile(r"<(s|e)(\d+)>")


def strip_image_token(text: str) -> str:
    
    return text.replace("<image>", "").strip()


def convert_frame_ranges_to_seconds(text: str, frame_to_sec: float = 2.0) -> str:
    
    def _repl(match: re.Match) -> str:
        start_frame = float(match.group(1))
        end_frame = float(match.group(2))
        start_sec = start_frame * frame_to_sec
        end_sec = end_frame * frame_to_sec
        return f"from {start_sec:.1f} seconds to {end_sec:.1f} seconds"

    return FRAME_RANGE_RE.sub(_repl, text)


def replace_meta_tokens_with_seconds(text: str, token_map: Dict[str, Any]) -> str:
    
    def _repl(match: re.Match) -> str:
        kind = match.group(1)   # 's' or 'e'
        idx = match.group(2)    # '0', '1', ...
        key = f"<{kind}{idx}>"
        if key in token_map:
            try:
                sec = float(token_map[key])
            except (TypeError, ValueError):
                return match.group(0)
            return f"{sec:.1f} seconds"
        return match.group(0)

    return META_TOKEN_RE.sub(_repl, text)


def is_vtg_pair(q_text: str, a_text: str) -> bool:
    
    if FRAME_RANGE_RE.search(a_text):
        return True
    if META_TOKEN_RE.search(q_text) or META_TOKEN_RE.search(a_text):
        return True
    return False


def process_item(item: Dict[str, Any], frame_to_sec: float = 2.0) -> List[Dict[str, Any]]:
    
    conversations = item.get("conversations", [])
    meta = item.get("meta") or {}
    token_map = meta.get("token") or {}

    base_id = item.get("id", "")
    video = item.get("video", "")
    source = item.get("source", "")
    duration = meta.get("duration", None)

    single_turn_samples: List[Dict[str, Any]] = []
    pair_idx = 0
    for i in range(0, len(conversations) - 1, 2):
        human = conversations[i]
        gpt = conversations[i + 1]
        if human.get("from") != "human" or gpt.get("from") != "gpt":
            continue

        q_raw = human.get("value", "") or ""
        a_raw = gpt.get("value", "") or ""
        if not is_vtg_pair(q_raw, a_raw):
            continue
        q_clean = strip_image_token(q_raw)
        q_clean = convert_frame_ranges_to_seconds(q_clean, frame_to_sec=frame_to_sec)
        q_clean = replace_meta_tokens_with_seconds(q_clean, token_map)
        a_clean = convert_frame_ranges_to_seconds(a_raw, frame_to_sec=frame_to_sec)
        a_clean = replace_meta_tokens_with_seconds(a_clean, token_map)

        sample_id = f"{base_id}_q{pair_idx}"
        pair_idx += 1

        single_turn_samples.append(
            {
                "id": sample_id,
                "source": source,
                "video": video,
                "duration": duration,
                "question": q_clean,
                "answer": a_clean,
            }
        )

    return single_turn_samples


def main():
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of items.")

    out_samples: List[Dict[str, Any]] = []
    total_items = 0
    total_pairs = 0
    total_vtg_pairs = 0

    for item in data:
        total_items += 1
        convs = item.get("conversations", []) or []
        total_pairs += len(convs) // 2

        samples = process_item(item, frame_to_sec=FRAME_TO_SEC)
        total_vtg_pairs += len(samples)
        out_samples.extend(samples)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out_samples, f, ensure_ascii=False, indent=2)

    print(f"[INFO] Input file        : {INPUT_PATH}")
    print(f"[INFO] Processed items   : {total_items}")
    print(f"[INFO] Total QA pairs    : {total_pairs}")
    print(f"[INFO] VTG QA extracted  : {total_vtg_pairs}")
    print(f"[INFO] Output written to : {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
