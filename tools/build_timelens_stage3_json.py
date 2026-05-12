#!/usr/bin/env python3


from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "data" / "raw" / "timelens" / "timelens-100k.jsonl"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "annotations" / "timelens" / "timelens_100k_stage3.json"


def format_seconds(value: float) -> str:
    text = f"{float(value):.6f}".rstrip("0").rstrip(".")
    if "." not in text:
        text = f"{text}.0"
    return text


def build_answer(start: float, end: float) -> str:
    return f"from {format_seconds(start)} seconds to {format_seconds(end)} seconds"


def make_sample(source: str, video_path: str, duration: float, event_idx: int, query: str, start: float, end: float) -> dict[str, Any]:
    video_stem = Path(video_path).stem
    return {
        "id": f"{source}__{video_stem}__{event_idx}",
        "source": source,
        "video": video_stem,
        "duration": float(duration),
        "question": query.strip(),
        "answer": build_answer(start, end),
        "spans": [{"start": float(start), "end": float(end)}],
    }


def convert_jsonl_to_stage3_json(input_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    input_path = Path(input_path)
    output_path = Path(output_path)

    converted: list[dict[str, Any]] = []
    source_counts: Counter[str] = Counter()
    skipped_events = 0
    videos = 0

    with input_path.open("r", encoding="utf-8") as infile:
        for line_no, line in enumerate(infile, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            videos += 1

            source = str(record.get("source", "")).strip()
            video_path = str(record.get("video_path", "")).strip()
            duration = float(record.get("duration", 0.0) or 0.0)
            events = record.get("events", [])
            if not source or not video_path or duration <= 0 or not isinstance(events, list):
                skipped_events += len(events) if isinstance(events, list) else 0
                continue

            for event_idx, event in enumerate(events):
                query = str(event.get("query", "")).strip()
                spans = event.get("span", [])
                if not query or not isinstance(spans, list) or not spans:
                    skipped_events += 1
                    continue
                first_span = spans[0]
                if not isinstance(first_span, list) or len(first_span) != 2:
                    skipped_events += 1
                    continue
                start, end = float(first_span[0]), float(first_span[1])
                if end <= start:
                    skipped_events += 1
                    continue

                converted.append(
                    make_sample(
                        source=source,
                        video_path=video_path,
                        duration=duration,
                        event_idx=event_idx,
                        query=query,
                        start=start,
                        end=end,
                    )
                )
                source_counts[source] += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as outfile:
        json.dump(converted, outfile, ensure_ascii=False, indent=2)

    return {
        "videos": videos,
        "samples": len(converted),
        "skipped_events": skipped_events,
        "source_counts": dict(sorted(source_counts.items())),
        "output_path": str(output_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Path to TimeLens-100K JSONL annotations.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Path to write the converted Stage-3 JSON.")
    parser.add_argument("--preview", type=int, default=3, help="How many converted samples to print after conversion.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = convert_jsonl_to_stage3_json(args.input, args.output)

    print(f"Converted videos: {stats['videos']}")
    print(f"Converted samples: {stats['samples']}")
    print(f"Skipped events: {stats['skipped_events']}")
    print("Samples per source:")
    for source, count in stats["source_counts"].items():
        print(f"  {source}: {count}")

    preview_count = max(0, int(args.preview))
    if preview_count:
        converted = json.loads(args.output.read_text(encoding="utf-8"))
        print(f"Previewing first {min(preview_count, len(converted))} samples:")
        for sample in converted[:preview_count]:
            print(json.dumps(sample, ensure_ascii=False))


if __name__ == "__main__":
    main()
