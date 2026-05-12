#!/usr/bin/env python3


from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INPUTS = {
    "activitynet": REPO_ROOT / "data" / "raw" / "timelens_bench" / "activitynet-timelens.json",
    "charades": REPO_ROOT / "data" / "raw" / "timelens_bench" / "charades-timelens.json",
}

DEFAULT_OUTPUTS = {
    "activitynet": REPO_ROOT / "data" / "annotations" / "timelens" / "activitynet_timelens_test.json",
    "charades": REPO_ROOT / "data" / "annotations" / "timelens" / "charades_timelens_test.json",
}


def convert_bench_json_to_test_json(input_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    input_path = Path(input_path)
    output_path = Path(output_path)

    with input_path.open("r", encoding="utf-8") as infile:
        data = json.load(infile)

    converted: list[dict[str, Any]] = []
    skipped_pairs = 0

    for video_id, payload in data.items():
        duration = float(payload.get("duration", 0.0) or 0.0)
        queries = payload.get("queries", [])
        spans = payload.get("spans", [])
        pair_count = min(len(queries), len(spans))

        for idx in range(pair_count):
            query = str(queries[idx]).strip()
            span = spans[idx]
            if not query or not isinstance(span, list) or len(span) != 2:
                skipped_pairs += 1
                continue

            start_time = float(span[0])
            end_time = float(span[1])
            if end_time <= start_time:
                skipped_pairs += 1
                continue

            converted.append(
                {
                    "id": f"{video_id}_{idx}",
                    "video": f"{video_id}.mp4",
                    "start_time": start_time,
                    "end_time": end_time,
                    "query": query,
                    "duration": duration,
                }
            )

        skipped_pairs += max(len(queries), len(spans)) - pair_count

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as outfile:
        json.dump(converted, outfile, ensure_ascii=False, indent=2)

    return {
        "videos": len(data),
        "samples": len(converted),
        "skipped_pairs": skipped_pairs,
        "output_path": str(output_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activitynet-input", type=Path, default=DEFAULT_INPUTS["activitynet"])
    parser.add_argument("--activitynet-output", type=Path, default=DEFAULT_OUTPUTS["activitynet"])
    parser.add_argument("--charades-input", type=Path, default=DEFAULT_INPUTS["charades"])
    parser.add_argument("--charades-output", type=Path, default=DEFAULT_OUTPUTS["charades"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    jobs = [
        ("activitynet", args.activitynet_input, args.activitynet_output),
        ("charades", args.charades_input, args.charades_output),
    ]

    for name, input_path, output_path in jobs:
        stats = convert_bench_json_to_test_json(input_path, output_path)
        print(f"[{name}] videos={stats['videos']} samples={stats['samples']} skipped_pairs={stats['skipped_pairs']}")
        preview = json.loads(output_path.read_text(encoding="utf-8"))
        if preview:
            print(json.dumps(preview[0], ensure_ascii=False))


if __name__ == "__main__":
    main()
