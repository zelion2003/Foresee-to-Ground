#!/usr/bin/env python3


from __future__ import annotations

import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_ROOT = REPO_ROOT / "data" / "raw" / "timelens_videos"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data" / "videos" / "timelens_100k_336"


@dataclass(frozen=True)
class VideoJob:
    source: str
    input_path: Path
    output_path: Path


def discover_jobs(input_root: str | Path, output_root: str | Path) -> list[VideoJob]:
    input_root = Path(input_root)
    output_root = Path(output_root)
    jobs: list[VideoJob] = []
    for source_dir in sorted(p for p in input_root.iterdir() if p.is_dir()):
        source = source_dir.name
        nested_dir = source_dir / source
        search_root = nested_dir if nested_dir.is_dir() else source_dir
        for input_path in sorted(search_root.rglob("*.mp4")):
            output_path = output_root / source / input_path.name
            jobs.append(VideoJob(source=source, input_path=input_path, output_path=output_path))
    return jobs


def build_ffmpeg_command(input_path: str | Path, output_path: str | Path) -> list[str]:
    input_path = Path(input_path)
    output_path = Path(output_path)
    return [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-vf",
        "fps=1,scale=336:336,setsar=1",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "23",
        str(output_path),
    ]


def transcode_one(job: VideoJob, dry_run: bool = False) -> tuple[str, VideoJob, str]:
    if job.output_path.exists() and job.output_path.stat().st_size > 0:
        return ("skipped", job, "output exists")

    job.output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = build_ffmpeg_command(job.input_path, job.output_path)
    if dry_run:
        return ("planned", job, " ".join(cmd))

    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode == 0 and job.output_path.exists() and job.output_path.stat().st_size > 0:
        return ("ok", job, "")

    if job.output_path.exists() and job.output_path.stat().st_size == 0:
        job.output_path.unlink()
    err = proc.stderr.strip().splitlines()
    return ("failed", job, err[-1] if err else f"ffmpeg exit code {proc.returncode}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel ffmpeg workers.")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N videos.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned actions without transcoding.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    jobs = discover_jobs(args.input_root, args.output_root)
    if args.limit is not None:
        jobs = jobs[: max(0, int(args.limit))]

    print(f"input_root={args.input_root}")
    print(f"output_root={args.output_root}")
    print(f"jobs={len(jobs)} workers={args.workers} dry_run={args.dry_run}")

    ok = 0
    skipped = 0
    failed: list[tuple[VideoJob, str]] = []

    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as executor:
        future_map = {executor.submit(transcode_one, job, args.dry_run): job for job in jobs}
        for idx, future in enumerate(as_completed(future_map), start=1):
            status, job, message = future.result()
            if status == "ok":
                ok += 1
            elif status in {"skipped", "planned"}:
                skipped += 1
            else:
                failed.append((job, message))
            if idx % 100 == 0 or idx == len(jobs):
                print(f"processed={idx}/{len(jobs)} ok={ok} skipped={skipped} failed={len(failed)}")

    print("summary:")
    print(f"  ok={ok}")
    print(f"  skipped={skipped}")
    print(f"  failed={len(failed)}")
    if failed:
        print("first_failures:")
        for job, message in failed[:10]:
            print(f"  {job.input_path} -> {job.output_path}: {message}")


if __name__ == "__main__":
    main()
