import os
import json
import argparse
from pathlib import Path

from moviepy import VideoFileClip
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip




def find_videos(input_dirs, exts=(".mp4", ".MP4")):
    
    video_paths = []
    for d in input_dirs:
        d = os.path.abspath(d)
        if not os.path.isdir(d):
            print(f"[WARN] Input dir not found or not a directory: {d}")
            continue
        for root, _, files in os.walk(d):
            for name in files:
                if name.endswith(exts):
                    video_paths.append(os.path.join(root, name))
    video_paths = sorted(set(video_paths))
    return video_paths


def safe_clip_name(output_dir, stem, clip_idx):
    
    base = f"{stem}_{clip_idx:04d}"
    candidate = base + ".mp4"
    out_path = os.path.join(output_dir, candidate)

    suffix = 2
    while os.path.exists(out_path):
        candidate = f"{base}_v{suffix:02d}.mp4"
        out_path = os.path.join(output_dir, candidate)
        suffix += 1

    return candidate, out_path


def process_single_video(
    video_path,
    output_dir,
    clip_len_frames=64,
    stride_frames=32,
):
    
    video_path = os.path.abspath(video_path)
    stem = Path(video_path).stem
    file_name = os.path.basename(video_path)

    try:
        vclip = VideoFileClip(video_path)
        duration_sec = float(vclip.duration)
        fps = float(vclip.fps) if vclip.fps is not None else 1.0
        vclip.close()
    except Exception as e:
        print(f"[ERROR] Failed to read video {video_path}: {e}")
        return None, 0
    total_frames = int(round(duration_sec * fps))

    if total_frames < clip_len_frames:
        return None, 0

    clips_meta = []
    clip_idx = 1
    num_clips = 0
    start_frame = 0
    while start_frame + clip_len_frames <= total_frames:
        end_frame = start_frame + clip_len_frames
        start_time_sec = start_frame / fps
        end_time_sec = end_frame / fps
        clip_filename, out_path = safe_clip_name(output_dir, stem, clip_idx)
        try:
            ffmpeg_extract_subclip(
                video_path,
                start_time_sec,
                end_time_sec,
                out_path,
            )
        except Exception as e:
            print(
                f"[ERROR] Failed to extract subclip for {video_path} "
                f"(start={start_time_sec}, end={end_time_sec}): {e}"
            )
            start_frame += stride_frames
            clip_idx += 1
            continue

        clips_meta.append(
            {
                "clip_id": Path(clip_filename).stem,
                "file_name": clip_filename,
                "start_frame": int(start_frame),
                "end_frame": int(end_frame),
                "start_time_sec": float(start_time_sec),
                "end_time_sec": float(end_time_sec),
            }
        )

        num_clips += 1
        clip_idx += 1
        start_frame += stride_frames

    if num_clips == 0:
        return None, 0

    video_meta = {
        "video_id": stem,
        "file_name": file_name,
        "abs_path": video_path,
        "duration_sec": float(duration_sec),
        "fps": float(fps),
        "num_frames_est": int(total_frames),
        "num_clips": int(num_clips),
        "clips": clips_meta,
    }
    return video_meta, num_clips


def main():
    parser = argparse.ArgumentParser(
        description="Build Stage-1 clips from 1 FPS source videos and write a metadata JSON file."
    )
    parser.add_argument(
        "--input_dirs",
        nargs="+",
        default=DEFAULT_INPUT_DIRS,
        help="One or more directories containing preprocessed 1 FPS videos.",
    )
    parser.add_argument(
        "--output_dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory for the generated Stage-1 clips.",
    )
    parser.add_argument(
        "--json_path",
        default=str(DEFAULT_JSON_PATH),
        help="Output path for the generated metadata JSON file.",
    )
    parser.add_argument(
        "--clip_len",
        type=int,
        default=48,
        help="Clip length in frames.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=24,
        help="Sliding-window stride in frames.",
    )
    args = parser.parse_args()

    input_dirs = args.input_dirs
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    json_path = (
        os.path.abspath(args.json_path)
        if args.json_path is not None
        else os.path.join(str(REPO_ROOT / "data" / "stage1"), "stage1_clips_meta.json")
    )

    clip_len_frames = args.clip_len
    stride_frames = args.stride

    video_paths = find_videos(input_dirs)
    num_source_videos = len(video_paths)

    print(f"[INFO] Found {num_source_videos} source videos.")

    videos_meta = []
    num_valid_videos = 0
    num_total_clips = 0

    for idx, vp in enumerate(video_paths, start=1):
        print(f"[INFO] ({idx}/{num_source_videos}) Processing: {vp}")
        video_meta, n_clips = process_single_video(
            vp,
            output_dir=output_dir,
            clip_len_frames=clip_len_frames,
            stride_frames=stride_frames,
        )
        if video_meta is not None:
            videos_meta.append(video_meta)
            num_valid_videos += 1
            num_total_clips += n_clips
    stats = {
        "clip_len_frames": clip_len_frames,
        "stride_frames": stride_frames,
        "num_source_videos": num_source_videos,
        "num_valid_videos": num_valid_videos,
        "num_total_clips": num_total_clips,
    }

    output_obj = {
        "videos": videos_meta,
        "stats": stats,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output_obj, f, ensure_ascii=False, indent=2)

    print("========== SUMMARY ==========")
    print(f"Source videos found      : {num_source_videos}")
    print(f"Valid videos (>=1 clip)  : {num_valid_videos}")
    print(f"Total clips generated    : {num_total_clips}")
    print(f"JSON saved to            : {json_path}")
    print(f"Clips saved under        : {output_dir}")


if __name__ == "__main__":
    main()
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIRS = [
    str(REPO_ROOT / "data" / "videos" / "anet"),
    str(REPO_ROOT / "data" / "videos" / "didemo"),
    str(REPO_ROOT / "data" / "videos" / "internvid"),
]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "stage1" / "clips"
DEFAULT_JSON_PATH = REPO_ROOT / "data" / "stage1" / "stage1_clips_meta.json"
