# Offline Tools

This directory contains offline data-processing utilities.

- `build_stage1_clips.py`: cut Stage-1 clips from 1 FPS source videos.
- `check_stage1_clips.py`: filter Stage-1 clips by basic decoding and metadata checks.
- `build_qwen_sft_json.py`: flatten multi-turn annotations into single-turn SFT examples.
- `build_timelens_stage3_json.py`: convert TimeLens-100K JSONL annotations to the Stage-3 format.
- `build_timelens_bench_test_json.py`: convert TimeLens-Bench files into flat evaluation JSON files.
- `resize_timelens_videos_336.py`: transcode TimeLens videos to 1 FPS and `336x336`.
- `check_sft_dataset.py`: validate the SFT dataset against the available videos.

These scripts are not imported by the stage training code. They are intended to be run explicitly when preparing data.
