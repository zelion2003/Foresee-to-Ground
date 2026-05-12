# Data Layout

This repository ships directory templates only. It does not ship raw videos, large annotation files, processed video shards, or model checkpoints.

## Expected Annotation Paths

- `annotations/training_sft.json`
- `annotations/qwen_training_sft.json`
- `annotations/stage2_vtg_sft.json`
- `annotations/stage3_ft.json`
- `annotations/timelens/timelens_100k_stage3.json`
- `annotations/timelens/activitynet_timelens_test.json`
- `annotations/timelens/charades_timelens_test.json`

These annotation files will be released later on **Hugging Face** together with the model checkpoints.

For the main non-TimeLens benchmarks, follow the public NumPro release:

- Repository: <https://github.com/yongliang-wu/NumPro>
- Data release: <https://drive.google.com/drive/folders/13NYRDC87Uc4AqaT5FBHA7QkHV5OMl-v8?usp=sharing>
- Training instructions: <https://drive.google.com/file/d/1X4VSdSpGEBeRDVGaZq6HsUjJxUj88jDc/view?usp=sharing>
- 1 FPS videos: <https://huggingface.co/datasets/Liang0223/NumPro_FT>

## Expected Video Directories

- `videos/anet/`
- `videos/didemo/`
- `videos/internvid/`
- `videos/timelens_100k_336/`
- `videos/timelens_bench_336/`

## Stage-1 Clips

Stage-1 uses short clips instead of full videos during pretraining. The public default location is:

- `stage1/clips/`

Use `tools/build_stage1_clips.py` to construct those clips from preprocessed 1 FPS videos.
