#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import argparse
import os
from pathlib import Path
from glob import glob
from typing import List, Dict, Any

import torch
from torch.utils.data import Dataset, DataLoader
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

from transformers import AutoProcessor, Qwen3VLForConditionalGeneration  # noqa: E402
import swanlab  # noqa: E402

from stage1.model_stage1 import Stage1LatentJEPA  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VIDEO_DIR = REPO_ROOT / "data" / "stage1" / "clips"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "stage1"


class VideoFolderDataset(Dataset):
    

    def __init__(self, video_dir: str):
        exts = ("*.mp4", "*.avi", "*.mov")
        paths: List[str] = []
        for e in exts:
            paths.extend(glob(os.path.join(video_dir, e)))
        if not paths:
            raise ValueError(f"No videos found in {video_dir}")
        self.paths = sorted(paths)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        return self.paths[idx]


def build_collate_fn(processor, num_frames, fps):
    

    def collate(batch: List[str]) -> Dict[str, Any]:
        proc_kwargs = dict(
            videos=list(batch),
            text=[""] * len(batch),
            return_tensors="pt",
        )
        if num_frames is not None:
            proc_kwargs["num_frames"] = num_frames
        else:
            proc_kwargs["fps"] = fps
        return processor(**proc_kwargs)

    return collate


def parse_args():
    ap = argparse.ArgumentParser("Train Stage1LatentJEPA with Qwen3-VL visual features (single GPU)")
    ap.add_argument("--video_dir", default=str(DEFAULT_VIDEO_DIR), help="Directory containing Stage-1 training clips.")
    ap.add_argument("--model_id", default="Qwen/Qwen3-VL-8B-Instruct", help="Base Qwen3-VL model identifier.")
    ap.add_argument("--batch_size", type=int, default=8, help="Per-step batch size.")
    ap.add_argument("--num_frames", type=int, default=None, help="Optional fixed number of sampled frames.")
    ap.add_argument("--fps", type=float, default=1.0, help="Sampling FPS when num_frames is not set.")
    ap.add_argument("--latent_dim", type=int, default=256, help="Latent dimension used by Stage-1.")
    ap.add_argument("--lambda_sig", type=float, default=0.3, help="Weight assigned to SIGReg relative to prediction loss.")
    ap.add_argument("--epochs", type=int, default=5, help="Number of epochs.")
    ap.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    ap.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay.")
    ap.add_argument(
        "--use_merger_preproj",
        default=False,
        action="store_true",
        help="Skip the merger projection layers and use normalized merged tokens directly.",
    )
    ap.add_argument("--log_interval", type=int, default=1, help="Logging interval in optimization steps.")
    ap.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory used to store Stage-1 checkpoints.")
    return ap.parse_args()


def dump_params(module, name, limit=20):
    print(f"== {name} params (first {limit}) ==")
    cnt = 0
    for n, p in module.named_parameters():
        cnt += 1
        req = "T" if p.requires_grad else "F"
        print(f"{n:60s} shape={tuple(p.shape)!s:20s} dtype={p.dtype} grad={req}")
        if cnt >= limit:
            print("..."); break
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    print(f"total={total:,}, trainable={trainable:,}\n")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    processor = AutoProcessor.from_pretrained(args.model_id)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_id,
        dtype=torch.bfloat16,
        device_map={"": device},
    )
    model.model.visual.deepstack_visual_indexes = []
    model.model.visual.deepstack_merger_list = torch.nn.ModuleList([])
    if args.use_merger_preproj:
        merger = model.model.visual.merger

        def merger_forward_no_proj(self, x):
            if self.use_postshuffle_norm:
                x = self.norm(x.view(-1, self.hidden_size)).view(-1, self.hidden_size)
            else:
                x = self.norm(x).view(-1, self.hidden_size)
            return x

        merger.forward = merger_forward_no_proj.__get__(merger, type(merger))
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    dataset = VideoFolderDataset(args.video_dir)
    collate_fn = build_collate_fn(processor, args.num_frames, args.fps)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)
    first_batch = next(iter(loader))
    grid_thw_key = "video_grid_thw" if "video_grid_thw" in first_batch else "image_grid_thw"
    merge_size = model.model.visual.spatial_merge_size
    tokens_per_frame = int(
        (first_batch[grid_thw_key][0, 1] * first_batch[grid_thw_key][0, 2] // (merge_size ** 2)).item()
    )
    d_in = (
        model.model.visual.merger.hidden_size
        if args.use_merger_preproj
        else model.config.vision_config.out_hidden_size
    )
    stage1 = Stage1LatentJEPA(
        d_in=d_in,
        d_model=d_in,
        latent_dim=args.latent_dim,
        lambda_sig=args.lambda_sig,
        tokens_per_frame=tokens_per_frame,
    ).to(device)

    optimizer = torch.optim.AdamW(stage1.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    dump_params(model.model.visual, "Qwen3-VL visual", limit=10)
    dump_params(stage1, "Stage1LatentJEPA", limit=50)
    swanlab.init(project="stage1_latent", config=vars(args))

    global_step = 0
    for epoch in range(args.epochs):
        for batch in loader:
            grid_thw = batch.get("video_grid_thw", batch.get("image_grid_thw")).to(device)
            pixel_key = "pixel_values_videos" if "pixel_values_videos" in batch else "pixel_values"
            pixel_values = batch[pixel_key].to(device)

            with torch.no_grad():
                embeds_list, _ = model.get_video_features(pixel_values, video_grid_thw=grid_thw) \
                    if "pixel_values_videos" in batch else model.get_image_features(pixel_values, image_grid_thw=grid_thw)
            H_base = torch.nn.utils.rnn.pad_sequence(embeds_list, batch_first=True).to(device).float()

            # Stage1
            out = stage1(H_base)
            loss = out["loss"]

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(stage1.parameters(), max_norm=1.0)

            optimizer.step()


            global_step += 1
            if global_step % args.log_interval == 0:
                print(f"[epoch {epoch} step {global_step}] loss={loss.item():.4f} "
                      f"pred={out['pred_loss'].item():.4f} sig={out['sig_loss'].item():.4f}")
                swanlab.log(
                    {
                        "loss/total": loss.item(),
                        "loss/pred": out["pred_loss"].item(),
                        "loss/sig": out["sig_loss"].item(),
                        "z_mean": out["z_mean"].item(),
                        "z_std": out["z_std"].item(),
                        "epoch": epoch,
                        "step": global_step,
                    },
                    step=global_step,
                )
        ckpt_path = os.path.join(args.output_dir, f"stage1_epoch{epoch}.pt")
        torch.save(
            {
                "stage1": stage1.state_dict(),
                "tokens_per_frame": tokens_per_frame,
                "d_in": d_in,
                "args": vars(args),
            },
            ckpt_path,
        )
        print(f"Saved checkpoint to {ckpt_path}")

    swanlab.finish()


if __name__ == "__main__":
    main()
