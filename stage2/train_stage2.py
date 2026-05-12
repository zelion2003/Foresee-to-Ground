#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

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

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
if CUR_DIR not in sys.path:
    sys.path.append(CUR_DIR)

from stage2.model_stage2 import Stage1BackboneWrapper, Stage2VTGModel, compute_vtg_losses  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANNOTATION_PATH = REPO_ROOT / "data" / "annotations" / "stage2_vtg_sft.json"
DEFAULT_STAGE1_CKPT = REPO_ROOT / "outputs" / "stage1" / "stage1_epoch4.pt"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "stage2"
DEFAULT_VIDEO_ROOTS = {
    "anet": str(REPO_ROOT / "data" / "videos" / "anet"),
    "didemo": str(REPO_ROOT / "data" / "videos" / "didemo"),
    "internvid": str(REPO_ROOT / "data" / "videos" / "internvid"),
}


class Stage2VTGDataset(Dataset):
    

    def __init__(
        self,
        ann_path: str,
        video_roots: Dict[str, str],
        min_spans: int = 1,
        max_samples: int | None = None,
        duration_eps: float = 1e-6,
        min_frames: int = 2,
    ):
        super().__init__()
        with open(ann_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.samples: List[Dict] = []

        for item in data:
            source = item.get("source")
            vid = item.get("video")
            duration = float(item.get("duration", 0.0) or 0.0)
            spans_raw = item.get("spans", [])

            if source not in video_roots:
                continue
            if duration <= duration_eps:
                continue

            video_path = os.path.join(video_roots[source], f"{vid}.mp4")
            if not os.path.isfile(video_path):
                continue
            if min_frames is not None and min_frames > 0:
                try:
                    import cv2

                    cap = cv2.VideoCapture(video_path)
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 0
                    cap.release()
                except Exception:
                    total_frames = 0
                if total_frames < min_frames:
                    continue

            spans_norm = []
            for s in spans_raw:
                try:
                    st = float(s.get("start", 0.0))
                    ed = float(s.get("end", 0.0))
                except Exception:
                    continue
                if ed <= st:
                    continue
                st_n = max(0.0, min(1.0, st / duration))
                ed_n = max(0.0, min(1.0, ed / duration))
                if ed_n <= st_n:
                    continue
                spans_norm.append([st_n, ed_n])

            if len(spans_norm) < min_spans:
                continue

            self.samples.append(
                {
                    "video_path": video_path,
                    "spans": torch.tensor(spans_norm, dtype=torch.float32),
                    "source": source,
                    "video": vid,
                    "duration": duration,
                }
            )

            if max_samples is not None and len(self.samples) >= max_samples:
                break

        if not self.samples:
            raise ValueError("The Stage-2 dataset is empty after filtering. Check the annotation file and video roots.")
        print(f"[Dataset] Kept {len(self.samples)} samples across {len(video_roots)} sources.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def parse_video_roots(arg: str) -> Dict[str, str]:
    if arg:
        roots = json.loads(arg)
    else:
        roots = DEFAULT_VIDEO_ROOTS
    roots = {k: os.path.abspath(v) for k, v in roots.items()}
    for k, v in roots.items():
        if not os.path.isdir(v):
            raise NotADirectoryError(f"Video root does not exist: {k} -> {v}")
    return roots


def build_collate_fn(processor, num_frames: int | None, fps: float, do_sample_frames: bool = True):
    def collate(batch: List[Dict]):
        videos = [b["video_path"] for b in batch]
        spans = [b["spans"] for b in batch]
        meta = [{"source": b["source"], "video": b["video"], "duration": b["duration"]} for b in batch]

        proc_kwargs = dict(
            videos=videos,
            text=[""] * len(videos),
            return_tensors="pt",
            do_sample_frames=do_sample_frames,
        )
        if num_frames is not None:
            proc_kwargs["num_frames"] = num_frames
        else:
            proc_kwargs["fps"] = fps
        model_inputs = processor(**proc_kwargs)
        return model_inputs, spans, meta

    return collate


def patch_qwen_merger(model, use_merger_preproj: bool):
    
    if not use_merger_preproj:
        return
    merger = model.model.visual.merger

    def merger_forward_no_proj(self, x):
        if self.use_postshuffle_norm:
            x = self.norm(x.view(-1, self.hidden_size)).view(-1, self.hidden_size)
        else:
            x = self.norm(x).view(-1, self.hidden_size)
        return x

    merger.forward = merger_forward_no_proj.__get__(merger, type(merger))


def parse_args():
    ap = argparse.ArgumentParser("Stage-2 VTG warm-up training")
    ap.add_argument("--annotation_path", default=str(DEFAULT_ANNOTATION_PATH), help="Path to the Stage-2 annotation JSON file.")
    ap.add_argument("--stage1_ckpt", default=str(DEFAULT_STAGE1_CKPT), help="Stage-1 checkpoint used to initialize the temporal backbone.")
    ap.add_argument("--model_id", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--video_roots", type=str, default="", help='JSON mapping such as {"anet": "...", "didemo": "..."}')
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--num_frames", type=int, default=None, help="Optional fixed number of sampled frames.")
    ap.add_argument("--fps", type=float, default=1.0, help="Sampling FPS when num_frames is not set.")
    ap.add_argument("--max_frames", type=int, default=48, help="Maximum number of video frames processed by the video processor.")
    ap.add_argument("--no_sample_frames", action="store_true", help="Disable frame sampling and use the processor defaults instead.")
    ap.add_argument("--min_frames", type=int, default=2, help="Filter out videos shorter than this frame threshold.")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--max_steps", type=int, default=None, help="Optional cap on the total number of optimization steps.")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--d_hidden", type=int, default=512)
    ap.add_argument("--dec_layers", type=int, default=2)
    ap.add_argument("--head_type", choices=["A", "B"], default="B")
    ap.add_argument("--alpha", type=float, default=0.5, help="Center-offset scale used by regression head A.")
    ap.add_argument("--beta", type=float, default=1.0, help="Segment-length scale used by regression head A.")
    ap.add_argument("--gamma_dist", type=float, default=0.5, help="Distance upper-bound scale used by regression head B.")
    ap.add_argument("--stage1_lr", type=float, default=None, help="Optional learning rate for the Stage-1 backbone.")
    ap.add_argument("--stage1_lr_mult", type=float, default=0.1, help="Fallback multiplier applied to the global learning rate for Stage-1.")
    ap.add_argument("--lambda_cls", type=float, default=1.0)
    ap.add_argument("--lambda_l1", type=float, default=2.0)
    ap.add_argument("--lambda_iou", type=float, default=2.0)
    ap.add_argument("--neg_weight", type=float, default=0.1, help="Weight applied to weak negatives.")
    ap.add_argument("--neg_iou_th", type=float, default=0.1, help="IoU threshold used to select weak negatives.")
    ap.add_argument("--freeze_stage1", action="store_true", default=True, help="Freeze the Stage-1 backbone.")
    ap.add_argument("--no_freeze_stage1", action="store_false", dest="freeze_stage1", help="Allow the Stage-1 backbone to train.")
    ap.add_argument("--use_merger_preproj", default=False, action="store_true", help="Match Stage-1 settings by skipping the merger projection layers.")
    ap.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory used to store Stage-2 checkpoints.")
    ap.add_argument("--save_every", type=int, default=1, help="Save a checkpoint every N epochs.")
    ap.add_argument("--save_steps", type=int, default=6000, help="Save a checkpoint every N optimization steps.")
    ap.add_argument("--log_interval", type=int, default=10)
    ap.add_argument("--max_samples", type=int, default=None, help="Optional cap on the number of loaded samples for debugging.")
    ap.add_argument("--swanlab_project", default="stage2_vtg_style", help="SwanLab project name.")
    ap.add_argument("--swanlab_run", default=None, help="Optional SwanLab run name.")
    ap.add_argument("--no_swanlab", action="store_true", help="Disable SwanLab logging.")
    return ap.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    video_roots = parse_video_roots(args.video_roots)
    dataset = Stage2VTGDataset(
        args.annotation_path,
        video_roots,
        max_samples=args.max_samples,
        min_frames=args.min_frames,
    )
    processor = AutoProcessor.from_pretrained(args.model_id)
    if hasattr(processor, "video_processor"):
        processor.video_processor.fps = args.fps
        processor.video_processor.max_frames = args.max_frames
        processor.video_processor.do_sample_frames = not args.no_sample_frames

    collate_fn = build_collate_fn(processor, args.num_frames, args.fps, do_sample_frames=not args.no_sample_frames)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
    )
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_id,
        dtype=torch.bfloat16,
        device_map={"": device},
    )
    model.model.visual.deepstack_visual_indexes = []
    model.model.visual.deepstack_merger_list = torch.nn.ModuleList([])
    patch_qwen_merger(model, args.use_merger_preproj)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    backbone = Stage1BackboneWrapper(
        ckpt_path=args.stage1_ckpt,
        device=device,
        freeze=args.freeze_stage1,
    )

    stage2 = Stage2VTGModel(
        backbone=backbone,
        d_hidden=args.d_hidden,
        dec_layers=args.dec_layers,
        reg_type=args.head_type,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma_dist,
    ).to(device)
    stage1_params = [p for p in stage2.backbone.parameters() if p.requires_grad]
    new_params = []
    for mod in (stage2.decoder, stage2.head):
        new_params.extend([p for p in mod.parameters() if p.requires_grad])

    param_groups = []
    if new_params:
        param_groups.append({"params": new_params, "lr": args.lr, "weight_decay": args.weight_decay})
    if stage1_params:
        lr_stage1 = args.stage1_lr if args.stage1_lr is not None else args.lr * args.stage1_lr_mult
        param_groups.append({"params": stage1_params, "lr": lr_stage1, "weight_decay": args.weight_decay})

    optimizer = torch.optim.AdamW(param_groups)
    use_swan = not args.no_swanlab
    if use_swan:
        swanlab.init(project=args.swanlab_project, run_name=args.swanlab_run, config=vars(args))
        swanlab.log({"steps_per_epoch": len(loader)}, step=0)

    global_step = 0
    stage2.train()

    for epoch in range(args.epochs):
        for batch in loader:
            model_inputs, gt_spans, meta = batch
            grid_key = "video_grid_thw" if "video_grid_thw" in model_inputs else "image_grid_thw"
            pixel_key = "pixel_values_videos" if "pixel_values_videos" in model_inputs else "pixel_values"
            grid_thw = model_inputs[grid_key].to(device)
            pixel_values = model_inputs[pixel_key].to(device)

            with torch.no_grad():
                embeds_list, _ = (
                    model.get_video_features(pixel_values, video_grid_thw=grid_thw)
                    if "pixel_values_videos" in model_inputs
                    else model.get_image_features(pixel_values, image_grid_thw=grid_thw)
                )

            H_base = torch.nn.utils.rnn.pad_sequence(embeds_list, batch_first=True).to(device).float()  # [B, L, D_in]
            if H_base.shape[-1] != backbone.d_in:
                raise ValueError(
                    f"Qwen feature width {H_base.shape[-1]} does not match Stage-1 d_in={backbone.d_in}. "
                    "Check whether use_merger_preproj and the visual tower settings match Stage-1."
                )

            out = stage2(H_base)
            gt_spans = [g.to(device) for g in gt_spans]
            loss_dict = compute_vtg_losses(
                out["t_start"],
                out["t_end"],
                out["logits"],
                gt_spans,
                lambda_cls=args.lambda_cls,
                lambda_l1=args.lambda_l1,
                lambda_iou=args.lambda_iou,
                gamma=args.neg_weight,
                tau_neg=args.neg_iou_th,
            )
            loss = (
                args.lambda_cls * loss_dict["loss_cls"]
                + args.lambda_l1 * loss_dict["loss_l1"]
                + args.lambda_iou * loss_dict["loss_giou"]
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            global_step += 1
            if global_step % args.log_interval == 0:
                print(
                    f"[epoch {epoch} step {global_step}] "
                    f"loss={loss.item():.4f} "
                    f"cls={loss_dict['loss_cls'].item():.4f} "
                    f"l1={loss_dict['loss_l1'].item():.4f} "
                    f"giou={loss_dict['loss_giou'].item():.4f} "
                    f"pos={loss_dict['pos_count'].item():.1f} "
                    f"neg={loss_dict['neg_count'].item():.1f}"
                )
                if use_swan:
                    swanlab.log(
                        {
                            "loss/total": loss.item(),
                            "loss/cls": loss_dict["loss_cls"].item(),
                            "loss/l1": loss_dict["loss_l1"].item(),
                            "loss/giou": loss_dict["loss_giou"].item(),
                            "stat/pos": loss_dict["pos_count"].item(),
                            "stat/neg": loss_dict["neg_count"].item(),
                            "epoch": epoch,
                        },
                        step=global_step,
                    )

            if args.save_steps and global_step % args.save_steps == 0:
                ckpt_path = os.path.join(args.output_dir, f"stage2_vtg_step{global_step}.pt")
                torch.save(
                    {
                        "stage2": stage2.state_dict(),
                        "config": vars(args),
                        "stage1_ckpt": args.stage1_ckpt,
                        "global_step": global_step,
                        "epoch": epoch,
                    },
                    ckpt_path,
                )
                print(f"[step {global_step}] Saved checkpoint to {ckpt_path}")

            if args.max_steps and global_step >= args.max_steps:
                break

        if (epoch + 1) % args.save_every == 0:
            ckpt_path = os.path.join(args.output_dir, f"stage2_vtg_epoch{epoch}.pt")
            torch.save(
                {
                    "stage2": stage2.state_dict(),
                    "config": vars(args),
                    "stage1_ckpt": args.stage1_ckpt,
                },
                ckpt_path,
            )
            print(f"[epoch {epoch}] Saved checkpoint to {ckpt_path}")

        if args.max_steps and global_step >= args.max_steps:
            break

    if use_swan:
        swanlab.finish()


if __name__ == "__main__":
    main()
