#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import argparse
from pathlib import Path
import torch
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

from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from stage1.model_stage1 import Stage1LatentJEPA


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    ap = argparse.ArgumentParser("Run Qwen3-VL vision -> Stage1LatentJEPA (single GPU)")
    ap.add_argument("--model_id", default="Qwen/Qwen3-VL-8B-Instruct", help="Base Qwen3-VL model identifier.")
    ap.add_argument("--video", default=str(REPO_ROOT / "data" / "stage1" / "clips" / "example.mp4"), help="Input video path.")
    ap.add_argument(
        "--num_frames",
        type=int,
        default=None,
        help="Optional fixed number of sampled frames.",
    )
    ap.add_argument("--fps", type=float, default=1.0, help="Sampling FPS when num_frames is not set.")
    ap.add_argument("--latent_dim", type=int, default=256, help="Latent dimension used by Stage-1.")
    ap.add_argument("--lambda_sig", type=float, default=0.1, help="Weight assigned to SIGReg relative to prediction loss.")
    ap.add_argument(
        "--use_merger_preproj",
        default=True,
        action="store_true",
        help="Skip the merger projection layers and use normalized merged tokens directly.",
    )
    return ap.parse_args()


def main():
    args = parse_args()
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
    def make_hook(name):
        def _hook(mod, inp, out):
            tensor = out[0] if isinstance(out, tuple) else out
            print(f"[HOOK] {name} -> {tuple(tensor.shape)}")
        return _hook

    handles = []
    for n, m in model.model.visual.named_modules():
        if n in ["patch_embed", "merger"]:
            handles.append(m.register_forward_hook(make_hook(n)))
    proc_kwargs = dict(
        videos=[args.video],
        text=[""],
        return_tensors="pt",
    )
    if args.num_frames is not None:
        proc_kwargs["num_frames"] = args.num_frames
    else:
        proc_kwargs["fps"] = args.fps
    batch = processor(**proc_kwargs)
    if "pixel_values_videos" in batch:
        pixel_values = batch["pixel_values_videos"].to(device)
        grid_thw = batch["video_grid_thw"].to(device)
        get_feats = model.get_video_features
    else:
        pixel_values = batch["pixel_values"].to(device)
        grid_thw = batch["image_grid_thw"].to(device)
        get_feats = model.get_image_features
    with torch.no_grad():
        embeds_list, _ = get_feats(pixel_values, grid_thw)
    H_base = torch.nn.utils.rnn.pad_sequence(embeds_list, batch_first=True)  # [B, L, C]
    # import pdb; pdb.set_trace()
    merge_size = model.model.visual.spatial_merge_size
    tokens_per_frame = int((grid_thw[0, 1] * grid_thw[0, 2] // (merge_size ** 2)).item())
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
    def dump_params(module, name, limit=50):
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

    dump_params(model.model.visual, "Qwen3-VL visual", limit=50)
    dump_params(stage1, "Stage1LatentJEPA", limit=200)
    H_base = H_base.to(device).float()
    out = stage1(H_base)
    print("z_latent:", out["z_latent"].shape)
    print("loss:", float(out["loss"]))
    print("pred_loss:", float(out["pred_loss"]))
    print("sig_loss:", float(out["sig_loss"]))
    for h in handles:
        h.remove()


if __name__ == "__main__":
    main()
