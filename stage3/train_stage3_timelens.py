#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import argparse
import math
import os
import signal
import re
from pathlib import Path
from typing import Dict, List, Tuple

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
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
from peft import LoraConfig, get_peft_model  # noqa: E402
import swanlab  # noqa: E402
import deepspeed  # noqa: E402


from stage3.dataset_stage3 import SPAN_PATTERN, Stage3VTGDataset
from stage3.tokens_stage3 import add_span_tokens
from stage3.model_stage3 import Stage1BackboneWrapper, Stage3SpanModel, proposal_loss_single_gt
from stage3.prompt_stage3 import render_prompt_b, find_insert_pos, mask_labels_before_assistant


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TIMELENS_DATA = REPO_ROOT / "data" / "annotations" / "timelens" / "timelens_100k_stage3.json"
DEFAULT_TIMELENS_OUTPUT = REPO_ROOT / "outputs" / "stage3_timelens"
DEFAULT_TIMELENS_VIDEO_ROOTS = {
    "cosmo_cap": str(REPO_ROOT / "data" / "videos" / "timelens_100k_336" / "cosmo_cap"),
    "didemo": str(REPO_ROOT / "data" / "videos" / "timelens_100k_336" / "didemo"),
    "internvid_vtime": str(REPO_ROOT / "data" / "videos" / "timelens_100k_336" / "internvid_vtime"),
    "queryd": str(REPO_ROOT / "data" / "videos" / "timelens_100k_336" / "queryd"),
    "hirest": str(REPO_ROOT / "data" / "videos" / "timelens_100k_336" / "hirest"),
    "hirest_step": str(REPO_ROOT / "data" / "videos" / "timelens_100k_336" / "hirest"),
    "hirest_grounding": str(REPO_ROOT / "data" / "videos" / "timelens_100k_336" / "hirest"),
}


def build_recovery_ckpt_label(reason: str, epoch: int, global_step: int, opt_step: int) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "_", str(reason).strip().lower()).strip("_")
    cleaned = cleaned or "interrupt"
    return f"{cleaned}_e{int(epoch)}_g{int(global_step)}_o{int(opt_step)}"


def resolve_model_init_source(model_id: str, resume_stage3_ckpt: str | None) -> str:
    return str(resume_stage3_ckpt) if resume_stage3_ckpt else str(model_id)


def resolve_resume_stage3_span_path(resume_stage3_ckpt: str) -> str:
    path = os.path.join(str(resume_stage3_ckpt), "stage3_span.pt")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"resume checkpoint is missing stage3_span.pt: {path}")
    return path


class SpanTokenRows(nn.Module):
    

    def __init__(self, span_token_ids: List[int], init_in: torch.Tensor, init_out: torch.Tensor):
        super().__init__()
        self.register_buffer("span_token_ids", torch.tensor(span_token_ids, dtype=torch.long))
        self.span_in = nn.Parameter(init_in)    # [K, D_llm]
        self.span_out = nn.Parameter(init_out)  # [K, D_llm]


def _encode_text_variants(tokenizer, text: str) -> List[List[int]]:
    
    variants = [text, f" {text}", f"\n{text}"]
    out: List[List[int]] = []
    seen = set()
    for v in variants:
        ids = tokenizer(v, add_special_tokens=False)["input_ids"]
        key = tuple(ids)
        if key and key not in seen:
            out.append(ids)
            seen.add(key)
    return out


def _find_subsequence_starts(seq: List[int], sub: List[int]) -> List[int]:
    if not sub or len(seq) < len(sub):
        return []
    starts: List[int] = []
    n = len(sub)
    for i in range(len(seq) - n + 1):
        if seq[i : i + n] == sub:
            starts.append(i)
    return starts


def _extract_time_strings(answer: str) -> Tuple[str, str] | None:
    
    m = SPAN_PATTERN.search(answer)
    if not m:
        return None
    return m.group(1), m.group(2)


def parse_video_roots(arg: str) -> Dict[str, str]:
    if arg:
        import json
        roots = json.loads(arg)
    else:
        roots = DEFAULT_TIMELENS_VIDEO_ROOTS
    roots = {k: os.path.abspath(v) for k, v in roots.items()}
    for k, v in roots.items():
        if not os.path.isdir(v):
            raise NotADirectoryError(f"Video root does not exist: {k} -> {v}")
    return roots


def build_messages(video_path: str, question: str, answer: str, fps: float):
    return [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": video_path, "fps": fps},
                {"type": "text", "text": question},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": answer}],
        },
    ]


def splice_ids_masks(input_ids_a: torch.Tensor, attn_a: torch.Tensor, input_ids_b: torch.Tensor, attn_b: torch.Tensor, insert_pos: int):
    # input_ids_a: [La], input_ids_b: [Lb]
    ids = torch.cat([input_ids_a[:insert_pos], input_ids_b, input_ids_a[insert_pos:]], dim=0)
    attn = torch.cat([attn_a[:insert_pos], attn_b, attn_a[insert_pos:]], dim=0)
    return ids, attn


def replace_tokens(inputs_embeds: torch.Tensor, input_ids: torch.Tensor, token_id: int, new_embeds: torch.Tensor):
    
    pos = (input_ids == token_id).nonzero(as_tuple=False).squeeze(-1)
    assert pos.numel() == new_embeds.size(0), f"token count {pos.numel()} != embed count {new_embeds.size(0)}"
    inputs_embeds[pos] = new_embeds
    return inputs_embeds


def _unwrap_hf_model(model_obj):
    
    # DeepSpeedEngine
    if hasattr(model_obj, "module") and "deepspeed" in type(model_obj).__module__:
        model_obj = model_obj.module

    # DDP
    if isinstance(model_obj, DDP):
        model_obj = model_obj.module
    if hasattr(model_obj, "base_model") and hasattr(model_obj.base_model, "model"):
        model_obj = model_obj.base_model.model
    elif hasattr(model_obj, "model") and hasattr(model_obj.model, "lm_head") and hasattr(model_obj.model, "model"):
        model_obj = model_obj.model

    if not (hasattr(model_obj, "model") and hasattr(model_obj, "lm_head")):
        raise TypeError(f"Failed to unwrap the underlying HF model. Got type: {type(model_obj)}")
    return model_obj


def save_model_checkpoint(args, base_model, processor, tokenizer, stage3_model, backbone, opt_step_label: str | None):
    
    save_dir = args.output_dir if opt_step_label is None else os.path.join(args.output_dir, opt_step_label)
    os.makedirs(save_dir, exist_ok=True)

    try:
        model_to_save = base_model.merge_and_unload()
    except Exception:
        model_to_save = base_model
    span_module = stage3_model.module if isinstance(stage3_model, DDP) else stage3_model
    if hasattr(span_module, "span_token_rows"):
        rows: SpanTokenRows = span_module.span_token_rows
        hf_model = _unwrap_hf_model(model_to_save)
        with torch.no_grad():
            span_ids = rows.span_token_ids.to(device=hf_model.get_input_embeddings().weight.device)
            emb_w = hf_model.get_input_embeddings().weight
            emb_w[span_ids] = rows.span_in.to(device=emb_w.device, dtype=emb_w.dtype)
            lm_w = hf_model.lm_head.weight
            lm_w[span_ids] = rows.span_out.to(device=lm_w.device, dtype=lm_w.dtype)

    model_to_save.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    try:
        processor.save_pretrained(save_dir)
    except Exception as e:
        print(f"[warn] processor.save_pretrained failed: {type(e).__name__}: {e}")

    torch.save(
        {
            "stage3_span": span_module.state_dict(),
            "config": vars(args),
            "tokens_per_frame": backbone.tokens_per_frame,
            "d_in": backbone.d_in,
        },
        os.path.join(save_dir, "stage3_span.pt"),
    )


def parse_args():
    ap = argparse.ArgumentParser("Stage-3 VTG Training")
    ap.add_argument("--local_rank", type=int, default=0, help="Local rank supplied by torchrun or DeepSpeed.")
    ap.add_argument("--data_path", default=str(DEFAULT_TIMELENS_DATA), help="Path to the converted TimeLens Stage-3 annotation JSON file.")
    ap.add_argument(
        "--video_roots",
        type=str,
        default="",
        help='JSON mapping such as {"cosmo_cap": "...", "hirest_step": "..."}',
    )
    ap.add_argument("--model_id", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--stage2_ckpt", required=True, help="Stage-2 checkpoint containing the temporal backbone, decoder, and proposal head.")
    ap.add_argument("--stage1_ckpt", default=None, help="Optional fallback Stage-1 checkpoint used only when Stage-2 metadata is incomplete.")
    ap.add_argument("--resume_stage3_ckpt", default=None, help="Optional Stage-3 checkpoint directory to resume from.")
    ap.add_argument("--output_dir", default=str(DEFAULT_TIMELENS_OUTPUT), help="Directory used to store TimeLens Stage-3 checkpoints.")
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max_steps", type=int, default=None)
    ap.add_argument("--save_every", type=float, default=None, help="Checkpoint frequency in epochs, for example 0.5 saves every half epoch.")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lr_stage1_mult", type=float, default=0.1)
    ap.add_argument("--warmup_ratio", type=float, default=0.05, help="Learning-rate warmup ratio.")
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--lambda_prop", type=float, default=0.1, help="Weight assigned to the proposal loss term.")
    ap.add_argument("--tau_span_ignore", type=float, default=0.0, help="Ignore span-id supervision when IoU falls below this threshold.")
    ap.add_argument("--span_id_weight", type=float, default=0.0, help="Weight assigned to <Span_k> label positions in the LM loss.")
    ap.add_argument(
        "--time_value_weight",
        type=float,
        default=0.0,
        help="Weight assigned to the start/end time value tokens in the LM loss.",
    )
    ap.add_argument("--k_top", type=int, default=8)
    ap.add_argument("--m_queries", type=int, default=4)
    ap.add_argument("--max_frames", type=int, default=48)
    ap.add_argument("--fps", type=float, default=1.0)
    ap.add_argument("--no_sample_frames", action="store_true")
    ap.add_argument("--log_interval", type=int, default=10)
    ap.add_argument("--grad_acc_steps", type=int, default=1, help="Gradient accumulation steps when DeepSpeed does not override the setting.")
    ap.add_argument("--swanlab_project", default="stage3_vtg_timelens", help="SwanLab project name.")
    ap.add_argument("--swanlab_run", default=None, help="Optional SwanLab run name.")
    ap.add_argument("--no_swanlab", action="store_true", help="Disable SwanLab logging.")
    ap.add_argument("--deepspeed", default=None, help="Path to a DeepSpeed configuration file, for example scripts/zero2.json.")
    ap.add_argument("--debug_generate_every", type=int, default=0, help="Run text generation every N global steps for debugging. Use 0 to disable.")
    ap.add_argument(
        "--debug_generate_until",
        type=int,
        default=0,
        help="Only enable debug generation for the first N global steps. Use 0 for no limit.",
    )
    ap.add_argument("--debug_generate_max_new_tokens", type=int, default=96, help="max_new_tokens used by debug generation.")
    ap.add_argument("--debug_generate_num_samples", type=int, default=1, help="Number of batch samples to print during debug generation.")
    ap.add_argument("--debug_print_prompt", action="store_true", help="Also print the prompt during debug generation.")
    ap.add_argument("--debug_prompt_chars", type=int, default=800, help="Maximum number of prompt characters printed during debugging.")
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Distributed init
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(local_rank)
        is_main = dist.get_rank() == 0
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        local_rank = args.local_rank
        is_main = True

    video_roots = parse_video_roots(args.video_roots)
    dataset = Stage3VTGDataset(args.data_path, video_roots=video_roots, min_frames=2)
    init_source = resolve_model_init_source(args.model_id, args.resume_stage3_ckpt)
    processor = AutoProcessor.from_pretrained(init_source, local_files_only=True)
    tokenizer = processor.tokenizer
    if hasattr(processor, "video_processor"):
        processor.video_processor.fps = args.fps
        processor.video_processor.max_frames = args.max_frames
        processor.video_processor.do_sample_frames = not args.no_sample_frames
    model_base = Qwen3VLForConditionalGeneration.from_pretrained(
        init_source, torch_dtype=torch.bfloat16, local_files_only=True
    ).to(device)
    try:
        model_base.model.visual.deepstack_visual_indexes = []
        model_base.model.visual.deepstack_merger_list = nn.ModuleList([])
    except Exception:
        pass
    new_ids, old_vocab = add_span_tokens(tokenizer, model_base, k=args.k_top, add_span_pad=True)
    llm_hidden = int(model_base.get_input_embeddings().weight.shape[1])
    span_token_ids = [tokenizer.convert_tokens_to_ids(f"<Span_{i}>") for i in range(args.k_top)]
    if hasattr(tokenizer, "unk_token_id") and any(tid == tokenizer.unk_token_id for tid in span_token_ids):
        raise ValueError("Could not find <Span_k> tokens in the tokenizer. Check whether add_span_tokens was applied correctly.")
    init_in = model_base.get_input_embeddings().weight.detach()[span_token_ids].clone().float()   # [K, D]
    init_out = model_base.lm_head.weight.detach()[span_token_ids].clone().float()                # [K, D]
    span_token_rows = SpanTokenRows(span_token_ids=span_token_ids, init_in=init_in, init_out=init_out).to(device)
    try:
        model_base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    except Exception:
        model_base.gradient_checkpointing_enable()
    model_base.config.use_cache = False
    lora_cfg = LoraConfig(
        r=64,
        lora_alpha=128,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model_base = get_peft_model(model_base, lora_cfg)
    for name, p in model_base.named_parameters():
        if "visual" in name:
            p.requires_grad = False
    for name, p in model_base.named_parameters():
        if any(k in name for k in ("embed_tokens", "lm_head")):
            p.requires_grad = False
    stage2_blob = torch.load(args.stage2_ckpt, map_location="cpu")
    stage2_cfg_all = stage2_blob.get("config", {}) or stage2_blob.get("args", {}) or {}
    stage2_state = stage2_blob.get("stage2", stage2_blob)
    stage2_cfg = {
        "d_in": stage2_blob.get("d_in"),
        "tokens_per_frame": stage2_blob.get("tokens_per_frame"),
        "d_model": stage2_blob.get("d_model"),
        "num_layers": stage2_blob.get("num_layers"),
        "num_heads": stage2_blob.get("num_heads"),
        "kernel_size": stage2_blob.get("kernel_size"),
    }
    def _infer_from_state(sd):
        meta = {}
        w_attn = sd.get("backbone.backbone.blocks.0.attn.in_proj_weight")
        if w_attn is not None:
            meta["d_model"] = w_attn.shape[1]
        w_fp = sd.get("backbone.frame_pool.mlp.0.weight")
        if w_fp is not None:
            meta["d_in"] = w_fp.shape[0]
        return meta

    inferred = _infer_from_state(stage2_state if isinstance(stage2_state, dict) else {})
    for k, v in inferred.items():
        if stage2_cfg.get(k) is None:
            stage2_cfg[k] = v
    hf_cfg = _unwrap_hf_model(model_base)
    d_in = stage2_cfg["d_in"] or hf_cfg.config.vision_config.out_hidden_size
    backbone = Stage1BackboneWrapper(
        device=device,
        ckpt_path=args.stage1_ckpt,
        freeze=False,
        d_in=d_in,
        tokens_per_frame=stage2_cfg.get("tokens_per_frame"),
        stage2_config=stage2_cfg if args.stage1_ckpt is None else None,
    )
    stage2_d_hidden = int(stage2_cfg_all.get("d_hidden", 512) or 512)
    stage2_dec_layers = int(stage2_cfg_all.get("dec_layers", 2) or 2)
    stage2_reg_type = str(stage2_cfg_all.get("head_type", stage2_cfg_all.get("reg_type", "B")) or "B")
    stage2_alpha = float(stage2_cfg_all.get("alpha", 0.5) or 0.5)
    stage2_beta = float(stage2_cfg_all.get("beta", 1.0) or 1.0)
    stage2_gamma = float(stage2_cfg_all.get("gamma_dist", stage2_cfg_all.get("gamma", 0.5)) or 0.5)

    stage3_model = Stage3SpanModel(
        stage2_backbone=backbone,
        d_hidden=stage2_d_hidden,
        dec_layers=stage2_dec_layers,
        reg_type=stage2_reg_type,
        alpha=stage2_alpha,
        beta=stage2_beta,
        gamma_dist=stage2_gamma,
        htre_heads=8,
        span_projector_dim=llm_hidden,
        k_top=args.k_top,
    ).to(device)
    stage3_model.span_token_rows = span_token_rows
    stage3_model.stage2.load_state_dict(stage2_state, strict=False)
    if args.resume_stage3_ckpt:
        resume_stage3_path = resolve_resume_stage3_span_path(args.resume_stage3_ckpt)
        resume_blob = torch.load(resume_stage3_path, map_location="cpu")
        resume_state = resume_blob.get("stage3_span", resume_blob)
        missing, unexpected = stage3_model.load_state_dict(resume_state, strict=False)
        if is_main:
            print(
                f"[resume-stage3] loaded from {args.resume_stage3_ckpt} "
                f"missing={len(missing)} unexpected={len(unexpected)}",
                flush=True,
            )
    params_stage1 = [p for p in stage3_model.stage2.backbone.parameters() if p.requires_grad]
    _stage1_ids = {id(p) for p in params_stage1}
    params_span = [p for p in stage3_model.parameters() if p.requires_grad and id(p) not in _stage1_ids]
    params_llm = [p for p in model_base.parameters() if p.requires_grad]

    optimizer_span = torch.optim.AdamW(
        [
            {"params": params_span, "lr": args.lr, "weight_decay": args.weight_decay},
            {"params": params_stage1, "lr": args.lr * args.lr_stage1_mult, "weight_decay": args.weight_decay},
        ]
    )

    def collate_fn(batch: List[Dict]):
        return batch

    if dist.is_initialized():
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True)
    else:
        sampler = None
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        sampler=sampler,
    )

    global_step = 0
    model_base.train(); stage3_model.train()
    try:
        _unwrap_hf_model(model_base).model.visual.eval()
    except Exception:
        pass
    if dist.is_initialized():
        stage3_model = DDP(stage3_model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    ds_config = args.deepspeed
    if args.deepspeed:
        import json as _json
        if isinstance(ds_config, str):
            with open(ds_config, "r") as f:
                ds_cfg_obj = _json.load(f)
        else:
            ds_cfg_obj = dict(ds_config)
        ds_cfg_obj["gradient_accumulation_steps"] = int(args.grad_acc_steps)
        micro_bs = int(args.batch_size)
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        train_bs = micro_bs * args.grad_acc_steps * world_size
        ds_cfg_obj["train_micro_batch_size_per_gpu"] = micro_bs
        ds_cfg_obj["train_batch_size"] = train_bs
        ds_cfg_obj["bf16"] = {"enabled": True}
        ds_cfg_obj["fp16"] = {"enabled": False}
        llm_optimizer = torch.optim.AdamW(params_llm, lr=args.lr, weight_decay=args.weight_decay)
        llm_engine, llm_optimizer, _, _ = deepspeed.initialize(
            model=model_base,
            model_parameters=params_llm,
            optimizer=llm_optimizer,
            config=ds_cfg_obj,
        )
        llm_model = llm_engine
    else:
        if dist.is_initialized():
            llm_model = DDP(model_base, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
        else:
            llm_model = model_base
        llm_optimizer = torch.optim.AdamW(params_llm, lr=args.lr, weight_decay=args.weight_decay)

    # ==========
    # ==========
    total_micro_steps = args.epochs * len(loader)
    if args.max_steps is not None:
        total_micro_steps = min(int(args.max_steps), int(total_micro_steps))
    total_opt_steps = max(1, math.ceil(total_micro_steps / max(1, int(args.grad_acc_steps))))
    warmup_steps = int(args.warmup_ratio * total_opt_steps)

    llm_base_lrs = [float(g.get("lr", args.lr)) for g in llm_optimizer.param_groups]
    span_base_lrs = [float(g.get("lr", args.lr)) for g in optimizer_span.param_groups]

    def _lr_mult(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        return max(0.0, float(total_opt_steps - step) / float(max(1, total_opt_steps - warmup_steps)))

    def _apply_lr(step: int) -> float:
        mult = _lr_mult(step)
        for g, base in zip(llm_optimizer.param_groups, llm_base_lrs):
            g["lr"] = base * mult
        for g, base in zip(optimizer_span.param_groups, span_base_lrs):
            g["lr"] = base * mult
        return mult
    _apply_lr(step=0)

    def llm_module():
        if args.deepspeed and isinstance(llm_model, deepspeed.DeepSpeedEngine):
            return llm_model.module
        if dist.is_initialized() and isinstance(llm_model, DDP):
            return llm_model.module
        return llm_model

    def stage3_module():
        return stage3_model.module if isinstance(stage3_model, DDP) else stage3_model

    def span_rows_module() -> SpanTokenRows:
        m = stage3_module()
        if not hasattr(m, "span_token_rows"):
            raise AttributeError("stage3_model is missing span_token_rows for span-token row training.")
        return m.span_token_rows
    use_swan = (not args.no_swanlab) and is_main
    if use_swan:
        swanlab.init(project=args.swanlab_project, run_name=args.swanlab_run, config=vars(args))
        swanlab.log({"steps_per_epoch": len(loader)}, step=0)

    save_interval_opt = None
    if args.save_every is not None and args.save_every > 0:
        est_steps_per_epoch = max(1, len(loader) // max(1, args.grad_acc_steps))
        save_interval_opt = max(1, int(args.save_every * est_steps_per_epoch))

    opt_step = 0
    progress = {"epoch": 0, "global_step": 0, "opt_step": 0}
    recovery_saved = {"done": False}
    signal_state = {"name": None}

    def _safe_recovery_save(reason: str):
        if recovery_saved["done"]:
            return
        if not is_main:
            return
        if progress["global_step"] <= 0 and progress["opt_step"] <= 0:
            print(f"[recovery-save] skip reason={reason} because no training progress yet", flush=True)
            return
        label = build_recovery_ckpt_label(
            reason=reason,
            epoch=progress["epoch"],
            global_step=progress["global_step"],
            opt_step=progress["opt_step"],
        )
        print(f"[recovery-save] start label={label}", flush=True)
        try:
            save_model_checkpoint(args, llm_module(), processor, tokenizer, stage3_model, backbone, opt_step_label=label)
        except Exception as e:
            print(f"[recovery-save] failed label={label} err={type(e).__name__}: {e}", flush=True)
        else:
            recovery_saved["done"] = True
            print(f"[recovery-save] done label={label}", flush=True)

    def _signal_handler(signum, frame):
        signal_state["name"] = signal.Signals(signum).name.lower()
        raise KeyboardInterrupt(signal_state["name"])

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)
    try:
        for epoch in range(args.epochs):
            progress["epoch"] = epoch
            if dist.is_initialized() and sampler is not None:
                sampler.set_epoch(epoch)
            for samples in loader:
                video_paths = [s["video_path"] for s in samples]
                durations = torch.tensor([s["duration"] for s in samples], device=device)
                questions = [s["question"] for s in samples]
                answers = [s["answer"] for s in samples]
                gt_spans = torch.stack([s["gt_span"] for s in samples], dim=0).to(device)  # [B,2] norm
                placeholder = "<Span_0>"
                init_span_list = " ".join([f"<Span_{i}>" for i in range(args.k_top)])
                msgs = []
                for vp, q, a in zip(video_paths, questions, answers):
                    user_txt = (
                        f"Question: {q}\n"
                        "Please answer naturally, and finally cite exactly ONE candidate span id token: "
                        f"{init_span_list}."
                    )
                    assistant_txt = f"{a}\nCorresponding span: {placeholder}."
                    msgs.append(build_messages(vp, user_txt, assistant_txt, fps=args.fps))
                inputs_A = processor.apply_chat_template(msgs, tokenize=True, add_generation_prompt=False, return_dict=True, return_tensors="pt")
                pixel_values = inputs_A["pixel_values_videos"].to(device)
                grid_thw = inputs_A["video_grid_thw"].to(device)
                if backbone.tokens_per_frame is None:
                    merge_size = model_base.model.visual.spatial_merge_size
                    tokens_per_frame = int(
                        (grid_thw[0, 1] * grid_thw[0, 2]) // (merge_size ** 2)
                    )
                    backbone.tokens_per_frame = tokens_per_frame
                with torch.no_grad():
                    embeds_list, _ = llm_module().get_video_features(pixel_values, video_grid_thw=grid_thw)
                H_base = pad_sequence(embeds_list, batch_first=True).to(device).float()
                if H_base.shape[-1] != backbone.d_in:
                    raise ValueError(
                        f"Qwen visual feature width mismatch: H_base[-1]={H_base.shape[-1]} vs backbone.d_in={backbone.d_in}. "
                        "Make sure Stage-3 uses the merger-projected output (vision_config.out_hidden_size) and matches Stage-2 training."
                    )
                prop = stage3_model(H_base)
                s_topk = prop["start"]  # [B,K]
                e_topk = prop["end"]
                span_embeds = prop["span_embeds"]  # [B,K,M,D_llm]
                K = s_topk.size(1)
                span_list = " ".join([f"<Span_{i}>" for i in range(K)])
                iou_mat = torch.zeros_like(s_topk)
                for b in range(s_topk.size(0)):
                    gs, ge = gt_spans[b, 0], gt_spans[b, 1]
                    inter = torch.clamp(torch.min(e_topk[b], ge) - torch.max(s_topk[b], gs), min=0.0)
                    union = torch.clamp(torch.max(e_topk[b], ge) - torch.min(s_topk[b], gs), min=1e-6)
                    iou_mat[b] = inter / union
                best_iou, best_k = iou_mat.max(dim=1)
                msgs = []
                for b, (vp, q, a) in enumerate(zip(video_paths, questions, answers)):
                    user_txt = (
                        f"Question: {q}\n"
                        "Please answer naturally, and finally cite exactly ONE candidate span id token: "
                        f"{span_list}."
                    )
                    assistant_txt = f"{a}\nCorresponding span: <Span_{best_k[b].item()}>."
                    msgs.append(build_messages(vp, user_txt, assistant_txt, fps=args.fps))

                inputs_A = processor.apply_chat_template(msgs, tokenize=True, add_generation_prompt=False, return_dict=True, return_tensors="pt")
                input_ids_A = inputs_A["input_ids"]           # [B, L]
                attn_A = inputs_A["attention_mask"]           # [B, L]
                s_sec = s_topk * durations.unsqueeze(1)
                e_sec = e_topk * durations.unsqueeze(1)
                prompt_B_list = render_prompt_b(s_sec.cpu(), e_sec.cpu(), k=K, m=span_embeds.size(2))
                tok_B = tokenizer(prompt_B_list, add_special_tokens=False, return_tensors="pt", padding=True)
                input_ids_B = tok_B["input_ids"]
                attn_B = tok_B["attention_mask"]
                ids_list = []
                attn_list = []
                for b in range(len(msgs)):
                    ins_pos = find_insert_pos(input_ids_A[b].tolist(), tokenizer)
                    ids_b, attn_b = splice_ids_masks(input_ids_A[b], attn_A[b], input_ids_B[b], attn_B[b], ins_pos)
                    ids_list.append(ids_b)
                    attn_list.append(attn_b)
                ids_padded = pad_sequence(ids_list, batch_first=True, padding_value=tokenizer.pad_token_id)
                attn_padded = pad_sequence(attn_list, batch_first=True, padding_value=0)
                ids_padded = ids_padded.to(device)
                attn_padded = attn_padded.to(device)

                # labels
                labels = ids_padded.clone()
                labels = mask_labels_before_assistant(labels, tokenizer)
                labels = labels.masked_fill(attn_padded == 0, -100)
                if args.tau_span_ignore > 0:
                    for b in range(len(msgs)):
                        if best_iou[b] < args.tau_span_ignore:
                            span_id = tokenizer.convert_tokens_to_ids(f"<Span_{best_k[b].item()}>")
                            pos = (ids_padded[b] == span_id).nonzero(as_tuple=False)
                            if pos.numel() > 0:
                                labels[b, pos[-1]] = -100
                hf = _unwrap_hf_model(llm_module())
                inputs_embeds = hf.get_input_embeddings()(ids_padded)
                inputs_embeds = inputs_embeds.clone()
                rows = span_rows_module()
                for i, tok_id in enumerate(rows.span_token_ids.tolist()):
                    mask = (ids_padded == tok_id)
                    if mask.any():
                        inputs_embeds[mask] = rows.span_in[i].to(dtype=inputs_embeds.dtype, device=inputs_embeds.device)
                vid_id = tokenizer.convert_tokens_to_ids("<|video_pad|>")
                # flatten embeds_list to match counts per sample
                for b in range(len(msgs)):
                    vid_pos = (ids_padded[b] == vid_id).nonzero(as_tuple=False).squeeze(-1)
                    vid_embed = embeds_list[b].to(device).to(inputs_embeds.dtype)
                    assert vid_pos.numel() == vid_embed.size(0), f"video_pad count {vid_pos.numel()} != vid_embed {vid_embed.size(0)}"
                    inputs_embeds[b, vid_pos, :] = vid_embed
                sp_id = tokenizer.convert_tokens_to_ids("<span_pad>")
                K, M = span_embeds.size(1), span_embeds.size(2)
                flat_span = span_embeds.view(len(msgs), K * M, -1).to(inputs_embeds.dtype)
                for b in range(len(msgs)):
                    sp_pos = (ids_padded[b] == sp_id).nonzero(as_tuple=False).squeeze(-1)
                    assert sp_pos.numel() == flat_span.size(1), f"span_pad count {sp_pos.numel()} != {flat_span.size(1)}"
                    inputs_embeds[b, sp_pos, :] = flat_span[b]

                # =========================
                # =========================
                if (
                    is_main
                    and args.debug_generate_every is not None
                    and int(args.debug_generate_every) > 0
                    and (int(getattr(args, "debug_generate_until", 0)) <= 0 or global_step < int(args.debug_generate_until))
                    and (global_step % int(args.debug_generate_every) == 0)
                ):
                    try:
                        hf_dbg = hf
                        rows_dbg = rows
                        span_ids_dbg = rows_dbg.span_token_ids.to(device=hf_dbg.get_input_embeddings().weight.device)
                        emb_w = hf_dbg.get_input_embeddings().weight
                        lm_w = hf_dbg.lm_head.weight
                        old_emb_rows = emb_w[span_ids_dbg].detach().clone()
                        old_lm_rows = lm_w[span_ids_dbg].detach().clone()
                        with torch.no_grad():
                            emb_w[span_ids_dbg] = rows_dbg.span_in.to(device=emb_w.device, dtype=emb_w.dtype)
                            lm_w[span_ids_dbg] = rows_dbg.span_out.to(device=lm_w.device, dtype=lm_w.dtype)

                        prefix_ids = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
                        with torch.inference_mode():
                            was_train = hf_dbg.training
                            old_cache = getattr(hf_dbg.config, "use_cache", None)
                            hf_dbg.eval()
                            if hasattr(hf_dbg, "config"):
                                hf_dbg.config.use_cache = True

                            n_show = min(int(args.debug_generate_num_samples), ids_padded.size(0))
                            for bi in range(n_show):
                                ids_list = ids_padded[bi].detach().cpu().tolist()
                                starts = _find_subsequence_starts(ids_list, prefix_ids)
                                if starts:
                                    start = starts[0] + len(prefix_ids)
                                else:
                                    start = int(attn_padded[bi].sum().item())
                                start = max(1, min(start, int(attn_padded.size(1))))

                                prompt_ids = ids_padded[bi : bi + 1, :start]
                                prompt_emb = inputs_embeds[bi : bi + 1, :start, :]
                                prompt_attn = torch.ones((1, start), dtype=torch.long, device=prompt_emb.device)

                                gen_ids = hf_dbg.generate(
                                    inputs_embeds=prompt_emb,
                                    attention_mask=prompt_attn,
                                    max_new_tokens=int(args.debug_generate_max_new_tokens),
                                    do_sample=False,
                                    num_beams=1,
                                )
                                gen_txt = tokenizer.decode(gen_ids[0].tolist(), skip_special_tokens=False)

                                print(f"\n[debug-generate] epoch={epoch} global_step={global_step} sample={bi}", flush=True)
                                print(f"[debug-generate] best_k={int(best_k[bi].item())} best_iou={float(best_iou[bi].item()):.3f}", flush=True)
                                if args.debug_print_prompt:
                                    prompt_txt = tokenizer.decode(prompt_ids[0].tolist(), skip_special_tokens=False)
                                    if len(prompt_txt) > int(args.debug_prompt_chars):
                                        prompt_txt = prompt_txt[: int(args.debug_prompt_chars)] + " ...[trunc]"
                                    print("=== prompt (trunc) ===", flush=True)
                                    print(prompt_txt, flush=True)
                                print("=== generated ===", flush=True)
                                print(gen_txt, flush=True)

                            # restore train/eval & cache flag
                            if hasattr(hf_dbg, "config") and old_cache is not None:
                                hf_dbg.config.use_cache = old_cache
                            if was_train:
                                hf_dbg.train()
                    finally:
                        try:
                            with torch.no_grad():
                                emb_w[span_ids_dbg] = old_emb_rows.to(device=emb_w.device, dtype=emb_w.dtype)
                                lm_w[span_ids_dbg] = old_lm_rows.to(device=lm_w.device, dtype=lm_w.dtype)
                        except Exception:
                            pass

                # =========================
                # =========================
                lm_out = hf.model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attn_padded,
                    return_dict=True,
                    use_cache=False,
                )
                hidden = lm_out.last_hidden_state  # [B, L, D]

                shift_labels = labels[:, 1:].contiguous()      # [B, L-1]
                shift_hidden = hidden[:, :-1, :].contiguous()  # [B, L-1, D]
                active_mask = shift_labels != -100
                active_labels = shift_labels[active_mask].long()     # [N]
                active_hidden = shift_hidden[active_mask]            # [N, D]

                if active_labels.numel() == 0:
                    loss_lm = torch.zeros([], device=device, dtype=torch.float32)
                    loss_lm_span = torch.zeros([], device=device, dtype=torch.float32)
                    loss_lm_time = torch.zeros([], device=device, dtype=torch.float32)
                    span_tok_count = 0
                    time_tok_count = 0
                else:
                    time_shift_mask = torch.zeros_like(shift_labels, dtype=torch.bool)
                    if args.time_value_weight is not None and args.time_value_weight > 1:
                        ids_cpu = ids_padded.detach().cpu().tolist()
                        labels_cpu = labels.detach().cpu().tolist()
                        for b in range(len(msgs)):
                            xy = _extract_time_strings(answers[b])
                            if xy is None:
                                continue
                            sup = [t != -100 for t in labels_cpu[b]]
                            for num_str in xy:
                                for sub_ids in _encode_text_variants(tokenizer, num_str):
                                    for st in _find_subsequence_starts(ids_cpu[b], sub_ids):
                                        idxs = list(range(st, st + len(sub_ids)))
                                        if not all(sup[p] for p in idxs):
                                            continue
                                        for p in idxs:
                                            if p > 0:
                                                time_shift_mask[b, p - 1] = True

                    logits = hf.lm_head(active_hidden).float()  # [N, V]
                    span_ids = rows.span_token_ids.to(device=logits.device)
                    span_logits = active_hidden.float() @ rows.span_out.to(device=active_hidden.device).t()  # [N, K]
                    logits.index_copy_(1, span_ids, span_logits)

                    loss_per = F.cross_entropy(logits, active_labels, reduction="none")  # [N]
                    is_span = (active_labels.unsqueeze(1) == span_ids.unsqueeze(0)).any(dim=1)
                    is_time = time_shift_mask[active_mask] if (args.time_value_weight is not None and args.time_value_weight > 1) else torch.zeros_like(is_span)
                    w = torch.ones_like(loss_per)
                    if args.span_id_weight is not None and args.span_id_weight > 1:
                        w[is_span] = float(args.span_id_weight)
                    if args.time_value_weight is not None and args.time_value_weight > 1 and is_time.any():
                        w_time = w.new_tensor(float(args.time_value_weight))
                        w[is_time] = torch.maximum(w[is_time], w_time)
                    loss_lm = (loss_per * w).sum() / (w.sum() + 1e-6)
                    loss_lm_span = loss_per[is_span].mean() if is_span.any() else torch.zeros([], device=device, dtype=torch.float32)
                    loss_lm_time = loss_per[is_time].mean() if is_time.any() else torch.zeros([], device=device, dtype=torch.float32)
                    span_tok_count = int(is_span.long().sum().item())
                    time_tok_count = int(is_time.long().sum().item())

                loss_prop = torch.zeros_like(loss_lm)
                if args.lambda_prop > 0:
                    loss_dict = proposal_loss_single_gt(
                        prop["start"],
                        prop["end"],
                        prop["score"].logit(),  # logits
                        gt_spans,
                    )
                    loss_prop = loss_dict["loss"]

                loss = loss_lm + args.lambda_prop * loss_prop

                if args.deepspeed:
                    is_boundary = llm_engine.is_gradient_accumulation_boundary()
                    if global_step % max(1, int(args.grad_acc_steps)) == 0:
                        optimizer_span.zero_grad()
                    llm_engine.backward(loss)
                    llm_engine.step()
                    if is_boundary:
                        optimizer_span.step()
                        opt_step += 1
                        _apply_lr(step=opt_step)
                else:
                    loss_to_back = loss / args.grad_acc_steps
                    loss_to_back.backward()
                    if (global_step + 1) % args.grad_acc_steps == 0:
                        optimizer_span.step()
                        llm_optimizer.step()
                        optimizer_span.zero_grad()
                        llm_optimizer.zero_grad()
                        opt_step += 1
                        _apply_lr(step=opt_step)

                global_step += 1
                progress["global_step"] = global_step
                progress["opt_step"] = opt_step
                if global_step % args.log_interval == 0 and is_main:
                    print(
                        f"[epoch {epoch} step {global_step}] loss={loss.item():.4f} "
                        f"lm={loss_lm.item():.4f} prop={loss_prop.item():.4f} "
                        f"best_iou={best_iou.mean().item():.3f}"
                    )
                    if use_swan:
                        swanlab.log(
                            {
                                "loss/total": loss.item(),
                                "loss/lm": loss_lm.item(),
                                "loss/prop": loss_prop.item(),
                                "loss/lm_span": float(loss_lm_span.item()),
                                "loss/lm_time": float(loss_lm_time.item()),
                                "stat/span_tok_count": span_tok_count,
                                "stat/time_tok_count": time_tok_count,
                                "stat/best_iou": best_iou.mean().item(),
                                "stat/opt_step": opt_step,
                                "stat/lr_mult": _lr_mult(opt_step),
                                "stat/lr_llm": float(llm_optimizer.param_groups[0]["lr"]) if llm_optimizer.param_groups else 0.0,
                                "stat/lr_span": float(optimizer_span.param_groups[0]["lr"]) if optimizer_span.param_groups else 0.0,
                                "stat/lr_stage1": float(optimizer_span.param_groups[1]["lr"]) if len(optimizer_span.param_groups) > 1 else 0.0,
                            },
                            step=global_step,
                        )
                if args.max_steps and global_step >= args.max_steps:
                    break
                if is_main and save_interval_opt is not None and opt_step > 0 and (opt_step % save_interval_opt == 0):
                    save_model_checkpoint(args, llm_module(), processor, tokenizer, stage3_model, backbone, opt_step_label=f"optstep_{opt_step}")

            if args.max_steps and global_step >= args.max_steps:
                break
            if is_main:
                save_model_checkpoint(args, llm_module(), processor, tokenizer, stage3_model, backbone, opt_step_label=f"epoch_{epoch}")

        if is_main:
            save_model_checkpoint(args, llm_module(), processor, tokenizer, stage3_model, backbone, opt_step_label=None)
    except KeyboardInterrupt:
        reason = signal_state["name"] or "interrupt"
        _safe_recovery_save(reason)
        raise
    except Exception as e:
        _safe_recovery_save(type(e).__name__)
        raise
    finally:
        if use_swan:
            swanlab.finish()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
