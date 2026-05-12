#!/usr/bin/env python3
# -*- coding: utf-8 -*-




import json
import os
import re
import sys
from pathlib import Path
from typing import Tuple, Optional, List

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

# --- Compatibility shims for torch 2.1.x + newer transformers ---
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
# ---------------------------------------------------------------

from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from transformers import AutoTokenizer
from transformers import AutoConfig


REPO_ROOT = Path(__file__).resolve().parents[2]
STAGE3_MODEL_DIR = str(REPO_ROOT / "outputs" / "stage3")
PROCESSOR_ID = "Qwen/Qwen3-VL-8B-Instruct"
LOCAL_FILES_ONLY = True
DATA_PATH = str(REPO_ROOT / "data" / "annotations" / "timelens" / "charades_timelens_test.json")
VIDEO_ROOT = str(REPO_ROOT / "data" / "videos" / "timelens_bench_336")
SAVE_PATH = str(REPO_ROOT / "outputs" / "eval" / "stage3_eval.json")
TORCH_DTYPE = "bf16"  # "bf16" / "fp16" / "auto"
DEVICE_MAP = "cuda:0"
ATTN_IMPL = "flash_attention_2"
K_TOP = 8
M_QUERIES = 4
MAX_NEW_TOKENS = 128
DO_SAMPLE = True
TEMPERATURE = 0.7
TOP_P = 0.8
TOP_K = 20
ADD_SPAN_FIELDS = True
ADD_RESPONSE_RAW = True
SAMPLE_LIMIT: Optional[int] = None


# =========================
# =========================

_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

from stage3.model_stage3 import Stage1BackboneWrapper, Stage3SpanModel  # noqa: E402
from stage3.prompt_stage3 import render_prompt_b, find_insert_pos  # noqa: E402


def load_model_and_processor(
    model_id: str,
    torch_dtype: str = "auto",
    device_map: str = "auto",
    attn_implementation: str = "flash_attention_2",
    local_files_only: bool = True,
):
    
    if torch_dtype == "bf16":
        dtype = torch.bfloat16
    elif torch_dtype == "fp16":
        dtype = torch.float16
    else:
        dtype = "auto"

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id,
        dtype=dtype,
        device_map=device_map,
        attn_implementation=attn_implementation,
        local_files_only=local_files_only,
    )
    processor = AutoProcessor.from_pretrained(model_id, local_files_only=local_files_only)
    return model, processor


def _get_model_device(model) -> torch.device:
    try:
        return model.device
    except Exception:
        pass
    return next(model.parameters()).device


def _infer_dtype(torch_dtype: str):
    if torch_dtype == "bf16":
        return torch.bfloat16
    if torch_dtype == "fp16":
        return torch.float16
    return "auto"


def _normalize_device_map(device_map):
    
    if isinstance(device_map, str):
        if device_map == "auto":
            return "auto"
        if device_map.startswith("cuda") or device_map == "cpu":
            return {"": device_map}
    return device_map


def _extract_last_span_token_id(raw_text: str) -> Optional[int]:
    
    ms = re.findall(r"<Span_(\d+)>", raw_text)
    if ms:
        try:
            return int(ms[-1])
        except Exception:
            return None
    cs = re.findall(r"candidate\s*(\d+)", raw_text, flags=re.IGNORECASE)
    if cs:
        try:
            cand = int(cs[-1])
            return cand - 1  # Candidate 1 <-> <Span_0>
        except Exception:
            return None
    return None


def _build_question_prompt(query: str, k: int) -> str:
    span_list = " ".join([f"<Span_{i}>" for i in range(k)])
    return (
        f"Question: During which time can we see {query}? "
        "Please answer in seconds in the form: From X seconds to Y seconds. "
        f"Finally cite exactly ONE candidate span id token from: {span_list}."
    )


def _make_conv(video_path: str, user_text: str):
    return [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": video_path},
                {"type": "text", "text": user_text},
            ],
        }
    ]


def _splice_ids_masks(
    input_ids_a: torch.Tensor,
    attn_a: torch.Tensor,
    input_ids_b: torch.Tensor,
    attn_b: torch.Tensor,
    insert_pos: int,
):
    ids = torch.cat([input_ids_a[:insert_pos], input_ids_b, input_ids_a[insert_pos:]], dim=0)
    attn = torch.cat([attn_a[:insert_pos], attn_b, attn_a[insert_pos:]], dim=0)
    return ids, attn


def load_stage3_models():
    
    if not STAGE3_MODEL_DIR:
        raise ValueError("Configure STAGE3_MODEL_DIR before running evaluation.")

    dtype = _infer_dtype(TORCH_DTYPE)
    device_map = _normalize_device_map(DEVICE_MAP)
    cfg = AutoConfig.from_pretrained(STAGE3_MODEL_DIR, local_files_only=LOCAL_FILES_ONLY)
    try:
        if hasattr(cfg, "vision_config") and hasattr(cfg.vision_config, "deepstack_visual_indexes"):
            cfg.vision_config.deepstack_visual_indexes = []
    except Exception:
        pass
    llm = Qwen3VLForConditionalGeneration.from_pretrained(
        STAGE3_MODEL_DIR,
        config=cfg,
        dtype=dtype,
        device_map=device_map,
        attn_implementation=ATTN_IMPL,
        local_files_only=LOCAL_FILES_ONLY,
    )
    llm.eval()
    llm.config.use_cache = True
    processor = AutoProcessor.from_pretrained(PROCESSOR_ID, local_files_only=LOCAL_FILES_ONLY)
    tokenizer = AutoTokenizer.from_pretrained(STAGE3_MODEL_DIR, use_fast=True, local_files_only=LOCAL_FILES_ONLY)
    processor.tokenizer = tokenizer
    sp_id = tokenizer.convert_tokens_to_ids("<span_pad>")
    if hasattr(tokenizer, "unk_token_id") and sp_id == tokenizer.unk_token_id:
        print("[warn] The tokenizer does not appear to include <span_pad>. Check whether the saved tokenizer contains the Stage-3 span tokens.")

    span_path = os.path.join(STAGE3_MODEL_DIR, "stage3_span.pt")
    blob = torch.load(span_path, map_location="cpu")
    span_state = blob["stage3_span"]
    tokens_per_frame = blob.get("tokens_per_frame", None)
    d_in = blob.get("d_in", None)
    cfg = blob.get("config", {}) or {}

    device = _get_model_device(llm)
    # d_model
    d_model = None
    w_attn = span_state.get("stage2.backbone.backbone.blocks.0.attn.in_proj_weight")
    if w_attn is not None:
        d_model = int(w_attn.shape[1])
    # num_layers
    import re as _re

    blk_ids = []
    for k in span_state.keys():
        m = _re.match(r"stage2\\.backbone\\.backbone\\.blocks\\.(\\d+)\\.", k)
        if m:
            blk_ids.append(int(m.group(1)))
    num_layers = (max(blk_ids) + 1) if blk_ids else 2
    # kernel_size
    kernel_size = None
    w_dw = span_state.get("stage2.backbone.backbone.blocks.0.dw_conv.weight")
    if w_dw is not None and w_dw.ndim == 3:
        kernel_size = int(w_dw.shape[-1])
    stage2_cfg = {
        "d_in": d_in,
        "tokens_per_frame": tokens_per_frame,
        "d_model": d_model or d_in,
        "num_layers": num_layers,
        "num_heads": 8,
        "kernel_size": kernel_size or 11,
    }
    backbone = Stage1BackboneWrapper(
        device=device,
        ckpt_path=None,
        freeze=True,
        d_in=d_in,
        tokens_per_frame=tokens_per_frame,
        stage2_config=stage2_cfg,
    )
    llm_hidden = int(llm.get_input_embeddings().weight.shape[1])
    stage2_ckpt = cfg.get("stage2_ckpt", None)
    stage2_cfg_all = {}
    if isinstance(stage2_ckpt, str) and stage2_ckpt and os.path.isfile(stage2_ckpt):
        try:
            _b = torch.load(stage2_ckpt, map_location="cpu")
            stage2_cfg_all = _b.get("config", {}) or _b.get("args", {}) or {}
        except Exception:
            stage2_cfg_all = {}

    # d_hidden
    d_hidden = stage2_cfg_all.get("d_hidden", None)
    if d_hidden is None:
        w_stem = span_state.get("stage2.decoder.stem.1.weight")
        d_hidden = int(w_stem.shape[0]) if w_stem is not None else 512
    d_hidden = int(d_hidden)
    # dec_layers
    dec_layers = stage2_cfg_all.get("dec_layers", None)
    if dec_layers is None:
        conv_ids = []
        for k in span_state.keys():
            m = _re.match(r"stage2\\.decoder\\.conv_blocks\\.(\\d+)\\.", k)
            if m:
                conv_ids.append(int(m.group(1)))
        dec_layers = (max(conv_ids) + 1) if conv_ids else 2
    dec_layers = int(dec_layers)
    # reg_type / alpha / beta / gamma
    reg_type = str(stage2_cfg_all.get("head_type", stage2_cfg_all.get("reg_type", "B")) or "B")
    alpha = float(stage2_cfg_all.get("alpha", 0.5) or 0.5)
    beta = float(stage2_cfg_all.get("beta", 1.0) or 1.0)
    gamma_dist = float(stage2_cfg_all.get("gamma_dist", stage2_cfg_all.get("gamma", 0.5)) or 0.5)
    k_top = int(cfg.get("k_top", K_TOP) or K_TOP)

    stage3_model = Stage3SpanModel(
        stage2_backbone=backbone,
        d_hidden=d_hidden,
        dec_layers=dec_layers,
        reg_type=reg_type,
        alpha=alpha,
        beta=beta,
        gamma_dist=gamma_dist,
        htre_heads=8,
        span_projector_dim=int(llm_hidden),
        k_top=k_top,
    ).to(device)
    missing = stage3_model.load_state_dict(span_state, strict=False)
    if getattr(missing, "missing_keys", None):
        print(f"[load_stage3_models] missing_keys={len(missing.missing_keys)}")
    if getattr(missing, "unexpected_keys", None):
        print(f"[load_stage3_models] unexpected_keys={len(missing.unexpected_keys)}")
    stage3_model.eval()
    try:
        span_in = span_state.get("span_token_rows.span_in", None)
        if span_in is None:
            span_in = span_state.get("module.span_token_rows.span_in", None)
        span_out = span_state.get("span_token_rows.span_out", None)
        if span_out is None:
            span_out = span_state.get("module.span_token_rows.span_out", None)
        if span_in is not None and span_out is not None:
            span_ids = [tokenizer.convert_tokens_to_ids(f"<Span_{i}>") for i in range(k_top)]
            if hasattr(tokenizer, "unk_token_id") and any(t == tokenizer.unk_token_id for t in span_ids):
                raise ValueError("Could not find <Span_k> tokens in the tokenizer. Check whether the saved tokenizer contains the Stage-3 span tokens.")
            span_ids_t = torch.tensor(span_ids, dtype=torch.long, device=llm.get_input_embeddings().weight.device)
            with torch.no_grad():
                emb_w = llm.get_input_embeddings().weight
                emb_w[span_ids_t] = span_in.to(device=emb_w.device, dtype=emb_w.dtype)
                lm_w = llm.lm_head.weight
                lm_w[span_ids_t] = span_out.to(device=lm_w.device, dtype=lm_w.dtype)
            print(f"[load_stage3_models] Restored embedding and lm_head rows for <Span_0..{k_top-1}> from stage3_span.pt.")
        else:
            print("[load_stage3_models] stage3_span.pt does not include span_token_rows.*. Falling back to the rows stored in the HF checkpoint.")
    except Exception as e:
        print(f"[warn] Failed to restore the saved <Span_k> rows: {type(e).__name__}: {e}")

    return llm, processor, stage3_model


def extract_span_seconds(text: str, mode: str, sec_per_index: float = 1.0) -> Optional[Tuple[float, float]]:
    
    text = text.strip().lower()

    if mode == "frame":
        pat = r"from\s*(?:frame\s*)?(\d+(?:\.\d+)?)\s*(?:to|-|~)\s*(?:frame\s*)?(\d+(?:\.\d+)?)"
        m = re.search(pat, text)
        if not m:
            nums = re.findall(r"\d+(?:\.\d+)?", text)
            if len(nums) >= 2:
                start_f, end_f = float(nums[0]), float(nums[1])
            else:
                return None
        else:
            start_f, end_f = float(m.group(1)), float(m.group(2))
        return min(start_f, end_f) * sec_per_index, max(start_f, end_f) * sec_per_index
    pat = r"from\s*(\d+(?:\.\d+)?)\s*(?:s|sec|seconds)?\s*(?:to|-|~)\s*(\d+(?:\.\d+)?)"
    m = re.search(pat, text)
    if not m:
        nums = re.findall(r"\d+(?:\.\d+)?", text)
        if len(nums) >= 2:
            start_s, end_s = float(nums[0]), float(nums[1])
        else:
            return None
    else:
        start_s, end_s = float(m.group(1)), float(m.group(2))
    return min(start_s, end_s), max(start_s, end_s)


def calc_iou_sec(pred: Tuple[float, float], gt: Tuple[float, float]) -> float:
    
    ps, pe = pred
    gs, ge = gt
    inter = max(0.0, min(pe, ge) - max(ps, gs))
    union = max(pe, ge) - min(ps, gs)
    if union <= 0:
        return 0.0
    return inter / union


def run_eval_stage3() -> List[dict]:
    
    llm, processor, stage3_model = load_stage3_models()
    tokenizer = processor.tokenizer
    device = _get_model_device(llm)
    k_top = int(getattr(stage3_model, "k_top", K_TOP) or K_TOP)

    with open(DATA_PATH, "r") as f:
        items = json.load(f)

    results = []
    if os.path.exists(SAVE_PATH):
        with open(SAVE_PATH, "r") as f:
            try:
                results = json.load(f)
            except Exception:
                results = []
    done_ids = {r["id"] for r in results}

    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

    for idx, sample in enumerate(items):
        if SAMPLE_LIMIT and idx >= SAMPLE_LIMIT:
            break
        if sample["id"] in done_ids:
            continue

        video_path = os.path.join(VIDEO_ROOT, sample["video"])
        if not os.path.exists(video_path):
            print(f"[warn][missing video] {video_path}")
            continue

        duration = float(sample.get("duration") or 0.0)
        gt_start, gt_end = sample.get("start_time"), sample.get("end_time")
        user_prompt = _build_question_prompt(sample["query"], k=k_top)
        conv = _make_conv(video_path, user_prompt)
        try:
            text_a = processor.apply_chat_template(
                conv,
                tokenize=False,
                add_generation_prompt=True,
            )
            model_inputs_a = processor(
                text=[text_a],
                videos=[video_path],
                padding=True,
                return_tensors="pt",
            )
        except Exception as e:
            print(f"[warn] apply_chat_template or video processing failed, skipping sample {sample['id']}: {type(e).__name__}: {e}")
            continue
        input_ids_a = model_inputs_a["input_ids"][0].cpu()
        attn_a = model_inputs_a["attention_mask"][0].cpu()
        pixel_values = model_inputs_a.get("pixel_values_videos", None)
        grid_thw = model_inputs_a.get("video_grid_thw", None)
        if pixel_values is None or grid_thw is None:
            print(f"[warn] The processor did not return a video tensor for sample {sample['id']}.")
            continue
        pixel_values = pixel_values.to(device)
        grid_thw = grid_thw.to(device)
        try:
            with torch.inference_mode():
                embeds_list, _ = llm.get_video_features(pixel_values, video_grid_thw=grid_thw)
        except Exception as e:
            print(f"[warn] get_video_features failed, skipping sample {sample['id']}: {type(e).__name__}: {e}")
            continue
        H_base = pad_sequence(embeds_list, batch_first=True).to(device).float()  # [1, L, D_in]
        expected_d_in = getattr(stage3_model.stage2.backbone, "d_in", None)
        if expected_d_in is not None and H_base.shape[-1] != int(expected_d_in):
            raise ValueError(
                f"Qwen visual feature width mismatch: H_base[-1]={H_base.shape[-1]} vs backbone.d_in={int(expected_d_in)}. "
                "Make sure inference uses the merger-projected output (vision_config.out_hidden_size) and matches Stage-3 training."
            )
        merge_size = llm.model.visual.spatial_merge_size
        tpf_cur = int((grid_thw[0, 1] * grid_thw[0, 2]) // (merge_size ** 2))
        tpf_prev = getattr(stage3_model.stage2.backbone, "tokens_per_frame", None)
        if tpf_prev is not None and int(tpf_prev) != int(tpf_cur):
            print(
                f"[warn] tokens_per_frame changed: {int(tpf_prev)} -> {int(tpf_cur)}. "
                f"id={sample['id']} grid_thw0={grid_thw[0].detach().cpu().tolist()}"
            )
        stage3_model.stage2.backbone.tokens_per_frame = int(tpf_cur)

        with torch.inference_mode():
            prop = stage3_model(H_base)
        s_topk = prop["start"][0]  # [K]
        e_topk = prop["end"][0]    # [K]
        span_embeds = prop["span_embeds"][0]  # [K, M, D_llm]
        K = int(s_topk.numel())
        M = int(span_embeds.size(1))
        if K < k_top:
            remove_ids = []
            for i in range(K, k_top):
                tid = tokenizer.convert_tokens_to_ids(f"<Span_{i}>")
                if hasattr(tokenizer, "unk_token_id") and tid == tokenizer.unk_token_id:
                    continue
                remove_ids.append(int(tid))
            if remove_ids:
                keep = torch.ones_like(input_ids_a, dtype=torch.bool)
                for rid in remove_ids:
                    keep &= (input_ids_a != rid)
                input_ids_a = input_ids_a[keep]
                attn_a = attn_a[keep]
        s_sec = (s_topk * duration).cpu().view(1, K)
        e_sec = (e_topk * duration).cpu().view(1, K)
        prompt_b_list = render_prompt_b(s_sec, e_sec, k=K, m=M)  # list[str] len=1
        tok_b = tokenizer(prompt_b_list, add_special_tokens=False, return_tensors="pt")
        input_ids_b = tok_b["input_ids"][0]
        attn_b = tok_b["attention_mask"][0]

        insert_pos = find_insert_pos(input_ids_a.tolist(), tokenizer)
        ids, attn = _splice_ids_masks(input_ids_a, attn_a, input_ids_b, attn_b, insert_pos)
        ids = ids.unsqueeze(0).to(device)
        attn = attn.unsqueeze(0).to(device)
        inputs_embeds = llm.get_input_embeddings()(ids).clone()
        vid_id = tokenizer.convert_tokens_to_ids("<|video_pad|>")
        sp_id = tokenizer.convert_tokens_to_ids("<span_pad>")
        vid_pos = (ids[0] == vid_id).nonzero(as_tuple=False).squeeze(-1)
        vid_embed = embeds_list[0].to(device).to(inputs_embeds.dtype)
        if vid_pos.numel() != vid_embed.size(0):
            print(
                f"[warn] video_pad count mismatch: tokens={vid_pos.numel()} feats={vid_embed.size(0)} id={sample['id']}"
            )
            continue
        inputs_embeds[0, vid_pos, :] = vid_embed
        flat_span = span_embeds.reshape(K * M, -1).to(device).to(inputs_embeds.dtype)
        sp_pos = (ids[0] == sp_id).nonzero(as_tuple=False).squeeze(-1)
        if sp_pos.numel() != flat_span.size(0):
            print(
                f"[warn] span_pad count mismatch: tokens={sp_pos.numel()} feats={flat_span.size(0)} id={sample['id']}"
            )
            continue
        inputs_embeds[0, sp_pos, :] = flat_span

        # ===== 5) LLM generate =====
        with torch.inference_mode():
            gen_ids = llm.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attn,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=DO_SAMPLE,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                top_k=TOP_K,
            )
        prompt_len = ids.size(1)
        if gen_ids.size(1) > prompt_len:
            gen_trim = gen_ids[:, prompt_len:]
        else:
            gen_trim = gen_ids

        resp_clean = processor.batch_decode(
            gen_trim, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()
        resp_raw = processor.batch_decode(
            gen_trim, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )[0].strip()
        span = extract_span_seconds(resp_clean, mode="seconds", sec_per_index=1.0)
        pred_start, pred_end = (span if span else (0.0, 0.0))
        gt_start, gt_end = sample.get("start_time"), sample.get("end_time")
        iou = 0.0
        if gt_start is not None and gt_end is not None and span:
            iou = calc_iou_sec((pred_start, pred_end), (gt_start, gt_end))

        rec = {
            "id": sample["id"],
            "video": sample.get("video"),
            "query": sample.get("query"),
            "response": resp_clean,
            "pred_start_sec": pred_start,
            "pred_end_sec": pred_end,
            "gt_start": gt_start,
            "gt_end": gt_end,
            "duration": sample.get("duration"),
            "iou": iou,
        }
        if ADD_RESPONSE_RAW:
            rec["response_raw"] = resp_raw
        if ADD_SPAN_FIELDS:
            span_k = _extract_last_span_token_id(resp_raw)
            rec["pred_span_token"] = (f"<Span_{span_k}>" if span_k is not None else None)
            span_start_sec = None
            span_end_sec = None
            span_iou = None
            if span_k is not None and 0 <= span_k < K and gt_start is not None and gt_end is not None:
                span_start_sec = float(s_sec[0, span_k].item())
                span_end_sec = float(e_sec[0, span_k].item())
                span_iou = calc_iou_sec((span_start_sec, span_end_sec), (float(gt_start), float(gt_end)))
            rec["span_start_sec"] = span_start_sec
            rec["span_end_sec"] = span_end_sec
            rec["span_iou"] = span_iou

        results.append(rec)

        with open(SAVE_PATH, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[{len(results)}/{len(items)}] {sample['id']} -> IoU={iou:.3f}")

    return results


def main():
    run_eval_stage3()


if __name__ == "__main__":
    main()
