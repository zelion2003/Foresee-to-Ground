#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from stage2.model_stage2 import (
    Stage1BackboneWrapper,
    Stage2VTGModel,
)


def topk_by_score(t_start: torch.Tensor, t_end: torch.Tensor, logits: torch.Tensor, k: int = 8):
    scores = logits.sigmoid()  # [B, T]
    T = scores.size(1)
    k_eff = min(k, T)
    topk_score, topk_idx = scores.topk(k_eff, dim=1)
    gather = lambda x: torch.gather(x, 1, topk_idx)
    return {
        "score": topk_score,
        "idx": topk_idx,
        "start": gather(t_start),
        "end": gather(t_end),
    }


def _interval_iou_pair(pred_s: torch.Tensor, pred_e: torch.Tensor, gt_s: torch.Tensor, gt_e: torch.Tensor, eps=1e-6):
    inter = torch.clamp(torch.min(pred_e, gt_e) - torch.max(pred_s, gt_s), min=0.0)
    union = torch.clamp(torch.max(pred_e, gt_e) - torch.min(pred_s, gt_s), min=eps)
    iou = inter / (union + eps)
    enc = torch.max(pred_e, gt_e) - torch.min(pred_s, gt_s)
    giou = iou - (enc - union) / (enc + eps)
    return iou, giou


class HTRE(nn.Module):
    

    def __init__(self, d_model: int, num_queries: int = 2, num_heads: int = 8):
        super().__init__()
        self.num_queries = num_queries
        self.Q = nn.Parameter(torch.randn(num_queries, d_model) / (d_model ** 0.5))
        self.ln_q = nn.LayerNorm(d_model)
        self.ln_kv = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)

    def forward(self, H: torch.Tensor, start_idx: torch.Tensor, end_idx: torch.Tensor) -> torch.Tensor:
        B, T, D = H.shape
        K = start_idx.shape[1]
        M = self.num_queries
        outputs = []
        for k in range(K):
            slices = []
            lengths = []
            for b in range(B):
                s = int(start_idx[b, k].item())
                e = int(end_idx[b, k].item())
                s = max(0, min(T, s))
                e = max(s + 1, min(T, e))
                slices.append(H[b, s:e, :])  # [Lk,D]
                lengths.append(e - s)
            Lmax = max(lengths)
            padded = []
            kpm = []
            for sl in slices:
                pad_len = Lmax - sl.size(0)
                if pad_len > 0:
                    pad = torch.zeros(pad_len, D, device=H.device, dtype=H.dtype)
                    padded.append(torch.cat([sl, pad], dim=0))
                    kpm.append(torch.cat([torch.zeros(sl.size(0), device=H.device, dtype=torch.bool),
                                          torch.ones(pad_len, device=H.device, dtype=torch.bool)]))
                else:
                    padded.append(sl)
                    kpm.append(torch.zeros(sl.size(0), device=H.device, dtype=torch.bool))
            Hk = torch.stack(padded, dim=0)  # [B,Lmax,D]
            key_mask = torch.stack(kpm, dim=0)  # [B,Lmax]

            Qb = self.Q.unsqueeze(0).expand(B, M, D)
            Qb = self.ln_q(Qb)
            Hk = self.ln_kv(Hk)
            Zk, _ = self.attn(Qb, Hk, Hk, key_padding_mask=key_mask)  # [B,M,D]
            outputs.append(Zk)

        Z = torch.stack(outputs, dim=1)  # [B,K,M,D]
        return Z


class SpanProjector(nn.Module):
    

    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.proj = nn.Linear(d_in, d_out)
        if d_in == d_out:
            nn.init.eye_(self.proj.weight)
            nn.init.zeros_(self.proj.bias)

    def forward(self, span_tokens: torch.Tensor) -> torch.Tensor:
        # span_tokens: [B, K, M, D_in]
        B, K, M, D = span_tokens.shape
        out = self.proj(span_tokens.view(B, K * M, D))
        return out.view(B, K, M, -1)


def proposal_loss_single_gt(
    t_start: torch.Tensor,
    t_end: torch.Tensor,
    logits: torch.Tensor,
    gt_span: torch.Tensor,
    lambda_cls: float = 1.0,
    lambda_l1: float = 1.0,
    lambda_iou: float = 1.0,
    tau_neg: float = 0.1,
    gamma_neg: float = 0.1,
):
    
    device = t_start.device
    B, T = t_start.shape
    pos_count = torch.zeros([], device=device)
    neg_count = torch.zeros([], device=device)
    pos_loss = torch.zeros([], device=device)
    neg_loss = torch.zeros([], device=device)
    l1_sum = torch.zeros([], device=device)
    giou_sum = torch.zeros([], device=device)
    for b in range(B):
        gt_s = gt_span[b, 0]
        gt_e = gt_span[b, 1]
        iou_mat, giou_mat = _interval_iou_pair(t_start[b], t_end[b], gt_s, gt_e)
        best_idx = torch.argmax(iou_mat)
        best_iou = iou_mat[best_idx]
        best_giou = giou_mat[best_idx]
        pos_logits = logits[b, best_idx]
        pos_loss = pos_loss + F.binary_cross_entropy_with_logits(pos_logits, torch.ones_like(pos_logits), reduction="sum")
        l1_sum = l1_sum + (t_start[b, best_idx] - gt_s).abs() + (t_end[b, best_idx] - gt_e).abs()
        giou_sum = giou_sum + (1.0 - best_giou)
        pos_count = pos_count + 1.0
        neg_mask = iou_mat < tau_neg
        neg_mask[best_idx] = False
        if neg_mask.any() and gamma_neg > 0:
            neg_logits = logits[b, neg_mask]
            neg_loss = neg_loss + F.binary_cross_entropy_with_logits(
                neg_logits, torch.zeros_like(neg_logits), reduction="sum"
            )
            neg_count = neg_count + float(neg_logits.numel())

    cls_denom = pos_count + gamma_neg * neg_count + 1e-6
    loss_cls = (pos_loss + gamma_neg * neg_loss) / cls_denom
    reg_denom = pos_count.clamp(min=1.0)
    loss_l1 = l1_sum / reg_denom
    loss_giou = giou_sum / reg_denom
    # loss = lambda_cls * loss_cls + lambda_l1 * loss_l1 + lambda_iou * loss_giou
    loss = lambda_l1 * loss_l1 + lambda_iou * loss_giou
    return {
        "loss": loss,
        "loss_cls": loss_cls.detach(),
        "loss_l1": loss_l1.detach(),
        "loss_giou": loss_giou.detach(),
        "pos": pos_count.detach(),
        "neg": neg_count.detach(),
    }


class Stage3SpanModel(nn.Module):
    

    def __init__(
        self,
        stage2_backbone: Stage1BackboneWrapper,
        d_hidden: int,
        dec_layers: int,
        reg_type: str,
        alpha: float,
        beta: float,
        gamma_dist: float,
        htre_heads: int,
        span_projector_dim: int,
        k_top: int = 8,
    ):
        super().__init__()
        self.k_top = k_top
        self.stage2 = Stage2VTGModel(
            backbone=stage2_backbone,
            d_hidden=d_hidden,
            dec_layers=dec_layers,
            reg_type=reg_type,
            alpha=alpha,
            beta=beta,
            gamma=gamma_dist,
        )
        d_model = stage2_backbone.hidden_size
        self.htre = HTRE(d_model=d_model, num_queries=2, num_heads=htre_heads)
        self.span_proj = SpanProjector(d_in=d_model, d_out=span_projector_dim)

    def forward(self, H_base: torch.Tensor):
        
        out = self.stage2(H_base)
        prop = topk_by_score(out["t_start"], out["t_end"], out["logits"], k=self.k_top)
        B, T, _ = out["H"].shape
        start_idx = torch.clamp((prop["start"] * T).floor().long(), min=0, max=T - 1)
        end_idx = torch.clamp((prop["end"] * T).ceil().long(), min=1, max=T)
        end_idx = torch.max(end_idx, start_idx + 1)
        span_tokens = self.htre(out["H"], start_idx, end_idx)  # [B,K,M,D_model]
        span_embeds = self.span_proj(span_tokens)              # [B,K,M,D_proj]
        prop["H"] = out["H"]
        prop["span_embeds"] = span_embeds
        if hasattr(self, "span_token_rows"):
            rows = getattr(self, "span_token_rows")
            prop["span_token_ids"] = rows.span_token_ids
            prop["span_in"] = rows.span_in
            prop["span_out"] = rows.span_out
            prop["_ddp_span_rows_dummy"] = (rows.span_in.sum() + rows.span_out.sum()) * 0.0
        return prop
