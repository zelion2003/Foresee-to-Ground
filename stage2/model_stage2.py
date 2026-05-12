from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from stage1.model_stage1 import Stage1LatentJEPA, FramePool, TemporalBackbone


class Stage1BackboneWrapper(nn.Module):
    

    def __init__(
        self,
        device: torch.device,
        ckpt_path: Optional[str] = None,
        freeze: bool = True,
        override_tokens_per_frame: Optional[int] = None,
        d_in: Optional[int] = None,
        tokens_per_frame: Optional[int] = None,
        stage2_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        if ckpt_path is not None:
            ckpt = torch.load(ckpt_path, map_location=device)
            args = ckpt.get("args", {}) or {}
            d_in_load = ckpt.get("d_in", args.get("d_in"))
            if d_in is None:
                d_in = d_in_load
            tokens_per_frame_load = ckpt.get("tokens_per_frame")
            tokens_per_frame = override_tokens_per_frame or tokens_per_frame or tokens_per_frame_load
            stage1 = Stage1LatentJEPA(
                d_in=d_in,
                d_model=args.get("d_model", d_in),
                n_heads=args.get("n_heads", 8),
                num_layers=args.get("num_layers", 2),
                kernel_size=args.get("kernel_size", 11),
                latent_dim=args.get("latent_dim", args.get("d_model", d_in)),
                lambda_sig=args.get("lambda_sig", 0.5),
                tokens_per_frame=tokens_per_frame,
            )
            missing = stage1.load_state_dict(ckpt["stage1"], strict=False)
            if missing.missing_keys:
                print(f"[Stage1BackboneWrapper] Missing keys: {missing.missing_keys}")
            if missing.unexpected_keys:
                print(f"[Stage1BackboneWrapper] Unexpected keys: {missing.unexpected_keys}")
            self.frame_pool = stage1.frame_pool
            self.input_proj = stage1.input_proj
            self.backbone = stage1.backbone
        else:
            if stage2_config is not None:
                d_in = d_in or stage2_config.get("d_in")
                tokens_per_frame = tokens_per_frame or stage2_config.get("tokens_per_frame")
                d_model = stage2_config.get("d_model", d_in)
                num_layers = stage2_config.get("num_layers", 2)
                num_heads = stage2_config.get("num_heads", 8)
                kernel_size = stage2_config.get("kernel_size", 11)
            if d_in is None:
                raise ValueError("One of d_in, ckpt_path, or stage2_config must be provided.")
            if tokens_per_frame is None and override_tokens_per_frame is not None:
                tokens_per_frame = override_tokens_per_frame
            self.frame_pool = FramePool(d_in=d_in)
            self.input_proj = nn.Identity()
            d_model = d_model if "d_model" in locals() else d_in
            num_layers = num_layers if "num_layers" in locals() else 2
            num_heads = num_heads if "num_heads" in locals() else 8
            kernel_size = kernel_size if "kernel_size" in locals() else 11
            self.backbone = TemporalBackbone(d_model=d_model, num_layers=num_layers, num_heads=num_heads, kernel_size=kernel_size)

        self.tokens_per_frame = tokens_per_frame
        self.d_in = d_in

        if freeze:
            for p in self.parameters():
                p.requires_grad = False
        self.to(device)

    @property
    def hidden_size(self) -> int:
        return self.backbone.blocks[0].attn.embed_dim

    def forward(self, H_base: torch.Tensor) -> torch.Tensor:
        
        if self.tokens_per_frame is None:
            raise ValueError("tokens_per_frame is not set. Provide it before calling forward().")
        H_frame = self.frame_pool(H_base, tokens_per_frame=self.tokens_per_frame)
        H_proj = self.input_proj(H_frame)
        H = self.backbone(H_proj, key_padding_mask=None)
        return H


class TemporalDecoderTrunk(nn.Module):
    

    def __init__(self, d_in: int, d_hidden: int = 512, num_layers: int = 2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, d_hidden),
            nn.GELU(),
        )
        self.conv_blocks = nn.ModuleList()
        num_groups = max(1, d_hidden // 32)
        for _ in range(num_layers):
            self.conv_blocks.append(
                nn.Sequential(
                    nn.Conv1d(d_hidden, d_hidden, kernel_size=3, padding=1, groups=1),
                    nn.GroupNorm(num_groups=num_groups, num_channels=d_hidden),
                    nn.GELU(),
                )
            )

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        
        F = self.stem(H)  # [B, T, D_h]
        U = F.transpose(1, 2)  # [B, D_h, T]
        for block in self.conv_blocks:
            U = block(U)  # [B, D_h, T]
        return U.transpose(1, 2)  # [B, T, D_h]


class ProposalHead(nn.Module):
    

    def __init__(
        self,
        d_hidden: int,
        reg_type: str = "A",
        alpha: float = 0.5,
        beta: float = 1.0,
        gamma: float = 0.5,
    ):
        super().__init__()
        assert reg_type in ("A", "B"), "reg_type must be either 'A' or 'B'."
        self.reg_type = reg_type
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.cls = nn.Sequential(
            nn.LayerNorm(d_hidden),
            nn.Linear(d_hidden, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, 1),
        )
        self.reg = nn.Sequential(
            nn.LayerNorm(d_hidden),
            nn.Linear(d_hidden, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, 2),
        )

    def forward(self, feats: torch.Tensor, H: torch.Tensor) -> Dict[str, torch.Tensor]:
        
        B, T, _ = feats.shape
        logits = self.cls(feats).squeeze(-1)  # [B, T]

        reg_out = self.reg(feats)  # [B, T, 2]
        tau = (torch.arange(T, device=feats.device, dtype=feats.dtype) + 0.5) / float(T)  # [T]
        tau = tau.unsqueeze(0).expand(B, T)  # [B, T]

        if self.reg_type == "A":
            delta = reg_out[..., 0]
            log_len = reg_out[..., 1]
            center = tau + self.alpha * torch.tanh(delta)  # [B, T]
            length = (self.beta / float(T)) * F.softplus(log_len)  # [B, T]
            t_start = torch.clamp(center - 0.5 * length, 0.0, 1.0)
            t_end = torch.clamp(center + 0.5 * length, 0.0, 1.0)
            extra = length
        else:
            d = self.gamma * F.softplus(reg_out)  # [B, T, 2], non-negative distances
            d_s, d_e = d[..., 0], d[..., 1]
            center = tau
            t_start = torch.clamp(center - d_s, 0.0, 1.0)
            t_end = torch.clamp(center + d_e, 0.0, 1.0)
            extra = d

        return {
            "t_start": t_start,
            "t_end": t_end,
            "logits": logits,
            "feature": H,
            "center": center,
            "extra": extra,
        }


class Stage2VTGModel(nn.Module):
    

    def __init__(
        self,
        backbone: Stage1BackboneWrapper,
        d_hidden: int = 512,
        dec_layers: int = 2,
        reg_type: str = "A",
        alpha: float = 0.5,
        beta: float = 1.0,
        gamma: float = 0.5,
    ):
        super().__init__()
        self.backbone = backbone
        self.decoder = TemporalDecoderTrunk(backbone.hidden_size, d_hidden=d_hidden, num_layers=dec_layers)
        self.head = ProposalHead(d_hidden=d_hidden, reg_type=reg_type, alpha=alpha, beta=beta, gamma=gamma)

    def forward(self, H_base: torch.Tensor) -> Dict[str, torch.Tensor]:
        
        H = self.backbone(H_base)          # [B, T, D_model]
        F = self.decoder(H)                # [B, T, D_h]
        head_out = self.head(F, H)         # proposals
        head_out["H"] = H
        head_out["F"] = F
        return head_out


# =====================
# =====================

def _pairwise_l1(pred_s: torch.Tensor, pred_e: torch.Tensor, gt_s: torch.Tensor, gt_e: torch.Tensor) -> torch.Tensor:
    
    pred = torch.stack([pred_s, pred_e], dim=-1)  # [P, 2]
    gt = torch.stack([gt_s, gt_e], dim=-1)        # [G, 2]
    diff = pred[:, None, :] - gt[None, :, :]      # [P, G, 2]
    return diff.abs().sum(dim=-1)                 # [P, G]


def _interval_iou(pred_s: torch.Tensor, pred_e: torch.Tensor, gt_s: torch.Tensor, gt_e: torch.Tensor, eps: float = 1e-6):
    
    inter = torch.clamp(torch.min(pred_e[:, None], gt_e[None, :]) - torch.max(pred_s[:, None], gt_s[None, :]), min=0.0)
    union = torch.clamp(torch.max(pred_e[:, None], gt_e[None, :]) - torch.min(pred_s[:, None], gt_s[None, :]), min=eps)
    iou = inter / (union + eps)
    enc = torch.max(pred_e[:, None], gt_e[None, :]) - torch.min(pred_s[:, None], gt_s[None, :])
    giou = iou - (enc - union) / (enc + eps)
    return iou, giou


def _giou_pair(pred_s: torch.Tensor, pred_e: torch.Tensor, gt_s: torch.Tensor, gt_e: torch.Tensor, eps: float = 1e-6):
    
    inter = torch.clamp(torch.min(pred_e, gt_e) - torch.max(pred_s, gt_s), min=0.0)
    union = torch.clamp(torch.max(pred_e, gt_e) - torch.min(pred_s, gt_s), min=eps)
    iou = inter / (union + eps)
    enc = torch.max(pred_e, gt_e) - torch.min(pred_s, gt_s)
    giou = iou - (enc - union) / (enc + eps)
    return giou


def hungarian_matching(cost: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    
    cost_np = cost.detach().cpu().numpy()
    try:
        from scipy.optimize import linear_sum_assignment  # type: ignore
        row_ind, col_ind = linear_sum_assignment(cost_np)
    except Exception:
        rows, cols = [], []
        cost_work = cost_np.copy()
        while len(rows) < min(cost_work.shape):
            r, c = np.unravel_index(np.argmin(cost_work), cost_work.shape)
            if np.isinf(cost_work[r, c]):
                break
            rows.append(r)
            cols.append(c)
            cost_work[r, :] = np.inf
            cost_work[:, c] = np.inf
        row_ind = np.array(rows, dtype=np.int64)
        col_ind = np.array(cols, dtype=np.int64)

    return torch.as_tensor(row_ind, device=cost.device), torch.as_tensor(col_ind, device=cost.device)


@torch.no_grad()
def build_cost_matrix(
    pred_s: torch.Tensor,
    pred_e: torch.Tensor,
    logits: torch.Tensor,
    gt_s: torch.Tensor,
    gt_e: torch.Tensor,
    lambda_cls: float,
    lambda_l1: float,
    lambda_iou: float,
) -> torch.Tensor:
    
    l1_mat = _pairwise_l1(pred_s, pred_e, gt_s, gt_e)  # [P, G]
    _, giou_mat = _interval_iou(pred_s, pred_e, gt_s, gt_e)  # [P, G]
    cls_prob = logits.sigmoid().unsqueeze(-1).expand_as(l1_mat)  # [P, G]
    cost = (
        -lambda_cls * cls_prob
        + lambda_l1 * l1_mat
        + lambda_iou * (1.0 - giou_mat)
    )  # [P, G]
    return cost.transpose(0, 1)  # [G, P]


def compute_vtg_losses(
    t_start: torch.Tensor,
    t_end: torch.Tensor,
    logits: torch.Tensor,
    gt_spans: Sequence[torch.Tensor],
    lambda_cls: float = 1.0,
    lambda_l1: float = 1.0,
    lambda_iou: float = 1.0,
    gamma: float = 0.1,
    tau_neg: float = 0.1,
) -> Dict[str, torch.Tensor]:
    
    B, T = t_start.shape
    device = t_start.device
    pos_count = torch.zeros([], device=device)
    neg_count = torch.zeros([], device=device)
    pos_loss_sum = torch.zeros([], device=device)
    neg_loss_sum = torch.zeros([], device=device)
    l1_sum = torch.zeros([], device=device)
    giou_sum = torch.zeros([], device=device)

    for b in range(B):
        gts = gt_spans[b]
        if gts.numel() == 0:
            continue

        gt_s = gts[:, 0]
        gt_e = gts[:, 1]

        cost = build_cost_matrix(
            t_start[b],
            t_end[b],
            logits[b],
            gt_s,
            gt_e,
            lambda_cls,
            lambda_l1,
            lambda_iou,
        )  # [G, P]
        gt_idx, pred_idx = hungarian_matching(cost)
        if gt_idx.numel() == 0:
            continue
        pos_logits = logits[b, pred_idx]
        pos_loss_sum = pos_loss_sum + F.binary_cross_entropy_with_logits(
            pos_logits,
            torch.ones_like(pos_logits),
            reduction="sum",
        )

        pred_s_pos = t_start[b, pred_idx]
        pred_e_pos = t_end[b, pred_idx]
        gt_s_pos = gt_s[gt_idx]
        gt_e_pos = gt_e[gt_idx]

        l1_sum = l1_sum + (pred_s_pos - gt_s_pos).abs().sum() + (pred_e_pos - gt_e_pos).abs().sum()
        giou_pos = _giou_pair(pred_s_pos, pred_e_pos, gt_s_pos, gt_e_pos)
        giou_sum = giou_sum + (1.0 - giou_pos).sum()
        pos_count = pos_count + float(gt_idx.numel())
        iou_mat, _ = _interval_iou(t_start[b], t_end[b], gt_s, gt_e)
        iou_mat = iou_mat.detach()
        max_iou = iou_mat.max(dim=1).values  # [P]
        neg_mask = (max_iou < tau_neg)
        if pred_idx.numel() > 0:
            neg_mask[pred_idx] = False
        if neg_mask.any() and gamma > 0:
            neg_logits = logits[b, neg_mask]
            neg_loss_sum = neg_loss_sum + F.binary_cross_entropy_with_logits(
                neg_logits,
                torch.zeros_like(neg_logits),
                reduction="sum",
            )
            neg_count = neg_count + float(neg_logits.numel())

    cls_denom = pos_count + gamma * neg_count + 1e-6
    loss_cls = (pos_loss_sum + gamma * neg_loss_sum) / cls_denom
    reg_denom = pos_count.clamp(min=1.0)
    loss_l1 = l1_sum / reg_denom
    loss_giou = giou_sum / reg_denom

    return {
        "loss_cls": loss_cls,
        "loss_l1": loss_l1,
        "loss_giou": loss_giou,
        "pos_count": pos_count,
        "neg_count": neg_count,
    }
