import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================
# =========================

class SIGReg(nn.Module):
    
    def __init__(self, knots: int = 17, num_dirs: int = 256):
        super().__init__()
        # t in [0, 3]
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3.0 / (knots - 1)
        weights = torch.full((knots,), 2.0 * dt, dtype=torch.float32)
        weights[0] = dt
        weights[-1] = dt
        window = torch.exp(-t.square() / 2.0)

        self.num_dirs = num_dirs
        self.register_buffer("t", t)                     # [K]
        self.register_buffer("phi", window)              # [K]
        self.register_buffer("weights", weights * window)  # [K]

    def forward(self, proj: torch.Tensor) -> torch.Tensor:
        
        device = proj.device
        N, D = proj.shape
        A = torch.randn(D, self.num_dirs, device=device)
        A = A / (A.norm(p=2, dim=0, keepdim=True) + 1e-6)  # [D, M]
        x = proj @ A  # [N, M]
        # self.t: [K]
        x_t = x.unsqueeze(-1) * self.t  # [N, M, K]
        cos_term = x_t.cos().mean(dim=0)  # [M, K]
        sin_term = x_t.sin().mean(dim=0)  # [M, K]
        err = (cos_term - self.phi) ** 2 + (sin_term ** 2)  # [M, K]
        statistic = err @ self.weights  # [M]
        statistic = statistic * float(N)
        return statistic.mean()


# =========================
# 2. Temporal Block & Backbone
# =========================

class TemporalBlock(nn.Module):
    
    def __init__(self, dim: int, num_heads: int, kernel_size: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim),
        )
        self.dw_conv = nn.Conv1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=dim,
        )
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None):
        h = self.norm1(x)
        attn_out, _ = self.attn(
            h, h, h,
            key_padding_mask=key_padding_mask
        )  # [Bv, T, D]
        x = x + attn_out

        h2 = self.norm2(x)
        ff_out = self.ffn(h2)
        x = x + ff_out  # [Bv, T, D]
        x_conv = x.transpose(1, 2)          # [Bv, D, T]
        x_conv = self.dw_conv(x_conv)       # [Bv, D, T]
        x_conv = x_conv.transpose(1, 2)     # [Bv, T, D]

        # x = x + self.alpha * x_conv
        return x


class TemporalBackbone(nn.Module):
    
    def __init__(self, d_model: int, num_layers: int, num_heads: int, kernel_size: int):
        super().__init__()
        self.blocks = nn.ModuleList([
            TemporalBlock(d_model, num_heads, kernel_size)
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None):
        
        for blk in self.blocks:
            x = blk(x, key_padding_mask=key_padding_mask)
        return x  # [Bv, T, D]


# =========================
# =========================

class QFormerAggregator(nn.Module):
    
    def __init__(self, d_model: int, num_heads: int, num_queries: int = 1):
        super().__init__()
        self.num_queries = num_queries
        self.query = nn.Parameter(
            torch.randn(num_queries, d_model) / math.sqrt(d_model)
        )
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        h: torch.Tensor,       # [Bv, T, D]
        mask: torch.Tensor,    # [B, V, T], 0/1
        B: int,
        V: int
    ) -> torch.Tensor:
        Bv, T, D = h.shape
        assert Bv == B * V, "Expected Bv == B * V."
        mask_flat = mask.view(Bv, T)            # [Bv, T]
        key_padding_mask = (mask_flat < 0.5)

        h_norm = self.norm(h)                   # [Bv, T, D]
        Q = self.query.unsqueeze(0).expand(Bv, self.num_queries, D)
        out, _ = self.attn(
            Q,            # query: [Bv, Q, D]
            h_norm,       # key:   [Bv, T, D]
            h_norm,       # value: [Bv, T, D]
            key_padding_mask=key_padding_mask
        )  # out: [Bv, Q, D]
        if self.num_queries == 1:
            z = out[:, 0, :]    # [Bv, D]
        else:
            z = out.mean(dim=1) # [Bv, D]

        return z


# =========================
# =========================

class FramePool(nn.Module):
    
    def __init__(self, d_in: int, hidden_mult: int = 2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, hidden_mult * d_in),
            nn.GELU(),
            nn.Linear(hidden_mult * d_in, d_in),
        )

    def forward(self, x: torch.Tensor, tokens_per_frame: int | None) -> torch.Tensor:
        
        if tokens_per_frame is not None:
            B, TP, D = x.shape
            assert TP % tokens_per_frame == 0, "T * P must be divisible by tokens_per_frame."
            T = TP // tokens_per_frame
            x = x.reshape(B, T, tokens_per_frame, D).mean(dim=2)  # [B, T, D]
        return self.mlp(x)


class Stage1LatentJEPA(nn.Module):
    
    def __init__(
        self,
        d_in: int,
        d_model: int | None = None,
        n_heads: int = 8,
        num_layers: int = 2,
        kernel_size: int = 11,
        num_views: int = 4,
        latent_dim: int | None = None,
        lambda_sig: float = 0.5,
        num_queries: int = 1,
        tokens_per_frame: int | None = None,
    ):
        super().__init__()

        if d_model is None:
            d_model = d_in
        if latent_dim is None:
            latent_dim = d_model

        assert num_views == 4, "This implementation assumes num_views=4: [global, prefix, suffix, strided]."
        self.num_views = num_views
        self.lambda_sig = lambda_sig
        self.d_model = d_model
        self.latent_dim = latent_dim
        self.tokens_per_frame = tokens_per_frame
        self.frame_pool = FramePool(d_in=d_in)
        if d_in == d_model:
            self.input_proj = nn.Identity()
        else:
            self.input_proj = nn.Linear(d_in, d_model)

        # Temporal backbone
        self.backbone = TemporalBackbone(
            d_model=d_model,
            num_layers=num_layers,
            num_heads=n_heads,
            kernel_size=kernel_size,
        )
        self.aggregator = QFormerAggregator(
            d_model=d_model,
            num_heads=n_heads,
            num_queries=num_queries,
        )

        # Projector: D_model -> latent_dim
        self.proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 4 * latent_dim),
            nn.GELU(),
            nn.Linear(4 * latent_dim, latent_dim),
        )
        self.view_embed = nn.Embedding(num_views, latent_dim)
        self.pred_head = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
        )
        self.sigreg = SIGReg()

    def _sample_view_masks(self, B: int, T: int, device) -> torch.Tensor:
        
        V = self.num_views
        masks = torch.zeros(B, V, T, device=device, dtype=torch.float32)
        masks[:, 0, :] = 1.0
        min_pos_1 = int(0.2 * T)
        max_pos_1 = max(min_pos_1 + 1, int(0.3 * T))
        min_pos_2 = int(0.7 * T)
        max_pos_2 = max(min_pos_2 + 1, int(0.8 * T))

        for b in range(B):
            # prefix: [0, t1]
            t1 = torch.randint(min_pos_1, max_pos_1, (1,), device=device).item()
            masks[b, 1, : t1 + 1] = 1.0

            # suffix: [t2, T-1]
            t2 = torch.randint(min_pos_2, max_pos_2, (1,), device=device).item()
            masks[b, 2, t2:] = 1.0

            # strided-global: 0, 2, 4, ...
            stride = 4
            idx = torch.arange(0, T, stride, device=device)
            masks[b, 3, idx] = 1.0

        return masks  # [B, 4, T]

    def forward(self, H_base: torch.Tensor):
        H_base = self.frame_pool(H_base, tokens_per_frame=self.tokens_per_frame)

        B, T, D_in = H_base.shape
        device = H_base.device
        V = self.num_views
        Dm = self.d_model
        Dl = self.latent_dim
        masks = self._sample_view_masks(B, T, device=device)  # [B, V, T]
        H_views = H_base.unsqueeze(1).expand(B, V, T, D_in).contiguous()
        x = self.input_proj(H_views)  # [B, V, T, Dm]
        x = x * masks.unsqueeze(-1)
        x_flat = x.view(B * V, T, Dm)           # [B*V, T, Dm]
        mask_flat = masks.view(B * V, T)       # [B*V, T]
        key_padding_mask = (mask_flat < 0.5)

        h = self.backbone(x_flat, key_padding_mask=key_padding_mask)  # [B*V, T, Dm]
        z_view_flat = self.aggregator(h, masks, B, V)     # [B*V, Dm]
        z_views = z_view_flat.view(B, V, Dm)               # [B, V, Dm]

        # 6) Projector: z_views -> z_latent
        z_latent = self.proj(z_views)                      # [B, V, Dl]
        with torch.no_grad():
            z_mean = z_latent.mean().detach()
            z_std  = z_latent.std(unbiased=False).detach()
        # =====================================================
        z_global = z_latent[:, 0, :].detach()              # [B, Dl], stop-grad
        z_global_rep = z_global.unsqueeze(1).expand(-1, V - 1, -1)  # [B, V-1, Dl]

        z_local = z_latent[:, 1:, :]                       # [B, V-1, Dl]

        view_ids = torch.arange(1, V, device=device)       # [V-1]
        e = self.view_embed(view_ids)                      # [V-1, Dl]
        e = e.unsqueeze(0)                                 # [1, V-1, Dl]

        pred_in = z_local + e                              # [B, V-1, Dl]
        pred = self.pred_head(pred_in)                     # [B, V-1, Dl]

        pred_loss = F.mse_loss(pred, z_global_rep)
        proj_for_sig = z_latent.reshape(-1, Dl)            # [B*V, Dl]
        sig_loss = self.sigreg(proj_for_sig)
        loss = self.lambda_sig * sig_loss + (1.0 - self.lambda_sig) * pred_loss

        return {
            "loss": loss,
            "pred_loss": pred_loss,
            "sig_loss": sig_loss,
            "z_views": z_views,       # [B, V, Dm]
            "z_latent": z_latent,     # [B, V, Dl]
            "masks": masks,           # [B, V, T]
            "z_mean": z_mean,
            "z_std": z_std,
        }


# =========================
# =========================

if __name__ == "__main__":
    torch.manual_seed(0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    B, T, D_in = 1, 48, 1536
    P = 44 * 80
    H_base = torch.randn(B, T * P, D_in, device=device)

    model = Stage1LatentJEPA(
        d_in=D_in,
        d_model=D_in,
        n_heads=8,
        num_layers=4,
        kernel_size=11,
        num_views=4,
        latent_dim=1024,
        lambda_sig=0.5,
        num_queries=1,
        tokens_per_frame=P,
    ).to(device)

    out = model(H_base)
    print("z_views shape:", out["z_views"].shape)     # [B, 4, D_model]
    print("z_latent shape:", out["z_latent"].shape)   # [B, 4, latent_dim]
    print("masks shape:", out["masks"].shape)         # [B, 4, T]
    print("loss:", float(out["loss"]))
    print("pred_loss:", float(out["pred_loss"]))
    print("sig_loss:", float(out["sig_loss"]))
