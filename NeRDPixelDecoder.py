import torch.nn as nn
import torch
import SirenMLP
import helper
# ─────────────────────────────────────────────────────────
# FULL PIPELINE: Encoder output → RGB image
# ─────────────────────────────────────────────────────────

class NeRDPixelDecoder(nn.Module):
    """
    Encoder output [B, 128, H, W] leke RGB [B, 3, H, W] deta hai.
    Encoder is class ke bahar train hota hai.
    """
    def __init__(self, hidden: int = 256, out_channels: int = 3, omega_0: float = 30.0):
        super().__init__()
        self.out_ch = out_channels
        # in_dim = 3200 (local encoding) + 2 (coords)
        self.siren = SirenMLP.SirenMLP(in_dim=3202, hidden=hidden,
                               out_dim=self.out_ch, omega_0=omega_0)

    def forward(self, xi: torch.Tensor) -> torch.Tensor:
        """
        Input:  xi  [B, 128, H, W]  — encoder output
        Output: rgb [B, 3,   H, W]  — demosaicked image
        """
        B, C, H, W = xi.shape
        device = xi.device

        # ── Step 1: Local encoding extract karo ──
        local_enc = helper.extract_local_encoding(xi, patch_size=5)
        # [B, H, W, 3200]

        # ── Step 2: Spatial coords ──
        coords = helper.make_coords(B, H, W, device)
        # [B, H, W, 2]

        # ── Step 3: Concatenate ──
        mlp_input = torch.cat([local_enc, coords], dim=-1)
        # [B, H, W, 3202]

        # ── Step 4: Flatten pixels — SIREN ek pixel at a time process karta hai ──
        N = B * H * W
        mlp_input = mlp_input.view(N, 3202)
        # [200000, 3202]  (for B=5, H=W=200)

        # ── Step 5: SIREN forward ──
        rgb_flat = self.siren(mlp_input)
        # [200000, 3]

        # ── Step 6: Reshape back to image ──
        rgb = rgb_flat.view(B, H, W, self.out_ch)
        rgb = rgb.permute(0, 3, 1, 2)
        # [B, 3, H, W]  ← standard PyTorch image format

        return rgb