import torch.nn as nn
import torch
import SirenMLP
import helper
# ─────────────────────────────────────────────────────────
# FULL PIPELINE: Encoder output → RGB image
# ─────────────────────────────────────────────────────────

class NeRDPixelDecoder(nn.Module):
    """
    Feature map [B, C, H, W] ko per-pixel decoder mein bhejkar
    requested output channels [B, out_channels, H, W] banata hai.
    Local 5x5 patches ko flatten karke implicit decoder ko feed karta hai.
    """
    def __init__(self,
                 feature_channels: int = 128,
                 patch_size: int = 5,
                 hidden: int = 256,
                 out_channels: int = 3,
                 omega_0: float = 30.0):
        super().__init__()
        self.out_ch = out_channels
        self.feature_channels = feature_channels
        self.patch_size = patch_size
        self.local_dim = feature_channels * (patch_size ** 2)

        # in_dim = C * (patch_size^2) + 2 coordinates
        self.siren = SirenMLP.SirenMLP(in_dim=self.local_dim + 2, hidden=hidden,
                               out_dim=self.out_ch, omega_0=omega_0)

    def forward(self, xi: torch.Tensor) -> torch.Tensor:
        """
        Input:  xi  [B, C, H, W]  — deterministic or learned feature map
        Output: out [B, out_channels, H, W]
        """
        B, C, H, W = xi.shape
        device = xi.device
        if C != self.feature_channels:
            raise ValueError(
                f"Expected {self.feature_channels} feature channels, got {C}."
            )

        # ── Step 1: Local encoding extract karo ──
        local_enc = helper.extract_local_encoding(xi, patch_size=self.patch_size)
        # [B, H, W, C * patch_size^2]

        # ── Step 2: Spatial coords ──
        coords = helper.make_coords(B, H, W, device)
        # [B, H, W, 2]

        # ── Step 3: Concatenate ──
        mlp_input = torch.cat([local_enc, coords], dim=-1)
        # [B, H, W, C * patch_size^2 + 2]

        # ── Step 4: Flatten pixels — SIREN ek pixel at a time process karta hai ──
        N = B * H * W
        mlp_input = mlp_input.view(N, self.local_dim + 2)
        # [B*H*W, C * patch_size^2 + 2]

        # ── Step 5: SIREN forward ──
        out_flat = self.siren(mlp_input)
        # [B*H*W, out_channels]

        # ── Step 6: Reshape back to image ──
        out = out_flat.view(B, H, W, self.out_ch)
        out = out.permute(0, 3, 1, 2)
        # [B, out_channels, H, W]  ← standard PyTorch image format

        return out
