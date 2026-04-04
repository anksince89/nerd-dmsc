# ────────────────────────────────────────────────────────────
# MAIN ENCODER: EDSR + U-Net
# ────────────────────────────────────────────────────────────
#
# Architecture (Fig. 2 se):
#
#   Bayer [B,1,H,W]
#       ↓
#   Initial Conv (1 → 128)
#       ↓
#   8× EDSR ResBlock (128 → 128)       ← "8x" in figure
#       ↓
#   ┌── Down1: 128→128, H/2  ──────────────────────────┐
#   │   Down2: 128→256, H/4  ───────────────────────┐  │
#   │   Down3: 256→512, H/8  ────────────────────┐  │  │  (skip connections)
#   │   Down4: 512→512, H/16 (bottleneck)         │  │  │
#   │       ↓                                      │  │  │
#   │   Up1:  512→256, H/8   ←────────────────────┘  │  │
#   │   Up2:  256→128, H/4   ←───────────────────────┘  │
#   │   Up3:  128→128, H/2   ←──────────────────────────┘
#   └── Up4:  128→128, H     ←── skip from EDSR output
#       ↓
#   Global Encoding ξ: [B, 128, H, W]
#
# ────────────────────────────────────────────────────────────

import torch
import torch.nn as nn
import torch.nn.functional as F
import helper

class NeRDEncoder(nn.Module):

    def __init__(self, in_channels: int = 1,
                 base_ch:       int = 128):
        """
        Args:
            in_channels : Bayer pattern ke channels.
                          1 = single-channel mosaic (default).
                          4 = 4-channel packed (R,Gr,Gb,B) — aapka case!
            base_ch     : Base feature channels (paper mein 128).
        """
        super().__init__()

        # ── Step 1: Initial Feature Extraction ──────────────
        # Bayer pattern (1 channel) ko 128 channel feature map
        # mein convert karo. Spatial size same rehti hai.
        # Conv 3-1-1: kernel=3, stride=1, padding=1
        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channels, base_ch,
                      kernel_size=3, stride=1, padding=1)
        )
        # Shape: [B, 1, H, W] → [B, 128, H, W]

        # ── Step 2: EDSR — 8 Residual Blocks ────────────────
        # 8 baar ResBlock apply karo. Yeh network ko
        # deep patterns seekhne deta hai bina gradient
        # vanishing ke. Figure mein "8x" yahi dikhata hai.
        self.edsr_blocks = nn.Sequential(
            *[helper.EDSRResBlock(channels=base_ch) for _ in range(8)]
        )
        # Shape: [B, 128, H, W] → [B, 128, H, W]  (unchanged)

        # ── Step 3: Post EDSR gray Conv Block 3-1-1 ───
        self.post_edsr_conv = nn.Conv2d(base_ch, base_ch, kernel_size=3, stride=1, padding=1)

        # ── Step 4: U-Net Encoder (4 Downsampling Stages) ───
        # Har stage mein resolution half hoti hai,
        # aur channels badhte hain (zyada abstract features).

        # Stage 1: 128 → 128, H → H/2
        self.down1 = helper.DownBlock(base_ch, base_ch)

        # Stage 2: 128 → 256, H/2 → H/4
        self.down2 = helper.DownBlock(base_ch, base_ch * 2)

        # Stage 3: 256 → 512, H/4 → H/8
        self.down3 = helper.DownBlock(base_ch * 2, base_ch * 4)

        # Stage 4: 512 → 512, H/8 → H/16 (bottleneck — sabse abstract)
        self.down4 = helper.DownBlock(base_ch * 4, base_ch * 4)

        # ── Step 4: U-Net Decoder (4 Upsampling Stages) ─────
        # Har stage mein:
        #   1. Resolution double hoti hai
        #   2. Encoder ka skip feature concatenate hota hai
        #   3. Conv se refine karo

        # Up1: (512 from below + 512 skip from down3) → 256, H/8
        self.up1 = helper.UpBlock(in_ch=base_ch*4, skip_ch=base_ch*4,
                           out_ch=base_ch*2)

        # Up2: (256 from below + 256 skip from down2) → 128, H/4
        self.up2 = helper.UpBlock(in_ch=base_ch*2, skip_ch=base_ch*2,
                           out_ch=base_ch)

        # Up3: (128 from below + 128 skip from down1) → 128, H/2
        self.up3 = helper.UpBlock(in_ch=base_ch,   skip_ch=base_ch,
                           out_ch=base_ch)

        # Up4: (128 from below + 128 skip from EDSR) → 128, H
        # EDSR output directly yahan aata hai as skip connection
        self.up4 = helper.UpBlock(in_ch=base_ch,   skip_ch=base_ch,
                           out_ch=base_ch)

        # ── Step 5: Final 1×1 Conv ───────────────────────────
        # Output channels ko 128 par confirm karo.
        # 1×1 conv = channel mixing, spatial size same.
        self.final_conv = nn.Conv2d(base_ch, base_ch,
                                    kernel_size=1)
        # Final output: [B, 128, H, W] = global encoding ξ

    def forward(self, bayer: torch.Tensor) -> torch.Tensor:
        """
        Args:
            bayer: [B, 1, H, W] — normalized Bayer pattern [0,1]
        Returns:
            xi:    [B, 128, H, W] — global feature encoding ξ
        """

        # ── EDSR Part ────────────────────────────────────────
        # Initial conv: raw Bayer → 128 feature channels
        x0 = self.initial_conv(bayer)
        # [B, 1, H, W] → [B, 128, H, W]

        # 8 ResBlocks: local + global patterns seekho
        x_edsr = self.edsr_blocks(x0)
        x_edsr = self.post_edsr_conv(x_edsr)
        # [B, 128, H, W] → [B, 128, H, W]
        # x_edsr ko baad mein skip connection ke liye save kiya

        # ── U-Net Encoder (Downsampling) ─────────────────────
        d1 = self.down1(x_edsr)   # [B, 128, H/2,  W/2]
        d2 = self.down2(d1)        # [B, 256, H/4,  W/4]
        d3 = self.down3(d2)        # [B, 512, H/8,  W/8]
        d4 = self.down4(d3)        # [B, 512, H/16, W/16]  ← bottleneck

        # ── U-Net Decoder (Upsampling + Skip Connections) ────
        # Har up step mein corresponding down ka output
        # skip connection ke roop mein diya jaata hai.

        u1 = self.up1(d4, d3)      # d4 upsample + d3 skip → [B, 256, H/8, W/8]
        u2 = self.up2(u1, d2)      # u1 upsample + d2 skip → [B, 128, H/4, W/4]
        u3 = self.up3(u2, d1)      # u2 upsample + d1 skip → [B, 128, H/2, W/2]
        u4 = self.up4(u3, x_edsr)  # u3 upsample + EDSR skip→ [B, 128, H,   W]
        #                  ↑
        #   EDSR output = deepest skip connection
        #   Yeh sabse important skip hai — full resolution
        #   fine details directly final decoder tak pohonchti hain

        # [B, 128, H, W] = global encoding ξ

        return u4
