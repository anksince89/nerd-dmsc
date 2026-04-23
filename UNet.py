import torch
import torch.nn as nn
import torch.nn.functional as F

# ────────────────────────────────────────────────────────────
# U-Net Downsampling Block
# ────────────────────────────────────────────────────────────
# Reduces the spatial resolution by half (H → H/2, W → W/2)
# Conv 3-2-1 means: Kernel=3, Stride=2, Padding=1
# Stride=2 is responsible for downsampling
# ReLU is performed after convolution
# ────────────────────────────────────────────────────────────
class DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.down_conv = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(in_ch, out_ch,
                      kernel_size=3, stride=2, padding=0),  # downsample
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_conv(x)


# ────────────────────────────────────────────────────────────
# U-Net Upsampling Block
# ────────────────────────────────────────────────────────────
# Doubles the spatial resolution (H/2 → H, W/2 → W)
# using bilinear interpolation, then concatenates the feature map
# from encoder using a skip connection.
# Skip connection: It is the memory of UNet — preserves the fine details captured by encoder
# Concatenation - upsampled output + skip channel
# Conv uses replication padding followed by a 3x3 convolution and ReLU,
# matching the author release.
# ────────────────────────────────────────────────────────────
class UpBlock(nn.Module):

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(in_ch + skip_ch, out_ch,
                      kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor,
                skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[2:],
                          mode='bilinear', align_corners=False)

        # [B, in_ch, H, W] + [B, skip_ch, H, W] → [B, in_ch+skip_ch, H, W]
        x = torch.cat([x, skip], dim=1)

        return self.conv(x)
