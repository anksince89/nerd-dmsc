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
            nn.Conv2d(in_ch, out_ch,
                      kernel_size=3, stride=2, padding=1),  # downsample
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_conv(x)


# ────────────────────────────────────────────────────────────
# U-Net Upsampling Block
# ────────────────────────────────────────────────────────────
# Doubles the spatial resolution (H/2 → H, W/2 → W)
# Halves the number of channels
# and concatenates the feature map from encoder using skip connection
# Skip connection: It is the memory of UNet — preserves the fine details captured by encoder
# UpConvolution - Transposed convolution kernel = 2, Stride = 2
# Concatenation - Upconvolution output + skip channel
# Conv 3-1-1 means: Kernel=3, Stride=2, Padding=1
# ReLU is performed after convolution
# ────────────────────────────────────────────────────────────
class UpBlock(nn.Module):

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up_conv = nn.ConvTranspose2d(in_ch, in_ch // 2,
                                           kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch // 2 + skip_ch, out_ch,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor,
                skip: torch.Tensor) -> torch.Tensor:
        x = self.up_conv(x)

        # Crop in case of odd sized images
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:],
                              mode='bilinear', align_corners=False)

        # [B, in_ch, H, W] + [B, skip_ch, H, W] → [B, in_ch+skip_ch, H, W]
        x = torch.cat([x, skip], dim=1)

        return self.conv(x)
