# ────────────────────────────────────────────────────────────
# EDSR ResBlock
# ────────────────────────────────────────────────────────────
# Yeh NeRD ka basic building block hai.
# EDSR matlab "Enhanced Deep Super-Resolution" — isme
# Batch Normalization NAHI hoti, sirf do conv layers hote hain
# aur ek residual (shortcut) connection hota hai.
#
# Structure:
#   input → Conv → ReLU → Conv → (+input) → output
# ────────────────────────────────────────────────────────────
import torch
import torch.nn as nn

class EDSRResBlock(nn.Module):

    def __init__(self, channels: int = 128):
        super().__init__()

        # Pehli conv layer: kernel=3, stride=1, padding=1
        # "3-1-1" matlab kernel size 3, stride 1, padding 1
        # Padding=1 se feature map ka size same rehta hai (H×W)
        self.conv1 = nn.Conv2d(channels, channels,
                               kernel_size=3, stride=1, padding=1)

        # ReLU activation — non-linearity add karta hai
        self.relu  = nn.ReLU(inplace=True)

        # Doosri conv layer: same config
        self.conv2 = nn.Conv2d(channels, channels,
                               kernel_size=3, stride=1, padding=1)

        # NOTE: EDSR mein Batch Norm intentionally NAHI hai.
        # BatchNorm remove karne se high-frequency details
        # better preserve hoti hain — important for demosaicking.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x ko residual ke liye save karo
        residual = x

        # Conv → ReLU → Conv
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)

        # Residual addition: original input ko output mein add karo.
        # Isse gradient vanishing problem solve hota hai aur
        # network sirf "change" seekhta hai, full mapping nahi.
        return out + residual