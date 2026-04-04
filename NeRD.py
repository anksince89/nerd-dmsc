import torch
import torch.nn as nn
import NeRDEncoder
import NeRDPixelDecoder

class NeRD(nn.Module):
    def __init__(self, in_ch: int = 1, out_ch: int = 3):
        super().__init__()
        self.encoder = NeRDEncoder.NeRDEncoder(in_channels=in_ch, base_ch=128)
        self.decoder = NeRDPixelDecoder.NeRDPixelDecoder(hidden=256, out_channels=out_ch, omega_0=30.0)

    def forward(self, bayer: torch.Tensor) -> torch.Tensor:
        """bayer [B,1,H,W] → rgb [B,3,H,W]"""
        xi  = self.encoder(bayer)
        rgb = self.decoder(xi)
        return rgb