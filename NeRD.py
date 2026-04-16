import torch
import torch.nn as nn
import NeRDEncoder
import NeRDPixelDecoder

class NeRD(nn.Module):
    def __init__(self, in_ch: int = 1, out_ch: int = 3):
        super().__init__()
        self.encoder = NeRDEncoder.NeRDEncoder(in_channels=in_ch, base_ch=128)
        self.decoder = NeRDPixelDecoder.NeRDPixelDecoder(hidden=256, out_channels=out_ch, omega_0=30.0)

    def encode(self, bayer: torch.Tensor) -> torch.Tensor:
        return self.encoder(bayer)

    def decode_chunk(self,
                     xi: torch.Tensor,
                     row_start: int,
                     row_end: int,
                     col_start: int = 0,
                     col_end: int = None,
                     pixel_chunk_size: int = None) -> torch.Tensor:
        return self.decoder.decode_chunk(
            xi,
            row_start=row_start,
            row_end=row_end,
            col_start=col_start,
            col_end=col_end,
            pixel_chunk_size=pixel_chunk_size)

    def decode_image(self,
                     xi: torch.Tensor,
                     row_chunk_size: int = None,
                     col_chunk_size: int = None,
                     pixel_chunk_size: int = None) -> torch.Tensor:
        return self.decoder.decode_image(
            xi,
            row_chunk_size=row_chunk_size,
            col_chunk_size=col_chunk_size,
            pixel_chunk_size=pixel_chunk_size)

    def forward(self,
                bayer: torch.Tensor,
                row_chunk_size: int = None,
                col_chunk_size: int = None,
                pixel_chunk_size: int = None) -> torch.Tensor:
        """bayer [B,1,H,W] → rgb [B,3,H,W]"""
        xi  = self.encode(bayer)
        rgb = self.decode_image(
            xi,
            row_chunk_size=row_chunk_size,
            col_chunk_size=col_chunk_size,
            pixel_chunk_size=pixel_chunk_size)
        return rgb
