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
    def __init__(self,
                 hidden: int = 256,
                 out_channels: int = 3,
                 omega_0: float = 30.0,
                 patch_size: int = 5):
        super().__init__()
        self.out_ch = out_channels
        self.patch_size = patch_size
        # in_dim = 3200 (local encoding) + 2 (coords)
        self.siren = SirenMLP.SirenMLP(in_dim=3202, hidden=hidden,
                               out_dim=self.out_ch, omega_0=omega_0)

    def _decode_flat(self,
                     mlp_input: torch.Tensor,
                     pixel_chunk_size: int = None) -> torch.Tensor:
        if pixel_chunk_size is None or mlp_input.shape[0] <= pixel_chunk_size:
            return self.siren(mlp_input)

        rgb_chunks = []
        for start in range(0, mlp_input.shape[0], pixel_chunk_size):
            end = min(start + pixel_chunk_size, mlp_input.shape[0])
            rgb_chunks.append(self.siren(mlp_input[start:end]))
        return torch.cat(rgb_chunks, dim=0)

    def decode_chunk(self,
                     xi: torch.Tensor,
                     row_start: int,
                     row_end: int,
                     col_start: int = 0,
                     col_end: int = None,
                     pixel_chunk_size: int = None) -> torch.Tensor:
        """
        Decode a spatial chunk from the encoder output.
        """
        B, _, H, W = xi.shape
        if col_end is None:
            col_end = W

        local_enc = helper.extract_local_encoding_chunk(
            xi,
            row_start=row_start,
            row_end=row_end,
            col_start=col_start,
            col_end=col_end,
            patch_size=self.patch_size)

        coords = helper.make_coords_chunk(
            B,
            H,
            W,
            xi.device,
            row_start=row_start,
            row_end=row_end,
            col_start=col_start,
            col_end=col_end)

        mlp_input = torch.cat([local_enc, coords], dim=-1)
        mlp_input = mlp_input.reshape(-1, self.siren.in_dim)

        rgb_flat = self._decode_flat(mlp_input, pixel_chunk_size=pixel_chunk_size)

        chunk_h = row_end - row_start
        chunk_w = col_end - col_start
        rgb = rgb_flat.view(B, chunk_h, chunk_w, self.out_ch)
        rgb = rgb.permute(0, 3, 1, 2).contiguous()
        return rgb

    def decode_image(self,
                     xi: torch.Tensor,
                     row_chunk_size: int = None,
                     col_chunk_size: int = None,
                     pixel_chunk_size: int = None) -> torch.Tensor:
        """
        Decode the full image by stitching together smaller chunks.
        """
        _, _, H, W = xi.shape
        row_chunk_size = H if row_chunk_size is None else row_chunk_size
        col_chunk_size = W if col_chunk_size is None else col_chunk_size

        row_chunks = []
        for row_start in range(0, H, row_chunk_size):
            row_end = min(row_start + row_chunk_size, H)
            col_chunks = []
            for col_start in range(0, W, col_chunk_size):
                col_end = min(col_start + col_chunk_size, W)
                col_chunks.append(self.decode_chunk(
                    xi,
                    row_start=row_start,
                    row_end=row_end,
                    col_start=col_start,
                    col_end=col_end,
                    pixel_chunk_size=pixel_chunk_size))

            row_chunks.append(col_chunks[0] if len(col_chunks) == 1
                              else torch.cat(col_chunks, dim=3))

        return row_chunks[0] if len(row_chunks) == 1 else torch.cat(row_chunks, dim=2)

    def forward(self,
                xi: torch.Tensor,
                row_chunk_size: int = None,
                col_chunk_size: int = None,
                pixel_chunk_size: int = None) -> torch.Tensor:
        """
        Input:  xi  [B, 128, H, W]  — encoder output
        Output: rgb [B, 3,   H, W]  — demosaicked image
        """
        return self.decode_image(
            xi,
            row_chunk_size=row_chunk_size,
            col_chunk_size=col_chunk_size,
            pixel_chunk_size=pixel_chunk_size)
