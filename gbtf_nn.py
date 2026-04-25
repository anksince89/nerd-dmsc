import torch
import torch.nn as nn
import torch.nn.functional as F

import NeRDEncoder
import UNet


def _filter2d_same(x: torch.Tensor,
                   kernel: torch.Tensor,
                   padding_mode: str = "replicate") -> torch.Tensor:
    kh, kw = kernel.shape[-2:]
    pad_h = kh // 2
    pad_w = kw // 2
    x = F.pad(x, (pad_w, pad_w, pad_h, pad_h), mode=padding_mode)
    return F.conv2d(x, kernel)


def _box_sum(x: torch.Tensor, radius_h: int, radius_v: int) -> torch.Tensor:
    kernel = x.new_ones(1, 1, 2 * radius_v + 1, 2 * radius_h + 1)
    x = F.pad(x, (radius_h, radius_h, radius_v, radius_v), mode="constant", value=0.0)
    return F.conv2d(x, kernel)


class DirectionalWeightMLP(nn.Module):
    def __init__(self, out_dim: int,
                 feature_dim: int = 64 * 25,
                 coord_dim: int = 2,
                 hidden_dim: int = 64):
        super().__init__()
        self.feature_dim = feature_dim
        self.input_dim = feature_dim + coord_dim

        # Reuse the NeRD skip pattern, but with ReLU instead of sine.
        self.layer1 = nn.Linear(self.input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim + feature_dim, hidden_dim)
        self.layer4 = nn.Linear(hidden_dim, hidden_dim)
        self.layer5 = nn.Linear(hidden_dim + feature_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, out_dim)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feature_skip = x[..., :self.feature_dim]

        h = self.relu(self.layer1(x))
        h = self.relu(self.layer2(h))
        h = self.relu(self.layer3(torch.cat([feature_skip, h], dim=-1)))
        h = self.relu(self.layer4(h))
        h = self.relu(self.layer5(torch.cat([feature_skip, h], dim=-1)))

        weights = torch.sigmoid(self.out(h))
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)
        return weights


class RBInterpolationUNet(nn.Module):
    def __init__(self, in_channels: int = 2, base_ch: int = 32):
        super().__init__()
        self.stem = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(in_channels, base_ch, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),
        )

        self.down1 = UNet.DownBlock(base_ch, base_ch)
        self.down2 = UNet.DownBlock(base_ch, base_ch * 2)
        self.up1 = UNet.UpBlock(base_ch * 2, base_ch, base_ch)
        self.up2 = UNet.UpBlock(base_ch, base_ch, base_ch)

        self.head = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(base_ch, 2, kernel_size=3, stride=1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, raw: torch.Tensor, green: torch.Tensor) -> torch.Tensor:
        x0 = self.stem(torch.cat([raw, green], dim=1))
        d1 = self.down1(x0)
        d2 = self.down2(d1)
        u1 = self.up1(d2, d1)
        u2 = self.up2(u1, x0)
        return self.head(u2)


class gbtf_nn(nn.Module):
    def __init__(self,
                 in_ch: int = 1,
                 out_ch: int = 3,
                 base_ch: int = 64,
                 pattern: str = "gbrg",
                 use_unet: bool = False,
                 weight_hidden: int = 64,
                 weight_row_chunk: int = 16,
                 weight_point_chunk: int = 4096,
                 guided_radius_h: int = 5,
                 guided_radius_v: int = 5,
                 guided_eps: float = 1e-4):
        super().__init__()
        if in_ch != 1 or out_ch != 3:
            raise ValueError("gbtf_nn expects a single Bayer input channel and RGB output.")

        self.pattern = pattern.lower()
        self.use_unet = use_unet
        self.weight_row_chunk = weight_row_chunk
        self.weight_point_chunk = weight_point_chunk
        self.guided_radius_h = guided_radius_h
        self.guided_radius_v = guided_radius_v
        self.guided_eps = guided_eps
        self.guided_var_floor = 1e-5

        self.encoder = NeRDEncoder.NeRDEncoder(in_channels=in_ch, base_ch=base_ch)
        self.weight_head = DirectionalWeightMLP(
            out_dim = 4
            feature_dim=base_ch * 25,
            coord_dim=2,
            hidden_dim=weight_hidden,
        )

        if self.use_unet:
            self.rb_interpolator = RBInterpolationUNet(in_channels=2, base_ch=32)

        self.dif_dir = NeRDEncoder.NeRDEncoder(in_channels=2, base_ch=4)
        # Fixed filters from GBTF / RI.
        self.register_buffer(
            "ha_h",
            torch.tensor([[-0.25, 0.5, 0.5, 0.5, -0.25]], dtype=torch.float32).view(1, 1, 1, 5),
        )
        self.register_buffer("ha_v", self.ha_h.transpose(-1, -2))

        h_kernel = torch.tensor(
            [
                [0.25, 0.5, 0.25],
                [0.5, 1.0, 0.5],
                [0.25, 0.5, 0.25],
            ],
            dtype=torch.float32,
        )
        self.register_buffer("residual_kernel", h_kernel.view(1, 1, 3, 3))

    def _build_masks(self, raw: torch.Tensor):
        _, _, h, w = raw.shape
        mask_r = raw.new_zeros(1, 1, h, w)
        mask_gr = raw.new_zeros(1, 1, h, w)
        mask_gb = raw.new_zeros(1, 1, h, w)
        mask_b = raw.new_zeros(1, 1, h, w)

        if self.pattern == "gbrg":
            mask_gb[:, :, 0::2, 0::2] = 1.0
            mask_b[:, :, 0::2, 1::2] = 1.0
            mask_r[:, :, 1::2, 0::2] = 1.0
            mask_gr[:, :, 1::2, 1::2] = 1.0
        elif self.pattern == "grbg":
            mask_gr[:, :, 0::2, 0::2] = 1.0
            mask_r[:, :, 0::2, 1::2] = 1.0
            mask_b[:, :, 1::2, 0::2] = 1.0
            mask_gb[:, :, 1::2, 1::2] = 1.0
        elif self.pattern == "rggb":
            mask_r[:, :, 0::2, 0::2] = 1.0
            mask_gr[:, :, 0::2, 1::2] = 1.0
            mask_gb[:, :, 1::2, 0::2] = 1.0
            mask_b[:, :, 1::2, 1::2] = 1.0
        elif self.pattern == "bggr":
            mask_b[:, :, 0::2, 0::2] = 1.0
            mask_gb[:, :, 0::2, 1::2] = 1.0
            mask_gr[:, :, 1::2, 0::2] = 1.0
            mask_r[:, :, 1::2, 1::2] = 1.0
        else:
            raise ValueError(f"Unsupported Bayer pattern: {self.pattern}")

        mask_r = mask_r.expand(raw.shape[0], -1, -1, -1)
        mask_gr = mask_gr.expand(raw.shape[0], -1, -1, -1)
        mask_gb = mask_gb.expand(raw.shape[0], -1, -1, -1)
        mask_b = mask_b.expand(raw.shape[0], -1, -1, -1)
        mask_g = mask_gr + mask_gb

        return mask_r, mask_gr, mask_gb, mask_b, mask_g

    def _hamilton_adams_differences(self,
                                    raw: torch.Tensor,
                                    mask_r: torch.Tensor,
                                    mask_gr: torch.Tensor,
                                    mask_gb: torch.Tensor,
                                    mask_b: torch.Tensor,
                                    mask_g: torch.Tensor):
        raw_h = _filter2d_same(raw, self.ha_h.to(raw), padding_mode="replicate")
        raw_v = _filter2d_same(raw, self.ha_v.to(raw), padding_mode="replicate")

        mosaic_r = raw * mask_r
        mosaic_g = raw * mask_g
        mosaic_b = raw * mask_b

        gr_h = raw_h * mask_r
        gb_h = raw_h * mask_b
        r_h = raw_h * mask_gr
        b_h = raw_h * mask_gb

        gr_v = raw_v * mask_r
        gb_v = raw_v * mask_b
        r_v = raw_v * mask_gb
        b_v = raw_v * mask_gr

        dif_h = (gr_h - mosaic_r) + (gb_h - mosaic_b) + (-r_h - b_h + mosaic_g)
        dif_v = (gr_v - mosaic_r) + (gb_v - mosaic_b) + (-r_v - b_v + mosaic_g)

        dif_h = _filter2d_same(dif_h, self.gaussian5.to(raw), padding_mode="replicate")
        dif_v = _filter2d_same(dif_v, self.gaussian5.to(raw), padding_mode="replicate")

        return dif_h, dif_v

    def _coords_for_rows(self,
                         batch: int,
                         height: int,
                         width: int,
                         row_start: int,
                         row_end: int,
                         ref: torch.Tensor) -> torch.Tensor:
        ys = torch.linspace(-1.0, 1.0, height, device=ref.device, dtype=ref.dtype)[row_start:row_end]
        xs = torch.linspace(-1.0, 1.0, width, device=ref.device, dtype=ref.dtype)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        coords = torch.stack([grid_y, grid_x], dim=-1)
        return coords.unsqueeze(0).expand(batch, -1, -1, -1)

    def _predict_green(self,
                       raw: torch.Tensor,
                       encoded: torch.Tensor,
                       dif_n: torch.Tensor,
                       dif_s: torch.Tensor,
                       dif_w: torch.Tensor,
                       dif_e: torch.Tensor,
                       non_green_mask: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = encoded.shape
        pad = 2
        encoded_padded = F.pad(encoded, (pad, pad, pad, pad), mode="reflect")

        green_chunks = []

        for row_start in range(0, height, self.weight_row_chunk):
            row_end = min(row_start + self.weight_row_chunk, height)
            chunk_h = row_end - row_start

            encoded_rows = encoded_padded[:, :, row_start:row_end + 2 * pad, :]
            local = F.unfold(encoded_rows, kernel_size=5)
            local = local.transpose(1, 2).reshape(batch, chunk_h, width, channels * 25)

            coords = self._coords_for_rows(batch, height, width, row_start, row_end, encoded)
            mlp_input = torch.cat([local, coords], dim=-1).reshape(-1, channels * 25 + 2)

            weight_parts = []
            for start in range(0, mlp_input.shape[0], self.weight_point_chunk):
                end = min(start + self.weight_point_chunk, mlp_input.shape[0])
                weight_parts.append(self.weight_head(mlp_input[start:end]))

            weights = torch.cat(weight_parts, dim=0).reshape(batch, chunk_h, width, 4).permute(0, 3, 1, 2)
            w_n = weights[:, 0:1]
            w_s = weights[:, 1:2]
            w_e = weights[:, 2:3]
            w_w = weights[:, 3:4]

            dif_chunk = (
                w_n * dif_n[:, :, row_start:row_end, :]
                + w_s * dif_s[:, :, row_start:row_end, :]
                + w_w * dif_w[:, :, row_start:row_end, :]
                + w_e * dif_e[:, :, row_start:row_end, :]
            ) / (w_n + w_s + w_w + w_e + 1e-8)

            green_chunk = raw[:, :, row_start:row_end, :] + non_green_mask[:, :, row_start:row_end, :] * dif_chunk
            green_chunks.append(green_chunk)

        return torch.cat(green_chunks, dim=2).clamp(0.0, 1.0)

    def _guided_filter(self,
                       guide: torch.Tensor,
                       signal: torch.Tensor,
                       mask: torch.Tensor) -> torch.Tensor:
        n = _box_sum(mask, self.guided_radius_h, self.guided_radius_v)
        n = torch.where(n > 0, n, torch.ones_like(n))

        mean_i = _box_sum(guide * mask, self.guided_radius_h, self.guided_radius_v) / n
        mean_p = _box_sum(signal * mask, self.guided_radius_h, self.guided_radius_v) / n
        mean_ip = _box_sum(guide * signal * mask, self.guided_radius_h, self.guided_radius_v) / n
        mean_ii = _box_sum(guide * guide * mask, self.guided_radius_h, self.guided_radius_v) / n

        cov_ip = mean_ip - mean_i * mean_p
        var_i = mean_ii - mean_i * mean_i
        var_i = torch.clamp(var_i, min=self.guided_var_floor)

        a = cov_ip / (var_i + self.guided_eps)
        b = mean_p - a * mean_i

        n2 = _box_sum(torch.ones_like(guide), self.guided_radius_h, self.guided_radius_v)
        mean_a = _box_sum(a, self.guided_radius_h, self.guided_radius_v) / n2
        mean_b = _box_sum(b, self.guided_radius_h, self.guided_radius_v) / n2

        return mean_a * guide + mean_b

    def _guided_rb(self,
                   raw: torch.Tensor,
                   green: torch.Tensor,
                   mask_r: torch.Tensor,
                   mask_b: torch.Tensor):
        sparse_r = raw * mask_r
        sparse_b = raw * mask_b

        tentative_r = self._guided_filter(green, sparse_r, mask_r).clamp(0.0, 1.0)
        tentative_b = self._guided_filter(green, sparse_b, mask_b).clamp(0.0, 1.0)

        residual_r = mask_r * (sparse_r - tentative_r)
        residual_b = mask_b * (sparse_b - tentative_b)

        residual_r = _filter2d_same(residual_r, self.residual_kernel.to(raw), padding_mode="replicate")
        residual_b = _filter2d_same(residual_b, self.residual_kernel.to(raw), padding_mode="replicate")

        red = (tentative_r + residual_r).clamp(0.0, 1.0)
        blue = (tentative_b + residual_b).clamp(0.0, 1.0)

        # Preserve the observed samples exactly.
        red = red * (1.0 - mask_r) + raw * mask_r
        blue = blue * (1.0 - mask_b) + raw * mask_b
        return red, blue

    def _unet_rb(self,
                 raw: torch.Tensor,
                 green: torch.Tensor,
                 mask_r: torch.Tensor,
                 mask_b: torch.Tensor):
        rb = self.rb_interpolator(raw, green)
        red = rb[:, 0:1]
        blue = rb[:, 1:2]
        red = red * (1.0 - mask_r) + raw * mask_r
        blue = blue * (1.0 - mask_b) + raw * mask_b
        return red.clamp(0.0, 1.0), blue.clamp(0.0, 1.0)

    def forward(self, raw: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(raw)

        mask_r, mask_gr, mask_gb, mask_b, mask_g = self._build_masks(raw)
        non_green_mask = mask_r + mask_b

        dif_h, dif_v = self._hamilton_adams_differences(
            raw,
            mask_r,
            mask_gr,
            mask_gb,
            mask_b,
            mask_g,
        )

        dif_n, dif_s, dif_w, dif_e = self.dif_dir(torch.cat([dif_h, dif_v], dim=1))
        
        green = self._predict_green(
            raw=raw,
            encoded=encoded,
            dif_n=dif_n,
            dif_s=dif_s,
            dif_w=dif_w,
            dif_e=dif_e,
            non_green_mask=non_green_mask,
        )
        green = green * non_green_mask + raw * mask_g

        if self.use_unet:
            red, blue = self._unet_rb(raw, green, mask_r, mask_b)
        else:
            red, blue = self._guided_rb(raw, green, mask_r, mask_b)

        return torch.cat([red, green, blue], dim=1)
