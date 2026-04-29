import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import NeRDPixelDecoder


def _normalize_kernel(kernel: torch.Tensor) -> torch.Tensor:
    kernel = kernel - kernel.mean()
    denom = kernel.abs().sum().clamp_min(1e-6)
    return kernel / denom


class BayerFeatureExtractor(nn.Module):
    """
    Deterministic handcrafted Bayer feature stack.
    Input : [B, 1, H, W]
    Output: [B, 30, H, W]
    """

    def __init__(self, pattern: str = "gbrg"):
        super().__init__()
        self.pattern = pattern.lower()

        self.register_buffer(
            "laplacian_kernel",
            torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], dtype=torch.float32).view(1, 1, 3, 3),
        )
        self.register_buffer(
            "hessian_xx_kernel",
            torch.tensor([[0.0, 0.0, 0.0], [1.0, -2.0, 1.0], [0.0, 0.0, 0.0]], dtype=torch.float32).view(1, 1, 3, 3),
        )
        self.register_buffer(
            "hessian_yy_kernel",
            torch.tensor([[0.0, 1.0, 0.0], [0.0, -2.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32).view(1, 1, 3, 3),
        )
        self.register_buffer(
            "hessian_xy_kernel",
            (0.25 * torch.tensor([[1.0, 0.0, -1.0], [0.0, 0.0, 0.0], [-1.0, 0.0, 1.0]], dtype=torch.float32)).view(1, 1, 3, 3),
        )

        self.register_buffer(
            "grad_x_kernel",
            (0.25 * torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], dtype=torch.float32)).view(1, 1, 3, 3),
        )
        self.register_buffer(
            "grad_y_kernel",
            (0.25 * torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], dtype=torch.float32)).view(1, 1, 3, 3),
        )
        self.register_buffer(
            "grad_d1_kernel",
            (0.5 * torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], dtype=torch.float32)).view(1, 1, 3, 3),
        )
        self.register_buffer(
            "grad_d2_kernel",
            (0.5 * torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, -1.0]], dtype=torch.float32)).view(1, 1, 3, 3),
        )

        self.register_buffer(
            "ha_second_h_kernel",
            torch.tensor([[1.0, 0.0, -2.0, 0.0, 1.0]], dtype=torch.float32).view(1, 1, 1, 5),
        )
        self.register_buffer("ha_second_v_kernel", self.ha_second_h_kernel.transpose(-1, -2))
        self.register_buffer(
            "ha_interp_h_kernel",
            torch.tensor([[-0.25, 0.5, 0.5, 0.5, -0.25]], dtype=torch.float32).view(1, 1, 1, 5),
        )
        self.register_buffer("ha_interp_v_kernel", self.ha_interp_h_kernel.transpose(-1, -2))

        sinusoid_x = torch.sin(2.0 * math.pi * torch.arange(5, dtype=torch.float32) / 5.0)
        sinusoid_y = torch.sin(2.0 * math.pi * torch.arange(5, dtype=torch.float32) / 5.0)
        self.register_buffer(
            "sinusoid_x_kernel",
            _normalize_kernel(sinusoid_x.repeat(5, 1)).view(1, 1, 5, 5),
        )
        self.register_buffer(
            "sinusoid_y_kernel",
            _normalize_kernel(sinusoid_y.view(5, 1).repeat(1, 5)).view(1, 1, 5, 5),
        )

        self.register_buffer("gabor_0_kernel", self._make_gabor(size=7, theta=0.0))
        self.register_buffer("gabor_45_kernel", self._make_gabor(size=7, theta=math.pi / 4.0))
        self.register_buffer("gabor_90_kernel", self._make_gabor(size=7, theta=math.pi / 2.0))

        stripe = torch.tensor(
            [
                [1.0, -1.0, 1.0, -1.0, 1.0],
                [1.0, -1.0, 1.0, -1.0, 1.0],
                [1.0, -1.0, 1.0, -1.0, 1.0],
                [1.0, -1.0, 1.0, -1.0, 1.0],
                [1.0, -1.0, 1.0, -1.0, 1.0],
            ],
            dtype=torch.float32,
        )
        checker = torch.tensor(
            [
                [1.0, -1.0, 1.0, -1.0, 1.0],
                [-1.0, 1.0, -1.0, 1.0, -1.0],
                [1.0, -1.0, 1.0, -1.0, 1.0],
                [-1.0, 1.0, -1.0, 1.0, -1.0],
                [1.0, -1.0, 1.0, -1.0, 1.0],
            ],
            dtype=torch.float32,
        )
        dct = self._make_dct_like(size=5, u=2, v=2)
        self.register_buffer("stripe_h_kernel", _normalize_kernel(stripe).view(1, 1, 5, 5))
        self.register_buffer("stripe_v_kernel", _normalize_kernel(stripe.t()).view(1, 1, 5, 5))
        self.register_buffer("checker_kernel", _normalize_kernel(checker).view(1, 1, 5, 5))
        self.register_buffer("dct_kernel", dct)

        self.register_buffer(
            "green_est_kernel",
            torch.tensor([[0.0, 0.25, 0.0], [0.25, 0.0, 0.25], [0.0, 0.25, 0.0]], dtype=torch.float32).view(1, 1, 3, 3),
        )
        self.register_buffer(
            "diff_smooth_kernel",
            (1.0 / 16.0) * torch.tensor([[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]], dtype=torch.float32).view(1, 1, 3, 3),
        )

    def _make_gabor(self,
                    size: int,
                    theta: float,
                    sigma: float = 1.6,
                    wavelength: float = 3.5,
                    gamma: float = 0.6) -> torch.Tensor:
        radius = size // 2
        ys = torch.arange(-radius, radius + 1, dtype=torch.float32)
        xs = torch.arange(-radius, radius + 1, dtype=torch.float32)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")

        x_theta = xx * math.cos(theta) + yy * math.sin(theta)
        y_theta = -xx * math.sin(theta) + yy * math.cos(theta)

        envelope = torch.exp(-(x_theta.pow(2) + (gamma * y_theta).pow(2)) / (2.0 * sigma * sigma))
        carrier = torch.cos(2.0 * math.pi * x_theta / wavelength)
        kernel = envelope * carrier
        return _normalize_kernel(kernel).view(1, 1, size, size)

    def _make_dct_like(self, size: int, u: int, v: int) -> torch.Tensor:
        coords = torch.arange(size, dtype=torch.float32)
        basis_x = torch.cos(math.pi * (coords + 0.5) * u / size)
        basis_y = torch.cos(math.pi * (coords + 0.5) * v / size)
        kernel = torch.outer(basis_y, basis_x)
        return _normalize_kernel(kernel).view(1, 1, size, size)

    def _conv_same(self,
                   x: torch.Tensor,
                   kernel: torch.Tensor,
                   padding_mode: str = "reflect") -> torch.Tensor:
        kh, kw = kernel.shape[-2:]
        pad_h = kh // 2
        pad_w = kw // 2
        x = F.pad(x, (pad_w, pad_w, pad_h, pad_h), mode=padding_mode)
        return F.conv2d(x, kernel.to(dtype=x.dtype, device=x.device))

    def _avg_pool_same(self, x: torch.Tensor, kernel_size: int) -> torch.Tensor:
        pad = kernel_size // 2
        x = F.pad(x, (pad, pad, pad, pad), mode="reflect")
        return F.avg_pool2d(x, kernel_size=kernel_size, stride=1)

    def _normalized_sparse_fill(self,
                                signal: torch.Tensor,
                                mask: torch.Tensor) -> torch.Tensor:
        smooth_signal = self._conv_same(signal, self.diff_smooth_kernel, padding_mode="reflect")
        smooth_mask = self._conv_same(mask, self.diff_smooth_kernel, padding_mode="reflect")
        return smooth_signal / smooth_mask.clamp_min(1e-4)

    def _build_masks(self, bayer: torch.Tensor):
        _, _, h, w = bayer.shape
        r_mask = bayer.new_zeros(1, 1, h, w)
        g_mask = bayer.new_zeros(1, 1, h, w)
        b_mask = bayer.new_zeros(1, 1, h, w)

        if self.pattern == "gbrg":
            g_mask[:, :, 0::2, 0::2] = 1.0
            b_mask[:, :, 0::2, 1::2] = 1.0
            r_mask[:, :, 1::2, 0::2] = 1.0
            g_mask[:, :, 1::2, 1::2] = 1.0
        elif self.pattern == "grbg":
            g_mask[:, :, 0::2, 0::2] = 1.0
            r_mask[:, :, 0::2, 1::2] = 1.0
            b_mask[:, :, 1::2, 0::2] = 1.0
            g_mask[:, :, 1::2, 1::2] = 1.0
        elif self.pattern == "rggb":
            r_mask[:, :, 0::2, 0::2] = 1.0
            g_mask[:, :, 0::2, 1::2] = 1.0
            g_mask[:, :, 1::2, 0::2] = 1.0
            b_mask[:, :, 1::2, 1::2] = 1.0
        elif self.pattern == "bggr":
            b_mask[:, :, 0::2, 0::2] = 1.0
            g_mask[:, :, 0::2, 1::2] = 1.0
            g_mask[:, :, 1::2, 0::2] = 1.0
            r_mask[:, :, 1::2, 1::2] = 1.0
        else:
            raise ValueError(f"Unsupported Bayer pattern: {self.pattern}")

        row_parity = bayer.new_zeros(1, 1, h, w)
        col_parity = bayer.new_zeros(1, 1, h, w)
        row_parity[:, :, 0::2, :] = 1.0
        row_parity[:, :, 1::2, :] = -1.0
        col_parity[:, :, :, 0::2] = 1.0
        col_parity[:, :, :, 1::2] = -1.0

        return (
            r_mask.expand_as(bayer),
            g_mask.expand_as(bayer),
            b_mask.expand_as(bayer),
            row_parity.expand_as(bayer),
            col_parity.expand_as(bayer),
        )

    def forward(self, bayer: torch.Tensor) -> torch.Tensor:
        r_mask, g_mask, b_mask, row_parity, col_parity = self._build_masks(bayer)

        # A simple green estimate gives the handcrafted stack a cross-channel anchor.
        green_missing = self._conv_same(bayer, self.green_est_kernel, padding_mode="reflect")
        green_est = g_mask * bayer + (1.0 - g_mask) * green_missing

        # Sparse color-difference maps approximate R-G and B-G structure from Bayer phases.
        rg_sparse = r_mask * (bayer - green_est)
        bg_sparse = b_mask * (bayer - green_est)
        rg_diff = self._normalized_sparse_fill(rg_sparse, r_mask)
        bg_diff = self._normalized_sparse_fill(bg_sparse, b_mask)

        grad_x = self._conv_same(bayer, self.grad_x_kernel, padding_mode="reflect")
        grad_y = self._conv_same(bayer, self.grad_y_kernel, padding_mode="reflect")
        grad_d1 = self._conv_same(bayer, self.grad_d1_kernel, padding_mode="reflect")
        grad_d2 = self._conv_same(bayer, self.grad_d2_kernel, padding_mode="reflect")
        grad_mag = torch.sqrt(grad_x.pow(2) + grad_y.pow(2) + 1e-8)

        # Directional variance uses 1x5 and 5x1 neighborhoods to expose GBTF-like anisotropy.
        mean_h = self._conv_same(bayer, torch.ones(1, 1, 1, 5, device=bayer.device, dtype=bayer.dtype) / 5.0, padding_mode="reflect")
        mean_v = self._conv_same(bayer, torch.ones(1, 1, 5, 1, device=bayer.device, dtype=bayer.dtype) / 5.0, padding_mode="reflect")
        var_h = self._conv_same((bayer - mean_h).pow(2), torch.ones(1, 1, 1, 5, device=bayer.device, dtype=bayer.dtype) / 5.0, padding_mode="reflect")
        var_v = self._conv_same((bayer - mean_v).pow(2), torch.ones(1, 1, 5, 1, device=bayer.device, dtype=bayer.dtype) / 5.0, padding_mode="reflect")

        ha_second_h = self._conv_same(bayer, self.ha_second_h_kernel, padding_mode="reflect")
        ha_second_v = self._conv_same(bayer, self.ha_second_v_kernel, padding_mode="reflect")
        ha_interp_h = self._conv_same(bayer, self.ha_interp_h_kernel, padding_mode="reflect")
        ha_interp_v = self._conv_same(bayer, self.ha_interp_v_kernel, padding_mode="reflect")
        ha_res_h = bayer - ha_interp_h
        ha_res_v = bayer - ha_interp_v

        sinusoid_phase = torch.sqrt(
            self._conv_same(bayer, self.sinusoid_x_kernel, padding_mode="reflect").pow(2)
            + self._conv_same(bayer, self.sinusoid_y_kernel, padding_mode="reflect").pow(2)
            + 1e-8
        )

        gabor_orientation = (
            self._conv_same(bayer, self.gabor_0_kernel, padding_mode="reflect").abs()
            + self._conv_same(bayer, self.gabor_45_kernel, padding_mode="reflect").abs()
            + self._conv_same(bayer, self.gabor_90_kernel, padding_mode="reflect").abs()
        )

        dct_periodicity = (
            self._conv_same(bayer, self.dct_kernel, padding_mode="reflect").abs()
            + self._conv_same(bayer, self.stripe_h_kernel, padding_mode="reflect").abs()
            + self._conv_same(bayer, self.stripe_v_kernel, padding_mode="reflect").abs()
            + self._conv_same(bayer, self.checker_kernel, padding_mode="reflect").abs()
        )

        local_mean = self._avg_pool_same(bayer, kernel_size=5)
        local_variance = self._avg_pool_same((bayer - local_mean).pow(2), kernel_size=5)
        gradient_energy = self._avg_pool_same(grad_mag.pow(2), kernel_size=5)

        morph_gradient = F.max_pool2d(F.pad(bayer, (1, 1, 1, 1), mode="reflect"), kernel_size=3, stride=1) - (
            -F.max_pool2d(F.pad(-bayer, (1, 1, 1, 1), mode="reflect"), kernel_size=3, stride=1)
        )

        features = []

        # R-mask tells the decoder where red measurements are trustworthy anchor samples.
        features.append(r_mask)

        # G-mask identifies the denser green lattice that stabilizes luminance reconstruction.
        features.append(g_mask)

        # B-mask exposes blue sample positions so periodic CFA aliasing is easier to disambiguate.
        features.append(b_mask)

        # Row parity encodes CFA phase and helps the decoder separate vertical sampling aliases.
        features.append(row_parity)

        # Column parity encodes the complementary CFA phase needed for horizontal phase reasoning.
        features.append(col_parity)

        # Laplacian highlights sharp curvature and zipper-prone transitions around edges.
        features.append(self._conv_same(bayer, self.laplacian_kernel, padding_mode="reflect"))

        # Hessian xx captures horizontal curvature, useful when interpolation changes along columns.
        features.append(self._conv_same(bayer, self.hessian_xx_kernel, padding_mode="reflect"))

        # Hessian yy captures vertical curvature, complementing xx on anisotropic structures.
        features.append(self._conv_same(bayer, self.hessian_yy_kernel, padding_mode="reflect"))

        # Hessian xy reacts to saddle points and oblique edge crossings common in textured regions.
        features.append(self._conv_same(bayer, self.hessian_xy_kernel, padding_mode="reflect"))

        # Morphological gradient emphasizes edge extent without assuming linear image statistics.
        features.append(morph_gradient)

        # Horizontal gradient exposes left-right edge polarity for directional demosaicking.
        features.append(grad_x)

        # Vertical gradient exposes top-bottom edge polarity for directional demosaicking.
        features.append(grad_y)

        # Main-diagonal gradient helps the decoder follow thin slanted structures like roof slats.
        features.append(grad_d1)

        # Anti-diagonal gradient captures the opposite slant direction seen in alias-prone textures.
        features.append(grad_d2)

        # Gradient magnitude summarizes edge strength and helps the decoder gate high-frequency detail.
        features.append(grad_mag)

        # HA second-order horizontal derivative mirrors Hamilton-Adams horizontal edge tests.
        features.append(ha_second_h)

        # HA second-order vertical derivative mirrors Hamilton-Adams vertical edge tests.
        features.append(ha_second_v)

        # Horizontal HA residual measures how much the raw Bayer deviates from horizontal interpolation.
        features.append(ha_res_h)

        # Vertical HA residual measures how much the raw Bayer deviates from vertical interpolation.
        features.append(ha_res_v)

        # GBTF gradient difference compares horizontal and vertical curvature to expose preferred directions.
        features.append(ha_second_h.abs() - ha_second_v.abs())

        # GBTF variance indicator uses directional variance imbalance as a threshold-free edge confidence.
        features.append(var_h - var_v)

        # R-G difference approximates chroma structure around red samples, reducing false color.
        features.append(rg_diff)

        # B-G difference approximates chroma structure around blue samples, reducing false color.
        features.append(bg_diff)

        # Green estimation residual highlights where a simple green model is insufficient.
        features.append(green_est - bayer)

        # Sinusoidal phase projection helps the decoder model repetitive stripe-like periodic content.
        features.append(sinusoid_phase)

        # Gabor orientation energy responds to slanted edges and wood-grain-like directional textures.
        features.append(gabor_orientation)

        # DCT/stripe/checkerboard response targets periodic aliasing, barn-like planks, and CFA moire.
        features.append(dct_periodicity)

        # Local mean supplies a low-frequency luminance anchor under sparse Bayer sampling.
        features.append(local_mean)

        # Local variance separates flat regions from risky textured zones where interpolation must be cautious.
        features.append(local_variance)

        # Gradient energy pools edge activity over a neighborhood, helping periodic-detail reconstruction.
        features.append(gradient_energy)

        features = torch.cat(features, dim=1)
        if features.shape[1] != 30:
            raise RuntimeError(f"Expected 30 feature channels, got {features.shape[1]}.")

        return features


class NerdFeatureModel(nn.Module):
    def __init__(self,
                 in_ch: int = 1,
                 out_ch: int = 3,
                 pattern: str = "gbrg",
                 patch_size: int = 5,
                 hidden: int = 256,
                 omega_0: float = 30.0):
        super().__init__()
        if in_ch != 1:
            raise ValueError("NerdFeatureModel expects a single Bayer input channel.")

        self.feature_extractor = BayerFeatureExtractor(pattern=pattern)
        self.decoder = NeRDPixelDecoder.NeRDPixelDecoder(
            feature_channels=30,
            patch_size=patch_size,
            hidden=hidden,
            out_channels=out_ch,
            omega_0=omega_0,
        )

    def forward(self, bayer: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(bayer)
        rgb = self.decoder(features)
        return rgb