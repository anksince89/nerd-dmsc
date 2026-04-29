import math
from collections import OrderedDict
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import NeRDPixelDecoder


def _to_pair(value) -> Tuple[int, int]:
    if isinstance(value, tuple):
        return value
    return (value, value)


def rgb_to_bayer(rgb: torch.Tensor, pattern: str = "gbrg") -> torch.Tensor:
    """
    RGB [B,3,H,W] or [3,H,W] in [0,1] -> Bayer [B,1,H,W] / [1,H,W].
    """
    pattern = pattern.lower()
    squeeze = rgb.dim() == 3
    if squeeze:
        rgb = rgb.unsqueeze(0)

    if rgb.dim() != 4 or rgb.shape[1] != 3:
        raise ValueError("rgb_to_bayer expects [3,H,W] or [B,3,H,W] input.")

    bayer = torch.zeros(rgb.shape[0], 1, rgb.shape[2], rgb.shape[3],
                        device=rgb.device, dtype=rgb.dtype)

    if pattern == "gbrg":
        bayer[:, 0, 0::2, 0::2] = rgb[:, 1, 0::2, 0::2]  # Gb
        bayer[:, 0, 0::2, 1::2] = rgb[:, 2, 0::2, 1::2]  # B
        bayer[:, 0, 1::2, 0::2] = rgb[:, 0, 1::2, 0::2]  # R
        bayer[:, 0, 1::2, 1::2] = rgb[:, 1, 1::2, 1::2]  # Gr
    elif pattern == "grbg":
        bayer[:, 0, 0::2, 0::2] = rgb[:, 1, 0::2, 0::2]  # Gr
        bayer[:, 0, 0::2, 1::2] = rgb[:, 0, 0::2, 1::2]  # R
        bayer[:, 0, 1::2, 0::2] = rgb[:, 2, 1::2, 0::2]  # B
        bayer[:, 0, 1::2, 1::2] = rgb[:, 1, 1::2, 1::2]  # Gb
    elif pattern == "rggb":
        bayer[:, 0, 0::2, 0::2] = rgb[:, 0, 0::2, 0::2]  # R
        bayer[:, 0, 0::2, 1::2] = rgb[:, 1, 0::2, 1::2]  # Gr
        bayer[:, 0, 1::2, 0::2] = rgb[:, 1, 1::2, 0::2]  # Gb
        bayer[:, 0, 1::2, 1::2] = rgb[:, 2, 1::2, 1::2]  # B
    elif pattern == "bggr":
        bayer[:, 0, 0::2, 0::2] = rgb[:, 2, 0::2, 0::2]  # B
        bayer[:, 0, 0::2, 1::2] = rgb[:, 1, 0::2, 1::2]  # Gb
        bayer[:, 0, 1::2, 0::2] = rgb[:, 1, 1::2, 0::2]  # Gr
        bayer[:, 0, 1::2, 1::2] = rgb[:, 0, 1::2, 1::2]  # R
    else:
        raise ValueError(f"Unsupported Bayer pattern: {pattern}")

    return bayer.squeeze(0) if squeeze else bayer


class BayerFeatureExtractor(nn.Module):
    """
    Deterministic handcrafted Bayer features specialized for green recovery.
    """

    FEATURE_NAMES = (
        "r_mask",
        "g_mask",
        "b_mask",
        "gr_mask",
        "gb_mask",
        "row_parity",
        "col_parity",
        "laplacian",
        "hessian_xx",
        "hessian_yy",
        "hessian_xy",
        "morphological_gradient",
        "grad_x",
        "grad_y",
        "grad_diag_main",
        "grad_diag_anti",
        "grad_magnitude",
        "structure_coherence",
        "structure_anisotropy",
        "ha_second_h",
        "ha_second_v",
        "ha_green_h",
        "ha_green_v",
        "ha_hv_disagreement",
        "ha_residual_h",
        "ha_residual_v",
        "ha_residual_energy_h",
        "ha_residual_energy_v",
        "directional_grad_diff",
        "directional_second_diff",
        "line_variance_h",
        "line_variance_v",
        "line_variance_diff",
        "directional_confidence",
        "rg_difference",
        "bg_difference",
        "green_phase_difference",
        "mhc_green",
        "mhc_minus_ha_avg",
        "red_green_residual",
        "blue_green_residual",
        "stripe_x",
        "stripe_y",
        "checkerboard",
        "gabor_45",
        "gabor_135",
        "dct_periodic",
        "checkerboard_energy",
        "stripe_energy",
        "local_mean",
        "local_variance",
        "gradient_energy",
    )

    def __init__(self, pattern: str = "gbrg", eps: float = 1e-6):
        super().__init__()
        self.pattern = pattern.lower()
        self.eps = eps
        self.feature_names = self.FEATURE_NAMES
        self.num_features = len(self.feature_names)

        self.register_buffer(
            "laplacian_kernel",
            torch.tensor([[0.0, 1.0, 0.0],
                          [1.0, -4.0, 1.0],
                          [0.0, 1.0, 0.0]], dtype=torch.float32).view(1, 1, 3, 3),
        )
        self.register_buffer(
            "hessian_xx_kernel",
            torch.tensor([[0.0, 0.0, 0.0],
                          [1.0, -2.0, 1.0],
                          [0.0, 0.0, 0.0]], dtype=torch.float32).view(1, 1, 3, 3),
        )
        self.register_buffer(
            "hessian_yy_kernel",
            torch.tensor([[0.0, 1.0, 0.0],
                          [0.0, -2.0, 0.0],
                          [0.0, 1.0, 0.0]], dtype=torch.float32).view(1, 1, 3, 3),
        )
        self.register_buffer(
            "hessian_xy_kernel",
            (torch.tensor([[1.0, 0.0, -1.0],
                           [0.0, 0.0, 0.0],
                           [-1.0, 0.0, 1.0]], dtype=torch.float32) / 4.0).view(1, 1, 3, 3),
        )
        self.register_buffer(
            "sobel_x_kernel",
            (torch.tensor([[-1.0, 0.0, 1.0],
                           [-2.0, 0.0, 2.0],
                           [-1.0, 0.0, 1.0]], dtype=torch.float32) / 8.0).view(1, 1, 3, 3),
        )
        self.register_buffer(
            "sobel_y_kernel",
            (torch.tensor([[-1.0, -2.0, -1.0],
                           [0.0, 0.0, 0.0],
                           [1.0, 2.0, 1.0]], dtype=torch.float32) / 8.0).view(1, 1, 3, 3),
        )
        self.register_buffer(
            "diag_main_kernel",
            (torch.tensor([[-2.0, -1.0, 0.0],
                           [-1.0, 0.0, 1.0],
                           [0.0, 1.0, 2.0]], dtype=torch.float32) / 8.0).view(1, 1, 3, 3),
        )
        self.register_buffer(
            "diag_anti_kernel",
            (torch.tensor([[0.0, 1.0, 2.0],
                           [-1.0, 0.0, 1.0],
                           [-2.0, -1.0, 0.0]], dtype=torch.float32) / 8.0).view(1, 1, 3, 3),
        )
        self.register_buffer(
            "ha_second_h_kernel",
            torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.0, 0.0, 0.0, 0.0, 0.0],
                          [-0.5, 0.0, 1.0, 0.0, -0.5],
                          [0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32).view(1, 1, 5, 5),
        )
        self.register_buffer(
            "ha_second_v_kernel",
            torch.tensor([[0.0, 0.0, -0.5, 0.0, 0.0],
                          [0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.0, 0.0, 1.0, 0.0, 0.0],
                          [0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.0, 0.0, -0.5, 0.0, 0.0]], dtype=torch.float32).view(1, 1, 5, 5),
        )
        self.register_buffer(
            "ha_green_h_kernel",
            torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.0, 0.0, 0.0, 0.0, 0.0],
                          [-0.25, 0.5, 0.5, 0.5, -0.25],
                          [0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32).view(1, 1, 5, 5),
        )
        self.register_buffer(
            "ha_green_v_kernel",
            torch.tensor([[0.0, 0.0, -0.25, 0.0, 0.0],
                          [0.0, 0.0, 0.5, 0.0, 0.0],
                          [0.0, 0.0, 0.5, 0.0, 0.0],
                          [0.0, 0.0, 0.5, 0.0, 0.0],
                          [0.0, 0.0, -0.25, 0.0, 0.0]], dtype=torch.float32).view(1, 1, 5, 5),
        )
        self.register_buffer(
            "smooth5_kernel",
            (torch.tensor([[1.0, 2.0, 3.0, 2.0, 1.0],
                           [2.0, 4.0, 6.0, 4.0, 2.0],
                           [3.0, 6.0, 9.0, 6.0, 3.0],
                           [2.0, 4.0, 6.0, 4.0, 2.0],
                           [1.0, 2.0, 3.0, 2.0, 1.0]], dtype=torch.float32) / 81.0).view(1, 1, 5, 5),
        )
        self.register_buffer(
            "mhc_green_kernel",
            (torch.tensor([[0.0, 0.0, -1.0, 0.0, 0.0],
                           [0.0, 0.0, 2.0, 0.0, 0.0],
                           [-1.0, 2.0, 4.0, 2.0, -1.0],
                           [0.0, 0.0, 2.0, 0.0, 0.0],
                           [0.0, 0.0, -1.0, 0.0, 0.0]], dtype=torch.float32) / 8.0).view(1, 1, 5, 5),
        )
        self.register_buffer(
            "stripe_x_kernel",
            (torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0],
                           [0.0, 0.0, 0.0, 0.0, 0.0],
                           [1.0, -4.0, 6.0, -4.0, 1.0],
                           [0.0, 0.0, 0.0, 0.0, 0.0],
                           [0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32) / 4.0).view(1, 1, 5, 5),
        )
        self.register_buffer(
            "stripe_y_kernel",
            (torch.tensor([[0.0, 0.0, 1.0, 0.0, 0.0],
                           [0.0, 0.0, -4.0, 0.0, 0.0],
                           [0.0, 0.0, 6.0, 0.0, 0.0],
                           [0.0, 0.0, -4.0, 0.0, 0.0],
                           [0.0, 0.0, 1.0, 0.0, 0.0]], dtype=torch.float32) / 4.0).view(1, 1, 5, 5),
        )
        self.register_buffer(
            "checkerboard_kernel",
            (torch.tensor([[1.0, -1.0, 1.0],
                           [-1.0, 1.0, -1.0],
                           [1.0, -1.0, 1.0]], dtype=torch.float32) / 9.0).view(1, 1, 3, 3),
        )
        self.register_buffer(
            "gabor_45_kernel",
            self._make_gabor(theta=math.pi / 4.0, sigma=1.1, lambd=3.0, gamma=0.65).view(1, 1, 5, 5),
        )
        self.register_buffer(
            "gabor_135_kernel",
            self._make_gabor(theta=3.0 * math.pi / 4.0, sigma=1.1, lambd=3.0, gamma=0.65).view(1, 1, 5, 5),
        )
        self.register_buffer(
            "dct_periodic_kernel",
            self._make_dct_like(u=2, v=2, size=5).view(1, 1, 5, 5),
        )

    @staticmethod
    def _make_gabor(theta: float, sigma: float, lambd: float, gamma: float) -> torch.Tensor:
        coords = torch.arange(-2, 3, dtype=torch.float32)
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        x_theta = xx * math.cos(theta) + yy * math.sin(theta)
        y_theta = -xx * math.sin(theta) + yy * math.cos(theta)
        envelope = torch.exp(-(x_theta ** 2 + (gamma ** 2) * y_theta ** 2) / (2.0 * sigma ** 2))
        carrier = torch.cos(2.0 * math.pi * x_theta / lambd)
        kernel = envelope * carrier
        kernel = kernel - kernel.mean()
        return kernel / kernel.abs().sum().clamp_min(1e-6)

    @staticmethod
    def _make_dct_like(u: int, v: int, size: int) -> torch.Tensor:
        coords = torch.arange(size, dtype=torch.float32)
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        kernel = (
            torch.cos(math.pi * (2.0 * xx + 1.0) * u / (2.0 * size))
            * torch.cos(math.pi * (2.0 * yy + 1.0) * v / (2.0 * size))
        )
        kernel = kernel - kernel.mean()
        return kernel / kernel.abs().sum().clamp_min(1e-6)

    def _conv_same(self, x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        kh, kw = kernel.shape[-2:]
        pad = (kw // 2, kw // 2, kh // 2, kh // 2)
        return F.conv2d(F.pad(x, pad, mode="reflect"), kernel.to(dtype=x.dtype))

    def _avg_pool_same(self, x: torch.Tensor, kernel_size) -> torch.Tensor:
        kh, kw = _to_pair(kernel_size)
        pad = (kw // 2, kw // 2, kh // 2, kh // 2)
        return F.avg_pool2d(F.pad(x, pad, mode="reflect"), kernel_size=(kh, kw), stride=1)

    def _normalized_sparse_fill(self, signal: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        num = self._conv_same(signal, self.smooth5_kernel)
        den = self._conv_same(mask, self.smooth5_kernel).clamp_min(self.eps)
        return num / den

    def _build_masks(self, bayer: torch.Tensor) -> Dict[str, torch.Tensor]:
        _, _, H, W = bayer.shape
        shape = (1, 1, H, W)
        dtype = bayer.dtype
        device = bayer.device

        r_mask = torch.zeros(shape, dtype=dtype, device=device)
        g_mask = torch.zeros(shape, dtype=dtype, device=device)
        b_mask = torch.zeros(shape, dtype=dtype, device=device)
        gr_mask = torch.zeros(shape, dtype=dtype, device=device)
        gb_mask = torch.zeros(shape, dtype=dtype, device=device)
        row_parity = torch.zeros(shape, dtype=dtype, device=device)
        col_parity = torch.zeros(shape, dtype=dtype, device=device)

        row_parity[:, :, 1::2, :] = 1.0
        col_parity[:, :, :, 1::2] = 1.0

        if self.pattern == "gbrg":
            gb_mask[:, :, 0::2, 0::2] = 1.0
            b_mask[:, :, 0::2, 1::2] = 1.0
            r_mask[:, :, 1::2, 0::2] = 1.0
            gr_mask[:, :, 1::2, 1::2] = 1.0
        elif self.pattern == "grbg":
            gr_mask[:, :, 0::2, 0::2] = 1.0
            r_mask[:, :, 0::2, 1::2] = 1.0
            b_mask[:, :, 1::2, 0::2] = 1.0
            gb_mask[:, :, 1::2, 1::2] = 1.0
        elif self.pattern == "rggb":
            r_mask[:, :, 0::2, 0::2] = 1.0
            gr_mask[:, :, 0::2, 1::2] = 1.0
            gb_mask[:, :, 1::2, 0::2] = 1.0
            b_mask[:, :, 1::2, 1::2] = 1.0
        elif self.pattern == "bggr":
            b_mask[:, :, 0::2, 0::2] = 1.0
            gb_mask[:, :, 0::2, 1::2] = 1.0
            gr_mask[:, :, 1::2, 0::2] = 1.0
            r_mask[:, :, 1::2, 1::2] = 1.0
        else:
            raise ValueError(f"Unsupported Bayer pattern: {self.pattern}")

        g_mask = gr_mask + gb_mask
        return {
            "r": r_mask.expand_as(bayer),
            "g": g_mask.expand_as(bayer),
            "b": b_mask.expand_as(bayer),
            "gr": gr_mask.expand_as(bayer),
            "gb": gb_mask.expand_as(bayer),
            "row": row_parity.expand_as(bayer),
            "col": col_parity.expand_as(bayer),
        }

    def _mhc_green_estimate(self, mosaic: torch.Tensor, g_mask: torch.Tensor) -> torch.Tensor:
        filtered = self._conv_same(mosaic, self.mhc_green_kernel)
        return torch.where(g_mask > 0.5, mosaic, filtered)

    def extract_feature_dict(self, bayer: torch.Tensor) -> "OrderedDict[str, torch.Tensor]":
        if bayer.dim() != 4 or bayer.shape[1] != 1:
            raise ValueError("BayerFeatureExtractor expects [B,1,H,W] input.")

        masks = self._build_masks(bayer)
        r_mask = masks["r"]
        g_mask = masks["g"]
        b_mask = masks["b"]
        gr_mask = masks["gr"]
        gb_mask = masks["gb"]
        row_parity = masks["row"]
        col_parity = masks["col"]

        raw = bayer
        gx = self._conv_same(raw, self.sobel_x_kernel)
        gy = self._conv_same(raw, self.sobel_y_kernel)
        gdiag_main = self._conv_same(raw, self.diag_main_kernel)
        gdiag_anti = self._conv_same(raw, self.diag_anti_kernel)
        grad_mag = torch.sqrt(gx.square() + gy.square() + self.eps)

        jxx = self._avg_pool_same(gx.square(), 5)
        jyy = self._avg_pool_same(gy.square(), 5)
        jxy = self._avg_pool_same(gx * gy, 5)
        structure_den = (jxx + jyy).clamp_min(self.eps)
        structure_coherence = torch.sqrt((jxx - jyy).square() + 4.0 * jxy.square() + self.eps) / structure_den
        structure_anisotropy = (jxx - jyy) / structure_den

        laplacian = self._conv_same(raw, self.laplacian_kernel)
        hessian_xx = self._conv_same(raw, self.hessian_xx_kernel)
        hessian_yy = self._conv_same(raw, self.hessian_yy_kernel)
        hessian_xy = self._conv_same(raw, self.hessian_xy_kernel)

        # Morphological span highlights zippering and edge polarity flips that hurt green interpolation.
        dilated = F.max_pool2d(F.pad(raw, (1, 1, 1, 1), mode="reflect"), kernel_size=3, stride=1)
        eroded = -F.max_pool2d(F.pad(-raw, (1, 1, 1, 1), mode="reflect"), kernel_size=3, stride=1)
        morph_gradient = dilated - eroded

        ha_second_h = self._conv_same(raw, self.ha_second_h_kernel)
        ha_second_v = self._conv_same(raw, self.ha_second_v_kernel)
        ha_green_h = self._conv_same(raw, self.ha_green_h_kernel)
        ha_green_v = self._conv_same(raw, self.ha_green_v_kernel)
        ha_avg = 0.5 * (ha_green_h + ha_green_v)
        ha_hv_disagreement = (ha_green_h - ha_green_v).abs()
        ha_residual_h = ha_green_h - raw
        ha_residual_v = ha_green_v - raw
        ha_residual_energy_h = self._avg_pool_same(ha_residual_h.abs(), (1, 5))
        ha_residual_energy_v = self._avg_pool_same(ha_residual_v.abs(), (5, 1))

        line_mean_h = self._avg_pool_same(raw, (1, 5))
        line_mean_v = self._avg_pool_same(raw, (5, 1))
        line_var_h = self._avg_pool_same(raw.square(), (1, 5)) - line_mean_h.square()
        line_var_v = self._avg_pool_same(raw.square(), (5, 1)) - line_mean_v.square()
        directional_grad_diff = gx.abs() - gy.abs()
        directional_second_diff = ha_second_h.abs() - ha_second_v.abs()
        line_var_diff = line_var_h - line_var_v
        directional_confidence = (line_var_v - line_var_h) / (line_var_h + line_var_v + self.eps)

        r_sparse = raw * r_mask
        b_sparse = raw * b_mask
        g_sparse = raw * g_mask
        gr_sparse = raw * gr_mask
        gb_sparse = raw * gb_mask

        dense_r = self._normalized_sparse_fill(r_sparse, r_mask)
        dense_b = self._normalized_sparse_fill(b_sparse, b_mask)
        dense_g = self._normalized_sparse_fill(g_sparse, g_mask)
        dense_gr = self._normalized_sparse_fill(gr_sparse, gr_mask)
        dense_gb = self._normalized_sparse_fill(gb_sparse, gb_mask)

        rg_difference = dense_r - dense_g
        bg_difference = dense_b - dense_g
        green_phase_difference = dense_gr - dense_gb
        mhc_green = self._mhc_green_estimate(raw, g_mask)
        mhc_minus_ha_avg = mhc_green - ha_avg
        red_green_residual = dense_r - mhc_green
        blue_green_residual = dense_b - mhc_green

        stripe_x = self._conv_same(raw, self.stripe_x_kernel)
        stripe_y = self._conv_same(raw, self.stripe_y_kernel)
        checkerboard = self._conv_same(raw, self.checkerboard_kernel)
        gabor_45 = self._conv_same(raw, self.gabor_45_kernel)
        gabor_135 = self._conv_same(raw, self.gabor_135_kernel)
        dct_periodic = self._conv_same(raw, self.dct_periodic_kernel)
        checkerboard_energy = self._avg_pool_same(checkerboard.abs(), 5)
        stripe_energy = self._avg_pool_same(stripe_x.abs() + stripe_y.abs(), 5)

        local_mean = self._avg_pool_same(raw, 5)
        local_variance = self._avg_pool_same(raw.square(), 5) - local_mean.square()
        gradient_energy = self._avg_pool_same(grad_mag.square(), 5)

        features: "OrderedDict[str, torch.Tensor]" = OrderedDict()

        # CFA masks tell the decoder which color was physically measured at a site.
        features["r_mask"] = r_mask
        # Green support marks locations where the true target channel is already sampled.
        features["g_mask"] = g_mask
        # Blue support helps the decoder reason about color-opponent interpolation cases.
        features["b_mask"] = b_mask
        # The first green phase distinguishes one green lattice from the other.
        features["gr_mask"] = gr_mask
        # The second green phase separates the Bayer quincunx into parity-aware groups.
        features["gb_mask"] = gb_mask
        # Row parity acts as a cheap positional encoding for CFA phase.
        features["row_parity"] = row_parity
        # Column parity completes the deterministic Bayer positional code.
        features["col_parity"] = col_parity

        # Laplacian emphasizes local curvature where naive green interpolation rings.
        features["laplacian"] = laplacian
        # Horizontal Hessian detects curvature across columns, important at vertical edges.
        features["hessian_xx"] = hessian_xx
        # Vertical Hessian detects curvature across rows, important at horizontal edges.
        features["hessian_yy"] = hessian_yy
        # Mixed Hessian highlights slanted structures that often cause zippering.
        features["hessian_xy"] = hessian_xy
        # Morphological gradient captures edge span and aliasing bursts beyond linear filters.
        features["morphological_gradient"] = morph_gradient

        # Horizontal first derivative is the basic cue for edge-aware green interpolation.
        features["grad_x"] = gx
        # Vertical first derivative complements the directional edge test.
        features["grad_y"] = gy
        # Main-diagonal derivative helps with thin slanted structures and roof lines.
        features["grad_diag_main"] = gdiag_main
        # Anti-diagonal derivative helps where Bayer aliasing follows the opposite slant.
        features["grad_diag_anti"] = gdiag_anti
        # Gradient magnitude marks texture density and edge strength for confidence weighting.
        features["grad_magnitude"] = grad_mag
        # Structure coherence tells the decoder when a dominant orientation is reliable.
        features["structure_coherence"] = structure_coherence
        # Structure anisotropy distinguishes horizontal-vs-vertical dominance with sign.
        features["structure_anisotropy"] = structure_anisotropy

        # Hamilton-Adams horizontal second derivative measures cross-edge oscillation.
        features["ha_second_h"] = ha_second_h
        # Hamilton-Adams vertical second derivative is the vertical counterpart.
        features["ha_second_v"] = ha_second_v
        # Horizontal tentative green is a classical directional estimate to refine.
        features["ha_green_h"] = ha_green_h
        # Vertical tentative green provides the competing classical hypothesis.
        features["ha_green_v"] = ha_green_v
        # HA disagreement exposes ambiguous edge directions and color zipper risk.
        features["ha_hv_disagreement"] = ha_hv_disagreement
        # Horizontal HA residual reveals where raw and horizontal green conflict.
        features["ha_residual_h"] = ha_residual_h
        # Vertical HA residual reveals where raw and vertical green conflict.
        features["ha_residual_v"] = ha_residual_v
        # Horizontal residual energy is a confidence map for line-wise interpolation.
        features["ha_residual_energy_h"] = ha_residual_energy_h
        # Vertical residual energy is the competing confidence map.
        features["ha_residual_energy_v"] = ha_residual_energy_v

        # Gradient difference is a compact directional selector used by many green estimators.
        features["directional_grad_diff"] = directional_grad_diff
        # Second-derivative difference sharpens that selector around oscillatory textures.
        features["directional_second_diff"] = directional_second_diff
        # Horizontal line variance mirrors LMMSE-style local directional uncertainty.
        features["line_variance_h"] = line_var_h
        # Vertical line variance is the orthogonal uncertainty estimate.
        features["line_variance_v"] = line_var_v
        # Variance difference tells the decoder which axis is smoother for green completion.
        features["line_variance_diff"] = line_var_diff
        # Normalized directional confidence keeps that decision bounded and scale-robust.
        features["directional_confidence"] = directional_confidence

        # Dense R-G difference approximates a classical color-difference smoothness prior.
        features["rg_difference"] = rg_difference
        # Dense B-G difference supplies the blue-side version of the same prior.
        features["bg_difference"] = bg_difference
        # Green-phase difference reveals imbalance between the two green sub-lattices.
        features["green_phase_difference"] = green_phase_difference
        # Malvar-He-Cutler green is a strong linear baseline hypothesis on Bayer data.
        features["mhc_green"] = mhc_green
        # MHC-vs-HA disagreement highlights where linear and directional rules diverge.
        features["mhc_minus_ha_avg"] = mhc_minus_ha_avg
        # Red-green residual indicates cross-channel inconsistency around red samples.
        features["red_green_residual"] = red_green_residual
        # Blue-green residual indicates the same inconsistency around blue samples.
        features["blue_green_residual"] = blue_green_residual

        # X-stripe detector responds to narrow periodic vertical structures like fence slats.
        features["stripe_x"] = stripe_x
        # Y-stripe detector responds to narrow periodic horizontal textures.
        features["stripe_y"] = stripe_y
        # Checkerboard response detects Bayer-phase aliasing and moire-like alternation.
        features["checkerboard"] = checkerboard
        # A 45-degree Gabor response helps the decoder model slanted repeating edges.
        features["gabor_45"] = gabor_45
        # A 135-degree Gabor response covers the opposite slanted orientation family.
        features["gabor_135"] = gabor_135
        # The DCT-like periodic response captures compact local frequency content.
        features["dct_periodic"] = dct_periodic
        # Checkerboard energy turns oscillatory alternation into a local confidence map.
        features["checkerboard_energy"] = checkerboard_energy
        # Stripe energy summarizes repeating high-frequency texture strength.
        features["stripe_energy"] = stripe_energy

        # Local mean gives the decoder a low-frequency brightness anchor for green recovery.
        features["local_mean"] = local_mean
        # Local variance marks textured zones where interpolation should trust directionality.
        features["local_variance"] = local_variance
        # Gradient energy summarizes overall local activity for periodic-structure modeling.
        features["gradient_energy"] = gradient_energy

        if tuple(features.keys()) != self.feature_names:
            raise RuntimeError("Feature ordering mismatch in BayerFeatureExtractor.")

        return features

    def forward(self, bayer: torch.Tensor) -> torch.Tensor:
        feature_dict = self.extract_feature_dict(bayer)
        return torch.cat(list(feature_dict.values()), dim=1)


class NerdFeatureModel(nn.Module):
    """
    Green-only variant:
    Bayer [B,1,H,W] -> handcrafted Bayer features -> NeRD decoder -> green [B,1,H,W].
    """

    def __init__(self,
                 pattern: str = "gbrg",
                 patch_size: int = 5,
                 hidden: int = 256,
                 omega_0: float = 30.0):
        super().__init__()
        self.feature_extractor = BayerFeatureExtractor(pattern=pattern)
        self.decoder = NeRDPixelDecoder.NeRDPixelDecoder(
            feature_channels=self.feature_extractor.num_features,
            patch_size=patch_size,
            hidden=hidden,
            out_channels=1,
            omega_0=omega_0,
        )

    def forward(self, bayer: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(bayer)
        green = self.decoder(features)
        return green
