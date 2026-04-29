import math
from collections import OrderedDict
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

import NeRDPixelDecoder


def _normalize_kernel(kernel: torch.Tensor) -> torch.Tensor:
    kernel = kernel - kernel.mean()
    return kernel / kernel.abs().sum().clamp_min(1e-6)


def rgb_to_bayer(rgb: torch.Tensor, pattern: str = "gbrg") -> torch.Tensor:
    """
    RGB [B,3,H,W] or [3,H,W] in [0,1] -> Bayer [B,1,H,W] or [1,H,W].
    """
    pattern = pattern.lower()
    squeeze = rgb.dim() == 3
    if squeeze:
        rgb = rgb.unsqueeze(0)

    if rgb.dim() != 4 or rgb.shape[1] != 3:
        raise ValueError("rgb_to_bayer expects [3,H,W] or [B,3,H,W] input.")

    bayer = torch.zeros(
        rgb.shape[0], 1, rgb.shape[2], rgb.shape[3], device=rgb.device, dtype=rgb.dtype
    )

    if pattern == "gbrg":
        bayer[:, 0, 0::2, 0::2] = rgb[:, 1, 0::2, 0::2]
        bayer[:, 0, 0::2, 1::2] = rgb[:, 2, 0::2, 1::2]
        bayer[:, 0, 1::2, 0::2] = rgb[:, 0, 1::2, 0::2]
        bayer[:, 0, 1::2, 1::2] = rgb[:, 1, 1::2, 1::2]
    elif pattern == "grbg":
        bayer[:, 0, 0::2, 0::2] = rgb[:, 1, 0::2, 0::2]
        bayer[:, 0, 0::2, 1::2] = rgb[:, 0, 0::2, 1::2]
        bayer[:, 0, 1::2, 0::2] = rgb[:, 2, 1::2, 0::2]
        bayer[:, 0, 1::2, 1::2] = rgb[:, 1, 1::2, 1::2]
    elif pattern == "rggb":
        bayer[:, 0, 0::2, 0::2] = rgb[:, 0, 0::2, 0::2]
        bayer[:, 0, 0::2, 1::2] = rgb[:, 1, 0::2, 1::2]
        bayer[:, 0, 1::2, 0::2] = rgb[:, 1, 1::2, 0::2]
        bayer[:, 0, 1::2, 1::2] = rgb[:, 2, 1::2, 1::2]
    elif pattern == "bggr":
        bayer[:, 0, 0::2, 0::2] = rgb[:, 2, 0::2, 0::2]
        bayer[:, 0, 0::2, 1::2] = rgb[:, 1, 0::2, 1::2]
        bayer[:, 0, 1::2, 0::2] = rgb[:, 1, 1::2, 0::2]
        bayer[:, 0, 1::2, 1::2] = rgb[:, 0, 1::2, 1::2]
    else:
        raise ValueError(f"Unsupported Bayer pattern: {pattern}")

    return bayer.squeeze(0) if squeeze else bayer


class BayerFeatureExtractor(nn.Module):
    """
    Deterministic Bayer feature bank for RGB demosaicking.
    Output: [B, 30, H, W]
    """

    FEATURE_NAMES = (
        # Group A: structural reconstruction (11)
        "grad_x",
        "grad_y",
        "grad_diag_main",
        "grad_diag_anti",
        "grad_magnitude",
        "laplacian",
        "hessian_lambda_max",
        "hessian_lambda_min",
        "structure_anisotropy",
        "directional_variance",
        "orientation_energy",
        # Group B: cross-channel consistency (10)
        "r_mask",
        "g_mask",
        "b_mask",
        "green_phase_difference",
        "rg_difference",
        "bg_difference",
        "green_interp_residual",
        "directional_green_consistency",
        "chroma_residual_magnitude",
        "color_difference_variance",
        # Group C: aliasing / frequency detection (9)
        "checkerboard_energy",
        "stripe_horizontal",
        "stripe_vertical",
        "alternating_diff_x",
        "alternating_diff_y",
        "phase_shift_energy",
        "sinusoid_proj_x",
        "sinusoid_proj_y",
        "highband_alias_energy",
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
            "grad_x_kernel",
            _normalize_kernel(
                torch.tensor([[-1.0, 0.0, 1.0],
                              [-2.0, 0.0, 2.0],
                              [-1.0, 0.0, 1.0]], dtype=torch.float32)
            ).view(1, 1, 3, 3),
        )
        self.register_buffer(
            "grad_y_kernel",
            _normalize_kernel(
                torch.tensor([[-1.0, -2.0, -1.0],
                              [0.0, 0.0, 0.0],
                              [1.0, 2.0, 1.0]], dtype=torch.float32)
            ).view(1, 1, 3, 3),
        )
        self.register_buffer(
            "grad_diag_main_kernel",
            _normalize_kernel(
                torch.tensor([[-2.0, -1.0, 0.0],
                              [-1.0, 0.0, 1.0],
                              [0.0, 1.0, 2.0]], dtype=torch.float32)
            ).view(1, 1, 3, 3),
        )
        self.register_buffer(
            "grad_diag_anti_kernel",
            _normalize_kernel(
                torch.tensor([[0.0, 1.0, 2.0],
                              [-1.0, 0.0, 1.0],
                              [-2.0, -1.0, 0.0]], dtype=torch.float32)
            ).view(1, 1, 3, 3),
        )

        self.register_buffer(
            "green_cross_kernel",
            torch.tensor([[0.0, 0.25, 0.0],
                          [0.25, 0.0, 0.25],
                          [0.0, 0.25, 0.0]], dtype=torch.float32).view(1, 1, 3, 3),
        )
        self.register_buffer(
            "ha_green_h_kernel",
            torch.tensor([[-0.25, 0.5, 0.5, 0.5, -0.25]], dtype=torch.float32).view(1, 1, 1, 5),
        )
        self.register_buffer(
            "ha_green_v_kernel",
            torch.tensor([[-0.25], [0.5], [0.5], [0.5], [-0.25]], dtype=torch.float32).view(1, 1, 5, 1),
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
            "smooth5_kernel",
            (torch.tensor([[1.0, 2.0, 3.0, 2.0, 1.0],
                           [2.0, 4.0, 6.0, 4.0, 2.0],
                           [3.0, 6.0, 9.0, 6.0, 3.0],
                           [2.0, 4.0, 6.0, 4.0, 2.0],
                           [1.0, 2.0, 3.0, 2.0, 1.0]], dtype=torch.float32) / 81.0).view(1, 1, 5, 5),
        )
        self.register_buffer(
            "box5_kernel",
            (torch.ones(5, 5, dtype=torch.float32) / 25.0).view(1, 1, 5, 5),
        )
        self.register_buffer(
            "box_h_kernel",
            (torch.ones(1, 5, dtype=torch.float32) / 5.0).view(1, 1, 1, 5),
        )
        self.register_buffer(
            "box_v_kernel",
            (torch.ones(5, 1, dtype=torch.float32) / 5.0).view(1, 1, 5, 1),
        )

        self.register_buffer(
            "checkerboard_kernel",
            _normalize_kernel(
                torch.tensor([[1.0, -1.0, 1.0, -1.0, 1.0],
                              [-1.0, 1.0, -1.0, 1.0, -1.0],
                              [1.0, -1.0, 1.0, -1.0, 1.0],
                              [-1.0, 1.0, -1.0, 1.0, -1.0],
                              [1.0, -1.0, 1.0, -1.0, 1.0]], dtype=torch.float32)
            ).view(1, 1, 5, 5),
        )
        self.register_buffer(
            "stripe_horizontal_kernel",
            _normalize_kernel(
                torch.tensor([[1.0, -1.0, 1.0, -1.0, 1.0],
                              [1.0, -1.0, 1.0, -1.0, 1.0],
                              [1.0, -1.0, 1.0, -1.0, 1.0],
                              [1.0, -1.0, 1.0, -1.0, 1.0],
                              [1.0, -1.0, 1.0, -1.0, 1.0]], dtype=torch.float32)
            ).view(1, 1, 5, 5),
        )
        self.register_buffer(
            "stripe_vertical_kernel",
            _normalize_kernel(
                torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0],
                              [-1.0, -1.0, -1.0, -1.0, -1.0],
                              [1.0, 1.0, 1.0, 1.0, 1.0],
                              [-1.0, -1.0, -1.0, -1.0, -1.0],
                              [1.0, 1.0, 1.0, 1.0, 1.0]], dtype=torch.float32)
            ).view(1, 1, 5, 5),
        )
        self.register_buffer(
            "alt_x_kernel",
            _normalize_kernel(
                torch.tensor([[1.0, -1.0, 1.0, -1.0, 1.0]], dtype=torch.float32).repeat(5, 1)
            ).view(1, 1, 5, 5),
        )
        self.register_buffer(
            "alt_y_kernel",
            _normalize_kernel(
                torch.tensor([[1.0], [-1.0], [1.0], [-1.0], [1.0]], dtype=torch.float32).repeat(1, 5)
            ).view(1, 1, 5, 5),
        )

        sin_x = torch.sin(2.0 * math.pi * torch.arange(5, dtype=torch.float32) / 5.0)
        cos_x = torch.cos(2.0 * math.pi * torch.arange(5, dtype=torch.float32) / 5.0)
        sin_y = sin_x.clone()
        cos_y = cos_x.clone()
        self.register_buffer(
            "sinusoid_x_kernel",
            _normalize_kernel(sin_x.repeat(5, 1)).view(1, 1, 5, 5),
        )
        self.register_buffer(
            "sinusoid_y_kernel",
            _normalize_kernel(sin_y.view(5, 1).repeat(1, 5)).view(1, 1, 5, 5),
        )
        self.register_buffer(
            "phase_x_kernel",
            _normalize_kernel(cos_x.repeat(5, 1)).view(1, 1, 5, 5),
        )
        self.register_buffer(
            "phase_y_kernel",
            _normalize_kernel(cos_y.view(5, 1).repeat(1, 5)).view(1, 1, 5, 5),
        )

        self.register_buffer(
            "gabor_45_kernel",
            self._make_gabor(size=5, theta=math.pi / 4.0),
        )
        self.register_buffer(
            "gabor_135_kernel",
            self._make_gabor(size=5, theta=3.0 * math.pi / 4.0),
        )
        self.register_buffer(
            "dct_highband_kernel",
            self._make_dct_like(size=5, u=2, v=2),
        )
        self.register_buffer(
            "hf_band_kernel",
            _normalize_kernel(
                torch.tensor([[1.0, -2.0, 1.0],
                              [-2.0, 4.0, -2.0],
                              [1.0, -2.0, 1.0]], dtype=torch.float32)
            ).view(1, 1, 3, 3),
        )

    def _make_gabor(
        self,
        size: int,
        theta: float,
        sigma: float = 1.1,
        wavelength: float = 3.0,
        gamma: float = 0.65,
    ) -> torch.Tensor:
        radius = size // 2
        coords = torch.arange(-radius, radius + 1, dtype=torch.float32)
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        x_theta = xx * math.cos(theta) + yy * math.sin(theta)
        y_theta = -xx * math.sin(theta) + yy * math.cos(theta)
        envelope = torch.exp(-(x_theta.square() + (gamma * y_theta).square()) / (2.0 * sigma * sigma))
        carrier = torch.cos(2.0 * math.pi * x_theta / wavelength)
        return _normalize_kernel(envelope * carrier).view(1, 1, size, size)

    def _make_dct_like(self, size: int, u: int, v: int) -> torch.Tensor:
        coords = torch.arange(size, dtype=torch.float32)
        basis_x = torch.cos(math.pi * (coords + 0.5) * u / size)
        basis_y = torch.cos(math.pi * (coords + 0.5) * v / size)
        return _normalize_kernel(torch.outer(basis_y, basis_x)).view(1, 1, size, size)

    def _conv_same(self, x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        kh, kw = kernel.shape[-2:]
        pad = (kw // 2, kw // 2, kh // 2, kh // 2)
        x = F.pad(x, pad, mode="reflect")
        return F.conv2d(x, kernel.to(device=x.device, dtype=x.dtype))

    def _normalized_sparse_fill(self, signal: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        num = self._conv_same(signal, self.smooth5_kernel)
        den = self._conv_same(mask, self.smooth5_kernel).clamp_min(self.eps)
        return num / den

    def _build_masks(self, bayer: torch.Tensor) -> Dict[str, torch.Tensor]:
        _, _, h, w = bayer.shape
        shape = (1, 1, h, w)

        r_mask = bayer.new_zeros(shape)
        g_mask = bayer.new_zeros(shape)
        b_mask = bayer.new_zeros(shape)
        gr_mask = bayer.new_zeros(shape)
        gb_mask = bayer.new_zeros(shape)

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
        }

    def extract_feature_dict(self, bayer: torch.Tensor) -> "OrderedDict[str, torch.Tensor]":
        if bayer.dim() != 4 or bayer.shape[1] != 1:
            raise ValueError("BayerFeatureExtractor expects [B,1,H,W] input.")

        raw = bayer
        masks = self._build_masks(bayer)
        r_mask = masks["r"]
        g_mask = masks["g"]
        b_mask = masks["b"]
        gr_mask = masks["gr"]
        gb_mask = masks["gb"]

        gx = self._conv_same(raw, self.grad_x_kernel)
        gy = self._conv_same(raw, self.grad_y_kernel)
        gd_main = self._conv_same(raw, self.grad_diag_main_kernel)
        gd_anti = self._conv_same(raw, self.grad_diag_anti_kernel)
        grad_mag = torch.sqrt(gx.square() + gy.square() + self.eps)
        laplacian = self._conv_same(raw, self.laplacian_kernel)

        hxx = self._conv_same(raw, self.hessian_xx_kernel)
        hyy = self._conv_same(raw, self.hessian_yy_kernel)
        hxy = self._conv_same(raw, self.hessian_xy_kernel)
        h_disc = torch.sqrt((hxx - hyy).square() + 4.0 * hxy.square() + self.eps)
        h_lambda_max = 0.5 * (hxx + hyy + h_disc)
        h_lambda_min = 0.5 * (hxx + hyy - h_disc)

        jxx = self._conv_same(gx.square(), self.box5_kernel)
        jyy = self._conv_same(gy.square(), self.box5_kernel)
        jxy = self._conv_same(gx * gy, self.box5_kernel)
        structure_den = (jxx + jyy).clamp_min(self.eps)
        structure_anisotropy = torch.sqrt((jxx - jyy).square() + 4.0 * jxy.square() + self.eps) / structure_den

        mean_h = self._conv_same(raw, self.box_h_kernel)
        mean_v = self._conv_same(raw, self.box_v_kernel)
        var_h = self._conv_same(raw.square(), self.box_h_kernel) - mean_h.square()
        var_v = self._conv_same(raw.square(), self.box_v_kernel) - mean_v.square()
        directional_variance = var_h - var_v
        orientation_energy = torch.sqrt(self._conv_same(gx.square() + gy.square(), self.box5_kernel) + self.eps)

        green_bilinear = torch.where(g_mask > 0.5, raw, self._conv_same(raw, self.green_cross_kernel))
        ha_green_h = torch.where(g_mask > 0.5, raw, self._conv_same(raw, self.ha_green_h_kernel))
        ha_green_v = torch.where(g_mask > 0.5, raw, self._conv_same(raw, self.ha_green_v_kernel))
        mhc_green = torch.where(g_mask > 0.5, raw, self._conv_same(raw, self.mhc_green_kernel))

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

        rg_diff = dense_r - dense_g
        bg_diff = dense_b - dense_g
        green_phase_difference = dense_gr - dense_gb
        green_interp_residual = mhc_green - green_bilinear
        directional_green_consistency = (ha_green_h - ha_green_v).abs()
        chroma_residual_magnitude = torch.sqrt(rg_diff.square() + bg_diff.square() + self.eps)
        color_diff_variance = (
            self._conv_same(rg_diff.square(), self.box5_kernel) - self._conv_same(rg_diff, self.box5_kernel).square()
            + self._conv_same(bg_diff.square(), self.box5_kernel) - self._conv_same(bg_diff, self.box5_kernel).square()
        )

        checkerboard_energy = self._conv_same(raw, self.checkerboard_kernel).abs()
        stripe_horizontal = self._conv_same(raw, self.stripe_horizontal_kernel).abs()
        stripe_vertical = self._conv_same(raw, self.stripe_vertical_kernel).abs()
        alternating_diff_x = self._conv_same(raw, self.alt_x_kernel).abs()
        alternating_diff_y = self._conv_same(raw, self.alt_y_kernel).abs()
        sinusoid_proj_x = self._conv_same(raw, self.sinusoid_x_kernel)
        sinusoid_proj_y = self._conv_same(raw, self.sinusoid_y_kernel)
        phase_shift_energy = torch.sqrt(
            self._conv_same(raw, self.phase_x_kernel).square()
            + self._conv_same(raw, self.phase_y_kernel).square()
            + sinusoid_proj_x.square()
            + sinusoid_proj_y.square()
            + self.eps
        )
        highband_alias_energy = (
            self._conv_same(raw, self.hf_band_kernel).abs()
            + self._conv_same(raw, self.gabor_45_kernel).abs()
            + self._conv_same(raw, self.gabor_135_kernel).abs()
            + self._conv_same(raw, self.dct_highband_kernel).abs()
        )

        features: "OrderedDict[str, torch.Tensor]" = OrderedDict()

        # Structural: horizontal gradient measures left-right edge transitions for edge-directed reconstruction.
        features["grad_x"] = gx
        # Structural: vertical gradient measures top-bottom edge transitions where green must respect contour flow.
        features["grad_y"] = gy
        # Structural: main-diagonal gradient captures thin slanted lines that bilinear rules often blur.
        features["grad_diag_main"] = gd_main
        # Structural: anti-diagonal gradient captures the opposite slant family that appears in roof/fence textures.
        features["grad_diag_anti"] = gd_anti
        # Structural: gradient magnitude summarizes true edge strength and helps separate texture from flat regions.
        features["grad_magnitude"] = grad_mag
        # Structural: Laplacian measures local curvature where zippering and ringing usually appear first.
        features["laplacian"] = laplacian
        # Structural: major Hessian eigenvalue approximates dominant second-order curvature for sharp edges and ridges.
        features["hessian_lambda_max"] = h_lambda_max
        # Structural: minor Hessian eigenvalue approximates orthogonal curvature and helps distinguish lines from corners.
        features["hessian_lambda_min"] = h_lambda_min
        # Structural: structure-tensor anisotropy measures whether local energy is concentrated along one orientation.
        features["structure_anisotropy"] = structure_anisotropy
        # Structural: directional variance compares horizontal and vertical neighborhood variability for edge direction choice.
        features["directional_variance"] = directional_variance
        # Structural: pooled orientation energy keeps a stable local estimate of oriented detail strength for texture recovery.
        features["orientation_energy"] = orientation_energy

        # Cross-channel: R mask marks locations where red is the trustworthy anchor sample in the Bayer lattice.
        features["r_mask"] = r_mask
        # Cross-channel: G mask marks the dense green sampling support that stabilizes luminance-like reconstruction.
        features["g_mask"] = g_mask
        # Cross-channel: B mask marks blue support so color-difference priors remain phase-aware.
        features["b_mask"] = b_mask
        # Cross-channel: green-phase difference exposes imbalance between the two green sub-lattices in the CFA.
        features["green_phase_difference"] = green_phase_difference
        # Cross-channel: R-G difference encodes chroma consistency around red samples to suppress false color.
        features["rg_difference"] = rg_diff
        # Cross-channel: B-G difference provides the complementary chroma prior around blue samples.
        features["bg_difference"] = bg_diff
        # Cross-channel: green interpolation residual shows where classical green estimators disagree and need learned correction.
        features["green_interp_residual"] = green_interp_residual
        # Cross-channel: directional green consistency indicates whether horizontal and vertical green hypotheses agree.
        features["directional_green_consistency"] = directional_green_consistency
        # Cross-channel: chroma residual magnitude summarizes overall color-difference tension in the local neighborhood.
        features["chroma_residual_magnitude"] = chroma_residual_magnitude
        # Cross-channel: color-difference variance marks unstable chroma regions where naive interpolation leaks color artifacts.
        features["color_difference_variance"] = color_diff_variance

        # Aliasing: checkerboard energy responds to CFA-phase alternation and moire-like sampling artifacts.
        features["checkerboard_energy"] = checkerboard_energy
        # Aliasing: horizontal stripe response detects repeated vertical plank/slat patterns that are easy to alias.
        features["stripe_horizontal"] = stripe_horizontal
        # Aliasing: vertical stripe response detects repeated horizontal banding and zipper-prone line patterns.
        features["stripe_vertical"] = stripe_vertical
        # Aliasing: alternating x-difference is a direct indicator of high-frequency horizontal phase alternation.
        features["alternating_diff_x"] = alternating_diff_x
        # Aliasing: alternating y-difference is the vertical counterpart for row-wise aliasing and zippering.
        features["alternating_diff_y"] = alternating_diff_y
        # Aliasing: phase-shift energy captures quadrature responses that help the SIREN model periodic phase changes.
        features["phase_shift_energy"] = phase_shift_energy
        # Aliasing: sinusoidal x projection measures periodic horizontal frequency content created by real texture or CFA aliasing.
        features["sinusoid_proj_x"] = sinusoid_proj_x
        # Aliasing: sinusoidal y projection measures periodic vertical frequency content and complements the x projection.
        features["sinusoid_proj_y"] = sinusoid_proj_y
        # Aliasing: high-band alias energy pools Gabor, DCT-like, and high-pass responses to localize troublesome frequencies.
        features["highband_alias_energy"] = highband_alias_energy

        if tuple(features.keys()) != self.feature_names:
            raise RuntimeError("Feature ordering mismatch in BayerFeatureExtractor.")

        return features

    def forward(self, bayer: torch.Tensor) -> torch.Tensor:
        feature_dict = self.extract_feature_dict(bayer)
        features = torch.cat(list(feature_dict.values()), dim=1)
        if features.shape[1] != 30:
            raise RuntimeError(f"Expected 30 feature channels, got {features.shape[1]}.")
        return features


class NerdFeatureModel(nn.Module):
    """
    Bayer [B,1,H,W] -> handcrafted Bayer features [B,30,H,W] -> NeRD decoder -> RGB [B,3,H,W]
    """

    def __init__(
        self,
        in_ch: int = 1,
        out_ch: int = 3,
        pattern: str = "gbrg",
        patch_size: int = 5,
        hidden: int = 256,
        omega_0: float = 30.0,
    ):
        super().__init__()
        if in_ch != 1:
            raise ValueError("NerdFeatureModel expects a single Bayer input channel.")

        self.feature_extractor = BayerFeatureExtractor(pattern=pattern)
        self.decoder = NeRDPixelDecoder.NeRDPixelDecoder(
            feature_channels=self.feature_extractor.num_features,
            patch_size=patch_size,
            hidden=hidden,
            out_channels=out_ch,
            omega_0=omega_0,
        )

    def forward(self, bayer: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(bayer)
        rgb = self.decoder(features)
        return rgb
