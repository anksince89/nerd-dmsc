import math
from typing import Tuple

import torch
import torch.nn.functional as F


SUPPORTED_ALGORITHMS = ("GBTF", "RI", "MLRI", "WMLRI")


def _to_nchw(x: torch.Tensor, channels: int) -> torch.Tensor:
    if x.dim() == 2:
        if channels != 1:
            raise ValueError("2D input is only valid for single-channel tensors.")
        return x.unsqueeze(0).unsqueeze(0)

    if x.dim() == 3:
        if x.shape[0] == channels:
            return x.unsqueeze(0)
        if x.shape[-1] == channels:
            return x.permute(2, 0, 1).unsqueeze(0)
        if channels == 1 and x.shape[0] != 1 and x.shape[-1] != 1:
            return x.unsqueeze(0).unsqueeze(0)
        raise ValueError(f"Could not interpret 3D tensor with shape {tuple(x.shape)} as {channels}-channel image.")

    if x.dim() == 4:
        if x.shape[1] == channels:
            return x
        if x.shape[-1] == channels:
            return x.permute(0, 3, 1, 2)
        raise ValueError(f"Could not interpret 4D tensor with shape {tuple(x.shape)} as {channels}-channel batch.")

    raise ValueError(f"Unsupported tensor rank {x.dim()}.")


def _kernel_2d(kernel, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if isinstance(kernel, torch.Tensor):
        ker = kernel.to(device=device, dtype=dtype)
    else:
        ker = torch.tensor(kernel, device=device, dtype=dtype)
    if ker.dim() == 2:
        ker = ker.view(1, 1, ker.shape[0], ker.shape[1])
    elif ker.dim() == 4:
        pass
    else:
        raise ValueError("Kernel must be 2D or 4D.")
    return ker


def filter2d_replicate(x: torch.Tensor, kernel) -> torch.Tensor:
    ker = _kernel_2d(kernel, x.device, x.dtype)
    kh, kw = ker.shape[-2:]
    pad = (kw // 2, kw // 2, kh // 2, kh // 2)
    x_pad = F.pad(x, pad, mode="replicate")
    return F.conv2d(x_pad, ker)


def box_filter_constant(x: torch.Tensor, boxsz_wh: Tuple[int, int]) -> torch.Tensor:
    width, height = boxsz_wh
    kernel = torch.ones((1, 1, height, width), device=x.device, dtype=x.dtype)
    pad = (width // 2, width // 2, height // 2, height // 2)
    x_pad = F.pad(x, pad, mode="constant", value=0.0)
    return F.conv2d(x_pad, kernel)


def gaussian_kernel_1d(size: int, sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    coords = torch.arange(size, device=device, dtype=dtype) - (size - 1) / 2.0
    kernel = torch.exp(-(coords.square()) / (2.0 * sigma * sigma))
    return kernel / kernel.sum().clamp_min(1e-12)


def get_mosaic_masks(raw_or_shape, pattern: str, device=None, dtype=torch.float32):
    pattern = pattern.lower()
    if isinstance(raw_or_shape, tuple):
        if len(raw_or_shape) == 2:
            b = 1
            h, w = raw_or_shape
        elif len(raw_or_shape) == 4:
            b, _, h, w = raw_or_shape
        else:
            raise ValueError("Unsupported shape tuple for get_mosaic_masks.")
    else:
        raw = _to_nchw(raw_or_shape, 1)
        b, _, h, w = raw.shape
        device = raw.device
        dtype = raw.dtype

    mask_gr = torch.zeros((b, 1, h, w), device=device, dtype=dtype)
    mask_gb = torch.zeros((b, 1, h, w), device=device, dtype=dtype)
    mask_r = torch.zeros((b, 1, h, w), device=device, dtype=dtype)
    mask_b = torch.zeros((b, 1, h, w), device=device, dtype=dtype)

    if pattern == "grbg":
        mask_gr[:, :, 0::2, 0::2] = 1.0
        mask_gb[:, :, 1::2, 1::2] = 1.0
        mask_r[:, :, 0::2, 1::2] = 1.0
        mask_b[:, :, 1::2, 0::2] = 1.0
    elif pattern == "rggb":
        mask_r[:, :, 0::2, 0::2] = 1.0
        mask_gr[:, :, 0::2, 1::2] = 1.0
        mask_gb[:, :, 1::2, 0::2] = 1.0
        mask_b[:, :, 1::2, 1::2] = 1.0
    elif pattern == "gbrg":
        mask_gb[:, :, 0::2, 0::2] = 1.0
        mask_b[:, :, 0::2, 1::2] = 1.0
        mask_r[:, :, 1::2, 0::2] = 1.0
        mask_gr[:, :, 1::2, 1::2] = 1.0
    elif pattern == "bggr":
        mask_b[:, :, 0::2, 0::2] = 1.0
        mask_gb[:, :, 0::2, 1::2] = 1.0
        mask_gr[:, :, 1::2, 0::2] = 1.0
        mask_r[:, :, 1::2, 1::2] = 1.0
    else:
        raise ValueError(f"Unsupported Bayer pattern: {pattern}")

    return mask_gr, mask_gb, mask_r, mask_b


def mosaic_bayer(rgb: torch.Tensor, pattern: str):
    rgb = _to_nchw(rgb, 3).to(torch.float32)
    mask_gr, mask_gb, mask_r, mask_b = get_mosaic_masks(rgb[:, :1], pattern, device=rgb.device, dtype=rgb.dtype)
    mask = torch.zeros_like(rgb)
    mask[:, 0:1] = mask_r
    mask[:, 1:2] = mask_gr + mask_gb
    mask[:, 2:3] = mask_b
    mosaic = rgb * mask
    return mosaic, mask


def raw_to_mosaic(raw: torch.Tensor, pattern: str):
    raw = _to_nchw(raw, 1).to(torch.float32)
    mask_gr, mask_gb, mask_r, mask_b = get_mosaic_masks(raw, pattern)
    mosaic = torch.zeros((raw.shape[0], 3, raw.shape[2], raw.shape[3]), device=raw.device, dtype=raw.dtype)
    mosaic[:, 0:1] = raw * mask_r
    mosaic[:, 1:2] = raw * (mask_gr + mask_gb)
    mosaic[:, 2:3] = raw * mask_b
    mask = torch.zeros_like(mosaic)
    mask[:, 0:1] = mask_r
    mask[:, 1:2] = mask_gr + mask_gb
    mask[:, 2:3] = mask_b
    return mosaic, mask


def directs_smooth4_kernel(algorithm: str, sigma: float, device: torch.device, dtype: torch.dtype):
    algorithm = algorithm.upper()
    if algorithm == "GBTF":
        ke = torch.tensor([[0, 0, 0, 0, 26, 24, 21, 17, 12]], device=device, dtype=dtype) / 100.0
        kw = torch.tensor([[12, 17, 21, 24, 26, 0, 0, 0, 0]], device=device, dtype=dtype) / 100.0
    else:
        h = gaussian_kernel_1d(9, sigma, device, dtype).view(1, 9)
        ke = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1, 1]], device=device, dtype=dtype) * h
        kw = torch.tensor([[1, 1, 1, 1, 1, 0, 0, 0, 0]], device=device, dtype=dtype) * h
        ke = ke / ke.sum(dim=1, keepdim=True).clamp_min(1e-12)
        kw = kw / kw.sum(dim=1, keepdim=True).clamp_min(1e-12)

    ks = ke.t()
    kn = kw.t()
    return kn.view(1, 1, kn.shape[0], kn.shape[1]), ks.view(1, 1, ks.shape[0], ks.shape[1]), ke.view(1, 1, ke.shape[0], ke.shape[1]), kw.view(1, 1, kw.shape[0], kw.shape[1])


def means4weights(algorithm: str, difh2: torch.Tensor, difv2: torch.Tensor):
    algorithm = algorithm.upper()
    device = difh2.device
    dtype = difh2.dtype

    if algorithm == "GBTF":
        g = gaussian_kernel_1d(5, 2.0, device, dtype)
        k = torch.outer(g, g)
        kw = torch.tensor([[1.0, 0.0, 0.0]], device=device, dtype=dtype)
        ke = torch.tensor([[0.0, 0.0, 1.0]], device=device, dtype=dtype)
    elif algorithm == "RI":
        k = torch.ones((5, 5), device=device, dtype=dtype)
        kw = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0]], device=device, dtype=dtype)
        ke = torch.tensor([[0.0, 0.0, 0.0, 0.0, 1.0]], device=device, dtype=dtype)
    elif algorithm in ("MLRI", "WMLRI"):
        if algorithm == "MLRI":
            k = torch.ones((3, 3), device=device, dtype=dtype)
        else:
            g = gaussian_kernel_1d(5, 2.0, device, dtype)
            k = torch.outer(g, g)
        kw = torch.tensor([[1.0, 0.0, 0.0]], device=device, dtype=dtype)
        ke = torch.tensor([[0.0, 0.0, 1.0]], device=device, dtype=dtype)
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    wh = filter2d_replicate(difh2, k)
    wv = filter2d_replicate(difv2, k)
    ks = ke.t()
    kn = kw.t()

    ww = filter2d_replicate(wh, kw)
    we = filter2d_replicate(wh, ke)
    wn = filter2d_replicate(wv, kn)
    ws = filter2d_replicate(wv, ks)

    eps = 1e-32
    ww = 1.0 / (ww.square() + eps)
    we = 1.0 / (we.square() + eps)
    wn = 1.0 / (wn.square() + eps)
    ws = 1.0 / (ws.square() + eps)
    return wn, ws, we, ww


def ha_residual(rawq: torch.Tensor, mask: torch.Tensor, mask_gr: torch.Tensor, mask_gb: torch.Tensor, mosaic: torch.Tensor):
    f = torch.tensor([[-0.25, 0.5, 0.5, 0.5, -0.25]], device=rawq.device, dtype=rawq.dtype)
    rawh = filter2d_replicate(rawq, f)
    rawv = filter2d_replicate(rawq, f.t())

    mask_r = mask[:, 0:1]
    mask_b = mask[:, 2:3]

    grh = rawh * mask_r
    gbh = rawh * mask_b
    rh = rawh * mask_gr
    bh = rawh * mask_gb
    grv = rawv * mask_r
    gbv = rawv * mask_b
    rv = rawv * mask_gb
    bv = rawv * mask_gr

    difh = (grh - mosaic[:, 0:1]) + (gbh - mosaic[:, 2:3]) + (-rh - bh + mosaic[:, 1:2])
    difv = (grv - mosaic[:, 0:1]) + (gbv - mosaic[:, 2:3]) + (-rv - bv + mosaic[:, 1:2])

    kh = torch.tensor([[1.0, 0.0, -1.0]], device=rawq.device, dtype=rawq.dtype)
    kv = kh.t()
    avk = torch.tensor([[1.0, 1.0, 1.0]], device=rawq.device, dtype=rawq.dtype)
    difh2 = filter2d_replicate(filter2d_replicate(difh, kh).abs(), avk.t())
    difv2 = filter2d_replicate(filter2d_replicate(difv, kv).abs(), avk)
    return difh, difv, difh2, difv2


def guidedfilter3gf(I: torch.Tensor, p: torch.Tensor, M: torch.Tensor, h: int, v: int, eps: float, algorithm: str, Fker) -> torch.Tensor:
    algorithm = algorithm.upper()
    th = 1e-5 * 255.0 * 255.0
    boxsz = (2 * h + 1, 2 * v + 1)

    n = box_filter_constant(M, boxsz)
    n = torch.where(n == 0, torch.ones_like(n), n)

    mean_I = box_filter_constant(I * M, boxsz) / n
    mean_p = box_filter_constant(p * M, boxsz) / n

    if algorithm in ("MLRI", "WMLRI"):
        difIF = filter2d_replicate(I * M, Fker)
        difpF = filter2d_replicate(p, Fker)
        mean_Ip = box_filter_constant(difIF * difpF * M, boxsz) / n
        mean_II = box_filter_constant(difIF * difIF * M, boxsz) / n
        mean_II = torch.clamp(mean_II, min=th)
        a = mean_Ip / (mean_II + eps)
    else:
        mean_Ip = box_filter_constant(I * p * M, boxsz) / n
        mean_II = box_filter_constant(I * I * M, boxsz) / n
        cov_Ip = mean_Ip - mean_I * mean_p
        var_I = mean_II - mean_I * mean_I
        var_I = torch.clamp(var_I, min=th)
        a = cov_Ip / (var_I + eps)

    b = mean_p - a * mean_I

    if algorithm == "WMLRI":
        dif = (
            box_filter_constant(I * I * M, boxsz) * a * a
            + b * b * n
            + box_filter_constant(p * p * M, boxsz)
            + 2.0 * a * b * box_filter_constant(I * M, boxsz)
            - 2.0 * b * box_filter_constant(p * M, boxsz)
            - 2.0 * a * box_filter_constant(p * I * M, boxsz)
        ) / n
        dif = torch.clamp(dif, min=0.001)
        dif = 1.0 / dif
        wdif = torch.clamp(box_filter_constant(dif, boxsz), min=0.001)
        mean_a = box_filter_constant(a * dif, boxsz) / wdif
        mean_b = box_filter_constant(b * dif, boxsz) / wdif
    else:
        n2 = box_filter_constant(torch.ones_like(I), boxsz)
        mean_a = box_filter_constant(a, boxsz) / n2
        mean_b = box_filter_constant(b, boxsz) / n2

    return mean_a * I + mean_b


def guidefilter_residual(rawq: torch.Tensor, mask: torch.Tensor, mask_gr: torch.Tensor, mask_gb: torch.Tensor, mosaic: torch.Tensor, algorithm: str):
    mask_r = mask[:, 0:1]
    mask_b = mask[:, 2:3]

    kh = torch.tensor([[0.5, 0.0, 0.5]], device=rawq.device, dtype=rawq.dtype)
    kv = kh.t()
    rawh = filter2d_replicate(rawq, kh)
    rawv = filter2d_replicate(rawq, kv)

    guidegh = mosaic[:, 1:2] + rawh * mask[:, 0:1] + rawh * mask[:, 2:3]
    guiderh = mosaic[:, 0:1] + rawh * mask_gr
    guidebh = mosaic[:, 2:3] + rawh * mask_gb

    guidegv = mosaic[:, 1:2] + rawv * mask[:, 0:1] + rawv * mask[:, 2:3]
    guiderv = mosaic[:, 0:1] + rawv * mask_gb
    guidebv = mosaic[:, 2:3] + rawv * mask_gr

    algorithm = algorithm.upper()
    if algorithm == "RI":
        h, v = 5, 0
    elif algorithm in ("MLRI", "WMLRI"):
        h, v = 3, 3
    else:
        raise ValueError(f"guidefilter_residual only supports RI/MLRI/WMLRI, got {algorithm}.")

    eps = 0.0
    f = torch.tensor([[-1.0, 0.0, 2.0, 0.0, -1.0]], device=rawq.device, dtype=rawq.dtype)
    ft = f.t()

    tentativeRh = guidedfilter3gf(guidegh, mosaic[:, 0:1], mask_r, h, v, eps, algorithm, f)
    tentativeGrh = guidedfilter3gf(guiderh, mosaic[:, 1:2] * mask_gr, mask_gr, h, v, eps, algorithm, f)
    tentativeGbh = guidedfilter3gf(guidebh, mosaic[:, 1:2] * mask_gb, mask_gb, h, v, eps, algorithm, f)
    tentativeBh = guidedfilter3gf(guidegh, mosaic[:, 2:3], mask_b, h, v, eps, algorithm, f)

    tentativeRv = guidedfilter3gf(guidegv, mosaic[:, 0:1], mask_r, v, h, eps, algorithm, ft)
    tentativeGrv = guidedfilter3gf(guiderv, mosaic[:, 1:2] * mask_gb, mask_gb, v, h, eps, algorithm, ft)
    tentativeGbv = guidedfilter3gf(guidebv, mosaic[:, 1:2] * mask_gr, mask_gr, v, h, eps, algorithm, ft)
    tentativeBv = guidedfilter3gf(guidegv, mosaic[:, 2:3], mask_b, v, h, eps, algorithm, ft)

    tentativeGrh = tentativeGrh.clamp(0.0, 255.0)
    tentativeGrv = tentativeGrv.clamp(0.0, 255.0)
    tentativeGbh = tentativeGbh.clamp(0.0, 255.0)
    tentativeGbv = tentativeGbv.clamp(0.0, 255.0)
    tentativeRh = tentativeRh.clamp(0.0, 255.0)
    tentativeRv = tentativeRv.clamp(0.0, 255.0)
    tentativeBh = tentativeBh.clamp(0.0, 255.0)
    tentativeBv = tentativeBv.clamp(0.0, 255.0)

    residualGrh = (mosaic[:, 1:2] - tentativeGrh) * mask_gr
    residualGbh = (mosaic[:, 1:2] - tentativeGbh) * mask_gb
    residualRh = (mosaic[:, 0:1] - tentativeRh) * mask_r
    residualBh = (mosaic[:, 2:3] - tentativeBh) * mask_b
    residualGrv = (mosaic[:, 1:2] - tentativeGrv) * mask_gb
    residualGbv = (mosaic[:, 1:2] - tentativeGbv) * mask_gr
    residualRv = (mosaic[:, 0:1] - tentativeRv) * mask_r
    residualBv = (mosaic[:, 2:3] - tentativeBv) * mask_b

    residualGrh = filter2d_replicate(residualGrh, kh)
    residualGbh = filter2d_replicate(residualGbh, kh)
    residualRh = filter2d_replicate(residualRh, kh)
    residualBh = filter2d_replicate(residualBh, kh)
    residualGrv = filter2d_replicate(residualGrv, kv)
    residualGbv = filter2d_replicate(residualGbv, kv)
    residualRv = filter2d_replicate(residualRv, kv)
    residualBv = filter2d_replicate(residualBv, kv)

    Grh = (tentativeGrh + residualGrh) * mask_r
    Gbh = (tentativeGbh + residualGbh) * mask_b
    Rh = (tentativeRh + residualRh) * mask_gr
    Bh = (tentativeBh + residualBh) * mask_gb
    Grv = (tentativeGrv + residualGrv) * mask_r
    Gbv = (tentativeGbv + residualGbv) * mask_b
    Rv = (tentativeRv + residualRv) * mask_gb
    Bv = (tentativeBv + residualBv) * mask_gr

    Grh = Grh.clamp(0.0, 255.0)
    Grv = Grv.clamp(0.0, 255.0)
    Gbh = Gbh.clamp(0.0, 255.0)
    Gbv = Gbv.clamp(0.0, 255.0)
    Rh = Rh.clamp(0.0, 255.0)
    Rv = Rv.clamp(0.0, 255.0)
    Bh = Bh.clamp(0.0, 255.0)
    Bv = Bv.clamp(0.0, 255.0)

    difh = mosaic[:, 1:2] + Grh + Gbh - mosaic[:, 0:1] - mosaic[:, 2:3] - Rh - Bh
    difv = mosaic[:, 1:2] + Grv + Gbv - mosaic[:, 0:1] - mosaic[:, 2:3] - Rv - Bv

    kh_grad = torch.tensor([[1.0, 0.0, -1.0]], device=rawq.device, dtype=rawq.dtype)
    kv_grad = kh_grad.t()
    difh2 = filter2d_replicate(difh, kh_grad).abs()
    difv2 = filter2d_replicate(difv, kv_grad).abs()
    return difh, difv, difh2, difv2


def green_interpolation(mosaic: torch.Tensor, mask: torch.Tensor, pattern: str, sigma: float, algorithm: str):
    rawq = mosaic.sum(dim=1, keepdim=True)
    mask_gr, mask_gb, _, _ = get_mosaic_masks(rawq, pattern)

    if algorithm.upper() == "GBTF":
        difh, difv, difh2, difv2 = ha_residual(rawq, mask, mask_gr, mask_gb, mosaic)
    else:
        difh, difv, difh2, difv2 = guidefilter_residual(rawq, mask, mask_gr, mask_gb, mosaic, algorithm)

    kn, ks, ke, kw = directs_smooth4_kernel(algorithm, sigma, mosaic.device, mosaic.dtype)
    wn, ws, we, ww = means4weights(algorithm, difh2, difv2)

    difn = filter2d_replicate(difv, kn)
    difs = filter2d_replicate(difv, ks)
    dife = filter2d_replicate(difh, ke)
    difw = filter2d_replicate(difh, kw)

    wt = ww + we + wn + ws
    dif = (wn * difn + ws * difs + ww * difw + we * dife) / wt.clamp_min(1e-12)
    green = dif + rawq
    green = green * (1.0 - mask[:, 1:2]) + rawq * mask[:, 1:2]
    green = green.clamp(0.0, 255.0)
    return green, dif


def red_interpolation(green: torch.Tensor, mosaic: torch.Tensor, mask: torch.Tensor, pattern: str, h: int, v: int, eps: float, dif: torch.Tensor, algorithm: str):
    algorithm = algorithm.upper()
    if algorithm == "GBTF":
        prb = torch.tensor(
            [[0, 0, -1, 0, -1, 0, 0],
             [0, 0, 0, 0, 0, 0, 0],
             [-1, 0, 10, 0, 10, 0, -1],
             [0, 0, 0, 0, 0, 0, 0],
             [-1, 0, 10, 0, 10, 0, -1],
             [0, 0, 0, 0, 0, 0, 0],
             [0, 0, -1, 0, -1, 0, 0]],
            device=mosaic.device,
            dtype=mosaic.dtype,
        ) / 32.0
        aknl = torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]], device=mosaic.device, dtype=mosaic.dtype) / 4.0
        red = mosaic[:, 0:1] + mask[:, 2:3] * (green - filter2d_replicate(dif, prb))
        tempimg = mosaic[:, 1:2] - mask[:, 1:2] * filter2d_replicate(green, aknl) + mask[:, 1:2] * filter2d_replicate(red, aknl)
        red = red + tempimg
    else:
        f = torch.tensor(
            [[0, 0, -1, 0, 0],
             [0, 0, 0, 0, 0],
             [-1, 0, 4, 0, -1],
             [0, 0, 0, 0, 0],
             [0, 0, -1, 0, 0]],
            device=mosaic.device,
            dtype=mosaic.dtype,
        )
        hker = torch.tensor([[0.25, 0.5, 0.25], [0.5, 1.0, 0.5], [0.25, 0.5, 0.25]], device=mosaic.device, dtype=mosaic.dtype)
        tentativeR = guidedfilter3gf(green, mosaic[:, 0:1], mask[:, 0:1], h, v, eps, algorithm, f).clamp(0.0, 255.0)
        residualR = mask[:, 0:1] * (mosaic[:, 0:1] - tentativeR)
        residualR = filter2d_replicate(residualR, hker)
        red = residualR + tentativeR

    return red.clamp(0.0, 255.0)


def blue_interpolation(green: torch.Tensor, mosaic: torch.Tensor, mask: torch.Tensor, pattern: str, h: int, v: int, eps: float, dif: torch.Tensor, algorithm: str):
    algorithm = algorithm.upper()
    if algorithm == "GBTF":
        prb = torch.tensor(
            [[0, 0, -1, 0, -1, 0, 0],
             [0, 0, 0, 0, 0, 0, 0],
             [-1, 0, 10, 0, 10, 0, -1],
             [0, 0, 0, 0, 0, 0, 0],
             [-1, 0, 10, 0, 10, 0, -1],
             [0, 0, 0, 0, 0, 0, 0],
             [0, 0, -1, 0, -1, 0, 0]],
            device=mosaic.device,
            dtype=mosaic.dtype,
        ) / 32.0
        aknl = torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]], device=mosaic.device, dtype=mosaic.dtype) / 4.0
        blue = mosaic[:, 2:3] + mask[:, 0:1] * (green - filter2d_replicate(dif, prb))
        tempimg = mosaic[:, 1:2] - mask[:, 1:2] * filter2d_replicate(green, aknl) + mask[:, 1:2] * filter2d_replicate(blue, aknl)
        blue = blue + tempimg
    else:
        f = torch.tensor(
            [[0, 0, -1, 0, 0],
             [0, 0, 0, 0, 0],
             [-1, 0, 4, 0, -1],
             [0, 0, 0, 0, 0],
             [0, 0, -1, 0, 0]],
            device=mosaic.device,
            dtype=mosaic.dtype,
        )
        hker = torch.tensor([[0.25, 0.5, 0.25], [0.5, 1.0, 0.5], [0.25, 0.5, 0.25]], device=mosaic.device, dtype=mosaic.dtype)
        tentativeB = guidedfilter3gf(green, mosaic[:, 2:3], mask[:, 2:3], h, v, eps, algorithm, f).clamp(0.0, 255.0)
        residualB = mask[:, 2:3] * (mosaic[:, 2:3] - tentativeB)
        residualB = filter2d_replicate(residualB, hker)
        blue = residualB + tentativeB

    return blue.clamp(0.0, 255.0)


def demosaic_RI(mosaic: torch.Tensor, pattern: str, sigma: float = 1.0, algorithm: str = "GBTF") -> torch.Tensor:
    """
    Torch implementation of the residual interpolation demosaicking family.
    Accepts either a 3-channel Bayer mosaic or a 1-channel raw Bayer image.
    Uses 0..255 image range to match the IPOL reference implementation.
    Returns RGB in NCHW format.
    """
    algorithm = algorithm.upper()
    if algorithm not in SUPPORTED_ALGORITHMS:
        raise ValueError(f"Unsupported algorithm {algorithm}. Expected one of {SUPPORTED_ALGORITHMS}.")

    mosaic = mosaic.to(torch.float32)
    if mosaic.dim() == 2:
        mosaic_rgb, mask = raw_to_mosaic(mosaic, pattern)
    elif mosaic.dim() == 3:
        if 3 in (mosaic.shape[0], mosaic.shape[-1]):
            mosaic_rgb, mask = mosaic_bayer(mosaic, pattern)
        else:
            mosaic_rgb, mask = raw_to_mosaic(mosaic, pattern)
    elif mosaic.dim() == 4:
        if mosaic.shape[1] == 3 or mosaic.shape[-1] == 3:
            mosaic_rgb, mask = mosaic_bayer(mosaic, pattern)
        elif mosaic.shape[1] == 1 or mosaic.shape[-1] == 1:
            mosaic_rgb, mask = raw_to_mosaic(mosaic, pattern)
        else:
            raise ValueError(f"Unsupported 4D mosaic shape {tuple(mosaic.shape)}.")
    else:
        raise ValueError(f"Unsupported mosaic rank {mosaic.dim()}.")

    green, dif = green_interpolation(mosaic_rgb, mask, pattern, sigma, algorithm)
    h = 5
    v = 5
    eps = 0.0
    red = red_interpolation(green, mosaic_rgb, mask, pattern, h, v, eps, dif, algorithm)
    blue = blue_interpolation(green, mosaic_rgb, mask, pattern, h, v, eps, dif, algorithm)
    return torch.cat([red, green, blue], dim=1)

