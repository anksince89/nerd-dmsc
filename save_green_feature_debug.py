import argparse
import json
import os
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from nerd_feature_model import BayerFeatureExtractor, rgb_to_bayer


def _resample_bilinear():
    try:
        return Image.Resampling.BILINEAR
    except AttributeError:
        return Image.BILINEAR


def load_input_image(path: str) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Returns:
        bayer [1,1,H,W]
        rgb   [1,3,H,W] if source image is RGB, else None
    """
    img = Image.open(path)
    arr = np.array(img)

    if arr.ndim == 2:
        arr = arr.astype(np.float32)
        if arr.max() > 1.0:
            arr = arr / 255.0
        bayer = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
        return bayer, None

    if arr.ndim == 3 and arr.shape[2] >= 3:
        arr = arr[:, :, :3].astype(np.float32)
        if arr.max() > 1.0:
            arr = arr / 255.0
        rgb = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        return torch.empty(0), rgb

    raise ValueError(f"Unsupported image shape: {arr.shape}")


def normalize_feature_map(feature: np.ndarray) -> Tuple[np.ndarray, Dict[str, Union[float, str]]]:
    """
    Converts a feature map to uint8 for visualization and returns normalization metadata.
    """
    eps = 1e-8
    fmin = float(feature.min())
    fmax = float(feature.max())
    fmean = float(feature.mean())
    fstd = float(feature.std())

    if fmin < 0.0 and fmax > 0.0:
        scale = max(abs(fmin), abs(fmax), eps)
        vis = np.clip(feature / (2.0 * scale) + 0.5, 0.0, 1.0)
        meta = {
            "mode": "signed_symmetric",
            "scale": float(scale),
            "min": fmin,
            "max": fmax,
            "mean": fmean,
            "std": fstd,
        }
        return (vis * 255.0).astype(np.uint8), meta

    lo, hi = np.percentile(feature, [1.0, 99.0])
    if hi <= lo + eps:
        lo, hi = fmin, fmax + eps
    vis = np.clip((feature - lo) / (hi - lo + eps), 0.0, 1.0)
    meta = {
        "mode": "percentile_1_99",
        "lo": float(lo),
        "hi": float(hi),
        "min": fmin,
        "max": fmax,
        "mean": fmean,
        "std": fstd,
    }
    return (vis * 255.0).astype(np.uint8), meta


def save_grayscale_png(array: np.ndarray, path: str) -> None:
    Image.fromarray(array).save(path)


def save_rgb_png(array: np.ndarray, path: str) -> None:
    Image.fromarray(array).save(path)


def tensor_to_uint8_gray(tensor: torch.Tensor) -> np.ndarray:
    arr = tensor.squeeze().detach().cpu().numpy()
    arr = np.clip(arr, 0.0, 1.0)
    return (arr * 255.0).astype(np.uint8)


def tensor_to_uint8_rgb(tensor: torch.Tensor) -> np.ndarray:
    arr = tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    arr = np.clip(arr, 0.0, 1.0)
    return (arr * 255.0).astype(np.uint8)


def build_contact_sheet(feature_images: Dict[str, np.ndarray], path: str, tile_size: int = 160, cols: int = 4) -> None:
    labels = list(feature_images.keys())
    rows = (len(labels) + cols - 1) // cols
    label_h = 28
    canvas = Image.new("RGB", (cols * tile_size, rows * (tile_size + label_h)), color=(24, 24, 24))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    resample = _resample_bilinear()

    for idx, name in enumerate(labels):
        row = idx // cols
        col = idx % cols
        x0 = col * tile_size
        y0 = row * (tile_size + label_h)
        tile = Image.fromarray(feature_images[name]).resize((tile_size, tile_size), resample=resample).convert("RGB")
        canvas.paste(tile, (x0, y0))
        draw.text((x0 + 4, y0 + tile_size + 6), name, fill=(235, 235, 235), font=font)

    canvas.save(path)


def crop_around_peak(image: np.ndarray, center_y: int, center_x: int, crop_size: int) -> np.ndarray:
    h, w = image.shape[:2]
    half = crop_size // 2
    top = min(max(center_y - half, 0), max(h - crop_size, 0))
    left = min(max(center_x - half, 0), max(w - crop_size, 0))
    bottom = min(top + crop_size, h)
    right = min(left + crop_size, w)
    return image[top:bottom, left:right]


def main():
    parser = argparse.ArgumentParser(description="Save green-demosaicking feature debug maps.")
    parser.add_argument("--input", required=True, help="Input image path. RGB will be mosaicked to Bayer.")
    parser.add_argument("--output-dir", default="data/debug", help="Directory to save debug outputs.")
    parser.add_argument("--pattern", default="gbrg", choices=["gbrg", "grbg", "rggb", "bggr"], help="Bayer CFA pattern.")
    parser.add_argument("--crop-size", type=int, default=192, help="Diagnostic crop size around strongest periodic response.")
    args = parser.parse_args()

    input_name = os.path.splitext(os.path.basename(args.input))[0]
    out_dir = args.output_dir
    if os.path.basename(out_dir) != input_name:
        out_dir = os.path.join(out_dir, input_name)
    os.makedirs(out_dir, exist_ok=True)

    bayer_in, rgb = load_input_image(args.input)
    if rgb is not None:
        bayer = rgb_to_bayer(rgb, pattern=args.pattern)
    else:
        bayer = bayer_in

    extractor = BayerFeatureExtractor(pattern=args.pattern)
    extractor.eval()

    with torch.no_grad():
        feature_dict = extractor.extract_feature_dict(bayer)
        feature_stack = torch.cat(list(feature_dict.values()), dim=1)

    bayer_vis = tensor_to_uint8_gray(bayer)
    save_grayscale_png(bayer_vis, os.path.join(out_dir, "bayer_input.png"))

    if rgb is not None:
        rgb_vis = tensor_to_uint8_rgb(rgb)
        save_rgb_png(rgb_vis, os.path.join(out_dir, "rgb_input.png"))
        green_vis = tensor_to_uint8_gray(rgb[:, 1:2])
        save_grayscale_png(green_vis, os.path.join(out_dir, "green_ground_truth.png"))

    feature_images: Dict[str, np.ndarray] = {}
    feature_stats: Dict[str, Dict[str, Union[float, str]]] = {}

    for name, tensor in feature_dict.items():
        fmap = tensor[0, 0].cpu().numpy()
        vis, stats = normalize_feature_map(fmap)
        feature_images[name] = vis
        feature_stats[name] = stats
        save_grayscale_png(vis, os.path.join(out_dir, f"{name}.png"))

    build_contact_sheet(feature_images, os.path.join(out_dir, "contact_sheet.png"))

    periodic_keys = [
        "stripe_x",
        "stripe_y",
        "checkerboard_energy",
        "stripe_energy",
        "gabor_45",
        "gabor_135",
        "dct_periodic",
    ]
    periodic_score = sum(feature_dict[key].abs() for key in periodic_keys)
    periodic_map = periodic_score[0, 0].cpu().numpy()
    periodic_vis, periodic_stats = normalize_feature_map(periodic_map)
    save_grayscale_png(periodic_vis, os.path.join(out_dir, "periodicity_score.png"))

    peak_index = int(np.argmax(periodic_map))
    peak_y, peak_x = np.unravel_index(peak_index, periodic_map.shape)

    bayer_crop = crop_around_peak(bayer_vis, peak_y, peak_x, args.crop_size)
    save_grayscale_png(bayer_crop, os.path.join(out_dir, "bayer_periodic_crop.png"))

    if rgb is not None:
        green_crop = crop_around_peak(green_vis, peak_y, peak_x, args.crop_size)
        save_grayscale_png(green_crop, os.path.join(out_dir, "green_periodic_crop.png"))
        rgb_crop = crop_around_peak(rgb_vis, peak_y, peak_x, args.crop_size)
        save_rgb_png(rgb_crop, os.path.join(out_dir, "rgb_periodic_crop.png"))

    np.savez_compressed(
        os.path.join(out_dir, "feature_stack.npz"),
        feature_stack=feature_stack.squeeze(0).cpu().numpy(),
        feature_names=np.array(list(feature_dict.keys()), dtype=object),
    )

    manifest = {
        "input": os.path.abspath(args.input),
        "output_dir": os.path.abspath(out_dir),
        "pattern": args.pattern,
        "feature_channels": int(feature_stack.shape[1]),
        "height": int(feature_stack.shape[2]),
        "width": int(feature_stack.shape[3]),
        "periodic_peak": {"y": int(peak_y), "x": int(peak_x)},
        "periodicity_score": periodic_stats,
        "features": feature_stats,
    }
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Saved debug outputs to {os.path.abspath(out_dir)}")


if __name__ == "__main__":
    main()
