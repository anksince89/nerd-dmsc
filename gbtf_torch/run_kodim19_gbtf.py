import json
from pathlib import Path
import sys

import numpy as np
import torch
from PIL import Image

ROOT = Path("/Users/ankitkumar/Documents/nerd-dmsc")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gbtf_torch.ri_torch import demosaic_RI, mosaic_bayer


INPUT = ROOT / "data" / "val" / "kodim19.png"
OUTPUT = ROOT / "gbtf_torch" / "kodim_gbtf.png"
MOSAIC_OUTPUT = ROOT / "gbtf_torch" / "kodim_mosaic.png"
META_OUTPUT = ROOT / "gbtf_torch" / "kodim_gbtf_meta.json"


def load_rgb(path: Path) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    arr = np.array(img, dtype=np.float32)
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def save_rgb(path: Path, rgb: torch.Tensor) -> None:
    rgb = rgb.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    rgb = np.clip(rgb, 0.0, 255.0).astype(np.uint8)
    Image.fromarray(rgb).save(path)


def save_raw_preview(path: Path, mosaic: torch.Tensor) -> None:
    raw = mosaic.sum(dim=1, keepdim=True).squeeze().detach().cpu().numpy()
    raw = np.clip(raw, 0.0, 255.0).astype(np.uint8)
    Image.fromarray(raw).save(path)


def psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = torch.mean((pred - target) ** 2).item()
    return float(-10.0 * np.log10(mse / (255.0 * 255.0) + 1e-12))


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

    rgb = load_rgb(INPUT).to(device)
    pattern = "grbg"
    sigma = 1.0
    algorithm = "GBTF"

    mosaic, _ = mosaic_bayer(rgb, pattern)
    rgb_dem = demosaic_RI(mosaic, pattern=pattern, sigma=sigma, algorithm=algorithm).clamp(0.0, 255.0)

    save_rgb(OUTPUT, rgb_dem)
    save_raw_preview(MOSAIC_OUTPUT, mosaic)

    meta = {
        "input": str(INPUT),
        "output": str(OUTPUT),
        "mosaic_preview": str(MOSAIC_OUTPUT),
        "device": str(device),
        "pattern": pattern,
        "algorithm": algorithm,
        "sigma": sigma,
        "psnr_vs_source": psnr(rgb_dem, rgb),
        "shape": list(rgb_dem.shape),
    }
    META_OUTPUT.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
