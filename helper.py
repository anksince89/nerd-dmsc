# ============================================================
#   NeRD Encoder: EDSR + U-Net
#   Paper: "NeRD: Neural Field-Based Demosaicking"
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
import os
import random
import numpy as np

# ────────────────────────────────────────────────────────────
# LOCAL ENCODING EXTRACTION
# ────────────────────────────────────────────────────────────
# Paper kehta hai: global encoding (H×W×128) se
# har pixel ke liye ek 5×5 patch nikalo → flatten → 3200-dim vector.
# Yahi ξ_x hai jo MLP ko diya jaata hai.
# ────────────────────────────────────────────────────────────

def extract_local_encoding(xi: torch.Tensor,
                            patch_size: int = 5) -> torch.Tensor:
    """
    Global encoding se har pixel ke liye local 5×5 patch nikalo.

    Args:
        xi         : [B, 128, H, W] — encoder output
        patch_size : 5 (paper default)
    Returns:
        local_enc  : [B, H, W, 128*5*5] = [B, H, W, 3200]
    """
    B, C, H, W = xi.shape
    pad = patch_size // 2  # = 2

    # Border pixels ke liye reflect padding
    # Reflect = edge values mirror karo (aritifacts kam hote hain)
    xi_padded = F.pad(xi, (pad, pad, pad, pad), mode='reflect')
    # [B, 128, H+4, W+4]

    # unfold se sliding 5×5 windows extract karo
    # unfold(dimension, size, step)
    patches = xi_padded \
        .unfold(2, patch_size, 1) \
        .unfold(3, patch_size, 1)
    # [B, 128, H, W, 5, 5]

    # Flatten: 128 channels × 5×5 = 3200
    local_enc = patches.reshape(B, C, H, W, -1)
    # [B, 128, H, W, 25]

    local_enc = local_enc.reshape(B, C * patch_size * patch_size, H, W)
    # [B, 3200, H, W]

    # Permute for MLP-friendly format: pixel-first
    local_enc = local_enc.permute(0, 2, 3, 1)
    # [B, H, W, 3200]

    return local_enc

# ─────────────────────────────────────────────────────────
# STEP 2: Spatial Coordinates Generator
# Har pixel ke liye normalized (x,y) ∈ [-1, 1]
# ─────────────────────────────────────────────────────────

def make_coords(B: int, H: int, W: int,
                device: torch.device) -> torch.Tensor:
    """
    Output: coords [B, H, W, 2]
    coords[:,:,:,0] = y (row)   normalized to [-1, 1]
    coords[:,:,:,1] = x (col)   normalized to [-1, 1]
    """
    ys = torch.linspace(-1, 1, H, device=device)  # [H]
    xs = torch.linspace(-1, 1, W, device=device)  # [W]

    # meshgrid se 2D grid banao
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
    # grid_y, grid_x: [H, W]

    coords = torch.stack([grid_y, grid_x], dim=-1)
    # [H, W, 2]

    coords = coords.unsqueeze(0).expand(B, -1, -1, -1)
    # [B, H, W, 2]

    return coords

# ────────────────────────────────────────────────────────────
# DATASET — DIV2K / Flickr2K style
# Patch size: 200×200, Batch: 5 (paper config)
# ────────────────────────────────────────────────────────────

class BayerDataset(Dataset):
    """
    RGB images load karo → Bayer pattern banao → encoder ko do.
    Paper: 200×200 random crops, GBRG pattern.
    """

    def __init__(self, image_dir: str,
                 patch_size: int = 200,
                 augment: bool = True):

        # Saare image paths collect karo
        self.paths = sorted([
            os.path.join(image_dir, f)
            for f in os.listdir(image_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        self.patch_size = patch_size
        self.augment    = augment

        if len(self.paths) == 0:
            raise ValueError(f"No images found in {image_dir}")
        print(f"Dataset: {len(self.paths)} images found.")

    def _rgb_to_bayer_gbrg(self, rgb: np.ndarray) -> np.ndarray:
        """
        RGB numpy [H,W,3] → Bayer GBRG single-channel [H,W].
        GBRG layout:
          (even,even)=G  (even,odd)=B
          (odd, even)=R  (odd, odd)=G
        """
        H, W = rgb.shape[:2]
        bayer = np.zeros((H, W), dtype=np.float32)
        bayer[0::2, 0::2] = rgb[0::2, 0::2, 1]  # G (top-left)
        bayer[0::2, 1::2] = rgb[0::2, 1::2, 2]  # B (top-right)
        bayer[1::2, 0::2] = rgb[1::2, 0::2, 0]  # R (bottom-left)
        bayer[1::2, 1::2] = rgb[1::2, 1::2, 1]  # G (bottom-right)
        return bayer

    def __len__(self):
        # Paper: ek epoch mein 10,000 random patches
        # Practically: dataset size = image count
        return len(self.paths)

    def __getitem__(self, idx: int):
        # Image load karo aur [0,1] mein normalize karo
        img = Image.open(self.paths[idx]).convert('RGB')
        img = np.array(img, dtype=np.float32) / 255.0

        H, W = img.shape[:2]
        P    = self.patch_size

        # Chhoti image check
        if H < P or W < P:
            img = np.array(
                Image.fromarray((img * 255).astype(np.uint8))
                     .resize((max(W, P), max(H, P))),
                dtype=np.float32
            ) / 255.0
            H, W = img.shape[:2]

        # Random 200×200 crop (paper config)
        top  = random.randint(0, H - P)
        left = random.randint(0, W - P)
        img  = img[top:top+P, left:left+P]  # [200, 200, 3]

        # Data augmentation: random flip + rotation
        if self.augment:
            if random.random() > 0.5:
                img = np.fliplr(img).copy()   # horizontal flip
            if random.random() > 0.5:
                img = np.flipud(img).copy()   # vertical flip
            k = random.randint(0, 3)
            img = np.rot90(img, k).copy()     # 0/90/180/270 rotation

        # RGB → Bayer GBRG
        bayer = self._rgb_to_bayer_gbrg(img)

        # Numpy → Torch tensors
        # Bayer: [H,W] → [1,H,W] (channel dimension add karo)
        bayer_t = torch.from_numpy(bayer).unsqueeze(0)   # [1, 200, 200]

        # RGB target: [H,W,3] → [3,H,W] (channels first for PyTorch)
        rgb_t   = torch.from_numpy(
            img.transpose(2, 0, 1)                        # [3, 200, 200]
        )

        return bayer_t, rgb_t
    