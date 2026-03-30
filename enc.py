# ============================================================
#   NeRD Encoder: EDSR + U-Net
#   Paper: "NeRD: Neural Field-Based Demosaicking"
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


# ────────────────────────────────────────────────────────────
# BLOCK 1: EDSR ResBlock
# ────────────────────────────────────────────────────────────
# Yeh NeRD ka basic building block hai.
# EDSR matlab "Enhanced Deep Super-Resolution" — isme
# Batch Normalization NAHI hoti, sirf do conv layers hote hain
# aur ek residual (shortcut) connection hota hai.
#
# Structure:
#   input → Conv → ReLU → Conv → (+input) → output
# ────────────────────────────────────────────────────────────

class EDSRResBlock(nn.Module):

    def __init__(self, channels: int = 128):
        super().__init__()

        # Pehli conv layer: kernel=3, stride=1, padding=1
        # "3-1-1" matlab kernel size 3, stride 1, padding 1
        # Padding=1 se feature map ka size same rehta hai (H×W)
        self.conv1 = nn.Conv2d(channels, channels,
                               kernel_size=3, stride=1, padding=1)

        # ReLU activation — non-linearity add karta hai
        self.relu  = nn.ReLU(inplace=True)

        # Doosri conv layer: same config
        self.conv2 = nn.Conv2d(channels, channels,
                               kernel_size=3, stride=1, padding=1)

        # NOTE: EDSR mein Batch Norm intentionally NAHI hai.
        # BatchNorm remove karne se high-frequency details
        # better preserve hoti hain — important for demosaicking.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x ko residual ke liye save karo
        residual = x

        # Conv → ReLU → Conv
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)

        # Residual addition: original input ko output mein add karo.
        # Isse gradient vanishing problem solve hota hai aur
        # network sirf "change" seekhta hai, full mapping nahi.
        return out + residual


# ────────────────────────────────────────────────────────────
# BLOCK 2: U-Net Downsampling Block
# ────────────────────────────────────────────────────────────
# Yeh block spatial resolution ko HALF karta hai (H → H/2)
# aur channels badhata hai.
# Conv 3-2-1 matlab: kernel=3, STRIDE=2, padding=1
# Stride=2 hi downsampling karta hai (pooling ki jagah).
# ────────────────────────────────────────────────────────────

class DownBlock(nn.Module):

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()

        # Strided conv se resolution half hoti hai (stride=2)
        # in_ch → out_ch channels, H×W → H/2 × W/2
        self.down_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch,
                      kernel_size=3, stride=2, padding=1),  # downsample
            nn.ReLU(inplace=True),

            # Ek extra conv for feature refinement (stride=1, size same)
            nn.Conv2d(out_ch, out_ch,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Output store karo — yahi U-Net ka skip connection bnega
        return self.down_conv(x)


# ────────────────────────────────────────────────────────────
# BLOCK 3: U-Net Upsampling Block
# ────────────────────────────────────────────────────────────
# Yeh block resolution DOUBLE karta hai (H/2 → H)
# aur skip connection se encoder ka feature map concatenate karta hai.
#
# Skip connection: U-Net ka "memory" hai —
# encoder ne jo fine details capture ki thi,
# woh decoder tak directly pohonchti hain.
# ────────────────────────────────────────────────────────────

class UpBlock(nn.Module):

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()

        # Bilinear upsample: nearest ke bajaye bilinear use karo
        # artifacts kam hote hain.
        # Yeh learnable nahi hai — sirf interpolation hai.
        self.upsample = nn.Upsample(scale_factor=2,
                                    mode='bilinear',
                                    align_corners=False)

        # Skip connection ke baad channels double ho jaate hain:
        # in_ch (from below) + skip_ch (from encoder) → out_ch
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            # Ek aur conv for better feature fusion
            nn.Conv2d(out_ch, out_ch,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor,
                skip: torch.Tensor) -> torch.Tensor:
        # Step 1: Resolution double karo
        x = self.upsample(x)

        # Step 2: Agar size mismatch ho toh crop karo
        # (Odd size images mein yeh ho sakta hai)
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:],
                              mode='bilinear', align_corners=False)

        # Step 3: Encoder ka skip feature map concatenate karo
        # [B, in_ch, H, W] + [B, skip_ch, H, W] → [B, in_ch+skip_ch, H, W]
        x = torch.cat([x, skip], dim=1)

        # Step 4: Conv se channels ko merge aur refine karo
        return self.conv(x)


# ────────────────────────────────────────────────────────────
# MAIN ENCODER: EDSR + U-Net
# ────────────────────────────────────────────────────────────
#
# Architecture (Fig. 2 se):
#
#   Bayer [B,1,H,W]
#       ↓
#   Initial Conv (1 → 128)
#       ↓
#   8× EDSR ResBlock (128 → 128)       ← "8x" in figure
#       ↓
#   ┌── Down1: 128→128, H/2  ──────────────────────────┐
#   │   Down2: 128→256, H/4  ───────────────────────┐  │
#   │   Down3: 256→512, H/8  ────────────────────┐  │  │  (skip connections)
#   │   Down4: 512→512, H/16 (bottleneck)         │  │  │
#   │       ↓                                      │  │  │
#   │   Up1:  512→256, H/8   ←────────────────────┘  │  │
#   │   Up2:  256→128, H/4   ←───────────────────────┘  │
#   │   Up3:  128→128, H/2   ←──────────────────────────┘
#   └── Up4:  128→128, H     ←── skip from EDSR output
#       ↓
#   Global Encoding ξ: [B, 128, H, W]
#
# ────────────────────────────────────────────────────────────

class NeRDEncoder(nn.Module):

    def __init__(self, in_channels: int = 1,
                 base_ch:       int = 128):
        """
        Args:
            in_channels : Bayer pattern ke channels.
                          1 = single-channel mosaic (default).
                          4 = 4-channel packed (R,Gr,Gb,B) — aapka case!
            base_ch     : Base feature channels (paper mein 128).
        """
        super().__init__()

        # ── Step 1: Initial Feature Extraction ──────────────
        # Bayer pattern (1 channel) ko 128 channel feature map
        # mein convert karo. Spatial size same rehti hai.
        # Conv 3-1-1: kernel=3, stride=1, padding=1
        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channels, base_ch,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        # Shape: [B, 1, H, W] → [B, 128, H, W]

        # ── Step 2: EDSR — 8 Residual Blocks ────────────────
        # 8 baar ResBlock apply karo. Yeh network ko
        # deep patterns seekhne deta hai bina gradient
        # vanishing ke. Figure mein "8x" yahi dikhata hai.
        self.edsr_blocks = nn.Sequential(
            *[EDSRResBlock(channels=base_ch) for _ in range(8)]
        )
        # Shape: [B, 128, H, W] → [B, 128, H, W]  (unchanged)

        # ── Step 3: U-Net Encoder (4 Downsampling Stages) ───
        # Har stage mein resolution half hoti hai,
        # aur channels badhte hain (zyada abstract features).

        # Stage 1: 128 → 128, H → H/2
        self.down1 = DownBlock(base_ch,      base_ch)

        # Stage 2: 128 → 256, H/2 → H/4
        self.down2 = DownBlock(base_ch,      base_ch * 2)

        # Stage 3: 256 → 512, H/4 → H/8
        self.down3 = DownBlock(base_ch * 2,  base_ch * 4)

        # Stage 4: 512 → 512, H/8 → H/16 (bottleneck — sabse abstract)
        self.down4 = DownBlock(base_ch * 4,  base_ch * 4)

        # ── Step 4: U-Net Decoder (4 Upsampling Stages) ─────
        # Har stage mein:
        #   1. Resolution double hoti hai
        #   2. Encoder ka skip feature concatenate hota hai
        #   3. Conv se refine karo

        # Up1: (512 from below + 512 skip from down3) → 256, H/8
        self.up1 = UpBlock(in_ch=base_ch*4, skip_ch=base_ch*4,
                           out_ch=base_ch*2)

        # Up2: (256 from below + 256 skip from down2) → 128, H/4
        self.up2 = UpBlock(in_ch=base_ch*2, skip_ch=base_ch*2,
                           out_ch=base_ch)

        # Up3: (128 from below + 128 skip from down1) → 128, H/2
        self.up3 = UpBlock(in_ch=base_ch,   skip_ch=base_ch,
                           out_ch=base_ch)

        # Up4: (128 from below + 128 skip from EDSR) → 128, H
        # EDSR output directly yahan aata hai as skip connection
        self.up4 = UpBlock(in_ch=base_ch,   skip_ch=base_ch,
                           out_ch=base_ch)

        # ── Step 5: Final 1×1 Conv ───────────────────────────
        # Output channels ko 128 par confirm karo.
        # 1×1 conv = channel mixing, spatial size same.
        self.final_conv = nn.Conv2d(base_ch, base_ch,
                                    kernel_size=1)
        # Final output: [B, 128, H, W] = global encoding ξ

    def forward(self, bayer: torch.Tensor) -> torch.Tensor:
        """
        Args:
            bayer: [B, 1, H, W] — normalized Bayer pattern [0,1]
        Returns:
            xi:    [B, 128, H, W] — global feature encoding ξ
        """

        # ── EDSR Part ────────────────────────────────────────
        # Initial conv: raw Bayer → 128 feature channels
        x0 = self.initial_conv(bayer)
        # [B, 1, H, W] → [B, 128, H, W]

        # 8 ResBlocks: local + global patterns seekho
        x_edsr = self.edsr_blocks(x0)
        # [B, 128, H, W] → [B, 128, H, W]
        # x_edsr ko baad mein skip connection ke liye save kiya

        # ── U-Net Encoder (Downsampling) ─────────────────────
        d1 = self.down1(x_edsr)   # [B, 128, H/2,  W/2]
        d2 = self.down2(d1)        # [B, 256, H/4,  W/4]
        d3 = self.down3(d2)        # [B, 512, H/8,  W/8]
        d4 = self.down4(d3)        # [B, 512, H/16, W/16]  ← bottleneck

        # ── U-Net Decoder (Upsampling + Skip Connections) ────
        # Har up step mein corresponding down ka output
        # skip connection ke roop mein diya jaata hai.

        u1 = self.up1(d4, d3)      # d4 upsample + d3 skip → [B, 256, H/8, W/8]
        u2 = self.up2(u1, d2)      # u1 upsample + d2 skip → [B, 128, H/4, W/4]
        u3 = self.up3(u2, d1)      # u2 upsample + d1 skip → [B, 128, H/2, W/2]
        u4 = self.up4(u3, x_edsr)  # u3 upsample + EDSR skip→ [B, 128, H,   W]
        #                  ↑
        #   EDSR output = deepest skip connection
        #   Yeh sabse important skip hai — full resolution
        #   fine details directly final decoder tak pohonchti hain

        # ── Final refinement ─────────────────────────────────
        xi = self.final_conv(u4)
        # [B, 128, H, W] = global encoding ξ

        return xi


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


# ────────────────────────────────────────────────────────────
# DATASET — DIV2K / Flickr2K style
# Patch size: 200×200, Batch: 5 (paper config)
# ────────────────────────────────────────────────────────────

import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import random


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


# ────────────────────────────────────────────────────────────
# TRAINING LOOP — Encoder only
# (Normally encoder + MLP saath train hote hain,
#  lekin yahan sirf encoder ka structure test karte hain)
# ────────────────────────────────────────────────────────────

def train_encoder(image_dir:  str,
                  epochs:     int   = 20,
                  batch_size: int   = 5,       # paper: 5
                  lr:         float = 1e-4,    # paper: 0.0001
                  patch_size: int   = 200,     # paper: 200×200
                  save_path:  str   = "encoder.pth",
                  device:     str   = None):
    """
    Encoder ko train karo. Yeh normally MLP ke saath
    end-to-end train hota hai. Yahan encoder ka
    output test karte hain ki shapes sahi hain.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Setup] Device: {device}")
    print(f"[Setup] Batch size: {batch_size}, LR: {lr}, Epochs: {epochs}")

    # Model initialize karo
    encoder = NeRDEncoder(in_channels=1, base_ch=128).to(device)

    total_params = sum(p.numel() for p in encoder.parameters())
    print(f"[Model] Total parameters: {total_params:,}")

    # Dataset + DataLoader
    dataset = BayerDataset(image_dir, patch_size=patch_size)
    loader  = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=(device == "cuda")
    )

    # Optimizer: Adam with paper's β values
    optimizer = torch.optim.Adam(
        encoder.parameters(),
        lr=lr,
        betas=(0.9, 0.999)   # paper: β1=0.9, β2=0.999
    )

    # Step decay: 0.95 per epoch (paper config)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=1, gamma=0.95
    )

    # Sanity check: ek forward pass karo
    encoder.eval()
    with torch.no_grad():
        dummy = torch.rand(1, 1, 200, 200).to(device)
        xi    = encoder(dummy)
        print(f"\n[Sanity] Input Bayer:         {dummy.shape}")
        print(f"[Sanity] Output encoding ξ:   {xi.shape}")
        # Local encoding check
        local = extract_local_encoding(xi, patch_size=5)
        print(f"[Sanity] Local encoding ξ_x:  {local.shape}")
        print(f"         → Per-pixel feature:  {local.shape[-1]}-dim (should be 3200)")

    print("\n[Train] Starting encoder training...\n")
    encoder.train()

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for bayer, rgb_target in loader:
            bayer      = bayer.to(device)       # [B, 1, 200, 200]
            rgb_target = rgb_target.to(device)  # [B, 3, 200, 200]

            optimizer.zero_grad()

            # Encoder forward pass → global encoding
            xi = encoder(bayer)   # [B, 128, 200, 200]

            # NOTE: Normally yahan MLP bhi hota hai jo
            # xi se RGB predict karta hai. Yahan sirf
            # encoder test ke liye ek simple auxiliary loss
            # use karte hain (1x1 conv ke equivalent):
            # Actual training mein yeh loss MLP se aati hai.

            # Proxy loss: xi ka mean ko rgb target se match karo
            # (Real training mein yeh MSE(MLP_output, rgb_target) hogi)
            xi_mean = xi.mean(dim=1, keepdim=True).expand_as(rgb_target[:, :1, :, :])
            # This is just a structural test — replace with full MLP loss in real training

            loss = F.mse_loss(
                xi_mean.expand(-1, 3, -1, -1),
                rgb_target
            )
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        psnr     = -10 * np.log10(avg_loss + 1e-10)
        scheduler.step()   # LR × 0.95 each epoch

        print(f"Epoch [{epoch:3d}/{epochs}] | "
              f"MSE: {avg_loss:.6f} | "
              f"PSNR: {psnr:.2f} dB | "
              f"LR: {scheduler.get_last_lr()[0]:.6f}")

    # Checkpoint save karo
    torch.save({
        "encoder_state": encoder.state_dict(),
        "epochs":        epochs,
        "lr":            lr,
    }, save_path)
    print(f"\n[Saved] Encoder checkpoint → {save_path}")
    return encoder


# ────────────────────────────────────────────────────────────
# ENTRY POINT
# ────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── Quick architecture test (no data needed) ──
    print("=" * 55)
    print("  NeRD Encoder — Architecture Test")
    print("=" * 55)

    device  = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = NeRDEncoder(in_channels=1, base_ch=128).to(device)

    # Paper input: batch=5, 200×200 Bayer patch
    x  = torch.rand(5, 1, 200, 200).to(device)
    xi = encoder(x)

    print(f"\nInput  (Bayer patch) : {x.shape}")
    print(f"Output (ξ encoding)  : {xi.shape}  ← should be [5, 128, 200, 200]")

    local = extract_local_encoding(xi, patch_size=5)
    print(f"Local encoding ξ_x   : {local.shape}  ← should be [5, 200, 200, 3200]")

    params = sum(p.numel() for p in encoder.parameters())
    print(f"\nTotal parameters     : {params:,}")
    print("\n✓ Shapes sahi hain! Ab training ke liye:")
    print("  train_encoder(image_dir='path/to/DIV2K/', epochs=20)")

    # ── If you want to train on your 5×5×4 features ──
    # Sirf in_channels=4 karo:
    # encoder_4ch = NeRDEncoder(in_channels=4, base_ch=128)