# Full trainable SIREN-based demosaicking pipeline
# Bayer CFA → 450-dim handcrafted features → SIREN MLP → RGB

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np

# ─────────────────────────────────────────────────────────
# 1. BAYER SYNTHESIS (for training from RGB images)
# ─────────────────────────────────────────────────────────

def rgb_to_bayer(rgb: torch.Tensor) -> torch.Tensor:
    """RGB [H,W,3] → Bayer GBRG [H,W]"""
    bayer = torch.zeros(rgb.shape[:2])
    bayer[0::2, 0::2] = rgb[0::2, 0::2, 1]  # G
    bayer[0::2, 1::2] = rgb[0::2, 1::2, 2]  # B
    bayer[1::2, 0::2] = rgb[1::2, 0::2, 0]  # R
    bayer[1::2, 1::2] = rgb[1::2, 1::2, 1]  # G
    return bayer


# ─────────────────────────────────────────────────────────
# 2. FEATURE EXTRACTION: Bayer → 450-dim per pixel
# ─────────────────────────────────────────────────────────
# Group 1: 4 raw channels (R,Gr,Gb,B)       4 × 25 = 100
# Group 2: 3 color diffs (R-G, B-G, Gr-Gb)  3 × 25 =  75
# Group 3: 8 gradients (H+V per channel)    8 × 25 = 200
# Group 4: 3 guided (G_interp, R-G, B-G)    3 × 25 =  75
# ─────────────────────────────────────────────────────────

def extract_features(bayer: torch.Tensor, patch_size: int = 5):
    """
    Input:  bayer [H, W] in [0,1]
    Output: features [H*W, 450], coords [H*W, 2]
    """
    H, W = bayer.shape
    pad  = patch_size // 2  # = 2

    # ── Unpack 4 CFA channels (GBRG layout) ──
    Gr = torch.zeros_like(bayer); Gr[0::2, 0::2] = bayer[0::2, 0::2]
    B  = torch.zeros_like(bayer); B [0::2, 1::2] = bayer[0::2, 1::2]
    R  = torch.zeros_like(bayer); R [1::2, 0::2] = bayer[1::2, 0::2]
    Gb = torch.zeros_like(bayer); Gb[1::2, 1::2] = bayer[1::2, 1::2]

    # ── Bilinear Green interpolation ──
    G_raw   = Gr + Gb
    G_count = (G_raw > 0).float()
    G_sum   = G_raw.clone()
    for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
        G_sum   += torch.roll(torch.roll(G_raw,   di, 0), dj, 1)
        G_count += torch.roll(torch.roll(G_count, di, 0), dj, 1)
    G_interp = G_sum / G_count.clamp(min=1.0)

    # ── Color differences ──
    R_mG    = R  - G_interp
    B_mG    = B  - G_interp
    Gr_mGb  = Gr - Gb

    # ── Directional gradients (H and V per channel) ──
    def grad_h(c): return torch.roll(c, -1, 1) - torch.roll(c, 1, 1)
    def grad_v(c): return torch.roll(c, -1, 0) - torch.roll(c, 1, 0)

    grad_planes = []
    for ch in [Gr, B, R, Gb]:
        grad_planes.append(grad_h(ch))
        grad_planes.append(grad_v(ch))

    # ── Combine all 18 planes ──
    planes = [Gr, B, R, Gb,           # 4 raw
              R_mG, B_mG, Gr_mGb,     # 3 color diff
              *grad_planes,            # 8 gradients
              G_interp, R_mG, B_mG]   # 3 guided
    # Total: 4+3+8+3 = 18 planes × 25 = 450

    # ── Extract 5×5 patches for every pixel ──
    all_patches = []
    for plane in planes:
        p = F.pad(plane.unsqueeze(0).unsqueeze(0),
                  (pad, pad, pad, pad), mode='reflect').squeeze()
        patches = p.unfold(0, patch_size, 1).unfold(1, patch_size, 1)
        # [H, W, 5, 5] → [H, W, 25]
        all_patches.append(patches.reshape(H, W, -1))

    features = torch.cat(all_patches, dim=-1)   # [H, W, 450]

    # ── Normalized spatial coordinates ──
    ys = torch.linspace(-1, 1, H).unsqueeze(1).expand(H, W)
    xs = torch.linspace(-1, 1, W).unsqueeze(0).expand(H, W)
    coords = torch.stack([ys, xs], dim=-1)       # [H, W, 2]

    return features.reshape(H*W, 450), coords.reshape(H*W, 2)


# ─────────────────────────────────────────────────────────
# 3. SIREN LAYER
# ─────────────────────────────────────────────────────────

class SirenLayer(nn.Module):
    def __init__(self, in_dim, out_dim, omega_0=1.0, is_first=False):
        super().__init__()
        self.omega_0  = omega_0
        self.is_first = is_first
        self.linear   = nn.Linear(in_dim, out_dim)
        self._init_weights(in_dim)

    def _init_weights(self, fan_in):
        with torch.no_grad():
            bound = 1.0 / fan_in if self.is_first else np.sqrt(6.0 / fan_in)
            self.linear.weight.uniform_(-bound, bound)
            self.linear.bias.uniform_(-bound, bound)

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))


# ─────────────────────────────────────────────────────────
# 4. SIREN MLP (with skip connections)
# ─────────────────────────────────────────────────────────

class DemosaickSIREN(nn.Module):
    """
    Input  → 452-dim (450 features + x,y)
    Layer0 → 256, sin, ω₀=30  [first layer]
    Layer1 → 256, sin
    Layer2 → 256, sin          [skip: concat input at entry]
    Layer3 → 256, sin
    Layer4 → 256, sin          [skip: concat input at entry]
    Out    → 3 (RGB), sigmoid
    """
    def __init__(self, feat_dim=450, hidden=256, omega_0=30.0):
        super().__init__()
        in_dim = feat_dim + 2

        self.layer0 = SirenLayer(in_dim,          hidden, omega_0=omega_0, is_first=True)
        self.layer1 = SirenLayer(hidden,           hidden, omega_0=1.0)
        self.layer2 = SirenLayer(hidden + in_dim,  hidden, omega_0=1.0)   # skip
        self.layer3 = SirenLayer(hidden,           hidden, omega_0=1.0)
        self.layer4 = SirenLayer(hidden + in_dim,  hidden, omega_0=1.0)   # skip
        self.out    = nn.Linear(hidden, 3)

    def forward(self, feat, coords):
        x = torch.cat([feat, coords], dim=-1)     # [N, 452]
        h = self.layer0(x)
        h = self.layer1(h)
        h = self.layer2(torch.cat([h, x], dim=-1))   # skip
        h = self.layer3(h)
        h = self.layer4(torch.cat([h, x], dim=-1))   # skip
        return torch.sigmoid(self.out(h))         # [N, 3] in [0,1]


# ─────────────────────────────────────────────────────────
# 5. DATASET
# ─────────────────────────────────────────────────────────

class BayerDataset(Dataset):
    def __init__(self, image_paths, pixels_per_image=10000):
        self.data = []
        for path in image_paths:
            from PIL import Image
            rgb = torch.tensor(
                np.array(Image.open(path).convert('RGB')), dtype=torch.float32
            ) / 255.0
            bayer = rgb_to_bayer(rgb)
            H, W  = bayer.shape
            feats, coords = extract_features(bayer)
            targets = rgb.reshape(H*W, 3)

            idx = torch.randperm(H*W)[:pixels_per_image]
            self.data.append((feats[idx], coords[idx], targets[idx]))

        # Flatten all images into one pool
        self.feats   = torch.cat([d[0] for d in self.data], 0)
        self.coords  = torch.cat([d[1] for d in self.data], 0)
        self.targets = torch.cat([d[2] for d in self.data], 0)

        # Compute normalization stats
        self.feat_mean = self.feats.mean(0, keepdim=True)
        self.feat_std  = self.feats.std(0, keepdim=True).clamp(min=1e-6)
        self.feats     = (self.feats - self.feat_mean) / self.feat_std

    def __len__(self):
        return self.feats.shape[0]

    def __getitem__(self, idx):
        return self.feats[idx], self.coords[idx], self.targets[idx]


# ─────────────────────────────────────────────────────────
# 6. TRAINING LOOP
# ─────────────────────────────────────────────────────────

def train(image_paths,
          epochs=20, lr=1e-4, batch_size=2048,
          hidden=256, omega_0=30.0,
          pixels_per_image=10000,
          save_path="siren_demosaick.pth"):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    dataset = BayerDataset(image_paths, pixels_per_image=pixels_per_image)
    loader  = DataLoader(dataset, batch_size=batch_size, 
                         shuffle=True, num_workers=4, pin_memory=True)

    model     = DemosaickSIREN(feat_dim=450, hidden=hidden, omega_0=omega_0).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for feats, coords, targets in loader:
            feats, coords, targets = (feats.to(device), 
                                      coords.to(device), 
                                      targets.to(device))
            optimizer.zero_grad()
            loss = nn.MSELoss()(model(feats, coords), targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg  = total_loss / len(loader)
        psnr = -10 * np.log10(avg + 1e-10)
        scheduler.step()
        print(f"Epoch {epoch:3d}/{epochs} | MSE: {avg:.6f} | PSNR: {psnr:.2f} dB")

    torch.save({
        "model_state": model.state_dict(),
        "feat_mean":   dataset.feat_mean,
        "feat_std":    dataset.feat_std,
        "hidden": hidden, "omega_0": omega_0
    }, save_path)
    print(f"Saved → {save_path}")


# ─────────────────────────────────────────────────────────
# 7. INFERENCE
# ─────────────────────────────────────────────────────────

def demosaick(bayer: torch.Tensor, checkpoint_path: str) -> torch.Tensor:
    """bayer [H,W] → rgb [H,W,3]"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt   = torch.load(checkpoint_path, map_location=device)

    model = DemosaickSIREN(feat_dim=450,
                           hidden=ckpt["hidden"],
                           omega_0=ckpt["omega_0"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    H, W       = bayer.shape
    feats, coords = extract_features(bayer)
    feats      = ((feats.to(device) - ckpt["feat_mean"].to(device))
                  / ckpt["feat_std"].to(device))
    coords     = coords.to(device)

    chunks = []
    with torch.no_grad():
        for i in range(0, H*W, 4096):      # chunked to avoid OOM
            chunks.append(model(feats[i:i+4096], coords[i:i+4096]).cpu())

    return torch.cat(chunks).reshape(H, W, 3)


# ─────────────────────────────────────────────────────────
# 8. ENTRY POINT
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    # ── Training ──
    # Point to your RGB images (JPG/PNG)
    train(
        image_paths=["img1.png", "img2.png"],   # add your image paths
        epochs=20,
        lr=1e-4,
        batch_size=2048,
        hidden=256,
        pixels_per_image=10000,
        save_path="siren_demosaick.pth"
    )

    # ── Inference ──
    from PIL import Image
    rgb   = torch.tensor(np.array(Image.open("test.png").convert("RGB")),
                         dtype=torch.float32) / 255.0
    bayer = rgb_to_bayer(rgb)
    result = demosaick(bayer, "siren_demosaick.pth")   # [H, W, 3]
    out_img = Image.fromarray((result.numpy() * 255).astype(np.uint8))
    out_img.save("demosaicked_output.png")