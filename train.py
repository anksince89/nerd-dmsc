# train.py
# NeRD model NeRD.py se import ho raha hai — yahan sirf training logic hai
# pip install pytorch-msssim

import os
import math
import random
import numpy as np
from glob  import glob
from PIL   import Image

import torch
import torch.nn            as nn
import torch.nn.functional as F
from torch.utils.data      import Dataset, DataLoader
from pytorch_msssim        import ssim as compute_ssim   # SSIM metric

import NeRD


# ─────────────────────────────────────────────────────────────
# CONFIG  (paper exact values)
# ─────────────────────────────────────────────────────────────

CFG = {
    # Paper values
    "batch_size"   : 5,        # paper effective batch: 5
    "patch_size"   : 200,      # paper: 200×200
    "epoch_iters"  : 10_000,   # paper: 10000 iters per epoch
    "lr"           : 1e-4,     # paper: 0.0001
    "lr_gamma"     : 0.95,     # paper: decay × 0.95 per epoch
    "betas"        : (0.9, 0.999),

    # Training control
    "epochs"       : 50,
    "patience"     : 10,       # early stopping
    "num_workers"  : 4,
    "micro_batch_size": 1,     # fits low-VRAM GPUs
    "train_row_chunk_size": 8, # decoder rows per backward chunk
    "train_col_chunk_size": 200,
    "train_pixel_chunk_size": 4096,
    "val_row_chunk_size": 8,
    "val_col_chunk_size": 256,
    "val_pixel_chunk_size": 4096,
    "seed"         : 42,
    "save_dir"     : "checkpoints",

    # Paths
    "train_dir"    : "data/train",   # DIV2K + Flickr2K + OST
    "val_dir"      : "data/val",     # Kodak or McM
}

# ─────────────────────────────────────────────────────────────
# REPRODUCIBILITY
# ─────────────────────────────────────────────────────────────

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

# ─────────────────────────────────────────────────────────────
# BAYER SYNTHESIS  (GBRG — paper layout)
# ─────────────────────────────────────────────────────────────

def rgb_to_bayer(rgb: torch.Tensor) -> torch.Tensor:
    """RGB [3,H,W] in [0,1] → Bayer [1,H,W]"""
    b = torch.zeros(1, rgb.shape[1], rgb.shape[2], device=rgb.device)
    b[0, 0::2, 0::2] = rgb[1, 0::2, 0::2]   # Gr
    b[0, 0::2, 1::2] = rgb[2, 0::2, 1::2]   # B
    b[0, 1::2, 0::2] = rgb[0, 1::2, 0::2]   # R
    b[0, 1::2, 1::2] = rgb[1, 1::2, 1::2]   # Gb
    return b

# ─────────────────────────────────────────────────────────────
# DATASET
# Paper: 10000 random 200×200 crops synthesized per epoch
# ─────────────────────────────────────────────────────────────

class DemosaickDataset(Dataset):
    """
    Each call to __getitem__ picks a random image
    and returns a fresh random 200×200 crop.
    length = epoch_iters × effective_batch_size
    so that gradient accumulation still matches 10000 optimizer steps.
    """
    def __init__(self, image_paths, patch_size=200,
                 epoch_iters=10_000, batch_size=5):
        assert len(image_paths) > 0, "No images found!"
        self.paths      = image_paths
        self.patch_size = patch_size
        self.length     = epoch_iters * batch_size

    def __len__(self):
        return self.length

    def __getitem__(self, _):
        P   = self.patch_size
        img = None

        # Retry loop: skip corrupt/tiny images
        while img is None:
            path = random.choice(self.paths)
            try:
                img = Image.open(path).convert("RGB")
                W, H = img.size
                if W < P or H < P:
                    img = img.resize((max(W, P), max(H, P)),
                                     Image.BICUBIC)
                    W, H = img.size
            except Exception:
                img = None
                continue

        # Random 200×200 crop
        x     = random.randint(0, W - P)
        y     = random.randint(0, H - P)
        patch = img.crop((x, y, x + P, y + P))

        rgb   = torch.from_numpy(
                    np.array(patch, dtype=np.float32)
                ).permute(2, 0, 1) / 255.0        # [3, 200, 200]
        bayer = rgb_to_bayer(rgb)                  # [1, 200, 200]
        return bayer, rgb

# ─────────────────────────────────────────────────────────────
# METRICS  (paper uses PSNR + SSIM)
# ─────────────────────────────────────────────────────────────

def psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """pred, target in [0,1]  →  PSNR in dB"""
    mse = F.mse_loss(pred, target).item()
    return -10.0 * math.log10(mse + 1e-10)

@torch.no_grad()
def batch_metrics(pred: torch.Tensor,
                  target: torch.Tensor) -> dict:
    """
    pred, target: [B, 3, H, W] in [0, 1]
    Returns dict with psnr (dB) and ssim (0→1)
    """
    p = psnr(pred, target)
    s = compute_ssim(pred, target,
                     data_range=1.0,
                     size_average=True).item()
    return {"psnr": p, "ssim": s}


def pad_to_multiple(x: torch.Tensor,
                    multiple: int = 16) -> tuple[torch.Tensor, tuple[int, int]]:
    """
    Pad only on bottom/right so validation can be done on the whole image.
    """
    _, _, H, W = x.shape
    pad_h = (multiple - (H % multiple)) % multiple
    pad_w = (multiple - (W % multiple)) % multiple

    if pad_h == 0 and pad_w == 0:
        return x, (0, 0)

    x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
    return x, (pad_h, pad_w)


def remove_padding(x: torch.Tensor,
                   pad_h: int,
                   pad_w: int) -> torch.Tensor:
    if pad_h > 0:
        x = x[:, :, :-pad_h, :]
    if pad_w > 0:
        x = x[:, :, :, :-pad_w]
    return x

# ─────────────────────────────────────────────────────────────
# EARLY STOPPING
# ─────────────────────────────────────────────────────────────

class EarlyStopping:
    """mode='max' → PSNR improve hona chahiye"""
    def __init__(self, patience=10, mode="max", min_delta=1e-3):
        self.patience   = patience
        self.mode       = mode
        self.min_delta  = min_delta
        self.counter    = 0
        self.best       = None
        self.stop       = False

    def step(self, score: float) -> bool:
        if self.best is None:
            self.best = score
            return False

        improved = score > self.best + self.min_delta \
                   if self.mode == "max" \
                   else score < self.best - self.min_delta

        if improved:
            self.best    = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
        return self.stop

# ─────────────────────────────────────────────────────────────
# CHECKPOINT  SAVE / LOAD
# ─────────────────────────────────────────────────────────────

def save_ckpt(path, model, optimizer, scheduler, epoch, metrics):
    torch.save({
        "epoch"       : epoch,
        "model_state" : model.state_dict(),
        "optim_state" : optimizer.state_dict(),
        "sched_state" : scheduler.state_dict(),
        "metrics"     : metrics,           # {"psnr":..., "ssim":...}
    }, path)
    print(f"    ✓ Saved  → {path}  "
          f"[PSNR {metrics['psnr']:.2f} dB | SSIM {metrics['ssim']:.4f}]")


def load_ckpt(path, model, optimizer=None, scheduler=None):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    if optimizer and ckpt.get("optim_state"):
        optimizer.load_state_dict(ckpt["optim_state"])
    if scheduler and ckpt.get("sched_state"):
        scheduler.load_state_dict(ckpt["sched_state"])
    m = ckpt.get("metrics", {})
    print(f"  ✓ Resumed epoch {ckpt['epoch']} | "
          f"PSNR {m.get('psnr', 0):.2f} dB | SSIM {m.get('ssim', 0):.4f}")
    return ckpt["epoch"], m

# ─────────────────────────────────────────────────────────────
# TRAIN ONE EPOCH
# ─────────────────────────────────────────────────────────────

def train_one_epoch(model,
                    loader,
                    optimizer,
                    device,
                    accum_steps: int,
                    row_chunk_size: int,
                    col_chunk_size: int,
                    pixel_chunk_size: int):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad(set_to_none=True)
    micro_steps = 0

    for bayer, rgb_gt in loader:
        bayer  = bayer.to(device,  non_blocking=True)
        rgb_gt = rgb_gt.to(device, non_blocking=True)
        xi = model.encode(bayer)

        _, _, H, W = rgb_gt.shape
        batch_elems = rgb_gt.numel()
        row_chunk = H if row_chunk_size is None else row_chunk_size
        col_chunk = W if col_chunk_size is None else col_chunk_size

        num_row_tiles = math.ceil(H / row_chunk)
        num_col_tiles = math.ceil(W / col_chunk)
        total_tiles = num_row_tiles * num_col_tiles
        tile_idx = 0
        batch_sse = 0.0

        for row_start in range(0, H, row_chunk):
            row_end = min(row_start + row_chunk, H)

            for col_start in range(0, W, col_chunk):
                col_end = min(col_start + col_chunk, W)
                tile_idx += 1

                pred_tile = model.decode_chunk(
                    xi,
                    row_start=row_start,
                    row_end=row_end,
                    col_start=col_start,
                    col_end=col_end,
                    pixel_chunk_size=pixel_chunk_size)
                gt_tile = rgb_gt[:, :, row_start:row_end, col_start:col_end]

                tile_sse = F.mse_loss(pred_tile, gt_tile, reduction="sum")
                batch_sse += tile_sse.detach().item()

                scaled_loss = tile_sse / (batch_elems * accum_steps)
                scaled_loss.backward(retain_graph=(tile_idx < total_tiles))

        total_loss += batch_sse / batch_elems
        micro_steps += 1

        if micro_steps == accum_steps:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            micro_steps = 0

    if micro_steps > 0:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    avg_loss = total_loss / len(loader)
    avg_psnr = -10.0 * math.log10(avg_loss + 1e-10)
    return {"loss": avg_loss, "psnr": avg_psnr}

# ─────────────────────────────────────────────────────────────
# VALIDATE
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(model, val_paths, device):
    """
    Full-image validation (no resizing/cropping to 200×200).
    This is not the same protocol as Table 1 in the paper,
    which reports Kodak* / McM* after resizing and cropping.
    Returns avg PSNR (dB) and avg SSIM over all val images.
    """
    model.eval()
    psnr_list, ssim_list = [], []

    for path in val_paths:
        img  = Image.open(path).convert("RGB")
        rgb  = torch.from_numpy(
                   np.array(img, dtype=np.float32)
               ).permute(2, 0, 1).unsqueeze(0) / 255.0   # [1,3,H,W]

        rgb = rgb.to(device)
        rgb_pad, (pad_h, pad_w) = pad_to_multiple(rgb, multiple=16)
        bayer = rgb_to_bayer(rgb_pad.squeeze(0)).unsqueeze(0).to(device)

        xi = model.encode(bayer)
        pred_pad = model.decode_image(
            xi,
            row_chunk_size=CFG["val_row_chunk_size"],
            col_chunk_size=CFG["val_col_chunk_size"],
            pixel_chunk_size=CFG["val_pixel_chunk_size"])
        pred = remove_padding(pred_pad, pad_h, pad_w).clamp(0.0, 1.0)

        m = batch_metrics(pred, rgb)
        psnr_list.append(m["psnr"])
        ssim_list.append(m["ssim"])

    model.train()
    return {
        "psnr": float(np.mean(psnr_list)),
        "ssim": float(np.mean(ssim_list)),
    }

# ─────────────────────────────────────────────────────────────
# MAIN TRAINING
# ─────────────────────────────────────────────────────────────

def main(resume_path: str = None):
    set_seed(CFG["seed"])
    os.makedirs(CFG["save_dir"], exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert CFG["batch_size"] % CFG["micro_batch_size"] == 0, \
        "batch_size must be divisible by micro_batch_size"
    accum_steps = CFG["batch_size"] // CFG["micro_batch_size"]

    # ── Paths ──
    exts        = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
    train_paths = [f for e in exts
                   for f in glob(os.path.join(CFG["train_dir"], "**", e),
                                 recursive=True)]
    val_paths   = [f for e in exts
                   for f in glob(os.path.join(CFG["val_dir"], e))]

    print(f"Device      : {device}")
    print(f"Train images: {len(train_paths)}")
    print(f"Val images  : {len(val_paths)}")
    print(f"Effective BS: {CFG['batch_size']}")
    print(f"Micro BS    : {CFG['micro_batch_size']}")
    print(f"Accum steps : {accum_steps}")

    # ── Dataset & Loader ──
    train_ds = DemosaickDataset(
                   train_paths,
                   patch_size  = CFG["patch_size"],
                   epoch_iters = CFG["epoch_iters"],
                   batch_size  = CFG["batch_size"])

    train_loader = DataLoader(
                       train_ds,
                       batch_size  = CFG["micro_batch_size"],
                       shuffle     = True,
                       num_workers = CFG["num_workers"],
                       pin_memory  = True,
                       drop_last   = True)

    # ── Model (from NeRD.py) ──
    model = NeRD.NeRD(in_ch=1, out_ch=3).to(device)
    print(f"Parameters  : {sum(p.numel() for p in model.parameters()):,}")

    # ── Optimizer / Scheduler ──
    optimizer = torch.optim.Adam(model.parameters(),
                                  lr    = CFG["lr"],
                                  betas = CFG["betas"])
    scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer,
                    step_size = 1,
                    gamma     = CFG["lr_gamma"])   # ×0.95 per epoch

    es = EarlyStopping(patience=CFG["patience"], mode="max")

    # ── Resume ──
    start_epoch  = 1
    best_metrics = {"psnr": 0.0, "ssim": 0.0}

    if resume_path and os.path.exists(resume_path):
        start_epoch, best_metrics = load_ckpt(
            resume_path, model, optimizer, scheduler)
        start_epoch += 1

    print("=" * 65)
    print(f"{'Epoch':>6} | {'Train Loss':>10} | {'Train PSNR':>10} | "
          f"{'Val PSNR':>8} | {'Val SSIM':>8} | {'LR':>9}")
    print("=" * 65)

    # ── Epoch loop ──
    for epoch in range(start_epoch, CFG["epochs"] + 1):

        # Train
        t = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            accum_steps=accum_steps,
            row_chunk_size=CFG["train_row_chunk_size"],
            col_chunk_size=CFG["train_col_chunk_size"],
            pixel_chunk_size=CFG["train_pixel_chunk_size"])

        # Validate
        v = validate(model, val_paths, device)

        # Scheduler step (lr × 0.95)
        scheduler.step()
        lr = scheduler.get_last_lr()[0]

        # Log
        flag = " ◄" if v["psnr"] > best_metrics["psnr"] else ""
        print(f"{epoch:>6} | {t['loss']:>10.6f} | {t['psnr']:>10.2f} | "
              f"{v['psnr']:>8.2f} | {v['ssim']:>8.4f} | {lr:>9.7f}{flag}")

        # Save best checkpoint
        if v["psnr"] > best_metrics["psnr"]:
            best_metrics = v
            save_ckpt(os.path.join(CFG["save_dir"], "best.pth"),
                      model, optimizer, scheduler, epoch, v)

        # Periodic save every 10 epochs
        if epoch % 10 == 0:
            save_ckpt(os.path.join(CFG["save_dir"], f"epoch_{epoch:03d}.pth"),
                      model, optimizer, scheduler, epoch, v)

        # Early stopping check
        if es.step(v["psnr"]):
            print(f"\nEarly stopping at epoch {epoch} "
                  f"(no improvement for {CFG['patience']} epochs)")
            break

    print("=" * 65)
    print(f"Training complete.")
    print(f"Best Val PSNR : {best_metrics['psnr']:.2f} dB")
    print(f"Best Val SSIM : {best_metrics['ssim']:.4f}")


if __name__ == "__main__":
    main(resume_path=None)    # resume: main("checkpoints/best.pth")
