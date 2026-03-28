#train.py
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from models.unet import UNet
from utils.metrics import dice_score_np

os.makedirs("outputs", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

if device.type == "cuda":
    torch.backends.cudnn.benchmark = True

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else torch.amp.GradScaler(enabled=False)


class EarlyStopping:
    def __init__(self, patience=15, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def step(self, val_score):
        if self.best_score is None:
            self.best_score = val_score
            return False

        if val_score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_score
            self.counter = 0

        return self.early_stop


def _num_workers():
    if device.type != "cuda":
        return 0
    cpu_count = os.cpu_count() or 4
    return min(8, max(2, cpu_count // 2))


def dice_score_torch(logits, target, eps=1e-6):
    prob = torch.sigmoid(logits)
    pred = (prob > 0.5).float()

    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = (2 * intersection + eps) / (union + eps)

    return dice.mean().item()


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, target):
        prob = torch.sigmoid(logits)
        intersection = (prob * target).sum(dim=(2, 3))
        union = prob.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


def _save_best_worst_examples(model, val_loader):
    model.eval()
    results = []

    with torch.no_grad():
        for imgs, masks in val_loader:
            if imgs.ndim == 3:
                imgs = imgs.unsqueeze(1)
            if masks.ndim == 3:
                masks = masks.unsqueeze(1)

            imgs = imgs.float().to(device, non_blocking=True)
            masks = masks.float().to(device, non_blocking=True)

            logits = model(imgs)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            for b in range(imgs.shape[0]):
                img_show = imgs[b, 0].cpu().numpy()
                mask_show = masks[b, 0].cpu().numpy()
                pred_show = preds[b, 0].cpu().numpy()

                dice_val = dice_score_np(pred_show, mask_show)
                results.append((dice_val, img_show, mask_show, pred_show))

    if not results:
        return

    results.sort(key=lambda x: x[0])
    worst = results[0]
    best = results[-1]

    for name, sample in zip(["worst", "best"], [worst, best]):
        dice_val, img_show, mask_show, pred_show = sample
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(img_show, cmap="gray")
        plt.title(f"{name.capitalize()} Image")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(mask_show, cmap="gray")
        plt.title("Ground Truth")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(pred_show, cmap="gray")
        plt.title(f"Prediction (Dice={dice_val:.3f})")
        plt.axis("off")

        plt.savefig(f"outputs/{name}_sample.png", bbox_inches="tight", dpi=180)
        plt.close()


def train_model(dataset, batch_size=8, epochs=120, lr=5e-4):
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size

    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

    num_workers = _num_workers()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
        drop_last=False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
        drop_last=False
    )

    model = UNet().to(device)

    # Optional PyTorch 2 compile for faster GPU training
    if device.type == "cuda" and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
        except Exception:
            pass

    pos_weight = torch.tensor([6.0], device=device)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    dice = DiceLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )
    early_stopper = EarlyStopping(patience=15)

    train_losses, val_losses = [], []
    train_dices, val_dices = [], []

    best_val_dice = 0.0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_dice = 0.0

        for img, mask in train_loader:
            if img.ndim == 3:
                img = img.unsqueeze(1)
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)

            img = img.float().to(device, non_blocking=True)
            mask = mask.float().to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                logits = model(img)
                loss = 0.5 * bce(logits, mask) + 0.5 * dice(logits, mask)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            train_dice += dice_score_torch(logits.detach(), mask)

        model.eval()
        val_loss = 0.0
        val_dice = 0.0

        with torch.no_grad():
            for img, mask in val_loader:
                if img.ndim == 3:
                    img = img.unsqueeze(1)
                if mask.ndim == 3:
                    mask = mask.unsqueeze(1)

                img = img.float().to(device, non_blocking=True)
                mask = mask.float().to(device, non_blocking=True)

                with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                    logits = model(img)
                    loss = 0.5 * bce(logits, mask) + 0.5 * dice(logits, mask)

                val_loss += loss.item()
                val_dice += dice_score_torch(logits, mask)

        avg_train_loss = train_loss / max(len(train_loader), 1)
        avg_val_loss = val_loss / max(len(val_loader), 1)
        avg_train_dice = train_dice / max(len(train_loader), 1)
        avg_val_dice = val_dice / max(len(val_loader), 1)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_dices.append(avg_train_dice)
        val_dices.append(avg_val_dice)

        scheduler.step(avg_val_dice)

        if avg_val_dice > best_val_dice:
            best_val_dice = avg_val_dice
            torch.save(model.state_dict(), "best_model.pth")
            print("🔥 Saved Best Model")

        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss: {avg_train_loss:.4f}, Train Dice: {avg_train_dice:.4f} | "
            f"Val Loss: {avg_val_loss:.4f}, Val Dice: {avg_val_dice:.4f}"
        )

        if early_stopper.step(avg_val_dice):
            print("🛑 Early stopping triggered")
            break

    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.legend()
    plt.title("Loss Curve")
    plt.savefig("outputs/loss_curve.png", bbox_inches="tight", dpi=180)
    plt.close()

    plt.figure()
    plt.plot(train_dices, label="Train Dice")
    plt.plot(val_dices, label="Val Dice")
    plt.legend()
    plt.title("Dice Curve")
    plt.savefig("outputs/dice_curve.png", bbox_inches="tight", dpi=180)
    plt.close()

    _save_best_worst_examples(model, val_loader)

    return model