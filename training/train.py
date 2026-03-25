import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from models.unet import UNet
from utils.visualization import show_error_map
from utils.metrics import region_dice
from visualization.polar_map import generate_polar_map
import random
import numpy as np
import os

# -----------------------------
# Create necessary folders
# -----------------------------
folders = [
    "outputs/predictions",
    "outputs/overlays",
    "outputs/errors",
    "outputs/confidence",
    "outputs/polar"
]
for f in folders:
    os.makedirs(f, exist_ok=True)

# -----------------------------
# Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -----------------------------
# GradScaler for mixed precision
# -----------------------------
scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else torch.amp.GradScaler(enabled=False)

# -----------------------------
# Seed
# -----------------------------
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# -----------------------------
# EarlyStopping
# -----------------------------
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0):
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

# -----------------------------
# Dice Score & Dice Loss
# -----------------------------
def dice_score(pred, target):
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = (2 * intersection + 1e-6) / (union + 1e-6)
    return dice.mean().item()

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()

# -----------------------------
# Train Function
# -----------------------------
def train_model(dataset, batch_size=4, epochs=120, lr=5e-4):
    # -----------------------------
    # Split dataset
    # -----------------------------
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=(device.type=="cuda"))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=(device.type=="cuda"))

    # -----------------------------
    # Model, Loss, Optimizer
    # -----------------------------
    model = UNet().to(device)
    pos_weight = torch.tensor([6.0]).to(device)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    dice = DiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    early_stopper = EarlyStopping(patience=15)

    train_losses, val_losses, train_dices, val_dices = [], [], [], []
    best_val_dice = 0

    # -----------------------------
    # Training Loop
    # -----------------------------
    for epoch in range(epochs):
        model.train()
        train_loss, train_dice_val = 0, 0
        for img, mask in train_loader:
            if img.ndim == 3:
                img = img.unsqueeze(1)
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)
            img = img.float().to(device)
            mask = mask.float().to(device)

            optimizer.zero_grad()
            with torch.amp.autocast(device_type=device.type, enabled=(device.type=="cuda")):
                pred = model(img)
                loss = 0.5 * bce(pred, mask) + 0.5 * dice(pred, mask)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            train_dice_val += dice_score(pred, mask)

        # Validation
        model.eval()
        val_loss, val_dice_val = 0, 0
        with torch.no_grad():
            for img, mask in val_loader:
                if img.ndim == 3:
                    img = img.unsqueeze(1)
                if mask.ndim == 3:
                    mask = mask.unsqueeze(1)
                img = img.float().to(device)
                mask = mask.float().to(device)
                pred = model(img)
                loss = 0.5 * bce(pred, mask) + 0.5 * dice(pred, mask)
                val_loss += loss.item()
                val_dice_val += dice_score(pred, mask)

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        avg_train_dice = train_dice_val / len(train_loader)
        avg_val_dice = val_dice_val / len(val_loader)
        scheduler.step(avg_val_dice)

        if avg_val_dice > best_val_dice:
            best_val_dice = avg_val_dice
            torch.save(model.state_dict(), "best_model.pth")
            print("🔥 Saved Best Model")

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_dices.append(avg_train_dice)
        val_dices.append(avg_val_dice)

        print(
            f"Epoch {epoch} | "
            f"Train Loss: {avg_train_loss:.4f}, Train Dice: {avg_train_dice:.4f} | "
            f"Val Loss: {avg_val_loss:.4f}, Val Dice: {avg_val_dice:.4f}"
        )

        if early_stopper.step(avg_val_dice):
            print("🛑 Early stopping triggered")
            break

    # -----------------------------
    # Plot Loss & Dice
    # -----------------------------
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.legend(); plt.title("Loss Curve")
    plt.savefig("outputs/loss_curve.png"); plt.close()

    plt.plot(train_dices, label="Train Dice")
    plt.plot(val_dices, label="Val Dice")
    plt.legend(); plt.title("Dice Curve")
    plt.savefig("outputs/dice_curve.png"); plt.close()

    # -----------------------------
    # Visualization on Validation Set (FIXED)
    # -----------------------------
    model.eval()

    num_samples = 10
    collected = 0
    results = []

    with torch.no_grad():
        for imgs, masks in val_loader:

            if imgs.ndim == 3:
                imgs = imgs.unsqueeze(1)
            if masks.ndim == 3:
                masks = masks.unsqueeze(1)

            imgs = imgs.float().to(device)
            masks = masks.float().to(device)

            outputs = model(imgs)
            probs = torch.sigmoid(outputs).cpu().numpy()
            preds = (probs > 0.5).astype(np.float32)

            for b in range(imgs.shape[0]):

                if collected >= num_samples:
                    break

                img_show = imgs[b][0].cpu().numpy()
                mask_show = masks[b][0].cpu().numpy()

                pred_bin = preds[b].squeeze()
                pred_show = probs[b].squeeze()

                # Dice
                intersection = (pred_bin * mask_show).sum()
                union = pred_bin.sum() + mask_show.sum()
                dice_val = (2 * intersection + 1e-6) / (union + 1e-6)

                results.append((dice_val, img_show, mask_show, pred_bin))

                # -----------------------------
                # Prediction
                # -----------------------------
                plt.figure(figsize=(10, 4))

                plt.subplot(1, 3, 1)
                plt.imshow(img_show, cmap='gray')
                plt.title("Image")

                plt.subplot(1, 3, 2)
                plt.imshow(mask_show, cmap='gray')
                plt.title("Ground Truth")

                plt.subplot(1, 3, 3)
                plt.imshow(pred_bin, cmap='gray')
                plt.title(f"Prediction (Dice={dice_val:.3f})")

                plt.savefig(f"outputs/predictions/pred_{collected}.png")
                plt.close()

                # -----------------------------
                # Overlay
                # -----------------------------
                plt.figure(figsize=(5, 5))
                plt.imshow(img_show, cmap='gray')
                plt.imshow(mask_show, alpha=0.3, cmap='Greens')
                plt.imshow(pred_bin, alpha=0.3, cmap='Reds')
                plt.title(f"Overlay (Dice={dice_val:.3f})")
                plt.axis('off')
                plt.savefig(f"outputs/overlays/overlay_{collected}.png")
                plt.close()

                # -----------------------------
                # Error map (NOW ALWAYS 10)
                # -----------------------------
                show_error_map(
                    img_show,
                    mask_show,
                    pred_bin,
                    f"outputs/errors/pred_{collected}.png"
                )

                # -----------------------------
                # Confidence map
                # -----------------------------
                epsilon = 1e-6
                entropy = -(pred_show * np.log(pred_show + epsilon) +
                            (1 - pred_show) * np.log(1 - pred_show + epsilon))
                confidence_map = 1 - entropy

                plt.imshow(confidence_map, cmap='jet')
                plt.title("Confidence")
                plt.colorbar()
                plt.savefig(f"outputs/confidence/pred_{collected}.png")
                plt.close()

                np.save(f"outputs/confidence/conf_{collected}.npy", confidence_map)

                # -----------------------------
                # Failure detection
                # -----------------------------
                mean_conf = np.mean(confidence_map)

                failure = (dice_val < 0.6 and mean_conf < 0.5) or \
                        (dice_val < 0.4) or \
                        (mean_conf < 0.2)

                if failure:
                    print(f"⚠️ Failure detected (Sample {collected}) | Dice={dice_val:.2f}, Conf={mean_conf:.2f}")

                if dice_val < 0.5 and mean_conf > 0.7:
                    print(f"🚨 Overconfident WRONG prediction (Sample {collected})")

                # -----------------------------
                # Region Dice
                # -----------------------------
                region_scores = region_dice(pred_bin, mask_show)
                print(f"Sample {collected} Region Dice:", region_scores)

                # -----------------------------
                # Polar Map
                # -----------------------------
                generate_polar_map(pred_bin, f"outputs/polar/pred_{collected}.png")

                collected += 1

            if collected >= num_samples:
                break

    # -----------------------------
    # Best & Worst Samples
    # -----------------------------
    results.sort(key=lambda x: x[0])
    worst, best = results[0], results[-1]
    for name, sample in zip(["Worst", "Best"], [worst, best]):
        dice_val, img_show, mask_show, pred_show = sample
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 3, 1); plt.imshow(img_show, cmap='gray'); plt.title(f"{name} Image")
        plt.subplot(1, 3, 2); plt.imshow(mask_show, cmap='gray'); plt.title("GT")
        plt.subplot(1, 3, 3); plt.imshow(pred_show, cmap='gray'); plt.title(f"{name} (Dice={dice_val:.2f})")
        plt.savefig(f"outputs/{name.lower()}_sample.png"); plt.close()

    return model