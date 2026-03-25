import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from visualization.overlay import show_overlay
from visualization.polar_map import generate_polar_map
from explainability.gradcam import gradcam
from utils.metrics import region_dice


# -----------------------------
# PREPROCESS IMAGE
# -----------------------------
def preprocess_image(image):
    if isinstance(image, np.ndarray):
        image = torch.from_numpy(image)
    image = image.float()

    if image.ndim == 2:
        image = image.unsqueeze(0).unsqueeze(0)
    elif image.ndim == 3:
        if image.shape[-1] in [1, 3]:
            image = image.permute(2, 0, 1)
        image = image.unsqueeze(0)
    elif image.ndim == 4:
        pass
    else:
        raise ValueError(f"Unsupported shape: {image.shape}")

    if image.shape[1] != 1:
        image = image.mean(dim=1, keepdim=True)

    return image


# -----------------------------
# PREDICT
# -----------------------------
def predict(model, image):
    model.eval()
    image = preprocess_image(image)
    device = next(model.parameters()).device
    image = image.to(device)

    with torch.no_grad():
        logits = model(image)
        probs = torch.sigmoid(logits)

    pred = (probs > 0.5).float()
    return pred.cpu().numpy(), probs.cpu().numpy()


# -----------------------------
# UNCERTAINTY
# -----------------------------
def enable_dropout(model):
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()


def predict_with_uncertainty(model, image, n_samples=10):
    model.eval()
    enable_dropout(model)

    image = preprocess_image(image)
    device = next(model.parameters()).device
    image = image.to(device)

    preds = []
    for _ in range(n_samples):
        with torch.no_grad():
            preds.append(torch.sigmoid(model(image)))

    preds = torch.stack(preds)
    mean_pred = preds.mean(dim=0)
    uncertainty = preds.var(dim=0)

    return mean_pred.cpu().numpy(), uncertainty.cpu().numpy()


# -----------------------------
# FULL DATASET PREDICTION
# -----------------------------
def predict_dataset(model, dataset, output_dir="outputs", mc_samples=0, save_uncertainty=False):

    # -----------------------------
    # CREATE STRUCTURE
    # -----------------------------
    full_dirs = ["full_overlay", "full_gradcam", "full_uncertainty", "full_polarmap"]
    selected_dirs = ["predictions", "overlays", "errors", "confidence", "polar", "gradcam"]

    for d in full_dirs + selected_dirs:
        os.makedirs(os.path.join(output_dir, d), exist_ok=True)

    device = next(model.parameters()).device

    results = []
    region_dice_dict = {}

    # -----------------------------
    # LOOP ALL SLICES (2890)
    # -----------------------------
    for idx in range(len(dataset)):
        image, mask = dataset[idx]

        # Predict
        if mc_samples > 0:
            pred, uncertainty = predict_with_uncertainty(model, image, mc_samples)
        else:
            pred, _ = predict(model, image)
            uncertainty = None

        pred_bin = pred.squeeze()
        img_np = image.squeeze()
        mask_np = mask.squeeze()

        # Dice
        intersection = (pred_bin * mask_np).sum()
        union = pred_bin.sum() + mask_np.sum()
        dice_val = (2 * intersection + 1e-6) / (union + 1e-6)

        # Region Dice
        region_scores = region_dice(pred_bin, mask_np)
        region_dice_dict[idx] = region_scores

        results.append((idx, dice_val, pred_bin, img_np, mask_np, uncertainty))

        # -----------------------------
        # SAVE FULL OUTPUTS (ALL 2890)
        # -----------------------------
        show_overlay(img_np, pred_bin,
                     os.path.join(output_dir, "full_overlay", f"overlay_{idx}.png"))

        generate_polar_map(pred_bin,
                           os.path.join(output_dir, "full_polarmap", f"polarmap_{idx}.png"))

        # GradCAM
        heatmap = gradcam(model, image)
        plt.figure()
        plt.imshow(img_np, cmap='gray')
        plt.imshow(heatmap, cmap='jet', alpha=0.5)
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, "full_gradcam", f"gradcam_{idx}.png"))
        plt.close()

        if save_uncertainty and uncertainty is not None:
            np.save(os.path.join(output_dir, "full_uncertainty", f"uncertainty_{idx}.npy"),
                    uncertainty.squeeze())

    # -----------------------------
    # SELECT 10 SLICES
    # -----------------------------
    results.sort(key=lambda x: x[1])

    worst = results[:2]
    best = results[-2:]

    failures = [r for r in results if r[1] < 0.5][:3]
    normals = results[len(results)//2: len(results)//2 + 3]

    selected = best + worst + failures + normals
    selected = selected[:10]

    # -----------------------------
    # SAVE SELECTED (UI)
    # -----------------------------
    for i, (idx, dice_val, pred_bin, img_np, mask_np, uncertainty) in enumerate(selected):

        # Prediction
        plt.figure(figsize=(10,4))
        plt.subplot(1,3,1); plt.imshow(img_np, cmap='gray')
        plt.subplot(1,3,2); plt.imshow(mask_np, cmap='gray')
        plt.subplot(1,3,3); plt.imshow(pred_bin, cmap='gray')
        plt.title(f"Dice={dice_val:.3f}")
        plt.savefig(os.path.join(output_dir, "predictions", f"pred_{i}.png"))
        plt.close()

        # Overlay
        show_overlay(img_np, pred_bin,
                     os.path.join(output_dir, "overlays", f"overlay_{i}.png"))

        # Polar
        generate_polar_map(pred_bin,
                           os.path.join(output_dir, "polar", f"pred_{i}.png"))

        # Confidence
        if uncertainty is not None:
            conf = 1 - uncertainty.squeeze()
            plt.imshow(conf, cmap='jet')
            plt.savefig(os.path.join(output_dir, "confidence", f"pred_{i}.png"))
            np.save(os.path.join(output_dir, "confidence", f"conf_{i}.npy"), conf)
            plt.close()

        # GradCAM
        heatmap = gradcam(model, image)
        plt.imshow(img_np, cmap='gray')
        plt.imshow(heatmap, cmap='jet', alpha=0.5)
        plt.savefig(os.path.join(output_dir, "gradcam", f"gradcam_{i}.png"))
        plt.close()

    # Save region dice
    np.save(os.path.join(output_dir, "region_dice.npy"), region_dice_dict)

    print("✅ Full + Selected outputs generated")