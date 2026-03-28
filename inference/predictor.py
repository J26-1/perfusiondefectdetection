#predictor.py
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import cv2

from visualization.overlay import show_overlay
from visualization.polar_map import generate_polar_map
from visualization.defect_map import generate_defect_map
from utils.visualization import show_error_map
from utils.metrics import region_dice, dice_score_np
from explainability.gradcam import GradCAM


def preprocess_image(image):
    if isinstance(image, np.ndarray):
        image = torch.from_numpy(image)

    image = image.float()

    if image.ndim == 2:
        image = image.unsqueeze(0).unsqueeze(0)
    elif image.ndim == 3:
        image = image.unsqueeze(0)
    elif image.ndim == 4:
        pass
    else:
        raise ValueError(f"Unsupported image shape: {tuple(image.shape)}")

    if image.shape[1] != 1:
        image = image.mean(dim=1, keepdim=True)

    return image


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


def predict_batch(model, images, device=None):
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    preds = []
    probs = []

    for img in images:
        pred, prob = predict(model, img)
        preds.append(pred.squeeze())
        probs.append(prob.squeeze())

    return np.asarray(preds), np.asarray(probs)


def enable_dropout(model):
    for m in model.modules():
        if isinstance(m, nn.Dropout) or isinstance(m, nn.Dropout2d):
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
            pred = torch.sigmoid(model(image))
            preds.append(pred)

    preds = torch.stack(preds, dim=0)  # [T,B,C,H,W]
    mean_pred = preds.mean(dim=0)
    uncertainty = preds.var(dim=0)

    return mean_pred.cpu().numpy(), uncertainty.cpu().numpy()


def _normalize(x):
    x = x.astype(np.float32)
    x = x - np.min(x)
    if np.max(x) > 0:
        x = x / np.max(x)
    return x


def _roi_mask(pred_mask, dilation=5):
    pred_mask = (np.asarray(pred_mask).squeeze() > 0.5).astype(np.uint8)
    if pred_mask.sum() == 0:
        return pred_mask.astype(np.float32)
    kernel = np.ones((dilation, dilation), np.uint8)
    roi = cv2.dilate(pred_mask, kernel, iterations=1)
    return roi.astype(np.float32)


def compute_confidence_map(mean_pred, uncertainty, pred_mask):
    mean_pred = np.asarray(mean_pred).squeeze().astype(np.float32)
    uncertainty = np.asarray(uncertainty).squeeze().astype(np.float32)
    pred_mask = (np.asarray(pred_mask).squeeze() > 0.5).astype(np.uint8)

    p = np.clip(mean_pred, 1e-6, 1 - 1e-6)
    entropy = -(p * np.log(p) + (1 - p) * np.log(1 - p)) / np.log(2.0)  # 0..1
    var_norm = uncertainty / (uncertainty.max() + 1e-6)

    combined_unc = 0.6 * var_norm + 0.4 * entropy
    combined_unc = np.clip(combined_unc, 0, 1)

    grad_y, grad_x = np.gradient(mean_pred)
    boundary = np.sqrt(grad_x ** 2 + grad_y ** 2)
    boundary = boundary / (boundary.max() + 1e-6)

    conf = 1.0 - combined_unc
    conf = conf * (1.0 - 0.6 * boundary)

    roi = _roi_mask(pred_mask, dilation=7)
    conf = conf * roi
    conf = np.clip(conf, 0, 1)

    return conf.astype(np.float32), roi.astype(np.float32)


def save_prediction_panel(image, mask, pred, dice_val, save_path):
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(image, cmap="gray")
    plt.title("Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(mask, cmap="gray")
    plt.title("Ground Truth")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(pred, cmap="gray")
    plt.title(f"Prediction (Dice={dice_val:.3f})")
    plt.axis("off")

    plt.savefig(save_path, bbox_inches="tight", dpi=180)
    plt.close()


def save_confidence_map(image, confidence_map, roi_mask, save_path):
    plt.figure(figsize=(5, 5))
    plt.imshow(image, cmap="gray")

    cmap = plt.cm.jet.copy()
    masked_conf = np.ma.masked_where(roi_mask <= 0, confidence_map)
    plt.imshow(masked_conf, cmap=cmap, alpha=0.65, vmin=0, vmax=1)

    plt.title("Confidence Map")
    plt.axis("off")
    plt.savefig(save_path, bbox_inches="tight", dpi=180)
    plt.close()


def save_gradcam_overlay(image, heatmap, save_path):
    plt.figure(figsize=(5, 5))
    plt.imshow(image, cmap="gray")

    masked = np.ma.masked_where(heatmap <= 0.05, heatmap)
    plt.imshow(masked, cmap="jet", alpha=0.5, vmin=0, vmax=1)

    plt.title("Grad-CAM")
    plt.axis("off")
    plt.savefig(save_path, bbox_inches="tight", dpi=180)
    plt.close()


def _is_clinically_meaningful(item):
    img = item["img"]
    pred = item["pred"]
    mask = item["mask"]
    h, w = pred.shape
    area = h * w

    pred_ratio = pred.sum() / area
    mask_ratio = mask.sum() / area
    roi_ratio = max(pred_ratio, mask_ratio)

    # reject nearly empty / huge / flat slices
    if roi_ratio < 0.003:
        return False
    if roi_ratio > 0.20:
        return False
    if img.std() < 0.08:
        return False

    return True


def _select_ui_slices(full_results, target_n=10):
    meaningful = [r for r in full_results if _is_clinically_meaningful(r)]

    if len(meaningful) < target_n:
        # fallback: keep largest non-empty regions
        meaningful = sorted(
            full_results,
            key=lambda x: (x["pred"].sum() + x["mask"].sum(), x["dice"]),
            reverse=True
        )[:max(target_n, 12)]

    meaningful = sorted(meaningful, key=lambda x: x["dice"])

    # Buckets
    best = meaningful[-2:] if len(meaningful) >= 2 else meaningful
    caution = [r for r in meaningful if 0.35 <= r["dice"] < 0.75][:3]
    failures = [r for r in meaningful if r["dice"] < 0.35][:2]

    mid = len(meaningful) // 2
    normals = meaningful[max(0, mid - 3): min(len(meaningful), mid + 3)]

    selected = best + caution + failures + normals

    seen = set()
    selected_unique = []
    for item in selected:
        if item["full_idx"] not in seen:
            selected_unique.append(item)
            seen.add(item["full_idx"])

    if len(selected_unique) < target_n:
        for item in reversed(meaningful):
            if item["full_idx"] not in seen:
                selected_unique.append(item)
                seen.add(item["full_idx"])
            if len(selected_unique) >= target_n:
                break

    return selected_unique[:target_n]


def predict_dataset(model, dataset, output_dir="outputs", mc_samples=10, save_uncertainty=True, full_dataset_outputs=True, selected_ui_slices=10, max_samples=None):
    full_dirs = [
        "full_overlay",
        "full_error",
        "full_gradcam",
        "full_uncertainty",
        "full_confidence",
        "full_polarmap",
        "full_defect_map",
    ]
    selected_dirs = [
        "predictions",
        "overlays",
        "errors",
        "confidence",
        "uncertainty",
        "polar",
        "gradcam",
        "defect_map",
    ]

    for d in full_dirs + selected_dirs:
        os.makedirs(os.path.join(output_dir, d), exist_ok=True)

    device = next(model.parameters()).device
    gradcam = GradCAM(model, target_layer_name="down3")

    full_results = []
    full_region_dice_dict = {}

    total_items = len(dataset) if max_samples is None else min(len(dataset), max_samples)
    for idx in range(total_items):
        image, mask = dataset[idx]

        if mc_samples > 0:
            mean_pred, uncertainty = predict_with_uncertainty(model, image, n_samples=mc_samples)
            pred_bin = (mean_pred > 0.5).astype(np.float32).squeeze()
            mean_pred = mean_pred.squeeze()
            uncertainty = uncertainty.squeeze()
        else:
            pred_bin, probs = predict(model, image)
            pred_bin = pred_bin.squeeze().astype(np.float32)
            mean_pred = probs.squeeze()
            uncertainty = np.zeros_like(mean_pred, dtype=np.float32)

        img_np = image.squeeze().cpu().numpy()
        mask_np = mask.squeeze().cpu().numpy()

        confidence_map, roi_mask = compute_confidence_map(mean_pred, uncertainty, pred_bin)

        dice_val = dice_score_np(pred_bin, mask_np)
        region_scores = region_dice(pred_bin, mask_np)

        full_region_dice_dict[idx] = region_scores
        item = {
            "full_idx": idx,
            "dice": float(dice_val),
            "pred": pred_bin.astype(np.float32),
            "img": img_np.astype(np.float32),
            "mask": mask_np.astype(np.float32),
            "mean_pred": mean_pred.astype(np.float32),
            "uncertainty": uncertainty.astype(np.float32),
            "confidence": confidence_map.astype(np.float32),
            "roi_mask": roi_mask.astype(np.float32),
        }
        full_results.append(item)

        if full_dataset_outputs:
            show_overlay(
                img_np,
                pred_bin,
                mask=mask_np,
                save_path=os.path.join(output_dir, "full_overlay", f"overlay_{idx}.png")
            )

            show_error_map(
                img_np,
                mask_np,
                pred_bin,
                save_path=os.path.join(output_dir, "full_error", f"error_{idx}.png")
            )

            if _is_clinically_meaningful(item):
                generate_polar_map(
                    img_np,
                    pred_bin,
                    save_path=os.path.join(output_dir, "full_polarmap", f"polarmap_{idx}.png")
                )

                generate_defect_map(
                    img_np,
                    pred_bin,
                    save_path=os.path.join(output_dir, "full_defect_map", f"defect_{idx}.png")
                )

                image_tensor = preprocess_image(image).to(device)
                heatmap = gradcam(image_tensor, pred_mask=pred_bin)
                save_gradcam_overlay(
                    img_np,
                    heatmap,
                    os.path.join(output_dir, "full_gradcam", f"gradcam_{idx}.png")
                )

                save_confidence_map(
                    img_np,
                    confidence_map,
                    roi_mask,
                    os.path.join(output_dir, "full_confidence", f"confidence_{idx}.png")
                )

            if save_uncertainty:
                np.save(
                    os.path.join(output_dir, "full_uncertainty", f"uncertainty_{idx}.npy"),
                    uncertainty.astype(np.float32)
                )

    # -----------------------------
    # Select 10 clinically meaningful slices for UI
    # -----------------------------
    selected_unique = _select_ui_slices(full_results, target_n=selected_ui_slices)

    selected_region_dice = {}
    selected_meta = {}

    for i, item in enumerate(selected_unique):
        idx = item["full_idx"]
        dice_val = item["dice"]
        pred_bin = item["pred"]
        img_np = item["img"]
        mask_np = item["mask"]
        uncertainty = item["uncertainty"]
        confidence_map = item["confidence"]
        roi_mask = item["roi_mask"]

        selected_region_dice[i] = region_dice(pred_bin, mask_np)
        selected_meta[i] = {
            "full_idx": idx,
            "dice": float(dice_val),
            "meaningful": True
        }

        save_prediction_panel(
            img_np,
            mask_np,
            pred_bin,
            dice_val,
            os.path.join(output_dir, "predictions", f"pred_{i}.png")
        )

        show_overlay(
            img_np,
            pred_bin,
            mask=mask_np,
            save_path=os.path.join(output_dir, "overlays", f"overlay_{i}.png")
        )

        show_error_map(
            img_np,
            mask_np,
            pred_bin,
            save_path=os.path.join(output_dir, "errors", f"pred_{i}.png")
        )

        save_confidence_map(
            img_np,
            confidence_map,
            roi_mask,
            os.path.join(output_dir, "confidence", f"pred_{i}.png")
        )
        np.save(
            os.path.join(output_dir, "confidence", f"conf_{i}.npy"),
            confidence_map.astype(np.float32)
        )

        if uncertainty is not None:
            np.save(
                os.path.join(output_dir, "uncertainty", f"unc_{i}.npy"),
                uncertainty.astype(np.float32)
            )

        generate_polar_map(
            img_np,
            pred_bin,
            save_path=os.path.join(output_dir, "polar", f"pred_{i}.png")
        )

        generate_defect_map(
            img_np,
            pred_bin,
            save_path=os.path.join(output_dir, "defect_map", f"defect_{i}.png")
        )

        image_tensor = preprocess_image(torch.from_numpy(img_np))
        image_tensor = image_tensor.to(device)
        heatmap = gradcam(image_tensor, pred_mask=pred_bin)
        save_gradcam_overlay(
            img_np,
            heatmap,
            os.path.join(output_dir, "gradcam", f"gradcam_{i}.png")
        )

    np.save(os.path.join(output_dir, "region_dice.npy"), selected_region_dice)
    np.save(os.path.join(output_dir, "full_region_dice.npy"), full_region_dice_dict)
    np.save(os.path.join(output_dir, "selected_meta.npy"), selected_meta)

    print("✅ Full + Selected outputs generated")