#defect_map.py
import numpy as np
import matplotlib.pyplot as plt
import cv2


def _largest_component(mask):
    mask = mask.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return (labels == largest).astype(np.uint8)


def generate_defect_map(image, pred_mask, save_path=None, defect_threshold=0.55):
    image = np.asarray(image).squeeze().astype(np.float32)
    pred_mask = (np.asarray(pred_mask).squeeze() > 0.5).astype(np.uint8)

    if image.ndim != 2 or pred_mask.ndim != 2:
        raise ValueError("Defect map expects 2D image and 2D mask.")

    pred_mask = _largest_component(pred_mask)

    if pred_mask.sum() == 0:
        defect_score = np.zeros_like(image, dtype=np.float32)
    else:
        # Normalize first for stable perfusion scaling
        norm_img = image - image.min()
        norm_img = norm_img / (norm_img.max() + 1e-6)

        smooth = cv2.GaussianBlur(norm_img, (5, 5), 0)

        roi_vals = smooth[pred_mask > 0]
        ref = np.percentile(roi_vals, 95)
        ref = max(ref, 1e-6)

        norm_perf = np.clip(smooth / ref, 0, 1)

        defect_score = np.zeros_like(norm_perf, dtype=np.float32)
        defect_score[pred_mask > 0] = 1.0 - norm_perf[pred_mask > 0]

        # Sharpen strong defects and suppress mild variation
        defect_score = np.power(defect_score, 1.5)

        # Threshold inside ROI only
        defect_binary = (defect_score > defect_threshold).astype(np.uint8)

        kernel = np.ones((3, 3), np.uint8)
        defect_binary = cv2.morphologyEx(defect_binary, cv2.MORPH_OPEN, kernel)
        defect_binary = cv2.morphologyEx(defect_binary, cv2.MORPH_CLOSE, kernel)

        # Remove tiny isolated artifacts
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(defect_binary, connectivity=8)
        cleaned = np.zeros_like(defect_binary)
        for label in range(1, num_labels):
            area = stats[label, cv2.CC_STAT_AREA]
            if area >= 8:
                cleaned[labels == label] = 1

        defect_score = defect_score * cleaned.astype(np.float32)

    plt.figure(figsize=(5, 5))
    plt.imshow(image, cmap="gray")
    cmap = plt.cm.inferno.copy()
    masked = np.ma.masked_where(defect_score <= 0, defect_score)
    plt.imshow(masked, cmap=cmap, alpha=0.65, vmin=0, vmax=1)
    plt.title("Perfusion Defect Detection Map")
    plt.axis("off")

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=180)
        plt.close()
    else:
        plt.show()

    return defect_score