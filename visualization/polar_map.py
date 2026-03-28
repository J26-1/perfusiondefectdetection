#polar_map.py
import numpy as np
import matplotlib.pyplot as plt
import cv2


def _normalize_image(x):
    x = x.astype(np.float32)
    x = x - np.min(x)
    if np.max(x) > 0:
        x = x / np.max(x)
    return x


def _mask_centroid(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        h, w = mask.shape
        return w / 2.0, h / 2.0
    return float(xs.mean()), float(ys.mean())


def _largest_component(mask):
    mask = mask.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return (labels == largest).astype(np.uint8)


def generate_polar_map(image, mask, save_path=None, num_angles=180, num_radii=96):
    image = np.asarray(image).squeeze().astype(np.float32)
    mask = (np.asarray(mask).squeeze() > 0.5).astype(np.uint8)

    if image.ndim != 2 or mask.ndim != 2:
        raise ValueError("Polar map expects 2D image and 2D mask.")

    perf = _normalize_image(image)

    if mask.sum() == 0:
        polar_display = np.full((num_radii, num_angles), np.nan, dtype=np.float32)
    else:
        # Keep only main myocardium component
        mask = _largest_component(mask)

        # Slight smoothing to reduce speckle
        perf_smooth = cv2.GaussianBlur(perf, (5, 5), 0)

        cx, cy = _mask_centroid(mask)

        ys, xs = np.where(mask > 0)
        distances = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)

        # Robust radius: use percentile instead of max to avoid over-stretching
        max_r = float(np.percentile(distances, 98))
        max_r = max(max_r, 10.0)

        polar_img = cv2.warpPolar(
            perf_smooth,
            (num_angles, num_radii),
            (cx, cy),
            max_r,
            cv2.WARP_POLAR_LINEAR
        )

        polar_mask = cv2.warpPolar(
            mask.astype(np.float32),
            (num_angles, num_radii),
            (cx, cy),
            max_r,
            cv2.WARP_POLAR_LINEAR
        )
        polar_mask = (polar_mask > 0.2).astype(np.uint8)

        # Morphological cleanup in polar space
        kernel = np.ones((3, 3), np.uint8)
        polar_mask = cv2.morphologyEx(polar_mask, cv2.MORPH_CLOSE, kernel)

        row_support = polar_mask.mean(axis=1)
        col_support = polar_mask.mean(axis=0)

        valid_rows = row_support > 0.05
        valid_cols = col_support > 0.03

        polar_display = np.full_like(polar_img, np.nan, dtype=np.float32)
        polar_display[polar_mask > 0] = polar_img[polar_mask > 0]

        for r in range(num_radii):
            if not valid_rows[r]:
                polar_display[r, :] = np.nan

        for c in range(num_angles):
            if not valid_cols[c]:
                polar_display[:, c] = np.nan

        # Optional light smoothing only where valid
        valid = ~np.isnan(polar_display)
        filled = np.where(valid, polar_display, 0)
        filled = cv2.GaussianBlur(filled, (3, 3), 0)
        polar_display[valid] = filled[valid]

    plt.figure(figsize=(6, 5))
    cmap = plt.cm.turbo.copy()
    cmap.set_bad(color=(0.92, 0.92, 0.92, 1.0))
    plt.imshow(polar_display, cmap=cmap, aspect="auto", origin="lower", vmin=0, vmax=1)
    plt.title("Myocardial Perfusion Polar Map")
    plt.xlabel("Angle")
    plt.ylabel("Radius")
    plt.colorbar(label="Normalized Perfusion")

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=180)
        plt.close()
    else:
        plt.show()

    return polar_display