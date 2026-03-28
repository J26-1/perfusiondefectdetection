#visualization.py
import matplotlib.pyplot as plt
import numpy as np
import cv2


def _boundary(mask):
    mask = mask.astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernel).astype(bool)


def show_error_map(img, mask, pred, save_path=None, overlay_alpha=0.55):
    img = np.asarray(img).squeeze().astype(np.float32)
    mask = (np.asarray(mask).squeeze() > 0.5).astype(np.uint8)
    pred = (np.asarray(pred).squeeze() > 0.5).astype(np.uint8)

    tp = (pred == 1) & (mask == 1)
    fp = (pred == 1) & (mask == 0)
    fn = (pred == 0) & (mask == 1)

    fp_edge = _boundary(fp.astype(np.uint8))
    fn_edge = _boundary(fn.astype(np.uint8))

    rgb = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.float32)
    rgb[tp] = [0.0, 1.0, 0.0]       # TP fill: green
    rgb[fp_edge] = [1.0, 0.0, 0.0]  # FP edge: red
    rgb[fn_edge] = [0.0, 0.4, 1.0]  # FN edge: blue

    plt.figure(figsize=(5, 5))
    plt.imshow(img, cmap="gray")
    plt.imshow(rgb, alpha=overlay_alpha)
    plt.title("Error Map (Green=TP, Red=FP, Blue=FN)")
    plt.axis("off")

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=180)
        plt.close()
    else:
        plt.show()