#overlay.py
import matplotlib.pyplot as plt
import numpy as np


def show_overlay(image, prediction, mask=None, save_path=None):
    image = np.asarray(image).squeeze()
    pred = np.asarray(prediction).squeeze()

    plt.figure(figsize=(5, 5))
    plt.imshow(image, cmap="gray")

    if mask is not None:
        gt = np.asarray(mask).squeeze()
        plt.imshow(gt, alpha=0.35, cmap="Greens")

    plt.imshow(pred, alpha=0.35, cmap="Reds")
    plt.title("Overlay")
    plt.axis("off")

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=180)
        plt.close()
    else:
        plt.show()