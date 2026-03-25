import matplotlib.pyplot as plt
import numpy as np

def show_error_map(img, mask, pred, save_path=None, cmap="hot", overlay_alpha=0.6):

    error = np.abs(pred - mask)

    plt.figure(figsize=(5, 5))

    if img is not None:
        plt.imshow(img, cmap="gray")
        plt.imshow(error, cmap=cmap, alpha=overlay_alpha)
    else:
        plt.imshow(error, cmap=cmap)

    plt.title("Error Map")
    plt.axis("off")

    if save_path:
        plt.savefig(save_path)

    plt.close()