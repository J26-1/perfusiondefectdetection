import matplotlib.pyplot as plt
import numpy as np

def show_overlay(image, prediction, save_path=None):
    if hasattr(image, "cpu"):
        image = image.cpu().numpy()

    if isinstance(prediction, np.ndarray):
        pred = prediction.squeeze()
    else:
        pred = prediction

    if image.ndim == 3:
        image = image.squeeze()

    plt.figure(figsize=(5, 5))
    plt.imshow(image, cmap='gray')
    plt.imshow(pred, alpha=0.4, cmap='Reds')
    plt.axis('off')

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()