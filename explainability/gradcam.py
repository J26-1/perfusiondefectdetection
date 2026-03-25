import torch
import numpy as np


def gradcam(model, image):

    image = torch.tensor(image).unsqueeze(0).unsqueeze(0).float()

    image.requires_grad = True

    output = model(image)

    output.backward(torch.ones_like(output))

    gradients = image.grad

    heatmap = gradients.squeeze().numpy()

    heatmap = np.maximum(heatmap, 0)

    max_val = np.max(heatmap)
    if max_val != 0:
        heatmap /= max_val

    return heatmap