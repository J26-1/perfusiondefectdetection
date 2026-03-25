import numpy as np


def extract_slices(volume, mask):

    images = []
    masks = []

    for i in range(volume.shape[0]):

        img_slice = volume[i]
        mask_slice = mask[i]

        images.append(img_slice)
        masks.append(mask_slice)

    return np.array(images), np.array(masks)