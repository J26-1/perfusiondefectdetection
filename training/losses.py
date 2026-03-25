import torch


def dice_loss(pred, target, smooth=1):

    pred = pred.view(-1)
    target = target.view(-1)

    intersection = (pred * target).sum()

    dice = (2 * intersection + smooth) / (
        pred.sum() + target.sum() + smooth
    )

    return 1 - dice