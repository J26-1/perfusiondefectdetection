import numpy as np

def is_failure(dice, threshold=0.6):
    return dice < threshold

def dice_score(pred, gt):

    intersection = np.sum(pred * gt)

    return (2 * intersection) / (
        np.sum(pred) + np.sum(gt) + 1e-8
    )

def region_dice(pred, mask):

    h = pred.shape[0]

    regions = {
        "top": (slice(0, h//3), slice(None)),
        "middle": (slice(h//3, 2*h//3), slice(None)),
        "bottom": (slice(2*h//3, h), slice(None))
    }

    scores = {}

    for name, (r,c) in regions.items():

        p = pred[r,c]
        m = mask[r,c]

        intersection = (p*m).sum()
        union = p.sum() + m.sum()

        dice = (2*intersection + 1e-6)/(union + 1e-6)

        scores[name] = dice

    return scores