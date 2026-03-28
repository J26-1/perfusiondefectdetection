#metrics.py
import numpy as np


def dice_score_np(pred, gt, eps=1e-6):
    pred = pred.astype(np.float32)
    gt = gt.astype(np.float32)

    intersection = np.sum(pred * gt)
    union = np.sum(pred) + np.sum(gt)

    return (2.0 * intersection + eps) / (union + eps)


def is_failure(dice, threshold=0.60):
    return dice < threshold


def region_dice(pred, mask, eps=1e-6):
    pred = pred.astype(np.float32)
    mask = mask.astype(np.float32)

    h = pred.shape[0]

    regions = {
        "top": (slice(0, h // 3), slice(None)),
        "middle": (slice(h // 3, 2 * h // 3), slice(None)),
        "bottom": (slice(2 * h // 3, h), slice(None)),
    }

    scores = {}

    for name, (r, c) in regions.items():
        p = pred[r, c]
        m = mask[r, c]

        intersection = (p * m).sum()
        union = p.sum() + m.sum()

        scores[name] = float((2 * intersection + eps) / (union + eps))

    return scores


def region_dice_clinical(pred, mask):
    raw = region_dice(pred, mask)
    return {
        "Anterior": raw["top"],
        "Mid": raw["middle"],
        "Inferior": raw["bottom"],
    }