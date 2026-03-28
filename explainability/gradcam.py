# explainability/gradcam.py
import torch
import torch.nn.functional as F
import numpy as np
import cv2


class GradCAM:
    def __init__(self, model, target_layer_name="down3"):
        self.model = model
        self.model.eval()
        self.device = next(model.parameters()).device
        self.target_layer_name = target_layer_name

        self.fmap = None
        self.grads = None
        self._register_hooks()

    def _register_hooks(self):
        target_layer = dict(self.model.named_modules())[self.target_layer_name]

        def forward_hook(module, inputs, output):
            self.fmap = output

        def backward_hook(module, grad_input, grad_output):
            self.grads = grad_output[0]

        target_layer.register_forward_hook(forward_hook)
        target_layer.register_full_backward_hook(backward_hook)

    def _largest_component(self, mask):
        mask = mask.astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if num_labels <= 1:
            return mask
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        return (labels == largest).astype(np.uint8)

    def __call__(self, image, pred_mask=None, threshold=0.5):
        if isinstance(image, np.ndarray):
            image = torch.tensor(image, dtype=torch.float32)

        if image.ndim == 2:
            image = image.unsqueeze(0).unsqueeze(0)
        elif image.ndim == 3:
            image = image.unsqueeze(0)

        image = image.to(self.device).float()
        image.requires_grad_(True)

        logits = self.model(image)
        probs = torch.sigmoid(logits)

        if pred_mask is None:
            pred_mask = (probs > threshold).float()
        else:
            if isinstance(pred_mask, np.ndarray):
                pred_mask = torch.tensor(pred_mask, dtype=torch.float32, device=self.device)
            if pred_mask.ndim == 2:
                pred_mask = pred_mask.unsqueeze(0).unsqueeze(0)
            elif pred_mask.ndim == 3:
                pred_mask = pred_mask.unsqueeze(0)
            pred_mask = pred_mask.to(self.device).float()

        # Use image-normalized intensity to tighten target focus
        img_norm = image - image.min()
        img_norm = img_norm / (img_norm.max() + 1e-6)

        if pred_mask.sum() > 0:
            weighted_roi = pred_mask * img_norm
            target = (logits * weighted_roi).sum() / (weighted_roi.sum() + 1e-6)
        else:
            target = probs.mean()

        self.model.zero_grad(set_to_none=True)
        target.backward(retain_graph=False)

        weights = self.grads.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.fmap).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = cam.detach().squeeze().cpu().numpy()
        h, w = image.shape[-2:]
        cam = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)

        # Smooth CAM slightly
        cam = cv2.GaussianBlur(cam, (5, 5), 0)

        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()

        roi = pred_mask.detach().squeeze().cpu().numpy().astype(np.uint8)
        if roi.sum() > 0:
            roi = self._largest_component(roi)
            kernel = np.ones((7, 7), np.uint8)
            roi = cv2.dilate(roi, kernel, iterations=1)

            cam = cam * roi

            # Keep only meaningful hotspots
            cam[cam < 0.15] = 0.0

            cam -= cam.min()
            if cam.max() > 0:
                cam /= cam.max()

        return cam.astype(np.float32)