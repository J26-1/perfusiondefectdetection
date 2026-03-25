import os
import numpy as np
import torch
from models.unet import UNet
from inference.predictor import predict_batch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = UNet()
model.load_state_dict(torch.load("best_model.pth", map_location=device))

# Load test images
images = []

for file in os.listdir("test_images"):
    img = np.load(f"test_images/{file}")  # or cv2.imread
    images.append(img)

images = np.array(images)

# Batch inference
preds, probs = predict_batch(model, images, device)

# Save outputs
os.makedirs("outputs/test_preds", exist_ok=True)

for i in range(len(preds)):
    np.save(f"outputs/test_preds/pred_{i}.npy", preds[i])