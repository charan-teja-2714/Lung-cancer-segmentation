# backend/src/inference.py

import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from model import UNet

# ============================================================
# CONFIG
# ============================================================

IMAGE_SIZE = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = "checkpoints/best_model.pth"

# ============================================================
# INFERENCE
# ============================================================

def predict(image_path):
    model = UNet().to(DEVICE)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img_resized = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    img_norm = img_resized / 255.0

    tensor = torch.tensor(img_norm).float().unsqueeze(0).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred = model(tensor)[0, 0].cpu().numpy()

    mask = (pred > 0.5).astype(np.uint8)

    return img_resized, mask

def visualize(image, mask):
    overlay = image.copy()
    overlay[mask == 1] = 255

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.title("CT Image")
    plt.imshow(image, cmap="gray")

    plt.subplot(1, 3, 2)
    plt.title("Predicted Mask")
    plt.imshow(mask, cmap="gray")

    plt.subplot(1, 3, 3)
    plt.title("Overlay")
    plt.imshow(overlay, cmap="gray")

    plt.tight_layout()
    plt.show()

# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    img_path = "sample_ct.jpg"  # change this
    image, mask = predict(img_path)
    visualize(image, mask)
