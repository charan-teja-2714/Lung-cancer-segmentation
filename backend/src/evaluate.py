# backend/src/evaluate.py

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import LungCancerBinaryDataset
from model import UNet

# ============================================================
# CONFIG
# ============================================================

DATA_ROOT = "data/raw"
IMAGE_SIZE = 256
BATCH_SIZE = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CHECKPOINT_PATH = "checkpoints/best_model.pth"

# ============================================================
# METRICS
# ============================================================

def dice_score(pred, target, smooth=1e-6):
    pred = (pred > 0.5).float()
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    return (2 * intersection + smooth) / (pred.sum() + target.sum() + smooth)

def accuracy_score(pred, target):
    pred = (pred > 0.5).float()
    correct = (pred == target).float().sum()
    return correct / torch.numel(pred)

# ============================================================
# EVALUATION
# ============================================================

def evaluate():
    print("🔍 Evaluating model on TEST set")

    test_dataset = LungCancerBinaryDataset(
        data_root=DATA_ROOT,
        split="test",
        image_size=IMAGE_SIZE
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2
    )

    model = UNet().to(DEVICE)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    total_dice = 0.0
    total_acc = 0.0

    with torch.no_grad():
        for images, masks in tqdm(test_loader):
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            preds = model(images)

            total_dice += dice_score(preds, masks).item()
            total_acc += accuracy_score(preds, masks).item()

    avg_dice = total_dice / len(test_loader)
    avg_acc = total_acc / len(test_loader)

    print("✅ Evaluation completed")
    print(f"📊 Test Dice Score: {avg_dice:.4f}")
    print(f"📊 Test Accuracy:   {avg_acc:.4f}")

if __name__ == "__main__":
    evaluate()
