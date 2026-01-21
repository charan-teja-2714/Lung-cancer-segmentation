# backend/src/train.py

import os
import torch
import torch.nn as nn
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
EPOCHS = 40
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

BEST_CKPT = os.path.join(CHECKPOINT_DIR, "best_model.pth")
LAST_CKPT = os.path.join(CHECKPOINT_DIR, "last_model.pth")

# 🔴 EARLY STOPPING CONFIG
PATIENCE = 6
MIN_DELTA = 1e-4

# ============================================================
# METRICS & LOSSES
# ============================================================

def dice_loss(pred, target, smooth=1e-6):
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    return 1 - (2 * intersection + smooth) / (pred.sum() + target.sum() + smooth)

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
# TRAINING LOOP
# ============================================================

def train():
    print(f"🚀 Training started on device: {DEVICE}")

    train_dataset = LungCancerBinaryDataset(
        data_root=DATA_ROOT,
        split="train",
        image_size=IMAGE_SIZE
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    model = UNet().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    bce_loss = nn.BCELoss()

    best_dice = 0.0
    start_epoch = 1
    epochs_without_improve = 0

    # ========================================================
    # 🔄 RESUME TRAINING
    # ========================================================

    if os.path.exists(LAST_CKPT):
        print("🔄 Resuming training from last checkpoint")
        checkpoint = torch.load(LAST_CKPT, map_location=DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"➡️ Resuming from epoch {start_epoch}")

    # ========================================================
    # TRAIN
    # ========================================================

    for epoch in range(start_epoch, EPOCHS + 1):
        model.train()

        epoch_loss = 0.0
        epoch_dice = 0.0
        epoch_acc = 0.0

        loop = tqdm(train_loader, desc=f"Epoch [{epoch}/{EPOCHS}]")

        for images, masks in loop:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            preds = model(images)

            loss = bce_loss(preds, masks) + dice_loss(preds, masks)
            dice = dice_score(preds, masks)
            acc = accuracy_score(preds, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_dice += dice.item()
            epoch_acc += acc.item()

            loop.set_postfix(
                loss=f"{loss.item():.4f}",
                dice=f"{dice.item():.4f}",
                acc=f"{acc.item():.4f}"
            )

        epoch_loss /= len(train_loader)
        epoch_dice /= len(train_loader)
        epoch_acc /= len(train_loader)

        print(
            f"Epoch {epoch:02d} | "
            f"Loss: {epoch_loss:.4f} | "
            f"Dice: {epoch_dice:.4f} | "
            f"Accuracy: {epoch_acc:.4f}"
        )

        # ====================================================
        # 🏆 BEST MODEL + EARLY STOPPING
        # ====================================================

        if epoch_dice > best_dice + MIN_DELTA:
            best_dice = epoch_dice
            epochs_without_improve = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_dice": best_dice,
                },
                BEST_CKPT
            )
            print(f"🏆 Best model saved (Dice = {best_dice:.4f})")
        else:
            epochs_without_improve += 1
            print(f"⏳ No improvement for {epochs_without_improve} epoch(s)")

        # Save LAST model
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            LAST_CKPT
        )

        # ⏹️ EARLY STOP
        if epochs_without_improve >= PATIENCE:
            print(
                f"⏹️ Early stopping triggered "
                f"(no Dice improvement for {PATIENCE} epochs)"
            )
            break

    print("🎉 Training finished")
    print(f"🏆 Best Dice achieved: {best_dice:.4f}")
    print(f"📁 Checkpoints saved in: {CHECKPOINT_DIR}")

# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    train()
