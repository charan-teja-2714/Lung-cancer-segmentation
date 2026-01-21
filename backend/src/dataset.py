# backend/src/dataset.py

import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

class LungCancerBinaryDataset(Dataset):
    def __init__(self, data_root, split="train", image_size=256):
        """
        data_root: path to data folder
                   e.g. backend/data
        split: 'train' or 'test'
        """
        self.image_size = image_size
        self.samples = []

        ct_root = os.path.join(data_root, split, "CT")
        mask_root = os.path.join(data_root, split, "MASK")

        classes = ["ADC", "LCC", "SCC"]

        for cls in classes:
            ct_dir = os.path.join(ct_root, cls)
            if not os.path.exists(ct_dir):
                continue

            for fname in os.listdir(ct_dir):
                if fname.endswith(('.png', '.jpg', '.jpeg')):
                    self.samples.append(fname)

        # remove duplicates (same slice appears in multiple classes)
        self.samples = sorted(list(set(self.samples)))

        self.ct_root = ct_root
        self.mask_root = mask_root
        self.classes = classes

    def __len__(self):
        return len(self.samples)

    def _load_ct(self, fname):
        for cls in self.classes:
            path = os.path.join(self.ct_root, cls, fname)
            if os.path.exists(path):
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                return img
        return None

    def _load_binary_mask(self, fname):
        mask = None
        for cls in self.classes:
            path = os.path.join(self.mask_root, cls, fname)
            if os.path.exists(path):
                m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    mask = (m > 0).astype(np.uint8)
                else:
                    mask = mask | (m > 0).astype(np.uint8)

        if mask is None:
            mask = np.zeros((self.image_size, self.image_size), dtype=np.uint8)

        return mask

    def __getitem__(self, idx):
        fname = self.samples[idx]

        # Load CT
        image = self._load_ct(fname)
        if image is None:
            raise RuntimeError(f"CT image not found for {fname}")

        # Load & merge masks
        mask = self._load_binary_mask(fname)

        # Resize
        image = cv2.resize(image, (self.image_size, self.image_size))
        mask = cv2.resize(mask, (self.image_size, self.image_size),
                           interpolation=cv2.INTER_NEAREST)

        # Normalize image
        image = image.astype(np.float32) / 255.0
        mask = mask.astype(np.float32)  # Convert mask to float32

        # To tensor
        image = torch.tensor(image).unsqueeze(0)      # (1, H, W)
        mask = torch.tensor(mask).unsqueeze(0)        # (1, H, W)

        return image, mask
