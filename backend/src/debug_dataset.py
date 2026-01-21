from dataset import LungCancerBinaryDataset

dataset = LungCancerBinaryDataset(
    data_root="data/raw",
    split="train",
    image_size=256
)

print(f"Dataset length: {len(dataset)}")
print(f"Samples found: {len(dataset.samples)}")
print(f"CT root: {dataset.ct_root}")
print(f"Mask root: {dataset.mask_root}")

import os
print(f"CT root exists: {os.path.exists(dataset.ct_root)}")
print(f"Mask root exists: {os.path.exists(dataset.mask_root)}")

for cls in dataset.classes:
    ct_dir = os.path.join(dataset.ct_root, cls)
    print(f"{cls} CT dir exists: {os.path.exists(ct_dir)}")
    if os.path.exists(ct_dir):
        files = os.listdir(ct_dir)
        print(f"  Files in {cls}: {len(files)}")