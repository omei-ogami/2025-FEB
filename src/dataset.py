import os
import shutil
from PIL import Image
import cv2
import torch

class DisasterDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None, target_transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform
        self.image_pairs = []
        
        pre_dir = os.path.join(root_dir, "images", "pre")
        post_dir = os.path.join(root_dir, "images", "post")
        target_dir = os.path.join(root_dir, "targets")

        # List all pre-disaster images
        pre_files = {f.replace("_pre_disaster.png", ""): f for f in os.listdir(pre_dir) if f.endswith("_pre_disaster.png")}
        
        # Match each pre-disaster image with its corresponding post-disaster image and target mask
        for prefix, pre_filename in pre_files.items():
            pre_path = os.path.join(pre_dir, pre_filename)
            post_filename = f"{prefix}_post_disaster.png"
            target_filename = f"{prefix}_target.png"

            post_path = os.path.join(post_dir, post_filename)
            target_path = os.path.join(target_dir, target_filename)

            # Ensure that both post and target images exist
            if os.path.exists(post_path) and os.path.exists(target_path):
                self.image_pairs.append((pre_path, post_path, target_path))

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        pre_path, post_path, target_path = self.image_pairs[idx]
        
        pre_img = Image.open(pre_path).convert("RGB")
        post_img = Image.open(post_path).convert("RGB")
        target_img = Image.open(target_path).convert("L")  # Grayscale mask

        # Apply transformations to images
        if self.transform:
            pre_img = self.transform(pre_img)   # (3, H, W)
            post_img = self.transform(post_img) # (3, H, W)
            input_tensor = torch.cat([pre_img, post_img], dim=0)  # (6, H, W)

        if self.target_transform:
            target_img = self.target_transform(target_img)  # (1, H, W)

        return input_tensor, target_img  # (6, H, W), (1, H, W)
