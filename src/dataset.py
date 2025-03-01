import os
import shutil
from PIL import Image
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as T
import albumentations as A

class DisasterDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_pairs = []
        self.mean = None
        self.std = None
        
        pre_dir = os.path.join(root_dir, "images", "pre")
        post_dir = os.path.join(root_dir, "images", "post")
        pre_target_dir = os.path.join(root_dir, "targets", "pre")
        post_target_dir = os.path.join(root_dir, "targets", "post")

        # List all pre-disaster images
        pre_files = {f.replace("_pre_disaster.png", ""): f for f in os.listdir(pre_dir) if f.endswith("_pre_disaster.png")}
        
        # Match each pre-disaster image with its corresponding post-disaster image and target mask
        for prefix, pre_filename in pre_files.items():
            pre_path = os.path.join(pre_dir, pre_filename)
            post_filename = f"{prefix}_post_disaster.png"
            pre_target_filename = f"{prefix}_pre_disaster_target.png"
            post_target_filename = f"{prefix}_post_disaster_target.png"

            post_path = os.path.join(post_dir, post_filename)
            pre_target_path = os.path.join(pre_target_dir, pre_target_filename)
            post_target_path = os.path.join(post_target_dir, post_target_filename)

            # Ensure that both post and target images exist
            if os.path.exists(post_path) and os.path.exists(pre_target_path) and os.path.exists(post_target_path) and self.checkTarget(pre_target_path, post_target_path):
                self.image_pairs.append((pre_path, post_path, pre_target_path, post_target_path))

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        pre_path, post_path, pre_target_path, post_target_path = self.image_pairs[idx]

        pre_img = np.array(Image.open(pre_path).convert("RGB"))
        post_img = np.array(Image.open(post_path).convert("RGB"))
        pre_target = np.array(Image.open(pre_target_path).convert("P"))
        post_target = np.array(Image.open(post_target_path).convert("P"))

        if self.transform:
            transformed_pre = self.transform(image=pre_img, mask=pre_target)
            pre_img = transformed_pre['image']
            pre_target = transformed_pre['mask']
            replay = transformed_pre['replay']

            transformed_post = A.ReplayCompose.replay(replay, image=post_img, mask=post_target)
            post_img = transformed_post['image']
            post_target = transformed_post['mask']

        # Normalize the images and convert to tensor
        pre_img = T.ToTensor()(pre_img)
        post_img = T.ToTensor()(post_img)

        input_tensor = torch.cat([pre_img, post_img], dim=0)  # (6, H, W)

        if self.mean != None and self.std != None:
            input_tensor = T.Normalize(mean=self.mean, std=self.std)(input_tensor)
            #self.visualize(input_tensor[:3, :, :], input_tensor[3:, :, :], pre_target, post_target)

        target_img = post_target - pre_target

        # Clip the NumPy array FIRST
        target_img = np.maximum(target_img, 0)  # Or your preferred clipping method
        
        # Ensure that pixels are within the range [0, 3], others to 0
        target_img[target_img > 3] = 0

        # THEN convert to tensor
        target_img = torch.from_numpy(target_img).long()

        

        return input_tensor, target_img  # (6, H, W), (H, W)


    def visualize(self, pre_img, post_img, pre_target, post_target):
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))

        axes[0, 0].imshow(pre_img.permute(1, 2, 0).cpu().numpy())
        axes[0, 0].set_title("Pre-Disaster Image")
        axes[0, 0].axis("off")

        axes[0, 1].imshow(post_img.permute(1, 2, 0).cpu().numpy())
        axes[0, 1].set_title("Post-Disaster Image")
        axes[0, 1].axis("off")

        axes[1, 0].imshow(pre_target, cmap="tab10")
        axes[1, 0].set_title("Pre-Disaster Target")
        axes[1, 0].axis("off")

        axes[1, 1].imshow(post_target, cmap="tab10")
        axes[1, 1].set_title("Post-Disaster Target")
        axes[1, 1].axis("off")

        plt.tight_layout()
        plt.show()

    def setStat(self, mean, std):
        self.mean = mean
        self.std = std

    def checkTarget(self, pre_target_path, post_target_path):
        # Ensure that pixels are within the range [0, 3], others to 0
        pre_target = np.array(Image.open(pre_target_path).convert("P"))
        post_target = np.array(Image.open(post_target_path).convert("P"))

        pre_target = np.maximum(pre_target, 0)
        pre_target[pre_target > 3] = 0

        post_target = np.maximum(post_target, 0)
        post_target[post_target > 3] = 0

        return np.sum(pre_target) >= 1 and np.sum(post_target) >= 1