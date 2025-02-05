from dataset import DisasterDataset
from torchvision import transforms
from torch.utils.data import random_split
import torch

def main():
    # Create dataset and transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  
        transforms.ToTensor(),  
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  
    ])

    target_transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.long())  # 如果是类别
    ])

    dataset = DisasterDataset(root_dir="data/train", transform=transform, target_transform=target_transform)
    print(f"Total samples: {len(dataset)}")

    # split into training data (80) and validation data (20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    generator = torch.Generator().manual_seed(42)  # 确保可复现
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

if __name__ == '__main__':
    main()