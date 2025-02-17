from dataset import DisasterDataset
from Unet import UNet
from train import train
from torchvision import transforms
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from focal_losss import FocalLoss
import torch
import numpy as np
from plots import visualize_predictions
import torchvision.transforms.v2 as T
import albumentations as A
from albumentations.pytorch import ToTensorV2

def main():
    # Create dataset and transforms
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.Resize(512, 512),
    ], is_check_shapes=False)

    dataset = DisasterDataset(root_dir="data/train", transform=transform)

    # split into training data (80) and validation data (20)k,
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    generator = torch.Generator().manual_seed(42)  # 确保可复现
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False, pin_memory=True)

    # Experient Id
    exp_id = 11

    # Hyperparameters
    num_epochs = 100
    learning_rate = 6e-5

    # Define model, loss function, and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")
    model = UNet(in_channels=6, out_channels=4)
    alpha = torch.tensor([0.1, 0.3, 0.3, 0.3]).to(device)
    criterion = FocalLoss(gamma=2.0, alpha=alpha)
    # criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate) 


    # Train the model
    model.to(device)
    performance = train(model, optimizer, criterion, train_dataloader, val_dataloader, num_epochs, device=device, exp_id=exp_id)

    # Save the model and results
    model_name = f"models/model_{exp_id}.pth"
    torch.save(model.state_dict(), model_name)
    csv_name = f"runs/experiment_{exp_id}/performance.csv"
    performance.to_csv(csv_name, index=False)

    visualize_predictions(model, val_dataloader, device="cuda", exp_id=exp_id)


if __name__ == '__main__':
    main()