from dataset import DisasterDataset
from Unet import UNet
from train import train, plot_loss
from torchvision import transforms
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt
import numpy as np
from plots import visualize_predictions

def main():
    # Create dataset and transforms
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.ToTensor(),  
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  
    ])

    target_transform = transforms.Compose([
        # transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.Lambda(lambda img: torch.tensor(np.array(img), dtype=torch.long)) 
    ])

    dataset = DisasterDataset(root_dir="data/train", transform=transform, target_transform=target_transform)

    # split into training data (80) and validation data (20)k,
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    generator = torch.Generator().manual_seed(42)  # 确保可复现
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False, pin_memory=True)

    # Experient Id
    exp_id = 1

    # Hyperparameters
    num_epochs = 70
    learning_rate = 1e-3

    # Define model, loss function, and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")
    class_weights = torch.tensor([0.1, 1.5, 2.0, 2.5]).to(device) 
    model = UNet(in_channels=6, out_channels=4)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    model.to(device)
    performance = train(model, optimizer, criterion, train_dataloader, val_dataloader, num_epochs, device=device, exp_id=exp_id)
    plot_loss(performance)

    # Save the model and results
    model_name = f"models/model_{exp_id}.pth"
    torch.save(model.state_dict(), model_name)
    csv_name = f"runs/experiment_{exp_id}/performance.csv"
    performance.to_csv(csv_name, index=False)

    visualize_predictions(model, val_dataloader, device="cuda")


if __name__ == '__main__':
    main()