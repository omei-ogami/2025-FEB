import matplotlib.pyplot as plt
import numpy as np
import torch

def visualize_predictions(model, dataloader, device, num_samples=3):
    model.eval()  # Set to evaluation mode
    images, masks = next(iter(dataloader))  # Get a batch
    images, masks = images[:num_samples].to(device), masks[:num_samples].cpu().numpy()

    with torch.no_grad():
        preds = model(images).cpu()  # Get predictions
        preds = torch.argmax(preds, dim=1).numpy()  # Convert to class labels

    fig, axes = plt.subplots(num_samples, 3, figsize=(10, num_samples * 3))

    for i in range(num_samples):
        ax0 = axes[i, 0]
        ax1 = axes[i, 1]

        im0 = ax0.imshow(masks[i], cmap="tab10", vmin=0, vmax=3)  # True mask
        im1 = ax1.imshow(preds[i], cmap="tab10", vmin=0, vmax=3)  # Predicted mask

        ax0.set_title("Ground Truth")
        ax1.set_title("Prediction")

        # Hide axis
        for ax in [ax0, ax1]:
            ax.axis("off")

        # Add colorbar
        cbar = fig.colorbar(im1, ax=axes[i, 2], fraction=0.046, pad=0.04)
        cbar.set_label("Class Index")

    plt.tight_layout()
    plt.show()
