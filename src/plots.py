import matplotlib.pyplot as plt
import numpy as np
import torch

def visualize_predictions(model, dataloader, device, exp_id, batch_index=1, num_samples=3):
    model.eval()

    # Convert to list (only once to avoid RAM issues in large datasets)
    dataloader_list = list(dataloader)
    
    # Ensure batch index is within range
    batch_index = min(batch_index, len(dataloader_list) - 1)
    
    images, masks = dataloader_list[batch_index]
    images, masks = images[:num_samples].to(device), masks[:num_samples].cpu().numpy()

    with torch.no_grad():
        preds = model(images).cpu()
        preds = torch.argmax(preds, dim=1).numpy()

    fig, axes = plt.subplots(num_samples, 2, figsize=(10, num_samples * 2))

    for i in range(num_samples):
        ax0, ax1 = axes[i]

        im0 = ax0.imshow(masks[i], cmap="tab10", vmin=0, vmax=3)
        im1 = ax1.imshow(preds[i], cmap="tab10", vmin=0, vmax=3)

        ax0.set_title("Ground Truth")
        ax1.set_title("Prediction")

        for ax in [ax0, ax1]:
            ax.axis("off")

        cbar = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        cbar.set_label("Class Index")

    plt.tight_layout()
    plt.show()

    # save the figure
    fig_name = f"runs/experiment_{exp_id}/predictions.png"
    fig.savefig(fig_name)



