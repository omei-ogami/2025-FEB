import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from metric import mean_pixel_accuracy, IoU, dice_coeff, per_class_accuracy

def train(model, optimizer, criterion, train_loader, val_loader, num_epochs, device, exp_id):
    folder = f"runs/experiment_{exp_id}"
    writer = SummaryWriter(log_dir=folder)

    # Initialize a list to collect performance metrics
    performance_data = []

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.8, patience=3
    )
    
    train_loss = []
    eval_loss = []
    
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        total_loss = 0

        for inputs, targets in tqdm.tqdm(train_loader, unit="batch", desc=f"Epoch {epoch+1}/{num_epochs}"):
            optimizer.zero_grad()
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)

            # Choose correct loss function
            targets = targets.squeeze(1)  # (B, 1, H, W) → (B, H, W)
            loss = criterion(outputs, targets) 

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        scheduler.step(avg_loss)
        train_loss.append(avg_loss)

        # Evaluate the model correctly
        val_performance = evaluate(model, val_loader, criterion, device)
        eval_loss.append(val_performance["Val Loss"])

        # Write to tensorboard
        writer.add_scalar('Training Loss', avg_loss, epoch)
        for name, param in model.named_parameters():
            writer.add_histogram(name, param, epoch)
        writer.add_scalar('Validation Loss', val_performance["Val Loss"], epoch)

        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_loss:.4f}, Val Loss: {val_performance['Val Loss']:.4f}")

        # Record performance in a list of dictionaries
        performance_data.append({
            "Epoch": epoch + 1,  # Use epoch + 1 for human-friendly numbering
            "Train Loss": avg_loss,
            "Val Loss": val_performance["Val Loss"],
            "Mean Pixel Accuracy": val_performance["Mean Pixel Accuracy"],
            "IoU": val_performance["IoU"],
            "Dice Score": val_performance["Dice Score"],
            "Per-class Accuracy": val_performance["Per-class Accuracy"]
        })

    writer.close()

    # Convert the list of dictionaries to a DataFrame
    performance_df = pd.DataFrame(performance_data)

    return performance_df

def evaluate(model, val_loader, criterion, device, num_classes=4):
    model.eval()  # Set model to evaluation mode

    total_loss = 0
    total_acc = 0
    total_iou = [0] * num_classes
    total_dice = [0] * num_classes
    total_class_accuracy = [0] * num_classes

    with torch.no_grad():
        for inputs, targets in tqdm.tqdm(val_loader, unit="batch", desc="Validating"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            
            # If using CrossEntropyLoss, no need to apply argmax for loss calculation
            targets = targets.squeeze(1)  # (B, 1, H, W) → (B, H, W)
            loss = criterion(outputs, targets)  # Criterion should work directly with raw logits

            # If you need predictions for evaluation or visualization
            predicted_class = torch.argmax(outputs, dim=1)  # Get predicted class (logits to class)

            # print(f"Predicted Classes: {predicted_class.unique()}")
            # print(f"Target Classes: {targets.unique()}")

            # Accuracy (Mean Pixel Accuracy)
            acc = mean_pixel_accuracy(predicted_class, targets)
            total_acc += acc

            # IoU per class
            iou = IoU(predicted_class, targets, num_classes)
            total_iou = [x + y for x, y in zip(total_iou, iou)]

            # Dice score per class
            dice = dice_coeff(predicted_class, targets, num_classes)
            total_dice = [x + y for x, y in zip(total_dice, dice)]

            # Per-class accuracy
            class_accuracy = per_class_accuracy(predicted_class, targets, num_classes)
            total_class_accuracy = [x + y for x, y in zip(total_class_accuracy, class_accuracy)]

            total_loss += loss.item()

    avg_loss = total_loss / len(val_loader)
    avg_acc = total_acc / len(val_loader)
    avg_iou = [x / len(val_loader) for x in total_iou]
    avg_dice = [x / len(val_loader) for x in total_dice]
    avg_class_accuracy = [x / len(val_loader) for x in total_class_accuracy]

    performance = {
        "Val Loss": avg_loss,
        "Mean Pixel Accuracy": avg_acc,
        "IoU": avg_iou,
        "Dice Score": avg_dice,
        "Per-class Accuracy": avg_class_accuracy
    }

    return performance


def plot_loss(performance: pd.DataFrame):
    plt.figure(figsize=(10, 6))
    plt.plot(performance["Epoch"], performance["Train Loss"], label="Train Loss")
    plt.plot(performance["Epoch"], performance["Val Loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.show()