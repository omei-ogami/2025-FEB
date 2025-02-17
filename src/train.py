import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from metric import mean_pixel_accuracy, IoU, dice_coeff, per_class_accuracy
import torch.amp as amp

def train(model, optimizer, criterion, train_loader, val_loader, num_epochs, device, exp_id):
    folder = f"runs/experiment_{exp_id}"
    writer = SummaryWriter(log_dir=folder)

    performance_data = []
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-7)
    train_loss, eval_loss = [], []
    best_iou = 0.0
    patience = 7

    scaler = amp.GradScaler()

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for inputs, targets in tqdm.tqdm(train_loader, unit="batch", desc=f"Epoch {epoch+1}/{num_epochs}"):
            optimizer.zero_grad()
            inputs, targets = inputs.to(device), targets.to(device)

            with amp.autocast(device_type='cuda'):
                outputs = model(inputs)
                targets = targets.squeeze(1)
                loss = criterion(outputs, targets)


            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        train_loss.append(avg_loss)
        val_performance = evaluate(model, val_loader, criterion, device)
        eval_loss.append(val_performance["Val Loss"])
        scheduler.step(val_performance["Mean IoU"])
        
        writer.add_scalar('Training Loss', avg_loss, epoch)
        writer.add_scalar('Validation Loss', val_performance["Val Loss"], epoch)
        writer.add_scalar('Mean IoU', val_performance["Mean IoU"], epoch)
        writer.add_scalar('Dice Score', val_performance["Mean Dice Score"], epoch)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_loss:.4f}, Val Loss: {val_performance['Val Loss']:.4f}, Mean IoU: {val_performance['Mean IoU']:.4f}")
        
        performance_data.append({
            "Epoch": epoch + 1,
            "Train Loss": avg_loss,
            "Val Loss": val_performance["Val Loss"],
            "Mean Pixel Accuracy": val_performance["Mean Pixel Accuracy"],
            "Mean IoU": val_performance["Mean IoU"],
            "Mean Dice Score": val_performance["Mean Dice Score"],
            "Per-class IoU": val_performance["IoU"],
            "Per-class Dice": val_performance["Dice Score"]
        })

        if val_performance["Mean IoU"] > best_iou:
            best_iou = val_performance["Mean IoU"]
            patience = 10
        else:
            patience -= 1
            if patience == 0:
                print("Early stopping triggered at epoch", epoch + 1)
                break

    writer.close()
    return pd.DataFrame(performance_data)

def evaluate(model, val_loader, criterion, device, num_classes=4):
    model.eval()
    total_loss, total_acc = 0, 0
    total_iou = torch.zeros(num_classes, device=device)
    total_dice = torch.zeros(num_classes, device=device)
    total_class_accuracy = torch.zeros(num_classes, device=device)
    
    with torch.no_grad():
        for inputs, targets in tqdm.tqdm(val_loader, unit="batch", desc="Validating"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            targets = targets.squeeze(1)
            loss = criterion(outputs, targets)
            predicted_class = torch.argmax(outputs, dim=1)
            
            total_loss += loss.item()

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
    
    num_batches = len(val_loader)
    avg_loss = total_loss / len(val_loader)
    avg_acc = total_acc / len(val_loader)

    # Convert total_iou to a tensor and then compute the mean
    avg_iou = torch.tensor(total_iou, device=device) / len(val_loader)
    avg_dice = torch.tensor(total_dice, device=device) / len(val_loader)
    avg_class_accuracy = torch.tensor(total_class_accuracy, device=device) / len(val_loader)
    
    return {
        "Val Loss": avg_loss,
        "Mean Pixel Accuracy": avg_acc,
        "Mean IoU": avg_iou.mean().item(),  # Now it's a tensor, so we can call .mean()
        "Mean Dice Score": avg_dice.mean().item(),
        "IoU": avg_iou.tolist(),
        "Dice Score": avg_dice.tolist(),
        "Per-class Accuracy": avg_class_accuracy.tolist()
    }



def plot_loss(performance: pd.DataFrame):
    plt.figure(figsize=(10, 6))
    plt.plot(performance["Epoch"], performance["Train Loss"], label="Train Loss")
    plt.plot(performance["Epoch"], performance["Val Loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.show()