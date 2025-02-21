import torch
import segmentation_models_pytorch as smp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义损失函数
focal_loss = smp.losses.FocalLoss(mode="multiclass", gamma=2)
dice_loss = smp.losses.DiceLoss(mode="multiclass", from_logits=True)

# 类别权重（确保搬到 GPU）
alpha = torch.tensor([0.3, 1.0, 2.5, 2.5], device=device)

# 组合损失函数
def combined_loss(y_pred, y_true):
    dice = dice_loss(y_pred, y_true)  # 逐类别 Dice Loss, shape: (B, C, H, W)
    
    # 扩展 alpha 维度以匹配 dice
    weighted_dice = (dice * alpha.view(1, -1, 1, 1)).mean()  # 广播到 (1, num_classes, 1, 1)
    
    return 0.5 * weighted_dice + 0.5 * focal_loss(y_pred, y_true)
