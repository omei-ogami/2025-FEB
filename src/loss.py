import torch
import segmentation_models_pytorch as smp
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 类别权重（确保搬到 GPU）
alpha = torch.tensor([0.3, 2.5, 5.0, 3.5], device=device)

# 定义损失函数
focal_loss = smp.losses.FocalLoss(mode="multiclass", gamma=5, reduction="none")
dice_loss = smp.losses.DiceLoss(mode="multiclass", from_logits=True)

def combined_loss(y_pred, y_true):
    # 计算 Dice 损失
    dice = dice_loss(y_pred, y_true)
    
    # 扩展 alpha 使其形状与 focal 一致 (B, C, H, W)
    expanded_alpha = alpha.view(1, -1, 1, 1).expand(y_pred.size(0), alpha.size(0), y_pred.size(2), y_pred.size(3))

    # 计算 Focal Loss
    focal = focal_loss(y_pred, y_true)
    
    # 将 focal 的形状调整为 (B, C, H, W)
    focal_one_hot = F.one_hot(y_true, num_classes=y_pred.size(1)).permute(0, 3, 1, 2).float()
    
    # 使用 one-hot 编码的标签来加权 focal loss
    weighted_focal = (focal_one_hot * focal.unsqueeze(1) * expanded_alpha).mean()

    # 使用加权 Dice Loss
    weighted_dice = (dice * expanded_alpha).mean()
    
    return 0.45 * weighted_dice + 0.55 * weighted_focal


# test
y_pred = torch.rand(64, 4, 512, 512).to(device)
y_true = torch.randint(0, 4, (64, 512, 512)).to(device)
#combined_loss(y_pred, y_true)