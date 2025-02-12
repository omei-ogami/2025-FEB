import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet18_Weights

class UNet(nn.Module):
    def __init__(self, in_channels=6, out_channels=4):
        super(UNet, self).__init__()

        # Encoder (ResNet)
        self.resnet = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Additional Conv + Pool Blocks (1 Conv + 1 Pool each)
        self.conv1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Attention Block
        self.se1 = SEBlock(64)
        self.se2 = SEBlock(128)
        self.se3 = SEBlock(256)
        self.se4 = SEBlock(512)

        # Bottleneck
        self.bottleneck_conv = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        
        # Decoder
        self.upconv1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.upconv3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        
        # Decoder Convolutions
        self.dec_conv1 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.dec_conv2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.dec_conv3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        
        # Output
        self.outconv = nn.Conv2d(64, out_channels, kernel_size=1)
    
    def forward(self, x):
        # Encoder
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x1 = F.relu(self.conv1(x))
        x1 = self.se1(x1)
        x1_pool = self.pool1(x1)
        
        x2 = F.relu(self.conv2(x1_pool))
        x2 = self.se2(x2)
        x2_pool = self.pool2(x2)
        
        x3 = F.relu(self.conv3(x2_pool))
        x3 = self.se3(x3)
        x3_pool = self.pool3(x3)
        
        # Bottleneck
        bottleneck = self.bottleneck_conv(x3_pool)
        bottleneck = self.se4(bottleneck)
        
        # Decoder
        up1 = F.relu(self.upconv1(bottleneck))
        x3_up = F.interpolate(x3, size=up1.shape[2:], mode="bilinear", align_corners=True)
        up1 = torch.cat((up1, x3_up), dim=1)
        up1 = self.dec_conv1(up1)
        
        up2 = F.relu(self.upconv2(up1))
        x2_up = F.interpolate(x2, size=up2.shape[2:], mode="bilinear", align_corners=True)
        up2 = torch.cat((up2, x2_up), dim=1)
        up2 = self.dec_conv2(up2)
        
        up3 = F.relu(self.upconv3(up2))
        x1_up = F.interpolate(x1, size=up3.shape[2:], mode="bilinear", align_corners=True)
        up3 = torch.cat((up3, x1_up), dim=1)
        up3 = self.dec_conv3(up3)
        
        # Output
        x = self.outconv(up3)
        x = F.interpolate(x, size=(1024, 1024), mode="bilinear", align_corners=True)
        return x

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)  # Global Avg Pooling
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch, channels, _, _ = x.shape
        squeeze = self.global_avg_pool(x).view(batch, channels)  # [B, C, 1, 1] → [B, C]
        excite = self.fc(squeeze).view(batch, channels, 1, 1)  # [B, C] → [B, C, 1, 1]
        return x * excite  # Reweighting
