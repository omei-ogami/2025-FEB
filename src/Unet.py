import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet34_Weights

class UNet(nn.Module):
    def __init__(self, in_channels=6, out_channels=4):
        super(UNet, self).__init__()

        # Encoder (ResNet34)
        self.resnet = models.resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        self.resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # freeze the ResNet weights
        # for param in self.resnet.parameters():
        #    param.requires_grad = False

        # Replace fully connected layers for segmentation
        self.resnet.fc = nn.Identity()

        # ECA Block for each ResNet block
        self.eca1 = ECA(64)
        self.eca2 = ECA(128)
        self.eca3 = ECA(256)
        self.eca4 = ECA(512)

        # Skip Connection (1x1 Conv)
        self.skip1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            SEBlock(64)
        )
        self.skip2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            SEBlock(128)
        )
        self.skip3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            SEBlock(256)
        )
        self.skip4 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            SEBlock(512)
        )

        # Decoder
        self.upconv1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.upconv3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)


        # Decoder Convolutions (多層卷積)
        self.dec_conv1 = DecoderBlock(512, 256) # 修改
        self.dec_conv2 = DecoderBlock(256, 128) # 修改
        self.dec_conv3 = DecoderBlock(128, 64)   # 修改

        # Output layer
        self.outconv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder (ResNet34)
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        # Passing through the ResNet layers
        x1 = self.resnet.layer1(x)  # [B, 64, H/2, W/2]
        x2 = self.resnet.layer2(x1)  # [B, 128, H/4, W/4]
        x3 = self.resnet.layer3(x2)  # [B, 256, H/8, W/8]
        x4 = self.resnet.layer4(x3)  # [B, 512, H/16, W/16]

        # ECA Block
        x1 = self.eca1(x1)
        x2 = self.eca2(x2)
        x3 = self.eca3(x3)
        x4 = self.eca4(x4)

        # Skip Connection (1x1 Conv)
        x1_skip = self.skip1(x1)
        x2_skip = self.skip2(x2)
        x3_skip = self.skip3(x3)
        x4_skip = self.skip4(x4)

        # Decoder
        up1 = F.relu(self.upconv1(x4_skip)) # 修改
        up1 = torch.cat((up1, x3_skip), dim=1) # 修改
        up1 = self.dec_conv1(up1) # 修改

        up2 = F.relu(self.upconv2(up1))
        up2 = torch.cat((up2, x2_skip), dim=1) # 修改
        up2 = self.dec_conv2(up2) # 修改

        up3 = F.relu(self.upconv3(up2))
        up3 = torch.cat((up3, x1_skip), dim=1) # 修改
        up3 = self.dec_conv3(up3) # 修改

        # Output layer
        x = self.outconv(up3)
        x = F.interpolate(x, size=(512, 512), mode="bilinear", align_corners=True)

        return x

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_avg = torch.mean(x, dim=1, keepdim=True)
        x_max = torch.max(x, dim=1, keepdim=True)[0]
        x_concat = torch.cat([x_avg, x_max], dim=1)
        x_out = self.conv(x_concat)
        return self.sigmoid(x_out)
    
class ECA(nn.Module):
    def __init__(self, in_channels, kernel_size=3):
        super(ECA, self).__init__()
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size-1)//2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        y = torch.mean(x, dim=(2, 3), keepdim=True)  # Global Average Pooling
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        x_eca = x * y # 通道注意力輸出
        x_out = x_eca * self.spatial_attention(x_eca) # 結合空間注意力
        return x_out


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.5):  # Add dropout_rate parameter
        super(DecoderBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.dropout1 = nn.Dropout2d(dropout_rate)  # Dropout after ReLU
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout2 = nn.Dropout2d(dropout_rate)
        self.relu2 = nn.ReLU(inplace=True)
        

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)  # Apply dropout
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout2(x)  # Apply dropout
        return x
    
class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        x_se = self.avg_pool(x).squeeze(-1).squeeze(-1)
        x_se = self.fc(x_se)
        x_se = x_se.unsqueeze(-1).unsqueeze(-1)
        return x * x_se