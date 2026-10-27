"""
深度学习模型定义
用于三维形状重建的神经网络
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ReconstructionNet(nn.Module):
    """
    轻量级U-Net架构
    输入: 差分图像 [B, 1, H, W]
    输出: 高度图 [B, 1, H, W]
    """
    def __init__(self, in_channels=1, out_channels=1, base_features=32):
        super(ReconstructionNet, self).__init__()

        # Encoder
        self.enc1 = self._conv_block(in_channels, base_features)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = self._conv_block(base_features, base_features * 2)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = self._conv_block(base_features * 2, base_features * 4)
        self.pool3 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = self._conv_block(base_features * 4, base_features * 8)

        # Decoder
        self.upconv3 = nn.ConvTranspose2d(base_features * 8, base_features * 4,
                                          kernel_size=2, stride=2)
        self.dec3 = self._conv_block(base_features * 8, base_features * 4)

        self.upconv2 = nn.ConvTranspose2d(base_features * 4, base_features * 2,
                                          kernel_size=2, stride=2)
        self.dec2 = self._conv_block(base_features * 4, base_features * 2)

        self.upconv1 = nn.ConvTranspose2d(base_features * 2, base_features,
                                          kernel_size=2, stride=2)
        self.dec1 = self._conv_block(base_features * 2, base_features)

        # Output
        self.out = nn.Conv2d(base_features, out_channels, kernel_size=1)

    def _conv_block(self, in_channels, out_channels):
        """卷积块：Conv -> BN -> ReLU -> Conv -> BN -> ReLU"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        size = x.shape[-2:]
        x = F.pad(x, (0, (-size[1]) % 8, 0, (-size[0]) % 8), mode='replicate')
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        enc3 = self.enc3(self.pool2(enc2))

        # Bottleneck
        bottleneck = self.bottleneck(self.pool3(enc3))

        # Decoder with skip connections
        dec3 = self.upconv3(bottleneck)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.dec3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)

        # Output
        out = self.out(dec1)
        out = torch.relu(out)  # 确保输出非负

        return out[..., :size[0], :size[1]]


class SimpleReconstructionNet(nn.Module):
    """
    简化版模型（更快的训练速度）
    适合快速验证和数据不足的情况
    """
    def __init__(self, in_channels=1, out_channels=1):
        super(SimpleReconstructionNet, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, out_channels, kernel_size=1),
            nn.ReLU()  # 确保输出非负
        )

    def forward(self, x):
        size = x.shape[-2:]
        x = F.pad(x, (0, (-size[1]) % 4, 0, (-size[0]) % 4), mode='replicate')
        x = self.encoder(x)
        x = self.decoder(x)
        return x[..., :size[0], :size[1]]


class ResidualBlock(nn.Module):
    """残差块"""
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out


class ResNetReconstructionNet(nn.Module):
    """
    基于ResNet的重建网络
    更深的网络，适合大数据集
    """
    def __init__(self, in_channels=1, out_channels=1):
        super(ResNetReconstructionNet, self).__init__()

        self.conv_in = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # Residual blocks
        self.res_block1 = ResidualBlock(64)
        self.res_block2 = ResidualBlock(64)

        self.down1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.res_block3 = ResidualBlock(128)
        self.res_block4 = ResidualBlock(128)

        # Decoder
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )

        self.conv_out = nn.Sequential(
            nn.Conv2d(16, out_channels, kernel_size=1),
            nn.ReLU()
        )

    def forward(self, x):
        size = x.shape[-2:]
        x = F.pad(x, (0, (-size[1]) % 8, 0, (-size[0]) % 8), mode='replicate')
        x = self.conv_in(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.down1(x)
        x = self.res_block3(x)
        x = self.res_block4(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.conv_out(x)
        return x[..., :size[0], :size[1]]


def count_parameters(model):
    """统计模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class ResidualConv(nn.Module):
    """GroupNorm remains stable with small full-resolution training batches."""
    def __init__(self, cin, cout):
        super().__init__()
        groups = min(8, cout)
        while cout % groups:
            groups -= 1
        self.body = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.GroupNorm(groups, cout), nn.SiLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.GroupNorm(groups, cout))
        self.skip = nn.Conv2d(cin, cout, 1, bias=False) if cin != cout else nn.Identity()
        hidden = max(cout // 8, 4)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(cout, hidden, 1), nn.SiLU(),
            nn.Conv2d(hidden, cout, 1), nn.Sigmoid())

    def forward(self, x):
        y = self.body(x)
        return F.silu(self.skip(x) + y * self.channel_attention(y))


class AttentionResidualUNet(nn.Module):
    """Four-scale residual U-Net for dense height regression, no input resizing."""
    def __init__(self, base_features=48):
        super().__init__()
        widths = [base_features * 2 ** i for i in range(5)]
        self.encoders = nn.ModuleList([ResidualConv(1, widths[0])] + [
            ResidualConv(widths[i - 1], widths[i]) for i in range(1, 5)])
        self.reductions = nn.ModuleList([nn.Conv2d(widths[i + 1], widths[i], 1)
                                         for i in reversed(range(4))])
        self.gates = nn.ModuleList([nn.Conv2d(widths[i] * 2, 1, 1)
                                    for i in reversed(range(4))])
        self.decoders = nn.ModuleList([ResidualConv(widths[i] * 2, widths[i])
                                       for i in reversed(range(4))])
        self.output = nn.Conv2d(widths[0], 1, 1)
        nn.init.normal_(self.output.weight, std=0.001)
        nn.init.zeros_(self.output.bias)

    def forward(self, x):
        size = x.shape[-2:]
        x = F.pad(x, (0, (-size[1]) % 16, 0, (-size[0]) % 16), mode='replicate')
        skips = []
        for i, encoder in enumerate(self.encoders):
            x = encoder(F.avg_pool2d(x, 2) if i else x)
            skips.append(x)
        for reduce, gate, decode, skip in zip(
                self.reductions, self.gates, self.decoders, reversed(skips[:-1])):
            x = reduce(F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False))
            attention = torch.sigmoid(gate(torch.cat([x, skip], dim=1)))
            x = decode(torch.cat([x, skip * attention], dim=1))
        # Linear regression avoids permanently zero gradients from a final ReLU.
        return self.output(x)[..., :size[0], :size[1]]


def build_model(model_type='attention_unet', base_features=48):
    if model_type == 'attention_unet':
        return AttentionResidualUNet(base_features)
    if model_type == 'unet':
        return ReconstructionNet(base_features=base_features)
    if model_type == 'simple':
        return SimpleReconstructionNet()
    if model_type == 'resnet':
        return ResNetReconstructionNet()
    raise ValueError(f'Unknown model: {model_type}')


if __name__ == '__main__':
    # 测试模型
    print("测试模型架构...")

    # 测试输入
    batch_size = 4
    height, width = 160, 224  # 示例尺寸，需要是8的倍数
    x = torch.randn(batch_size, 1, height, width)

    print(f"\n输入形状: {x.shape}")

    # 测试 ReconstructionNet
    print("\n1. ReconstructionNet (U-Net)")
    model1 = ReconstructionNet(base_features=32)
    y1 = model1(x)
    print(f"   输出形状: {y1.shape}")
    print(f"   参数量: {count_parameters(model1):,}")

    # 测试 SimpleReconstructionNet
    print("\n2. SimpleReconstructionNet")
    model2 = SimpleReconstructionNet()
    y2 = model2(x)
    print(f"   输出形状: {y2.shape}")
    print(f"   参数量: {count_parameters(model2):,}")

    # 测试 ResNetReconstructionNet
    print("\n3. ResNetReconstructionNet")
    model3 = ResNetReconstructionNet()
    y3 = model3(x)
    print(f"   输出形状: {y3.shape}")
    print(f"   参数量: {count_parameters(model3):,}")

    print("\n推荐：")
    print("  - 数据充足（>500样本）: ReconstructionNet")
    print("  - 数据较少（<500样本）: SimpleReconstructionNet")
    print("  - 追求最高精度: ResNetReconstructionNet")
