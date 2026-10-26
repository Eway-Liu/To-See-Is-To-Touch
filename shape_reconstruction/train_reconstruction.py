"""
快速训练脚本 - 72小时方案
从差分图像预测高度图
"""
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from glob import glob


class TactileDataset(Dataset):
    """触觉数据集"""
    def __init__(self, data_dir, augment=False):
        self.diff_dir = os.path.join(data_dir, 'diff_images')
        self.height_dir = os.path.join(data_dir, 'height_maps')

        self.diff_files = sorted(glob(os.path.join(self.diff_dir, '*.npy')))
        self.augment = augment

        if len(self.diff_files) == 0:
            raise ValueError(f"在 {self.diff_dir} 中没有找到数据文件")

        print(f"加载数据集: {data_dir}")
        print(f"  样本数量: {len(self.diff_files)}")

    def __len__(self):
        return len(self.diff_files)

    def __getitem__(self, idx):
        # 加载数据
        diff = np.load(self.diff_files[idx]).astype(np.float32)
        height = np.load(self.diff_files[idx].replace('diff_images', 'height_maps')).astype(np.float32)

        # 归一化
        diff = diff / 255.0  # [0, 1]
        height = height / 10.0  # 假设最大高度 10mm

        # 数据增强
        if self.augment:
            # 随机水平翻转
            if np.random.rand() > 0.5:
                diff = np.fliplr(diff)
                height = np.fliplr(height)

            # 随机垂直翻转
            if np.random.rand() > 0.5:
                diff = np.flipud(diff)
                height = np.flipud(height)

            # 随机旋转 90度
            if np.random.rand() > 0.5:
                k = np.random.randint(1, 4)
                diff = np.rot90(diff, k)
                height = np.rot90(height, k)

        # 转换为 tensor [C, H, W]
        diff = torch.from_numpy(diff[None, ...])  # [1, H, W]
        height = torch.from_numpy(height[None, ...])  # [1, H, W]

        return diff, height


class UNet(nn.Module):
    """简化的 U-Net 模型"""
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet, self).__init__()

        # Encoder
        self.enc1 = self.conv_block(in_channels, 32)
        self.enc2 = self.conv_block(32, 64)
        self.enc3 = self.conv_block(64, 128)
        self.enc4 = self.conv_block(128, 256)

        # Bottleneck
        self.bottleneck = self.conv_block(256, 512)

        # Decoder
        self.up4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec4 = self.conv_block(512, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = self.conv_block(256, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = self.conv_block(128, 64)

        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = self.conv_block(64, 32)

        # Output
        self.out = nn.Conv2d(32, out_channels, 1)

        self.pool = nn.MaxPool2d(2)

    def conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Bottleneck
        b = self.bottleneck(self.pool(e4))

        # Decoder
        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        out = self.out(d1)
        return out


def train_epoch(model, dataloader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0

    for diff, height in tqdm(dataloader, desc="Training", leave=False):
        diff = diff.to(device)
        height = height.to(device)

        # 前向传播
        pred = model(diff)
        loss = criterion(pred, height)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    """验证"""
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for diff, height in tqdm(dataloader, desc="Validating", leave=False):
            diff = diff.to(device)
            height = height.to(device)

            pred = model(diff)
            loss = criterion(pred, height)

            total_loss += loss.item()

    return total_loss / len(dataloader)


def save_comparison(model, val_dataset, device, save_path, num_samples=4):
    """保存对比图"""
    model.eval()

    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4*num_samples))
    if num_samples == 1:
        axes = axes[None, :]

    indices = np.random.choice(len(val_dataset), num_samples, replace=False)

    with torch.no_grad():
        for i, idx in enumerate(indices):
            diff, height_gt = val_dataset[idx]
            diff = diff.unsqueeze(0).to(device)

            height_pred = model(diff).cpu().squeeze()
            diff = diff.cpu().squeeze()
            height_gt = height_gt.squeeze()

            # 反归一化
            diff = diff.numpy()
            height_gt = height_gt.numpy() * 10.0
            height_pred = height_pred.numpy() * 10.0

            # 绘图
            axes[i, 0].imshow(diff, cmap='gray')
            axes[i, 0].set_title('Input: Diff Image')
            axes[i, 0].axis('off')

            axes[i, 1].imshow(height_gt, cmap='jet')
            axes[i, 1].set_title(f'GT: {height_gt.max():.2f}mm')
            axes[i, 1].axis('off')

            axes[i, 2].imshow(height_pred, cmap='jet')
            axes[i, 2].set_title(f'Pred: {height_pred.max():.2f}mm')
            axes[i, 2].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"对比图保存: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='训练形状重建模型')
    parser.add_argument('--train_dir', type=str, default='./reconstruction_data/train',
                        help='训练数据目录')
    parser.add_argument('--val_dir', type=str, default='./reconstruction_data/val',
                        help='验证数据目录')
    parser.add_argument('--epochs', type=int, default=100,
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='批次大小')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='学习率')
    parser.add_argument('--save_dir', type=str, default='./models',
                        help='模型保存目录')
    parser.add_argument('--load_model', type=str, default=None,
                        help='加载预训练模型')

    args = parser.parse_args()

    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)

    print("\n" + "="*60)
    print("形状重建模型训练")
    print("="*60)

    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 加载数据
    print("\n加载数据...")
    train_dataset = TactileDataset(args.train_dir, augment=True)
    val_dataset = TactileDataset(args.val_dir, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                             shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                           shuffle=False, num_workers=4)

    # 创建模型
    print("\n创建模型...")
    model = UNet(in_channels=1, out_channels=1).to(device)

    if args.load_model:
        print(f"加载预训练模型: {args.load_model}")
        model.load_state_dict(torch.load(args.load_model))

    # 损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)

    # 训练
    print("\n开始训练...")
    print("="*60)

    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    for epoch in range(args.epochs):
        # 训练
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)

        # 验证
        val_loss = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)

        # 调整学习率
        scheduler.step(val_loss)

        # 打印
        print(f"Epoch [{epoch+1}/{args.epochs}] "
              f"Train Loss: {train_loss:.6f} | "
              f"Val Loss: {val_loss:.6f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'best_model.pth'))
            print(f"  ✓ 保存最佳模型 (val_loss: {val_loss:.6f})")

        # 定期保存对比图
        if (epoch + 1) % 10 == 0:
            save_comparison(model, val_dataset, device,
                          os.path.join(args.save_dir, f'comparison_epoch_{epoch+1}.png'))

    # 保存最终模型
    torch.save(model.state_dict(), os.path.join(args.save_dir, 'final_model.pth'))

    # 绘制训练曲线
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(args.save_dir, 'training_curve.png'))
    print(f"\n训练曲线保存: {os.path.join(args.save_dir, 'training_curve.png')}")

    print("\n" + "="*60)
    print("训练完成！")
    print("="*60)
    print(f"最佳验证损失: {best_val_loss:.6f}")
    print(f"模型保存在: {args.save_dir}")
    print(f"  - best_model.pth: 最佳模型")
    print(f"  - final_model.pth: 最终模型")
    print(f"  - training_curve.png: 训练曲线")
    print(f"  - comparison_epoch_*.png: 对比图")

    # 最终评估
    print("\n最终评估...")
    save_comparison(model, val_dataset, device,
                  os.path.join(args.save_dir, 'final_comparison.png'), num_samples=8)


if __name__ == '__main__':
    main()
