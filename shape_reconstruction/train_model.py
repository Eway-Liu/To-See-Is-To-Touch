"""
训练脚本
训练深度学习模型进行三维形状重建
"""
import os
import sys
import time
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm

# 添加路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dataset import ShapeReconstructionDataset
from model import ReconstructionNet, SimpleReconstructionNet, ResNetReconstructionNet


class EdgeLoss(nn.Module):
    """边缘损失：保持边缘清晰"""
    def __init__(self):
        super(EdgeLoss, self).__init__()

    def forward(self, pred, target):
        # 计算梯度
        pred_grad_x = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        pred_grad_y = pred[:, :, 1:, :] - pred[:, :, :-1, :]

        target_grad_x = target[:, :, :, 1:] - target[:, :, :, :-1]
        target_grad_y = target[:, :, 1:, :] - target[:, :, :-1, :]

        # 梯度差的L1损失
        loss_x = torch.abs(pred_grad_x - target_grad_x).mean()
        loss_y = torch.abs(pred_grad_y - target_grad_y).mean()

        return loss_x + loss_y


class CombinedLoss(nn.Module):
    """组合损失函数"""
    def __init__(self, alpha=1.0, beta=0.1):
        super(CombinedLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        self.edge_loss = EdgeLoss()
        self.alpha = alpha  # MSE权重
        self.beta = beta    # 边缘损失权重

    def forward(self, pred, target):
        mse = self.mse_loss(pred, target)
        l1 = self.l1_loss(pred, target)
        edge = self.edge_loss(pred, target)

        total_loss = self.alpha * mse + l1 + self.beta * edge
        return total_loss, {'mse': mse.item(), 'l1': l1.item(), 'edge': edge.item()}


def train_epoch(model, train_loader, optimizer, criterion, device, epoch):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    total_mse = 0.0
    total_l1 = 0.0
    total_edge = 0.0

    pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]')
    for diff_img, height_map in pbar:
        diff_img = diff_img.to(device)
        height_map = height_map.to(device)

        # 前向传播
        optimizer.zero_grad()
        pred_height = model(diff_img)

        # 计算损失
        loss, loss_dict = criterion(pred_height, height_map)

        # 反向传播
        loss.backward()
        optimizer.step()

        # 记录
        total_loss += loss.item()
        total_mse += loss_dict['mse']
        total_l1 += loss_dict['l1']
        total_edge += loss_dict['edge']

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'mse': f"{loss_dict['mse']:.4f}"
        })

    num_batches = len(train_loader)
    return {
        'loss': total_loss / num_batches,
        'mse': total_mse / num_batches,
        'l1': total_l1 / num_batches,
        'edge': total_edge / num_batches
    }


def validate(model, val_loader, criterion, device, epoch):
    """验证"""
    model.eval()
    total_loss = 0.0
    total_mse = 0.0
    total_l1 = 0.0
    total_edge = 0.0
    total_mae = 0.0
    total_rmse = 0.0

    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f'Epoch {epoch} [Val]')
        for diff_img, height_map in pbar:
            diff_img = diff_img.to(device)
            height_map = height_map.to(device)

            # 前向传播
            pred_height = model(diff_img)

            # 计算损失
            loss, loss_dict = criterion(pred_height, height_map)

            # 额外指标
            mae = torch.abs(pred_height - height_map).mean()
            rmse = torch.sqrt(((pred_height - height_map) ** 2).mean())

            total_loss += loss.item()
            total_mse += loss_dict['mse']
            total_l1 += loss_dict['l1']
            total_edge += loss_dict['edge']
            total_mae += mae.item()
            total_rmse += rmse.item()

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'rmse': f'{rmse.item():.4f}'
            })

    num_batches = len(val_loader)
    return {
        'loss': total_loss / num_batches,
        'mse': total_mse / num_batches,
        'l1': total_l1 / num_batches,
        'edge': total_edge / num_batches,
        'mae': total_mae / num_batches,
        'rmse': total_rmse / num_batches
    }


def main():
    parser = argparse.ArgumentParser(description='训练形状重建模型')
    parser.add_argument('--data_dir', type=str, default='./reconstruction_data',
                        help='数据目录')
    parser.add_argument('--model', type=str, default='unet',
                        choices=['unet', 'simple', 'resnet'],
                        help='模型架构')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='批次大小')
    parser.add_argument('--epochs', type=int, default=50,
                        help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='学习率')
    parser.add_argument('--save_dir', type=str, default='../saved_models',
                        help='模型保存目录')
    parser.add_argument('--log_dir', type=str, default='./logs',
                        help='日志目录')
    parser.add_argument('--resume', type=str, default=None,
                        help='恢复训练的检查点路径')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='设备')

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("训练形状重建模型")
    print("=" * 60)
    print(f"\n配置:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")

    # 设置设备
    device = torch.device(args.device)
    print(f"\n使用设备: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # 创建数据集
    print("\n加载数据集...")
    train_dataset = ShapeReconstructionDataset(
        mode='train',
        data_dir=args.data_dir,
        augmentation=True
    )
    val_dataset = ShapeReconstructionDataset(
        mode='val',
        data_dir=args.data_dir,
        augmentation=False
    )

    print(f"训练集: {len(train_dataset)} 个样本")
    print(f"验证集: {len(val_dataset)} 个样本")

    if len(train_dataset) == 0:
        print("\n错误：训练集为空！")
        print("请先运行数据准备脚本：")
        print("  python shape_reconstruction/prepare_data.py --method synthetic")
        return

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # 创建模型
    print("\n创建模型...")
    if args.model == 'unet':
        model = ReconstructionNet(base_features=32)
    elif args.model == 'simple':
        model = SimpleReconstructionNet()
    elif args.model == 'resnet':
        model = ResNetReconstructionNet()
    else:
        raise ValueError(f"未知模型: {args.model}")

    model = model.to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型: {args.model}")
    print(f"参数量: {num_params:,}")

    # 损失函数
    criterion = CombinedLoss(alpha=1.0, beta=0.1)

    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    # TensorBoard
    writer = SummaryWriter(args.log_dir)

    # 恢复训练
    start_epoch = 0
    best_val_loss = float('inf')

    if args.resume:
        if os.path.isfile(args.resume):
            print(f"\n从 {args.resume} 恢复训练...")
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            print(f"从 epoch {start_epoch} 继续训练")
        else:
            print(f"\n警告：未找到检查点 {args.resume}")

    # 训练循环
    print("\n" + "=" * 60)
    print("开始训练")
    print("=" * 60)

    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 60)

        # 训练
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device, epoch + 1)

        print(f"\n[训练] Loss: {train_metrics['loss']:.4f} | "
              f"MSE: {train_metrics['mse']:.4f} | "
              f"L1: {train_metrics['l1']:.4f} | "
              f"Edge: {train_metrics['edge']:.4f}")

        # 验证
        if len(val_dataset) > 0:
            val_metrics = validate(model, val_loader, criterion, device, epoch + 1)

            print(f"[验证] Loss: {val_metrics['loss']:.4f} | "
                  f"RMSE: {val_metrics['rmse']:.4f} | "
                  f"MAE: {val_metrics['mae']:.4f}")

            # 学习率调度
            scheduler.step(val_metrics['loss'])

            # 记录到TensorBoard
            writer.add_scalar('Loss/train', train_metrics['loss'], epoch)
            writer.add_scalar('Loss/val', val_metrics['loss'], epoch)
            writer.add_scalar('Metrics/RMSE', val_metrics['rmse'], epoch)
            writer.add_scalar('Metrics/MAE', val_metrics['mae'], epoch)
            writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)

            # 保存最佳模型
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                best_model_path = os.path.join(args.save_dir, 'reconstruction_best.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': val_metrics['loss'],
                    'rmse': val_metrics['rmse'],
                    'mae': val_metrics['mae'],
                    'best_val_loss': best_val_loss,
                    'model_type': args.model
                }, best_model_path)
                print(f"✓ 保存最佳模型到 {best_model_path}")
        else:
            # 没有验证集，记录训练指标
            writer.add_scalar('Loss/train', train_metrics['loss'], epoch)
            writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)

        # 定期保存检查点
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(args.save_dir, f'reconstruction_epoch_{epoch + 1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'model_type': args.model
            }, checkpoint_path)
            print(f"✓ 保存检查点到 {checkpoint_path}")

    # 训练完成
    print("\n" + "=" * 60)
    print("训练完成！")
    print("=" * 60)

    # 保存最终模型
    final_model_path = os.path.join(args.save_dir, 'reconstruction_final.pth')
    torch.save({
        'epoch': args.epochs - 1,
        'model_state_dict': model.state_dict(),
        'model_type': args.model
    }, final_model_path)
    print(f"\n最终模型保存到: {final_model_path}")
    print(f"最佳模型保存到: {os.path.join(args.save_dir, 'reconstruction_best.pth')}")
    print(f"TensorBoard日志: {args.log_dir}")
    print("\n查看训练日志:")
    print(f"  tensorboard --logdir {args.log_dir}")

    writer.close()


if __name__ == '__main__':
    main()
