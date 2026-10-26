"""
Shape Reconstruction Dataset
用于训练深度学习模型的数据集类
"""
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import random


class ShapeReconstructionDataset(Dataset):
    """
    三维形状重建数据集
    输入：差分图像（灰度图）
    输出：高度图
    """
    def __init__(self, mode='train', data_dir='./reconstruction_data',
                 transform=None, augmentation=True):
        """
        Args:
            mode: 'train' 或 'val'
            data_dir: 数据根目录
            transform: 数据变换
            augmentation: 是否使用数据增强
        """
        self.mode = mode
        self.transform = transform
        self.augmentation = augmentation and (mode == 'train')

        # 加载数据路径
        self.diff_images = []
        self.height_maps = []

        data_path = os.path.join(data_dir, mode)
        diff_dir = os.path.join(data_path, 'diff_images')
        height_dir = os.path.join(data_path, 'height_maps')

        if os.path.exists(diff_dir) and os.path.exists(height_dir):
            diff_files = sorted([f for f in os.listdir(diff_dir) if f.endswith('.npy')])
            for diff_file in diff_files:
                height_file = diff_file  # 相同的文件名
                diff_path = os.path.join(diff_dir, diff_file)
                height_path = os.path.join(height_dir, height_file)

                if os.path.exists(height_path):
                    self.diff_images.append(diff_path)
                    self.height_maps.append(height_path)

        print(f"Loaded {len(self.diff_images)} samples for {mode} set")

    def __len__(self):
        return len(self.diff_images)

    def __getitem__(self, idx):
        # 加载数据
        diff_img = np.load(self.diff_images[idx]).astype(np.float32)
        height_map = np.load(self.height_maps[idx]).astype(np.float32)

        # 数据增强
        if self.augmentation:
            diff_img, height_map = self._augment(diff_img, height_map)

        # 归一化
        diff_img = self._normalize_diff(diff_img)
        height_map = self._normalize_height(height_map)

        # 转换为tensor，添加通道维度
        diff_img = torch.from_numpy(diff_img[np.newaxis, :, :])  # [1, H, W]
        height_map = torch.from_numpy(height_map[np.newaxis, :, :])  # [1, H, W]

        if self.transform:
            diff_img = self.transform(diff_img)
            height_map = self.transform(height_map)

        return diff_img, height_map

    def _augment(self, diff_img, height_map):
        """数据增强"""
        # 随机水平翻转
        if random.random() > 0.5:
            diff_img = np.fliplr(diff_img)
            height_map = np.fliplr(height_map)

        # 随机垂直翻转
        if random.random() > 0.5:
            diff_img = np.flipud(diff_img)
            height_map = np.flipud(height_map)

        # 随机旋转90度的倍数
        k = random.randint(0, 3)
        if k > 0:
            diff_img = np.rot90(diff_img, k)
            height_map = np.rot90(height_map, k)

        # 添加随机噪声
        if random.random() > 0.7:
            noise = np.random.normal(0, 2, diff_img.shape).astype(np.float32)
            diff_img = diff_img + noise

        return diff_img, height_map

    def _normalize_diff(self, diff_img):
        """归一化差分图像"""
        # 裁剪到合理范围
        diff_img = np.clip(diff_img, 0, 255)
        # 归一化到[0, 1]
        diff_img = diff_img / 255.0
        return diff_img

    def _normalize_height(self, height_map):
        """归一化高度图"""
        # 假设高度图范围是0-10mm
        height_map = np.clip(height_map, 0, 10)
        # 归一化到[0, 1]
        height_map = height_map / 10.0
        return height_map


class ShapeReconstructionDatasetFromSensor(Dataset):
    """
    从传感器实时数据生成训练集
    使用传统方法生成伪标签
    """
    def __init__(self, sensor, num_samples=1000, save_dir='./reconstruction_data/train'):
        """
        Args:
            sensor: Sensor对象实例
            num_samples: 要采集的样本数
            save_dir: 保存目录
        """
        self.sensor = sensor
        self.num_samples = num_samples
        self.save_dir = save_dir

        self.diff_dir = os.path.join(save_dir, 'diff_images')
        self.height_dir = os.path.join(save_dir, 'height_maps')

        os.makedirs(self.diff_dir, exist_ok=True)
        os.makedirs(self.height_dir, exist_ok=True)

    def collect_data(self):
        """采集数据"""
        print(f"开始采集{self.num_samples}个样本...")
        count = 0

        while count < self.num_samples and self.sensor.cap.isOpened():
            # 获取图像
            img = self.sensor.get_rectify_crop_image()
            img_GRAY = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # 计算差分图像
            diff = self.sensor.ref_GRAY.astype(np.float32) - img_GRAY.astype(np.float32)
            diff = np.clip(diff, 0, 255)

            # 使用传统方法生成高度图（作为伪标签）
            height_map = self.sensor.raw_image_2_height_map(img_GRAY)

            # 只保存有接触的数据（高度图不全为0）
            if np.max(height_map) > 0.1:
                # 保存
                np.save(os.path.join(self.diff_dir, f'{count:06d}.npy'), diff)
                np.save(os.path.join(self.height_dir, f'{count:06d}.npy'), height_map)
                count += 1

                if count % 10 == 0:
                    print(f"已采集 {count}/{self.num_samples} 个样本")

            # 显示进度
            cv2.imshow('Current Image', img)
            key = cv2.waitKey(1)
            if key == ord('q'):
                break

        cv2.destroyAllWindows()
        print(f"数据采集完成！共采集 {count} 个样本")
        return count


if __name__ == '__main__':
    # 测试数据集
    print("测试ShapeReconstructionDataset...")
    dataset = ShapeReconstructionDataset(mode='train', data_dir='./reconstruction_data')
    if len(dataset) > 0:
        diff, height = dataset[0]
        print(f"Diff image shape: {diff.shape}")
        print(f"Height map shape: {height.shape}")
        print(f"Diff range: [{diff.min():.3f}, {diff.max():.3f}]")
        print(f"Height range: [{height.min():.3f}, {height.max():.3f}]")
    else:
        print("数据集为空，请先运行数据准备脚本")
