"""
数据准备脚本
生成用于训练的差分图像和高度图配对数据
"""
import os
import sys
import cv2
import yaml
import numpy as np
from tqdm import tqdm
import argparse

# 添加路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
from shape_reconstruction import Sensor


def collect_data_from_camera(sensor, num_samples=500, save_dir='./reconstruction_data/train'):
    """
    方法1：从摄像头实时采集数据
    使用传统方法生成的高度图作为伪标签
    """
    print("\n=== 方法1：从摄像头实时采集数据 ===")
    print(f"目标样本数: {num_samples}")
    print("操作说明：")
    print("  - 用不同物体接触传感器表面")
    print("  - 尝试不同的接触位置、角度和力度")
    print("  - 按 'q' 键退出")
    print("\n开始采集...")

    diff_dir = os.path.join(save_dir, 'diff_images')
    height_dir = os.path.join(save_dir, 'height_maps')
    os.makedirs(diff_dir, exist_ok=True)
    os.makedirs(height_dir, exist_ok=True)

    count = 0
    last_saved_height = None

    while count < num_samples and sensor.cap.isOpened():
        # 获取图像
        img = sensor.get_rectify_crop_image()
        img_GRAY = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 计算差分图像
        diff_raw = sensor.ref_GRAY.astype(np.float32) - img_GRAY.astype(np.float32)
        diff = np.clip(diff_raw, 0, 255).astype(np.float32)

        # 生成高度图（传统方法作为伪标签）
        height_map = sensor.raw_image_2_height_map(img_GRAY)

        # 只保存有明显接触的数据
        contact_threshold = 0.05  # mm
        max_height = np.max(height_map)

        if max_height > contact_threshold:
            # 避免保存相似的连续帧
            is_different = True
            if last_saved_height is not None:
                height_diff = np.abs(height_map - last_saved_height).mean()
                if height_diff < 0.01:  # 变化太小
                    is_different = False

            if is_different:
                # 保存数据
                np.save(os.path.join(diff_dir, f'{count:06d}.npy'), diff)
                np.save(os.path.join(height_dir, f'{count:06d}.npy'), height_map)

                last_saved_height = height_map.copy()
                count += 1

                # 显示进度
                print(f"\r已采集: {count}/{num_samples} | 最大高度: {max_height:.3f}mm", end='')

        # 可视化
        vis_img = sensor.visualize_gel_deformation(img)
        depth_map = sensor.height_map_2_depth_map(height_map)

        display = np.hstack([img, vis_img, cv2.cvtColor(depth_map, cv2.COLOR_GRAY2BGR)])
        cv2.imshow('采集中: 原始图 | 形变可视化 | 深度图', display)

        key = cv2.waitKey(1)
        if key == ord('q'):
            print("\n用户中断采集")
            break

    cv2.destroyAllWindows()
    print(f"\n\n数据采集完成！共采集 {count} 个样本")
    print(f"数据保存在: {save_dir}")
    return count


def generate_data_from_existing(image_dir, sensor_cfg, output_dir='./reconstruction_data',
                                 train_ratio=0.8):
    """
    方法2：从已有图像数据生成训练集
    如果你已经有采集的图像数据
    """
    print("\n=== 方法2：从已有图像生成数据集 ===")

    # 加载传感器（用于处理图像）
    sensor = Sensor(sensor_cfg, calibrated=True)

    # 查找图像文件
    image_files = []
    if os.path.exists(image_dir):
        for root, dirs, files in os.walk(image_dir):
            for file in files:
                if file.endswith(('.png', '.jpg', '.jpeg')):
                    image_files.append(os.path.join(root, file))

    if len(image_files) == 0:
        print(f"错误：在 {image_dir} 中未找到图像文件")
        return 0

    print(f"找到 {len(image_files)} 张图像")

    # 创建输出目录
    train_diff_dir = os.path.join(output_dir, 'train', 'diff_images')
    train_height_dir = os.path.join(output_dir, 'train', 'height_maps')
    val_diff_dir = os.path.join(output_dir, 'val', 'diff_images')
    val_height_dir = os.path.join(output_dir, 'val', 'height_maps')

    for dir_path in [train_diff_dir, train_height_dir, val_diff_dir, val_height_dir]:
        os.makedirs(dir_path, exist_ok=True)

    # 处理每张图像
    train_count = 0
    val_count = 0

    for img_path in tqdm(image_files, desc="处理图像"):
        try:
            # 读取图像
            img = cv2.imread(img_path)
            if img is None:
                continue

            # 裁剪和校正（如果需要）
            # img = sensor.rectify_image(img)  # 根据需要

            img_GRAY = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # 计算差分
            diff = sensor.ref_GRAY.astype(np.float32) - img_GRAY.astype(np.float32)
            diff = np.clip(diff, 0, 255).astype(np.float32)

            # 生成高度图
            height_map = sensor.raw_image_2_height_map(img_GRAY)

            # 过滤无接触数据
            if np.max(height_map) < 0.05:
                continue

            # 划分训练集/验证集
            is_train = np.random.rand() < train_ratio

            if is_train:
                np.save(os.path.join(train_diff_dir, f'{train_count:06d}.npy'), diff)
                np.save(os.path.join(train_height_dir, f'{train_count:06d}.npy'), height_map)
                train_count += 1
            else:
                np.save(os.path.join(val_diff_dir, f'{val_count:06d}.npy'), diff)
                np.save(os.path.join(val_height_dir, f'{val_count:06d}.npy'), height_map)
                val_count += 1

        except Exception as e:
            print(f"\n处理 {img_path} 时出错: {e}")
            continue

    print(f"\n数据生成完成！")
    print(f"训练集: {train_count} 个样本")
    print(f"验证集: {val_count} 个样本")
    print(f"数据保存在: {output_dir}")

    return train_count + val_count


def generate_synthetic_data(sensor_cfg, num_samples=1000, output_dir='./reconstruction_data'):
    """
    方法3：生成合成数据（快速测试用）
    使用高斯分布模拟接触
    """
    print("\n=== 方法3：生成合成测试数据 ===")
    print(f"生成 {num_samples} 个合成样本用于快速测试...")

    train_ratio = 0.8
    num_train = int(num_samples * train_ratio)
    num_val = num_samples - num_train

    # 创建目录
    train_diff_dir = os.path.join(output_dir, 'train', 'diff_images')
    train_height_dir = os.path.join(output_dir, 'train', 'height_maps')
    val_diff_dir = os.path.join(output_dir, 'val', 'diff_images')
    val_height_dir = os.path.join(output_dir, 'val', 'height_maps')

    for dir_path in [train_diff_dir, train_height_dir, val_diff_dir, val_height_dir]:
        os.makedirs(dir_path, exist_ok=True)

    # 图像尺寸（根据实际传感器调整）
    height, width = 150, 200  # 示例尺寸

    def generate_sample():
        # 随机生成接触位置和形状
        cx = np.random.randint(30, width - 30)
        cy = np.random.randint(30, height - 30)
        sigma_x = np.random.uniform(10, 30)
        sigma_y = np.random.uniform(10, 30)
        amplitude = np.random.uniform(1.0, 5.0)  # mm

        # 生成高度图（高斯分布）
        y, x = np.ogrid[0:height, 0:width]
        height_map = amplitude * np.exp(-((x - cx) ** 2 / (2 * sigma_x ** 2) +
                                          (y - cy) ** 2 / (2 * sigma_y ** 2)))
        height_map = height_map.astype(np.float32)

        # 生成对应的差分图像
        # 模拟：接触越深，灰度差越大
        diff = (height_map / amplitude) * 80 + np.random.randn(height, width) * 3
        diff = np.clip(diff, 0, 255).astype(np.float32)

        return diff, height_map

    # 生成训练集
    for i in tqdm(range(num_train), desc="生成训练集"):
        diff, height_map = generate_sample()
        np.save(os.path.join(train_diff_dir, f'{i:06d}.npy'), diff)
        np.save(os.path.join(train_height_dir, f'{i:06d}.npy'), height_map)

    # 生成验证集
    for i in tqdm(range(num_val), desc="生成验证集"):
        diff, height_map = generate_sample()
        np.save(os.path.join(val_diff_dir, f'{i:06d}.npy'), diff)
        np.save(os.path.join(val_height_dir, f'{i:06d}.npy'), height_map)

    print(f"\n合成数据生成完成！")
    print(f"训练集: {num_train} 个样本")
    print(f"验证集: {num_val} 个样本")
    print(f"数据保存在: {output_dir}")

    return num_samples


def split_train_val(data_dir='./reconstruction_data', train_ratio=0.8):
    """
    将数据划分为训练集和验证集
    如果数据都在一个目录下
    """
    print("\n=== 划分训练集和验证集 ===")

    source_diff_dir = os.path.join(data_dir, 'diff_images')
    source_height_dir = os.path.join(data_dir, 'height_maps')

    if not os.path.exists(source_diff_dir):
        print(f"错误：{source_diff_dir} 不存在")
        return

    # 获取所有文件
    diff_files = sorted([f for f in os.listdir(source_diff_dir) if f.endswith('.npy')])

    num_total = len(diff_files)
    num_train = int(num_total * train_ratio)

    # 随机打乱
    indices = np.random.permutation(num_total)
    train_indices = indices[:num_train]
    val_indices = indices[num_train:]

    # 创建目标目录
    train_diff_dir = os.path.join(data_dir, 'train', 'diff_images')
    train_height_dir = os.path.join(data_dir, 'train', 'height_maps')
    val_diff_dir = os.path.join(data_dir, 'val', 'diff_images')
    val_height_dir = os.path.join(data_dir, 'val', 'height_maps')

    for dir_path in [train_diff_dir, train_height_dir, val_diff_dir, val_height_dir]:
        os.makedirs(dir_path, exist_ok=True)

    # 复制文件
    import shutil

    for i, idx in enumerate(tqdm(train_indices, desc="复制到训练集")):
        filename = diff_files[idx]
        shutil.copy(os.path.join(source_diff_dir, filename),
                    os.path.join(train_diff_dir, f'{i:06d}.npy'))
        shutil.copy(os.path.join(source_height_dir, filename),
                    os.path.join(train_height_dir, f'{i:06d}.npy'))

    for i, idx in enumerate(tqdm(val_indices, desc="复制到验证集")):
        filename = diff_files[idx]
        shutil.copy(os.path.join(source_diff_dir, filename),
                    os.path.join(val_diff_dir, f'{i:06d}.npy'))
        shutil.copy(os.path.join(source_height_dir, filename),
                    os.path.join(val_height_dir, f'{i:06d}.npy'))

    print(f"\n划分完成！")
    print(f"训练集: {len(train_indices)} 个样本")
    print(f"验证集: {len(val_indices)} 个样本")


def main():
    parser = argparse.ArgumentParser(description='准备形状重建训练数据')
    parser.add_argument('--method', type=str, default='camera',
                        choices=['camera', 'existing', 'synthetic', 'split'],
                        help='数据准备方法')
    parser.add_argument('--num_samples', type=int, default=500,
                        help='采集/生成的样本数')
    parser.add_argument('--config', type=str, default='shape_config.yaml',
                        help='传感器配置文件')
    parser.add_argument('--image_dir', type=str, default='../Dataset/image',
                        help='已有图像目录（method=existing时使用）')
    parser.add_argument('--output_dir', type=str, default='./reconstruction_data',
                        help='输出数据目录')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='训练集比例')

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("形状重建数据准备工具")
    print("=" * 60)

    if args.method == 'camera':
        # 方法1：从摄像头采集
        print("\n加载传感器配置...")
        with open(args.config, 'r', encoding='utf-8') as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
        sensor = Sensor(cfg, calibrated=True)

        print("\n准备采集数据...")
        print("提示：")
        print("  1. 确保传感器已正确标定")
        print("  2. 使用多种物体（球形、圆柱、平面等）")
        print("  3. 尝试不同的接触位置和力度")
        print("  4. 采集过程中保持光照稳定")

        count = collect_data_from_camera(sensor, args.num_samples,
                                        os.path.join(args.output_dir, 'train'))

        # 自动划分一部分作为验证集
        if count > 10:
            print("\n自动生成验证集...")
            split_train_val(args.output_dir, args.train_ratio)

    elif args.method == 'existing':
        # 方法2：从已有图像生成
        with open(args.config, 'r', encoding='utf-8') as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)

        count = generate_data_from_existing(args.image_dir, cfg, args.output_dir,
                                           args.train_ratio)

    elif args.method == 'synthetic':
        # 方法3：生成合成数据
        count = generate_synthetic_data(args.config, args.num_samples, args.output_dir)

    elif args.method == 'split':
        # 划分已有数据
        split_train_val(args.output_dir, args.train_ratio)
        return

    print("\n" + "=" * 60)
    print("数据准备完成！")
    print("=" * 60)
    print(f"\n输出目录: {args.output_dir}")
    print("\n目录结构:")
    print("  reconstruction_data/")
    print("    ├── train/")
    print("    │   ├── diff_images/")
    print("    │   └── height_maps/")
    print("    └── val/")
    print("        ├── diff_images/")
    print("        └── height_maps/")
    print("\n下一步：运行训练脚本")
    print("  python shape_reconstruction/train_reconstruction.py")


if __name__ == '__main__':
    main()
