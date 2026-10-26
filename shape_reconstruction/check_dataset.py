"""
数据集质量检查脚本
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import argparse


def analyze_dataset(data_dir='./reconstruction_data/train'):
    """分析数据集统计信息"""

    diff_dir = os.path.join(data_dir, 'diff_images')
    height_dir = os.path.join(data_dir, 'height_maps')

    if not os.path.exists(diff_dir):
        print(f"错误：{diff_dir} 不存在")
        return

    diff_files = sorted(glob(os.path.join(diff_dir, '*.npy')))
    height_files = sorted(glob(os.path.join(height_dir, '*.npy')))

    print("\n" + "="*60)
    print("数据集统计信息")
    print("="*60)
    print(f"数据目录: {data_dir}")
    print(f"样本数量: {len(diff_files)}")

    if len(diff_files) == 0:
        print("警告：没有找到数据文件")
        return

    # 统计高度信息
    max_heights = []
    mean_heights = []
    contact_areas = []

    print("\n正在分析数据...")
    for height_file in height_files[:min(100, len(height_files))]:  # 采样最多100个
        height_map = np.load(height_file)
        max_heights.append(np.max(height_map))
        mean_heights.append(np.mean(height_map[height_map > 0]))
        contact_areas.append(np.sum(height_map > 0.05))  # 接触阈值 0.05mm

    max_heights = np.array(max_heights)
    mean_heights = np.array(mean_heights)
    contact_areas = np.array(contact_areas)

    print("\n高度统计 (mm):")
    print(f"  最大高度范围: {max_heights.min():.3f} - {max_heights.max():.3f}")
    print(f"  平均最大高度: {max_heights.mean():.3f} ± {max_heights.std():.3f}")
    print(f"  接触区域均值: {np.mean(mean_heights):.3f} mm")

    print("\n接触区域统计 (像素):")
    print(f"  接触面积范围: {contact_areas.min():.0f} - {contact_areas.max():.0f}")
    print(f"  平均接触面积: {contact_areas.mean():.0f} ± {contact_areas.std():.0f}")

    # 数据质量检查
    print("\n数据质量检查:")
    low_quality = np.sum(max_heights < 0.1)
    medium_quality = np.sum((max_heights >= 0.1) & (max_heights < 1.0))
    high_quality = np.sum(max_heights >= 1.0)

    total = len(max_heights)
    print(f"  轻触样本 (<0.1mm): {low_quality} ({low_quality/total*100:.1f}%)")
    print(f"  中等样本 (0.1-1mm): {medium_quality} ({medium_quality/total*100:.1f}%)")
    print(f"  深度样本 (>1mm): {high_quality} ({high_quality/total*100:.1f}%)")

    # 建议
    print("\n数据质量建议:")
    if total < 500:
        print("  ⚠️  样本数量较少，建议继续采集到至少 500 个")
    elif total < 1000:
        print("  ✓  样本数量基本够用，建议继续采集到 1000+ 以提升效果")
    else:
        print("  ✓  样本数量充足")

    if low_quality > total * 0.6:
        print("  ⚠️  轻触样本过多，建议增加中等和深度接触的样本")

    if high_quality < total * 0.1:
        print("  ⚠️  深度接触样本较少，建议增加更大压力的接触")

    if contact_areas.std() / contact_areas.mean() < 0.3:
        print("  ⚠️  接触面积变化较小，建议使用更多不同大小的物体")

    return max_heights, mean_heights, contact_areas


def visualize_samples(data_dir='./reconstruction_data/train', num_samples=9):
    """可视化随机样本"""

    diff_dir = os.path.join(data_dir, 'diff_images')
    height_dir = os.path.join(data_dir, 'height_maps')

    diff_files = sorted(glob(os.path.join(diff_dir, '*.npy')))

    if len(diff_files) == 0:
        print("没有找到数据文件")
        return

    # 随机选择样本
    indices = np.random.choice(len(diff_files), min(num_samples, len(diff_files)), replace=False)

    fig, axes = plt.subplots(3, 6, figsize=(18, 9))
    fig.suptitle('数据集样本展示 (左: 差分图像, 右: 高度图)', fontsize=14)

    for i, idx in enumerate(indices):
        row = i // 3
        col_base = (i % 3) * 2

        # 加载数据
        diff = np.load(diff_files[idx])
        height = np.load(diff_files[idx].replace('diff_images', 'height_maps'))

        # 显示差分图像
        axes[row, col_base].imshow(diff, cmap='gray')
        axes[row, col_base].set_title(f'Sample {idx}')
        axes[row, col_base].axis('off')

        # 显示高度图
        im = axes[row, col_base + 1].imshow(height, cmap='jet')
        axes[row, col_base + 1].set_title(f'Max: {height.max():.2f}mm')
        axes[row, col_base + 1].axis('off')
        plt.colorbar(im, ax=axes[row, col_base + 1], fraction=0.046)

    plt.tight_layout()
    output_path = 'dataset_samples.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n样本可视化已保存: {output_path}")
    plt.close()


def plot_statistics(max_heights, mean_heights, contact_areas):
    """绘制统计图表"""

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 最大高度分布
    axes[0, 0].hist(max_heights, bins=30, edgecolor='black', alpha=0.7)
    axes[0, 0].set_xlabel('最大高度 (mm)')
    axes[0, 0].set_ylabel('样本数量')
    axes[0, 0].set_title('最大高度分布')
    axes[0, 0].axvline(max_heights.mean(), color='r', linestyle='--',
                      label=f'均值: {max_heights.mean():.2f}mm')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    # 平均高度分布
    axes[0, 1].hist(mean_heights, bins=30, edgecolor='black', alpha=0.7, color='orange')
    axes[0, 1].set_xlabel('接触区域平均高度 (mm)')
    axes[0, 1].set_ylabel('样本数量')
    axes[0, 1].set_title('接触区域平均高度分布')
    axes[0, 1].axvline(mean_heights.mean(), color='r', linestyle='--',
                      label=f'均值: {mean_heights.mean():.2f}mm')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    # 接触面积分布
    axes[1, 0].hist(contact_areas, bins=30, edgecolor='black', alpha=0.7, color='green')
    axes[1, 0].set_xlabel('接触面积 (像素)')
    axes[1, 0].set_ylabel('样本数量')
    axes[1, 0].set_title('接触面积分布')
    axes[1, 0].axvline(contact_areas.mean(), color='r', linestyle='--',
                      label=f'均值: {contact_areas.mean():.0f}px')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    # 高度 vs 面积散点图
    axes[1, 1].scatter(contact_areas, max_heights, alpha=0.5, s=20)
    axes[1, 1].set_xlabel('接触面积 (像素)')
    axes[1, 1].set_ylabel('最大高度 (mm)')
    axes[1, 1].set_title('接触面积 vs 最大高度')
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    output_path = 'dataset_statistics.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"统计图表已保存: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='检查数据集质量')
    parser.add_argument('--data_dir', type=str, default='./reconstruction_data/train',
                        help='数据集目录')
    parser.add_argument('--visualize', action='store_true',
                        help='生成可视化图表')

    args = parser.parse_args()

    # 分析数据集
    result = analyze_dataset(args.data_dir)

    if result is None:
        return

    max_heights, mean_heights, contact_areas = result

    # 可视化
    if args.visualize:
        print("\n生成可视化...")
        visualize_samples(args.data_dir, num_samples=9)
        plot_statistics(max_heights, mean_heights, contact_areas)
        print("\n可视化完成！")
        print("  - dataset_samples.png: 随机样本展示")
        print("  - dataset_statistics.png: 统计分布图表")


if __name__ == '__main__':
    main()
