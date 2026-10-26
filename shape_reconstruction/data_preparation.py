"""
数据准备脚本 - 从摄像头实时采集训练数据
使用传统方法生成的高度图作为伪标签
"""
import os
import sys
import cv2
import yaml
import numpy as np
import time
import argparse
from glob import glob

# 添加路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
from shape_reconstruction import Sensor


def get_existing_count(save_dir):
    """获取已有数据集的数量"""
    diff_dir = os.path.join(save_dir, 'diff_images')
    if not os.path.exists(diff_dir):
        return 0
    existing_files = glob(os.path.join(diff_dir, '*.npy'))
    return len(existing_files)


def collect_data_from_camera(sensor, save_dir='./reconstruction_data/train'):
    """
    从摄像头实时采集数据
    按 'q' 键退出
    """
    diff_dir = os.path.join(save_dir, 'diff_images')
    height_dir = os.path.join(save_dir, 'height_maps')
    os.makedirs(diff_dir, exist_ok=True)
    os.makedirs(height_dir, exist_ok=True)

    # 获取已有数据数量，从下一个编号开始
    start_count = get_existing_count(save_dir)
    count = start_count
    new_samples = 0

    print(f"\n已有数据: {start_count} 个样本")
    print(f"新数据将从编号 {start_count:06d} 开始保存")
    print("\n=== 开始采集数据 ===")
    print("操作说明：")
    print("  - 用不同物体接触传感器表面")
    print("  - 尝试不同的接触位置、角度和力度")
    print("  - 按 'q' 键退出并保存")
    print("  - 按 's' 键手动保存当前帧")
    print("\n采集中...")

    last_saved_height = None

    while sensor.cap.isOpened():
        # 获取图像
        img = sensor.get_rectify_crop_image()
        if img is None:
            continue

        img_GRAY = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 计算差分图像
        diff_raw = sensor.ref_GRAY.astype(np.float32) - img_GRAY.astype(np.float32)
        diff = np.clip(diff_raw, 0, 255).astype(np.float32)

        # 生成高度图（传统方法作为伪标签）
        height_map = sensor.raw_image_2_height_map(img_GRAY)

        # 检测接触
        contact_threshold = 0.05  # mm
        max_height = np.max(height_map)

        # 可视化
        vis_img = sensor.visualize_gel_deformation(img)
        depth_map = sensor.height_map_2_depth_map(height_map)
        display = np.hstack([img, vis_img, cv2.cvtColor(depth_map, cv2.COLOR_GRAY2BGR)])

        # 显示状态信息
        status_text = f"Total: {count} | New: {new_samples} | Height: {max_height:.3f}mm | Press 'q' to quit, 's' to save"
        cv2.putText(display, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                   0.6, (0, 255, 0), 2)

        cv2.imshow('Data Collection: Original | Deformation | Depth', display)

        key = cv2.waitKey(1)

        # 手动保存
        if key == ord('s') and max_height > contact_threshold:
            np.save(os.path.join(diff_dir, f'{count:06d}.npy'), diff)
            np.save(os.path.join(height_dir, f'{count:06d}.npy'), height_map)
            count += 1
            new_samples += 1
            print(f"\r手动保存: 总计 {count} | 新增 {new_samples} | 最大高度: {max_height:.3f}mm", end='')
            last_saved_height = height_map.copy()

        # 自动保存有效接触
        elif max_height > contact_threshold:
            # 避免保存相似的连续帧
            is_different = True
            if last_saved_height is not None:
                height_diff = np.abs(height_map - last_saved_height).mean()
                if height_diff < 0.01:  # 变化太小
                    is_different = False

            if is_different:
                np.save(os.path.join(diff_dir, f'{count:06d}.npy'), diff)
                np.save(os.path.join(height_dir, f'{count:06d}.npy'), height_map)
                last_saved_height = height_map.copy()
                count += 1
                new_samples += 1
                print(f"\r自动保存: 总计 {count} | 新增 {new_samples} | 最大高度: {max_height:.3f}mm", end='')

        # 退出
        if key == ord('q'):
            print("\n\n用户退出采集")
            break

    cv2.destroyAllWindows()
    print(f"\n\n{'='*60}")
    print(f"数据采集完成！")
    print(f"{'='*60}")
    print(f"原有样本: {start_count}")
    print(f"新增样本: {new_samples}")
    print(f"总计样本: {count}")
    print(f"数据保存在: {save_dir}")
    return new_samples


def main():
    parser = argparse.ArgumentParser(description='从摄像头采集形状重建训练数据')
    parser.add_argument('--config', type=str, default='shape_config.yaml',
                        help='传感器配置文件')
    parser.add_argument('--output_dir', type=str, default='./reconstruction_data/train',
                        help='输出数据目录')
    parser.add_argument('--auto_ref', action='store_true',
                        help='自动捕获参考图像（3秒倒计时）')

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("形状重建数据采集工具")
    print("=" * 60)

    print("\n加载传感器配置...")
    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    print("\n初始化传感器...")

    if args.auto_ref:
        # 自动捕获参考图像
        print("自动捕获模式：")
        print("确保传感器表面无任何物体接触...")
        temp_camera = Sensor.__bases__[0](cfg, calibrated=True)  # 初始化Camera部分

        for i in range(3, 0, -1):
            print(f"  {i}秒后自动捕获参考图像...")
            time.sleep(1)

        # 捕获参考图像
        print("  正在捕获参考图像...")
        img_add = np.zeros_like(temp_camera.get_rectify_crop_image(), float)
        for _ in range(10):
            img_add += temp_camera.get_rectify_crop_image()
        ref = (img_add / 10).astype(np.uint8)

        # 使用捕获的参考图像初始化传感器
        sensor = Sensor(cfg, calibrated=True, ref=ref)
        print("✓ 参考图像已自动捕获！")
    else:
        # 交互式捕获参考图像
        print("提示：将显示摄像头画面，请按 'y' 确认参考图像（无接触状态），或按 'q' 退出")
        print("重要：确保传感器表面无任何物体接触，画面稳定后按 'y'")
        sensor = Sensor(cfg, calibrated=True)
        print("✓ 参考图像已确认！")

    print("\n准备采集数据...")
    print("提示：")
    print("  1. 使用多种物体（球形、圆柱、平面等）")
    print("  2. 尝试不同的接触位置、角度和力度")
    print("  3. 采集过程中保持光照稳定")
    print("  4. 数据会自动在原有基础上累加")

    count = collect_data_from_camera(sensor, args.output_dir)

    print("\n" + "=" * 60)
    print(f"本次新增 {count} 个样本")
    print("=" * 60)
    print(f"\n数据保存在: {args.output_dir}")
    print("\n目录结构:")
    print(f"  {args.output_dir}/")
    print("    ├── diff_images/")
    print("    └── height_maps/")
    print("\n提示：")
    print("  - 可以多次运行此脚本继续采集数据")
    print("  - 建议至少采集 500-1000 个样本用于训练")
    print("  - 采集完成后运行训练脚本：")
    print("    python shape_reconstruction/train_reconstruction.py")


if __name__ == '__main__':
    main()

