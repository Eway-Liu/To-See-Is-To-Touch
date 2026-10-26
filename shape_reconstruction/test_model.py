"""
测试脚本
测试训练好的深度学习模型并与传统方法对比
"""
import os
import sys
import time
import argparse
import cv2
import yaml
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from sensor import Sensor
from model import ReconstructionNet, SimpleReconstructionNet, ResNetReconstructionNet
from dataset import ShapeReconstructionDataset


class ReconstructionModel:
    """封装的推理模型"""
    def __init__(self, model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = torch.device(device)

        # 加载检查点
        checkpoint = torch.load(model_path, map_location=self.device)
        model_type = checkpoint.get('model_type', 'unet')

        # 创建模型
        if model_type == 'unet':
            self.model = ReconstructionNet(base_features=32)
        elif model_type == 'simple':
            self.model = SimpleReconstructionNet()
        elif model_type == 'resnet':
            self.model = ResNetReconstructionNet()
        else:
            raise ValueError(f"未知模型类型: {model_type}")

        # 加载权重
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()

        print(f"加载模型: {model_type}")
        print(f"检查点: {model_path}")
        print(f"设备: {self.device}")

    def predict(self, diff_image):
        """
        预测高度图
        Args:
            diff_image: numpy array, shape [H, W], 差分图像
        Returns:
            height_map: numpy array, shape [H, W], 高度图
        """
        # 预处理
        diff_norm = np.clip(diff_image, 0, 255) / 255.0
        diff_tensor = torch.from_numpy(diff_norm).float().unsqueeze(0).unsqueeze(0)
        diff_tensor = diff_tensor.to(self.device)

        # 推理
        with torch.no_grad():
            height_tensor = self.model(diff_tensor)

        # 后处理
        height_map = height_tensor.squeeze().cpu().numpy()
        height_map = height_map * 10.0  # 反归一化
        height_map = np.clip(height_map, 0, 10)

        return height_map


def evaluate_on_dataset(model, dataset, device, num_samples=None):
    """在数据集上评估模型"""
    print("\n在验证集上评估...")

    if num_samples is None:
        num_samples = len(dataset)
    else:
        num_samples = min(num_samples, len(dataset))

    total_mse = 0.0
    total_mae = 0.0
    total_rmse = 0.0

    model.model.eval()
    with torch.no_grad():
        for i in tqdm(range(num_samples), desc="评估"):
            diff_img, height_map_true = dataset[i]
            diff_img = diff_img.unsqueeze(0).to(device)
            height_map_true = height_map_true.unsqueeze(0).to(device)

            # 预测
            height_map_pred = model.model(diff_img)

            # 计算指标
            mse = ((height_map_pred - height_map_true) ** 2).mean().item()
            mae = torch.abs(height_map_pred - height_map_true).mean().item()
            rmse = np.sqrt(mse)

            total_mse += mse
            total_mae += mae
            total_rmse += rmse

    # 平均指标
    avg_mse = total_mse / num_samples
    avg_mae = total_mae / num_samples
    avg_rmse = total_rmse / num_samples

    print(f"\n评估结果 ({num_samples} 个样本):")
    print(f"  MSE:  {avg_mse:.6f}")
    print(f"  MAE:  {avg_mae:.6f}")
    print(f"  RMSE: {avg_rmse:.6f}")

    return {'mse': avg_mse, 'mae': avg_mae, 'rmse': avg_rmse}


def compare_methods(sensor, dl_model, num_samples=10, save_dir='./comparison_results'):
    """对比传统方法和深度学习方法"""
    print("\n对比传统方法和深度学习方法...")
    print(f"将采集 {num_samples} 个样本进行对比")
    print("请用物体接触传感器...")

    os.makedirs(save_dir, exist_ok=True)

    traditional_times = []
    dl_times = []
    traditional_heights = []
    dl_heights = []

    count = 0
    while count < num_samples and sensor.cap.isOpened():
        # 获取图像
        img = sensor.get_rectify_crop_image()
        img_GRAY = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 计算差分
        diff = sensor.ref_GRAY.astype(np.float32) - img_GRAY.astype(np.float32)

        # 只处理有接触的情况
        if np.sum(diff > 5) < 100:
            cv2.imshow('等待接触...', img)
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            continue

        # 方法1：传统方法
        start_time = time.time()
        height_map_trad = sensor.raw_image_2_height_map(img_GRAY)
        trad_time = (time.time() - start_time) * 1000  # ms

        # 方法2：深度学习方法
        start_time = time.time()
        height_map_dl = dl_model.predict(diff)
        dl_time = (time.time() - start_time) * 1000  # ms

        # 记录
        traditional_times.append(trad_time)
        dl_times.append(dl_time)
        traditional_heights.append(height_map_trad)
        dl_heights.append(height_map_dl)

        # 可视化对比
        depth_trad = sensor.height_map_2_depth_map(height_map_trad)
        depth_dl = sensor.height_map_2_depth_map(height_map_dl)

        # 拼接显示
        display = np.hstack([
            img,
            cv2.cvtColor(depth_trad, cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(depth_dl, cv2.COLOR_GRAY2BGR)
        ])

        # 添加文字
        cv2.putText(display, 'Original', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display, f'Traditional ({trad_time:.1f}ms)', (img.shape[1] + 10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display, f'Deep Learning ({dl_time:.1f}ms)', (2 * img.shape[1] + 10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow('对比：原始 | 传统方法 | 深度学习', display)

        # 保存
        save_path = os.path.join(save_dir, f'comparison_{count:03d}.png')
        cv2.imwrite(save_path, display)

        count += 1
        print(f"\r进度: {count}/{num_samples} | "
              f"传统: {trad_time:.1f}ms | DL: {dl_time:.1f}ms", end='')

        key = cv2.waitKey(1000)  # 等待1秒
        if key == ord('q'):
            break

    cv2.destroyAllWindows()

    if count == 0:
        print("\n未采集到数据")
        return

    # 统计结果
    print(f"\n\n对比结果 ({count} 个样本):")
    print(f"平均推理时间:")
    print(f"  传统方法: {np.mean(traditional_times):.2f} ± {np.std(traditional_times):.2f} ms")
    print(f"  深度学习: {np.mean(dl_times):.2f} ± {np.std(dl_times):.2f} ms")
    print(f"  加速比: {np.mean(traditional_times) / np.mean(dl_times):.2f}x")

    # 计算高度差异
    height_diffs = []
    for h_trad, h_dl in zip(traditional_heights, dl_heights):
        diff = np.abs(h_trad - h_dl).mean()
        height_diffs.append(diff)

    print(f"\n平均高度差异: {np.mean(height_diffs):.4f} mm")
    print(f"最大高度差异: {np.max(height_diffs):.4f} mm")

    print(f"\n对比图像保存在: {save_dir}")

    return {
        'traditional_times': traditional_times,
        'dl_times': dl_times,
        'height_diffs': height_diffs
    }


def realtime_demo(sensor, dl_model):
    """实时演示"""
    print("\n实时演示模式")
    print("按 't' 切换传统方法/深度学习方法")
    print("按 'q' 退出")

    use_dl = True

    while sensor.cap.isOpened():
        # 获取图像
        img = sensor.get_rectify_crop_image()
        img_GRAY = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 计算差分
        diff = sensor.ref_GRAY.astype(np.float32) - img_GRAY.astype(np.float32)

        # 生成高度图
        start_time = time.time()
        if use_dl:
            height_map = dl_model.predict(diff)
            method_name = "Deep Learning"
        else:
            height_map = sensor.raw_image_2_height_map(img_GRAY)
            method_name = "Traditional"
        inference_time = (time.time() - start_time) * 1000

        # 可视化
        depth_map = sensor.height_map_2_depth_map(height_map)
        vis_img = sensor.visualize_gel_deformation(img)

        # 拼接
        display = np.hstack([
            img,
            vis_img,
            cv2.cvtColor(depth_map, cv2.COLOR_GRAY2BGR)
        ])

        # 添加信息
        info_text = f'{method_name} | {inference_time:.1f}ms | FPS: {1000 / inference_time:.1f}'
        cv2.putText(display, info_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow('实时演示：原始 | 形变 | 深度', display)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('t'):
            use_dl = not use_dl

    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='测试形状重建模型')
    parser.add_argument('--model_path', type=str,
                        default='../saved_models/reconstruction_best.pth',
                        help='模型路径')
    parser.add_argument('--config', type=str, default='shape_config.yaml',
                        help='传感器配置文件')
    parser.add_argument('--mode', type=str, default='demo',
                        choices=['demo', 'evaluate', 'compare'],
                        help='测试模式')
    parser.add_argument('--data_dir', type=str, default='./reconstruction_data',
                        help='数据目录（evaluate模式）')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='对比样本数（compare模式）')
    parser.add_argument('--save_dir', type=str, default='./comparison_results',
                        help='结果保存目录')

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("测试形状重建模型")
    print("=" * 60)

    # 检查模型文件
    if not os.path.exists(args.model_path):
        print(f"\n错误：模型文件不存在: {args.model_path}")
        print("请先训练模型：")
        print("  python shape_reconstruction/train_reconstruction.py")
        return

    # 加载模型
    print("\n加载深度学习模型...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dl_model = ReconstructionModel(args.model_path, device=device)

    if args.mode == 'evaluate':
        # 在数据集上评估
        print("\n加载验证集...")
        val_dataset = ShapeReconstructionDataset(
            mode='val',
            data_dir=args.data_dir,
            augmentation=False
        )
        if len(val_dataset) == 0:
            print("错误：验证集为空")
            return

        evaluate_on_dataset(dl_model, val_dataset, torch.device(device))

    elif args.mode == 'compare':
        # 对比测试
        print("\n加载传感器...")
        with open(args.config, 'r', encoding='utf-8') as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
        sensor = Sensor(cfg, calibrated=True)

        compare_methods(sensor, dl_model, args.num_samples, args.save_dir)

    elif args.mode == 'demo':
        # 实时演示
        print("\n加载传感器...")
        with open(args.config, 'r', encoding='utf-8') as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
        sensor = Sensor(cfg, calibrated=True)

        realtime_demo(sensor, dl_model)

    print("\n测试完成！")


if __name__ == '__main__':
    main()
