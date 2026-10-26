"""
集成深度学习的传感器类
继承原有Sensor类，添加深度学习推理功能
"""
import os
import numpy as np
import cv2
import torch
from ..core.sensor import Sensor
from ..config import load_config
from .model import ReconstructionNet, SimpleReconstructionNet, ResNetReconstructionNet


class SensorDL(Sensor):
    """
    集成深度学习的传感器
    可以选择使用传统方法或深度学习方法进行形状重建
    """
    def __init__(self, cfg, model_path=None, use_dl=True,
                 device='cuda' if torch.cuda.is_available() else 'cpu', calibrated=True, ref=None):
        """
        Args:
            cfg: 传感器配置
            model_path: 深度学习模型路径
            use_dl: 是否使用深度学习方法
            device: 计算设备
            calibrated: 是否已标定
            ref: 参考图像
        """
        super().__init__(cfg, calibrated=calibrated, ref=ref)

        self.use_dl = use_dl
        self.device = torch.device(device)
        self.dl_model = None

        # 加载深度学习模型
        if use_dl and model_path:
            self.load_dl_model(model_path)

    def load_dl_model(self, model_path):
        """加载深度学习模型"""
        try:
            print(f"加载深度学习模型: {model_path}")

            # 加载检查点
            checkpoint = torch.load(model_path, map_location=self.device)
            model_type = checkpoint.get('model_type', 'unet')

            # 创建模型
            if model_type == 'unet':
                self.dl_model = ReconstructionNet(base_features=32)
            elif model_type == 'simple':
                self.dl_model = SimpleReconstructionNet()
            elif model_type == 'resnet':
                self.dl_model = ResNetReconstructionNet()
            else:
                raise ValueError(f"未知模型类型: {model_type}")

            # 加载权重
            self.dl_model.load_state_dict(checkpoint['model_state_dict'])
            self.dl_model = self.dl_model.to(self.device)
            self.dl_model.eval()

            print(f"✓ 深度学习模型加载成功 (类型: {model_type}, 设备: {self.device})")
            self.use_dl = True

        except Exception as e:
            print(f"✗ 加载深度学习模型失败: {e}")
            print("  回退到传统方法")
            self.dl_model = None
            self.use_dl = False

    def raw_image_2_height_map_dl(self, img_GRAY):
        """
        使用深度学习方法生成高度图
        Args:
            img_GRAY: 灰度图像
        Returns:
            height_map: 高度图
        """
        if self.dl_model is None:
            raise RuntimeError("深度学习模型未加载")

        # 计算差分
        diff = self.ref_GRAY.astype(np.float32) - img_GRAY.astype(np.float32)
        diff = np.clip(diff, 0, 255)

        # 预处理
        diff_norm = diff / 255.0
        diff_tensor = torch.from_numpy(diff_norm).float().unsqueeze(0).unsqueeze(0)
        diff_tensor = diff_tensor.to(self.device)

        # 推理
        with torch.no_grad():
            height_tensor = self.dl_model(diff_tensor)

        # 后处理
        height_map = height_tensor.squeeze().cpu().numpy()
        height_map = height_map * 10.0  # 反归一化
        height_map = np.clip(height_map, 0, 10)

        return height_map.astype(np.float32)

    def raw_image_2_height_map(self, img_GRAY):
        """
        重写父类方法，根据use_dl选择方法
        Args:
            img_GRAY: 灰度图像
        Returns:
            height_map: 高度图
        """
        if self.use_dl and self.dl_model is not None:
            return self.raw_image_2_height_map_dl(img_GRAY)
        else:
            return super().raw_image_2_height_map(img_GRAY)

    def set_method(self, use_dl):
        """
        切换方法
        Args:
            use_dl: True使用深度学习，False使用传统方法
        """
        if use_dl and self.dl_model is None:
            print("警告：深度学习模型未加载，无法切换")
            return False

        self.use_dl = use_dl
        method_name = "深度学习" if use_dl else "传统"
        print(f"切换到 {method_name} 方法")
        return True

    def get_current_method(self):
        """获取当前使用的方法"""
        if self.use_dl and self.dl_model is not None:
            return "deep_learning"
        else:
            return "traditional"


if __name__ == '__main__':
    # 测试集成
    print("测试SensorDL...")

    # 加载配置
    cfg = load_config()

    # 创建传感器（先不加载模型）
    sensor = SensorDL(cfg, model_path=None, use_dl=False)

    print(f"\n当前方法: {sensor.get_current_method()}")

    # 尝试加载模型
    model_path = './saved_models/reconstruction_best.pth'
    if os.path.exists(model_path):
        sensor.load_dl_model(model_path)
        sensor.set_method(True)
        print(f"当前方法: {sensor.get_current_method()}")
    else:
        print(f"\n模型文件不存在: {model_path}")
        print("使用传统方法")

    # 实时测试
    print("\n实时测试（按't'切换方法，按'q'退出）")

    while sensor.cap.isOpened():
        img = sensor.get_rectify_crop_image()
        img_GRAY = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 计时
        import time
        start = time.time()
        height_map = sensor.raw_image_2_height_map(img_GRAY)
        inference_time = (time.time() - start) * 1000

        # 可视化
        depth_map = sensor.height_map_2_depth_map(height_map)
        vis_img = sensor.visualize_gel_deformation(img)

        display = np.hstack([
            img,
            vis_img,
            cv2.cvtColor(depth_map, cv2.COLOR_GRAY2BGR)
        ])

        # 显示信息
        method = "DL" if sensor.use_dl else "Trad"
        cv2.putText(display, f'{method} | {inference_time:.1f}ms',
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow('SensorDL Test', display)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('t'):
            current = sensor.get_current_method()
            new_method = not (current == "deep_learning")
            sensor.set_method(new_method)

    cv2.destroyAllWindows()
