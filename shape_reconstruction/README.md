# Shape Reconstruction 模块

所有命令在仓库根目录执行，统一使用 `python -m`。原先顶层的脚本已经移动到对应子包，直接运行旧脚本路径的命令需要更新。

## 目录职责

| 目录 / 文件 | 内容 |
| --- | --- |
| `calibration/` | 相机、深度标定脚本，以及 `sensor_<id>/camera_calibration/`、`sensor_<id>/depth_calibration/` 下的原始图片和标定结果 |
| `configs/shape_config.yaml` | 相机、标定与重建参数 |
| `config.py` | 加载配置，以 YAML 所在目录解析相对标定路径 |
| `core/` | `Camera`、传统重建 `Sensor` |
| `data/` | 摄像头数据采集、数据质量检查；采集数据写入命令行指定目录 |
| `learning/` | 数据集、网络、训练、模型评估与 `SensorDL` |
| `visualization/` | Open3D 点云显示和 `sensor_obj/` 网格资源 |
| `demos/` | 实时三维形状重建演示 |

标定图片和 `.npy` 文件继续保存在原来的传感器目录，不需要重新标定。

## 启动命令

```bash
# 标定：先相机，再深度
python -m shape_reconstruction.calibration.camera_calibration
python -m shape_reconstruction.calibration.sensor_calibration

# 采集和检查
python -m shape_reconstruction.data.data_preparation --auto_ref
python -m shape_reconstruction.data.check_dataset --data_dir reconstruction_data/train --visualize

# 快速 U-Net 训练，对应 GUIDE.md 的流程
python -m shape_reconstruction.learning.train_reconstruction \
    --train_dir reconstruction_data/train --val_dir reconstruction_data/val --save_dir models

# 多模型训练与评估（unet / simple / resnet）
python -m shape_reconstruction.learning.train_model --model unet --save_dir saved_models
python -m shape_reconstruction.learning.test_model --mode evaluate \
    --model_path saved_models/reconstruction_best.pth

# 传统方法三维演示（需要 Open3D 和传感器）
python -m shape_reconstruction.demos.shape_reconstruction_demo

# 配置与导入回归检查，无需连接硬件
python -m unittest discover -s tests -v
```

两套训练入口保留各自的模型和检查点格式：`train_reconstruction` 输出 `models/best_model.pth` 的纯权重，配合 GUIDE.md 中生成的 `demo_reconstruction.py`；`train_model` 输出带 `model_type` 等信息的检查点，配合 `learning.test_model` / `SensorDL` 使用。两者不能直接混用。`train_model` 需要额外安装 `tensorboard`，三维演示需要额外安装 `open3d`。

## 配置与 Python 调用

```python
from shape_reconstruction import Camera, Sensor
from shape_reconstruction.config import load_config
from shape_reconstruction.learning.sensor_dl import SensorDL

cfg = load_config()
# cfg = load_config('/absolute/path/to/custom_config.yaml')
sensor = Sensor(cfg)
```

`from shape_reconstruction import Camera, Sensor, Visualizer` 公共接口继续可用；子模块改用新路径，例如 `shape_reconstruction.learning.model`。普通导入、离线数据检查和训练不会自动加载 Open3D。

默认配置从包内固定位置读取。`calibration_root_dir` 的相对路径以配置文件目录为基准，绝对路径直接使用；网格文件按 `visualizer.py` 的位置定位。数据、模型、日志和结果的命令行路径仍相对于当前工作目录，已有文件可通过对应参数继续使用。
