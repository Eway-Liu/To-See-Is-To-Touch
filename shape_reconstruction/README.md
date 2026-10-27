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

标定图片和 `.npy` 文件继续保存在原来的传感器目录。只有对应同一传感器、相机位置与光照设置的有效标定才能沿用。

## 启动命令

```bash
# 标定：先相机，再深度
python -m shape_reconstruction.calibration.camera_calibration
python -m shape_reconstruction.calibration.sensor_calibration

# 采集：同一实物沿用同一个 object_id；接触组由脚本自动记录
python -m shape_reconstruction.data.data_preparation --auto_ref --object_id steel_ball_8mm --output_dir reconstruction_data/raw

# 筛选配对数据并按接触组划分 train/val/test，不移动原始数据
python -m shape_reconstruction.data.prepare_dataset --data_dir reconstruction_data/raw --output reconstruction_data/splits.json

# Windows 4080 上训练：新残差注意力 U-Net，CUDA 自动混合精度
python -m shape_reconstruction.learning.train_model --manifest reconstruction_data/splits.json --model attention_unet --base_features 32 --batch_size 8 --epochs 120 --save_dir saved_models/attention_unet

# 以独立 test split 评估，并可对传统方法同帧比较（需采集 raw/reference 图）
python -m shape_reconstruction.learning.evaluation --manifest reconstruction_data/splits.json --split test --model_path saved_models/attention_unet/reconstruction_best.pth --compare_traditional

# 实时三维并排演示；需要传感器和 Open3D
python -m shape_reconstruction.demos.shape_reconstruction_demo --mode compare --model_path saved_models/attention_unet/reconstruction_best.pth --auto_ref

# 配置与导入回归检查，无需连接硬件
python -m unittest discover -s tests -v
```

`train_reconstruction` 是旧入口兼容层，和 `train_model` 使用一致的新检查点格式。旧 U-Net 纯权重仍可由推理器读取。在线三维演示需要安装 Open3D；离线回放/导出不需要相机。传统算法输出是伪标签时，指标表示网络对传统结果的拟合程度，不代表物理精度。

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
