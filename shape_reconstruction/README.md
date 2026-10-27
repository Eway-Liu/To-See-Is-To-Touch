# Shape Reconstruction 模块

所有命令在仓库根目录执行，统一使用 `python -m`。原先顶层的脚本已经移动到对应子包，直接运行旧脚本路径的命令需要更新。

## 目录职责

| 目录 / 文件 | 内容 |
| --- | --- |
| `calibration/` | 相机、深度标定脚本，以及 `sensor_<id>/camera_calibration/`、`sensor_<id>/depth_calibration/` 下的原始图片和标定结果 |
| `configs/shape_config.yaml` | 相机、标定与重建参数 |
| `config.py` | 加载配置，以 YAML 所在目录解析相对标定路径 |
| `core/` | `Camera`、传统重建 `Sensor` |
| `data/` | 随机合成图像或读取已有图像，生成传统方法伪标签、检查质量与划分数据集 |
| `learning/` | 数据集、网络、训练、模型评估与 `SensorDL` |
| `visualization/` | Open3D 点云显示和 `sensor_obj/` 网格资源 |
| `demos/` | 实时三维形状重建演示 |

标定图片和 `.npy` 文件继续保存在原来的传感器目录。只有对应同一传感器、相机位置与光照设置的有效标定才能沿用。

## 启动命令

```bash
# 从空目录开始：随机生成 3,000 条、600 组合成数据，并用现有传统标定生成伪标签
python -m shape_reconstruction.data.generate_pseudolabels --synthetic --num_samples 3000 --samples_per_group 5 --seed 42 --output_dir reconstruction_data/raw

# 筛选配对数据并按接触组划分 train/val/test，不移动原始数据
python -m shape_reconstruction.data.prepare_dataset --data_dir reconstruction_data/raw --output reconstruction_data/splits.json

# Windows 4080 上训练：新残差注意力 U-Net，CUDA 自动混合精度
python -m shape_reconstruction.learning.train_model --manifest reconstruction_data/splits.json --model attention_unet --base_features 32 --batch_size 8 --epochs 120 --save_dir saved_models/attention_unet

# 以独立 test split 评估，并可用保存的 raw/reference 图对传统方法同帧比较
python -m shape_reconstruction.learning.evaluation --manifest reconstruction_data/splits.json --split test --model_path saved_models/attention_unet/reconstruction_best.pth --compare_traditional

# 实时三维并排演示；需要传感器和 Open3D
python -m shape_reconstruction.demos.shape_reconstruction_demo --mode compare --model_path saved_models/attention_unet/reconstruction_best.pth --auto_ref

# 配置与导入回归检查，无需连接硬件
python -m unittest discover -s tests -v
```

`train_reconstruction` 是旧入口兼容层，和 `train_model` 使用一致的新检查点格式。旧 U-Net 纯权重仍可由推理器读取。在线三维演示需要安装 Open3D；离线回放/导出不需要相机。传统算法输出是伪标签时，指标表示网络对传统结果的拟合程度，不代表物理精度。

## 没有图像：随机合成

`--synthetic` 自动生成无接触参考图，以及椭圆圆顶、高斯凸起、棱脊、平顶、圆环、胶囊、多边形、超椭圆八类接触灰度图。每组随机生成一个或 2～4 个接触斑，支持取最大值或相加后限幅的组合接触；位置覆盖图像边缘，随机改变尺寸、方向、曲率、对比度、背景纹理、光照梯度、暗角和噪声。每 10 组合成接触中有一组纯背景。分辨率来自配置中的 `crop_size`，当前为 `300×400`，无需先采集或矫正图片。

先生成图像，再调用与实时重建相同的传统查找表和滤波算法计算高度。所有高度标签都来自传统算法，合成轮廓不作为几何真值。使用仓库已有的深度查找表和像素尺度文件，无需相机或重新标定。

`--num_samples` 表示本次生成的新增样本数，默认 120；`--samples_per_group` 默认 5，`--seed` 默认 42。相同生成器版本、随机种子、起始组号和参数可复现数据。每组共享一个合成物体、接触形状组合和参考图，组内帧只有小幅位移、缩放、光照漂移、对比度及噪声变化。`prepare_dataset` 按组划分，避免把同一接触的相似帧分到训练集和测试集。至少生成 3 组才能划分。

元数据自动写入 `data_source=synthetic`、`label_source=synthetic_traditional_pseudolabel`、`generator_version=2`、随机种子、每个接触斑的参数、光照参数、帧内变化和分组。120 条数据约占 144 MB，3,000 条约占 3.6 GB，生成物已由 Git 忽略。旧版 120 条数据继续保留，使用新版生成器不能仅凭旧种子重新生成旧版图像。

### 在原数据目录增量补充

已有 120 条、24 组时，以下命令在同一个目录新增 2,880 条、576 组，达到 3,000 条、600 组；不会复制或重写原有样本：

```bash
python -m shape_reconstruction.data.generate_pseudolabels --synthetic --append --num_samples 2880 --samples_per_group 5 --seed 42 --output_dir reconstruction_data/raw
python -m shape_reconstruction.data.prepare_dataset --data_dir reconstruction_data/raw --output reconstruction_data/splits.json --extend
```

`--append` 会检查已有合成元数据、四类数组是否配齐、像素尺度和重建参数，再按现有样本数、组数分配新编号。不带 `--append` 时输出目录必须不存在。不要对已经扩到 3,000 条的数据再次执行上面的 2,880 条追加命令，除非确实需要继续增量。

`--extend` 全量筛选后原子更新原清单，检查原样本记录是否变化，并保留原样本和接触组的 train/val/test 归属；只将新组分配到各划分。扩充后应使用新的训练保存目录，原训练运行的 manifest 指纹已改变，不能直接 `--resume`。

合成数据适合跑通训练、预训练和演示。该生成器使用程序化灰度图，不是完整的触觉传感器光学仿真；合成测试集上的指标只能说明在该分布上复现传统算法的程度，真实传感器泛化能力需要真实接触图验证。

## 离线伪标签输入

以后有真实图像时，也可以继续使用：

```bash
python -m shape_reconstruction.data.generate_pseudolabels --input_dir recorded/touch001 --reference recorded/reference.png --group touch001 --object_id steel_ball_8mm --output_dir reconstruction_data/real
```

`--input_dir` 下的 `raw_images/` 接收 `.npy`、PNG、JPEG、BMP、TIFF 图像。输入必须已经按当前传感器标定完成矫正和裁剪；参考图必须来自相同相机位置、光照和无接触状态。像素为 `uint8`，彩色图会转换为灰度；不对原始相机画面自动矫正，也不将浮点输入强制转为整数。

所有帧共用参考图时传入 `--reference`；参考随帧变化时，省略该参数并将参考图放入 `reference_images/`，文件名与 `raw_images/` 完全一致。原 `data_preparation` 摄像头采集入口及 `ShapeReconstructionDatasetFromSensor` 已移除。

每次接触的全部帧使用同一个 `--group`，不同接触的标识必须不同；同一个物体沿用同一个 `--object_id`。已有多接触数据可在输入目录放置 `samples.jsonl`，每行包含 `file`、`group`、`object_id`，此时无需传这两个参数。例如：

```json
{"file":"touch001_frame001.png","group":"touch001","object_id":"steel_ball_8mm"}
{"file":"touch002_frame001.png","group":"touch002","object_id":"steel_ball_8mm"}
```

输出保持现有的 `diff_images/`、`height_maps/`、`raw_images/`、`reference_images/` 和 `samples.jsonl` 格式，图像统一保存为同名 `.npy`。差分为有符号 `float32`，高度为毫米单位的 `float32`，来源标记为 `traditional_pseudolabel`。可将不同接触追加到同一输出目录，但输入文件名去掉扩展名后必须唯一；已有样本不会被覆盖。一个批次中途遇到坏图会报错并停止，之前已成功保存的样本会保留。

至少准备 3 个真实独立接触组，再执行 `prepare_dataset` 划分。不会把逐帧编号当成独立接触组。

真实传感器需要重新标定时，按顺序运行 `python -m shape_reconstruction.calibration.camera_calibration` 和 `python -m shape_reconstruction.calibration.sensor_calibration`。

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
