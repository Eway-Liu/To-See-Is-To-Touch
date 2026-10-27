# 三维形状重建答辩流程

所有命令在仓库根目录执行。项目同时保留传统查找表方法和神经网络方法；两者可以接收同一帧差分图并在同一物理尺度下显示。

## 1. 随机生成图像和传统伪标签

无需连接摄像头，也无需准备任何图片。已有 120 条、24 组数据时，在原目录增量追加 2,880 条、576 组：

```bash
python -m shape_reconstruction.data.generate_pseudolabels \
  --synthetic --append --num_samples 2880 --samples_per_group 5 --seed 42 \
  --output_dir reconstruction_data/raw
```

最终共 3,000 条、600 组。`--num_samples` 是本次新增数量，不是目标总量，不要重复执行这次增量命令。如果从空目录开始，省略 `--append`，并使用 `--num_samples 3000`。

随机生成椭圆圆顶、高斯凸起、棱脊、平顶、圆环、胶囊、多边形、超椭圆，并以 2～4 个接触斑构成组合接触。位置覆盖边缘，随机改变大小、方向、曲率、光照梯度、暗角、纹理和噪声，并包含纯背景组。每组默认 5 帧，共享一个合成物体和形状组合。相同生成器版本、随机种子、起始组号和参数可复现数据。默认分辨率为配置中的 `300×400`。

高度伪标签由原有传统查找表和滤波流程计算。脚本保存有符号灰度差分、毫米高度伪标签、合成接触灰度图、参考图和元数据；来源明确标为 `synthetic_traditional_pseudolabel`。3,000 帧约 3.6 GB。增量模式保持旧文件不变，在同一目录中写入新编号；元数据记录生成器版本、组内形状参数和光照变化。

以后有真实图像时仍可用 `--input_dir` 模式，输入要求见 [模块说明](shape_reconstruction/README.md)。旧摄像头采集脚本已移除。

## 2. 筛选和划分

```bash
python -m shape_reconstruction.data.prepare_dataset \
  --data_dir reconstruction_data/raw \
  --output reconstruction_data/splits.json \
  --group_by contact --extend
```

`--extend` 保留原有训练、验证和测试归属，原子更新原清单；如果第一次创建清单，省略该参数。需要至少 3 个独立生成的接触组。同一组合成帧会放在同一个划分里。脚本只报告坏样本和重复输入，不删除原始数据。合成组标识和物体标识由生成器自动写入。

扩充后应在新的 `--save_dir` 中开始训练，不能使用原清单下的运行直接 `--resume`，因为数据指纹已改变。

## 3. Windows RTX 4080 环境

安装 Python 3.10 或 3.11，在 PyTorch 官方安装选择器中选择 Windows、pip、CUDA 版本并执行生成命令，然后：

```powershell
pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

`requirements.txt` 已包含实时窗口所需的 Open3D。将 `reconstruction_data/`（包含 `raw/` 和 `splits.json`）一起复制到 Windows，复制后用 `--data_dir` 指向包含 `diff_images/` 的 `reconstruction_data/raw/`。

## 4. 训练

稳定基线：

```powershell
python -m shape_reconstruction.learning.train_model `
  --manifest reconstruction_data/splits.json --data_dir reconstruction_data/raw `
  --model attention_unet --base_features 32 --batch_size 8 `
  --epochs 120 --workers 4 --save_dir saved_models/attention32
```

显存允许时的精度候选：

```powershell
python -m shape_reconstruction.learning.train_model `
  --manifest reconstruction_data/splits.json --data_dir reconstruction_data/raw `
  --model attention_unet --base_features 48 --batch_size 4 `
  --epochs 160 --workers 4 --save_dir saved_models/attention48
```

根据验证集 MAE 选择模型，最后只在独立 test 划分上评估一次。继续训练必须使用原 `--save_dir`、原检查点和原 manifest：

```powershell
python -m shape_reconstruction.learning.train_model `
  --manifest reconstruction_data/splits.json --data_dir reconstruction_data/raw `
  --model attention_unet --base_features 48 --batch_size 4 `
  --epochs 240 --save_dir saved_models/attention48 `
  --resume saved_models/attention48/latest.pth
```

## 5. 评估和独立参考

```powershell
python -m shape_reconstruction.learning.evaluation `
  --manifest reconstruction_data/splits.json --split test `
  --data_dir reconstruction_data/raw `
  --model_path saved_models/attention48/reconstruction_best.pth `
  --compare_traditional --output evaluation/attention48
```

输出 `metrics.json`、逐样本 CSV 和对比图。合成数据上的指标只能说明网络在合成分布中复现传统算法的程度，不能证明真实物理精度或真实传感器上的泛化能力。

获得真实接触图以后，可用已知半径球补一个独立几何参考。下面的球参考只适用于真实球接触图，不能用于随机合成的灰度斑。人工确认接触圆心和边界后执行：

```bash
python -m shape_reconstruction.learning.ball_reference \
  --raw reconstruction_data/raw/raw_images/000000.npy \
  --reference reconstruction_data/raw/reference_images/000000.npy \
  --output_dir reconstruction_data/ball_reference --sample ball000 \
  --center 200 150 --rim_radius_px 45 \
  --ball_radius_mm 4 --mm_per_pixel 0.05
```

先检查 `ball000_alignment.png` 的红色接触圆。它是几何参考，不是高精度计量设备的真值。

## 6. 同帧三维展示

实时摄像头对比：

```bash
python -m shape_reconstruction.demos.shape_reconstruction_demo \
  --mode compare --model_path saved_models/attention48/reconstruction_best.pth \
  --auto_ref --output demo_results
```

无相机时回放伪标签输出目录并循环：

```bash
python -m shape_reconstruction.demos.shape_reconstruction_demo \
  --mode compare --model_path saved_models/attention48/reconstruction_best.pth \
  --replay reconstruction_data/raw --loop
```

导出而不打开窗口：

```bash
python -m shape_reconstruction.demos.shape_reconstruction_demo \
  --mode compare --model_path saved_models/attention48/reconstruction_best.pth \
  --replay reconstruction_data/raw --headless --output demo_results
```

画面和 3D 导出使用同一 `mm/pixel` 与同一高度色标；传统算法仍使用原查找表实现。`--headless` 不需要 Open3D。

## 7. 答辩前检查

```bash
python -m unittest discover -s tests -v
python -m compileall -q shape_reconstruction
```

CUDA 速度和实时窗口需要在目标电脑现场确认。随机伪标签生成不需要输入图片、传感器或 Open3D，使用仓库已有标定。项目不附带训练好的检查点；合成数据可用于预训练和流程验证，真实传感器上的效果仍需实测。
