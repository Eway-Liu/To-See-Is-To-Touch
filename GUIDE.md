# 三维形状重建答辩流程

所有命令在仓库根目录执行。项目同时保留传统查找表方法和神经网络方法；两者可以接收同一帧差分图并在同一物理尺度下显示。

## 1. 采集

传感器电脑需要相机和现有标定文件。每个实物保持一个 `object_id`：

```bash
python -m shape_reconstruction.data.data_preparation \
  --auto_ref --object_id ball_R8mm \
  --output_dir reconstruction_data/raw --max_samples 300
```

采集时让物体完全离开传感器，再进行下一次接触。脚本保存有符号灰度差分、传统高度伪标签、原始灰度图、参考图和元数据。按 `q` 结束。

## 2. 筛选和划分

```bash
python -m shape_reconstruction.data.prepare_dataset \
  --data_dir reconstruction_data/raw \
  --output reconstruction_data/splits.json \
  --group_by contact
```

需要至少 3 个独立接触组。更可靠的答辩实验是采集多个物体后使用 `--group_by object`，并确保每个样本都有真实的 `object_id`。脚本只报告坏样本和重复输入，不删除原始数据。

## 3. Windows RTX 4080 环境

安装 Python 3.10 或 3.11，在 PyTorch 官方安装选择器中选择 Windows、pip、CUDA 版本并执行生成命令，然后：

```powershell
pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

实时 Open3D 窗口还需要 `pip install -r requirements-demo.txt`。将 `reconstruction_data/` 和 `splits.json` 一起复制到 Windows，复制后用 `--data_dir` 指向 Windows 数据根目录。

## 4. 训练

稳定基线：

```powershell
python -m shape_reconstruction.learning.train_model `
  --manifest reconstruction_data/splits.json --data_dir reconstruction_data `
  --model attention_unet --base_features 32 --batch_size 8 `
  --epochs 120 --workers 4 --save_dir saved_models/attention32
```

显存允许时的精度候选：

```powershell
python -m shape_reconstruction.learning.train_model `
  --manifest reconstruction_data/splits.json --data_dir reconstruction_data `
  --model attention_unet --base_features 48 --batch_size 4 `
  --epochs 160 --workers 4 --save_dir saved_models/attention48
```

根据验证集 MAE 选择模型，最后只在独立 test 划分上评估一次。继续训练必须使用原 `--save_dir`、原检查点和原 manifest：

```powershell
python -m shape_reconstruction.learning.train_model `
  --manifest reconstruction_data/splits.json --data_dir reconstruction_data `
  --model attention_unet --base_features 48 --batch_size 4 `
  --epochs 240 --save_dir saved_models/attention48 `
  --resume saved_models/attention48/latest.pth
```

## 5. 评估和独立参考

```powershell
python -m shape_reconstruction.learning.evaluation `
  --manifest reconstruction_data/splits.json --split test `
  --data_dir reconstruction_data `
  --model_path saved_models/attention48/reconstruction_best.pth `
  --compare_traditional --output evaluation/attention48
```

输出 `metrics.json`、逐样本 CSV 和对比图。当前标签来自传统算法时，指标只能说明网络复现传统算法的程度，不能证明真实物理精度提升。

可用已知半径球补一个独立几何参考。人工确认接触圆心和边界后执行：

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

无相机时回放采集目录并循环：

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

当前机器没有 Windows 4080、传感器和 Open3D，因此 CUDA 速度、相机采集和实时窗口必须在目标电脑现场确认。项目不附带训练好的检查点；没有真实数据前，不应宣称神经网络已提升物理重建精度。
