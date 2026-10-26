# 触觉传感器形状重建 - 完整指南

## 📋 目录

1. [项目概述](#项目概述)
2. [环境配置](#环境配置)
3. [72小时执行计划](#72小时执行计划)
4. [数据采集指南](#数据采集指南)
5. [模型训练](#模型训练)
6. [常见问题](#常见问题)

---

## 项目概述

本项目使用深度学习方法从触觉传感器的差分图像预测3D形状（高度图）。

### 方案选择

**采用方案：传统方法伪标签训练**
- 使用现有的传统算法（查找表方法）生成训练标签
- 快速生成大量训练数据
- 虽然有性能上限，但实现快速、成本低

### 数据流程

```
触觉传感器 → 差分图像 → 传统算法 → 高度图（伪标签）
                ↓
            差分图像 → 神经网络 → 高度图（预测）
```

---

## 环境配置

### 快速安装

```bash
# 方法1：pip安装（推荐）
pip install torch torchvision opencv-python numpy matplotlib pyyaml tqdm scipy

# 方法2：使用requirements.txt
pip install -r requirements.txt

# 方法3：使用conda
conda env create -f environment.yml
conda activate tactile_reconstruction
```

### 验证安装

```bash
python -c "import torch; import cv2; import numpy; print('✓ 所有依赖安装成功')"
```

### 依赖列表

- **PyTorch** >= 2.0.0 - 深度学习框架
- **OpenCV** >= 4.8.0 - 图像处理
- **NumPy** >= 1.24.0 - 数值计算
- **Matplotlib** >= 3.7.0 - 可视化
- **PyYAML** >= 6.0 - 配置文件
- **tqdm** >= 4.65.0 - 进度条

---

## 72小时执行计划

### ⏰ 时间分配

| 时间段 | 任务 | 预计时间 |
|--------|------|---------|
| **Day 1 (0-24h)** | 数据采集 | 8-10小时 |
| **Day 2 (24-48h)** | 模型训练 | 8-10小时 |
| **Day 3 (48-72h)** | 测试优化 | 4-6小时 |

### Day 1：数据采集（目标：1000-1500样本）

#### 准备工作（30分钟）

```bash
# 测试采集脚本
cd shape_reconstruction
python data_preparation.py --auto_ref --output_dir ./test_data

# 采集50-100个样本测试，按'q'退出
# 检查数据质量
python check_dataset.py --data_dir ./test_data --visualize

# 确认正常后删除
rm -rf ./test_data
```

#### 准备物体

至少准备 **8-10 种不同物体**：
- **球形**：弹珠、乒乓球、玻璃球（不同大小）
- **圆柱**：笔、螺丝刀、管道
- **平面**：手指、卡片、积木
- **不规则**：钥匙、橡皮、硬币

#### 正式采集（6-8小时）

```bash
# 开始采集
python data_preparation.py --auto_ref --output_dir ./reconstruction_data/train
```

**采集策略：**
- 目标：1000-1500 个有效样本
- 速度：每小时 150-200 个样本
- 每种物体：100-150 个样本

**系统性采集（每个物体30-40分钟）：**
1. 不同位置：中心、边缘、各个区域（50个）
2. 不同力度：轻触、中等、深压（50个）
3. 不同角度：垂直、倾斜、滚动（50个）

**关键技巧：**
- 接触 → 停留1秒 → 移开 → 换位置 → 重复
- 每200个样本休息5分钟
- 观察屏幕计数器确保样本在增加

#### 中途检查（每2小时）

```bash
# 查看已采集数量
ls reconstruction_data/train/diff_images/ | wc -l

# 检查数据质量
python check_dataset.py --data_dir reconstruction_data/train --visualize
```

### Day 2：模型训练

#### 数据准备（1小时）

```bash
# 划分训练集和验证集（80/20）
python -c "
import os, shutil, numpy as np
from glob import glob

diff_files = sorted(glob('reconstruction_data/train/diff_images/*.npy'))
total = len(diff_files)
val_size = int(total * 0.2)

print(f'总样本数: {total}')
print(f'训练集: {total - val_size}')
print(f'验证集: {val_size}')

indices = np.random.permutation(total)
val_indices = indices[:val_size]

os.makedirs('reconstruction_data/val/diff_images', exist_ok=True)
os.makedirs('reconstruction_data/val/height_maps', exist_ok=True)

for idx in val_indices:
    diff_file = diff_files[idx]
    height_file = diff_file.replace('diff_images', 'height_maps')
    shutil.move(diff_file, diff_file.replace('train', 'val'))
    shutil.move(height_file, height_file.replace('train', 'val'))

print('✓ 数据划分完成')
"
```

#### 开始训练（6-8小时）

```bash
# 训练模型
python train_reconstruction.py \
    --train_dir ./reconstruction_data/train \
    --val_dir ./reconstruction_data/val \
    --epochs 100 \
    --batch_size 32 \
    --lr 0.001 \
    --save_dir ./models
```

**训练输出：**
- `models/best_model.pth` - 最佳模型
- `models/final_model.pth` - 最终模型
- `models/training_curve.png` - 训练曲线
- `models/comparison_epoch_*.png` - 对比图

#### 训练调试

**如果损失不下降：**
```bash
# 降低学习率
python train_reconstruction.py \
    --train_dir ./reconstruction_data/train \
    --val_dir ./reconstruction_data/val \
    --epochs 100 \
    --lr 0.0001 \
    --save_dir ./models_v2
```

**如果过拟合：**
- 训练损失很低，验证损失很高
- 解决：增加数据增强、减少训练轮数

### Day 3：测试和完善

#### 创建测试脚本

```bash
cat > test_model.py << 'EOF'
import cv2
import yaml
import torch
import numpy as np
import sys
sys.path.append('..')
from shape_reconstruction import Sensor
from train_reconstruction import UNet

# 加载模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet().to(device)
model.load_state_dict(torch.load('models/best_model.pth', map_location=device))
model.eval()
print('✓ 模型加载完成')

# 初始化传感器
with open('shape_config.yaml', 'r') as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

import time
temp_camera = Sensor.__bases__[0](cfg, calibrated=True)
for i in range(3, 0, -1):
    print(f'{i}秒后捕获参考图像...')
    time.sleep(1)
img_add = np.zeros_like(temp_camera.get_rectify_crop_image(), float)
for _ in range(10):
    img_add += temp_camera.get_rectify_crop_image()
ref = (img_add / 10).astype(np.uint8)
sensor = Sensor(cfg, calibrated=True, ref=ref)
print('✓ 传感器初始化完成\n按q退出...\n')

# 主循环
while sensor.cap.isOpened():
    img = sensor.get_rectify_crop_image()
    if img is None:
        continue
    
    img_GRAY = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    diff = sensor.ref_GRAY.astype(np.float32) - img_GRAY.astype(np.float32)
    diff = np.clip(diff, 0, 255).astype(np.float32)
    
    # 传统方法
    height_trad = sensor.raw_image_2_height_map(img_GRAY)
    
    # 神经网络
    with torch.no_grad():
        diff_tensor = torch.from_numpy(diff/255.0).unsqueeze(0).unsqueeze(0).to(device)
        height_nn = model(diff_tensor).cpu().squeeze().numpy() * 10.0
    
    # 显示对比
    depth_trad = sensor.height_map_2_depth_map(height_trad)
    depth_nn = sensor.height_map_2_depth_map(height_nn)
    
    depth_trad_bgr = cv2.cvtColor(depth_trad, cv2.COLOR_GRAY2BGR)
    depth_nn_bgr = cv2.cvtColor(depth_nn, cv2.COLOR_GRAY2BGR)
    
    cv2.putText(depth_trad_bgr, f'Traditional: {height_trad.max():.2f}mm', 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(depth_nn_bgr, f'Neural Net: {height_nn.max():.2f}mm', 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
    display = np.hstack([img, depth_trad_bgr, depth_nn_bgr])
    cv2.imshow('Test: Original | Traditional | Neural Network', display)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
EOF

# 运行测试
python test_model.py
```

---

## 数据采集指南

### 核心原则

#### 1. 多样性最重要

- **物体形状**：球、圆柱、平面、不规则
- **接触位置**：中心、边缘、各个区域
- **接触力度**：轻触、中等、深压
- **接触角度**：垂直、倾斜、滚动

#### 2. 数据质量

- 确保每个样本有明显接触（高度 > 0.05mm）
- 避免重复相似样本
- **保持光照和环境稳定**（非常重要！）

#### 3. 数量要求

- **最少**：800 个样本
- **推荐**：1000-1500 个样本
- **理想**：2000+ 个样本（如果时间允许）

#### 4. 平衡性

不同类型样本应该均衡分布：

| 类别 | 建议占比 | 样本数（基于1000总数） |
|------|----------|------------------------|
| 球形物体 | 20% | 200 |
| 圆柱形物体 | 20% | 200 |
| 平面物体 | 25% | 250 |
| 不规则形状 | 25% | 250 |
| 多点接触 | 10% | 100 |

### 采集操作

#### 启动采集

```bash
cd shape_reconstruction
python data_preparation.py --auto_ref --output_dir ./reconstruction_data/train
```

#### 操作说明

- **自动保存**：检测到有效接触且与上次不同时自动保存
- **手动保存**：按 's' 键强制保存当前帧
- **退出**：按 'q' 键退出并保存

#### 采集节奏

1. 接触传感器表面
2. 保持 1-2 秒
3. 移开物体
4. 换位置/角度
5. 重复

**不要**快速连续按压（会产生大量相似样本）

### 质量检查

每采集200-300个样本检查一次：

```bash
# 查看数量
ls reconstruction_data/train/diff_images/ | wc -l

# 检查质量
python check_dataset.py --data_dir reconstruction_data/train --visualize
```

**检查项目：**
- 样本数量是否增长
- 高度分布是否合理（轻、中、重接触都有）
- 接触面积分布是否多样
- 是否有明显的数据问题

### 常见问题

#### Q1: 摄像头打开后程序卡住？
**A**: 需要按 'y' 确认参考图像，或使用 `--auto_ref` 自动捕获

#### Q2: 采集速度太慢？
**A**: 
- 减少每个物体的变化幅度
- 使用自动保存，不要频繁手动按's'
- 加快操作节奏

#### Q3: 样本数量不增加？
**A**: 
- 检查接触是否足够深（高度 > 0.05mm）
- 确保与上次样本有明显不同
- 查看终端输出的提示信息

#### Q4: 光照变化了怎么办？
**A**: **必须重新开始！** 光照变化会导致参考图像失效

---

## 模型训练

### 模型架构

使用 **U-Net** 架构：
- 编码器：提取特征
- 解码器：重建高度图
- 跳跃连接：保留细节

### 训练参数

```bash
python train_reconstruction.py \
    --train_dir ./reconstruction_data/train \
    --val_dir ./reconstruction_data/val \
    --epochs 100 \              # 训练轮数
    --batch_size 32 \            # 批次大小
    --lr 0.001 \                 # 学习率
    --save_dir ./models          # 保存目录
```

### 参数调整

| 参数 | 默认值 | 调整建议 |
|------|--------|---------|
| `--epochs` | 100 | 如果过拟合，减少到50-80 |
| `--batch_size` | 32 | 显存不足时减少到16或8 |
| `--lr` | 0.001 | 不收敛时降低到0.0001 |

### 数据增强

训练脚本自动应用：
- 随机水平翻转
- 随机垂直翻转
- 随机旋转90度

### 监控训练

**正常训练应该看到：**
- 训练损失和验证损失都在下降
- 验证损失略高于训练损失（正常）
- 对比图显示预测越来越接近真实值

**异常情况：**
1. **损失不下降** → 降低学习率
2. **训练损失低，验证损失高** → 过拟合，增加数据或减少轮数
3. **损失突然爆炸** → 学习率过大，重新训练

### 输出文件

训练完成后会生成：
- `best_model.pth` - 验证损失最低的模型（推荐使用）
- `final_model.pth` - 最后一轮的模型
- `training_curve.png` - 训练曲线图
- `comparison_epoch_*.png` - 每10轮的对比图
- `final_comparison.png` - 最终8个样本的对比

---

## 常见问题

### 环境问题

#### Q: PyTorch安装失败？
```bash
# 先安装CPU版本
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 或GPU版本（CUDA 11.8）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

#### Q: OpenCV导入错误？
```bash
pip uninstall opencv-python opencv-contrib-python
pip install opencv-python==4.8.0.76
```

### 数据问题

#### Q: 数据量不够怎么办？
- 最少需要800个样本才能训练
- 如果实在来不及，可以：
  - 降低采集质量要求（但不推荐）
  - 减少物体种类（但影响泛化）
  - 使用更多的数据增强

#### Q: 数据质量不好？
查看 `check_dataset.py` 的输出：
- 如果轻触样本过多（>60%），增加深压样本
- 如果接触面积变化小，使用更多不同大小的物体
- 如果某种类型样本太少，专门补充采集

### 训练问题

#### Q: 显存不足？
```bash
# 减小batch size
--batch_size 16  # 或 8, 4

# 或使用CPU训练（虽然慢）
CUDA_VISIBLE_DEVICES="" python train_reconstruction.py ...
```

#### Q: 训练太慢？
- 使用GPU：比CPU快10-50倍
- 减少数据量：用800个样本快速验证
- 减少训练轮数：50轮可能就够了

#### Q: 模型效果不好？
可能的原因和解决方案：
1. **数据量不足** → 继续采集
2. **数据多样性不够** → 增加物体种类
3. **过拟合** → 增加数据或减少模型复杂度
4. **学习率问题** → 尝试0.0001或0.01

---

## 检查清单

### ✅ 24小时检查点
- [ ] 采集了 1000+ 样本
- [ ] 数据质量良好（check_dataset.py检查）
- [ ] 有多种物体类型
- [ ] 高度分布合理

### ✅ 48小时检查点
- [ ] 数据划分完成（训练集/验证集）
- [ ] 模型训练完成
- [ ] 训练损失收敛
- [ ] 验证损失合理（无严重过拟合）
- [ ] 有保存的模型文件

### ✅ 72小时检查点
- [ ] 模型能在真实传感器上运行
- [ ] 效果可接受
- [ ] 代码整理完成
- [ ] 有演示材料

---

## 快速开始

```bash
# 1. 安装依赖
pip install torch torchvision opencv-python numpy matplotlib pyyaml tqdm scipy

# 2. 验证安装
python -c "import torch; import cv2; print('✓ 安装成功')"

# 3. 开始采集数据
cd shape_reconstruction
python data_preparation.py --auto_ref

# 4. 检查数据（采集一段时间后）
python check_dataset.py --data_dir reconstruction_data/train --visualize

# 5. 划分数据集（采集完成后）
python -c "..." # 见上文

# 6. 训练模型
python train_reconstruction.py \
    --train_dir ./reconstruction_data/train \
    --val_dir ./reconstruction_data/val \
    --epochs 100

# 7. 测试模型
python test_model.py
```

---

## 项目结构

```
.
├── shape_reconstruction/
│   ├── data_preparation.py      # 数据采集
│   ├── train_reconstruction.py  # 模型训练
│   ├── check_dataset.py         # 数据检查
│   ├── sensor.py                # 传感器类
│   └── camera.py                # 相机类
├── reconstruction_data/         # 数据目录（不提交git）
│   ├── train/
│   │   ├── diff_images/
│   │   └── height_maps/
│   └── val/
│       ├── diff_images/
│       └── height_maps/
├── models/                      # 模型目录（不提交git）
│   ├── best_model.pth
│   └── training_curve.png
├── requirements.txt             # Python依赖
├── environment.yml              # Conda环境
└── GUIDE.md                     # 本文档
```

---

## 预期效果

使用伪标签训练的模型：
- **理论上限**：接近传统方法的性能
- **实际效果**：70-80% 的精度
- **优势**：推理速度快，适合实时应用

---

**祝你顺利完成！如有问题随时查看本文档。💪**
