#!/bin/bash
# 快速启动脚本 - 深度学习优化三维形状重建

echo "=========================================="
echo "深度学习优化三维形状重建 - 快速启动"
echo "=========================================="
echo ""

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 检查Python环境
echo "检查Python环境..."
python3 --version
if [ $? -ne 0 ]; then
    echo -e "${RED}错误：未找到Python3${NC}"
    exit 1
fi

# 检查依赖
echo ""
echo "检查依赖..."
python3 -c "import torch" 2>/dev/null
if [ $? -ne 0 ]; then
    echo -e "${YELLOW}警告：PyTorch未安装${NC}"
    echo "正在安装PyTorch..."
    pip install torch torchvision torchaudio
fi

python3 -c "import cv2, numpy, tqdm, yaml" 2>/dev/null
if [ $? -ne 0 ]; then
    echo -e "${YELLOW}警告：缺少部分依赖${NC}"
    echo "正在安装依赖..."
    pip install opencv-python numpy tqdm pyyaml matplotlib
fi

echo -e "${GREEN}✓ 环境检查通过${NC}"

# 菜单选择
echo ""
echo "请选择启动方案："
echo "1) 快速验证（合成数据 + 简单模型，2小时）"
echo "2) 完整流程（真实数据 + U-Net模型，4-6小时，需要硬件）"
echo "3) 仅训练（假设数据已准备好）"
echo "4) 仅测试（假设模型已训练好）"
echo "5) 查看帮助"
read -p "请输入选项 (1-5): " choice

case $choice in
    1)
        echo ""
        echo "=========================================="
        echo "方案1：快速验证"
        echo "=========================================="

        # 步骤1：生成合成数据
        echo ""
        echo "[1/3] 生成合成数据..."
        python3 prepare_data.py --method synthetic --num_samples 1000
        if [ $? -ne 0 ]; then
            echo -e "${RED}✗ 数据生成失败${NC}"
            exit 1
        fi
        echo -e "${GREEN}✓ 数据生成完成${NC}"

        # 步骤2：训练模型
        echo ""
        echo "[2/3] 训练模型（预计30-60分钟）..."
        python3 train_reconstruction.py \
            --model simple \
            --batch_size 16 \
            --epochs 30 \
            --lr 0.001
        if [ $? -ne 0 ]; then
            echo -e "${RED}✗ 训练失败${NC}"
            exit 1
        fi
        echo -e "${GREEN}✓ 训练完成${NC}"

        # 步骤3：测试评估
        echo ""
        echo "[3/3] 测试评估..."
        python3 test_reconstruction.py --mode evaluate
        if [ $? -ne 0 ]; then
            echo -e "${YELLOW}! 测试失败，但模型已训练完成${NC}"
        else
            echo -e "${GREEN}✓ 测试完成${NC}"
        fi

        echo ""
        echo -e "${GREEN}=========================================="
        echo "快速验证完成！"
        echo "==========================================${NC}"
        echo "模型保存在: ../saved_models/reconstruction_best.pth"
        echo "日志保存在: ./logs/"
        echo ""
        echo "下一步："
        echo "  - 查看实验结果"
        echo "  - 制作PPT和演示"
        ;;

    2)
        echo ""
        echo "=========================================="
        echo "方案2：完整流程"
        echo "=========================================="

        # 检查传感器配置
        if [ ! -f "shape_config.yaml" ]; then
            echo -e "${RED}✗ 未找到传感器配置文件 shape_config.yaml${NC}"
            exit 1
        fi

        # 步骤1：采集数据
        echo ""
        echo "[1/3] 采集真实数据..."
        echo "请按照提示操作传感器..."
        python3 prepare_data.py --method camera --num_samples 500
        if [ $? -ne 0 ]; then
            echo -e "${RED}✗ 数据采集失败${NC}"
            exit 1
        fi
        echo -e "${GREEN}✓ 数据采集完成${NC}"

        # 步骤2：训练模型
        echo ""
        echo "[2/3] 训练U-Net模型（预计1-3小时）..."
        python3 train_reconstruction.py \
            --model unet \
            --batch_size 16 \
            --epochs 50 \
            --lr 0.001
        if [ $? -ne 0 ]; then
            echo -e "${RED}✗ 训练失败${NC}"
            exit 1
        fi
        echo -e "${GREEN}✓ 训练完成${NC}"

        # 步骤3：对比测试
        echo ""
        echo "[3/3] 对比测试（需要传感器）..."
        python3 test_reconstruction.py \
            --mode compare \
            --num_samples 10
        if [ $? -ne 0 ]; then
            echo -e "${YELLOW}! 对比测试失败，但模型已训练完成${NC}"
        else
            echo -e "${GREEN}✓ 对比测试完成${NC}"
        fi

        echo ""
        echo -e "${GREEN}=========================================="
        echo "完整流程完成！"
        echo "==========================================${NC}"
        echo "模型保存在: ../saved_models/reconstruction_best.pth"
        echo "对比结果在: ./comparison_results/"
        echo "日志保存在: ./logs/"
        ;;

    3)
        echo ""
        echo "=========================================="
        echo "仅训练模型"
        echo "=========================================="

        # 检查数据
        if [ ! -d "reconstruction_data/train" ]; then
            echo -e "${RED}✗ 未找到训练数据${NC}"
            echo "请先运行数据准备："
            echo "  python3 prepare_data.py --method synthetic --num_samples 1000"
            exit 1
        fi

        echo "选择模型架构："
        echo "1) simple - 简单模型，快速训练"
        echo "2) unet - U-Net模型，推荐"
        echo "3) resnet - ResNet模型，精度最高"
        read -p "请选择 (1-3): " model_choice

        case $model_choice in
            1) MODEL="simple"; EPOCHS=30 ;;
            2) MODEL="unet"; EPOCHS=50 ;;
            3) MODEL="resnet"; EPOCHS=100 ;;
            *) MODEL="unet"; EPOCHS=50 ;;
        esac

        echo ""
        echo "开始训练 $MODEL 模型..."
        python3 train_reconstruction.py \
            --model $MODEL \
            --batch_size 16 \
            --epochs $EPOCHS \
            --lr 0.001

        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✓ 训练完成${NC}"
        else
            echo -e "${RED}✗ 训练失败${NC}"
        fi
        ;;

    4)
        echo ""
        echo "=========================================="
        echo "测试模型"
        echo "=========================================="

        # 检查模型
        if [ ! -f "../saved_models/reconstruction_best.pth" ]; then
            echo -e "${RED}✗ 未找到训练好的模型${NC}"
            echo "请先训练模型"
            exit 1
        fi

        echo "选择测试模式："
        echo "1) evaluate - 在验证集上评估"
        echo "2) demo - 实时演示（需要传感器）"
        echo "3) compare - 对比实验（需要传感器）"
        read -p "请选择 (1-3): " test_choice

        case $test_choice in
            1)
                python3 test_reconstruction.py --mode evaluate
                ;;
            2)
                python3 test_reconstruction.py --mode demo
                ;;
            3)
                python3 test_reconstruction.py --mode compare --num_samples 10
                ;;
            *)
                echo "无效选择"
                ;;
        esac
        ;;

    5)
        echo ""
        cat << EOF
========================================
帮助信息
========================================

快速启动流程：

1. 快速验证（2小时）
   - 生成合成数据
   - 训练简单模型
   - 评估精度
   适合：快速验证概念

2. 完整流程（4-6小时）
   - 采集真实数据
   - 训练U-Net模型
   - 对比实验
   适合：完整的项目

3. 仅训练
   - 假设数据已准备好
   - 选择模型架构训练

4. 仅测试
   - 假设模型已训练好
   - 评估或演示

详细文档：
  查看 深度学习优化实施指南.md

常用命令：

# 生成数据
python3 prepare_data.py --method synthetic --num_samples 1000

# 训练模型
python3 train_reconstruction.py --model unet --epochs 50

# 测试模型
python3 test_reconstruction.py --mode evaluate

# 查看训练日志
tensorboard --logdir ./logs

问题排查：
  查看 深度学习优化实施指南.md 的"常见问题"部分

========================================
EOF
        ;;

    *)
        echo -e "${RED}无效选项${NC}"
        exit 1
        ;;
esac

echo ""
echo "脚本执行完毕"
