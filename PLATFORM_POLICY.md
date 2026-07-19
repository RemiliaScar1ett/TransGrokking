# TransGrokking 环境与平台约束摘要

## Conda 环境

项目只使用仓库根目录下的 `./env`：

```bash
conda env create --prefix ./env --file environment.yml
conda env update --prefix ./env --file environment.yml --prune
conda run --prefix ./env python -m pip install -e .
```

自动化命令统一使用 `conda run --prefix ./env`。`env/` 需要加入 `.gitignore`。

## 正式实验平台

```text
NVIDIA GeForce RTX 4060 Laptop GPU
8 GB VRAM
cuda:0
```

CPU 用于单元测试和 smoke run。正式 Grokking 轨迹需要通过硬件严格检查。

## 数值基线

主基线采用 FP32，关闭 TF32、AMP、BF16、FP16 和 `torch.compile`。其他精度模式作为单独实验条件。

## 显存策略

完整 logits 可以逐 checkpoint 构造。activation 按指定模块和指定 checkpoint 提取，完成后转移到 CPU。分析过程记录峰值显存、batch size 与 offload 策略。
