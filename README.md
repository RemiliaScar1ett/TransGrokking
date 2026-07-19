# TransGrokking

TransGrokking 是面向模加法 Grokking 复现和机制分析的实验平台。理论定义见
[`docs/intro.md`](docs/intro.md)。当前已实现范围为 M0：确定性数据、透明
Transformer、显式 AdamW parameter groups、CE-only full-batch 训练、安全 checkpoint
恢复和硬件预检。
M1 及之后的行为时间线、Fourier、群对称性和干预分析仍为 planned。

## 环境

唯一支持的环境位置是仓库根目录 `./env`。当前环境基线为 Python 3.10、
PyTorch 2.2 和 PyTorch CUDA 12.1；不要求为重构主动升级环境。

```bash
conda run --prefix ./env python -m pip install -e .
conda run --prefix ./env python -m transgrokking.cli doctor
```

正式实验目标为 `cuda:0` 上的 NVIDIA GeForce RTX 4060 Laptop GPU 8GB：

```bash
conda run --prefix ./env python -m transgrokking.cli doctor --require-cuda --expected-device "NVIDIA GeForce RTX 4060 Laptop GPU" --expected-vram-gb 8
```

## M0 命令

生成确定性划分：

```bash
conda run --prefix ./env python -m transgrokking.cli generate-data --config configs/smoke.yaml
```

执行三步 CPU smoke：

```bash
conda run --prefix ./env python -m transgrokking.cli train --config configs/smoke.yaml
```

恢复指定 checkpoint：

```bash
conda run --prefix ./env python -m transgrokking.cli train --config configs/smoke.yaml --resume-from runs/<run_id>/checkpoints/step_000001.pt --resume-mode auto
```

`auto` 仅在 interrupted run 的最新 checkpoint 上原地继续；completed run、历史 checkpoint
或非最新 checkpoint 自动创建 child run。可显式选择 `inplace` 或 `branch`，不满足原地
恢复条件时命令会失败。恢复允许提高 `max_steps`，但目标必须严格大于 checkpoint step。

`configs/baseline_ce.yaml` 是尚未执行的正式 CE-only 配置。本轮不得把 CPU smoke
结果解释为 Grokking 实验结果。

## 运行产物

每个 `runs/<run_id>/` 保存 resolved config、环境 metadata、数据划分及哈希、状态、
JSONL scalar、checkpoint 和 manifest。Checkpoint 包含模型、优化器、step、配置、
split hash 及 Python/NumPy/Torch RNG 状态，并采用临时文件加原子替换写入。
Metadata 同时记录 scientific config hash、父子 run 关系、AdamW group 名称及参数清单、
最终显存峰值。正式 CE 基线采用 `final_norm: false`，与历史原型架构保持一致。

当前状态：

```text
M0 implementation: completed after fixes
formal CE-only baseline: not yet run
M1 analysis: planned
```

历史原型位于 [`legacy/`](legacy/README.md)，仅用于追溯，不是受支持入口。
