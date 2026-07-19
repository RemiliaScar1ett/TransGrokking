# 实现记录

## 2026-07-19 — M0 最小可复现训练平台

**背景**

历史原型混合 CE、congruence loss、TF32 和 BF16 autocast，且没有 checkpoint 恢复协议。

**选择**

采用严格 YAML dataclass 配置、确定性全表划分、显式 causal Transformer、CE-only
full-batch AdamW，以及带 schema、split hash 和完整 RNG 状态的原子 checkpoint。
`global_step` 表示已经完成的 optimizer update；step 0 表示初始化状态。

**理由**

接口保持透明、可 hook，并优先保证科学配置可检查和中断恢复可复现。M0 日志仅包含
管线健康所需的 CE 与 accuracy，不提前实现 M1 分析。

**影响**

历史文件保存在 `legacy/`，不再作为入口。运行产物遵循 `runs/<run_id>/` 协议。

**待验证事项**

正式 Grokking 轨迹尚未执行。M1 事件与机制指标尚未实现。

## 2026-07-19 — 现有 Conda 环境优先

**背景**

仓库 `./env` 已包含 Python 3.10.20、PyTorch 2.2.0、CUDA runtime 12.1、NumPy 1.26.4
和 PyYAML 6.0.3，并可识别目标 GPU。

**选择**

`environment.yml` 记录该实际环境口径。仅在缺失时增量安装 pytest 和 ruff，不升级
或重建 PyTorch、CUDA、Python。

**理由**

优先复用已配置且可运行的本地环境，避免不必要的数值与依赖变化。

**影响**

PyTorch 2.2/CUDA 12.1 是 M0 的已支持基线，不作为 doctor 失败条件。

**待验证事项**

环境中曾存在 pip 的 `Ignoring invalid distribution -umpy` 警告；当前 NumPy 导入、测试
和 smoke 均可运行，因此本轮未重建或升级环境。该警告后续若影响安装再单独处理。

## 2026-07-19 — M0 实际验证

**背景**

按敏捷计划只保留最小关键路径测试和一次真实 CPU smoke。

**选择**

实际执行以下命令：

```bash
conda run --prefix ./env python -m pytest -q
conda run --prefix ./env python -m ruff check .
conda run --prefix ./env python -m ruff format --check .
conda run --prefix ./env python -m transgrokking.cli doctor
conda run --prefix ./env python -m transgrokking.cli doctor --require-cuda --expected-device "NVIDIA GeForce RTX 4060 Laptop GPU" --expected-vram-gb 8
conda run --prefix ./env python -m transgrokking.cli train --config configs/smoke.yaml
```

**理由**

这些检查覆盖数据与配置、模型前后向、doctor、checkpoint、连续/中断恢复等价和真实 CLI
产物链路，避免重复 smoke 或长程运行。

**影响**

- pytest：6 passed；恢复测试逐项比较模型参数和 optimizer tensor。
- ruff lint 与 format check：通过。
- 普通和严格 doctor：通过；实际检测 Python 3.10.20、PyTorch 2.2.0、PyTorch CUDA
  12.1、RTX 4060 Laptop GPU、8,585,216,000 bytes VRAM、compute capability 8.9。
- CPU smoke：完成 3 个 update，生成 step 0–3 checkpoint、manifest、split、metadata、
  JSONL scalars 和 completed 状态。

以上仅是实现与管线验证，不是 Grokking 实验结果。

**待验证事项**

`configs/baseline_ce.yaml` 尚未执行；正式 GPU 数值基线、显存峰值和 Grokking 轨迹均
尚未验证。
