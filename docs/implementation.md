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

## 2026-07-19 — M0 GitHub 审核修复

**背景**

初版 checkpoint 逐字段比较完整配置，无法安全延长训练；历史 checkpoint 可能在原 run
追加重复 step。全部参数统一 decay、固定 final LayerNorm 和不完整生命周期也会削弱正式
CE-only 基线的可解释性。

**选择**

- Scientific config 包含 task、model、optimizer 类型/学习率/decay policy、loss、数值与
  seed/deterministic、device 和正式硬件约束；execution config 包含训练上限、记录频率、
  runs 路径及分析资源字段。Checkpoint 和 metadata 保存稳定 scientific SHA-256 hash。
- `resume-mode=auto` 仅将 interrupted run 的最新且可安全追加 checkpoint 原地恢复；其他
  来源创建 child run。Child 延续绝对 global step，并记录 parent run/checkpoint/step。
- 删除未实现的 `optimizer_checkpoint_interval`，M0 只保留完整 checkpoint 的单一频率。
- AdamW 使用稳定 `decay`/`no_decay` groups。基线 matrix weights 与 embeddings decay，
  bias 与 LayerNorm 不 decay；metadata 保存参数名和实际超参数，checkpoint 校验 group
  signature。
- `model.final_norm` 显式控制最终归一化。正式基线设为 `false`，保持历史原型 residual
  直接进入 unembedding；未来 `true` 只能作为单独对照配置。
- 生命周期为 `initializing → running → completed`，异常转为 `failed`，中断转为
  `interrupted`。KeyboardInterrupt 在状态可用时保存 emergency checkpoint；CUDA loop
  前重置 peak counters，最终 status 与 metadata 同步显存峰值。

**理由**

科学兼容性与执行调度分离后可以延长同一实验而不放松模型、优化或数值约束；安全分支
避免覆盖历史证据。显式 parameter groups 和 final norm 消除了后续范数与 margin 解释中的
架构歧义。

**影响**

Checkpoint schema 与 manifest schema 升级为 v2；初版 v1 checkpoint 不会被静默加载。
Cache 中 `residual.final` 固定表示归一化前的最终 residual，启用 final norm 时另有
`residual.final_normalized`。

**待验证事项**

正式 `configs/baseline_ce.yaml` 仍未运行。Final LayerNorm 对照、M1 指标和 Grokking 轨迹
均保持 planned。

## 2026-07-19 — 审核修复实际验证

**背景**

本节只记录本轮实际执行的软件、恢复和单步硬件验证，不将任何 smoke 数据解释为科学结果。

**选择**

执行 pytest、ruff、普通/严格 doctor、真实 CPU smoke、中断—原地恢复以及 completed 历史
checkpoint child branch。测试套件包含真实 subprocess CLI 和标记为 `cuda` 的 1-update
目标设备检查。

**理由**

覆盖审核指出的所有可信度边界，同时禁止启动正式长程基线。

**影响**

- `conda run --prefix ./env python -m pytest -q`：22 passed；目标 GPU 可用，因此两个
  `@pytest.mark.cuda` 测试均实际执行，没有 skip。
- `ruff check` 与 `ruff format --check`：通过。
- 普通与严格 doctor：通过；识别 RTX 4060 Laptop GPU、8,585,216,000 bytes VRAM、
  compute capability 8.9、PyTorch 2.2.0/CUDA runtime 12.1。
- 真实 CPU CLI smoke：完成 3 个 update，status/metadata/split/scalars/checkpoint/manifest
  完整；CPU metadata 的 peak VRAM 正确记录为 0，且 `formal_run=false`。
- 独立 CUDA integration：在目标 GPU 执行 1 个 update，metadata/status 均记录非零 peak
  allocated 与 reserved VRAM。该运行只验证硬件路径，不是正式 CE 基线。
- 中断—最新 checkpoint 原地恢复和提高 `max_steps`：通过；scalar 为严格递增且来源
  checkpoint 内容未改变。
- completed run 与非最新历史 checkpoint：均创建 child run，父级 metadata、绝对 step、
  manifest 和 scalar 单调性验证通过。

**待验证事项**

```text
M0 implementation: completed after fixes
formal CE-only baseline: not yet run
M1 analysis: planned
```
