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

## 2026-07-19 — M1 行为时间线协议

**背景**

M0 只记录 loss 与 accuracy，无法检查分类 margin、错误类型、参数范数及行为事件，也缺少
只读 checkpoint 行为评估入口。

**选择**

- 新增独立 `metrics/` 层，逐 split 计算 CE、accuracy、正确类排除后的样本 margin、错误
  count/rate 和错误样本循环 offset。
- 参数范数使用 FP64 accumulator 和非重叠模块桶；`final_norm=false` 使用 JSON `null`，
  optimizer group 范数复用现有稳定 parameter names。
- 顶层 `events` 配置定义 `t_fit`、`t_grok50`、`t_grok99` 的阈值和连续 evaluation 数，
  属于 execution/measurement config，不进入 scientific hash。
- Metrics schema v1 使用 `scalars.jsonl`、`error_offsets.jsonl`、`events.json`。Offset 先原子
  写入，scalar 作为 commit marker，events 从 committed scalar 幂等重建；恢复截断未提交
  offset 尾部。
- Child run 复制父 checkpoint 之前的 committed M1 timeline，再以绝对 global step 追加。
  同一 run 已达到事件保持首次检测结果。
- `evaluate --run-dir ... [--checkpoint ...]` 校验 config、split、manifest 和 checkpoint 后
  只向终端输出单 checkpoint JSON，不修改 run。

**理由**

行为指标必须来自真实前向，且恢复、分支和离线重算需要共享同一数学实现。Scalar commit
协议在缺少跨文件事务的普通文件系统上提供明确的安全恢复点。

**影响**

Resolved config 新增 `events` section；旧 M0-only scalar 文件不会被静默当作完整 M1
timeline。Margin 与错误 offset 是 M1 行为量，不包含 Reynolds、三维 Fourier 或其他 M2
函数空间分析。

实际验证：

- `conda run --prefix ./env python -m pytest -q`：34 passed；目标 GPU 可用，CUDA RNG 与
  M1 1-update smoke 实际执行，无 skip。
- `ruff check`、`ruff format --check`：通过。
- 普通与严格 doctor：通过，识别 RTX 4060 Laptop GPU、8,585,216,000 bytes VRAM、
  compute capability 8.9、PyTorch 2.2.0/CUDA runtime 12.1。
- CPU smoke run `20260719T085709446544Z_274e145d`：3 个 update，三个 metrics 文件完整，
  scalar step 为 1/2/3，每个 step 有 train/test offset pair，事件文件 last step 为 3。
- 对该 run 最新 checkpoint 的只读 `evaluate`：通过，未追加 timeline。
- 独立中断—原地恢复测试：通过，M1 scalar/offset/events 安全继续追加。
- 首次组合验收脚本因 PowerShell 选中 Conda 输出末尾空行，使 `evaluate --run-dir` 缺少参数
  而失败；使用明确 smoke run path 重新执行后成功。该失败未修改 run 产物。

**待验证事项**

正式 CE-only 长程基线尚未运行，M2 function-space 指标尚未实现。

该记录完成时的阶段状态为：

```text
M0 implementation: completed
M1-A behavior measurement: completed
M1-B CE-reference seed 1: planned
M2 function-space analysis: planned
```

## 2026-07-21 — 正式 CUDA 主链路与 M1 审计

**背景**

训练器原先在模型迁移到配置设备前建立 AdamW。正式运行还要求确定性环境变量先于 doctor 的
首次 CUDA 查询生效，并需要机器可读的最终轨迹审计。

**选择**

训练入口先调用 `configure_reproducibility`，再执行 doctor；模型迁移至 device/FP32 后建立
parameter groups 和 optimizer，并严格比较模型与 optimizer 的参数对象集合。正式配置冻结为
evaluation interval 50、checkpoint interval 100。新增 `audit` 命令核验 lineage、hash、M1
时间轴、manifest、离线 evaluator、显存和停止规则。

**理由**

该顺序避免 optimizer 引用迁移前参数的风险，并确保 cuBLAS 确定性约束在首次 CUDA API 前设置。
审计产物把正式运行验收从人工检查变为可重复的 schema 化检查。

**影响**

PyTorch 2.2 AdamW 的参数形状状态（`exp_avg`、`exp_avg_sq`）位于 GPU，标量 step counter 默认
位于 CPU。为避免启用 `capturable=True` 改变正式优化路径，GPU 测试显式验证前者及标量有限性。
正式 run 和实际测试结果将在软件验收通过后追加记录。

### M1-B 实际执行记录

- 正式训练代码提交：`98c5347dafe28479817a45f5a2dc61434d869d00`。
- CUDA checkpoint RNG 恢复修复提交：`0f96f3a21409894b1f03cee7c792ecb9d79ea0d6`。
- 软件验收：`37 passed`；Ruff check 与 format check 通过；普通和严格 doctor 通过；
  CPU smoke run 为 `20260720T192718702298Z_274e145d`。
- Root formal run：`20260721T045021566841Z_ef3ee07b`，5000 step，completed。
- 第一次 20000-step continuation：`20260721T045158050567Z_30c62ebc`，在加载 checkpoint
  RNG state 时失败，未执行 optimizer update；失败状态和 traceback 已保留。
- CUDA RNG 修复完成全量回归后，从 root step 5000 重试并创建 canonical child：
  `20260721T045433955396Z_30c62ebc`，20000 step，completed。
- Scientific config hash：
  `b167674594bf0944f0b2afb877d2d8c8f5647c0e4e60c64ebb2a511a9f1f7729`。
- Split hash：`d0ec6ff924ecc411b9a9d40786f057ec869076b98308e2ecb75da2756c308237`。
- 行为事件：`t_fit=100`（step 300 确认）、`t_grok50=6050`（step 6150 确认）、
  `t_grok99=7000`（step 7100 确认）。这些是行为事件，不构成 M2 机制结论。
- 固定停止规则要求训练至至少 step 8000；canonical run 的 final step 20000 已满足，故未执行
  50000-step 延长。最终 step 的单点评估不用于改写已冻结的首次事件。
- Canonical peak allocated/reserved VRAM：230345216 / 373293056 bytes。
- 最终 checkpoint 离线 evaluator 成功；`audit/m1_ce_reference.json` 的全部检查通过，未发现
  congruence、Reynolds、Fourier 或其他 M2+ artifact。

实际命令包括完整 pytest/Ruff、两个 doctor、CPU smoke、5000-step 基线、20000-step安全恢复、
最终 `evaluate` 与 `audit`。按当时的 20000-step 首次事件协议，M1-B 已完成；canonical run
和 audit JSON 是该阶段的唯一正式 CE-reference 证据链。

## 2026-07-23 — Add instability-aware analysis

**背景**

M1 seed 1 行为轨迹在首次拟合与首次 Grokking 后多次离开高性能区并恢复。首次
`t_grok99` 之后的八个候选起点同时出现 train/test accuracy 下降；更早的三个候选起点只满足
train collapse 的时间前提。参数 L2 曲线还显示 decay/no-decay group 的显著尺度重分配。
这些均为行为层观察，尚未通过 checkpoint 函数重建确认，也不构成优化或函数机制结论。

**选择**

- 保留原 M1 canonical run、首次事件与 `results/m1_ce_reference/`，不覆盖或重新导出替换。
- 从 canonical run 的最新 checkpoint 创建 child run，把 seed 1 延长至 50000 step。
- 在函数空间分析前增加 M2-A 失稳真实性验证与坍塌窗口分析。
- 区分首次 Grokking 事件与稳定 Grokking，并规划稳定窗口和坍塌 episode 派生指标。
- 在延长 run 中增加最小优化诊断；0–20000 step 不补写不存在的逐步诊断。
- M2-A/M2-B 管线稳定后，提前执行 seed 2、3 的行为层与失稳统计复现；完整 WD 网格仍留在 M4。

**理由**

首次连续阈值事件只标记模型第一次进入高性能区域，不能描述其后是否长期保持。先确认失稳
是否对应真实 checkpoint 状态，再分析函数结构与优化同步关系，可以避免从单条行为曲线直接
推断机制。

**影响**

- 阶段门控细化为 M1-C、M2-A、M2-B 与 Gate 2；M3 和完整多 seed/WD 网格顺延。
- 规划新增 stability artifacts 和 extension 起点之后的最小优化诊断。
- 首次 `t_grok99` 后 20 个 evaluation interval 只表示首次事件后的继续训练，不再是稳定阶段
  的充分条件。
- 固定 PDF 本轮不修改；待仓库文档方案审核后再发布新版规范。

**待验证事项**

- 坍塌是否跨 seed 复现；
- 坍塌是否与 `Gamma`、`I`、`D_eq` 变化同步；
- 坍塌是否与 update norm、Adam moment 或 parameter-group norm 同步；
- 失稳是否在 50000 step 前消失。

当前候选坍塌起点为：

```text
1400
3450
5850
7350
8450
10050
12350
14250
15350
17500
18350
```

这些 step 来自原始 behavior timeline：前三个是 `t_grok99` 前的 train-collapse 候选，后八个
是首次 `t_grok99` 后的 joint-collapse 候选。它们尚未通过 checkpoint 离线重算确认；最后一个
候选截至 20000-step 轨迹终点也尚未满足 test recovery 判据。

```text
M0 engineering foundation: completed
M1-A behavior measurement: completed
M1-B CE-reference 20000-step: completed
M1-C CE-reference 50000-step extension: planned
M2-A instability analysis: planned
M2-B function-space analysis: planned
Gate 2 seed 2/3 replication: planned
M3 Fourier analysis: planned
```
