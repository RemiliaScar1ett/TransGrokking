# TransGrokking

TransGrokking 是面向模加法 Grokking 复现和机制分析的实验平台。理论定义见
[`docs/intro.md`](docs/intro.md)，仓库可执行协议见
[`docs/experiment_protocol.md`](docs/experiment_protocol.md)。M0 工程基础、M1-A 行为测量和
M1-B 20000-step CE-reference seed 1 已完成。该轨迹显示首次泛化后仍会反复失稳并恢复；
M1-C 50000-step 延长及稳定性测量也已完成并通过审计。M2-A 失稳真实性分析和 M2-B
函数空间分析仍待实施。

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

## 常用命令

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

普通训练的 `auto` 仅在 interrupted run 的最新 checkpoint 上原地继续；completed run、历史 checkpoint
或非最新 checkpoint 自动创建 child run。可显式选择 `inplace` 或 `branch`，不满足原地
恢复条件时命令会失败。带 M1-C measurement sidecar 的恢复始终创建 child，以免在同一 run
混合不同代码 provenance。恢复允许提高 `max_steps`，但目标必须严格大于 checkpoint step。

`configs/baseline_ce.yaml` 是已冻结并完成 seed 1 正式运行的 CE-only 基准配置。

## 运行产物

每个 `runs/<run_id>/` 保存 resolved config、环境 metadata、数据划分及哈希、状态、
JSONL scalar、checkpoint 和 manifest。Checkpoint 包含模型、优化器、step、配置、
split hash 及 Python/NumPy/Torch RNG 状态，并采用临时文件加原子替换写入。
Metadata 同时记录 scientific config hash、父子 run 关系、AdamW group 名称及参数清单、
最终显存峰值。正式 CE 基线采用 `final_norm: false`，与历史原型架构保持一致。

每次 evaluation 将实际前向结果写入：

```text
metrics/scalars.jsonl       # loss、accuracy、margin、error count/rate、参数范数
metrics/error_offsets.jsonl # train/test 错误样本的循环 offset 直方图
metrics/events.json         # t_fit、t_grok50、t_grok99
metrics/stability.json      # t_stable99、最终稳定性状态和汇总
metrics/collapse_episodes.json # train/test primitive 与 joint composite episode
metrics/optimization.jsonl  # M1-C extension-only AdamW 优化诊断
measurement.resolved.yaml   # M1-C measurement sidecar 的冻结副本
```

错误 offset 定义为 `(prediction-label) mod p`，只统计误分类，因此长度为 `p` 的 counts
中第 0 项固定为 0。正确类别 margin 定义为正确 logit 减去排除正确类别后的最大错误
logit。Child run 继承父 checkpoint 之前的 committed M1 时间线，并继续使用绝对 step。

只读重算最新或指定 checkpoint 的 M1 指标：

```bash
conda run --prefix ./env python -m transgrokking.cli evaluate \
  --run-dir runs/<run_id>

conda run --prefix ./env python -m transgrokking.cli audit \
  --run-dir runs/<canonical_run_id>
```

`evaluate` 只向终端输出 JSON，不追加训练 timeline 或修改 run 状态；`audit` 将只读核验
结果原子写入 run 的 `audit/` 子目录。

M1-C 使用独立 measurement sidecar，并提供专用审计与只读导出：

```bash
conda run --prefix ./env python -m transgrokking.cli train \
  --config configs/ce_reference_extend_50000.yaml \
  --measurement-config configs/analysis/m1c_stability.yaml \
  --resume-from runs/20260721T045433955396Z_30c62ebc/checkpoints/step_020000.pt \
  --resume-mode auto

conda run --prefix ./env python -m transgrokking.cli audit \
  --run-dir runs/<m1c_run_id> \
  --profile m1c-extension

conda run --prefix ./env python -m transgrokking.cli export-m1c \
  --run-dir runs/<m1c_run_id> \
  --output-dir results/m1_ce_reference_extended
```

当前状态：

```text
M0 engineering foundation: completed
M1-A behavior measurement: completed
M1-B CE-reference 20000-step: completed
M1-C CE-reference 50000-step extension: completed
M1 overall: completed
M2-A instability analysis: planned
M2-B function-space analysis: planned
```

M1 canonical run 为 `20260721T045433955396Z_30c62ebc`，最终 step 为 20000；行为事件为
`t_fit=100`、`t_grok50=6050`、`t_grok99=7000`，M1 audit 已通过。这些时间点仅描述行为
时间线。原始导出证据位于 [`results/m1_ce_reference/`](results/m1_ce_reference/)，保持不可变。
曲线还显示模型反复进入和离开高性能区域，但坍塌窗口的逐 checkpoint 离线重算与机制解释
尚未完成；相关工作分别安排在 M2-A 和 M2-B。完整执行记录见
[`docs/implementation.md`](docs/implementation.md)。

M1-C terminal child 为 `20260724T091041024473Z_c6434d8a`，最终 step 为 50000，完整
M1-C audit 通过。预注册的 `t_stable99` 未达到；操作性检测得到 26 个 train primitive、
10 个 test primitive 和 10 个 joint composite episode，最后 onset 为 step 49950，
terminal 状态为 `recovering`。扩展证据位于
[`results/m1_ce_reference_extended/`](results/m1_ce_reference_extended/)，行为层讨论见
[`docs/M1-disc.md`](docs/M1-disc.md)。这些结果说明轨迹非单调，但尚不提供函数空间或机制解释。

历史原型位于 [`legacy/`](legacy/README.md)，仅用于追溯，不是受支持入口。
