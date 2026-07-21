# TransGrokking 实验协议

固定理论与总实验规范由仓库根目录的
[`TransGrokking_Theory_Experiment_Specification.pdf`](../TransGrokking_Theory_Experiment_Specification.pdf)
控制。本文件维护仓库当前可执行协议并登记阶段状态；历史工程决定、命令、失败与正式运行
详情见[实现与运行记录](implementation.md)。本文件不复制完整实验日志。

## 基线

- 任务：`p=97` 模加法，固定 seed 的 40%/60% 划分。
- 模型与优化参数以 `configs/baseline_ce.yaml` 为唯一入口。
- 基线使用 `final_norm: false`：最后 residual stream 直接进入 unembedding，与历史原型一致。
- CE-reference 目标函数仅为 cross-entropy；congruence loss 禁用。
- 正式设备为 `cuda:0` 上的 NVIDIA GeForce RTX 4060 Laptop GPU 8GB。
- 数值策略为 FP32，并关闭 TF32、AMP、BF16、FP16 和 `torch.compile`。

正式运行必须先通过严格 doctor。配置、划分或数值策略不得在运行中静默改变。
失败与中断运行保留配置、状态和错误摘要。

AdamW 使用两个稳定 parameter groups：matrix weights 与 embeddings 接受配置的 weight
decay，bias 和 LayerNorm 参数默认不接受 decay。每组名称、学习率、weight decay 和参数
名称写入 metadata 与 checkpoint compatibility signature。

## 配置与恢复边界

Scientific config 包含 task、model、optimizer 类型、学习率、weight decay 及其参数策略、
loss、precision/TF32/AMP、seed、deterministic、device 和正式硬件约束；恢复时 hash 必须一致。

Execution config 可调整 `max_steps`、eval/checkpoint interval、activation steps、runs directory、
analysis batch size 和 activation offload。新 `max_steps` 必须严格大于来源 checkpoint step。

默认 `resume-mode=auto`：interrupted run 的最新 checkpoint 且 scalar 未超前时原地继续；
completed run、历史 checkpoint、非最新 checkpoint或无法安全追加的情况创建 child run。Child
使用绝对 global step，并记录 parent run、checkpoint 和 parent step。Manifest step 唯一，scalar
step 严格递增，已有 checkpoint 禁止覆盖。

## 生命周期

状态依次为 `initializing → running → completed`，中断或失败分别进入 `interrupted`、
`failed`。模型与 optimizer 可用后的 KeyboardInterrupt 保存 emergency checkpoint。CUDA 训练
在 loop 前重置 peak counters，结束或失败时记录 allocated/reserved peak VRAM。

## M1 行为时间线

每次 `logging.eval_interval` evaluation 在 `model.eval()` 与 `torch.no_grad()` 下对 train/test
分别前向，完成后恢复训练模式。M1 不保存完整 logits。

样本 margin 为

\[
m(a,b)=z_y(a,b)-\max_{c\ne y}z_c(a,b),
\]

其中最大值显式排除正确类别。每个 split 保存 mean、min、q01、q05、q25、median、q75、
q95、q99。错误 offset 为 `(prediction-y) mod p`，只统计错误样本；直方图长度为 `p` 且
第 0 项为 0。

参数范数使用 FP64 accumulator。模块桶互不重叠：token/position embedding、各 block 的
attention、MLP、LayerNorm、final norm、unembedding；平方和重建总范数。`final_norm=false`
时对应字段为 JSON `null`。Decay/no-decay 范数复用 optimizer 的稳定参数清单。

事件定义来自顶层 `events` 配置：

- `t_fit`：train accuracy ≥ 0.999，连续 5 次 evaluation；
- `t_grok50`：test accuracy ≥ `(1+1/p)/2`，连续 3 次；
- `t_grok99`：test accuracy ≥ 0.99，连续 3 次。

事件 step 是首次连续窗口起点，`detected_at_evaluation_step` 是确认窗口末端。同一 run
已经达到的事件不可因恢复或 measurement 配置调整而改写。

Metrics schema v1 包含 `scalars.jsonl`、`error_offsets.jsonl`、`events.json`。JSON 禁止
NaN/Infinity；JSONL 采用原子替换。Offset 先写，scalar 作为 evaluation commit marker，
events 随后原子重建；恢复会截断没有 scalar commit 的尾部 offset。Child run 复制父
checkpoint 之前的 committed M1 前缀后继续绝对 step。

## 当前阶段状态

```text
M0 engineering foundation: completed
M1-A behavior measurement: completed
M1-B CE-reference seed 1: completed
M2 function-space analysis: planned
```

Canonical CE-reference run：`20260721T045433955396Z_30c62ebc`。M1 的正式执行、失败重试、
行为事件和审计结果见[实现与运行记录](implementation.md)。行为事件用于定位时间线，函数空间
与机制解释属于 M2，当前仍为 planned。

正式 CE-reference 固定 `eval_interval=50` 与 `checkpoint_interval=100`。训练入口必须先设置
确定性环境和随机状态，再调用任何 CUDA 诊断；模型迁移到 `cuda:0`/FP32 后才能建立 AdamW
parameter groups。PyTorch 2.2 的 AdamW 一阶、二阶矩与参数同处 GPU，标量 step counter 默认位于
CPU；该标量不参与参数张量设备绑定判断，也不通过 `capturable=True` 改变正式优化路径。

最终 run 使用 `transgrokking.cli audit` 生成 `audit/m1_ce_reference.json`。只有生命周期、
lineage、hash、时间轴、manifest、离线 evaluator 和预注册停止规则全部通过时，M1-B 才标记完成。
