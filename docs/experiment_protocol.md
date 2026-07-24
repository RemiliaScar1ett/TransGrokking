# TransGrokking 实验协议

固定理论与总实验规范由仓库根目录的
[`TransGrokking_Theory_Experiment_Specification.pdf`](../TransGrokking_Theory_Experiment_Specification.pdf)
控制。本文件维护仓库当前可执行协议并登记阶段状态；历史工程决定、命令、失败与正式运行
详情见[实现与运行记录](implementation.md)。本文件不复制完整实验日志。M1-C、M2-A、
M2-B 与 Gate 2 是 M0–M7 总体体系内的仓库执行门细化，不改写本轮保持固定的 PDF。

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
M1-C 的稳定性阈值、冻结来源和诊断频率由独立 measurement sidecar
`configs/analysis/m1c_stability.yaml` 管理；其 resolved 副本与 hash 写入 run metadata，
但不进入 scientific config hash。

默认 `resume-mode=auto`：interrupted run 的最新 checkpoint 且 scalar 未超前时原地继续；
completed run、历史 checkpoint、非最新 checkpoint或无法安全追加的情况创建 child run。Child
使用绝对 global step，并记录 parent run、checkpoint 和 parent step。Manifest step 唯一，scalar
step 严格递增，已有 checkpoint 禁止覆盖。使用 measurement sidecar 的 instrumented run
始终创建 child，以免把不同代码 provenance 的诊断记录混入同一 run。Branch source 只读
校验；只有 interrupted inplace run 才能修复自身未提交的指标尾部。

## 生命周期

状态依次为 `initializing → running → completed`，中断或失败分别进入 `interrupted`、
`failed`。模型与 optimizer 可用后的 KeyboardInterrupt 保存 emergency checkpoint。CUDA 训练
在 loop 前重置 peak counters，结束或失败时记录 allocated/reserved peak VRAM。

## M1 行为时间线

每次 `logging.eval_interval` evaluation 在 `model.eval()` 与 `torch.no_grad()` 下对 train/test
分别前向，完成后恢复训练模式。M1 不保存完整 logits。

样本 margin 为

$$
m(a,b)=z_y(a,b)-\max_{c\ne y}z_c(a,b),
$$

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

基础 M1 metrics schema v1 包含 `scalars.jsonl`、`error_offsets.jsonl`、`events.json`。
M1-C 在不改写基础 schema 的前提下增加 `optimization.jsonl`、`stability.json` 和
`collapse_episodes.json`。JSON 禁止 NaN/Infinity；JSONL 采用原子替换。M1-C 的一次
evaluation 按 offset pair、optimization record、scalar commit marker 的顺序写入，
随后从完整 committed scalar 时间线原子重建 events、stability 和 collapse。恢复会截断
没有 scalar commit 的 offset/optimization 尾部。Child run 复制来源 checkpoint 之前的
committed 前缀后继续绝对 step；0–20000 不补写不存在的 optimization 记录。

## 当前阶段状态

```text
M0 engineering foundation: completed
M1-A behavior measurement: completed
M1-B CE-reference 20000-step: completed
M1-C CE-reference 50000-step extension: completed
M1 overall: completed
M2-A instability analysis: planned
M2-B function-space analysis: planned
Gate 2 seed 2/3 replication: planned
M3 Fourier analysis: planned
```

Canonical CE-reference run：`20260721T045433955396Z_30c62ebc`。M1-B 的正式执行、失败重试、
首次行为事件和审计结果见[实现与运行记录](implementation.md)。行为曲线显示反复进入和离开
高性能区域；这些是行为层观察，坍塌窗口的逐 checkpoint 离线重算与机制解释仍为 planned。

M1-C terminal child 为 `20260724T091041024473Z_c6434d8a`，完成至 step 50000 并通过
`m1c-extension` 审计。首次事件仍冻结为 `t_fit=100`、`t_grok50=6050` 和
`t_grok99=7000`。`t_stable99` 未达到，最长 test accuracy 不低于 0.99 的连续区间为
44 次 evaluation（2150-step span），末端状态为 `recovering`。这些结果完成了 M1
行为证据协议，但不等于形成稳定泛化或机制结论。

## 证据冻结

[`results/m1_ce_reference/`](../results/m1_ce_reference/) 是 20000-step canonical run 的不可变
M1 证据。后续工作不得覆盖、重写或重新导出替换该目录，也不得改写其中冻结的 `t_fit`、
`t_grok50` 和 `t_grok99`。[`results/m1_ce_reference_extended/`](../results/m1_ce_reference_extended/)
是已审计的 50000-step 延长证据，同样不得覆盖或替换。后续 M2-A 结果写入：

```text
results/m2a_stability/
```

派生结果必须保留来源 run、checkpoint、scientific config hash、split hash 和生成代码 commit。

## M1-C：50000-step extension

M1-C 已从 canonical run `20260721T045433955396Z_30c62ebc` 的 `step_020000.pt` 创建 terminal
child `20260724T091041024473Z_c6434d8a`，仅把 `max_steps` 调整为 50000。Scientific config
hash `b167674594bf0944f0b2afb877d2d8c8f5647c0e4e60c64ebb2a511a9f1f7729` 与 split hash
`d0ec6ff924ecc411b9a9d40786f057ec869076b98308e2ecb75da2756c308237` 保持不变，行为时间线
继续使用绝对 global step。

Terminal child 包含 step 50–50000 的 1000 条 scalar、2000 条 train/test offset record、
step 20050–50000 的 600 条 optimization record，以及 step 20000–50000 的 301 个 manifested
checkpoint。逐步优化诊断仅从 extension 起点之后提供；0–20000 没有回填。首次行为事件未被
改写。M1-C 用于测量坍塌是否继续出现、间隔是否变化以及长期稳定区间是否形成，不承担
checkpoint 窗口真实性确认或机制解释。

## M2-A：失稳真实性与坍塌窗口

M2-A 先通过已有 checkpoint 离线重算确认候选坍塌是否属于真实模型状态，再比较坍塌前、谷底、
恢复后的行为、参数尺度、optimizer 状态和 logit 尺度。稳定窗口、坍塌 episode、checkpoint
重算、最小优化诊断和失稳中心化分析使用同一绝对 step 时间线。

20000-step 轨迹登记以下候选 checkpoint 窗口：

```text
1300–1650
3350–4000
5700–6200
7250–7600
8350–8650
9950–10300
12250–12800
14150–14650
15250–15800
17400–17900
18250–18850
```

每个窗口从已有 checkpoint 中选择最接近 `pre-collapse`、`onset`、`trough`、
`early-recovery` 和 `recovered` 的状态。若窗口或轨迹终点尚未满足恢复判据，必须明确记录
为部分恢复或未恢复，不得虚构 `recovered` checkpoint。候选窗口只定位行为异常，不预设
优化或函数空间机制。

## M2-B 与 Gate 2

M2-B 在共享时间线上计算 `D_eq`、`Gamma`、`I`、`L_parallel`、centered-logit Frobenius
norm、prediction entropy、normalized margin、`t_alg` 和 `t_dom`，并叠加 M2-A 的失稳
episode。Reynolds、中心化与函数分解的数学定义保持不变。

M2-A 与 M2-B 分析管线达到稳定状态后，Gate 2 执行 `WD=0.5` 的 CE seed 2、3，仅做行为层
与失稳统计复现。完整多 seed 与 weight-decay 网格仍属于 M4。

## 停止与稳定性规则

M1-C 保留 50000 optimizer step 的预注册上限。首次 `t_grok99` 后继续 20 个 evaluation
interval 只验证首次事件后的后续训练，不再作为进入稳定阶段的充分条件。长期稳定性由
$t_{\mathrm{stable99}}(100)$、longest stable window 和 last collapse step 共同描述；其中
100 个 evaluation interval 需要连续 101 条记录，对应当前协议下首尾相差 5000 optimizer
steps。

本次正式 M1-C 到达预注册 step 50000 上限时，`t_stable99` 仍为 `not_reached`，last
collapse onset 为 step 49950，final state 为 `recovering`。因此停止原因是达到协议上限，
而不是达到稳定窗口。

## GPU 主链路与 M1-B 验收

正式 CE-reference 固定 `eval_interval=50` 与 `checkpoint_interval=100`。训练入口必须先设置
确定性环境和随机状态，再调用任何 CUDA 诊断；模型迁移到 `cuda:0`/FP32 后才能建立 AdamW
parameter groups。PyTorch 2.2 的 AdamW 一阶、二阶矩与参数同处 GPU，标量 step counter 默认位于
CPU；该标量不参与参数张量设备绑定判断，也不通过 `capturable=True` 改变正式优化路径。

最终 run 使用 `transgrokking.cli audit` 生成 `audit/m1_ce_reference.json`。只有生命周期、
lineage、hash、时间轴、manifest、离线 evaluator 和预注册停止规则全部通过时，M1-B 才标记完成。

M1-C 使用：

```bash
conda run --prefix ./env python -m transgrokking.cli audit \
  --run-dir runs/20260724T091041024473Z_c6434d8a \
  --profile m1c-extension
```

该审计原子写入 `audit/m1c_extension.json`，并核验 lineage、唯一配置差异、冻结 hash 与
首次事件、精确 scalar/offset/optimization/checkpoint 时间格点、最终 checkpoint 离线
evaluator、原 M1-B 23 个冻结文件以及不存在 M2+ artifact。全部检查已通过；具体运行数据
见[实现与运行记录](implementation.md)，行为讨论见 [M1 CE-reference 行为讨论](M1-disc.md)。
