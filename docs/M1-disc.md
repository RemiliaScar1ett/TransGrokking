# M1 CE-reference 行为讨论

## 1. 文档范围

本文汇总 M1-B 与 M1-C 已审计的行为层证据。它描述首次事件、稳定窗口、坍塌
episode、参数尺度和 M1-C extension 期间的最小优化诊断，不解释 Reynolds 结构、
Fourier 结构、内部表征或因果机制。M2-A 将负责 checkpoint 窗口重算，M2-B 将负责
函数空间与群对称性分析。

原 20000-step 证据目录
[`results/m1_ce_reference/`](../results/m1_ce_reference/) 保持不可变；50000-step
延长证据位于
[`results/m1_ce_reference_extended/`](../results/m1_ce_reference_extended/)。

## 2. 证据身份

| 项目 | 值 |
| --- | --- |
| M1-B canonical run | `20260721T045433955396Z_30c62ebc` |
| M1-C terminal child | `20260724T091041024473Z_c6434d8a` |
| M1-C 训练代码 commit | `8d620a84930c85d617131d1062a01e640b667a32` |
| Parent checkpoint | `step_020000.pt` |
| Final step | 50000 |
| Scientific config hash | `b167674594bf0944f0b2afb877d2d8c8f5647c0e4e60c64ebb2a511a9f1f7729` |
| Split hash | `d0ec6ff924ecc411b9a9d40786f057ec869076b98308e2ecb75da2756c308237` |
| Measurement config hash | `182d1b705795f54acc683e751312b314bd8fd42f531f95d5c1491bb739a7b733` |
| Evaluation interval | 50 optimizer steps |
| Checkpoint interval | 100 optimizer steps |
| M1-C audit | passed |

M1-C 相对 M1-B 只把 `optimization.max_steps` 从 20000 改为 50000。任务、模型、
AdamW、learning rate、weight decay 与 parameter groups、CE-only loss、FP32 数值策略、
seed、split 和设备均保持不变。完整 provenance 与审计结果见
[`provenance.json`](../results/m1_ce_reference_extended/provenance.json) 和
[`audit/m1c_extension.json`](../results/m1_ce_reference_extended/audit/m1c_extension.json)。

## 3. 操作性定义

原 M1 首次事件保持冻结：

| 事件 | 首次窗口起点 | 确认 step |
| --- | ---: | ---: |
| `t_fit` | 100 | 300 |
| `t_grok50` | 6050 | 6150 |
| `t_grok99` | 7000 | 7100 |

`t_stable99` 要求从 `t_grok99` 起存在连续 101 条 evaluation record，其 test
accuracy 均不低于 0.99；在当前间隔下，该窗口首尾相差 5000 optimizer steps。

Train collapse 在 `t_fit` 确认后、train accuracy 首次低于 0.9 时开启；test collapse
在 `t_grok99` 确认后、test accuracy 首次低于 0.9 时开启。恢复要求对应 accuracy
连续 3 次回到 train 0.999 或 test 0.99。Train/test onset 相差不超过一个 evaluation
时，另建立引用两个 primitive episode 的 joint composite。以上都是操作性测量定义，
不是机制判据。

## 4. 稳定性结果

| 指标 | 实际结果 |
| --- | ---: |
| `t_stable99` | `not_reached` |
| Train primitive episodes | 26 |
| Test primitive episodes | 10 |
| Joint composite episodes | 10 |
| Last collapse onset | 49950 |
| Test accuracy ≥ 0.99 的最长连续区间 | step 10150–12300 |
| 最长区间长度 | 44 evaluations / 2150 steps |
| Test evaluations 中 accuracy ≥ 0.99 的比例 | 0.202 |
| Final state | `recovering` |
| Final train accuracy | 1.0 |
| Final test accuracy | 0.963513970375061 |

因此，首次 `t_grok99=7000` 只表明模型第一次连续三次进入 99% test accuracy 区域。
到 step 50000 为止，没有出现预注册的 5000-step stable99 窗口。

20000-step 前已有 11 个 train、8 个 test 和 8 个 joint episode。Extension 新增
15 个 train episode，并新增 onset 为 21900 和 27300 的两个 test/joint episode。
`test_008` 从 parent 的 step 18350 开始，在 child 的 step 21300 才进入恢复窗口，
并在 step 21400 被确认，说明派生器能够在同一绝对时间线上处理跨 parent/child 的
episode。

`test_010` 从 step 27300 开始，到终点仍未满足连续三次 test accuracy ≥ 0.99 的恢复
判据。最后一个 train episode 在 step 49950 开启；step 50000 的 train accuracy 已回到
1.0，但只有一条恢复记录，因此仍为 `not_recovered`。这两点共同解释 terminal
`final_state=recovering`，而不是用终点单点替代恢复事件。

全部 onset、trough、depth、recovery 和确认 step 保存在
[`collapse_episodes.json`](../results/m1_ce_reference_extended/collapse_episodes.json)；
原始曲线和标注图保存在扩展结果目录，不含平滑、插值、异常点删除或缺失 step 补齐。

## 5. 参数尺度与优化诊断

| 指标 | Step 20000 | Step 50000 |
| --- | ---: | ---: |
| Total parameter L2 | 37.0234510911 | 45.2118762728 |
| Decay-group parameter L2 | 18.0065635094 | 18.0784628163 |
| No-decay-group parameter L2 | 32.3496460766 | 41.4401126724 |

在这两个端点之间，总参数 L2 的增加主要与 no-decay group 的尺度增加同时出现，而
decay-group L2 接近原尺度。这是参数分组的描述性比较，不能单独推出函数复杂度、
优化原因或因果方向。

M1-C 从 step 20050 到 50000 每 50 step 记录一次 optimization diagnostic，共 600 条。
字段包括 gradient L2、实际 total update、解析的 AdamW decay update、残差 data update、
data/decay ratio 与 cosine，以及 Adam 一阶矩 L2 和二阶矩 mean/RMS/max。参数、gradient、
update 与 moment 使用分离的统计口径；0–20000 不存在逐步优化诊断，也没有进行回填。

这些曲线与 collapse 标记共享时间轴，可用于 M2-A 选择候选关系，但本阶段没有进行
统计显著性检验、checkpoint 状态重算或因果干预。观察到的同步变化不得写成失稳机制。

## 6. 审计与限制

M1-C audit 验证了：

- terminal lifecycle 为 `completed`；
- canonical lineage、checkpoint SHA、scientific config hash 和 split hash 一致；
- canonical 的 0–20000 scalar/offset 前缀逐条一致；
- scalar、offset、optimization 和 checkpoint 时间格点完整；
- 301 个 extension checkpoint 与 manifest 一致；
- step 50000 离线 evaluator 与最终时间线记录一致；
- 稳定性与 episode 文件可以从 scalar 时间线幂等重建；
- 原 M1-B 的 23 个冻结结果文件逐字节不变；
- run 中没有 congruence、Reynolds、Fourier 或其他 M2+ artifact。

该审计只对最终 checkpoint 执行完整离线 behavior evaluator。各 collapse onset、trough
和 recovery checkpoint 的逐窗口离线重算仍属于 M2-A。Seed 2/3 尚未运行，因此目前不能
判断反复失稳是否跨 seed 复现。

## 7. M2 交接

M1-C 的工程、正式运行、审计和导出门均已通过，允许进入 M2-A。M2-A 应优先：

1. 对代表性 onset、trough、early-recovery 和 recovered checkpoint 重算行为指标；
2. 覆盖跨 lineage 的 `test_008`、长期未恢复的 `test_010` 和 terminal
   `train_026`；
3. 将 parameter-group L2 与 extension-only optimization diagnostics 对齐，但保持
   相关性与因果解释分离；
4. 明确哪些 evaluation step 没有同 step checkpoint，并使用最近的已存在 checkpoint；
5. 在 checkpoint 真实性确认后，再由 M2-B 计算函数空间与群对称性指标。

在 M2-A/M2-B 完成前，不产生 Reynolds、Fourier、表征、电路或因果机制结论。
