# analysis/AGENTS.md

## 1. 作用域

本文件适用于 `analysis/`。分析脚本读取已有 run artifacts，生成派生数据、图表和报告。分析命令不得隐式训练模型。

## 2. 数据与资源边界

- 每次只加载一个 checkpoint。
- 完整 logits 支持分批生成。
- Logits 与 activation 优先转移到 CPU。
- Reynolds、FFT、统计和绘图默认在 CPU 执行。
- 派生数据写入独立目录，禁止覆盖训练日志。
- 多运行对比先检查任务、模型、split、schema 和科学配置兼容性。
- OOM 重试记录 batch size 与 offload 策略。

## 3. M1 分析

生成行为时间线：

```text
train/test CE
train/test accuracy
margin quantiles
error count/rate
parameter norms
t_fit
t_grok50
t_grok99
```

正式 CE-reference 报告需要验证停止规则、设备 metadata、checkpoint manifest 和离线 evaluator 一致性。

## 4. M1-C 稳定性导出

M1-C 的 scalar-derived stability 已完成正式运行、审计和导出。只读证据为：

```text
source run: 20260724T091041024473Z_c6434d8a
output: results/m1_ce_reference_extended/
t_stable99: not_reached
final_state: recovering
```

导出包含原始行为与 optimization CSV、stable-window timeline、collapse timeline、episode
JSON、参数/更新曲线、PNG/SVG、provenance 和 audit 摘要。所有曲线使用原始 evaluation
step，不平滑、不插值。原 M1 首次事件和 `results/m1_ce_reference/` 保持冻结。

M1-C 从 scalar 时间线观察到 train primitive 26 个、test primitive 10 个、joint composite
10 个；joint 计数不是新增的独立 primitive episode。最长 test accuracy 不低于 99% 的区间为
44 条 evaluation、2150-step span，尚不足 101 条 evaluation 的稳定窗口。以上只属于行为与
optimizer 记录层的 `observed` 结果，不构成 checkpoint 函数真实性或机制结论。

## 5. M2-A 坍塌窗口报告

M2-A 仍为 `planned`。它必须在代表 checkpoint 上离线重算行为与函数量，不能把 M1-C 的
scalar-derived episode 或最终 checkpoint 审计直接当作每个坍塌窗口的真实性证据。

### 代表窗口

每个代表 episode 展示：

```text
pre-collapse
onset
trough
early-recovery
recovered
```

共同展示：

- train/test CE；
- train/test accuracy；
- train/test margin；
- parameter-group L2；
- `D_eq`、`Gamma`、`I`；
- centered-logit Frobenius norm；
- prediction entropy；
- normalized margin；
- 可用时的 optimization diagnostics。

报告必须区分行为时间线观察、checkpoint 离线重算和机制解释。若 episode 在时间线终点仍未恢复，显式记录 `not_recovered`，不得用终点代替 recovery step。

### 跨事件对齐

M2-A 需要生成 collapse depth、recovery duration、relative-event aligned curves，并按
pre-collapse、onset、trough、early-recovery、recovered 对齐。尚未恢复的 episode 必须与
完整 episode 分开汇总。

## 6. M2-B 函数空间分析

生成：

```text
D_eq
Gamma
I
L_parallel
centered_logit_frobenius_norm
prediction_entropy
normalized_margin
t_alg
t_dom
```

第一张机制图在共享时间轴展示 \(\Gamma_t\)、\(I_t\)、\(D_{\mathrm{eq}}(t)\) 与 train/test accuracy。图中标注首次行为事件、稳定窗口、坍塌 episode 和函数事件。M2-B 不得把行为层同步变化直接表述为机制因果。

## 7. Gate 2 多 seed 行为层复现

在 M2-A 与 M2-B 形成稳定分析管线后，对 CE、WD=0.5、seed 2/3 生成行为与失稳统计。跨 seed 比较同时采用：

- 绝对 step；
- 相对 collapse onset；
- 相对 first grokking event。

不得只按绝对 step 直接平均。Gate 2 不包含完整 WD 网格，也不替代 M4 的函数空间、Fourier 和全矩阵汇总。

## 8. M3 分析

至少包含：

- target-line energy；
- 各频率 frequency-time heatmap；
- 非目标频率能量；
- restricted/excluded loss；
- Reynolds/Fourier 一致性误差；
- Parseval 误差。

图注写明 FFT norm、零频处理、频率集合选择规则和 checkpoint 来源。

## 9. M4 分析

汇总 seed 复制和 WD 网格：

```text
event order
grokking delay
Gamma/I/D_eq relation
target-line energy
margin-norm efficiency
failure/interruption rate
```

报告单次轨迹与跨 seed 聚合。样本量较小时保留全部点，不使用误导性的平滑分布。

## 10. M5 分析

表征报告包含 embedding frequency energy、circle-fit \(R^2\)、hidden-state effective rank、probe accuracy 和候选 head/MLP attribution。

优化报告包含 parameter-group norm、data/decay ratio、radial/tangential update、update cosine 和 checkpoint distance。

因果分支图显示父 checkpoint、修改项、绝对 step、相对 step、状态及核心函数指标。

## 11. M6 分析

Congruence 成对报告比较：

```text
CE
full congruence
late-on
early-only
```

核心量包括 `t_alg`、`t_dom`、各目标频率、梯度范数/夹角和最终函数结构。配对运行使用相同初始化和 split。

## 12. M7 统一报告

推荐结构：

```text
analysis/reports/<run_or_group_id>/
├── summary.md
├── provenance.json
├── derived_metrics.parquet
├── events.json
└── figures/
```

`summary.md` 分开记录观察、解释和限制。每条机制解释附指标、图表或干预证据。

## 13. 图形规范

- 保存原始曲线。
- 平滑、插值和对数变换显式标注。
- 同类图使用稳定颜色与线型。
- 图题包含 run ID 或运行组 ID。
- 同时保存绘图所用派生表。
- 事件缺失时显示 `not_reached`，禁止用训练终点替代。

## 14. 完成条件

分析脚本通过 fixture 和 smoke run 测试。正式报告检查空数据、NaN、事件标记、schema、checkpoint 来源和 provenance。
