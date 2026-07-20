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

## 4. M2 分析

生成：

```text
D_eq
Gamma
I
L_parallel
t_alg
t_dom
```

第一张机制图在共享时间轴展示 \(\Gamma_t\)、\(I_t\)、\(D_{\mathrm{eq}}(t)\) 与 train/test accuracy。图中标注行为事件和函数事件。

## 5. M3 分析

至少包含：

- target-line energy；
- 各频率 frequency-time heatmap；
- 非目标频率能量；
- restricted/excluded loss；
- Reynolds/Fourier 一致性误差；
- Parseval 误差。

图注写明 FFT norm、零频处理、频率集合选择规则和 checkpoint 来源。

## 6. M4 分析

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

## 7. M5 分析

表征报告包含 embedding frequency energy、circle-fit \(R^2\)、hidden-state effective rank、probe accuracy 和候选 head/MLP attribution。

优化报告包含 parameter-group norm、data/decay ratio、radial/tangential update、update cosine 和 checkpoint distance。

因果分支图显示父 checkpoint、修改项、绝对 step、相对 step、状态及核心函数指标。

## 8. M6 分析

Congruence 成对报告比较：

```text
CE
full congruence
late-on
early-only
```

核心量包括 `t_alg`、`t_dom`、各目标频率、梯度范数/夹角和最终函数结构。配对运行使用相同初始化和 split。

## 9. M7 统一报告

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

## 10. 图形规范

- 保存原始曲线。
- 平滑、插值和对数变换显式标注。
- 同类图使用稳定颜色与线型。
- 图题包含 run ID 或运行组 ID。
- 同时保存绘图所用派生表。
- 事件缺失时显示 `not_reached`，禁止用训练终点替代。

## 11. 完成条件

分析脚本通过 fixture 和 smoke run 测试。正式报告检查空数据、NaN、事件标记、schema、checkpoint 来源和 provenance。
