# tests/AGENTS.md

## 1. 作用域

本文件适用于 `tests/`。测试验证数学定义、确定性、恢复能力、阶段门和产物 schema。

## 2. 基础原则

- 默认使用 CPU。
- 单元测试保持秒级。
- 长程 Grokking 结果不得作为测试前提。
- CUDA 测试使用 `@pytest.mark.cuda`。
- 目标 GPU 不可用时允许跳过硬件测试，并在报告中说明。
- 所有命令通过 `conda run --prefix ./env` 执行。

## 3. M0 测试

覆盖环境 doctor、数据完整性、模型 causal 行为、hook shape、checkpoint round-trip、RNG 恢复、原子写入和 child-run 恢复。

GPU smoke 需要验证：

- 设备名称与显存；
- optimizer 引用与模型参数对象一致；
- 至少一个模型参数发生非零更新；
- optimizer state 位于目标设备；
- peak allocated/reserved VRAM 有效。

## 4. M1 测试

验证：

- CE、accuracy、margin 和 quantile；
- error offset 模运算与直方图总数；
- 模块范数平方和重建总范数；
- decay/no-decay group 覆盖；
- `t_fit`、`t_grok50`、`t_grok99` 连续窗口；
- scalar/offset/events 原子协议；
- resume 和 child-run 时间线前缀；
- read-only evaluator 无副作用；
- 固定基准配置中的 evaluation/checkpoint interval。

M1-B 正式 run 验收使用独立审计脚本。现有 20000-step canonical run、首次事件和原 `events.json` 保持冻结。

## 5. M1-C 已测试覆盖

M1-C 自动测试已覆盖：

- stable-window 的 101-record 边界、时间格点缺口和乱序拒绝；
- 首次达到 99% 后再次坍塌；
- 多次坍塌与恢复；
- episode 横跨 parent/child run；
- 序列终点尚未恢复；
- 稳定窗口从未达到；
- 未恢复 episode 内的再次低值不拆分，以及 joint tolerance 配对；
- 重复或乱序 step；
- 稳定性派生文件重算幂等；
- 原 `events.json` 保持不变；
- instrumented child 的 scalar/offset/optimization 前缀与原子失败恢复；
- M1-C lineage audit 的正例和故障注入反例；
- 优化诊断无副作用；
- data update 与 decay update 重构；
- 参数、gradient、update 和 Adam moment 的 L2/summary 数值；
- `grad=None`、零分母和未定义 cosine 的 JSON `null`；
- 真实 CLI child run、目标 CUDA update 和 measurement 启用前后的确定性参数一致性；
- `export-m1c` 的拒绝覆盖、原子发布、provenance、CSV 和 figure 清单。

长期 Grokking 是否发生、具体坍塌时间或 canonical run 中的候选时间点均不得成为单元测试前提。测试使用人工时间线、小模型或 fixture checkpoint，验证定义和协议而不是重现某条正式科学轨迹。

## 6. M2-A 已测试覆盖

M2-A 自动测试已覆盖：

- physical checkpoint、canonical state 与 branch-anchor alias 的 lineage 解析；
- semantic state hash、相同语义去重和冲突状态拒绝；
- exact checkpoint 与 50-step deterministic replay；
- 两次独立 midpoint 一致性及继续到下一 manifested checkpoint 的 bridge；
- model、optimizer、RNG、global step、parameter-group signature 与行为指标比较；
- root/M1-B/M1-C segment ownership、parent/child boundary 和 source zero-write；
- episode role、多角色 state、split-specific trough 和 `terminal_unrecovered`；
- checkpoint 重算与 committed scalar 的容差对齐；
- analysis audit 的 schema、source inventory、故障注入和失败门控。

正式 M2-A audit 还现场验证了 503 个物理 checkpoint、501 个 regular state、48 个 replay
bridge 与全部强制 episode 状态。长期 Grokking 或具体 episode step 仍不得成为单元测试前提。

## 7. M2-B 已测试覆盖

参数化小模数：

```text
p in {2,3,5,7,8}
```

M2-B 自动测试覆盖中心化、Reynolds 幂等性、正交性、重构、Pythagorean energy、群生成元
不变性、offset profile 与 brute-force 轨道平均、`sum(g)` 和 residual projection。`Gamma/I`
测试显式排除正确类别，覆盖 implication buffer、数值歧义和 projected split 不变量。

同时测试 normalized tolerance、finite JSON、regular-grid crossing/run/exit 计数、FP32
forward 与 FP64 reduction、CPU/CUDA 容差、selected tensor policy、原子 export/no-overwrite，
以及 portable export audit 的路径、SHA、schema、冻结 M1 清单和 M3+ 禁止项。

## 8. Gate 2 计划测试

Gate 2 只验证 CE、WD=0.5、seed 2/3 的行为时间线、稳定性派生结果、provenance 和 lineage。不得把完整 WD 网格或 Fourier 结果提前纳入 Gate 2。

## 9. M3 计划测试

验证 FFT/IFFT round-trip、Parseval、目标线索引、负频模索引、共轭对称性、Reynolds/Fourier 等价和 \(E_{\mathrm{line}}=1-D_{\mathrm{eq}}\)。

Restricted 与 excluded logits 需要通过重构测试。

## 10. M4 测试

验证运行矩阵展开结果、seed/WD 配置组合、run ID 唯一性和 scientific config 差异。批量调度器不得共享模型或 optimizer state。

## 11. M5 测试

验证 hook 命名与 shape、circle fit、effective rank、probe 冻结、update 分解、WD 分支、optimizer reset、模块冻结、frequency ablation、patching 和模块移植兼容性。

干预测试使用小模型和短程分支，避免依赖真实 Grokking。

## 12. M6 测试

验证 congruence loss 数值、梯度、关闭时零值、schedule 切换、共享初始化和成对配置一致性。CE 与 congruence 梯度范数及夹角计算需要处理零梯度。

## 13. M7 报告测试

使用 fixture run 生成完整报告，检查图表、事件表、派生数据、NaN、缺失 checkpoint 和旧 schema 错误。

## 14. 完成条件

当前阶段相关测试必须通过。正式 GPU 结果只由目标 RTX 4060 Laptop 8GB 运行提供。跳过、失败和近似测试均写入阶段报告。
