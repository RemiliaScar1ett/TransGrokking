# src/AGENTS.md

## 1. 作用域

本文件适用于 `src/` 及其全部子目录。根目录阶段顺序和固定协议继续有效。

## 2. 依赖方向

```text
config / data / models
          ↓
training / checkpoint
          ↓
metrics / interventions
          ↓
cli
```

`models/` 不得导入训练器。`metrics/` 只接受张量、模型状态或 run artifacts，不得隐式启动训练。`interventions/` 通过明确 branch 接口创建 child run。

## 3. 训练与设备

- `global_step` 表示已经完成的 optimizer update 数量。
- step 0 checkpoint 表示初始化状态。
- 确定性环境变量在首次 CUDA 调用前设置。
- 模型迁移到目标设备后创建 AdamW。
- GPU 测试验证模型参数真实更新。
- 正式基线保持 CE-only、FP32、TF32/AMP 关闭。
- 训练器不得包含绘图、Reynolds、FFT 或大型 activation 分析。

## 4. M1-A 与 M1-B 已有职责

M1 行为指标放在 `metrics/behavior.py`、`metrics/norms.py` 和 `metrics/events.py`。

要求：

- margin 排除正确类别；
- error offset 只统计错误样本；
- 参数模块桶互不重叠；
- FP64 accumulator 用于参数平方和；
- event detector 只读取已提交 scalar；
- offline evaluator 只读 checkpoint 与 split；
- scalar、offset、event 文件支持恢复与 child-run 前缀复制。

M1-B 正式配置固定 `eval_interval=50`、`checkpoint_interval=100`。现有 20000-step CE-reference 及其首次事件保持冻结；延长训练不得改写 `t_fit`、`t_grok50` 或 `t_grok99`。

## 5. M1-C 与 M2-A 已实现职责

M1-C stability metrics、最小 optimization diagnostics、instrumented child-run、审计与导出均已
实现、测试并完成正式运行。正式 terminal child 为
`20260724T091041024473Z_c6434d8a`。M2-A 的 checkpoint 重算、坍塌窗口真实性验证、
deterministic replay 和 bridge audit 也已实现、测试并完成正式分析。

### Stability metrics

```text
src/transgrokking/metrics/stability.py
```

稳定性纯函数包括：

```text
detect_stable_window(...)
detect_collapse_episodes(...)
summarize_stability(...)
```

输出：

```text
t_stable99
collapse_count
last_collapse_step
longest_stable_window
fraction_of_time_above_99
collapse episodes
```

稳定窗口固定为 100 个 evaluation interval，即 101 条连续记录和 5000 optimizer steps。
Train/test primitive episode 独立保存，joint episode 仅作为对二者的一对一 composite 引用。
稳定窗口、坍塌 episode 和恢复状态均从已提交的行为时间线幂等派生。原 `events.json`、
原 M1 schema 和 `results/m1_ce_reference/` 保持不可变。

### Optimization diagnostics

`training/diagnostics.py` 在目标 evaluation update 的 backward 后、optimizer step 前捕获
FP64 CPU snapshot，并在 step 完成后记录：

- gradient L2 norm；
- data-update L2 norm；
- decay-update L2 norm；
- data/decay ratio；
- update cosine；
- Adam first-moment L2 norm；
- Adam second-moment mean、RMS 和 max；
- decay/no-decay group 的 gradient、update 和 moment 统计。

参数和更新统一使用欧氏 L2；矩阵及高阶张量展平后的数值与 Frobenius norm 等价。Optimizer state 统计与模型参数范数分开保存。诊断过程不得修改 gradient、optimizer state 或训练更新。

0–20000 step 的 canonical run 没有逐步 optimization diagnostics；正式 M1-C 只在
20050–50000 step 每 50 step 保存一条，共 600 条，不得回填或推测。LayerNorm 与
embedding parameter L2 继续由行为 scalar 的稳定模块范数提供，不伪装成 0–20000 step
optimizer diagnostics。

### M1-C artifacts 与提交顺序

Instrumented child run 使用：

```text
measurement.resolved.yaml
metrics/stability.json
metrics/collapse_episodes.json
metrics/optimization.jsonl
audit/m1c_extension.json
```

每次 evaluation 按 error-offset pair、optimization record、scalar commit marker、
events/stability/collapse 原子重建的顺序提交。Resume 只允许修复当前 interrupted inplace run；
branch source 必须只读验证。Retry child 继承恢复 checkpoint 之前的已提交前缀和原始
diagnostics origin。

这些文件属于新增派生或延长运行证据，不能覆盖原 M1 schema、原 canonical run 或
`results/m1_ce_reference/`。`export-m1c` 只读已通过审计的 run，以临时目录原子发布并拒绝
覆盖已有目标。

### M2-A 已实现职责

`analysis/checkpoint_resolver.py` 只读解析 root、M1-B child 和 M1-C child 的 manifest，保留
物理 checkpoint 的 raw SHA，并以稳定 parameter name、optimizer state 和 RNG 的规范摘要
识别 branch-anchor semantic alias。正式 M2 现场确认 503 个物理 checkpoint、501 个唯一
regular state 和 step 5000/20000 的语义等价 alias。

`analysis/replay.py` 按 lineage segment 从前一 100-step checkpoint 重放目标 50-step 状态，
重复生成 midpoint 后继续到下一 manifested checkpoint 完成 bridge。Bridge 比较 model、
optimizer、RNG、global step、parameter-group signature 和行为指标，并在前后核对 source 与
endpoint checkpoint SHA；replay cache 只能写入 `analysis_runs/`。

M2-A 已对强制 collapse state 执行离线行为重算和 committed scalar 对齐。M1-C optimization
diagnostics 可用于描述性对齐，但不得单独解释为因果机制。未恢复 episode 保持
`terminal_unrecovered`，不得填充 recovery 字段。

## 6. M2-B 已实现职责

实现路径包括：

```text
src/transgrokking/metrics/function_space.py
src/transgrokking/metrics/audit_m2.py
src/transgrokking/reporting/m2.py
```

完整 logits 统一 shape：

```text
[p, p, p]
[a, b, candidate_c]
```

中心化：

\[
\widetilde z(a,b,c)=z(a,b,c)-\frac1p\sum_jz(a,b,j).
\]

Reynolds offset profile：

\[
g(d)=\frac1{p^2}\sum_{a,b}\widetilde z(a,b,a+b+d).
\]

返回：

```text
centered_logits
offset_profile
equivariant_logits
residual_logits
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

M2-B 数学函数保持纯函数性质。完整 logits evaluator 使用可配置 batch size、FP32 CUDA
forward、CPU offload 和 FP64 reduction；逐 checkpoint 流式释放 logits。正式分析覆盖 501 个
regular state 和 48 个 replay state，关键 tensor 留在 `analysis_runs/`，Git 导出只包含 manifest、
offset profile 和聚合结果。

M2 analysis audit 检查只读 lineage、replay bridge、行为重算与 Reynolds 不变量；portable
export audit 再检查相对 POSIX provenance、文件 SHA、schema、冻结 M1 清单和 M3+ 禁止项。
实际接口为 `analyze-m2`、`audit --profile m2-function-space` 和 `export-m2`。Gate 2 与 M3
仍为 `planned`，不得由 M2 代码隐式启动。

## 7. M3 计划职责

统一 FFT：

```python
torch.fft.fftn(x, dim=(0, 1, 2), norm="ortho")
```

目标线索引：

\[
(r,r,-r\bmod p).
\]

实现 target-line mask、频率能量、inverse transform、restricted/excluded logits。Complex tensor 保留到指标层，绘图层再转换为能量和相位。

## 8. M4 实现职责

M4 主要增加运行矩阵配置与批量调度。每个运行独立创建 run ID。禁止复用模型状态模拟独立 seed 或 WD 条件。

完整矩阵固定包含：

```text
seed 1,2,3 with WD=0.5
WD=0,0.1,1.0 with seed 1,2,3
```

其中 seed 2/3、WD=0.5 的行为层 Gate 2 可在 M3 前执行，但 M4 仍负责完整函数/Fourier 汇总与 WD 网格。批量调度器只负责串行或受控并发启动，不能改变单次 run 的科学配置。

## 9. M5 实现职责

### 表征

稳定 hook 名称覆盖：

```text
embed.token
embed.position
blocks.<i>.attention.head_output
blocks.<i>.residual.mid
blocks.<i>.mlp.pre
blocks.<i>.mlp.post
blocks.<i>.residual.post
residual.final
residual.final_normalized
logits
```

实现 embedding Fourier、circle fit、hidden-state DFT、effective rank 和 linear probe。

### 优化动力学

按实际 AdamW parameter group 记录：

```text
parameter_norm
gradient_norm
data_update_norm
decay_update_norm
radial_update_norm
tangential_update_norm
data_decay_cosine
```

更新分解必须与真实 optimizer step 对齐。近似实现需要写入 metadata。

### 干预

Branch runner 支持 WD 分支、optimizer reset、模块冻结、frequency ablation、activation patching 和模块移植。M5 只处理 CE-only 条件。

## 10. M6 实现职责

Congruence loss：

\[
L_{\mathrm{cong}}=
\sum_k P_\theta(k\mid a,b)
\left[1-\cos\frac{2\pi(k-y)}p\right].
\]

返回：

```text
total
cross_entropy
congruence
```

实现 loss schedule、共享初始化、成对运行和模块级梯度范数/夹角。Schedule 事件来源于冻结的 M1 事件定义。

## 11. Checkpoint 与 schema

Checkpoint 加载验证：

- schema version；
- scientific config hash；
- split hash；
- model state shape；
- optimizer type 与 parameter-group signature；
- global step 与 RNG state。

旧 schema 禁止静默加载。迁移函数必须显式调用并有回归测试。

## 12. 资源约束

完整 logits 逐 checkpoint 生成。Activation 只在指定 checkpoint 和模块提取。分析完成后立即转移到 CPU 并释放 GPU 引用。禁止保存全时间线的完整 activation。

## 13. 局部完成条件

```bash
conda run --prefix ./env python -m pytest -q tests/unit
conda run --prefix ./env python -m ruff check src tests
conda run --prefix ./env python -m ruff format --check src tests
```

影响训练、CLI、checkpoint 或 branch runner 时继续运行对应 integration smoke。
