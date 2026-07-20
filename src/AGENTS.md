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

## 4. M1 实现职责

M1 行为指标放在 `metrics/behavior.py`、`metrics/norms.py` 和 `metrics/events.py`。

要求：

- margin 排除正确类别；
- error offset 只统计错误样本；
- 参数模块桶互不重叠；
- FP64 accumulator 用于参数平方和；
- event detector 只读取已提交 scalar；
- offline evaluator 只读 checkpoint 与 split；
- scalar、offset、event 文件支持恢复与 child-run 前缀复制。

M1 正式配置固定 `eval_interval=50`、`checkpoint_interval=100`。训练控制器支持 5000、20000、50000 step 的安全延长，并验证 `t_grok99` 后 20 个 evaluation interval。

## 5. M2 实现职责

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
```

M2 函数保持纯函数性质。完整 logits evaluator 支持 batch size、device、dtype 和 CPU offload。关键 checkpoint 可以持久化张量，其余 checkpoint 允许离线重建。

## 6. M3 实现职责

统一 FFT：

```python
torch.fft.fftn(x, dim=(0, 1, 2), norm="ortho")
```

目标线索引：

\[
(r,r,-r\bmod p).
\]

实现 target-line mask、频率能量、inverse transform、restricted/excluded logits。Complex tensor 保留到指标层，绘图层再转换为能量和相位。

## 7. M4 实现职责

M4 主要增加运行矩阵配置与批量调度。每个运行独立创建 run ID。禁止复用模型状态模拟独立 seed 或 WD 条件。

矩阵固定包含：

```text
seed 2,3 with WD=0.5
WD=0,0.1,1.0 with seed 1,2,3
```

批量调度器只负责串行或受控并发启动，不能改变单次 run 的科学配置。

## 8. M5 实现职责

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

## 9. M6 实现职责

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

## 10. Checkpoint 与 schema

Checkpoint 加载验证：

- schema version；
- scientific config hash；
- split hash；
- model state shape；
- optimizer type 与 parameter-group signature；
- global step 与 RNG state。

旧 schema 禁止静默加载。迁移函数必须显式调用并有回归测试。

## 11. 资源约束

完整 logits 逐 checkpoint 生成。Activation 只在指定 checkpoint 和模块提取。分析完成后立即转移到 CPU 并释放 GPU 引用。禁止保存全时间线的完整 activation。

## 12. 局部完成条件

```bash
conda run --prefix ./env python -m pytest -q tests/unit
conda run --prefix ./env python -m ruff check src tests
conda run --prefix ./env python -m ruff format --check src tests
```

影响训练、CLI、checkpoint 或 branch runner 时继续运行对应 integration smoke。
