# src/AGENTS.md

## 1. 作用域

本文件适用于 `src/` 目录及其全部子目录。根目录 `AGENTS.md` 的规则继续有效。

## 2. 包结构与依赖方向

推荐包结构：

```text
src/transgrokking/
├── cli.py
├── config.py
├── data.py
├── models/
├── training/
├── metrics/
├── interventions/
└── utils/
```

依赖方向保持单向：

```text
config/data/models
        ↓
training
        ↓
metrics/interventions
        ↓
cli
```

`models/` 不得导入训练器或分析脚本。`metrics/` 可以依赖张量与模型输出，不得启动训练。`analysis/` 不属于可安装核心包。

## 3. 数据模块

数据模块需要提供：

- 完整模加法表的确定性生成；
- 固定 seed 的划分；
- train/test 索引；
- split hash；
- 完整输入网格；
- 配置一致性校验。

基础标签满足：

\[
y=(a+b)\bmod p.
\]

必须验证：

- 样本总数为 \(p^2\)；
- 每个有序输入对只出现一次；
- train/test 互斥；
- train/test 并集覆盖完整表；
- 相同 seed 产生相同划分。

数据量很小，基础训练采用 full-batch tensor。接口仍需允许分析前向分批执行。

## 4. 模型模块

实现透明的小型 Transformer，要求：

- 显式 token embedding、position embedding、attention、MLP、LayerNorm 和 unembedding；
- dropout 由配置控制，基线默认 0；
- causal mask 行为有测试；
- 支持返回最终 logits；
- 支持按名称注册 hook；
- 支持捕获 per-head output 和 MLP activation；
- 所有张量 shape 在 docstring 中说明；
- 初始化完全受 seed 控制。

推荐 forward 接口：

```python
logits = model(tokens)
logits, cache = model.run_with_cache(tokens, names_filter=...)
```

`cache` 使用稳定的层级名称。名称一旦进入分析文件，修改时需要提供迁移说明。

## 5. 训练模块

训练器负责：

- 构建模型、损失和优化器；
- full-batch step；
- 定期评估；
- checkpoint 保存与恢复；
- 运行状态更新；
- scalar logging；
- 失败时写入状态。

训练器不得内置绘图和大型 Fourier 分析。重型指标通过 evaluator 在指定 step 运行。

训练循环中的 step 定义必须统一：`global_step` 表示已经完成的 optimizer update 数量。step 0 checkpoint 表示初始化状态。

## 6. 损失模块

交叉熵是基础目标。congruence loss 作为可配置附加项：

\[
L=L_{\mathrm{CE}}+\lambda_{\mathrm{cong}}L_{\mathrm{cong}}.
\]

返回结构至少包含：

```text
total
cross_entropy
congruence
```

关闭某个损失时仍记录明确的零值或禁用状态。禁止通过删除日志字段表达禁用。

## 7. 完整 logit 张量

统一 shape：

```text
[p, p, p]
```

轴顺序固定为：

```text
[a, b, candidate_c]
```

输入批次可以展平为 `[p*p, 2]`，输出 reshape 回 `[p, p, p]`。

中心化定义：

\[
\widetilde z(a,b,c)
=
z(a,b,c)-\frac1p\sum_jz(a,b,j).
\]

中心化函数需要支持任意浮点 dtype，并避免原地修改输入。

## 8. Reynolds 投影

推荐直接计算 offset profile：

\[
g(d)
=
\frac1{p^2}
\sum_{a,b}
\widetilde z(a,b,a+b+d).
\]

再构造：

\[
z^\parallel(a,b,c)=g(c-a-b),
\qquad
z^\perp=\widetilde z-z^\parallel.
\]

函数返回：

```text
centered_logits
offset_profile
equivariant_logits
residual_logits
```

必须满足：

- \(\Pi^2z=\Pi z\)；
- \(z^\parallel\) 对群作用保持不变；
- \(\langle z^\parallel,z^\perp\rangle\approx0\)；
- 重构误差接近浮点精度；
- 常数类别偏置在中心化后消失。

## 9. Margin 与干扰

统一正确标签：

\[
y=(a+b)\bmod p.
\]

算法 margin：

\[
\Gamma=g(0)-\max_{d\ne0}g(d).
\]

残差干扰：

\[
I=
\max_{a,b,c\ne y}
[z^\perp(a,b,c)-z^\perp(a,b,y)].
\]

实现时禁止把正确类别放入错误类别最大值。测试应构造 \(\Gamma>I\) 和 \(\Gamma\le I\) 的人工案例。

## 10. Fourier 约定

使用 `torch.fft.fftn` 和 `torch.fft.ifftn`。整个项目统一 `norm="ortho"`，除非文档明确选择其他约定。

目标频率线索引：

\[
(r,r,-r\bmod p).
\]

需要处理 Python 负索引与模索引的差异。函数返回 complex tensor；能量定义为

\[
|\widehat z|^2
=
\operatorname{Re}(\widehat z)^2+
\operatorname{Im}(\widehat z)^2.
\]

Reynolds mask 与显式投影需要在容差内一致。禁止只比较归一化比例，原始张量也要检查。

## 11. 表征指标

embedding Fourier 分析沿 token 轴执行。circle fit 使用正弦和余弦基，并返回：

```text
frequency
explained_variance
cos_direction
sin_direction
residual_norm
```

effective rank 默认使用 participation ratio：

\[
r_{\mathrm{eff}}
=
\frac{(\sum_i\lambda_i)^2}{\sum_i\lambda_i^2}.
\]

当协方差总能量接近零时，返回受控值并记录警告。

linear probe 需要固定训练 split、正则化和随机种子。probe 结果属于诊断指标，禁止回传梯度到主模型。

## 12. 优化动力学

AdamW 数据更新与 decay 更新需要按实际 parameter group 记录。每个 parameter group 的 learning rate 和 weight decay 都可能不同。

记录模块级：

```text
parameter_norm
gradient_norm
data_update_norm
decay_update_norm
radial_update_norm
tangential_update_norm
data_decay_cosine
```

零范数和零梯度情况需要安全处理。任何近似分解都要在返回 metadata 中标记。

LayerNorm 与 bias 是否施加 decay 必须由配置明确决定。默认配置应提供单独 parameter groups。

## 13. Checkpoint

checkpoint 写入包含版本号。加载时验证：

- schema version；
- model config；
- modulus；
- split hash；
- 参数 shape；
- optimizer 类型；
- global step。

版本不兼容时给出清晰错误。允许提供显式迁移函数，禁止静默忽略字段。

## 14. RTX 4060 Laptop 8GB 资源约束

基础 \(p=97\) 规模下优先选择可读的向量化实现。正式训练目标设备为 `cuda:0` 上的 NVIDIA GeForce RTX 4060 Laptop GPU 8GB。

完整 FP32 logit 张量约占 3.5 MiB，可以在单个 checkpoint 上完整构造。中间 activation 的规模随模块、样本和 checkpoint 数量增长，需要遵守以下规则：

- 训练期间只记录 scalar 和少量聚合张量。
- activation hook 默认关闭，只在配置指定的 checkpoint 与模块开启。
- 分析按单个 checkpoint 顺序执行，完成后把派生张量转移到 CPU 并释放引用。
- full-table forward 提供可配置批量，默认值需通过 8GB 平台 smoke profile 确定。
- Fourier 与 Reynolds 派生计算默认允许在 CPU 完成，避免长期占用显存。
- 禁止缓存全部 checkpoint 的 hidden state、attention pattern 或 MLP activation。
- 不在训练内循环中反复调用 `torch.cuda.empty_cache()`。该函数只可用于明确的分析阶段边界或 OOM 恢复前清理。
- 记录峰值 allocated/reserved VRAM，并把分析批量写入产物 metadata。
- CUDA OOM 时优先减小分析批量或启用 CPU offload。训练超参数保持原配置，失败运行保留。

函数参数支持 `device`、`dtype`、analysis batch size 和 CPU offload。CPU 单元测试不得依赖 CUDA。正式 GPU 运行需要先通过根目录规定的 `doctor` 严格检查。

## 15. 局部完成条件

修改 `src/` 后至少运行：

```bash
conda run --prefix ./env python -m pytest -q tests/unit
conda run --prefix ./env python -m ruff check src tests
conda run --prefix ./env python -m ruff format --check src tests
```

涉及 CLI、训练或 checkpoint 时，继续运行 smoke training 和恢复测试。
