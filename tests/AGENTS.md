# tests/AGENTS.md

## 1. 作用域

本文件适用于 `tests/`。测试承担数学定义、确定性和恢复能力的主要验证责任。

## 2. 测试原则

- 默认在 CPU 上运行。
- 单元测试保持秒级。
- 禁止依赖长程 Grokking 的随机出现。
- 禁止下载外部数据。
- 固定随机种子。
- 浮点比较使用与 dtype 匹配的容差。
- 测试失败信息应包含 shape、最大误差和关键配置。
- CUDA 测试使用显式 marker，GPU 不可用时跳过。
- CPU smoke run 使用小模数和很少 step，验证管线完整性。
- 正式 GPU integration test 只在 NVIDIA GeForce RTX 4060 Laptop GPU 8GB 上执行。
- 测试命令通过 `conda run --prefix ./env` 调用，禁止使用仓库外解释器。

## 3. 必需测试组

### 环境与 GPU 诊断

验证：

- `sys.prefix` 指向仓库根目录的 `env`；
- `doctor` 在 CPU 模式下能够输出环境摘要；
- CUDA 可用时能读取设备名称、总显存、compute capability 和 PyTorch CUDA runtime；
- 严格 GPU 模式在目标 RTX 4060 Laptop 8GB 上通过；
- 非目标设备或显存不满足时，正式运行检查给出明确失败；
- 显存 smoke test 记录峰值 allocated/reserved 数值；
- CPU-only CI 不会伪装成正式实验通过。

### 数据

验证：

- \(p^2\) 个唯一有序输入；
- 标签公式正确；
- train/test 互斥且覆盖完整表；
- seed 复现；
- split hash 稳定；
- 非法训练比例被拒绝。

### 模型

验证：

- logits shape；
- causal mask；
- dropout=0 时确定性；
- hook 名称和 activation shape；
- CPU forward/backward；
- 配置 round-trip。

### 中心化 logits

验证：

\[
\sum_c\widetilde z(a,b,c)\approx0.
\]

验证类别公共偏置不会改变中心化结果、softmax 和 argmax。

### Reynolds 投影

验证：

\[
\Pi^2z\approx\Pi z,
\]

\[
\langle z^\parallel,z^\perp\rangle\approx0,
\]

\[
\widetilde z\approx z^\parallel+z^\perp.
\]

对人工构造的 \(g(c-a-b)\)，投影结果应保持不变。对随机张量，群变换后的投影值应一致。

### Fourier

验证：

- FFT/IFFT round-trip；
- Parseval；
- 目标线 mask；
- Fourier mask 与 Reynolds 投影一致；
- 实数输入的共轭对称性；
- 模索引在奇数和偶数 \(p\) 下正确。

### Margin 阈值

人工构造：

- \(\Gamma>I\) 时完整表分类正确；
- 干扰增加后可破坏指定输入；
- 正确类别不会进入错误最大值；
- \(p=2\)、\(p=3\) 的边界情况。

### Checkpoint

验证：

- 保存后加载得到相同 logits；
- optimizer state 恢复；
- RNG state 恢复；
- 连续训练和中断恢复在容差内一致；
- split hash 不匹配时拒绝加载；
- 原子写入不会留下伪装成有效 checkpoint 的半成品。

### 训练 smoke

小配置建议：

```text
p=7
train_fraction=0.5
d_model=16
n_heads=2
n_layers=1
max_steps=5
precision=fp32
device=cpu
```

验证：

- 训练命令退出码为 0；
- metrics、metadata、status 和 checkpoint 存在；
- scalar step 单调；
- run 可被 evaluator 和 analyzer 读取；
- resume 后 global step 正确增加。

### 干预

验证：

- branch run 记录父 run 与 checkpoint；
- 冻结模块参数保持不变；
- optimizer reset 清除 moment；
- Fourier projection 只保留指定频率；
- 模块移植检查 shape 和配置兼容性。

## 4. 属性测试

优先使用多个小模数参数化测试：

```text
p in {2, 3, 5, 7, 8}
```

允许使用 Hypothesis，前提是依赖已经正式声明。随机 property test 必须输出失败 seed。

## 5. 回归 fixture

发现数学错误或恢复错误后，增加最小回归测试。fixture 只保存必要张量，附带来源说明和预期性质。

## 6. 测试命名

推荐格式：

```text
test_<unit>_<condition>_<expected>
```

示例：

```text
test_reynolds_projection_is_idempotent
test_resume_training_matches_continuous_run
test_target_line_mask_uses_modular_negative_frequency
```

## 7. 完成条件

涉及核心数学函数的修改，要求相关单元测试全部通过。涉及训练和 checkpoint 的修改，要求 integration smoke 与 resume equivalence 通过。跳过的测试需要在任务报告中说明原因。
