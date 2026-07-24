# AGENTS.md

## 1. 作用域与优先级

本文件适用于整个 TransGrokking 仓库。进入子目录工作前，继续读取该目录中的 `AGENTS.md`。目录层级更深的规则负责补充局部实现约束。用户当前任务与系统指令保持最高优先级。

## 2. 项目使命

TransGrokking 用有限循环群上的模加法研究小型 Transformer 的延迟泛化过程：

\[
a,b\in\mathbb Z_p,\qquad y=(a+b)\bmod p.
\]

项目把每个 checkpoint 视为独立模型状态，从行为、函数结构、内部表征、计算电路和优化动力学五个层级追踪训练过程。工作目标是建立可恢复、可重算、可干预的训练时间线，并定位记忆形成、算法结构形成、残差清理和测试性能跃迁之间的关系。

代码清晰度、数学一致性、可复现性和可审计性优先于运行吞吐率。

## 3. 规范来源

开始任何工作前，按顺序阅读：

1. `README.md`
2. `docs/intro.md`
3. `docs/experiment_protocol.md`
4. 当前目录树中的全部 `AGENTS.md`
5. 相关配置、测试和已有实现

固定实验规范控制任务、阶段顺序、基准参数、停止规则和验收逻辑。`docs/intro.md` 统一理论符号，`docs/experiment_protocol.md` 维护仓库内可执行协议，`docs/implementation.md` 记录工程决定与实际验证。

发现冲突时停止推进正式实验，先在 `docs/implementation.md` 记录差异并完成同步。

## 4. 科学证据原则

- 禁止生成、补写或推测实验结果。
- 未实际执行的功能标记为 `implemented` 或 `tested`，正式运行完成后才使用 `run`。
- 机制结论需要函数指标、表征指标或因果干预支持。
- 每条正式结果绑定 Git commit、run ID、resolved config、split hash 和 checkpoint step。
- 理论假说与经验观察分开记录。
- 失败与中断运行保留配置、状态和错误摘要。
- 超参数选择遵循预先冻结的协议。

## 5. 固定环境与平台

项目只使用仓库根目录下的 Conda prefix：

```text
./env
```

正式实验平台固定为：

```text
GPU: NVIDIA GeForce RTX 4060 Laptop GPU
VRAM: 8 GB
CUDA device: cuda:0
```

主基线采用 FP32，并关闭 TF32、AMP、BF16、FP16 和 `torch.compile`。CPU 用于数据生成、单元测试、静态检查和短程 smoke run。

所有命令通过本地 prefix 执行：

```bash
conda run --prefix ./env python -m pytest -q
conda run --prefix ./env python -m transgrokking.cli doctor
```

正式运行前执行严格设备检查：

```bash
conda run --prefix ./env python -m transgrokking.cli doctor \
  --require-cuda \
  --expected-device "NVIDIA GeForce RTX 4060 Laptop GPU" \
  --expected-vram-gb 8
```

确定性环境变量必须在首次 CUDA API 调用前设置。模型必须先迁移至目标设备，再创建引用其参数的优化器。GPU smoke test 需要验证至少一个模型参数在 optimizer step 后发生有限且非零的更新。

## 6. 工程边界

推荐结构：

```text
TransGrokking/
├── AGENTS.md
├── README.md
├── environment.yml
├── pyproject.toml
├── configs/
├── docs/
├── src/
│   ├── AGENTS.md
│   └── transgrokking/
│       ├── cli.py
│       ├── config.py
│       ├── data.py
│       ├── models/
│       ├── training/
│       ├── metrics/
│       ├── interventions/
│       └── utils/
├── tests/
│   └── AGENTS.md
├── analysis/
│   └── AGENTS.md
├── legacy/
└── runs/
```

训练循环、数学指标、分析绘图和干预执行保持职责分离。配置文件承担实验参数入口，运行目录承担科学证据载体。导入模块时不得启动训练、创建运行目录或修改外部环境。

## 7. 运行产物协议

每个正式 run 至少保存：

```text
runs/<run_id>/
├── config.resolved.yaml
├── metadata.json
├── split.pt
├── status.json
├── metrics/
│   ├── scalars.jsonl
│   ├── error_offsets.jsonl
│   └── events.json
├── checkpoints/
│   ├── step_000000.pt
│   └── manifest.json
├── tensors/
├── figures/
└── logs/
```

Checkpoint 至少包含模型、优化器、global step、配置、scientific config hash、split hash、parameter-group signature 和 Python/NumPy/Torch RNG 状态。文件写入采用临时文件加原子替换。

Child run 记录父 run、父 checkpoint 和父 step。Scalar step 严格递增，manifest step 保持唯一，已有 checkpoint 禁止覆盖。

## 8. 统一命令接口

稳定入口包括：

```bash
conda run --prefix ./env python -m transgrokking.cli generate-data --config configs/baseline_ce.yaml
conda run --prefix ./env python -m transgrokking.cli train --config configs/baseline_ce.yaml
conda run --prefix ./env python -m transgrokking.cli evaluate --run-dir runs/<run_id>
conda run --prefix ./env python -m transgrokking.cli analyze --run-dir runs/<run_id>
conda run --prefix ./env python -m transgrokking.cli branch --run-dir runs/<run_id> --checkpoint <step> --config <branch.yaml>
```

CLI 发生变化时同步更新 README、协议文档和 subprocess 测试。

## 9. 固定阶段体系

阶段顺序与固定实验规范保持一致。Codex 不得跳过阶段门，不得把后续阶段的科学条件提前混入当前阶段。

### M0：工程与可复现基础

交付范围：

- 包结构、严格配置、`./env` 环境和测试框架；
- 硬件 doctor、确定性数据和透明 Transformer；
- CE-only full-batch 训练、运行生命周期和安全 checkpoint；
- AdamW parameter groups、恢复与 child-run 语义；
- CPU smoke 和目标 GPU 单步 smoke。

M0 完成后保持历史原型可追溯。后续阶段不得破坏 M0 的恢复等价性和产物协议。

### M1：CE-only 基准轨迹与行为时间线

M1 对应固定规范的第一阶段，包含行为测量、已冻结的 20000-step seed 1 基准证据和后续 50000-step 延长轨迹。

#### M1-A 行为测量

实现并记录：

- train/test cross-entropy 与 accuracy；
- 正确类别 margin 的均值、最小值和固定分位数；
- 错误样本数量、error rate 和循环 error offset；
- 参数总范数、模块范数及 decay/no-decay group 范数；
- `t_fit`、`t_grok50`、`t_grok99` 的连续窗口事件。

指标写入 schema 化的机器可读文件。离线 evaluator 只读 run artifacts，不向训练时间线重复写入记录。

#### M1-B 20000-step CE-reference

固定配置：

```text
p=97
train_fraction=0.4
split_seed=42
d_model=128
n_heads=4
n_layers=2
d_mlp=512
dropout=0
norm_first=true
final_norm=false
optimizer=AdamW
learning_rate=1e-3
weight_decay=0.5
model_seed=1
device=cuda:0
precision=fp32
eval_interval=50
checkpoint_interval=100
congruence_weight=0
```

M1-B 的 20000-step canonical run 与 `results/m1_ce_reference/` 是已冻结证据。首次事件 `t_fit`、`t_grok50` 和 `t_grok99` 保持原定义与原记录，不得因后续稳定性分析被改写。该轨迹只支持行为层观察，不支持机制结论。

#### M1-C seed 1 延长至 50000 step

M1-C 必须满足：

- 原 `results/m1_ce_reference/` 和 20000-step canonical run 保持不可变；
- 从 canonical run 的最新 checkpoint 创建 child run；
- scientific config hash 与 split hash 保持一致；
- 只把执行上限延长为 `max_steps=50000`，时间线继续使用绝对 global step；
- 运行前 stability metrics 与最小 optimization diagnostics 均达到 `implemented` 和 `tested` 状态；
- 检查坍塌是否继续出现、episode 间隔是否变化，以及是否形成长期稳定区间；
- 不改写 M1-B 的 `t_fit`、`t_grok50` 和 `t_grok99`。

M1-C 只延长行为证据并增加预注册诊断，不得提前产生函数空间、Fourier 或机制性结论。

### M2：失稳验证、函数空间与群对称性

#### M2-A 失稳真实性验证与坍塌窗口分析

M2-A 优先回答：

- 已观察到的坍塌能否通过 checkpoint 离线重算复现；
- 坍塌是否属于真实模型状态，而非在线记录或产物异常；
- 坍塌前、谷底和恢复后的行为及参数尺度如何变化；
- 坍塌是否与优化状态、parameter-group 范数或 logit 尺度同步。

M2-A 至少包含 stable window、collapse episode、checkpoint 重算、最小 optimization diagnostics 和失稳中心化分析。行为同步只构成待验证关系；在函数证据或因果证据建立前不得写成机制结论。

#### M2-B 函数空间与群对称性

M2-B 对应固定规范的函数空间分析阶段。

实现完整中心化 logits：

\[
\widetilde z(a,b,c)=z(a,b,c)-\frac1p\sum_j z(a,b,j).
\]

实现 Reynolds 投影及函数分解：

\[
z^\parallel=\Pi\widetilde z,\qquad z^\perp=(I-\Pi)\widetilde z.
\]

记录：

\[
D_{\mathrm{eq}}=\frac{\|z^\perp\|_2^2}{\|\widetilde z\|_2^2},
\qquad
\Gamma=g(0)-\max_{d\ne0}g(d),
\]

\[
I=\max_{a,b,c\ne y}[z^\perp(a,b,c)-z^\perp(a,b,y)].
\]

除 `D_eq`、`Gamma` 和 `I` 外，M2-B 还记录 `L_parallel`、centered-logit Frobenius norm、prediction entropy 和 normalized margin，并实现 `t_alg` 与 `t_dom`。所有失稳 episode 必须叠加到共享函数时间线。完整 logits evaluator 支持分批前向和 CPU offload。

M2-B 验收包括中心化、幂等性、正交性、重构、群不变性、人工阈值案例和失稳 episode 时间对齐。

### Gate 2：seed 2、3 行为层复现

M2-A 与 M2-B 形成稳定分析管线后，执行以下独立初始化条件：

```text
loss=CE
weight_decay=0.5
seed=2,3
```

Gate 2 只进行行为时间线与失稳统计复现，并分别按绝对 step、相对 collapse onset 和相对 first grokking event 比较。完整多 seed 汇总和 WD 网格仍属于 M4。

### M3：Fourier 分析

M3 对应固定规范的 Fourier 阶段。统一采用：

```python
torch.fft.fftn(centered_logits, dim=(0, 1, 2), norm="ortho")
```

实现：

- 目标线 \((r,r,-r)\) 能量及各频率时间线；
- 非目标频率能量；
- Reynolds mask 与显式投影等价；
- Parseval 与 \(E_{\mathrm{line}}=1-D_{\mathrm{eq}}\) 校验；
- restricted/excluded logits 与对应 loss。

在最终 grokked checkpoint 上选择解释目标线 95% 能量的最小频率集合，固定该集合回看 M1 正式轨迹。

M3 数学测试全部通过后才能进入 M4。

### M4：完整多 seed 与权重衰减网格

M4 对应固定规范的第四阶段。函数空间、失稳和 Fourier 管线稳定后执行完整运行矩阵。

正式条件：

```text
CE-reference: WD=0.5, seed=1
CE-replication: WD=0.5, seed=2,3
WD-grid: WD=0,0.1,1.0, seed=1,2,3
```

各运行保持任务、模型、数据协议和数值设置一致。汇总事件顺序、Grokking 延迟、\(\Gamma/I/D_{\mathrm{eq}}\) 关系、Fourier 结构和 margin–norm 效率。

Checkpoint 后改变 WD 的分支实验属于 M5；M4 研究从初始化开始的独立训练条件。

### M5：表征、电路、优化动力学与因果干预

M5 对应固定规范的第五阶段。

#### 表征与候选电路定位

实现 embedding Fourier 谱、circle fit、hidden-state 二维 DFT、effective rank 和 layerwise linear probe。Hook 覆盖 embedding、residual stream、per-head output、MLP activation、unembedding 输入和 logits。

#### 优化动力学测量

记录实际 AdamW parameter groups 上的参数、梯度、data update 与 decay update 范数，并计算径向/切向分量、更新夹角以及 checkpoint 到最终状态的距离。

#### 因果干预

统一 branch runner 至少支持：

- WD-off、WD-low、WD-high；
- optimizer state reset 与模块冻结；
- key-frequency projection/ablation；
- activation patching 与模块移植。

每个分支生成独立 run ID，记录父运行、父 checkpoint、修改项和 scientific config 差异。M5 保持 CE-only，禁止启用 congruence 条件。

### M6：Congruence 成对实验

M6 对应固定规范的第六阶段。在 CE 基准、函数指标、Fourier 管线和分支系统稳定后加入：

\[
L_{\mathrm{cong}}=
\sum_kP_\theta(k\mid a,b)
\left[1-\cos\frac{2\pi(k-y)}p\right].
\]

实现可配置 loss 权重、CE/congruence 独立日志和模块级梯度范数及夹角。正式成对条件共享数据划分、初始化和数值配置：

```text
CE
全程 congruence
`t_fit` 后开启
`t_fit` 后关闭
```

比较 `t_alg`、`t_dom`、目标频率增长、最终函数结构和作用阶段。M6 结束前不得把 congruence 结果合并进 CE-reference 结论。

### M7：统一分析报告

M7 对应固定规范的统一报告协议。报告脚本只读取已有 artifacts，至少生成：

- 行为时间线与事件表；
- \(D_{\mathrm{eq}}\)、\(\Gamma\)、\(I\) 与 \(L_\parallel\)；
- frequency-time heatmap 与 restricted/excluded 指标；
- 表征、电路和优化动力学结果；
- 多 seed、WD 网格、checkpoint 分支和 congruence 对比；
- 配置、commit、run ID、checkpoint 与限制摘要。

报告中的观察与机制解释分开书写。每条解释附对应指标、图表和干预证据。

## 10. 阶段状态与门控

`docs/implementation.md` 使用以下状态：

```text
planned
implemented
tested
run
observed
```

每个阶段完成后记录：完成范围、实际命令、测试结果、运行产物、已知限制和下一阶段进入条件。

严格门控：

```text
M0
→ M1-A
→ M1-B
→ 文档修订
→ 稳定性代码实现与测试
→ M1-C
→ M2-A
→ M2-B
→ Gate 2
→ M3
→ M4
→ M5
→ M6
→ M7
```

允许在当前阶段内完成代码、测试和 smoke。正式科学分析需要前置阶段的 run artifacts 已通过验收。当前文档修订不得被标记为 stability、M1-C 或 M2 已实现；这些阶段仍须依次通过各自的实现、测试和运行门。

## 11. 配置规则

配置解析采用严格 schema。未知字段、非法类型、超范围值和 shape 不一致立即报错。科学配置与执行配置分离；恢复时校验 scientific config hash。

基准配置的 `eval_interval=50`、`checkpoint_interval=100` 属于固定协议。任何调整都需要建立独立运行条件并记录理由。

## 12. RTX 4060 Laptop 8GB 资源约束

- 每次只加载一个 checkpoint 到 GPU。
- 完整 logits 允许逐 checkpoint 构造，并及时转移到 CPU。
- Activation 只提取指定模块和指定 checkpoint。
- Fourier、Reynolds、统计和绘图默认允许在 CPU 执行。
- 禁止缓存整条时间线的完整 hidden state。
- CUDA OOM 时保留失败状态，允许减小分析 batch 或启用 CPU offload。
- 正式训练科学配置不得因显存问题静默变化。

## 13. 验证要求

完成任何代码修改后执行：

```bash
conda run --prefix ./env python -m pytest -q
conda run --prefix ./env python -m ruff check .
conda run --prefix ./env python -m ruff format --check .
```

影响训练、checkpoint 或 CLI 时继续运行 CPU smoke、目标 GPU 单步更新测试和恢复测试。影响 Reynolds 或 Fourier 时执行全部数学不变量测试。影响正式协议时更新 README 和 docs。

## 14. Git 工作规范

- 开始前查看 `git status` 和当前分支。
- 保留用户已有未提交修改。
- 禁止强制重置、强制推送或删除未知文件。
- 每个提交覆盖一个清晰主题。
- 完成后确认测试结果与工作区状态。
- 禁止修改已有提交历史。

## 15. 完成报告格式

每次任务完成后报告：

```text
完成范围
关键设计
修改文件
运行命令
测试结果
生成产物
未解决问题
下一阶段进入条件
```

只报告实际完成和实际验证的内容。
