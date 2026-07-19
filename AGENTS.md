# AGENTS.md

## 1. 作用域与优先级

本文件适用于整个 TransGrokking 仓库。

进入任意子目录工作前，继续读取该目录及其父目录中的 `AGENTS.md`。更深层文件负责补充局部规则。出现冲突时，采用作用域更具体的规则；用户当前任务和系统指令保持最高优先级。

## 2. 项目使命

TransGrokking 是一个面向学习与机制分析的 Grokking 实验平台。基础任务为有限循环群 \(\mathbb Z_p\) 上的模加法：

\[
a,b\in\mathbb Z_p,\qquad y=(a+b)\bmod p.
\]

项目需要完整复现 Grokking 训练轨迹，并通过行为、函数、表征、电路和优化五个层级追踪模型变化。最终产物应允许研究者回答：

- 模型何时完成训练集拟合；
- 可泛化的模加法结构何时形成；
- 非等变残差何时开始清理；
- 测试准确率跃迁与 margin 阈值有何关系；
- Fourier 模式如何进入 embedding、hidden state 和 logits；
- 权重衰减如何改变算法回路与样本记忆成分；
- congruence loss 对训练动力学和最终函数结构产生何种影响。

本项目服务于理解过程。代码清晰度、数学一致性、可复现性和可检查性优先于吞吐率与代码长度。

## 3. 必读材料

开始任何实现前，按顺序阅读：

1. `README.md`
2. `docs/intro.md`
3. 当前目录树中的全部 `AGENTS.md`
4. 相关配置、测试和已有实现

`docs/intro.md` 是理论符号、指标定义和机制假说的主要来源。代码与文档存在差异时，先定位差异并记录，再进行修改。任何理论定义变更都需要同步更新测试和文档。

## 4. 当前最高优先级

从现有最小原型重构出完整实验平台。基础 CE-only Grokking 轨迹必须先稳定复现。congruence loss、复杂干预和可视化扩展安排在基础管线通过验收之后。

现有 `main.py` 与 `data.py` 只承担历史参考作用。重构期间保留可追溯性。允许把它们移入 `legacy/`，前提是新入口已经可运行，迁移记录已写入文档。

## 5. 强制原则

### 5.1 科学性

- 禁止生成、补写或推测实验结果。
- 未实际运行的代码只能标记为“已实现，尚未执行”。
- 指标必须严格对应 `docs/intro.md` 中的数学定义。
- 理论假说和经验观察分开记录。
- 测试集可用于轨迹观测。超参数选择需要固定规则或独立验证方案。
- 深度分析必须绑定 `git commit`、`run_id`、配置摘要和 checkpoint step。
- 任何随机过程都要显式记录种子。
- 失败运行也要保留状态、错误信息和配置，除非文件损坏或包含敏感信息。

### 5.2 可复现性

每个正式运行至少保存：

- 完整解析后的配置；
- 数据划分索引与哈希；
- Python、PyTorch、CUDA 和设备信息；
- 模型初始化种子与数据种子；
- 模型参数、优化器状态和当前 step；
- 指标文件、checkpoint 清单和日志；
- 当前 Git commit；
- 运行状态：`running`、`completed`、`failed` 或 `interrupted`。

恢复训练必须保持数据划分、优化器状态、scheduler 状态和随机数状态一致。

### 5.3 固定执行平台与数值策略

正式实验平台固定为：

```text
GPU: NVIDIA GeForce RTX 4060 Laptop GPU
VRAM: 8 GB
CUDA device: cuda:0
```

CPU 只承担数据生成、单元测试、静态分析和短程 smoke run。正式 Grokking 轨迹、GPU 数值基线和显存评估必须在上述 RTX 4060 Laptop 8GB 平台执行。

- 主基线采用 FP32，并显式关闭 TF32、BF16、FP16、AMP 和 `torch.compile`。
- 混合精度与 TF32 只能作为独立实验条件，配置、run ID 和报告中需要清楚标记。
- 正式运行前执行硬件预检，记录 GPU 名称、可用显存、驱动版本、PyTorch 版本、PyTorch CUDA runtime 和 compute capability。
- 运行元数据记录 `torch.cuda.max_memory_allocated()` 与 `torch.cuda.max_memory_reserved()`。
- 训练代码保持通用设备接口，正式配置固定使用 `cuda:0`。
- 完整 logits 可按分析批次生成。中间 activation 只在指定 checkpoint 和指定模块保存。
- 禁止把多个 checkpoint 的完整 activation 同时驻留在 GPU。
- 发生 CUDA OOM 时立即保留失败状态和显存摘要。允许降低分析批量、改为逐 checkpoint 处理或把派生计算转移到 CPU；训练科学配置不得静默变化。
- Codex 当前执行环境若没有目标 GPU，只能完成代码、CPU 测试和 smoke run。正式实验状态必须标记为尚未执行。

### 5.4 工程边界

- 训练循环、指标计算、分析绘图和干预逻辑保持模块分离。
- 配置文件是实验参数的唯一入口。库代码中不得散落实验超参数。
- 公开函数提供类型标注和简洁 docstring。
- 核心数学函数保持纯函数特征，输入输出 shape 明确。
- 禁止隐藏全局状态。
- 禁止在导入模块时启动训练、创建目录或修改环境。
- 禁止用 broad `except Exception` 吞掉错误。
- 禁止引入与目标无关的大型框架。
- 新依赖需要写入环境文件，并说明用途。

## 6. 目标目录

重构后的推荐结构如下。允许小幅调整，职责边界需要保持清晰。

```text
TransGrokking/
├── AGENTS.md
├── README.md
├── environment.yml
├── pyproject.toml
├── configs/
│   ├── baseline_ce.yaml
│   ├── baseline_no_wd.yaml
│   ├── congruence.yaml
│   └── smoke.yaml
├── docs/
│   ├── AGENTS.md
│   ├── intro.md
│   ├── implementation.md
│   └── experiment_protocol.md
├── src/
│   ├── AGENTS.md
│   └── transgrokking/
│       ├── __init__.py
│       ├── cli.py
│       ├── config.py
│       ├── data.py
│       ├── models/
│       ├── training/
│       ├── metrics/
│       ├── interventions/
│       └── utils/
├── tests/
│   ├── AGENTS.md
│   ├── unit/
│   └── integration/
├── analysis/
│   ├── AGENTS.md
│   ├── build_timeline.py
│   ├── checkpoint_report.py
│   └── make_figures.py
├── legacy/
└── runs/
```

`runs/` 与项目本地 Conda 环境目录 `env/` 默认进入 `.gitignore`。小型测试 fixture 可以存入 `tests/fixtures/`。

## 7. 统一命令接口

项目应提供稳定 CLI。推荐入口：

```bash
conda run --prefix ./env python -m transgrokking.cli generate-data --config configs/baseline_ce.yaml
conda run --prefix ./env python -m transgrokking.cli train --config configs/baseline_ce.yaml
conda run --prefix ./env python -m transgrokking.cli evaluate --run-dir runs/<run_id>
conda run --prefix ./env python -m transgrokking.cli analyze --run-dir runs/<run_id>
conda run --prefix ./env python -m transgrokking.cli branch --run-dir runs/<run_id> --checkpoint <step> --config <branch.yaml>
```

允许增加子命令。已有命令名称发生变更时，同步更新 README、文档和 smoke tests。

## 8. 环境管理

### 8.1 唯一环境位置

整个项目只使用仓库根目录下的 Conda prefix：

```text
./env
```

禁止创建或依赖全局命名环境、`venv`、Poetry 环境、系统 Python site-packages 或仓库外部 Conda prefix。`env/` 必须加入 `.gitignore`。

Codex 执行自动化命令时优先使用 `conda run --prefix ./env`，避免 shell 激活状态不一致。人工交互时可以执行 `conda activate ./env`。

### 8.2 创建与更新

仓库缺少 `environment.yml` 时，M0 必须先创建该文件。文件不得写入绝对 `prefix`。Python 主次版本、核心科学计算依赖和开发工具需要固定到可复现范围。

标准命令：

```bash
conda env create --prefix ./env --file environment.yml
conda env update --prefix ./env --file environment.yml --prune
conda run --prefix ./env python -m pip install -e .
conda run --prefix ./env python -m pytest -q
```

环境已经存在时使用 `conda env update`。禁止删除已有 `./env` 后无理由重建。依赖变更后同步修改 `environment.yml`、`pyproject.toml` 和锁定说明。

所有 `pip` 命令必须通过该环境中的 Python 调用：

```bash
conda run --prefix ./env python -m pip ...
```

### 8.3 PyTorch 与 CUDA

目标机器使用 NVIDIA GeForce RTX 4060 Laptop GPU 8GB。安装 PyTorch 前执行 `nvidia-smi`，记录驱动信息，并选择与驱动兼容的官方 PyTorch CUDA 构建。

- 不修改 NVIDIA 驱动。
- 不安装独立系统级 CUDA Toolkit，除非某个已批准依赖明确需要编译 CUDA 扩展。
- 优先使用 PyTorch 随包提供的 CUDA runtime。
- 不把本机驱动版本写死到 Python 依赖文件。
- 环境创建完成后验证 CUDA 可用性、GPU 名称和显存容量。

M0 需要提供硬件诊断命令：

```bash
conda run --prefix ./env python -m transgrokking.cli doctor
```

正式 GPU 运行增加严格检查：

```bash
conda run --prefix ./env python -m transgrokking.cli doctor --require-cuda --expected-device "NVIDIA GeForce RTX 4060 Laptop GPU" --expected-vram-gb 8
```

设备名称允许处理厂商字符串中的轻微差异，检查结果需要写入 metadata。未通过严格检查时禁止把该运行标记为正式实验。

### 8.4 环境自检

每次开始阶段性实现前，确认解释器来自 `./env`：

```bash
conda run --prefix ./env python -c "import pathlib,sys; print(pathlib.Path(sys.prefix).resolve())"
```

输出路径必须指向仓库根目录的 `env`。若 Conda 不可用，停止环境相关工作并报告缺失条件；不得改用其他环境管理方案。

## 9. 实施阶段

Codex 接到“按 AGENTS 完成重构”之类的完整任务时，按下列阶段推进。每个阶段完成后运行对应测试并更新 `docs/implementation.md`。

### M0：仓库引导与基线规范化

交付内容：

- 建立包结构、配置系统、`./env` Conda 环境文件和测试框架；
- 实现 `doctor` 硬件与环境预检，支持 RTX 4060 Laptop 8GB 严格模式；
- 将数据生成改为确定性函数；
- 实现透明、可 hook 的小型 Transformer；
- 实现 CE-only full-batch 训练；
- 保存配置、划分、checkpoint、优化器和随机状态；
- 提供 CPU smoke 配置；
- 保留原型代码的历史位置或迁移说明。

验收条件：

```bash
conda run --prefix ./env python -m pytest -q
conda run --prefix ./env python -m transgrokking.cli doctor
conda run --prefix ./env python -m transgrokking.cli train --config configs/smoke.yaml
```

两条命令成功，smoke run 可恢复，输出目录符合约定。

### M1：行为时间线

实现以下指标：

- train/test cross-entropy；
- train/test accuracy；
- 正确类别 margin 的均值、最小值和分位数；
- 参数总范数与模块范数；
- `t_fit`、`t_grok50`、`t_grok99` 的事件检测；
- 错误样本数量与错误 offset 分布。

所有指标写入机器可读文件。推荐 Parquet 或 JSONL；大型数组使用 `.pt` 或 `.npz`。

### M2：函数空间与群对称性

实现完整中心化 logit 张量：

\[
\widetilde z(a,b,c)
=
z(a,b,c)-\frac1p\sum_j z(a,b,j).
\]

实现雷诺兹投影、非等变残差和指标：

\[
z^\parallel=\Pi\widetilde z,\qquad
z^\perp=(I-\Pi)\widetilde z,
\]

\[
D_{\mathrm{eq}}
=
\frac{\|z^\perp\|_2^2}{\|\widetilde z\|_2^2},
\]

\[
\Gamma=g(0)-\max_{d\ne0}g(d),
\]

\[
I=\max_{a,b,c\ne y}
[z^\perp(a,b,c)-z^\perp(a,b,y)].
\]

实现 `t_alg` 与 `t_dom`。对 \(p=97\) 的完整 logits 评估需要支持分批前向。

### M3：Fourier 分析

实现统一归一化约定下的 DFT：

\[
\widehat z(r_a,r_b,r_c)
=
\operatorname{DFT}_{a,b,c}[\widetilde z].
\]

实现：

- 目标线 \((r,r,-r)\) 能量；
- 各目标频率随 step 的能量；
- 非目标频率能量；
- Reynolds 投影与 Fourier mask 的一致性校验；
- Parseval 数值校验；
- restricted/excluded logits 接口。

FFT 输出允许保留复数。可视化层负责转换为能量或相位。

### M4：表征与模块观测

实现可选 hook，覆盖：

- token embedding；
- position embedding；
- 每层 residual stream；
- 每个 attention head 输出；
- MLP pre-activation 与 post-activation；
- unembedding 前表示；
- 最终 logits。

实现 embedding Fourier 谱、circle fit、hidden-state effective rank 和 layerwise linear probe。大规模 activation 仅在指定 checkpoint 保存。

### M5：优化动力学

实现：

- 模块级参数范数；
- 数据更新与 AdamW decay 更新的范数；
- 数据更新的径向与切向分量；
- 更新方向夹角；
- checkpoint 到最终参数的距离和余弦；
- optimizer state reset 分支。

更新分解需与实际优化器 step 对齐。无法精确复原时，在文档中写出近似条件。

### M6：因果干预

实现统一 branch runner。至少支持：

- checkpoint 后修改 weight decay；
- 开关 congruence loss；
- 模块冻结；
- key Fourier frequency projection 与 ablation；
- optimizer state reset；
- embedding、Transformer blocks、unembedding 的模块移植。

每个分支生成独立 `run_id`，记录父运行和父 checkpoint。

### M7：Congruence loss

在 CE 基线稳定后加入：

\[
L_{\mathrm{cong}}
=
\sum_kP_\theta(k\mid a,b)
\left[
1-\cos\frac{2\pi(k-y)}p
\right].
\]

实现：

- 可配置损失权重；
- CE 与 congruence 的独立日志；
- 两项梯度的模块级范数与夹角；
- 从指定 checkpoint 开启或关闭该损失；
- 与 CE 基线共享初始化的成对运行。

### M8：分析报告

生成统一报告，至少包含：

- 行为时间线；
- \(D_{\mathrm{eq}}\)、\(\Gamma\)、\(I\)；
- Fourier frequency-time 热图；
- 表征有效秩和 probe；
- 参数范数与更新分解；
- 分支干预对比；
- 配置、commit 和事件时间摘要。

报告脚本读取现有 run artifacts，不得隐式重新训练。

## 10. Checkpoint 与运行目录协议

推荐布局：

```text
runs/<run_id>/
├── config.resolved.yaml
├── metadata.json
├── split.pt
├── status.json
├── metrics/
│   ├── scalars.jsonl
│   └── events.json
├── checkpoints/
│   ├── step_000000.pt
│   └── manifest.json
├── tensors/
├── figures/
└── logs/
```

checkpoint 至少包含：

```text
model_state
optimizer_state
scheduler_state
global_step
config
split_hash
python_rng_state
numpy_rng_state
torch_cpu_rng_state
torch_cuda_rng_state
```

文件写入采用临时文件加原子替换，降低中断导致损坏的风险。

## 11. 配置规则

配置需要显式声明：

```text
task:
  modulus
  train_fraction
  split_seed

model:
  d_model
  n_heads
  n_layers
  d_mlp
  dropout
  activation
  norm_first

optimization:
  optimizer
  learning_rate
  weight_decay
  max_steps
  precision
  allow_tf32
  use_amp
  deterministic
  seed
  device

hardware:
  expected_device
  expected_vram_gb
  formal_run
  analysis_batch_size
  activation_offload

loss:
  cross_entropy_weight
  congruence_weight

logging:
  eval_interval
  checkpoint_interval
  optimizer_checkpoint_interval
  activation_steps
```

配置解析后执行严格校验。未知字段、非法取值和不一致 shape 需要立即报错。

## 12. 验证要求

完成任何代码修改后，执行与改动范围匹配的检查：

```bash
conda run --prefix ./env python -m pytest -q
conda run --prefix ./env python -m ruff check .
conda run --prefix ./env python -m ruff format --check .
```

若项目选择其他 formatter 或 linter，应保持单一工具链并更新本文档。

影响训练、checkpoint 或 CLI 的改动还需运行：

```bash
conda run --prefix ./env python -m transgrokking.cli train --config configs/smoke.yaml
```

影响恢复逻辑的改动必须执行中断—恢复等价测试。

影响 Fourier 或 Reynolds 指标的改动必须执行数学不变量测试。

## 13. Git 工作规范

- 开始前查看 `git status` 和当前分支。
- 保留用户已有未提交修改。
- 禁止执行 `git reset --hard`、强制推送或删除未知文件。
- 每个提交只覆盖一个清晰主题。
- 提交信息使用简洁英文 conventional style，例如 `feat: add reynolds projection metrics`。
- 完成后确认测试结果和 `git status`。
- 用户未要求提交时，可以保留工作区修改，并在结果中列出变更。
- 禁止修改已有提交历史。

## 14. 决策记录

遇到下列情况时，在 `docs/implementation.md` 中记录：

- 理论公式存在多个实现约定；
- PyTorch API 导致可观测性限制；
- 精确更新分解代价过高；
- GPU 与 CPU 路径存在数值差异；
- 目录结构偏离本文件建议；
- 测试只验证了近似性质。

记录内容应包含选择、理由、替代方案和已知限制。

## 15. 完成报告格式

每次任务完成后，报告以下内容：

```text
完成范围
关键设计
修改文件
运行命令
测试结果
生成产物
未解决问题
建议的下一阶段
```

只报告实际完成和实际验证的内容。失败检查需要保留命令、错误摘要和影响判断。
