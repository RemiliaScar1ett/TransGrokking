# docs/AGENTS.md

## 1. 作用域

本文件适用于 `docs/`。文档维护理论定义、固定协议、工程决策、阶段状态和使用命令。

## 2. 文档职责

- `intro.md`：理论符号、指标、假说和解释边界。
- `experiment_protocol.md`：固定配置、阶段顺序、运行矩阵、停止规则和验收。
- `implementation.md`：代码架构、schema、工程决定和实际验证。
- `README.md`：仓库入口、当前状态和可复制命令。

新增文档需要明确职责，禁止复制大段已有内容。

## 3. 阶段名称

仓库统一使用：

```text
M0 工程基础
M1-A 行为测量
M1-B 20000-step CE-reference
M1-C 50000-step extension
M2-A instability analysis
M2-B function-space analysis
Gate 2 seed replication
M3 Fourier
M4 multi-seed / WD grid
M5 representation / circuit / optimization
M6 congruence
M7 report
```

旧阶段名称出现时完成迁移。README、协议、实现记录和报告标题保持一致。

## 4. Markdown 数学公式

- 行内数学统一使用 `$...$`，且不得跨行。
- 独立数学统一使用各占一行的 `$$` 定界符。
- 禁止在 `docs/**/*.md` 中使用 `\(...\)` 和 `\[...\]` 作为 Markdown 数学定界符。
- fenced code block 与反引号包围的内容不参与公式定界符转换。
- 不得把 shell 变量、货币符号、路径、CLI 参数、配置字段或普通文本误判为数学公式。
- 多行推导可在 `$$` 内使用 `aligned`，保留既有 LaTeX 命令、上下标、矩阵和对齐结构。

完成文档修改后，应在代码围栏与反引号之外确认旧定界符为零；本节用于说明禁用语法的反引号示例不构成公式。

## 5. 数学符号

```text
z                    raw logits
\widetilde z         centered logits
z^\parallel          Reynolds-projected logits
z^\perp              residual logits
g(d)                 offset profile
D_eq                  nonequivariant energy ratio
E_line                target Fourier-line energy ratio
Gamma                 algorithmic margin
I                     residual interference
t_fit                 train-fit event
t_alg                 algorithm-margin event
t_grok50              midpoint test event
t_dom                 Gamma > I event
t_grok99              high-generalization event
t_stable99            stable-generalization event
collapse_onset        collapse-episode onset
```

公式变化后同步检查代码、测试、README、图表标签和产物字段。

## 6. 固定协议

`experiment_protocol.md` 至少规定：

- CE-reference 配置；
- evaluation interval 50 与 checkpoint interval 100；
- M1-B 5000/20000-step 证据与 M1-C 50000-step child-run 延长规则；
- `t_grok99` 后 20 个 evaluation interval 只作为首次事件后的继续训练证据；
- `t_stable99`、稳定窗口与坍塌 episode 的长期稳定性口径；
- Gate 2 seed 复制与 M4 完整多 seed/WD 网格的边界；
- 分析 checkpoint；
- 表征、电路和优化干预；
- Congruence 配对条件；
- RTX 4060 Laptop 8GB 资源策略；
- 失败、中断和结果汇总规则。

正式运行前冻结协议版本并写入 metadata。

## 7. 事实状态

使用：

```text
planned
implemented
tested
run
observed
```

代码存在时使用 `implemented`，自动测试通过后使用 `tested`，正式实验完成后使用 `run`，结果文件支持具体陈述后使用 `observed`。

## 8. 决策记录

`implementation.md` 使用：

```markdown
## YYYY-MM-DD — 决策标题

**背景**

**选择**

**理由**

**影响**

**待验证事项**
```

影响数学含义、可复现性、设备兼容性、阶段顺序或 schema 的决定均需记录。

## 9. 命令与配置

所有命令默认采用：

```bash
conda run --prefix ./env ...
```

CLI、配置或目录变化后同步更新 README、协议和 smoke tests。示例配置必须来自仓库真实文件。

## 10. 阶段完成记录

每个阶段结束时，`implementation.md` 记录：

```text
阶段状态
Git commit
实际命令
测试结果
正式 run ID
主要产物
已知限制
下一阶段门
```

当前阶段未完成正式运行时，禁止标记整个阶段为 completed。

## 11. 完成条件

检查内部链接、公式、命令、阶段名称、配置值和状态词。涉及代码接口时运行对应 smoke 命令。
