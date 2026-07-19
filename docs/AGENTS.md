# docs/AGENTS.md

## 1. 作用域

本文件适用于 `docs/`。文档承担理论定义、实验协议、实现决策和使用方法的记录责任。

## 2. 文档职责

- `intro.md`：理论符号、指标定义、机制假说和解释边界；
- `implementation.md`：代码架构、接口、实现约定和决策记录；
- `experiment_protocol.md`：基线、运行矩阵、checkpoint 频率和分析步骤；
- `README.md`：面向仓库入口的项目概述和快速使用说明。

新增文档应有明确职责，避免复制整段已有内容。

## 3. 数学一致性

以下符号应保持统一：

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
```

修改公式后，同步检查：

- `src/transgrokking/metrics/`；
- 对应单元测试；
- README 中的简化说明；
- 图表标签和报告字段。

## 4. 事实状态

文档使用明确状态词：

```text
planned
implemented
tested
run
observed
```

“implemented”表示代码已存在。“tested”表示自动检查通过。“run”表示正式实验执行完成。“observed”表示结果文件支持该描述。

禁止将计划指标写成已经观测到的结论。

## 5. 命令与配置

README 中的命令必须能够直接复制执行。所有项目命令默认写成 `conda run --prefix ./env ...`，并说明正式实验平台为 NVIDIA GeForce RTX 4060 Laptop GPU 8GB。CLI 变更后同步更新：

- 快速开始；
- 配置示例；
- 输出目录；
- 恢复训练；
- 分析命令。

示例配置应来自仓库中的真实文件。

## 6. 决策记录

`implementation.md` 中采用以下格式：

```markdown
## YYYY-MM-DD — 决策标题

**背景**

**选择**

**理由**

**影响**

**待验证事项**
```

记录会影响数学含义、可复现性、设备兼容性或文件 schema 的决定。

## 7. 实验协议

`experiment_protocol.md` 至少定义：

- 基线配置；
- seed 数量；
- 数据划分；
- 最大 step 和停止规则；
- 评估及 checkpoint 间隔；
- 关键事件定义；
- 深度分析 checkpoint；
- 对照与干预分支；
- 失败运行处理；
- RTX 4060 Laptop 8GB 的显存预算、分析批量和 CPU offload 规则；
- 结果汇总规则。

任何正式运行前先冻结协议版本，并写入 run metadata。

## 8. 写作规范

中文解释为主，保留必要英文术语。公式使用 Markdown LaTeX。代码、路径和字段使用反引号。结论限制在数据支持范围内。已知限制放在对应章节附近。

## 9. 完成条件

文档修改后检查内部链接、命令、路径和符号。涉及代码接口时，运行相应 smoke 命令确认文档示例可用。
