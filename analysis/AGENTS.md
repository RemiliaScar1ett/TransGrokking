# analysis/AGENTS.md

## 1. 作用域

本文件适用于 `analysis/`。该目录负责读取已有运行产物、生成诊断数据、图表和报告。

## 2. 数据边界

- 分析脚本默认只读 `runs/<run_id>/`。
- 禁止在分析命令中隐式启动训练。
- 缺失 checkpoint 或指标时给出明确错误。
- 重建完整 logits 时记录 checkpoint、device、dtype 和 batch size。
- 派生指标保存到独立文件，避免覆盖原始日志。
- 多运行对比必须检查任务、模数、数据划分和指标版本是否兼容。

## 3. RTX 4060 Laptop 8GB 分析策略

分析实现必须针对 8GB 显存设计：

- 每次只加载一个 checkpoint 到 GPU。
- 完整输入前向使用配置化 batch size。
- logits 生成后优先转移到 CPU；Reynolds、FFT、统计和绘图默认在 CPU 执行。
- activation 只提取报告所需模块，并立即写入派生文件或在线聚合。
- 禁止把多 checkpoint activation 堆叠到 GPU。
- 多运行比较按 run 顺序读取，使用磁盘上的派生表完成合并。
- 每份 checkpoint 报告记录峰值显存、batch size、dtype 和 offload 策略。
- OOM 后可以减小分析批量并重新执行该分析任务，报告中保留重试信息。

正式训练曲线来源必须是 RTX 4060 Laptop GPU 8GB 运行。CPU 或其他 GPU 生成的结果只能用于代码验证，并在报告中显式标注。

## 4. 时间线

统一读取事件定义：

```text
t_fit
t_alg
t_grok50
t_dom
t_grok99
```

事件不存在时写入 `null` 和原因。禁止将训练结束 step 代替缺失事件。

建议生成：

- train/test loss；
- train/test accuracy；
- margin 分位数；
- \(D_{\mathrm{eq}}\)；
- \(\Gamma\) 与 \(I\)；
- 参数范数和模块范数；
- 关键事件竖线。

## 5. Fourier 报告

至少包含：

- 目标频率线能量；
- 各 \(r\) 的 frequency-time heatmap；
- 非目标频率能量；
- restricted/excluded loss；
- Reynolds 与 Fourier 一致性误差。

热图需标注 Fourier norm、log scale 状态和零频处理方式。

## 6. 表征报告

至少支持：

- embedding frequency energy；
- circle-fit \(R^2\)；
- hidden-state effective rank；
- layerwise probe accuracy；
- attention head 和 MLP 的结构贡献。

降维图仅作辅助展示。PCA、t-SNE 或 UMAP 的参数和 seed 需要写入图注或 metadata。

## 7. 优化报告

至少支持：

- 总参数范数；
- 模块参数范数；
- data/decay update ratio；
- radial/tangential update；
- update cosine；
- 距最终 checkpoint 的参数距离。

对不同 parameter group 分开展示，避免将无 decay 参数混入 decay 解释。

## 8. 因果分支对比

分支图需要显示：

- 父 checkpoint；
- 分支修改项；
- 相对 step 与绝对 step；
- \(\Gamma\)、\(I\)、\(D_{\mathrm{eq}}\)；
- test accuracy；
- 分支运行状态。

只比较共享父 checkpoint 的分支，或在报告中明确额外差异。

## 9. 图形规范

- 轴标签包含量名和单位或归一化方式。
- 图题包含 run ID 的短形式。
- 颜色映射在同类图中保持稳定。
- 训练集和测试集的线型保持稳定。
- 保存 PNG 和矢量格式中的至少一种。
- 同时保存绘图所用的派生表。
- 禁止只输出平滑曲线；原始曲线需要保留。
- 平滑窗口、插值和对数变换必须显式标注。

## 10. 自动报告

推荐生成：

```text
analysis/reports/<run_id>/
├── summary.md
├── events.json
├── derived_metrics.parquet
└── figures/
```

`summary.md` 只陈述可由当前运行数据支持的观察。机制解释要附对应图表和指标。

## 11. 完成条件

分析脚本应通过 smoke run 产物测试。对缺失数据、失败运行和旧 schema 提供可读错误。生成报告后检查图表数量、空数据、NaN 和事件标注。
