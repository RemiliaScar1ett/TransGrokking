# M1 CE-reference behavior curves

本目录从 canonical run `20260721T045433955396Z_30c62ebc` 只读导出 M1 连续行为证据。
原始来源、哈希、导出时间和 CSV schema 见 `provenance.json`。

## 实验配置

- 模加法：`p=97`，train fraction `0.4`，split seed `42`
- 模型：`d_model=128`、4 heads、2 layers、`d_mlp=512`、ReLU、`final_norm=false`
- 优化：AdamW，learning rate `0.001`，weight decay `0.5`，seed `1`
- 数值与设备：FP32，TF32/AMP disabled，`cuda:0`
- 记录：evaluation interval `50`，checkpoint interval `100`
- 损失：CE-only，congruence weight `0`

完整 resolved config 保存在 `config.resolved.yaml`。

## 行为事件

| Event | First qualifying window step | Detected at evaluation step |
|---|---:|---:|
| `t_fit` | 100 | 300 |
| `t_grok50` | 6050 | 6150 |
| `t_grok99` | 7000 | 7100 |

事件定义和阈值来自复制的 `events.json`。所有图均使用竖直虚线标注上述三个行为事件。

## 数据文件

- `loss_curve.csv`：step、train/test cross-entropy。
- `accuracy_curve.csv`：step、train/test accuracy。
- `margin_curve.csv`：step，以及 train/test 的 mean、min、q01、q05、q25、median、q75、
  q95、q99 margin。
- `parameter_norm_curve.csv`：step、总参数范数、AdamW decay/no-decay group 范数，以及
  全部模块范数字段。由于正式模型配置 `final_norm=false`，
  `parameter_norm_final_norm` 保留为空值。

四份 CSV 均直接来自 400 条原始 evaluation record。没有平滑、插值、异常点删除、
step 补齐或重新采样。

## 图形

- `loss_linear`：train/test cross-entropy，线性纵轴。
- `loss_log`：同一 train/test cross-entropy，仅使用对数纵轴展示。
- `accuracy`：train/test accuracy，并标注随机准确率 `1/97`。
- `margin_train`：全部 train margin 汇总字段，并标注 `margin=0`。
- `margin_test`：全部 test margin 汇总字段，并标注 `margin=0`。
- `parameter_norm_groups`：总参数范数及 decay/no-decay group 范数。
- `parameter_norm_modules`：所有具有数值的模块范数；禁用的 final norm 不绘制。

每张图同时提供 PNG 和 SVG，横轴均为未经变换的 optimizer update step，标题包含 canonical
run 短 ID `30c62ebc`。

## 解释边界

这些导出结果只支持行为层面的 loss、accuracy、margin、错误事件和参数范数描述。
M2 函数空间与群对称性分析尚未完成；本目录不包含 Reynolds、Fourier 或其他机制指标，
也不提供机制性结论。
