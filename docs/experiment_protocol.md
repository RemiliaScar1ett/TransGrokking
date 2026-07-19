# M0 实验协议

## 基线

- 任务：`p=97` 模加法，固定 seed 的 40%/60% 划分。
- 模型与优化参数以 `configs/baseline_ce.yaml` 为唯一入口。
- 目标函数仅为 cross-entropy；congruence loss 在 M0 禁用。
- 正式设备为 `cuda:0` 上的 NVIDIA GeForce RTX 4060 Laptop GPU 8GB。
- 数值策略为 FP32，并关闭 TF32、AMP、BF16、FP16 和 `torch.compile`。

正式运行必须先通过严格 doctor。配置、划分或数值策略不得在运行中静默改变。
失败与中断运行保留配置、状态和错误摘要。

## 当前执行边界

M0 只允许 CPU 单元测试、恢复等价测试和极短 smoke。`baseline_ce.yaml` 的长程轨迹
尚未执行。测试集指标仅用于管线观测，不用于本轮得出 Grokking 结论。

M1+ 的事件检测、完整 logits、Reynolds/Fourier 指标、activation 分析和干预不属于
本阶段实现范围。

