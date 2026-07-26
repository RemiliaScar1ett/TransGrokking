# M2：失稳验证与函数空间分析

## 1. 阶段目标与边界

M2-A 检查 M1 行为时间线中的坍塌是否能由真实 checkpoint 或确定性 replay 状态复现；
M2-B 在同一绝对 step 时间线上计算完整函数、类别中心化、Reynolds 投影、等变/非等变
分解与函数事件。本阶段保持原 CE-only、FP32、full-batch 训练状态不变，所有分析只读访问
M1 artifacts。

本阶段没有执行三维 Fourier、target-line energy、seed 2/3、weight-decay grid、表征 probe、
电路定位、模块干预或 congruence 实验。因此本文只报告 checkpoint 真实性、函数量与已有
优化诊断之间的时间对齐，不把描述性同步写成电路或优化因果机制。

## 2. 证据身份

| 项目 | 值 |
| --- | --- |
| M1 root run | `20260721T045021566841Z_ef3ee07b` |
| M1-B canonical parent | `20260721T045433955396Z_30c62ebc` |
| M1-C terminal child | `20260724T091041024473Z_c6434d8a` |
| M2 analysis ID | `20260726T185412278703Z_5337a9bb` |
| Source training commit | `8d620a84930c85d617131d1062a01e640b667a32` |
| Analysis implementation commit | `e119ff42f96920e9e4fcf54eafd287bb2dd8ecbf` |
| Analysis audit commit | `02087b717b0971e33af3f8cb59412e0d00b47101` |
| Results commit | `4ef234d` |
| Scientific config hash | `b167674594bf0944f0b2afb877d2d8c8f5647c0e4e60c64ebb2a511a9f1f7729` |
| Split hash | `d0ec6ff924ecc411b9a9d40786f057ec869076b98308e2ecb75da2756c308237` |
| Analysis config hash | `5337a9bb82bd3192f8d3e64f7ebf327a436651d2b3785a73f81cf33d879c6ba7` |
| Physical / regular checkpoints | 503 / 501 |
| Exact / replay validation states | 47 / 48 |
| Function records | 501 regular + 48 replay = 549 |
| Hardware | NVIDIA GeForce RTX 4060 Laptop GPU, 8,585,216,000 bytes |
| Forward / reduction | CUDA FP32, batch 1024 / CPU FP64 |
| Peak allocated / reserved | 231,535,104 / 329,252,864 bytes |

三个 manifest 的现场计数分别为 51、151 和 301。Step 5000 与 20000 各有两个物理
checkpoint：raw SHA 不同，但字段级 semantic state hash 相同，因此函数时间线按语义状态
去重，物理别名仍完整保存在 provenance 和 checkpoint tables 中。

M1 两个冻结 results 目录在 analysis audit 和 portable export audit 中均按完整 path、size、
SHA-256 清单复核，未发生变化。

## 3. M2-A checkpoint 真实性验证

### 3.1 Resolver 与 replay

Replay 来源按训练 lineage segment 固定：

- step 0–5000 使用 root run；
- step 5000–20000 使用 M1-B child；
- step 20000–50000 使用 M1-C child。

缺少 50-step checkpoint 的状态从前一个同 segment checkpoint 恢复完整 model、optimizer 和
RNG。每个目标执行两次独立 replay，并继续到后一个 100-step manifested checkpoint 做 bridge
比较。48/48 midpoint 自一致、48/48 endpoint semantic state 与真实 checkpoint 一致；source
和 endpoint 文件 SHA 在 replay 前后均未改变。

### 3.2 行为重算

M2-A 验证 95 个唯一目标，47 个来自 exact checkpoint，48 个来自 deterministic replay，
`unresolved=0`。Accuracy 与 error count 使用精确比较；CE、margin 和参数范数使用
`atol=1e-6, rtol=1e-6`。94 个存在 committed scalar 的状态，其 reference evaluator 重算
字段最大绝对差均为 0；step 0 没有 committed scalar，只验证内部一致性。95/95 error-offset
直方图一致。

M2-B 的 batch-1024 前向在 33100、33300、34300、37000、37700、40500、42300 和 45900
出现单个近边界 argmax 与原 full-batch evaluator 不同；最大 CE 差为
$1.53\times10^{-7}$。这 8 个状态均通过未分批 reference recheck，并被显式标记为
`batch_sensitive_predictions`，没有被静默当作精确 prediction match。

### 3.3 强制 episode

- `test_008`：step 18300 为 pre-collapse；18350 replay 为 onset/test trough；train 在
  18800 确认恢复，test 在 21300 开始恢复、21400 确认、21450 post-recovery。该 episode
  跨越 step 20000 lineage anchor，连续性与恢复均已确认。
- `test_010`：27250 replay 为 pre-collapse，27300 onset，31400 trough；到 50000 仍为
  `terminal_unrecovered`，`recovery_start` 和 `recovery_confirmed` 保持 `null`。
- `train_026`：49900 pre-collapse，49950 replay 为 onset/train trough；50000 仅为
  `terminal_unrecovered`。终点 train accuracy 单点回升不替代三次连续恢复确认。
- Train-only 对照 `train_001`、`train_015`、`train_024` 均完成 pre/onset/trough/recovery/
  post-recovery 解析；后两者分别位于活动 `test_009` 和 `test_010` 内。

全部 10 个 test primitive、10 个 joint composite 和三个预注册 train-only 对照均已解析。
Joint episode 保留独立 `train_trough` 与 `test_trough`，未构造虚假的单一 trough。

## 4. 函数空间定义

对输入 $(a,b)$ 和候选类别 $c$，原始 logits 记为 $z(a,b,c)$。类别中心化为

$$
\widetilde z(a,b,c)=z(a,b,c)-\frac{1}{p}\sum_{j=0}^{p-1}z(a,b,j).
$$

模加法群作用为

$$
T_{u,v}(a,b,c)=(a+u,b+v,c+u+v)\pmod p.
$$

令 $d=c-a-b\pmod p$，Reynolds offset profile 为

$$
g(d)=\frac{1}{p^2}\sum_{a,b}\widetilde z(a,b,a+b+d).
$$

等变投影和非等变残差定义为

$$
z^\parallel(a,b,c)=g(c-a-b),
\qquad
z^\perp=\widetilde z-z^\parallel.
$$

函数能量与非等变比例为

$$
E_\parallel=\lVert z^\parallel\rVert_F^2,
\qquad
E_\perp=\lVert z^\perp\rVert_F^2,
\qquad
D_{\mathrm{eq}}=\frac{E_\perp}{\lVert\widetilde z\rVert_F^2}.
$$

算法 margin、最坏残差干扰和充分界余量为

$$
\Gamma=g(0)-\max_{d\ne0}g(d),
$$

$$
I=\max_{a,b,c\ne y}\left[z^\perp(a,b,c)-z^\perp(a,b,y)\right],
\qquad
\Delta_{\mathrm{dom}}=\Gamma-I.
$$

投影模型的 split loss 为

$$
L_\parallel^{\mathrm{split}}
=\frac{1}{\lvert\mathcal D_{\mathrm{split}}\rvert}
\sum_{(a,b)\in\mathcal D_{\mathrm{split}}}
\operatorname{CE}\!\left(z^\parallel(a,b,:),y\right).
$$

预测熵和尺度归一化量为

$$
H(a,b)=-\sum_cP(c\mid a,b)\log P(c\mid a,b),
\qquad
s_z=\frac{\lVert\widetilde z\rVert_F}{\sqrt{p^3}},
$$

$$
\Gamma_{\mathrm{rms}}=\frac{\Gamma}{s_z},
\qquad
I_{\mathrm{rms}}=\frac{I}{s_z},
\qquad
\eta_\Gamma=\frac{\Gamma}{\lVert\theta\rVert_2}.
$$

函数首次事件保持固定定义：

$$
t_{\mathrm{alg}}=\min\{t:\Gamma_t>0\},
\qquad
t_{\mathrm{dom}}=\min\{t:\Gamma_t>I_t\}.
$$

事件只在 100-step regular grid 上重建。充分条件 audit 使用显式数值缓冲：只有
$\Gamma>I+10^{-10}$ 或 $\Gamma>10^{-10}$ 时才执行对应 full-accuracy 强断言。

## 5. 全时间线结果

| Step | Train / test / full accuracy | $D_{\mathrm{eq}}$ | $\Gamma$ | $I$ | $\Gamma-I$ | Projected full CE | Full normalized entropy | Centered RMS |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.0133 / 0.00868 / 0.0105 | 1.0000 | -0.000627 | 6.1628 | -6.1635 | 4.5746 | 0.9087 | 0.9047 |
| 100 | 0.9992 / 0.000177 / 0.3997 | 0.9940 | 3.5103 | 47.4606 | -43.9503 | 1.0014 | 0.2586 | 5.8811 |
| 6050 | 0.9981 / 0.7040 / 0.8217 | 0.2446 | 2.6965 | 45.9046 | -43.2081 | 0.1368 | 0.1612 | 15.7110 |
| 7000 | 1.0000 / 0.9904 / 0.9943 | 0.0893 | 6.7488 | 71.4084 | -64.6597 | 0.00362 | 0.02075 | 34.1262 |
| 11000 | 1.0000 / 1.0000 / 1.0000 | 0.2090 | 6.6197 | 68.5007 | -61.8810 | 0.00519 | 0.01245 | 35.3140 |
| 20000 | 1.0000 / 0.9897 / 0.9938 | 0.3334 | 17.2708 | 106.9401 | -89.6692 | $9.92\times10^{-8}$ | 0.01308 | 51.4341 |
| 50000 | 1.0000 / 0.9635 / 0.9781 | 0.7196 | 1.9649 | 122.7797 | -120.8148 | 0.2423 | 0.1531 | 64.2470 |

Regular grid 上，$D_{\mathrm{eq}}$ 的最小值为 0.0524（step 14000），终点回升到
0.7196；$\Gamma$ 在 step 20000 达到 17.2708，而 $I$ 在 step 36700 达到 260.1591。
$\Gamma-I$ 全程为负，最大值仍只有 -3.3673（step 3500）。Centered-logit RMS 从
step 3500 的 0.3001 到 step 47200 的 116.2164 跨越很大范围；normalized full entropy
则在 0.00399 与 0.99058 之间变化。

Projected train/test/full CE、accuracy 和 margin 分布在数值容差内一致。549 个状态中仅
step 0、7350、14250、32800、36500、36700、40800 的 $\Gamma\le0$，这些状态的
projected full accuracy 为 0；其余状态 $\Gamma>0$ 且 projected full accuracy 为 1。

## 6. 首次 Grokking 与函数事件

行为与函数事件的观测顺序为：

```text
t_fit = 100, detected_at = 300
t_alg regular-grid estimate = 100, interval = (0, 100]
t_grok50 = 6050, detected_at = 6150
t_grok99 = 7000, detected_at = 7100
t_dom = not_reached through step 50000
```

$t_{\mathrm{alg}}$ 与 $t_{\mathrm{fit}}$ 在当前 100-step 分辨率上重合，并明显早于首次测试
泛化事件。Gamma-positive regular window 共 5 段，占 501 个 regular states 的
0.99002；有 5 次 false-to-true crossing、4 次 true-to-false exit，最后一次 exit 在
step 40800，最长正窗口为 32600 steps。Replay 状态不参与这些事件统计。

$t_{\mathrm{dom}}$ 未达到不是审计失败。$\Gamma>I$ 是 raw full-table accuracy 为 1 的充分
条件而非必要条件：step 11000 的 raw full accuracy 为 1，但 $\Gamma-I=-61.8810$。因此
当前最坏样本干扰界不能解释首次测试跃迁。

## 7. 失稳 episode 的函数变化

10 个 test primitive 的 onset 相对 pre-collapse 的中位变化为：$\Delta\Gamma=-7.3526$、
$\Delta I=-63.3652$、$\Delta D_{\mathrm{eq}}=+0.4578$、normalized entropy
$+0.8019$、centered RMS $-34.6176$。Gamma 和 $I$ 在 onset 同时缩小，因此数据不支持
“残差干扰增强导致坍塌”这一表述；更直接的共同特征是函数尺度下降、offset margin 收缩、
非等变能量比例和预测熵上升。8/10 onset 的 projected full accuracy 仍为 1，2/10 降为 0。

9 个已恢复 test episode 在 recovery-confirmed 相对 pre-collapse 的中位变化为：
$\Delta\Gamma=+0.2679$、$\Delta I=+16.4653$、$\Delta D_{\mathrm{eq}}=+0.0489$、
normalized entropy $+0.00893$、centered RMS $+3.9920$。恢复并不要求 $I$ 下降到 pre-collapse
以下，也不要求 $\Gamma-I$ 转正。

`test_008` 是跨 lineage 的代表案例：step 18300 到 18350，$\Gamma$ 从 10.8189 降到
0.0646，centered RMS 从 41.5177 降到 2.8114，$D_{\mathrm{eq}}$ 从 0.2360 升到
0.8712，entropy 从 0.0123 升到 0.7798；到 test recovery-confirmed step 21400，RMS
回到 42.7692、$\Gamma$ 回到 10.1003、entropy 回到 0.0108。

`test_010` 从 step 27300 起长期未恢复。其 step 31400 trough 的 $\Gamma=0.00557$、
$D_{\mathrm{eq}}=0.9460$、centered RMS $=0.4451$、normalized entropy $=0.9816$；
step 50000 虽有显著尺度回升，test accuracy 仍为 0.9635，episode 按协议保持
`terminal_unrecovered`。`train_026` 同样在终点只有一次回升记录，不能标为 recovered。

## 8. 与参数尺度和优化诊断的对齐

参数总 L2 从 initialization 的 118.9148 降到 step 50000 的 45.2119；decay group 从
116.7173 降到 18.0785，no-decay group 从 22.7552 升到 41.4401，说明存在明显的组间尺度
重分配。全局参数范数不能单独代表函数复杂度：同一时期 centered-logit RMS、$D_{\mathrm{eq}}$
和行为仍发生大幅非单调变化。

Optimization diagnostics 只覆盖 step 20050–50000，不能外推到此前 episode。在该区间的
15 个 train-collapse onset 中，相对前一 evaluation 的中位倍数为：gradient 26.75 倍、
total update 13.53 倍、data update 8.45 倍、decay update 1.02 倍、data/decay ratio
9.61 倍、Adam first moment 170.15 倍；total/data update 和 first moment 在 15/15 onset
上升。仅有 2 个 test onset，样本不足以形成稳定总体估计。

这些记录支持“collapse onset 与 data-update/Adam-state 突增同步”的描述性关联。它们不证明
optimizer 导致坍塌：诊断频率为 50 step、没有干预分支，且多个函数量同时变化。

## 9. 主要发现

1. **Checkpoint 真实性（直接验证）**：所有强制失稳状态可由 exact checkpoint 或通过
   bridge 的 deterministic replay 解析，M1 scalar 与 reference evaluator 一致。
2. **早期等变 profile（直接测量）**：step 100 已有 $\Gamma>0$ 和 projected full
   accuracy 1，但 raw test accuracy 接近 0；投影结构早于 raw 泛化。
3. **非单调函数结构（直接测量）**：$\Gamma$ 后续有四次退出正区间，$D_{\mathrm{eq}}$、
   logit scale、entropy 和 projected loss 均随 collapse/recovery 大幅往复。
4. **充分界未穿越（直接测量）**：$\Gamma-I$ 在完整 regular grid 上始终为负，
   $t_{\mathrm{dom}}$ 未达到；raw 高准确率不要求该最坏样本充分界成立。
5. **失稳中心化特征（支持的关联）**：test onset 的典型模式是 Gamma、$I$ 和整体 logit
   尺度共同下降，同时 $D_{\mathrm{eq}}$ 比例与 entropy 上升。
6. **优化同步（支持的关联）**：M1-C 范围内多数 train onset 同步出现 data update 与
   Adam first moment 的尖峰；没有干预证据可把这一同步升级为原因。

## 10. 初步解释与证据等级

- **Observation**：Reynolds-projected profile 在训练拟合时已经能正确分类完整表，但 raw
  logits 仍受输入特异残差影响。
- **Supported association**：严重 collapse 同时伴随算法 margin 与 centered-logit scale
  收缩、entropy 上升和等变能量占比下降；恢复呈反向变化。
- **Hypothesis**：重复 collapse 可能反映整体函数尺度与输入特异残差在 AdamW 更新中的
  暂态重排，而不是单向 residual cleanup。
- **Unverified mechanism**：具体 parameter group、head、MLP 或 optimizer moment 是否驱动
  这些变化，需要 Gate 2 的复现和 M5 的干预；M2 数据本身不能回答。

## 11. 核心假设评估

本节按固定 PDF 的 H1–H4 编号，不使用仓库后续阶段新增工作假说改写编号。

- **H1（结构提前形成）：partially supported。** $t_{\mathrm{alg}}$ 的网格估计为 100，
  早于 $t_{\mathrm{grok99}}=7000$，且 step 100 的 projected full accuracy 已为 1。
  但 $L_\parallel$ 在后续 collapse 中反复上升，不满足全程持续下降；restricted/key-frequency
  证据属于 M3，尚未检验。
- **H2（阈值穿越解释跃迁）：not supported in this seed。** $\Gamma-I$ 到 50000 始终
  为负，$t_{\mathrm{dom}}$ 未达到，而 raw full accuracy 已多次接近或达到 1。Residual
  scaling 干预尚未实施。
- **H3（cleanup 对应非等变残差清理）：partially supported。** $D_{\mathrm{eq}}$ 从
  step 100 的 0.994 降到 step 7000 的 0.089，并在 collapse 时上升、恢复时下降；但长期
  轨迹不单调，终点为 0.720，$I$ 也不持续下降。非目标 Fourier 能量和 WD-off 分支尚未检验。
- **H4（算法回路具有更高 margin–norm 效率）：inconclusive。** $\Gamma/\lVert\theta\rVert_2$
  在 step 20000 达到 0.4665，终点降到 0.04346；单 seed 轨迹不显示持续稳定上升，且没有
  module-level efficiency 或 WD branch 对照。

## 12. 限制

- 只有 CE-reference seed 1，尚不能判断失稳与函数变化能否跨 seed 复现。
- $I$ 是全表最坏样本充分界，可能比实现高准确率实际所需条件保守。
- 50-step replay 提高了 episode 分辨率，但 regular 函数事件仍只有 100-step 区间分辨率。
- Batch-1024 前向有 8 个近边界 prediction 差异；这些状态已由 reference full-batch evaluator
  复核，但说明离散 argmax 对数值批处理敏感。
- 参数范数受重参数化和 LayerNorm/no-decay 尺度迁移影响，不能等同于函数复杂度。
- 完整函数指标仍不能单独证明 head/MLP 电路或 optimizer 的因果作用。
- 没有 Fourier、restricted logits、表征或干预证据。

## 13. M2 结论

M2-A 确认 M1 的反复坍塌与恢复属于可重算的模型状态，不是 JSONL、lineage 或 checkpoint
异常。M2-B 显示等变 offset profile 很早出现，但 raw 函数是否泛化取决于更丰富且非单调的
函数状态；collapse 的共同特征是算法 margin 和函数尺度收缩、相对非等变能量及 entropy
上升。最坏残差充分界从未被跨越，故不能用单一 $\Gamma>I$ 事件解释这条轨迹。

以上结论建立了函数空间层面的描述性证据，并排除了若干过于单调的解释；它没有识别具体
电路，也没有证明 optimization cause。M2 overall 的完成表示管线、正式单 seed 分析和两阶段
audit 完成，不表示机制研究结束。

## 14. Gate 2 交接

Gate 2 对 CE、WD=0.5、seed 2/3 预注册复现以下内容：

- M1 首次事件、stable window、train/test/joint collapse episode 数量与持续时间；
- $t_{\mathrm{alg}}$、$t_{\mathrm{dom}}$、Gamma-positive runs 和 dominance runs；
- collapse onset 前后的 $\Gamma$、$I$、$D_{\mathrm{eq}}$、centered RMS、entropy 与
  projected behavior；
- 绝对 step、相对 first grokking event 和相对 collapse onset 三种对齐；
- checkpoint/replay 真实性与 source zero-write audit。

Gate 2 只做行为层复现和预注册函数指标核验。M3 Fourier、M4 完整 WD grid、M5 表征/
电路/干预、M6 congruence 继续保持 planned。

## 15. 可复核文件

- 本地 analysis：`analysis_runs/20260726T185412278703Z_5337a9bb/`；
- Portable results：[`results/m2_function_space/`](../results/m2_function_space/README.md)；
- Analysis audit：
  [`m2_analysis.json`](../results/m2_function_space/audit/m2_analysis.json)；
- Export audit：[`m2_export.json`](../results/m2_function_space/audit/m2_export.json)；
- Checkpoint index：
  [`checkpoint_index.csv`](../results/m2_function_space/checkpoint_index.csv)；
- M2-A validation：
  [`checkpoint_validation.csv`](../results/m2_function_space/m2a/checkpoint_validation.csv)；
- Function timeline：
  [`function_metrics.csv`](../results/m2_function_space/m2b/function_metrics.csv)；
- Function events：
  [`function_events.json`](../results/m2_function_space/m2b/function_events.json)；
- Offset profiles：
  [`offset_profiles.npz`](../results/m2_function_space/m2b/offset_profiles.npz)；
- Figures：[`figures/`](../results/m2_function_space/figures/)；
- Selected tensor manifest：
  [`manifest.json`](../results/m2_function_space/selected_tensors/manifest.json)；
- Portable provenance：
  [`provenance.json`](../results/m2_function_space/provenance.json)。

17 个 selected tensor 共 252.61 MiB，超过 100 MiB portable export 阈值，因此完整 `.npz`
保留在本地 analysis run；Git results 只保存其 repository-relative path、shape、dtype、size 和
SHA-256 清单。所有正式图表使用原始 step，不平滑、不插值、不删除异常点。
