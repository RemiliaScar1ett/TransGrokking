# TransGrokking 理论框架

## 1. 文档目的

本文档为 TransGrokking 重构实验提供统一的理论语言、可计算指标与可证伪假设。研究对象是小型 Transformer 在有限群算法任务上的 Grokking 过程。关注重点是训练集拟合之后的内部演化，包括函数结构形成、非结构残差清理、参数复杂度压力和最终泛化跃迁。

项目采用“训练时间显微镜”的思路。每一个 checkpoint 都被视为一个独立模型状态，并从行为、函数、表征、电路和优化五个层级进行测量。最终测试准确率只承担阶段标记作用，机制判断需要依赖更细粒度的连续指标和因果干预。

---

## 2. 基础任务与符号

### 2.1 模加法任务

给定素数或正整数 $p$，定义有限循环群

$$
\mathbb Z_p=\{0,1,\ldots,p-1\}.
$$

输入为

$$
a,b\in\mathbb Z_p,
$$

目标为

$$
y=(a+b)\bmod p.
$$

完整运算表包含 $p^2$ 个输入对。训练集 $S_{\mathrm{tr}}$ 从完整集合中固定抽取，测试集 $S_{\mathrm{te}}$ 由剩余输入对构成。基础实验使用 $p=97$ 和 40% 训练比例。

### 2.2 模型 logits

模型对输入 $(a,b)$ 输出长度为 $p$ 的 logit 向量：

$$
z_\theta(a,b)
=
\bigl(z_\theta(a,b,0),\ldots,z_\theta(a,b,p-1)\bigr).
$$

预测概率为

$$
P_\theta(c\mid a,b)
=
\frac{\exp z_\theta(a,b,c)}
{\sum_{j=0}^{p-1}\exp z_\theta(a,b,j)}.
$$

预测类别为

$$
\widehat y_\theta(a,b)
=
\arg\max_c z_\theta(a,b,c).
$$

将所有输入和候选输出放在一起，可以得到完整 logit 张量

$$
z_\theta:\mathbb Z_p^3\rightarrow\mathbb R.
$$

softmax 对每个输入上的公共平移保持不变。函数分析前使用中心化 logits：

$$
\widetilde z(a,b,c)
=
z(a,b,c)
-
\frac1p\sum_{j=0}^{p-1}z(a,b,j).
$$

该操作消除不影响预测的类别公共偏置。

---

## 3. Grokking 的操作性定义

Grokking 指训练准确率已经接近饱和后，测试性能经过明显延迟才快速改善的训练动力学。实验中预先定义事件时间，避免根据单条曲线临时调整阶段边界。

训练拟合时间定义为

$$
t_{\mathrm{fit}}
=
\min\left\{
 t:\operatorname{Acc}_{\mathrm{tr}}(t)\ge 99.9\%
\right\},
$$

并要求该条件在连续若干次评估中成立。

测试跃迁中点定义为

$$
t_{\mathrm{grok50}}
=
\min\left\{
 t:
\operatorname{Acc}_{\mathrm{te}}(t)
\ge
\frac{1+1/p}{2}
\right\}.
$$

高泛化时间定义为

$$
t_{\mathrm{grok99}}
=
\min\left\{
 t:\operatorname{Acc}_{\mathrm{te}}(t)\ge99\%
\right\}.
$$

Grokking 延迟可以写为

$$
\Delta t_{\mathrm{grok}}
=
t_{\mathrm{grok99}}-t_{\mathrm{fit}}.
$$

这些行为指标用于定位阶段。内部机制需要由后续结构指标解释。

### 3.1 首次泛化与稳定泛化

上述 $t_{\mathrm{fit}}$、$t_{\mathrm{grok50}}$ 和
$t_{\mathrm{grok99}}$ 都是首次连续阈值事件：事件时间记录第一次满足既定连续窗口的
起点。它们用于定位首次拟合与首次泛化，不表示模型此后会永久停留在高性能区域。

为区分首次泛化与长期稳定泛化，定义稳定泛化事件

$$
t_{\mathrm{stable99}}(H)
=
\min
\left\{
t:
\operatorname{Acc}_{\mathrm{test}}(s)\ge 0.99,
\ \forall s\in[t,t+H]
\right\}.
$$

初始 measurement protocol 取 $H=100$ 个 evaluation；在
`eval_interval=50` 时对应 5000 个 optimizer steps。该窗口长度属于测量协议，可以在
后续版本中单独评估敏感性，不改写原有首次事件。

### 3.2 坍塌与恢复 episode

在首次事件之后，使用以下操作性定义识别性能坍塌：

- **train collapse**：已经达到 $t_{\mathrm{fit}}$ 后，train accuracy 下降至 0.9
  以下；
- **test collapse**：已经达到 $t_{\mathrm{grok99}}$ 后，test accuracy 下降至 0.9
  以下；
- **joint collapse**：train collapse 与 test collapse 出现在同一次 evaluation，或
  相邻一次 evaluation 内。

恢复判据分别为：

- **train recovery**：train accuracy 连续 3 次不低于 0.999；
- **test recovery**：test accuracy 连续 3 次不低于 0.99。

每个 episode 记录 `collapse_onset`、`collapse_trough`、`collapse_depth`、
`recovery_step` 和 `recovery_duration`。序列结束时仍未满足恢复判据的 episode
明确标记为未恢复。这些阈值是失稳分析的操作性定义，不等价于机制结论。

### 3.3 样本分类 margin 与错误 offset

M1 对每个样本记录分类 margin：

$$
m(a,b)=z_y(a,b)-\max_{c\ne y}z_c(a,b).
$$

错误类别最大值必须排除正确类别。正 margin 表示当前样本分类正确，负 margin 表示至少
一个错误类别 logit 更大。时间线分别记录 train/test 的均值、最小值和固定分位数。

对误分类样本定义循环错误 offset：

$$
\Delta=(\widehat y-y)\bmod p.
$$

M1 直方图只统计误分类，因此 offset 0 固定为零。该量描述预测相对正确模加法结果的循环
偏移，不等同于后续 M2 的 Reynolds offset profile。

---

## 4. 压缩视角

“压缩即智能”在本项目中被限定为以下命题：

> 在保持训练似然和分类 margin 的条件下，模型逐渐消除与任务对称性无关的信息，并形成能够跨样本共享的低复杂度函数结构。

函数复杂度可以采用最小参数实现代价：

$$
\mathcal C(z)
=
\inf_{\theta:f_\theta=z}
\|\theta\|_2^2.
$$

它描述模型架构实现给定 logit 函数所需的最低平方范数。该定义依赖参数化，因此需要配合函数空间指标使用。

对于模加法，结构化解可以共享同一个群规律

$$
c-a-b\equiv0\pmod p.
$$

样本记忆解通常需要维护大量输入对特定的响应。若当前 Transformer 参数化满足

$$
\mathcal C(z_{\mathcal A})
<
\mathcal C(z_{\mathcal M}),
$$

其中 $z_{\mathcal A}$ 表示算法解，$z_{\mathcal M}$ 表示记忆解，权重衰减将长期偏向前者。

---

## 5. 模加法的群对称性

### 5.1 群作用

对任意 $u,v\in\mathbb Z_p$，定义

$$
T_{u,v}(a,b,c)
=
(a+u,b+v,c+u+v),
$$

所有运算均在 $\mathbb Z_p$ 中完成。

该变换保持差值

$$
d=c-a-b\pmod p
$$

不变。正确类别统一对应 $d=0$。

对固定 $(a,b,c)$，集合

$$
\mathcal O(a,b,c)
=
\left\{
T_{u,v}(a,b,c):u,v\in\mathbb Z_p
\right\}
$$

称为其群轨道。轨道中的所有三元组拥有相同的 $c-a-b$。

### 5.2 雷诺兹投影

对任意函数 $z:\mathbb Z_p^3\to\mathbb R$，定义

$$
(\Pi z)(a,b,c)
=
\frac1{p^2}
\sum_{u,v\in\mathbb Z_p}
z(a+u,b+v,c+u+v).
$$

该算子沿群轨道取平均。投影结果满足

$$
(\Pi z)(a+s,b+t,c+s+t)
=
(\Pi z)(a,b,c).
$$

因此存在一维函数 $g:\mathbb Z_p\to\mathbb R$，使得

$$
(\Pi z)(a,b,c)
=
g(c-a-b).
$$

实际计算可以直接使用

$$
g(d)
=
\frac1{p^2}
\sum_{a,b\in\mathbb Z_p}
\widetilde z(a,b,a+b+d).
$$

雷诺兹算子满足幂等性

$$
\Pi^2=\Pi.
$$

在均匀内积

$$
\langle f,h\rangle
=
\frac1{p^3}
\sum_{a,b,c}f(a,b,c)h(a,b,c)
$$

下，$\Pi$ 是正交投影。

### 5.3 函数分解

定义

$$
z^\parallel=\Pi\widetilde z,
\qquad
z^\perp=(I-\Pi)\widetilde z.
$$

于是

$$
\widetilde z=z^\parallel+z^\perp,
\qquad
\langle z^\parallel,z^\perp\rangle=0.
$$

其中：

- $z^\parallel$ 表示满足模加法平移对称性的函数成分；
- $z^\perp$ 表示轨道内部变化与非等变成分。

定义非等变能量比例

$$
D_{\mathrm{eq}}
=
\frac{\|z^\perp\|_2^2}
{\|\widetilde z\|_2^2}.
$$

对应的等变能量比例为

$$
A_{\mathrm{eq}}=1-D_{\mathrm{eq}}.
$$

该指标直接作用于模型函数，能够避开部分参数缩放冗余。

---

## 6. Fourier 空间

### 6.1 有限循环群上的 Fourier 基

令

$$
\omega=e^{2\pi i/p}.
$$

对每个频率 $r\in\mathbb Z_p$，定义 character

$$
\chi_r(x)=\omega^{rx}.
$$

任意函数 $f:\mathbb Z_p\to\mathbb C$ 都有展开

$$
f(x)
=
\sum_{r=0}^{p-1}
\widehat f(r)\omega^{rx},
$$

其中

$$
\widehat f(r)
=
\frac1p\sum_{x=0}^{p-1}f(x)\omega^{-rx}.
$$

character 满足

$$
\chi_r(x+y)=\chi_r(x)\chi_r(y),
$$

因此适合表示群加法结构。

### 6.2 三维 Fourier 展开

完整 logits 可以写为

$$
\widetilde z(a,b,c)
=
\sum_{r_a,r_b,r_c}
\widehat z(r_a,r_b,r_c)
\omega^{r_a a+r_b b+r_c c}.
$$

模加法目标张量为

$$
Y(a,b,c)=\mathbf1[c=a+b\pmod p].
$$

利用有限群正交关系

$$
\mathbf1[d=0]
=
\frac1p\sum_{r=0}^{p-1}\omega^{rd},
$$

可以得到

$$
Y(a,b,c)
=
\frac1p\sum_{r=0}^{p-1}
\omega^{r(a+b-c)}.
$$

目标 Fourier 支撑集中在

$$
(r_a,r_b,r_c)=(r,r,-r).
$$

这条频率线记为

$$
\mathcal L
=
\{(r,r,-r):r\in\mathbb Z_p\}.
$$

### 6.3 雷诺兹投影的频域含义

对单个 Fourier 模式

$$
\phi_{r_a,r_b,r_c}(a,b,c)
=
\omega^{r_a a+r_b b+r_c c}
$$

进行群平均，会产生因子

$$
\frac1p\sum_u\omega^{(r_a+r_c)u}
\cdot
\frac1p\sum_v\omega^{(r_b+r_c)v}.
$$

只有以下条件同时成立时该因子非零：

$$
r_a+r_c=0,
\qquad
r_b+r_c=0.
$$

因此

$$
r_a=r_b=-r_c.
$$

雷诺兹投影在 Fourier 空间中等价于保留 $\mathcal L$ 上的系数。

定义目标频率线能量比例

$$
E_{\mathrm{line}}
=
\frac{
\sum_r|\widehat z(r,r,-r)|^2
}{
\sum_{r_a,r_b,r_c}
|\widehat z(r_a,r_b,r_c)|^2
}.
$$

在变换归一化一致时，Parseval 定理给出

$$
E_{\mathrm{line}}
=
A_{\mathrm{eq}}.
$$

这个等式可以用于校验雷诺兹投影与 Fourier 实现。

---

## 7. Margin 阈值与泛化跃迁

令正确类别为

$$
y=a+b\pmod p.
$$

算法投影只依赖 offset $d=c-a-b$。定义全局算法 margin

$$
\Gamma
=
g(0)-\max_{d\ne0}g(d).
$$

定义非等变残差的最大对抗干扰

$$
I
=
\max_{a,b,c\ne y}
\left[
 z^\perp(a,b,c)-z^\perp(a,b,y)
\right].
$$

对任意错误类别 $c\ne y$，有

$$
\begin{aligned}
\widetilde z(a,b,y)-\widetilde z(a,b,c)
&=
\left[z^\parallel(a,b,y)-z^\parallel(a,b,c)\right]\\
&\quad+
\left[z^\perp(a,b,y)-z^\perp(a,b,c)\right]\\
&\ge
\Gamma-I.
\end{aligned}
$$

因此得到充分条件：

$$
\boxed{\Gamma>I\Longrightarrow
\widehat y(a,b)=a+b\pmod p
\text{ 对全部输入成立}.}
$$

这一定理允许连续内部变量解释离散准确率跃迁。训练期间 $\Gamma_t$ 可以逐渐增加，$I_t$ 可以逐渐下降。当两条曲线发生穿越时，大量测试样本可能在相近时间改变 argmax。

定义结构形成时间

$$
t_{\mathrm{alg}}
=
\min\{t:\Gamma_t>0\}.
$$

定义结构支配时间

$$
t_{\mathrm{dom}}
=
\min\{t:\Gamma_t>I_t\}.
$$

若

$$
t_{\mathrm{alg}}<t_{\mathrm{grok99}}
$$

且

$$
t_{\mathrm{dom}}\approx t_{\mathrm{grok99}},
$$

则实验支持“算法结构提前形成，残差清理触发可见跃迁”的过程解释。

---

## 8. 权重范数惩罚

### 8.1 参数动力学

带平方范数惩罚的目标可写为

$$
\mathcal L_{\mathrm{reg}}(\theta)
=
\mathcal L_{\mathrm{data}}(\theta)
+
\frac\lambda2\|\theta\|_2^2.
$$

连续梯度流满足

$$
\dot\theta
=
-\nabla_\theta\mathcal L_{\mathrm{data}}
-
\lambda\theta.
$$

AdamW 采用解耦权重衰减。简化更新形式为

$$
\theta_{t+1}
=
(1-\eta_t\lambda)\theta_t
-
\eta_tP_t\nabla_\theta\mathcal L_{\mathrm{data}},
$$

其中 $P_t$ 表示自适应预条件。

权重衰减直接作用于参数。它对 logits 的影响通过 Jacobian

$$
J_\theta=\frac{\partial z}{\partial\theta}
$$

传递：

$$
\dot z
=
J_\theta\dot\theta.
$$

因此，权重衰减不会自动执行雷诺兹投影。任务结构、架构参数化和数据梯度共同决定不同函数成分的收缩速度。

### 8.2 两回路近似

将模型函数近似分解为

$$
z_t
=
a_tz_{\mathcal A}+b_tz_{\mathcal M},
$$

其中 $z_{\mathcal A}$ 表示算法回路，$z_{\mathcal M}$ 表示记忆回路。

设二者的有效参数代价为 $c_{\mathcal A}$ 和 $c_{\mathcal M}$，正则项近似为

$$
\mathcal R(a,b)
=
\frac\lambda2
\left(
 c_{\mathcal A}a^2+c_{\mathcal M}b^2
\right).
$$

若

$$
c_{\mathcal M}>c_{\mathcal A},
$$

记忆回路会受到更强的长期复杂度压力。训练早期的数据梯度可能使 $b_t$ 快速增长并完成训练集拟合。训练后期的参数收缩和共享结构梯度可以提高算法回路的相对占比。

在非等变干扰近似指数下降时，写作

$$
I_t
\approx
I_{t_0}
\exp[-\lambda_{\mathrm{eff}}(t-t_0)].
$$

若算法 margin 近似稳定为 $\Gamma_\infty>0$，则阈值穿越时间满足

$$
t_{\mathrm{dom}}-t_0
\approx
\frac1{\lambda_{\mathrm{eff}}}
\log\frac{I_{t_0}}{\Gamma_\infty}.
$$

该表达式给出一种可检验的延迟律。权重衰减过弱时，残差清理可能极慢；权重衰减过强时，算法 margin 也可能难以形成。

### 8.3 参数范数的解释边界

Transformer 含有层间缩放、LayerNorm 和多种重参数化自由度。原始参数范数无法唯一确定函数复杂度。项目需要同时记录：

$$
\|\theta_t\|_2,
\qquad
D_{\mathrm{eq}}(t),
\qquad
E_{\mathrm{line}}(t),
\qquad
\Gamma_t,
\qquad
I_t.
$$

解释参数尺度时同时遵守以下边界：

- 全局参数 L2 下降不保证函数复杂度单调下降；
- decay 与 no-decay 参数组可能发生显著的尺度重分配；
- LayerNorm 参数增长可能改变参数范数与函数尺度之间的对应关系；
- 机制判断需要函数空间指标与优化动力学证据共同支持。

参数指标与函数指标共同变化时，复杂度解释才具有较强可信度。

---

## 9. Congruence loss

当前原型使用圆周距离惩罚：

$$
L_{\mathrm{cong}}
=
\sum_{k=0}^{p-1}
P_\theta(k\mid a,b)
\left[
1-
\cos\frac{2\pi(k-y)}p
\right].
$$

令 $\omega=e^{2\pi i/p}$，可写为

$$
L_{\mathrm{cong}}
=
1-
\operatorname{Re}
\left[
\omega^{-y}
\sum_kP_\theta(k\mid a,b)\omega^k
\right].
$$

该损失直接监督预测分布的第一圆周 Fourier 矩。它可能提高低频循环坐标的学习速度，也可能改变最终频谱分布。

实验需要区分两类效应：

1. **动力学效应**：算法模式更早形成，最终函数结构接近 CE 基线；
2. **结构效应**：最终 Fourier 能量分配、margin 或表征维度发生稳定改变。

核心比较指标为

$$
t_{\mathrm{alg}},
\quad
t_{\mathrm{dom}},
\quad
E_{\mathrm{line}}(t),
\quad
e_r(t)=|\widehat z(r,r,-r)|^2.
$$

---

## 10. 训练阶段假说

本项目以四个基础阶段和三个失稳扩展阶段组织观察结果。阶段边界由连续指标和行为事件
共同决定；这些阶段可以重叠、重复或缺失，不预设训练轨迹单调前进。

### 10.1 Memorization

训练误差快速下降，测试表现仍接近随机水平。可能出现的内部特征包括：

- $D_{\mathrm{eq}}$ 较高；
- 非目标 Fourier 模式能量广泛分布；
- 训练集 residual logits 获得较大 margin；
- 参数范数或局部模块范数快速变化。

### 10.2 Circuit formation

算法投影开始获得可用结构：

$$
\Gamma_t\uparrow,
\qquad
L_\parallel(t)\downarrow,
\qquad
E_{\mathrm{line}}(t)\uparrow.
$$

测试准确率可能仍然较低，因为 $I_t$ 足以推翻结构化预测。

### 10.3 Cleanup

非等变干扰和非目标频率逐渐下降：

$$
D_{\mathrm{eq}}(t)\downarrow,
\qquad
I_t\downarrow.
$$

当 $\Gamma_t>I_t$ 后，测试准确率进入快速提升区间。

### 10.4 Stable algorithmic regime

训练与测试性能均处于高位，目标频率线和模块归因趋于稳定。后续训练主要改变 margin 尺度、参数范数或校准性质。

### 10.5 Instability / relapse

模型在已经达到首次拟合或首次高泛化事件后离开高性能区域。该阶段首先由行为坍塌
episode 定位；是否对应真实模型状态，以及是否伴随函数结构、logit 尺度或优化状态变化，
需要 checkpoint 重算和后续指标验证。

### 10.6 Recovery

模型从坍塌谷底重新满足 train recovery 或 test recovery 判据。恢复可以回到原先的函数
状态，也可能形成不同的函数与参数尺度组合，二者必须通过共享时间线区分。

### 10.7 Repeated transition

同一轨迹可以多次进入和离开高性能区域，形成反复失稳与恢复。Grokking 因而可能表现为
非单调的重复转变，而非一次阈值穿越后永久稳定。

该阶段模板属于待检验假说。不同架构和损失可能产生阶段重叠、顺序变化、重复或缺失。

---

## 11. 多层观测体系

### 11.1 行为层

主要指标包括：

$$
L_{\mathrm{tr}},
\quad
L_{\mathrm{te}},
\quad
\operatorname{Acc}_{\mathrm{tr}},
\quad
\operatorname{Acc}_{\mathrm{te}},
$$

以及正确类别 margin 的分位数、错误 offset 分布和完整 $p\times p$ 错误热图。

### 11.2 函数层

主要对象为完整 logits。核心指标包括：

$$
D_{\mathrm{eq}},
\quad
E_{\mathrm{line}},
\quad
\Gamma,
\quad
I,
\quad
L_\parallel.
$$

该层构成全过程分析的主轴。

### 11.3 表征层

对 token embedding $E(x,:)$ 沿 token 轴做 DFT：

$$
\widehat E(r,:)
=
\frac1p\sum_xE(x,:)e^{-2\pi irx/p}.
$$

记录频率能量

$$
e_E(r)=\|\widehat E(r,:)\|_2^2.
$$

对 hidden state $H_l(a,b,:)$ 做二维 DFT，并训练冻结表示的线性 probe。有效秩使用 participation ratio：

$$
r_{\mathrm{eff}}
=
\frac{(\sum_i\lambda_i)^2}
{\sum_i\lambda_i^2}.
$$

Fourier 对齐、probe 泛化和有效秩需要联合解释。

### 11.4 电路层

对 attention head、MLP neuron 和 unembedding 进行模块归因。主要因果工具包括：

- 单模块消融；
- 关键 Fourier 频率投影与删除；
- early/late checkpoint 模块移植；
- activation patching；
- 模块冻结后继续训练。

相关性分析用于定位候选电路，干预实验用于检验必要性和充分性。

### 11.5 优化层

记录每个模块的参数范数、梯度范数和实际 AdamW 更新。将一步更新拆分为

$$
\Delta\theta_{\mathrm{data}}
$$

和

$$
\Delta\theta_{\mathrm{decay}}
=-\eta\lambda\theta.
$$

定义相对强度

$$
R_{\mathrm{decay/data}}
=
\frac{\|\Delta\theta_{\mathrm{decay}}\|}
{\|\Delta\theta_{\mathrm{data}}\|}.
$$

checkpoint 分支可以分别关闭数据更新或 decay，并测量 $D_{\mathrm{eq}}$、$\Gamma$ 和 $I$ 的一步变化。

---

## 12. 核心可证伪假设

### H1：结构提前形成

在测试准确率明显提升以前，满足

$$
\Gamma_t>0
$$

且 $L_\parallel(t)$ 已经持续下降。

### H2：阈值穿越解释跃迁

测试准确率快速提升发生在

$$
\Gamma_t-I_t
$$

由负转正的邻近区间。

### H3：cleanup 对应非等变残差下降

训练后期的主要函数变化表现为

$$
D_{\mathrm{eq}}(t)\downarrow
$$

和非目标 Fourier 能量下降。

### H4：适度权重衰减缩短延迟

在训练能够完成拟合的范围内，提高 weight decay 会增大有效残差清理速率，并改变 $t_{\mathrm{dom}}$。

### H5：算法回路具有更高 margin—norm 效率

定义

$$
\eta_{\mathrm{margin}}
=
\frac{\Gamma}{\|\theta\|_2}
$$

或模块级对应量。算法阶段的该指标应高于记忆阶段。

### H6：congruence loss 优先增强低阶循环模式

在配对初始化实验中，congruence loss 会更早提高 $r=1$ 或邻近低阶目标模式能量。

### H7：关键频率具有因果作用

删除最终模型的关键目标频率会显著降低泛化性能；保留关键频率的受限模型能够复现主要算法行为。

任何假设都允许被单独否定。项目目标是建立清晰的过程图景，实验结果无需迎合预设阶段模型。

---

## 13. 理论边界

### 13.1 等变残差与记忆

$z^\perp$ 精确描述对称性破坏。某些具有群对称性的复杂函数仍然位于 $z^\parallel$ 中。因此，$D_{\mathrm{eq}}$ 不能覆盖全部形式的过拟合。

### 13.2 Fourier 稀疏与算法性

目标频率线能量提高说明函数更加符合模加法对称性。频谱相关性本身不构成电路因果证据，仍需频率消融、投影和模块干预。

### 13.3 权重范数与复杂度

参数范数受到架构和归一化影响。跨架构比较应优先使用函数指标、归一化 margin 或最小实现代价的近似量。

### 13.4 测试集使用

全过程机制分析需要访问完整运算表。超参数选择应使用预先固定的规则或独立验证划分，避免根据最终测试曲线反复调参。

### 13.5 数值精度

TF32、BF16、dropout 和未固定随机数都可能改变 Grokking 时间。基准轨迹应使用显式精度、固定 seed 和确定的 dropout 配置。

---

## 14. 实验设计原则

1. **密集 checkpoint**：保存参数时间序列，关键阶段同时保存 optimizer state。
2. **完整函数评估**：对全部 $p^2$ 个输入计算 logits，函数指标不依赖训练/测试抽样噪声。
3. **配对运行**：损失或正则化比较使用同一数据划分与同一初始化。
4. **多 seed 检查**：深度分析选择代表轨迹，主要结论需要在多个初始化上验证。
5. **因果分支**：从同一 checkpoint 修改 weight decay、冻结模块或重置 optimizer state。
6. **固定指标定义**：事件阈值和分析公式在运行前确定。
7. **保存环境元数据**：记录 commit、PyTorch/CUDA 版本、随机种子和参数组。

---

## 15. 预期主时间线

每条代表轨迹最终应能够形成以下叙述模板：

1. 模型在 $t_{\mathrm{fit}}$ 完成训练集拟合；
2. 算法投影在 $t_{\mathrm{alg}}$ 获得正 margin；
3. 非等变干扰在随后阶段持续下降；
4. $\Gamma_t$ 在 $t_{\mathrm{dom}}$ 超过 $I_t$；
5. 测试准确率在相邻区间进入快速提升；
6. 轨迹可能经历一次或多次 instability / relapse 与 recovery；
7. 后期电路与频谱在足够长的稳定窗口内逐渐稳定。

若实际轨迹偏离该模板，应以测量结果为准，并重新构建阶段解释。首次行为事件与稳定窗口
分别记录，不用后续坍塌改写首次事件。

---

## 16. 参考工作

- Power, Alet, Bakhtin, Misra. *Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets*. 2022.
- Nanda, Chan, Lieberum, Smith, Steinhardt. *Progress Measures for Grokking via Mechanistic Interpretability*. 2023.

本文档将随实验重构持续更新。理论公式负责定义测量对象，实验干预负责决定哪些解释可以保留。
