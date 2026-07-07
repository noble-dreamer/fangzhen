1.加强x的条件注入方式

x_matrix:[B,7,15,16,16]

reshape -> [B,105,16,16]

CNN -> global avg pool -> [B,256]

也就是说，整个 TX-RX-frequency 信息最后被压缩成一个全局向量：
然后通过 FiLM 注入 U-Net。

这有一个明显问题：

> xx**x** 的空间结构、TX-RX 几何关系、频率选择信息被压缩得太狠了。

对于超声反演，某个 TX-RX 对应某条传播路径；某些频率对某些缺陷尺度敏感。如果全部 global average pool 成一个向量，模型可能学不到精细的几何对应关系。

## 1.1 改进：让 XMatrixEncoder 输出 tokens，而不是一个向量

现在是：

$E_x(x) \in \mathbb{R}^{B\times 256}$

建议改成：

$E_x(x) \in \mathbb{R}^{B\times N\times d}$

例如：

$x \in \mathbb{R}^{B\times7\times15\times16\times16}$

reshape 后：

$\tilde x \in \mathbb{R}^{B\times105\times16\times16}$

经过 CNN 得到：

$f_x \in \mathbb{R}^{B\times C\times 16\times16}$

然后 flatten：

$T_x \in \mathbb{R}^{B\times256\times C}$

再投影：

$T_x \in \mathbb{R}^{B\times256\times d}$

这里 256 个 token 对应 TX-RX 平面位置。

然后在 U-Net 的中低分辨率层加入 cross-attention：

$Q = F_{\mathrm{image}}W_Q$

$K = T_xW_K$

$V = T_xW_V$

$\mathrm{CrossAttn}(F,T_x)

=

\mathrm{softmax}

\left(

\frac{QK^\top}{\sqrt d}

\right)V$

这样每个图像位置可以主动查询物理测量 token。

这比单纯 FiLM 更强。

结构变成：

h←SelfAttn(h)h \leftarrow \mathrm{SelfAttn}(h)**h**←**SelfAttn**(**h**)
h←h+CrossAttn(h,Tx)h \leftarrow h + \mathrm{CrossAttn}(h,T_x)**h**←**h**+**CrossAttn**(**h**,**T**x)
其中：

Tx=Extoken(x)T_x = E_x^{token}(x)**T**x=**E**x**t**o**k**e**n****(**x**)**

保留 FiLM，同时增加 cross-attention：

2. 引入 ControlNet/T2I-Adapter 思路：让粗图走独立控制分支

<pre node="[object Object]"><div class="sc-cgHfjM rztuA code-block"></div></pre>

$[y_t,\mathrm{pic}(x)]$

直接 concat：

这是最简单有效的方法。

但现在扩散模型领域中，处理空间条件常见做法是类似 ControlNet / Adapter：

* 主 U-Net 负责去噪；
* 条件网络单独编码 `pic(x)`；
* 把条件特征注入到 U-Net 的多尺度层。

## 2.2 改进做法

建立一个 `PicEncoder`：

$p \in \mathbb{R}^{B\times8\times256\times256}$
输出多尺度条件：

$\{p_{256},p_{128},p_{64},p_{32}\}$
其中：

**$p_{256} \in \mathbb{R}^{B\times48\times256\times256}$
$p_{128} \in \mathbb{R}^{B\times96\times128\times128}$
$p_{64} \in \mathbb{R}^{B\times192\times64\times64}$
**$p_{32} \in \mathbb{R}^{B\times192\times32\times32}$
然后注入 U-Net：

$h_l \leftarrow h_l + Z_l(p_l)$
其中 ZlZ_l**Z**l 是 zero-conv，初始化为零：

$Z_l(p_l)=\mathrm{Conv}_{1\times1}(p_l)$
初始时不影响原网络，训练稳定。

这就是 ControlNet 的核心思想。

3. 物理一致性：从“损失项”升级为“采样过程中的数据一致性引导”

## 3.1 现在的物理约束是训练后验

目前大概是：

$$
\mathcal L =
\mathcal L_{\mathrm{diff}}
+
\lambda_{\mathrm{ray}}\mathcal L_{\mathrm{ray}}
+
\lambda_{\mathrm{TV}}\mathcal L_{\mathrm{TV}}
+
...
$$

这只能让模型训练时更符合物理。
但是采样时，每一步去噪没有显式执行物理一致性。

## 3.2 改进：DPS / posterior sampling 思路

近年来 inverse problem diffusion 很常用一类方法：

$\nabla_{y_t}\log p(x\mid y_t)$

也就是在每一步去噪时，用物理测量误差的梯度修正采样方向。

你的观测模型可以写成：

$o = \mathcal A(y_0)+\eta$

其中：

* A\mathcal A**A**：物理前向算子，例如射线积分、Born 近似、频域波动算子；
* oo**o**：由 xx**x** 提取的观测；
* η\eta**η**：噪声。

假设：

$\eta\sim \mathcal N(0,\sigma_o^2I)$

则似然：

$$
p(o\mid y_0)
\propto
\exp
\left(
-\frac{1}{2\sigma_o^2}
\|\mathcal A(y_0)-o\|_2^2
\right)
$$

所以：

$$
\nabla_{y_0}\log p(o\mid y_0)
=
-\frac{1}{\sigma_o^2}
\nabla_{y_0}
\frac12
\|\mathcal A(y_0)-o\|_2^2
$$

如果 $\mathcal A$ 是线性算子：

$$
\mathcal A(y)=Ay
$$

则：

$$
\nabla_{y}
\frac12\|Ay-o\|_2^2
=
A^\top(Ay-o)
$$

所以数据一致性梯度是：

$$
g_{\mathrm{phys}}
=
-A^\top(A\hat y_0-o)
$$

采样时可以做：

$$
\hat y_0
\leftarrow
\hat y_0
-
\lambda_t
\nabla_{\hat y_0}
\mathcal L_{\mathrm{phys}}(\hat y_0,x)
$$

然后再带回 DDPM/DDIM 更新

这比只在训练损失里面加 `ray consistency` 更强。



## 4.1物理前向模型可以比 RayOperator 更强

它很快、可微、稳定。

但超声传播本身是波动问题，尤其多频信息很重要。

现在 `RayOperator.observed_from_x` 是obs = x_matrix[:, feature_index].mean(dim=1)

这相当于把频率维度平均：

$o_{tx,rx}

=

\frac1F\sum_f x_{feature,f,tx,rx}$

这会损失很多信息。


不要一开始就上全波 PINN。建议分三层：

### Level 1：现有 RayOperator

o=Ayo = Ay**o**=**A**y
最稳定。

### Level 2：Frequency-aware RayOperator

of=Afyo_f=A_fy**o**f=**A**fy
结合选频信息。

### Level 3：Born/Rytov wave operator



# 5 用更现代的 diffusion 训练形式：EDM / flow matching / consistency model

主要考虑EDM



6.自条件 Self-conditioning：低成本高收益，这种训练的小trick加上






<pre node="[object Object]"><div class="sc-cgHfjM rztuA code-block"></div></pre>
