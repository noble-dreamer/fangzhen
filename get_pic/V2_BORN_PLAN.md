# V2 Born 近似成像目标

V2 的目标是在 V1 ray-tube 粗图基础上，引入复数相位和散射路径补偿，使正确缺陷位置在多频、多 tx-rx 叠加时相干增强。

## 1. Born 近似物理图像

健康管道中有背景波场。缺陷引起局部壁厚减薄、等效刚度变化和波速扰动。若缺陷造成的扰动相对背景不太强，可以用一阶 Born 近似：

```text
DeltaH(tx, rx, f) ≈ ∫ G(rx, p, f) q(p) G(p, tx, f) dp
```

其中：

- `DeltaH = Hd - H0` 是健康-损伤复频响差；
- `G` 是背景 Green 函数或其近似；
- `q(p)` 是缺陷散射强度，和局部厚度损失、波速扰动、散射系数相关；
- 积分在展开管壁上进行。

这说明一个像素 `p` 对 `tx-rx` 复差分的贡献相位主要由两段路径决定：

```text
L_scat(p) = L(tx, p) + L(p, rx)
```

## 2. 相位补偿反投影

如果用简化传播模型：

```text
G(a,b,f) ~ A(a,b) * exp(-i*k(f)*L(a,b))
```

那么可以对每个候选像素做相位补偿：

```text
B(p) = sum_{tx,rx,f} W * DeltaH(tx,rx,f) * exp(+i*k(f)*L_scat(p))
pic_born(p) = abs(B(p)) / coverage(p)
```

若 `p` 接近真实缺陷位置，各个频点和路径的相位更可能对齐，叠加后幅值变大；错误位置的相位更容易互相抵消。

## 3. 需要校准的关键量

Born V2 的效果取决于有效波数：

```text
k(f) = 2*pi*f / c_phase(f)
```

当前可先用常数近似：

```text
c_phase(f) ≈ 2522 m/s
```

但这只是初值。更好的做法是用健康时域或健康频域数据拟合：

```text
unwrap(angle(H0_ij(f))) ≈ -2*pi*f*L_order/c_phase + phase0
```

可以按低/中/高频段分别估计 `c_phase`。

## 4. 为什么 V2 不应完全替代 V1

导波管道问题不是简单声学单模传播。实际响应包含：

- 多模态；
- 频散；
- 端部和阵列反射；
- PZT 窗口效应；
- 缺陷散射的角度依赖；
- 壳厚变化导致的复杂模态转换。

因此 Born V2 的相位补偿可能局部有效，但不应作为唯一粗图。更稳妥的做法是把 V2 作为额外通道：

```text
pic = [
  V1 ray_relative_delta,
  V1 ray_log_amp_loss,
  V1 ray_phase_change,
  V2 born_coherent_abs,
  V2 born_real_positive,
  path_coverage,
  reliability_mask
]
```

diffusion 同时看到 V1 稳健定位先验和 V2 相干散射先验。

## 5. V2 实施阶段

建议分三步：

1. 单频 Born 通道：
   - 对每个频点单独生成 `born_coherent_abs_f`；
   - 用 label 评价哪些频点相干定位更好。

2. 多频相干叠加：
   - 使用 `select_sensitive_frequencies.py` 的频点；
   - 根据健康响应强度和 V1 label-guided 得分加权。

3. 校准 `c_phase(f)`：
   - 使用时域健康波形或健康频域相位；
   - 比较常数速度和分频段速度对成像指标的影响。

V2 完成后，可以把 Born 的隐式前向核保存下来，作为后续 diffusion 物理一致性损失的基础。
