# VITS 核心概念详解

> VITS: Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech
> (Kim et al., 2021)

---

## 1. 背景：为什么需要端到端 TTS？

### 两阶段 pipeline 的问题

在 VITS 之前，TTS 的主流做法是**两阶段**：

```
Text → [Tacotron2 / FastSpeech] → Mel 频谱 → [HiFi-GAN] → 波形
```

每个阶段独立训练，中间用 Mel 频谱连接。这条路有几个根本问题：

1. **信息瓶颈** — Mel 频谱是 80 维的压缩表示，丢掉了相位信息和部分高频细节。第一阶段产生的误差无法被第二阶段补偿（第二阶段只能"在错的 Mel 上做到最好"）
2. **独立训练 ≠ 联合最优** — Tacotron2 的训练目标（Mel L1）和 HiFi-GAN 的训练目标（波形逼真）不对齐。两个模型各自最优不代表串联起来最优
3. **中间表示手工设计** — 为什么一定是 Mel？能不能让模型自己学会最有利于波形生成的中间表示？

### VITS 的答案

VITS 把整个流程统一成一个模型：

```
Text → [VITS 联合模型] → 波形
```

所有组件**端到端联合训练**——文本编码器、对齐模块、声码器都在同一个 loss 下优化，中间表示不是手工设计的 Mel 而是模型自学的隐变量 `z`。

---

## 2. 架构总览

```
                     ┌──────────────────────────────────────┐
                     │            Posterior Encoder          │
                     │       (非因果 WaveNet, 从波形提取)      │
                     │                                      │
   真实波形 x ────────┤  q(z|x)                             │
                     │     ↓                                │
                     │   隐变量 z  (随机采样)                 │
                     │     ↓                                │
                     │   Decoder (HiFi-GAN Generator)       │
                     │     ↓                                │
                     │  重建波形 ˆx                          │
                     └──────────────────────────────────────┘

   文本 c ──────→ [Text Encoder (Transformer)] ──→ prior 分布 p(z|c)
                                                       ↑
                                                  Normalizing Flow
                                                  (增强先验表达能力)

                   Duration Predictor ←── MAS (单调对齐搜索)
                   (flow-based, 随机性)       ↑
                                           真实时长（训练时从对齐提取）
```

### 训练时数据流

1. 真实波形 `x` → Posterior Encoder → 隐变量 `z ~ q(z|x)`
2. 文本 `c` → Text Encoder → prior 分布 `p(z|c)`（经过 flow 增强）
3. MAS 找到文本帧和隐变量帧之间的**单调对齐** → 得到每个音素的时长
4. Duration Predictor 学习预测这个时长分布
5. Decoder 从 `z` 重建波形 `ˆx`
6. Discriminator（MSD + MPD）判断真假

### 推理时数据流

1. 文本 `c` → Text Encoder → prior 分布
2. Duration Predictor 预测每个音素的时长 → 展开对齐
3. 从 prior 采样 `z`
4. Decoder 从 `z` 生成波形

注意推理时**不需要 Posterior Encoder**——它只在训练时用（VAE 的标准做法）。

---

## 3. VAE 框架：VITS 为什么是 conditional VAE

### 核心直觉

VITS 本质上在做一件事：**给定文本 c，生成波形 x**。但文本 c 不包含所有信息——比如说话人的语气、语速、音色细节。这些"额外信息"用一个**隐变量 z** 来表示。

- `p(z|c)` — 给定文本，隐变量的**先验分布**（Text Encoder 输出）
- `q(z|x)` — 给定真实波形，隐变量的**后验分布**（Posterior Encoder 输出）
- 推理时：先从 `p(z|c)` 采样 z，再通过 Decoder 生成波形

训练的目标是让**先验和后验尽量接近**——这样推理时（没有 posterior）只靠先验也能采样出好的 z。

### ELBO 的直觉

VITS 的 loss 可以理解为三部分：

```
Loss = 重建 Loss (波形要像) 
     + KL 散度 (先验和后验要接近) 
     + Duration Loss (时长要准)
```

**重建 Loss** — Decoder 输出的波形逼真吗？这包括了 Mel Loss + GAN Loss + FM Loss（和阶段 2 一模一样）

**KL 散度** — `KL(q(z|x) || p(z|c))`。如果先验和后验差距很大，KL 项会惩罚模型。这迫使 Text Encoder 输出的先验分布尽可能包含生成波形所需的信息。

**为什么叫 conditional VAE？** — 因为先验和后验都**以输入为条件**：`p(z|c)` 以文本为条件，`q(z|x)` 以波形为条件。标准的 VAE 先验是 `p(z)`（标准正态），而 VITS 的 `p(z|c)` 包含了文本信息，更有信息量。

---

## 4. Posterior Encoder：从波形提取隐变量

### 作用

Posterior Encoder 的输入是真实波形 `x`，输出是隐变量 `z` 的分布参数（均值 µ 和方差 σ）。

```
波形 x → [非因果 WaveNet] → µ, σ → 采样 z ~ N(µ, σ)
```

### 结构

Posterior Encoder 是一个**非因果 WaveNet**：

- 多层 dilated convolution（扩张卷积）
- 每层有 skip connection
- **非因果**意味着每层的输出可以依赖未来信息（不像 WaveNet 原版那样是自回归的，不需要 mask）

### 为什么需要它？

- **训练时**：Posterior Encoder 从真实波形提取 `z`，告诉 Decoder "好波形对应的隐变量长什么样"
- **推理时**：Posterior Encoder 被丢弃。生成器从 Text Encoder 的 prior 采样 `z`

Posterior Encoder 的存在让训练更稳定——因为 `z` 是从真实波形提取的，Decoder 一开始就有足够好的输入信号，不需要从头摸索。

---

## 5. Text Encoder（Prior Encoder）：从文本到先验分布

### 作用

Text Encoder 把音素序列映射为**先验分布** `p(z|c)` 的参数。

```
音素序列 c → [Text Encoder (Transformer)] → µ, σ (每个音素对应的分布)
```

### 结构

- 多层 Transformer（或 FFN + 卷积的混合结构）
- 输入是音素 embedding + 位置编码
- 输出是每个音素对应的 µ 和 σ

### 对齐的问题

Text Encoder 输出的 µ/σ 序列长度 = 音素个数（比如 "hello" 有 4 个音素），但隐变量 z 的序列长度 = 波形帧数（几十到几百帧）。需要把两者**对齐**——这就是 MAS 要做的事。

---

## 6. Monotonic Alignment Search (MAS)

这是 VITS 最关键的创新，理解它你就理解了 VITS 的核心。

### 问题定义

```
文本:  [h] [e] [l] [o]        ← 4 个音素, 每个音素产生一个 µ/σ
波形帧: f1 f2 f3 f4 f5 ... fN  ← N 帧隐变量 z

需要找到: 每个帧对应哪个音素？
          每个音素持续了多少帧？
```

约束条件：**单调性**——发音顺序不能颠倒，"hello" 不能先发 "o" 再发 "h"。

### 对齐是什么

对齐是一个映射 A：把文本位置映射到波形帧位置。

```
音素:    h     e     l     o
时长:    3     2     5     4
帧:     f1-f3 f4-f5 f6-f10 f11-f14
```

### MAS 的具体做法

MAS 的目标：**找到最可能的单调对齐**。

衡量标准是**概率**——给定对齐 A，隐变量 z 在音素 i 对应的帧上的概率是：

```
log p(z_frames_of_i | µ_i, σ_i)
```

MAS 用**动态规划**找到使总概率最大的对齐：

```
dp[i][j] = 前 i 个音素、处理到第 j 帧时的最大对数概率
dp[i][j] = max(dp[i][j-1], dp[i-1][j-1]) + log p(z_j | µ_i, σ_i)
```

递推公式的含义：
- `dp[i][j-1]` — 音素 i 继续（当前音素多占一帧）
- `dp[i-1][j-1]` — 切换到下一个音素

这保证了对齐是**单调**的（只能前进或停留，不能后退）。

### 为什么不需要标注数据？

MAS 是对齐**无需标注数据**（无监督），因为它只需要：
1. 文本端给出的先验分布 `N(µ_i, σ_i)`
2. 波形端提取的隐变量 z
3. 假设对齐是单调的

三者都来自模型本身的输出，不需要外部对齐工具（如 MFA）。

### 一个关键的理解

MAS 本质上是在做：

> "波形的这一帧最像哪个音素对应的分布，就把它对齐到哪个音素"

这是 VITS 能端到端训练的根本原因——对齐是训练过程中**动态计算**的，不需要外部标注。

---

## 7. Normalizing Flow：为什么 prior 需要增强

### 问题

Text Encoder 输出的先验分布 `p(z|c) = N(µ, σ)` 是**高斯分布**。但真实隐变量 z 的分布可能非常复杂——它不是高斯分布的。

如果我们直接用原始高斯先验去匹配后验 `q(z|x)`，KL 散度会很大，导致：
- 先验包含的信息不足 → 生成质量差
- 后验被迫承载太多信息 → 训练不稳定

### Flow 的解决方案

VITS 在 Text Encoder 之后加了一串 **Normalizing Flow** `f`，把简单的高斯分布变换成复杂分布：

```
文本 c → [Text Encoder] → N(µ, σ) → [Flow f] → 复杂分布 p'(z|c)
```

推理时采样路径：
```
z' ~ 复杂分布 → [Flow f^{-1}] → z → Decoder → 波形
```

### Normalizing Flow 的核心思想

一个可逆变换 `f`，把简单分布（高斯）映射为复杂分布（真实隐变量的分布）。

关键性质：
- **可逆**：`f` 必须有逆变换 `f^{-1}`（训练时正向，推理时反向）
- **可微**：可以端到端训练
- **概率显式可算**：通过 change of variable formula（变量变换公式）

### 为什么不用更复杂的编码器？

一个思路是增加 Text Encoder 的复杂度让它直接输出复杂分布。但这样编码器会非常庞大，而 Flow 用一个轻量级的可逆变换网络就能达到同样的表达能力——代价是推理时多一次逆变换计算。

---

## 8. Stochastic Duration Predictor：为什么时长需要随机性

### 确定性时长预测器的问题

FastSpeech 2 的 duration predictor 输出一个**确定的时长**（每个音素持续多少帧）。但人的发音时长本身就有**随机性**——同一个音素在不同语境下时长不同，甚至同一句话每次说时长都不一样。

确定性模型的问题：
- 每次生成同一句话的语速、节奏完全一样 → 缺乏自然感
- 无法捕捉"音素时长分布"（有的音素时长方差大，有的方差小）

### 随机时长预测器

VITS 的 Duration Predictor 输出的是一个**分布**而不是一个值：

```
音素特征 → [Flow-based Duration Predictor] → duration 的分布
```

训练时：从分布中采样 duration，计算 duration loss
推理时：从分布中采样 duration（或取均值），来决定对齐

### 为什么用 Flow？

因为需要捕捉 duration 的任意分布——高斯分布不够灵活（时长分布常常是偏态的）。Flow 在这里的任务和 prior 增强一样：把简单分布映射为复杂分布。

### 实际效果

随机时长预测器让 VITS 能生成**同一句话不同节奏的版本**——这是一个重要的能力，确定性模型做不到。

---

## 9. Decoder (HiFi-GAN Generator)

Decoder 直接就是阶段 2 学过的 HiFi-GAN Generator：

```
隐变量 z → [Conv1d × N + MRF × N + ConvTranspose1d × 4 ...] → 波形
```

### 和独立 HiFi-GAN 的区别

| | 独立 HiFi-GAN | VITS 中的 Decoder |
|---|---|---|
| 输入 | Mel 频谱 (80 维) | 隐变量 z (通常也是 80 维) |
| 训练 | 单独训练 | 和整个 VITS 联合训练 |
| 梯度来源 | Mel Loss + GAN Loss + FM Loss | 同上 + KL loss 回传 |
| 输入分布 | 固定 Mel（来自真实音频） | 训练时来自 posterior，推理时来自 prior |

Mel 频谱和隐变量 z 的维度可以相同，但含义不同——Mel 是手工设计的频谱特征，z 是模型自学的潜在表示。

---

## 10. 训练目标总结

### 总 Loss

```
L_total = L_vae + L_dur + L_adv + L_fm + L_mel
```

#### L_vae — VAE 损失

```
L_vae = 重建损失 (Mel L1) + KL(q(z|x) || p(z|c))
```

- 重建损失：Decoder 输出波形 → Mel → 和真实 Mel 算 L1
- KL 散度：让先验（文本→分布）和后验（波形→分布）接近

#### L_dur — Duration 损失

Duration Predictor 预测的 duration 分布和 MAS 提取的"真实"时长之间的 loss（负对数似然）。

#### L_adv — GAN 对抗损失

MSD + MPD 的判别器损失（和阶段 2 完全一样）。

#### L_fm — Feature Matching Loss

判别器中间特征的 L1 损失（和阶段 2 完全一样）。

#### L_mel — Mel Loss

生成波形和真实波形的 Mel 频谱 L1 损失（和阶段 2 完全一样）。

### 联合训练的意义

所有组件**同时优化**：

- Text Encoder 收到的梯度来自：KL loss（要产生产生接近后验的先验）+ duration loss（要对齐要准）
- Duration Predictor 收到的梯度来自：duration loss
- Posterior Encoder 收到的梯度来自：KL loss + 重建 loss
- Decoder 收到的梯度来自：重建 loss + GAN loss + FM loss
- Discriminator 收到的梯度来自：GAN loss

这意味着**文本编码器能感知到声码器的行为**，反过来声码器也能适应文本编码器的输出分布——这是两阶段 pipeline 做不到的。

---

## 概览图

```
训练时:
波形 x ──→ Posterior Encoder ──→ z ~ q(z|x) ──→ Decoder ──→ ˆx ──→ Mel Loss + GAN
                                 ↕ KL                              ↑
文本 c ──→ Text Encoder ──→ Flow ──→ p(z|c)                      MSD + MPD
                                 ↓
                              MAS (对齐 z 和文本)
                                 ↓
                              Duration ←─ Duration Predictor

推理时:
文本 c ──→ Text Encoder ──→ Flow ──→ p(z|c) ──→ 采样 z ──→ Decoder ──→ 波形
                                 ↓
                           Duration Predictor
                           (预测每个音素时长)
```

## 回答"为什么要超越两阶段 pipeline"的答案

| 问题 | 两阶段 (Tacotron 2 + HiFi-GAN) | VITS (端到端) |
|------|------|------|
| 中间表示 | Mel 频谱（固定设计） | 隐变量 z（模型自学） |
| 训练方式 | 独立训练 | 联合训练 |
| 文本→对齐 | Attention（可能漏词/重复） | MAS（单调保证，无监督） |
| 时长 | 确定性或 external predictor | 随机性 flow-based |
| 推理速度 | 慢（两步） | 快（一步） |
| 自然度 | 好 | 更好（同参数量下 MOS 更高） |

VITS 证明了**联合训练 + 隐变量表示 + 单调对齐**的组合足以取代了两阶段 pipeline，是 TTS 迈向更自然合成的关键一步。