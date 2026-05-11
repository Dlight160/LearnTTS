# TTS 系统学习路线：从语音基础到前沿扩散/LLM TTS

> 目标：学完后能理解 CosyVoice、FishSpeech 等主流模型，并能在开源模型上做优化。
> 前置条件：4年开发经验、C++/Python、线性代数+统计。
> 预计周期：6-8 个月（每天 2-3 小时）。

---

## 阶段总览

```
阶段 1: 语音信号基础        →  项目：从零构建 Mel 频谱分析器
阶段 2: 神经声码器           →  项目：训练 HiFi-GAN / 对比不同声码器
阶段 3: 端到端 TTS（传统）    →  项目：复现 VITS 并在 LJSpeech 上训练
阶段 4: 音频 Tokenization   →  项目：训练 VQ-VAE / 理解离散语音表示
阶段 5: LLM-based TTS       →  项目：精读 FishSpeech 源码并做微调实验
阶段 6: 扩散/Flow Matching  →  项目：精读 CosyVoice 源码并做风格迁移优化
```

---

## 阶段 1：语音信号处理基础 (3-4 周)

### 理论知识
- **语音产生模型**：声源-滤波器模型（source-filter model）、声道、共振峰
- **时频分析**：短时傅里叶变换（STFT）、窗口函数选择（Hann/Hamming）、时间/频率分辨率权衡
- **感知特征**：Mel 尺度、梅尔频谱（Mel-spectrogram）、MFCC、基频 F0 估计
- **音频基础**：采样率、量化、比特率、PCM 编码

### 关键理解点
- 为什么 Mel 频谱是 TTS 的"通用中间表示"
- 相位信息为何重要（以及为什么频谱映射会丢失相位）
- Griffin-Lim 算法的局限：它能从幅度谱重建信号，但质量有限，这正是神经声码器的动机

### 实践项目：构建 Mel 频谱分析器
```
输入: 任意 wav 音频
输出: Mel 频谱图可视化 + Griffin-Lim 重建音频

核心用 torchaudio/librosa 实现:
  1. 加载音频 → STFT → Mel 滤波器组 → Mel 频谱图
  2. 实现 Griffin-Lim 算法从频谱重建音频
  3. 可视化对比原始音频与重建音频
  4. 分析不同 FFT 参数、Mel bin 数对重建质量的影响
```

---

## 阶段 2：神经声码器 (4-5 周)

### 理论知识
- **声码器问题定义**：Mel 频谱 → 线性频谱 → 波形（Mel → Linear → Waveform）
- **WaveNet (2016)**：自回归声码器——开创性但太慢，理解 dilated causal convolution
- **HiFi-GAN (2020)**：GAN 声码器——多尺度判别器（MSD + MPD）、feature matching loss
- **WaveGlow (2018)**：基于流的声码器
- **BigVGAN (2022)**：大规模 GAN 声码器，周期性归纳偏置

### 架构对比

| 声码器 | 范式 | 速度 | 质量 | 训练难度 |
|--------|------|------|------|----------|
| WaveNet | AR | 极慢 | 极高 | 中 |
| WaveGlow | Flow | 快 | 高 | 高 |
| HiFi-GAN | GAN | 快 | 高 | 中 |
| BigVGAN | GAN | 快 | 极高 | 中 |

### 实践项目：训练并对比声码器
```
1. 在 LJSpeech / VCTK 上完整训练 HiFi-GAN
2. 实现多尺度判别器 (MSD) 和多周期判别器 (MPD)
3. 理解 mel-spectrogram loss + GAN loss + feature matching loss 的作用
4. 对比: Griffin-Lim vs WaveGlow vs HiFi-GAN 的重建质量
5. 指标: MOS, PESQ, MCD (Mel Cepstral Distortion), STOI
6. 分析: 哪些声音类型对声码器最有挑战（摩擦音、爆破音）
```

---

## 阶段 3：端到端 TTS（传统范式）(5-6 周)

这是关键阶段——理解 TTS 的核心问题分解。

### 理论知识

**两阶段 pipeline (2017-2020)——先理解它，再理解为什么要超越它：**

- **Tacotron 2 (2017)**：Text → Mel 频谱（seq2seq + attention + CBHG 后处理网络）
  - 自回归解码、teacher forcing
  - 注意力机制的对齐问题（漏词、重复）
- **FastSpeech (2019)**：非自回归 + duration predictor——解决速度和对齐问题
  - 长度调节器（Length Regulator）
  - 需要从自回归教师模型中提取 duration 信息
- **FastSpeech 2 (2020)**：更进一步，预测 pitch, energy, duration
  - Variance Adaptor 结构
  - 单阶段、快速、可控

**架构演化路径——这是理解的核心：**
```
Text → [Tacotron2: AR decoder + attention] → Mel → [WaveNet] → Audio
         ↓ 速度慢、attention 不稳定
Text → [FastSpeech: non-AR + duration] → Mel → [HiFi-GAN] → Audio
         ↓ 需要外部 duration 提取
Text → [FastSpeech2: + pitch/energy prediction] → Mel → [HiFi-GAN] → Audio
         ↓ 两阶段 pipeline 仍然有信息瓶颈
Text → [VITS: end-to-end, one-stage] → Audio  ← 开启新时代
```

**VITS (2021)——必须深入理解：**
- 核心创新：通过变分推断实现端到端的文本到波形
- 组件：posterior encoder + prior encoder + decoder + discriminator + flow-based duration predictor
- Monotonic Alignment Search (MAS)——无监督对齐（关键突破）
- 随机时长预测器（Stochastic Duration Predictor）——Flow-based
- 如何同时解决对齐问题 + 生成质量 + 端到端训练

**你需要在 VITS 中理解的数学直觉：**
- ELBO 的推导：为什么 VAE 目标函数允许端到端训练
- MAS 如何寻找单调对齐（动态规划寻找最优路径）
- Normalizing Flow 为何用于 duration predictor：需要的不只是 point estimate，而是 duration 的分布

### 实践项目：从零复现 VITS 核心组件
```
1. 在 LJSpeech 上复现 VITS 训练流程
2. 重点实现和理解:
   - Monotonic Alignment Search (MAS)
   - Posterior Encoder + Prior Encoder
   - Flow-based Stochastic Duration Predictor
   - HiFi-GAN decoder 作为 VITS 的 decoder
3. 消融实验:
   - 去除 flow-based duration predictor，用确定性 duration → 观察多样性下降
   - 去除 MAS，用外部 MFA 对齐 → 观察是否还能端到端训练
4. 分析: 对比 Tacotron2+HiFiGAN vs VITS 在注意力对齐上的差异
```

---

## 阶段 4：音频 Tokenization & 离散表示 (4-5 周)

这是理解 LLM-based TTS 的前提——你需要理解"音频如何变成 token"。

### 理论知识
- **为什么需要离散化**：LLM 只处理离散 token——离散化是连接音频和语言模型的关键
- **VQ-VAE 基础**：Encoder → Codebook (Vector Quantization) → Decoder
  - Straight-through estimator：前向用 argmax，反向用 straight-through 梯度
  - Codebook collapse 问题及解决方案（EMA update、codebook reset、commitment loss）
- **RVQ (Residual Vector Quantization)**：层级量化——用多个 codebook 逐步细化表示
  - 为什么需要 RVQ：单个量化层的表达能力有限，RVQ 可以指数级扩展码本空间
  - 这就是 FishSpeech 和 CosyVoice 的基础技术
- **音频 Codec 演进**：
  - SoundStream (Google, 2021)：首个端到端神经音频编解码器
  - EnCodec (Meta, 2022)：引入 RVQ + discriminator
  - SpeechTokenizer / Hubert / WavLM：语义驱动的离散化
  - DAC (Descript Audio Codec, 2023)：进一步改进的 RVQ codec

### 关键洞察
```
连续音频 ──[VQ-VAE/RVQ]──→ 离散 token 序列 ──[LLM]──→ 生成的 token 序列 ──[Decoder]──→ 波形
```
- **Codec 的质量直接决定了 LLM-based TTS 的上限**
- 不同的 codebook 层编码不同的信息（低层 → 音色/韵律，高层 → 内容/语义）
- 理解 token rate：24kHz 音频 → EnCodec 75 tokens/秒 (vs LLM 文字的 ~3 tokens/秒)

### 实践项目：训练音频 Tokenizer + 分析
```
1. 在高质量语音数据上训练一个简化版 VQ-VAE + RVQ codec
   参考: EnCodec / SoundStream 架构
2. 关键实验:
   - 对比不同 codebook 大小 (256, 512, 1024) 的影响
   - 对比不同 RVQ 层数 (4, 8, 12) 的重建质量
   - 可视化每层 codebook 的使用情况（检测 codebook collapse）
3. 分析: 用 t-SNE 可视化 token embedding，观察:
   - 不同音素是否聚集在不同 token 区域
   - 不同说话人的 token 分布差异
4. 重建质量评估: MCD, PESQ, 主观聆听
```

---

## 阶段 5：LLM-based TTS —— FishSpeech 为核心 (6-8 周)

### 理论知识

**AR + NAR 两阶段范式（VALL-E 路线）：**

- **VALL-E (Microsoft, 2023)**：第一个将 LLM 应用于 TTS 的方案
  - AR 模型生成粗粒度 token，NAR 模型细化
  - 用 EnCodec 作为 tokenizer
  - 3-second prompt → zero-shot 声音克隆
- **VALL-E 2 (2024)**：引入 Repetition Aware Sampling 和改进的编解码策略

**FishSpeech 架构（需理解的要点）：**

- **核心流程**：
  ```
  Text → [LLM (GPT-like)] → 语义 token → [Dual-AR] → 声学 token → [VQ-GAN Decoder] → Audio
                              ↗ Reference Audio → [VQ-GAN Encoder] → 音色 token
  ```

- **关键改进**：
  - 引入 RVQ 进行更精细的音频表示
  - 使用 multi-stream Transformer 预测多个 codebook
  - 音色解耦：如何分离内容、音色、韵律
  - 引入 prompt-based in-context learning

- **你需要深入理解的问题**：
  1. AR 模型如何对多个 codebook 层级建模——是层级 AR 还是交错 AR？
  2. 如何做说话人音色分离——adapter？prompt embedding？speaker conditioning？
  3. token 的采样策略——top-k/top-p/beam search 对语音质量和多样性有什么影响
  4. 如何处理中文 TTS 中特有的韵律和声调问题

### 实践项目：精读 FishSpeech 源码 + 微调实验
```
1. 环境搭建 + 完整推理流程跑通
2. 逐模块阅读:
   - 文本前端: G2P, 分词, 韵律标注
   - Tokenizer (VQ-GAN): encoder, codebook, decoder
   - LLM 骨干: 模型结构, 训练目标, 推理采样
   - 合成后端: token → waveform 的解码
3. 微调实验:
   - 在新说话人数据上做 LoRA 微调
   - 对比不同 LoRA rank / alpha 对音色相似度的影响
   - 分析: 哪些层对音色敏感，哪些层对内容敏感
4. 优化尝试:
   - 分析 tokenizer 的 codebook 利用率
   - 尝试调整采样策略改善生成质量
   - 尝试在特定文本类型（如数字、英文混合）上针对性优化
```

**关键论文（需精读）：**
- VALL-E: Neural Codec Language Models are Zero-Shot TTS
- VALL-E 2: Neural Codec Language Models are Human Parity Zero-Shot TTS
- FishSpeech 技术报告 + GitHub README/wiki

---

## 阶段 6：扩散模型 / Flow Matching TTS —— CosyVoice 为核心 (6-8 周)

### 理论知识——这部分就是你最想研究的前沿

**扩散模型基础：**
- **DDPM (2020)**：前向加噪过程、反向去噪过程
  - 理解为什么扩散模型擅长高质量生成
  - Score-based 视角：学的是 score function ∇_x log p(x)
- **DDIM**：确定性采样加速
- **Classifier-free Guidance (CFG)**：如何在 TTS 中做条件控制
  - CFG 强度：太弱 → 多样性高但质量差；太强 → 质量好但多样性低

**Flow Matching——这是 CosyVoice 的核心范式：**
- 为什么要从 DDPM 升级到 Flow Matching：
  - DDPM 需要预先定义噪声调度（noise schedule），但这不是最优的
  - Flow Matching 学习的是概率路径（probability path）——更灵活、更快的训练
  - 本质区别：DDPM 是 SDE 的离散化，Flow Matching 是 ODE
- **Conditional Flow Matching (CFM)**：
  - 学习从简单分布（噪声）到目标分布（Mel 频谱）的向量场 v(x, t)
  - ODE solver 进行推理采样
  - 与 classifier-free guidance 的结合

**CosyVoice 架构（需理解的要点）：**
- 核心范式：Flow Matching 用于 Mel 频谱生成 + 神经声码器
- 与 VALL-E 路线的根本区别：
  - VALL-E: 离散 token → LLM → 离散 token → decoder
  - CosyVoice: 文本 → Flow Matching → 连续 Mel 频谱 → 声码器
- 关键组件：
  - 文本编码器 + 说话人编码器
  - Flow Matching Transformer（UNet 或 DiT 风格的 backbone）
  - Optimal Transport Conditional Flow Matching (OT-CFM)
- CosyVoice 2 的创新：更高效的 flow matching、更好的零样本克隆

### 实践项目：精读 CosyVoice 源码 + 风格迁移优化
```
1. 环境搭建 + 完整推理流程跑通
2. 逐模块阅读:
   - 文本编码器: 如何编码音素/字符
   - Flow Matching 模型: 噪声调度、向量场预测、ODE solver
   - 说话人编码器: 如何提取说话人表示
   - 声码器: Mel 频谱 → 波形
3. Flow Matching 深度理解:
   - 从零实现一个简化的 Flow Matching 生成 2D 分布（玩具实验）
   - 然后对比 DDPM vs Flow Matching 在 Mel 频谱生成上的差异
4. 优化实验:
   - 分析 classifier-free guidance 强度对音质的影响
   - 尝试改进说话人表示（对比不同的 speaker encoder）
   - 尝试 emotion/style embedding 的注入方式
   - 分析: Flow Matching 的采样步数对质量的影响（1步→2步→4步→8步→32步）
```

**关键论文（需精读）：**
- Flow Matching for Generative Modeling (Lipman et al., 2023)
- Voicebox: Text-Guided Multilingual Universal Speech Generation (Meta, 2023)
- Matcha-TTS: A Fast TTS Architecture with Conditional Flow Matching
- CosyVoice 技术报告
- NaturalSpeech 2/3 (Microsoft) —— 扩散 TTS 的标杆工作

---

## 贯穿全程的学习资源

### 必读综述论文
1. **A Survey on Neural Speech Synthesis** (Tan et al., 2021) —— TTS 全景图
2. **A Survey on Audio Diffusion Models** —— 扩散模型在音频中的应用

### 核心工具链
- `torchaudio` / `librosa`：音频处理
- `espnet`：完整的 TTS 框架（含多种 baseline）
- `Amphion` (HKUST)：音频生成的统一框架，支持 VITS/NaturalSpeech/VALL-E
- `torchdiffeq`：ODE solver（用于 Flow Matching）

### 数据集
- LJSpeech (24h, 单说话人, 英文) → 阶段 1-3
- VCTK (44h, 109 说话人, 英文) → 阶段 2
- AISHELL-3 (85h, 218 说话人, 中文) → 阶段 5-6
- LibriTTS (585h, 多说话人, 英文) → 阶段 3-4

---

## 预期学习成果

| 阶段 | 完成后你能 |
|------|-----------|
| 1 | 理解音频的数字表示，独立实现频谱分析工具 |
| 2 | 理解 GAN 在波形生成中的作用，能训练/调试声码器 |
| 3 | 理解端到端 TTS 的核心问题（对齐、时长、韵律），能复现 VITS |
| 4 | 理解离散音频表示，能训练音频 tokenizer |
| 5 | 理解 LLM-based TTS 的全链路，能在 FishSpeech 上做微调和优化 |
| 6 | 理解 Flow Matching 原理，能在 CosyVoice 上做风格迁移等改进 |

**最终目标达成：**
- 当看 FishSpeech 源码时，你能理解：VQ-GAN encoder → RVQ tokenization → LLM AR 生成 → VQ-GAN decoder 每个环节的设计决策
- 当看 CosyVoice 源码时，你能理解：文本编码 → CFM 生成 Mel → 声码器的完整流程，以及为什么选择 Flow Matching 而不是 DDPM 或 AR
- 你具备在开源模型上做针对性优化的能力，比如改善特定说话人的克隆质量、优化特定语言的韵律等

---

*这份路线图专注于理解和研究前沿 TTS 模型，不涉及工程部署或 C++ 高性能推理。每个阶段以实践项目驱动，循序渐进地构建完整的 TTS 知识体系。*
