# Handoff — 2026-05-15 21:00

## 上次进度（2026-05-11）
完成了 git 仓库初始化、权限白名单、QA_Log.md 和 /handoff 技能创建。

## 本阶段完成（2026-05-15）

### 已完成
- **阶段 1 学习完成**
  - 完整阅读 `stage1_speech_signal/mel_analyzer.py` 全部代码（~600 行）
  - `load_audio` 重构：从 `torchaudio.load` 改为 `soundfile.read`，避免 torchcodec 依赖问题
  - 参数扫描实验代码已就绪（`--sweep` 模式）：n_fft、n_mels、GL 迭代次数的正交对比
- **阶段 1 知识点考核**（6 道题，全部通过）：
  - ✅ Mel 尺度动机：人耳感知非线性压缩，低频敏感、高频不敏感
  - ⚠️ hop_length 权衡：时间分辨率与重建质量，75% 重叠是经验值
  - ✅ GL 随机相位：避免对称陷阱，但始终无法恢复原始相位（信息论局限）
  - ✅ SNR 与 n_mels：先升后 plateau，Mel 滤波本质是信息压缩
  - ✅ GL 迭代上限：~30-60 次收敛，1000 次也无法逼近原始 SNR
  - ✅ Mel bin ↔ Hz 映射：低频 bin 窄、高频 bin 宽的非线性关系

## 本阶段完成（2026-05-21）

### 已完成
- **阶段 2：HiFi-GAN 模型构建**
  - `stage2_hifi_gan/model.py`：完整实现 Generator（含 MRF）、MSD、MPD
  - `stage2_hifi_gan/train.py`：完整训练循环（Mel Loss + GAN Loss + FM Loss）
  - 单卡训练 + 多卡 DDP 支持（`--multi-gpu` 参数）
  - LJSpeech-1.1 数据手动下载到 `data/LJSpeech-1.1/`
  - 训练验证通过：单卡（GPU 2）和 4 卡 DDP（GPU 2-5）均正常跑通

- **阶段 2 理论学习（QA_Log.md）**
  - 卷积层 kernel / stride / dilation / padding / bias 机制
  - LeakyReLU 激活函数的作用
  - MRF 多感受野融合的设计动机
  - ResBlock 残差连接的原理
  - MSD（多尺度判别器）+ MPD（多周期判别器）架构详解
  - GAN 对抗训练的必要性（Mel Loss 局限性）

- **环境迁移**
  - 从 Windows 11 迁移到 Linux（RTX 3090 ×8）
  - conda 环境重建为本地 `conda_env/`
  - CLAUDE.md 命令更新为 Linux 路径

### 代码改动
| 文件 | 改动 |
|------|------|
| `stage1_speech_signal/mel_analyzer.py` | `load_audio` 改用 `sf.read`，多声道自动混缩 |
| `stage2_hifi_gan/model.py` | 新建：HiFiGANGenerator / MSD / MPD 完整实现 |
| `stage2_hifi_gan/train.py` | 新建：训练循环 + 多卡 DDP 支持 |
| `stage2_hifi_gan/infer.py` | 新建：推理脚本 |
| `stage2_hifi_gan/compare.py` | 新建：GL vs HiFi-GAN 对比 |
| `CLAUDE.md` | 更新为 Linux 环境配置和命令 |
| `.claude/CLAUDE.md` | 添加 conda 运行指令 |
| `QA_Log.md` | 新增 10 条深度学习/声码器概念问答 |
| `.gitignore` | 新增 `conda_env/`、`data/` 条目 |

### 存在问题的点
- 多声道自动混缩用 `data.mean(axis=1)` 代替了原来的 `waveform.mean(dim=0, keepdim=True)` 逻辑——混缩后的幅度略有不同
- `conda_env/` 和 `data/` 已加入 `.gitignore`，不纳入版本控制

### 代码改动
| 文件 | 改动 |
|------|------|
| `stage1_speech_signal/mel_analyzer.py` | `load_audio` 改用 `sf.read`，多声道自动混缩 |
| `conda_env/` | 未追踪的本地 conda 环境文件 |
| `stage1_speech_signal/prompt_audio.wav` | 未追踪的测试音频 |

### 存在问题的点
- 多声道自动混缩用 `data.mean(axis=1)` 代替了原来的 `waveform.mean(dim=0, keepdim=True)` 逻辑——混缩后的幅度略有不同
- `conda_env/` 和 `data/` 已加入 `.gitignore`，不纳入版本控制

## 本阶段完成（2026-06-01）

### 已完成
- **阶段 3：端到端 TTS 系统深入理解（概念学习）**
  - `stage3_vits/vits_concepts.md`：VITS 完整概念文档（VAE 框架、MAS、Normalizing Flow、Duration Predictor）
  - VITS 核心机制理解：MAS 无监督单调对齐、Posterior/Prior Encoder、ELBO 训练目标
  - 跨模型对比：VITS vs Tacotron2、CosyVoice vs VITS、FishSpeech vs VITS
  - AR vs NAR 范式深入理解：误差累积、暴露偏差、离散 token 容错

- **阶段 2 深度学习考核（9 道题，全部通过）**：
  - ✅ AdamW 优化器原理 & betas=(0.8,0.99) 的动机
  - ✅ MRF 多感受野融合的作用（消除阶梯感 + 多尺度感知）
  - ✅ MPD 周期性重排机制 vs MSD 下采样机制的区别
  - ✅ 缺少 MPD 声音糊的根本原因（MSD 下采样丢失时序分辨率）
  - ✅ detach() 切断计算图的原理（autograd 动态图 + grad_fn 链）
  - ✅ Feature Matching Loss 的动机（稠密梯度）+ 为什么 detach
  - ✅ GAN 推理时音高/语速控制不可行的架构原因
  - ✅ Mel Loss 权重 45× 的设计原理（GAN 训练的锚点）
  - ✅ 三种 Loss 的分工与平衡

- **阶段 3 深度学习考核（14 道题，全部通过）**：
  - ✅ Normalizing Flow 变换原理（Coupling Layer、可逆性、训练方式）
  - ✅ Normalizing Flow vs Flow Matching 本质区别
  - ✅ VAE 范式：为什么是"分布"、条件 VAE、ELBO 直觉
  - ✅ z 隐变量的本质：80 维无预设含义、Mel 的超集、代码里是具体向量
  - ✅ AR vs NAR：误差累积、暴露偏差、序列长度的影响
  - ✅ VITS vs Tacotron2：MAS + NAR + VAE 解决了对齐/端到端/多样性
  - ✅ CosyVoice vs VITS：Flow Matching 的架构自由 vs VAE 的架构受限
  - ✅ FishSpeech vs VITS：离散 token + LLM scaling law vs VAE 扩展瓶颈
  - ✅ FishSpeech 双 AR 为什么没有 Tacotron2 的误差累积问题
  - ✅ CosyVoice LLM 阶段的角色（AR 语义骨架 + NAR Flow Matching 细节填充）
  - ✅ VQ-GAN Decoder vs HiFi-GAN Generator 对比（Snake、串行 ResidualUnit）
  - ✅ 两阶段 pipeline 真正的问题不是"两阶段"，而是"自回归"
  - ✅ CosyVoice 不走端到端也能成功的原因（Flow Matching 多步容错）
  - ✅ 三条路线演化图：Tacotron2 → VITS / FishSpeech / CosyVoice

### 代码改动
| 文件 | 改动 |
|------|------|
| `stage2_hifi_gan/model.py` | 修正 Conv2d weight_norm 为 `nn.utils.parametrizations.weight_norm` |
| `stage2_hifi_gan/train.py` | 优化训练循环，修复 DDP 参数传递 |
| `stage3_vits/vits_concepts.md` | **新建**：VITS 核心概念详解文档（10 章） |
| `QA_Log.md` | 新增 16 条阶段 2/3 深入学习问答 |

### 当前知识地图（路线对比）

```
Tacotron2 —— AR + Attention + 两阶段
    ↓ 对齐不稳定、误差累积、信息瓶颈
VITS —— NAR + MAS + VAE + 端到端
    ├── 优势：隐空间可操作、推理快、小数据友好
    └── 局限：Flow 架构受限、扩展性差
        ↓
    ┌── FishSpeech：离散 token + AR + RVQ（LLM 路线，Scaling law）
    └── CosyVoice：语义 token + Flow Matching（连续路线，质量上限高）
```

### 存在问题的点
- 训练只跑到 21.5k steps，未完成完整的 500k steps 收敛
- 阶段 3 目前只做了概念学习，未开始代码实现

## 本阶段完成（2026-06-02）

### 已完成
- **阶段 3：VITS 完整代码实现**
  - `stage3_vits/model.py`：SynthesizerTrn（TextEncoder、PosteriorEncoder、ResidualCouplingBlock、StochasticDurationPredictor、Generator/MSD/MPD）
  - `stage3_vits/modules.py`：WN 非因果 WaveNet、AffineCouplingLayer（Flow 核心）、FFTBlock、MRF/ResBlock、ConvReluNorm
  - `stage3_vits/mas.py`：Monotonic Alignment Search（动态规划无监督对齐）
  - `stage3_vits/discriminator.py`：复用 stage2 的 MSD + MPD（import 复用，无冗余代码）
  - `stage3_vits/text_symbols.py`：88 音素 IPA 词汇表 + gruut 英文文本转音素
  - `stage3_vits/train.py`：完整训练循环（VAE KL + Duration NLL + GAN + FM + Mel Loss）
  - 单卡训练 + 多卡 DDP 支持（`--multi-gpu` 参数）
  - AMP 混合精度 + TF32 优化
  - 验证通过：GPU 200 步快速模式正常跑通（KL loss 从 19000+ → 12000+）

### 代码改动
| 文件 | 改动 |
|------|------|
| `stage3_vits/model.py` | **新建**：完整 VITS 模型（~740 行） |
| `stage3_vits/modules.py` | **新建**：WN、AffineCoupling、FFT、MRF 等共享模块 |
| `stage3_vits/mas.py` | **新建**：MAS 动态规划算法 |
| `stage3_vits/discriminator.py` | **新建**：判别器复用 bridge |
| `stage3_vits/text_symbols.py` | **新建**：IPA 音素表 + gruut g2p |
| `stage3_vits/train.py` | **新建**：完整训练循环 + DDP |

### 模型架构参数
| 组件 | 参数量 |
|------|--------|
| SynthesizerTrn (生成器) | ~22M |
| TextEncoder | 6 层 FFT (2 head self-attention) |
| PosteriorEncoder | 16 层 WN, dilation_rate=1 |
| Flow | 4 层 AffineCouplingLayer |
| DurationPredictor | Flow-based (4 coupling layers) |
| Decoder | HiFi-GAN Generator (4× upsample) |
| MSD + MPD | ~191M (判别器合集) |

## 下一步
- **阶段 3 继续：LJSpeech 完整训练**
  - 启动 500k steps 完整训练（单卡或 DDP）
  - 监控 loss 收敛（目标: KL~100, Mel~0.5, FM~0.1）
  - 推理 demo 音频音质评估
- **阶段 4 开始：音频 Tokenization（VQ-VAE / RVQ）**

## 关键上下文（继承）
- 代理配置：git push 需走 `http://127.0.0.1:7897` 代理
- 技能文件路径：`.claude/skills/handoff/SKILL.md`
- 项目 CLAUDE.md 有两级：根目录（项目信息）和 `.claude/` 目录（指令）
- 多卡训练使用 `CUDA_VISIBLE_DEVICES` 指定 GPU + `--multi-gpu` 参数
