# QA Log

## 2026-06-03（第三阶段·训练性能诊断）

### Q: stage3 训练启动后 GPU 计算资源没跑满,一直在 0 和 100 之间跳,100 的时间很少,日志也没有 step 更新,为什么？

A: **数据加载瓶颈——GPU 在饿肚子等 CPU 预处理。** 典型症状就是 GPU 利用率 0/100 抖动(算完一个 batch 几十毫秒,然后长时间空等下一批数据)。

**诊断方法(逐步定位,而非猜)**：
1. `nvidia-smi` 看 GPU 利用率 → 全 0%,但显存占着(进程还在)。
2. `ps aux` 看进程 CPU% → 8 个 DDP 进程每个 877–900%(各吃 ~9 核),说明卡在 CPU 计算,不是死锁。
3. 拆解 `__getitem__` 各环节耗时 → **g2p 1211ms/句,io 5ms,mel 1ms**。真凶是 gruut 文本→IPA 音素转换。

**根因链条**：
- gruut g2p 确定性但极慢(~1.2s/真实 LJSpeech 句),却每样本每 epoch 重算。
- DDP 下 `num_workers=0`(原为绕开 mp.spawn + espeak fork 死锁),g2p 全堵在训练主进程同步执行 → batch=16 每步光 g2p ~19s。
- 次要:`torch.get_num_threads()=128` 默认,8 进程 ×128 线程抢 256 核,过度订阅。

**修复(按收益排序)**：
1. **预计算 + 缓存音素**(核心):文本→音素是确定性的,多进程跑一次存 `phonemes_cache.pt`,训练时查表。数据加载 1253ms → 3.9ms/样本,**加速 322×**。
2. 缓存后 worker 不再碰 gruut,死锁风险消失 → DDP 也开 `num_workers=4`。
3. `torch.set_num_threads(4)` + worker 单线程,防过度订阅。

**通用教训**:GPU 利用率抖动 + step 不推进,先怀疑数据管道,用 `ps` 的 CPU% 和分环节计时定位,别盲改超参或模型。

## 2026-06-01（第二阶段）

### Q: Normalizing Flow 具体如何变换？参数是训练出来的吗？
A: 是训练出来的。Flow 的每一层 Coupling Layer 里有一个小型 NN（可训练参数）输出 scale 和 shift。

**Affine Coupling Layer 具体计算**：
1. 把 z 沿通道切成两半：z₁, z₂
2. z₁ 直接复制到输出（不变）
3. z₁ 输入 NN → 输出 scale, shift
4. z₂' = z₂ × exp(scale) + shift
5. 拼接输出：concat(z₁, z₂')

**可逆性关键**：z₁ 没变，所以反向时同一个 NN 算出的 scale/shift 和正向一样，只需算术逆：(z₂' - shift) × exp(-scale)。NN 本身不需要可逆。

多层交替切哪一半 → 信息充分混合，表达能力累积。

### Q: Normalizing Flow（VITS）和 Flow Matching（CosyVoice）有什么区别？
A: 本质完全不同的两样东西。

| | Normalizing Flow | Flow Matching |
|---|---|---|
| **本质** | 可逆神经网络 | 微分方程（速度场） |
| **生成方式** | 一步映射 | 多步 ODE 积分（16-32 步） |
| **架构约束** | 必须可逆（Coupling Layer） | 无约束（Transformer/UNet 都行） |
| **概率计算** | 精确 | 近似 |
| **训练方式** | 最大似然 / KL | MSE 回归速度场 |

Flow Matching 没有可逆约束——这是它相比 Normalizing Flow 最大的优势：可以用强大架构，不受 Coupling Layer 限制。

### Q: Flow Matching 直接生成频谱，那隐变量在哪？
A: CosyVoice **没有隐变量 z** 这个概念。整个生成过程就是从噪声到 Mel 频谱的轨迹。

信息编码方式变了：
- VITS：文本外信息显式编码到 z 的分布 p(z|c) 里
- CosyVoice：文本外信息隐含在生成路径和条件注入（speaker embedding）里

随机性来源也不同：VITS 从 p(z|c) 采样不同 z；CosyVoice 从不同初始噪声 x₀ 出发。

### Q: VITS 和 CosyVoice 各自的优缺点？

**VITS（Normalizing Flow + VAE）**：
- 优点：推理快（一步）、隐空间可操作（声音克隆/编辑 z）、KL loss 天然防过拟合、小数据友好
- 缺点：架构受限（Coupling Layer 可逆约束）、扩展性差（大数据提升不明显）、文本条件注入间接

**CosyVoice（Flow Matching）**：
- 优点：架构自由（Transformer/DiT）、扩展性好（数据越大优势越大）、生成质量上限高、条件注入灵活、社区工具可复用（CFG、ODE solver）
- 缺点：推理慢（16-32 步）、无显式隐空间（控制靠条件注入）、训练更贵、概率计算近似

核心权衡：VITS = 可控性强 + 快，但上限受限；CosyVoice = 质量上限高 + 可扩展，但计算贵。

### Q: VITS 为什么不扩展 Mel bin 数（比如 80→128）？
A: 因为 VITS 的信息瓶颈是 **KL Loss**，不是 Mel bin 数。

在两阶段 pipeline 里，Mel 是唯一的中间表示，加 bin = 加信息。VITS 的 z 虽然也是 80 维，但每维没有固定频率含义——模型自由分配维度编码不同类型的信息（频谱、音色、韵律等）。KL loss 动态调节每维的信息量，不需要靠加维度来提升质量。

论文消融实验也验证了：80 vs 128 没有显著差异。

### Q: 为什么 VITS 要用隐变量 z 而不是保留 Mel 频谱（像 CosyVoice 一样）？
A: 因为 VITS 是 **VAE 框架**。VAE 需要先验 p(z|c) 和后验 q(z|x) 的 KL 散度来训练——没有 z 这个隐变量，KL 散度无法计算。z 是 VAE 框架的必要组件。

Posterior Encoder 在训练时给 Decoder 开了一个"作弊通道"——从真实波形提取 z，告诉 Decoder"好波形对应的隐变量长什么样"。推理时扔掉 Posterior Encoder，只靠先验也能生成。

如果没有 z，VITS 无法端到端训练——梯度从 Decoder 到 Text Encoder 跨度太大，GAN 的不稳定梯度传不回去。CosyVoice 不需要 z，因为它用 Flow Matching 的 MSE 回归替代了 VAE 的 KL + GAN，训练信号稳定得多。

### Q: VAE 范式是什么？
A: 变分自编码器（Variational Autoencoder）。

**普通自编码器**：Encoder 输入 x → 固定向量 z → Decoder 重建 ˆx。问题是 z 固定，无法生成新样本。

**VAE**：Encoder 输出 N(µ, σ) → 采样 z ~ N(µ, σ) → Decoder 重建 ˆx。训练目标 = 重建 Loss + KL(N(µ, σ) || N(0,1))。

"变分"指用简单分布 q(z|x) 去逼近真实后验 p(z|x)，最小化两者的差距。

**条件 VAE（VITS）**：先验不是标准正态而是以文本 c 为条件 p(z|c) = TextEncoder(c)。推理时从 p(z|c) 采样 z，生成包含正确文本内容但细节多样的波形。

### Q: 为什么 Flow Matching 不直接生成波形，而是生成 Mel 频谱？
A: 技术上可以，但工程上不划算：

1. **维度差太多** — 1 秒音频：Mel(80,86) vs 波形(1,22050)。波形序列长 250倍，Transformer 自注意力 O(T²) 扛不住
2. **信息密度低** — 波形每个采样点是单值，Mel 每帧包含完整频谱分布。同样信息量波形帧数多得多
3. **不能复用 HiFi-GAN** — Mel→波形已被验证接近无损，没必要重复造
4. **波形 MSE 不反映感知距离** — 相位偏移导致波形 MSE 大但听感一样，反过来轻微时间偏移 MSE 小但听感不同。Mel 空间的 MSE 和人耳感知更相关

### Q: 隐变量 z 是分布怎么理解？在代码里不就是一组向量吗？
A: 你说得对，z 在代码里就是一组向量（具体的张量）。

"z 是分布"是数学描述，完整说法是：**z 由一组从分布中采样的向量来定义，并通过 KL loss 约束这些样本的行为来"塑造"这个分布**。

具体对应：
- mu, sigma = Encoder(x) → 定义了"所有合格 z 的集合"（分布）
- z = mu + sigma × epsilon → 从这个集合中取一个具体样本（向量）
- KL loss = f(mu, sigma) → 约束这个集合的形状靠近先验

在代码世界只有具体值——分布是 mu 和 sigma 共同编码的数学含义，加上 KL loss 的训练约束。

### Q: 为什么说两阶段 pipeline 不行？CosyVoice 看起来也是两阶段啊
A: 你说得对，"两阶段"不是问题——CosyVoice 和 Tacotron2 都是两阶段（第一阶段生成 → 第二阶段声码器）。

真正问题不是"几个阶段"，而是 **自回归 vs 非自回归**：

**Tacotron2** 自回归逐帧预测 Mel（86帧/秒），误差累积 + 暴露偏差 + 注意力对齐不稳定。
**CosyVoice** 的 Flow Matching 非自回归一次性生成所有 Mel 帧，没有误差累积，训练和推理条件一致。

所以更好的说法是：Tacotron2 不是输在"两阶段"，是输在"自回归的架构缺陷"。

### Q: 什么是自回归，什么是非自回归？
A: **自回归（AR）** = "用已经生成的，决定下一步生成什么"。Tacotron2 逐帧预测 Mel，每帧依赖前一帧的预测结果。慢、误差累积、暴露偏差。

**非自回归（NAR）** = "一次性生成所有帧"，不依赖之前的生成结果。VITS / CosyVoice / FastSpeech。快、无误差累积、训练和推理条件一致。

非自回归的代价：需要额外的 duration predictor 告诉模型每个音素占多少帧，且容易"平均化"（用 Flow / 多步积分来缓解）。

### Q: CosyVoice 怎么解决时间对齐问题？
A: CosyVoice 1 用 **外部工具 MFA（Montreal Forced Aligner）** 做对齐 + Duration Predictor 预测时长。和 VITS 的 MAS（无监督）走了不同路线。

流程：MFA 对齐 → 提取每个音素时长 → 训练 Duration Predictor → 展开帧级别文本条件 → Flow Matching 以展开条件生成 Mel。

为什么 CosyVoice 可以接受外部对齐？因为 Flow Matching 多步积分有容错性——每一步可以修正之前的不完美，而 VITS 是一步映射，对齐必须准。

### Q: CosyVoice 的 LLM 阶段作用是什么？LLM 也是自回归的吧？
A: 是的，LLM 是自回归的。CosyVoice 是 **AR + NAR 混合架构**：

```
LLM（AR）→ 语义 token → Flow Matching（NAR）→ Mel → HiFi-GAN → 波形
```

LLM 负责：语义理解、长程韵律建模（语调走向、断句）、说话人风格注入。输出粗粒度的"语义骨架"。

Flow Matching 负责：在 LLM 定好的"韵律骨架"上填充频谱细节。

AR + NAR 混合的原因：AR 适合建模序列依赖但慢且误差累积，NAR 适合高质量细节生成但需要外部对齐。两者互补。

### Q: FishSpeech 的双 AR 结构怎么解决 Tacotron2 的问题？
A: FishSpeech 也是 AR，但 AR 的粒度不同。核心区别：

1. **离散 token vs 连续 Mel** — Tacotron2 预测连续 Mel 值（飘一点就偏），FishSpeech 预测离散 token ID（分类问题，要么对要么错）
2. **序列长度不同** — Tacotron2 ~86帧/秒，FishSpeech ~10-20 token/秒（短 4-8倍），误差累积机会少
3. **离散 embedding 有语义结构** — 即使预测错了相近 token，embedding 空间里接近，后续不会崩
4. **RVQ 分层容错** — Level-1 错了，Level-2/3/4 可以在细粒度上弥补。Tacotron2 没有分层机制

一句话：Tacotron2 的问题不是 AR 本身，而是连续值 AR 的高速误差累积。

### Q: FishSpeech 中对应 Flow Matching 的阶段是什么？
A: **没有对应阶段**。FishSpeech 和 CosyVoice 走了完全不同的路线：

- CosyVoice：离散 token → **Flow Matching（回到连续空间）** → Mel → 声码器
- FishSpeech：离散 token → **更多离散 token（全程离散）** → VQ-GAN Decoder → 波形

如果非要找功能对应，VQ-GAN Decoder ≈ Flow Matching + HiFi-GAN 的组合，但过程完全不同——Flow Matching 是多步积分，VQ-GAN Decoder 是一次前向。

### Q: FishSpeech 的 VQ-GAN Decoder 和 HiFi-GAN Generator 对比？
A: **范式相同，架构不同**。

**相同**：纯卷积上采样（ConvTranspose1d + 多重空洞卷积精修）+ GAN 对抗训练 + weight norm + MPD（部分）

**不同**：
- 激活函数：HiFi-GAN 用 LeakyReLU，FishSpeech 用 **Snake**（周期性偏置，更适合音频）
- 残差块结构：HiFi-GAN 用并行 MRF（3路求和），FishSpeech 用**串行 ResidualUnit**（3层 dilation 递增 1→3→9）
- 通道数：HiFi-GAN 512→32，FishSpeech **1536→96**（大得多，~54M 参数 vs ~14M）
- 判别器：HiFi-GAN 用 MSD+MPD，FishSpeech 用 **MRD+MPD**（多分辨率频谱判别器代替 MSD）
- GAN loss：HiFi-GAN 用 LSGAN(MSE)，FishSpeech 用 **HingeGAN**

FishSpeech 站在 HiFi-GAN 肩膀上，吸收了 BigVGAN 的 Snake 激活函数和 DAC 的串行残差块。

### Q: 总结：VITS vs Tacotron2、CosyVoice vs VITS、FishSpeech vs VITS 各解决了什么问题？

**VITS 相比 Tacotron2**：
- 对齐不稳定 → MAS 保证单调对齐
- 两阶段信息瓶颈 → 端到端联合训练
- AR 误差累积 → NAR 一次生成
- 多样性差 → VAE + 随机时长预测

**CosyVoice 相比 VITS**：
- Flow 架构受限 → Flow Matching 无架构约束（可用 Transformer）
- 扩展性差 → Flow Matching 数据越大提升越明显
- 生成质量上限 → 多步积分每步修正
- 条件注入间接 → 每一步直接注入文本/说话人条件

**FishSpeech 相比 VITS**：
- 无法享受 LLM scaling law → 离散 token + AR，统一 LLM 范式
- VAE + Flow 架构限制了模型规模 → 纯 LLM 架构，随数据/算力稳定提升
- 独立声学框架无法复用社区工具 → 复用 LLM 基础设施（KV cache、投机解码等）

### Q: 隐变量 z 为什么是分布而不是一组固定向量？
A: 文本到语音是一对多映射——同一句话可以有无数种发音方式（不同语气、语速、音色）。
如果 z 是固定向量，模型只能学到"平均值"，输出固定不变。
分布 `z ~ N(µ, σ)` 编码了"可能性空间"，σ 大的维度对应文本没约束的自由度（音色细节），σ 小的维度对应必须遵守的文本约束（音素内容）。
采样得到的是"在合理范围内的一个可行实例"，不是随机噪声。

### Q: z 隐变量的复杂分布到底是什么？
A: z 是一个 80 维向量（和 Mel 同维度），但每个维度没有预设的物理含义——模型自己决定怎么用这 80 维空间来编码文本之外的语音信息（音色、韵律、语气等）。

"分布"的意思是 z 的取值在空间中有高低概率区域。因为影响语音的因素不独立（如"疑问语气=高音+慢速"和"确认语气=低音+中速"是分开的两团），真实分布不是单个高斯能描述的——需要 Normalizing Flow 把高斯揉成任意形状去匹配。

### Q: 我理解 Mel 也属于一种分布，但被人为定义了。z 是模型自学的，VITS 就是训练另一种没人定义的 Mel 谱，对吗？
A: 核心直觉完全正确。微调一处：Mel 是确定性变换（固定公式算出来的值），不是"分布"——同样的音频进来 Mel 永远一样，没有概率概念。但从功能等价角度看，VITS 就是在做"让模型自己学一个更适合生成波形的中间表示，而不是用人工设计的 Mel"。

z 是 Mel 的超集——包含 Mel 信息（因为 Mel Loss 还在约束 Decoder 输出波形的 Mel），又额外携带 Mel 丢弃的信息（相位、音色细节等）。

## 2026-05-28

### Q: HiFi-GAN 的优化器为什么选 AdamW？betas=(0.8, 0.99) 的含义？
A: AdamW = Adam + 解耦权重衰减。原始 Adam 把 weight decay 混在梯度里，再被自适应 lr 缩放，导致正则化强度跟 lr 绑定。AdamW 把 weight decay 移到参数更新步骤，在自适应缩放之后才减，两者解耦。

betas=(β₁, β₂) 控制两个指数移动平均：
- β₁=0.8（默认 0.9）：动量衰减更快，对近期梯度更敏感。GAN 训练中生成器和判别器交替更新，梯度方向频繁反转，太慢的动量会拖着错误方向不放。
- β₂=0.99（默认 0.999）：自适应 lr 的衰减范围，用 ~100 步的梯度方差归一化。

生成器和判别器各用一个优化器，因为它们是交替更新，各走各的计算图。

### Q: HiFi-GAN 论文中最终 loss 是多少？
A: 论文不报告 loss 值（loss 跟 lambda 权重、batch size、segment length 直接相关），只报告 MOS 主观评分。经验参考（LJSpeech, 500k steps 收敛后）：
- Mel L1: 0.3 ~ 0.6（最重要）
- Feature Matching: 0.05 ~ 0.12
- GAN Loss: 1.0 ~ 3.0
- D Loss: 1.5 ~ 2.5（稳定说明没被生成器压制）
- G Total: 15 ~ 40（Mel loss ×45 占大头）

### Q: MRF 的作用？为什么反卷积后还要加残差块？
A: MRF = Multi-Receptive Field Fusion。两个作用：

1. **多感受野融合**：3 路 ResBlock 用不同 kernel（3/5/7）和 dilation（[1,3,5]），小 dilation 捕捉精细局部纹理（谐波细节），大 dilation 捕捉长程依赖（基频周期性、音高轮廓）。让模型同时兼顾精细和全局。

2. **消除上采样阶梯感**：反卷积 stride 不整除 kernel 时会产生 checkerboard artifacts，MRF 的多分支融合能缓解网格伪影。

ResBlock 的残差连接让网络只学"调整量"而非完整值，降低学习难度。

### Q: MPD 和 MSD 各捕捉音频的什么维度？为什么两个都要？
A: **MPD（Multi-Period Discriminator）** — 捕捉时间轴上的周期性模式。把波形按周期 p 重排为 2D（reshape + permute），用 Conv2d 判断周期内部结构是否自然。periods=[2,3,5,7,11] 覆盖不同谐波周期。

**MSD（Multi-Scale Discriminator）** — 捕捉不同时间尺度的整体音质。用 AvgPool1d 下采样到原始/1/2/1/4 三个尺度，用 Conv1d 判断：原始尺度看高频细节，1/2 看中频质感，1/4 看整体包络。

只用 MPD → 周期对了但每个周期内部波形粗糙（音高对但沙哑）
只用 MSD → 整体平滑但周期边界模糊（糊，像蒙布）
两个制衡 → 生成器同时做到"周期性正确"和"局部波形干净"

### Q: 缺少 MPD 为什么声音会糊？
A: MSD 的下采样（AvgPool1d）直接丢失时序分辨率。4× 下采样后相邻 4 个采样点融合成 1 个值，高频细节全丢。如果波形周期大致对了但每个周期内部偏移了几个采样点，MSD 下采样后分辨不出。

MPD 没有下采样，只是重排。Conv2d 在 period 轴上检查每个周期内精确的样本间关系。时序偏移了半个采样点，MPD 展开的 2D 图里会出现锯齿状边缘。

没有 MPD → 生成器只知道"大方向对就行"，±1 采样的偏差不会被惩罚。多个周期叠加后谐波相位乱了，高频细节模糊。

### Q: detach() 的作用是什么？计算图在 autograd 里怎么工作的？
A: `detach()` 切断计算图——新建一个张量共享底层数据但 `.grad_fn = None`，反向传播到它就终止。

**前向传播时，边执行边搭图**：每个张量操作注册一个 `Function` 节点，记录输入/输出 id 和反向所需信息。fake_wav 的 `.grad_fn` 指针指向 `TanhBackward`，链回 mel。

**backward 时，沿着 grad_fn 链反向拓扑遍历**：从 loss 出发，逐节点链式法则算梯度，累加到参数 `.grad`。

判别器训练时 detach fake_wav：`loss_d.backward()` 到 fake_wav_detached 发现 `grad_fn=None`，路径终结，不给生成器算梯度。如果不 detach，梯度会穿过 fake_wav 回传到生成器 — 浪费算力且可能污染下一轮更新。

生成器训练时不 detach：需要梯度穿过判别器回到生成器参数，所以保留完整计算图。

### Q: Feature Matching Loss 的作用？为什么对 fake_feats 做 detach？
A: **FM Loss 的作用**：判别器只输出一个 final logit，信号太稀疏。FM Loss 取判别器各中间层对真实/生成音频的特征输出，算 L1 loss —— 把单一"真/假"信号分解成几十个维度的具体误差，训练早期尤其有用。相当于告诉生成器"你在判别器第 3 层的输出跟真实音频差在这"。

三个 loss 的分工：
- **GAN Loss（1×）**：逼真度，让生成器骗过判别器
- **Mel Loss（45×）**：内容保真度，约束频谱包络
- **FM Loss（10×）**：加速收敛，提供稠密梯度信号

**detach 的原因**：生成器应该去匹配判别器当前的特征提取方式，而不是试图改变判别器的中间特征表示来降低 loss。判别器参数已冻结（`requires_grad=False`），如果不 detach，生成器在做一件"不可能的事"（改变一个固定函数的输出），梯度方向可能扭曲，浪费优化容量。

### Q: 推理时如何控制音高和语速？
A: 纯 HiFi-GAN 架构（卷积声码器）的控制能力非常有限。

**语速（播放变速）** — 可近似：对 Mel 频谱做时间维线性插值（拉伸放慢、缩短加快），生成器输出相应变长/变短的波形。这只是"播放变速"，不是"改变发音速度"。

**真实发音速度控制** — 不可：需要时长预测器（duration predictor），VITS / FastSpeech 才有。

**音高控制** — 不可：Mel 频谱的分布模式隐含了基频信息，直接修改会破坏结构。需要外接 F0 预测器提取基频，修改后 merge 回 Mel，再重新训练支持 F0 条件的生成器。

HiFi-GAN 的定位是神经声码器（Mel → Waveform），控制不在设计范围内。

### Q: 为什么 Mel Loss 权重设得这么大（lambda_mel=45）？
A: GAN 训练天然不稳定。Mel Loss 提供**物理约束**，把生成器输出强行绑定在"频谱看起来像目标"的范围内，不让 GAN 的对抗信号把生成器带偏。

没有 Mel Loss（或权重太小）：生成器只需要骗过判别器，可能发生 mode collapse — 输出听起来还行但内容完全不对的音频。

**如果 lambda_mel=1.0**：
- FM loss（10×）成了最大项，生成器优先满足"判别器特征接近"而非"频谱接近"
- 训练早期内容信息丢失，后期 GAN loss 主导
- 主观听感：音频可能"自然"但发音模糊、音素丢失、语义不清

比喻：Mel Loss 是**地图**，GAN Loss 是"看起来像在走路"。45 是论文消融实验得到的经验值——<45 内容保真不够，>100 压制 GAN 的高频细节提升空间。

## 2026-05-18

### Q: 卷积层的作用是什么，kernel 又是什么？
A: 卷积是一个"滑动窗口扫描仪"，kernel 是窗口内的权重模板。卷积的每个 kernel 是形状为 `(in_channels, kernel_size)` 的矩阵，在时间维度上滑动，在每个位置做加权求和。一个 Conv1d 有 `out_channels` 个不同的 kernel，每个检测不同的局部模式。

参数量公式：`in_channels × kernel_size × out_channels + out_channels（bias）`
- **in_channels**: kernel 要看多少输入通道
- **kernel_size**: kernel 每个通道看几个时间位置
- **out_channels**: 有多少个不同的 kernel
- **stride 和 dilation 不影响参数量**，只影响计算行为和输出形状

对比：Conv1d(kernel_size=1) 等价于在时间轴上滑动的全连接层。

### Q: stride 是什么，怎么工作的？
A: Stride 控制卷积核每次在时间轴上移动多少步。

- **Conv1d 的 stride**：每次跳 stride 步，输出长度 ≈ T/stride（下采样）
- **ConvTranspose1d 的 stride**：先在每两个值之间插 stride-1 个零，再卷积 → 输出长度 ×stride（上采样）
- 输出长度公式（Conv1d）：`floor((T + 2×padding - kernel_size) / stride) + 1`
- Stride 只影响滑动步长，kernel 的权重数量不变

### Q: dilation 空洞卷积是什么？
A: Dilation 在 kernel 元素之间插入间隔，让 kernel 在不增加参数的前提下看得更远。

- dilation=1（默认）：kernel 元素相邻，看连续的区域
- dilation=3：kernel 元素之间间隔 2 个位置，视野扩大但参数不变
- MRF 中用 dilation=[1,3] 配合 kernel=3/5/7，组合出不同感受野（~7/~13/~19 个采样点）

### Q: padding 怎么设置的？
A: Padding 在时间轴两端补零，用于控制输出长度。

常见策略：
- `padding = (kernel_size - 1) // 2` → stride=1 时输出长度 = 输入长度
- 转置卷积 `padding = (kernel_size - stride) // 2` → 输出长度 = 输入长度 × stride

### Q: bias 是什么？
A: Bias 是一个可学习的常数偏置，加到卷积的加权求和结果上：`output = sum(x×w) + bias`。没有 bias 时输入全零则输出必为零，bias 让每层有独立于输入的"基准激活值"。参数量 = out_channels 个。

### Q: HiFi-GAN Generator 每层的输入输出和参数设计？
A: Generator = 上采样网络：Mel 频谱 (B, 80, T) → 波形 (B, 1, T×256)

| 层 | 输入→输出 | 作用 |
|------|-----------|------|
| conv_pre Conv1d(80,512,7) | (80,T)→(512,T) | Mel 80维 → 512维特征展开 |
| up1 ConvTranspose1d(512,256,16,8) | (512,T)→(256,T×8) | ×8 上采样，展开粗略周期结构 |
| MRF1 | 256→256 | 三种感受野精修波形 |
| up2 ConvTranspose1d(256,128,16,8) | (256,T×8)→(128,T×64) | ×8 上采样，补充中间结构 |
| MRF2 | 128→128 | 中频细节打磨 |
| up3 ConvTranspose1d(128,64,4,2) | (128,T×64)→(64,T×128) | ×2 上采样，高频填充 |
| MRF3 | 64→64 | 高频打磨 |
| up4 ConvTranspose1d(64,32,4,2) | (64,T×128)→(32,T×256) | ×2 上采样，最后填充 |
| MRF4 | 32→32 | 超高频抛光 |
| conv_post Conv1d(32,1,7) + tanh | (32,T×256)→(1,T×256) | 映射到波形输出，tanh 限幅 [-1,1] |

MRF = 3 路并行的 ResBlock（kernel=3/5/7），每路 2 层空洞卷积 + 残差连接，三路求和融合多尺度信息。设计动机：语音中基频（大尺度）、共振峰过渡（中尺度）、摩擦音（小尺度）需要不同感受野。

### Q: MRF 的作用是什么，ResBlock 的作用又是什么？
A: **MRF** = 3 种不同 kernel（3/5/7）的 ResBlock 并行处理 + 求和融合，让每个采样点综合了三种感受野的信息。上采样后的波形粗糙有阶梯感，MRF 负责打磨平滑。

**ResBlock（残差块）** = `输出 = Conv(x) + x`。网络只需学"调整量"而非完整值，降低学习难度。残差连接为梯度提供了"短路通道"，防止深层网络梯度消失。kernel 3/5/7 的三路分别捕捉极高频细节、中频过渡、低频周期结构。

### Q: LeakyReLU 是什么，作用是什么？
A: `LeakyReLU(x) = x if x > 0 else 0.2x`。正数保留不变，负数保留一个小坡度（0.2），不让信息完全消失。相比 ReLU 负数全砍成 0 导致神经元"坏死"，LeakyReLU 在负半轴仍有梯度，神经元永远不会停止学习。波形信号正负对称，不能丢掉负半轴信息，所以 HiFi-GAN 用 LeakyReLU。

### Q: dilations=[1, 3] 表示什么？
A: 表示 ResBlock 有 2 层卷积：第 1 层 dilation=1（普通卷积，看局部），第 2 层 dilation=3（空洞卷积，看得更远）。两层叠加用更少的参数实现了更大的感受野，还多了一层非线性（LeakyReLU），表达能力更强。

### Q: MSD 和 MPD 是什么，作用是什么？
A: MSD（多尺度判别器）和 MPD（多周期判别器）是 HiFi-GAN 的两个判别器。Generator 生成假波形，判别器判断真假，两者对抗训练互相提升。

**MSD**：3 个子判别器，分别在原始/2×下采样/4×下采样的波形上判断。捕捉不同时间尺度的伪影（高频噪声、包络异常、长程结构）。

**MPD**：5 个子判别器（period=2/3/5/7/11），把 1D 波形按 period reshape 为 2D，用 Conv2d 判断周期结构是否自然。利用语音的基频周期性防止周期抖动。

**合计 8 个判别器**，Generator 必须在所有维度上都逼真才能骗过它们。

### Q: 为什么需要判别器和生成对抗网络？
A: 只用 Mel Loss (L1) 训练的问题：Mel 是 80 维压缩表示，两个听起来完全不同的波形 Mel 可能很接近 —— 模型会输出"Mel 看起来对但听起来糊"的波形（回归到均值）。

GAN 解决了这个问题：判别器直接看完整波形（12800 维），能看到 Mel 看不到的高频噪声、基频抖动、摩擦音纹理。Generator 如果偷懒输出模糊波形，判别器一眼识破。

三种 Loss 的分工：
- **Mel Loss**：保证内容大致对齐（"关键词写对了"）
- **GAN Loss**：逼真度（"语气像人说的"）
- **Feature Matching Loss**：用判别器中间层特征的 L1 稳定训练，防止模式崩塌

## 2026-05-11

### Q: 为什么你每次操作一个命令都要征求我的同意？
A: 这是 Claude Code 的权限系统。`settings.local.json` 中的 `permissions.allow` 白名单规则默认是**精确匹配**，只有命令字符串完全匹配规则时才自动放行。之前我们的规则太窄（如 `Bash(git -C "D:/Work/Code/LearnTTS" rev-parse --git-dir)` 只匹配那一条 git 命令），换个参数就触发审批。解决方案是改成了宽泛的通配规则（如 `Bash(git *)`、`Bash(ls *)`），覆盖一类操作而非单个命令。

### Q: 和 Claude Code 的聊天记录能放到 git 仓库里同步吗？
A: 不能直接放。聊天记录存储在 `~/.claude/projects/D--Work-Code-LearnTTS/` 和 `~/.claude/history.jsonl`，在用户目录下，不在项目文件夹内。替代方案是利用 CLAUDE.md 已规定的 QA_Log.md 机制——每次概念性/学习性问题解答后自动追加记录，这个文件会被 git 追踪，从而实现知识跨设备同步。
