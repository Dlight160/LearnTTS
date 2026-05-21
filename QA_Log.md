# QA Log

## 2026-05-11

### Q: 为什么你每次操作一个命令都要征求我的同意？
A: 这是 Claude Code 的权限系统。`settings.local.json` 中的 `permissions.allow` 白名单规则默认是**精确匹配**，只有命令字符串完全匹配规则时才自动放行。之前我们的规则太窄（如 `Bash(git -C "D:/Work/Code/LearnTTS" rev-parse --git-dir)` 只匹配那一条 git 命令），换个参数就触发审批。解决方案是改成了宽泛的通配规则（如 `Bash(git *)`、`Bash(ls *)`），覆盖一类操作而非单个命令。

### Q: 和 Claude Code 的聊天记录能放到 git 仓库里同步吗？
A: 不能直接放。聊天记录存储在 `~/.claude/projects/D--Work-Code-LearnTTS/` 和 `~/.claude/history.jsonl`，在用户目录下，不在项目文件夹内。替代方案是利用 CLAUDE.md 已规定的 QA_Log.md 机制——每次概念性/学习性问题解答后自动追加记录，这个文件会被 git 追踪，从而实现知识跨设备同步。

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
