"""
HiFi-GAN 模型定义
==================
Generator + Multi-Scale Discriminator (MSD) + Multi-Period Discriminator (MPD)

参考文献: HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity
          Speech Synthesis (Kong et al., 2020)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm
from torch.nn.utils.parametrize import remove_parametrizations


# ============================================================
# 工具函数
# ============================================================

def init_weights(m: nn.Module, std: float = 0.01) -> None:
    """权重初始化: 正态分布 N(0, std)。"""
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
        m.weight.data.normal_(0, std)
        if m.bias is not None:
            m.bias.data.zero_()


# ============================================================
# MRF (Multi-Receptive Field Fusion)
# ============================================================

class ResBlock(nn.Module):
    """残差块: 2 层空洞卷积 + 残差连接。

    不同 kernel_size 和 dilation 组合产生不同感受野，
    MRF 通过并行多个 ResBlock 融合多尺度信息。
    """

    def __init__(self, channels: int, kernel_size: int, dilations: list[int]):
        super().__init__()
        self.convs = nn.ModuleList()
        for d in dilations:
            self.convs.append(
                weight_norm(nn.Conv1d(
                    channels, channels,
                    kernel_size=kernel_size,
                    padding=(kernel_size * d - d) // 2,  # 保持长度不变
                    dilation=d,
                ))
            )
        # 每层 conv 前有 LeakyReLU
        self.activations = nn.ModuleList(
            [nn.LeakyReLU(0.2) for _ in range(len(dilations))]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv, act in zip(self.convs, self.activations):
            x = act(x)
            x = conv(x)
        return x

    def remove_weight_norm(self):
        for conv in self.convs:
            remove_parametrizations(conv, 'weight')


class MRF(nn.Module):
    """Multi-Receptive Field Fusion。

    3 路并行的 ResBlock (kernel=3,7,11)，每路 3 层空洞卷积，
    输出求和融合不同感受野的信息。
    """

    def __init__(self, channels: int):
        super().__init__()
        self.blocks = nn.ModuleList([
            ResBlock(channels, kernel_size=3, dilations=[1, 3, 5]),
            ResBlock(channels, kernel_size=7, dilations=[1, 3, 5]),
            ResBlock(channels, kernel_size=11, dilations=[1, 3, 5]),
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = 0
        for block in self.blocks:
            out = out + block(x)
        return out

    def remove_weight_norm(self):
        for block in self.blocks:
            block.remove_weight_norm()


# ============================================================
# Generator
# ============================================================

class HiFiGANGenerator(nn.Module):
    """HiFi-GAN 生成器。

    架构:
        Mel 频谱 → Conv1d → [Upsample ×4, 每步后接 MRF] → Conv1d → tanh → 波形

    上采样路径:
        Mel (80, T) → 512 → 256 (×8) → 128 (×8) → 64 (×2) → 32 (×2) → 1
        总上采样率: 8 × 8 × 2 × 2 = 256
    """

    def __init__(
        self,
        in_channels: int = 80,
        upsample_rates: list[int] = [8, 8, 2, 2],
        upsample_kernel_sizes: list[int] = [16, 16, 4, 4],
        upsample_initial_channel: int = 512,
    ):
        super().__init__()

        self.num_kernels = len(upsample_rates)
        self.num_upsamples = len(upsample_rates)

        # 初始映射: Mel 80 维 → 高维隐空间
        self.conv_pre = weight_norm(nn.Conv1d(in_channels, upsample_initial_channel, 7, padding=3))

        # 上采样块
        self.ups = nn.ModuleList()
        self.mrfs = nn.ModuleList()

        current_channels = upsample_initial_channel
        for i, (rate, kernel) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            out_channels = max(32, current_channels // 2)
            self.ups.append(
                weight_norm(nn.ConvTranspose1d(
                    current_channels, out_channels,
                    kernel_size=kernel,
                    stride=rate,
                    padding=(kernel - rate) // 2,
                ))
            )
            self.mrfs.append(MRF(out_channels))
            current_channels = out_channels

        # 最终映射回 1 通道波形
        self.conv_post = weight_norm(nn.Conv1d(current_channels, 1, 7, padding=3))

        # 激活函数
        self.leaky = nn.LeakyReLU(0.2)

        # 初始化权重
        self.apply(lambda m: init_weights(m, 0.01))

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Args:
            mel: Mel 频谱 (B, n_mels, T)

        Returns:
            波形 (B, 1, T_wave)
        """
        x = self.leaky(self.conv_pre(mel))

        for up, mrf in zip(self.ups, self.mrfs):
            x = up(x)
            x = mrf(x)
            x = self.leaky(x)

        x = self.conv_post(x)
        x = torch.tanh(x)
        return x

    def remove_weight_norm(self):
        remove_parametrizations(self.conv_pre, 'weight')
        remove_parametrizations(self.conv_post, 'weight')
        for up in self.ups:
            remove_parametrizations(up, 'weight')
        for mrf in self.mrfs:
            mrf.remove_weight_norm()


# ============================================================
# Multi-Scale Discriminator (MSD)
# ============================================================

class ScaleDiscriminator(nn.Module):
    """单个尺度的判别器。

    7 层 stride=2 的 Conv1d，输出 logits 序列 + 最终 logit。
    """

    def __init__(self):
        super().__init__()
        self.convs = nn.ModuleList([
            weight_norm(nn.Conv1d(1, 128, 15, stride=1, padding=7)),
            weight_norm(nn.Conv1d(128, 128, 41, stride=2, padding=20, groups=4)),
            weight_norm(nn.Conv1d(128, 256, 41, stride=2, padding=20, groups=16)),
            weight_norm(nn.Conv1d(256, 512, 41, stride=4, padding=20, groups=16)),
            weight_norm(nn.Conv1d(512, 1024, 41, stride=4, padding=20, groups=16)),
            weight_norm(nn.Conv1d(1024, 1024, 41, stride=1, padding=20)),
            weight_norm(nn.Conv1d(1024, 1024, 5, stride=1, padding=2)),
        ])
        self.conv_post = weight_norm(nn.Conv1d(1024, 1, 3, stride=1, padding=1))
        self.leaky = nn.LeakyReLU(0.2)

    def forward(self, x: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor]:
        """返回 (中间层特征列表, 最终 logit)。"""
        feats = []
        for conv in self.convs:
            x = self.leaky(conv(x))
            feats.append(x)
        x = self.conv_post(x)
        feats.append(x)
        return feats, x


class MultiScaleDiscriminator(nn.Module):
    """多尺度判别器 (MSD)。

    3 个子判别器，分别在原始 / 2×下采样 / 4×下采样的波形上判断真假。
    """

    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList([
            ScaleDiscriminator(),
            ScaleDiscriminator(),
            ScaleDiscriminator(),
        ])
        # 下采样用的平均池化
        self.pool = nn.AvgPool1d(4, stride=2, padding=2)

    def forward(self, x: torch.Tensor) -> list[tuple[list[torch.Tensor], torch.Tensor]]:
        """返回 [(feats, logit), ...] for each scale。"""
        results = []
        for i, disc in enumerate(self.discriminators):
            if i == 0:
                # 原始尺度
                results.append(disc(x))
            else:
                # 下采样尺度
                x_pool = self.pool(x) if i == 1 else self.pool(self.pool(x))
                results.append(disc(x_pool))
        return results


# ============================================================
# Multi-Period Discriminator (MPD)
# ============================================================

class PeriodDiscriminator(nn.Module):
    """单个周期的判别器。

    将 1D 波形按 period reshape 为 2D，然后用 Conv2d 判断真假。
    """

    def __init__(self, period: int):
        super().__init__()
        self.period = period

        self.convs = nn.ModuleList([
            weight_norm(nn.Conv2d(1, 32, (5, 1), stride=(3, 1), padding=(2, 0))),
            weight_norm(nn.Conv2d(32, 128, (5, 1), stride=(3, 1), padding=(2, 0))),
            weight_norm(nn.Conv2d(128, 512, (5, 1), stride=(3, 1), padding=(2, 0))),
            weight_norm(nn.Conv2d(512, 1024, (5, 1), stride=(3, 1), padding=(2, 0))),
            weight_norm(nn.Conv2d(1024, 1024, (5, 1), stride=1, padding=(2, 0))),
        ])
        self.conv_post = weight_norm(nn.Conv2d(1024, 1, (3, 1), stride=1, padding=(1, 0)))
        self.leaky = nn.LeakyReLU(0.2)

    def forward(self, x: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor]:
        """返回 (中间层特征列表, 最终 logit)。"""
        B, C, T = x.shape

        # 1D → 2D: reshape 为 (B, 1, period, T//period)
        pad = (self.period - T % self.period) % self.period
        if pad > 0:
            x = F.pad(x, (0, pad), "reflect")
        x = x.view(B, C, -1, self.period).permute(0, 1, 3, 2)  # (B, C, period, T/period)

        feats = []
        for conv in self.convs:
            x = self.leaky(conv(x))
            feats.append(x)
        x = self.conv_post(x)
        feats.append(x)
        return feats, x


class MultiPeriodDiscriminator(nn.Module):
    """多周期判别器 (MPD)。

    5 个子判别器，period = [2, 3, 5, 7, 11]。
    利用语音的周期性结构，从不同周期角度判断真假。
    """

    def __init__(self, periods: list[int] = [2, 3, 5, 7, 11]):
        super().__init__()
        self.discriminators = nn.ModuleList([
            PeriodDiscriminator(p) for p in periods
        ])

    def forward(self, x: torch.Tensor) -> list[tuple[list[torch.Tensor], torch.Tensor]]:
        """返回 [(feats, logit), ...] for each period。"""
        return [disc(x) for disc in self.discriminators]


# ============================================================
# 完整 HiFi-GAN 模型容器
# ============================================================

class HiFiGAN(nn.Module):
    """HiFi-GAN 完整模型 (Generator + MSD + MPD)。"""

    def __init__(self, **gen_kwargs):
        super().__init__()
        self.generator = HiFiGANGenerator(**gen_kwargs)
        self.msd = MultiScaleDiscriminator()
        self.mpd = MultiPeriodDiscriminator()

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """仅生成器前向 (推理时用)。"""
        return self.generator(mel)


# ============================================================
# 快速测试
# ============================================================

def test_model():
    """验证模型前向传播是否正常。"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"测试设备: {device}")

    # 生成器
    gen = HiFiGANGenerator().to(device)
    mel = torch.randn(2, 80, 50).to(device)  # (B, n_mels, T)
    wav = gen(mel)
    print(f"Generator: {mel.shape} → {wav.shape}")
    assert wav.shape[-1] == mel.shape[-1] * 256, f"上采样率不对: {wav.shape[-1]} != {mel.shape[-1] * 256}"

    # MSD
    msd = MultiScaleDiscriminator().to(device)
    msd_results = msd(wav)
    print(f"MSD: {len(msd_results)} scales")
    for i, (feats, logit) in enumerate(msd_results):
        print(f"  scale {i}: logit={logit.shape}, {len(feats)} feat layers")

    # MPD
    mpd = MultiPeriodDiscriminator().to(device)
    mpd_results = mpd(wav)
    print(f"MPD: {len(mpd_results)} periods")
    for i, (feats, logit) in enumerate(mpd_results):
        print(f"  period {[2,3,5,7,11][i]}: logit={logit.shape}, {len(feats)} feat layers")

    print("✓ 模型前向测试通过")
    return gen


if __name__ == "__main__":
    test_model()