"""
VITS 模型定义
=============
完整 VITS: Conditional VAE + Adversarial Learning for TTS

架构:
  TextEncoder → Normalizing Flow → PosteriorEncoder
                                   ↓
                              MAS (对齐)
                                   ↓
                         StochasticDurationPredictor
                                   ↓
                             Generator (HiFi-GAN)
                                   ↓
                              波形 + GAN

参考: VITS: Conditional Variational Autoencoder with Adversarial Learning
      for End-to-End Text-to-Speech (Kim et al., 2021)
"""

import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrize import remove_parametrizations
from torch.nn.utils.parametrizations import weight_norm

# Add parent dir for imports when run directly
sys.path.insert(0, str(Path(__file__).parent.parent))

from stage3_vits.modules import WN, AffineCouplingLayer, ConvReluNorm, FFTBlock, MRF, init_weights
from stage3_vits.mas import mas_hardcoded


# ============================================================
# TextEncoder — 从音素序列到先验分布
# ============================================================

class TextEncoder(nn.Module):
    """文本编码器: 音素序列 → 先验分布参数 (m_p, logs_p)。

    使用多层 FFT (self-attention + FFN) 提取上下文表示。

    Args:
        n_vocab: 音素词汇表大小
        out_channels: 输出通道数 (隐变量维度, = z_channels)
        hidden_channels: 隐层通道数
        n_layers: FFT 层数
        n_heads: 自注意力头数
        kernel_size: FFN conv kernel 大小
        p_dropout: dropout 概率
    """

    def __init__(
        self,
        n_vocab: int,
        out_channels: int,
        hidden_channels: int,
        n_layers: int = 6,
        n_heads: int = 2,
        kernel_size: int = 3,
        p_dropout: float = 0.1,
    ):
        super().__init__()

        self.emb = nn.Embedding(n_vocab, hidden_channels)
        nn.init.normal_(self.emb.weight, 0, 0.1)

        # 输入投影
        self.pre = nn.Conv1d(hidden_channels, hidden_channels, 1)

        # FFT 块
        self.fft_blocks = nn.ModuleList([
            FFTBlock(hidden_channels, kernel_size, 2, n_heads, p_dropout)
            for _ in range(n_layers)
        ])

        # 输出投影到分布参数
        self.proj_m = nn.Conv1d(hidden_channels, out_channels, 1)
        self.proj_logs = nn.Conv1d(hidden_channels, out_channels, 1)

        self.hidden_channels = hidden_channels

    def forward(
        self, x: torch.Tensor, x_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """编码文本为先验分布。

        Args:
            x: (B, T_text), 音素 ID
            x_mask: (B, 1, T_text), text mask

        Returns:
            m_p: (B, C, T_text), 均值
            logs_p: (B, C, T_text), 对数方差
            x_mask: (B, 1, T_text), mask (传递)
        """
        x = self.emb(x) * math.sqrt(self.hidden_channels)
        x = x.permute(0, 2, 1)  # (B, C, T)
        x = self.pre(x) * x_mask

        for block in self.fft_blocks:
            x = block(x, x_mask)

        m_p = self.proj_m(x) * x_mask
        logs_p = self.proj_logs(x) * x_mask
        # Clamp logs_p 防止极端方差导致 KL 爆炸
        logs_p = torch.clamp(logs_p, min=-10, max=5)
        return m_p, logs_p, x_mask


# ============================================================
# PosteriorEncoder — 从波形提取后验分布
# ============================================================

class PosteriorEncoder(nn.Module):
    """后验编码器: 波形 → 后验分布参数 (z ~ q(z|x))。

    使用非因果 WaveNet (WN) 提取隐变量表示。

    Args:
        in_channels: 输入通道数 (Mel 频谱维度)
        out_channels: 输出通道数 (隐变量维度)
        hidden_channels: 隐层通道数
        kernel_size: WN kernel 大小
        dilation_rate: WN dilation 增长率
        n_layers: WN 层数
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        kernel_size: int = 5,
        dilation_rate: int = 2,
        n_layers: int = 16,
    ):
        super().__init__()
        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.wn = WN(hidden_channels, kernel_size, dilation_rate, n_layers)
        self.proj_m = nn.Conv1d(hidden_channels, out_channels, 1)
        self.proj_logs = nn.Conv1d(hidden_channels, out_channels, 1)

    def forward(
        self, x: torch.Tensor, x_mask: torch.Tensor,
        g: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """编码波形为后验分布。

        Args:
            x: (B, C_in, T), 输入 (Mel 频谱)
            x_mask: (B, 1, T), mask
            g: (B, gin_channels, T) or None, 全局条件

        Returns:
            z: (B, C_out, T), 采样的隐变量
            m_q: (B, C_out, T), 后验均值
            logs_q: (B, C_out, T), 后验对数方差
        """
        x = self.pre(x) * x_mask
        x = self.wn(x, x_mask, g)
        m_q = self.proj_m(x) * x_mask
        logs_q = self.proj_logs(x) * x_mask
        logs_q = torch.clamp(logs_q, min=-10, max=5)
        z = m_q + torch.randn_like(m_q) * torch.exp(logs_q)
        return z, m_q, logs_q


# ============================================================
# ResidualCouplingBlock — Normalizing Flow
# ============================================================

class ResidualCouplingBlock(nn.Module):
    """残差耦合层块 (Normalizing Flow)。

    多层 AffineCouplingLayer 堆叠。
    将简单的高斯先验变换为复杂分布，提升先验表达能力。

    Args:
        channels: 通道数
        hidden_channels: WN 隐层通道数
        kernel_size: WN kernel 大小
        dilation_rate: WN dilation 增长率
        n_layers: 每层 coupling 的 WN 层数
        n_flows: coupling layer 数量
        gin_channels: 全局条件通道数
    """

    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        kernel_size: int = 5,
        dilation_rate: int = 2,
        n_layers: int = 4,
        n_flows: int = 4,
        gin_channels: int = 0,
    ):
        super().__init__()
        # mean_only=True: 与官方 VITS 先验 flow 一致, shift-only (logdet=0),
        # 使 KL 解析式 (无 logdet 项) 严格成立, 训练稳定。
        self.flows = nn.ModuleList()
        for i in range(n_flows):
            self.flows.append(
                AffineCouplingLayer(channels, hidden_channels, kernel_size,
                                    dilation_rate, n_layers, gin_channels,
                                    mean_only=True)
            )

    def forward(
        self, x: torch.Tensor, x_mask: torch.Tensor,
        g: torch.Tensor | None = None, reverse: bool = False,
        return_logdet: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """前向/反向通过所有 coupling layers。

        每层之后翻转通道 (torch.flip on dim=1), 使两个半区轮流被变换;
        否则 coupling 只变换后半通道, 前半永远是恒等映射。flip 是置换, logdet=0。

        Args:
            x: (B, C, T)
            x_mask: (B, 1, T)
            g: (B, gin_channels, T) or None
            reverse: False=正向(flow) / True=反向(inv flow)
            return_logdet: True 且 reverse=False 时返回 (x, total_logdet)

        Returns:
            reverse=True 或 return_logdet=False: (B, C, T)
            reverse=False 且 return_logdet=True: ((B, C, T), (B, T))
        """
        if not reverse:
            total_logdet = None
            for flow in self.flows:
                if return_logdet:
                    x, ld = flow(x, x_mask, g, reverse=False, return_logdet=True)
                    total_logdet = ld if total_logdet is None else total_logdet + ld
                else:
                    x = flow(x, x_mask, g, reverse=False)
                x = torch.flip(x, [1])
            if return_logdet:
                return x, total_logdet
            return x
        else:
            for flow in reversed(self.flows):
                x = torch.flip(x, [1])
                x = flow(x, x_mask, g, reverse=True)
            return x


# ============================================================
# StochasticDurationPredictor — 随机时长预测器
# ============================================================

class StochasticDurationPredictor(nn.Module):
    """随机时长预测器 (Flow-based)。

    不是输出确定的时长值，而是输出时长的分布。
    使用 AffineCouplingLayer 堆叠构建可逆变换。

    训练: durations → log → flow → N(0,1), 计算 NLL
    推理: N(0,1) → inverse flow → exp → durations

    Args:
        in_channels: 输入通道数 (text encoder 输出)
        hidden_channels: Flow 通道数
        kernel_size: WN kernel 大小
        dilation_rate: WN dilation 增长率
        n_layers: 每层 coupling 的 WN 层数
        n_flows: coupling layer 数量
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        kernel_size: int = 3,
        dilation_rate: int = 2,
        n_layers: int = 4,
        n_flows: int = 4,
    ):
        super().__init__()

        # 输入投影: in_channels → hidden_channels
        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)

        # Conv block for text features
        self.convs = nn.ModuleList()
        for i in range(3):
            in_ch = hidden_channels if i > 0 else hidden_channels
            out_ch = hidden_channels
            self.convs.append(
                weight_norm(nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1))
            )

        # Flow: stack of coupling layers
        self.flows = nn.ModuleList()
        for i in range(n_flows):
            self.flows.append(
                AffineCouplingLayer(hidden_channels, hidden_channels // 2,
                                    kernel_size, dilation_rate, n_layers)
            )

        # Duration projection
        self.dur_proj = nn.Conv1d(1, hidden_channels, 1)
        self.log_dur_proj = nn.Conv1d(hidden_channels, 1, 1)

        self.leaky = nn.LeakyReLU(0.1)

    def forward(
        self, x: torch.Tensor, x_mask: torch.Tensor,
        durations: torch.Tensor | None = None,
        reverse: bool = False,
    ) -> torch.Tensor:
        """前向 / 反向传播。

        Args:
            x: (B, C, T_text), text encoder 输出
            x_mask: (B, 1, T_text)
            durations: (B, 1, T_text) or None
            reverse: False=训练 NLL / True=推理采样

        Returns:
            训练: log_duration_prob (B, 1, T_text)
            推理: sampled_durations (B, 1, T_text)
        """
        # Pre-process text features
        h = self.pre(x) * x_mask
        for conv in self.convs:
            h = self.leaky(conv(h)) * x_mask

        if not reverse:
            # Training: durations → noise, compute NLL
            # Map durations (log) to hidden_channels
            log_dur = torch.log(durations + 1e-8)
            d_emb = self.dur_proj(log_dur) * x_mask
            flow_in = h + d_emb

            # Through flow to noise space (累积 logdet, 每层后翻转通道)
            total_logdet = None
            for flow in self.flows:
                flow_in, ld = flow(flow_in, x_mask, reverse=False, return_logdet=True)
                total_logdet = ld if total_logdet is None else total_logdet + ld
                flow_in = torch.flip(flow_in, [1])

            z = flow_in

            # 变量代换: log p(dur) = log N(z|0,1) + Σ logdet
            # 缺少 logdet 时, flow 会把 scale 压向 -∞ 来作弊降低 z² → 发散。
            log_prob = -0.5 * (math.log(2 * math.pi) + z ** 2)
            log_prob = log_prob.sum(dim=1, keepdim=True)  # (B,1,T)
            log_prob = log_prob + total_logdet.unsqueeze(1)  # 加入 Jacobian 行列式
            log_prob = log_prob * x_mask
            return log_prob
        else:
            # Inference: sample noise → durations
            z = torch.randn_like(h)

            # Inverse flow (与正向严格互逆: 先翻转再逆变换)
            for flow in reversed(self.flows):
                z = torch.flip(z, [1])
                z = flow(z, x_mask, reverse=True)

            # Project to duration
            log_dur = self.log_dur_proj(z)
            dur = torch.exp(log_dur) * x_mask
            dur = torch.clamp(dur, min=0.01)
            return dur


# ============================================================
# Generator (Decoder) — HiFi-GAN Generator
# ============================================================

class Generator(nn.Module):
    """VITS Decoder = HiFi-GAN Generator。

    与阶段 2 的 HiFiGANGenerator 一致。
    输入: 隐变量 z (或 Mel 频谱) → 波形
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

        self.conv_pre = weight_norm(nn.Conv1d(in_channels, upsample_initial_channel, 7, padding=3))

        self.ups = nn.ModuleList()
        self.mrfs = nn.ModuleList()

        current_channels = upsample_initial_channel
        for i, (rate, kernel) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            out_channels = max(32, current_channels // 2)
            self.ups.append(
                weight_norm(nn.ConvTranspose1d(
                    current_channels, out_channels,
                    kernel_size=kernel, stride=rate,
                    padding=(kernel - rate) // 2,
                ))
            )
            self.mrfs.append(MRF(out_channels))
            current_channels = out_channels

        self.conv_post = weight_norm(nn.Conv1d(current_channels, 1, 7, padding=3))
        self.leaky = nn.LeakyReLU(0.2)

        self.apply(lambda m: init_weights(m, 0.01))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, C, T_z) → wav: (B, 1, T_wav)"""
        x = self.leaky(self.conv_pre(z))
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
# SynthesizerTrn — 完整 VITS 模型
# ============================================================

class SynthesizerTrn(nn.Module):
    """VITS 完整模型: 训练 & 推理。

    整合 TextEncoder, PosteriorEncoder, Flow, DurationPredictor,
    Generator (HiFi-GAN Decoder), MSD, MPD。
    """

    def __init__(
        self,
        n_vocab: int = 88,           # 音素词汇表大小
        spec_channels: int = 80,      # Mel 频谱维度
        segment_size: int = 8192,     # 训练音频片段长度
        inter_channels: int = 192,    # 隐变量通道数 (z_channels)
        hidden_channels: int = 192,   # 隐层通道数
        kernel_size: int = 5,         # WN kernel 大小
        dilation_rate: int = 2,       # WN dilation 增长率
        n_layers: int = 4,           # WN/FFT 层数 (TextEncoder)
        n_flows: int = 4,            # Flow coupling layer 数
        n_heads: int = 2,            # 自注意力头数
        p_dropout: float = 0.1,      # dropout 概率
        gin_channels: int = 0,       # 全局条件通道 (multi-speaker)
    ):
        super().__init__()

        self.segment_size = segment_size
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels

        # Encoder
        self.enc_p = TextEncoder(
            n_vocab, inter_channels, hidden_channels,
            n_layers=n_layers, n_heads=n_heads,
            kernel_size=kernel_size, p_dropout=p_dropout,
        )

        # Flow (先验增强)
        self.flow = ResidualCouplingBlock(
            inter_channels, hidden_channels,
            kernel_size=5, dilation_rate=1,
            n_layers=4, n_flows=n_flows,
            gin_channels=gin_channels,
        )

        # Posterior Encoder
        self.enc_q = PosteriorEncoder(
            spec_channels, inter_channels, hidden_channels,
            kernel_size=5, dilation_rate=1,
            n_layers=16,
        )

        # Duration Predictor
        self.dp = StochasticDurationPredictor(
            hidden_channels, 192,
            kernel_size=3, dilation_rate=2,
            n_layers=4, n_flows=4,
        )

        # Decoder (Generator)
        self.dec = Generator(
            in_channels=inter_channels,
            upsample_rates=[8, 8, 2, 2],
            upsample_kernel_sizes=[16, 16, 4, 4],
            upsample_initial_channel=512,
        )

        # 判别器在 train.py 中独立创建
        self.msd = None
        self.mpd = None

    @property
    def device(self):
        return next(self.parameters()).device

    # ================================================================
    # 训练前向 (VAE 训练)
    # ================================================================

    def forward(
        self,
        x: torch.Tensor,           # (B, T_text), 音素 IDs
        x_mask: torch.Tensor,      # (B, 1, T_text), text mask
        mel: torch.Tensor,         # (B, spec_channels, T_mel), Mel 频谱
        y: torch.Tensor,           # (B, 1, T_wav), 原始波形 (slice + GAN)
        y_mask: torch.Tensor,      # (B, 1, T_mel), mel mask
        g: torch.Tensor | None = None,  # (B, gin_ch, T) or None
    ) -> dict[str, torch.Tensor]:
        """完整训练前向。

        Returns:
            dict with keys: z_p, m_p, logs_p, z_q, m_q, logs_q,
                           attn, y_hat, log_dur_prob, kl_loss
        """
        B = x.shape[0]

        # --- Posterior Encoder ---
        z_q, m_q, logs_q = self.enc_q(mel, y_mask, g=g)

        # --- Text Encoder ---
        m_p, logs_p, x_mask = self.enc_p(x, x_mask)

        # --- MAS: 在 no_grad 下用 z_q 发现对齐 (官方 VITS 在 flow 之前做 MAS) ---
        with torch.no_grad():
            attn = mas_hardcoded(z_q, m_p, logs_p, x_mask, y_mask)

        # --- Flow: 先验增强 (mean-only coupling, logdet=0) ---
        z_p = self.flow(z_q, y_mask, g=g, reverse=False)

        # --- 用 attn 展开先验: (B, C, T_text) → (B, C, T_mel) ---
        attn_float = attn.float()
        m_p_exp = torch.bmm(m_p, attn_float)
        logs_p_exp = torch.bmm(logs_p, attn_float)

        # --- KL Loss (官方 VITS 解析 KL) ---
        # KL(N(m_q,σ_q²) || N(m_p,σ_p²))
        #   = 0.5 * [2(logσ_p-logσ_q) + (μ_q-μ_p)²/σ_p² + σ_q²/σ_p² - 1]
        # 其中 (μ_q-μ_p)² 用采样近似: (z_p - m_p)² (z_p = flow(z_q), 是 μ_q 的变换)
        kl = (logs_p_exp - logs_q - 0.5 +
              0.5 * ((z_p - m_p_exp) ** 2) * torch.exp(-2 * logs_p_exp) +
              0.5 * torch.exp(2 * (logs_q - logs_p_exp)))
        kl_loss = (kl * y_mask).sum() / y_mask.sum()

        # --- Duration Predictor ---
        # 从对齐提取时长
        dur = self._attn_to_dur(attn, x_mask)
        log_dur_prob = self.dp(m_p.detach(), x_mask, durations=dur, reverse=False)
        dur_loss = -log_dur_prob.mean() / B

        # --- Generator ---
        # 训练时用 posterior z (short segments for efficiency)
        z_slice = self._slice_z(z_q, y_mask, self.segment_size)
        y_hat = self.dec(z_slice)

        # 裁剪真实波形
        slice_len = y_hat.shape[-1]
        if y.shape[-1] > slice_len:
            start = torch.randint(0, y.shape[-1] - slice_len, (B,), device=y.device)
            y_slice = torch.stack([y[b, :, s:s + slice_len] for b, s in enumerate(start)])
        else:
            y_slice = y

        return {
            'y_hat': y_hat,
            'y_slice': y_slice,
            'z_p': z_p,
            'm_p': m_p,
            'logs_p': logs_p,
            'z_q': z_q,
            'm_q': m_q,
            'logs_q': logs_q,
            'attn': attn,
            'dur': dur,
            'log_dur_prob': log_dur_prob,
            'kl_loss': kl_loss,
            'dur_loss': dur_loss,
        }

    # ================================================================
    # 推理
    # ================================================================

    @torch.no_grad()
    def inference(
        self,
        x: torch.Tensor,           # (1, T_text), 音素 ID
        x_mask: torch.Tensor,      # (1, 1, T_text)
        g: torch.Tensor | None = None,
        noise_scale: float = 1.0,  # z 采样噪声尺度
        noise_scale_dur: float = 1.0,  # duration 噪声尺度
    ) -> torch.Tensor:
        """推理: 文本 → 波形。

        Args:
            x: (1, T_text), 音素 IDs
            x_mask: (1, 1, T_text)
            g: (1, gin_ch, T) or None
            noise_scale: z 采样随机性 (越大越多样)
            noise_scale_dur: duration 随机性

        Returns:
            y_hat: (1, 1, T_wav), 生成波形
        """
        # Text Encoder → prior
        m_p, logs_p, x_mask = self.enc_p(x, x_mask)

        # Duration Predictor → 每个音素的帧数
        dur = self.dp(m_p, x_mask, reverse=True)
        dur = dur * noise_scale_dur

        # 构建 y_mask
        T_mel = int(dur.sum().item())
        y_mask = torch.ones(1, 1, T_mel, device=x.device)

        # 从对齐展开 (dur → attn)
        attn = self._dur_to_attn(dur, x_mask, y_mask)

        # 展开先验: attn (1, T_text, T_mel) × m_p (1, C, T_text) → (1, C, T_mel)
        m_p_exp = torch.bmm(attn.transpose(1, 2), m_p.transpose(1, 2)).transpose(1, 2)
        logs_p_exp = torch.bmm(attn.transpose(1, 2), logs_p.transpose(1, 2)).transpose(1, 2)

        # 采样 z
        z = m_p_exp + torch.randn_like(m_p_exp) * torch.exp(logs_p_exp) * noise_scale

        # Flow reverse → Decoder
        z = self.flow(z, y_mask, g=g, reverse=True)
        y_hat = self.dec(z)
        return y_hat

    # ================================================================
    # 辅助方法
    # ================================================================

    def _gaussian_log_prob(
        self, m: torch.Tensor, logs: torch.Tensor, z: torch.Tensor
    ) -> torch.Tensor:
        """计算 log N(z | m, exp(logs))."""
        return -0.5 * (math.log(2 * math.pi) + 2 * logs +
                       (z - m) ** 2 * torch.exp(-2 * logs))

    def _attn_to_dur(
        self, attn: torch.Tensor, x_mask: torch.Tensor
    ) -> torch.Tensor:
        """对齐矩阵 → 每个音素的时长。"""
        dur = attn.sum(dim=-1, keepdim=False).float()  # (B, T_text)
        dur = dur.unsqueeze(1)  # (B, 1, T_text)
        dur = dur * x_mask
        return dur

    def _dur_to_attn(
        self, dur: torch.Tensor, x_mask: torch.Tensor, y_mask: torch.Tensor
    ) -> torch.Tensor:
        """时长 → 对齐矩阵 (用于推理展开)。"""
        B, _, T_text = x_mask.shape
        T_mel = y_mask.shape[-1]
        # Round durations to integer frames
        dur_int = torch.round(dur).long()
        dur_int = torch.clamp(dur_int, min=1)

        attn = torch.zeros(B, T_text, T_mel, device=dur.device)
        for b in range(B):
            pos = 0
            for i in range(T_text):
                d = dur_int[b, 0, i].item()
                if pos + d > T_mel:
                    d = T_mel - pos
                if d > 0:
                    attn[b, i, pos:pos + d] = 1.0
                    pos += d
                if pos >= T_mel:
                    break
        return attn

    def _slice_z(
        self, z: torch.Tensor, y_mask: torch.Tensor, segment_size: int
    ) -> torch.Tensor:
        """在训练时裁剪 z 到 segment_size 的对应长度。"""
        T_mel = z.shape[-1]
        hop_length = 256
        z_seg_len = segment_size // hop_length

        if T_mel <= z_seg_len:
            return z

        start = torch.randint(0, T_mel - z_seg_len, (1,)).item()
        return z[:, :, start:start + z_seg_len]

    def remove_weight_norm(self):
        self.dec.remove_weight_norm()
        if hasattr(self, 'enc_p'):
            for block in self.enc_p.fft_blocks:
                if hasattr(block, 'remove_weight_norm'):
                    block.remove_weight_norm()


# ============================================================
# 快速测试
# ============================================================

def test_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"测试设备: {device}")

    model = SynthesizerTrn(n_vocab=88).to(device)
    model.eval()

    B = 2
    T_text = 20
    T_mel = 50

    x = torch.randint(3, 50, (B, T_text), device=device)
    x_mask = torch.ones(B, 1, T_text, device=device)
    mel = torch.randn(B, 80, T_mel, device=device)
    y = torch.randn(B, 1, T_mel * 256, device=device)
    y_mask = torch.ones(B, 1, T_mel, device=device)

    # 训练模式
    model.train()
    out = model(x, x_mask, mel, y, y_mask)
    print(f"y_hat: {out['y_hat'].shape}")
    print(f"kl_loss: {out['kl_loss'].item():.4f}")
    print(f"dur_loss: {out['dur_loss'].item():.4f}")
    print(f"attn: {out['attn'].shape}")
    print(f"dur: {out['dur'].shape}")

    # 推理模式
    model.eval()
    y_hat = model.inference(x[:1], x_mask[:1])
    print(f"inference: {y_hat.shape}")

    print("✓ VITS 模型测试通过")


if __name__ == "__main__":
    test_model()