"""
VITS 共享模块
=============
WN (WaveNet-like), AffineCouplingLayer, FFT, MultiHeadAttention 等构建块。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrize import remove_parametrizations
from torch.nn.utils.parametrizations import weight_norm


# ============================================================
# 权重初始化
# ============================================================

def init_weights(m: nn.Module, std: float = 0.01) -> None:
    if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
        m.weight.data.normal_(0, std)
        if m.bias is not None:
            m.bias.data.zero_()


# ============================================================
# WN — Non-causal WaveNet 核心块
# ============================================================

class WN(nn.Module):
    """Non-causal WaveNet 模块。

    多层 dilated conv，每层有 gated activation + skip connection。
    输出是所有层 skip 输出的和，再经过最终变换。

    Args:
        hidden_channels: 隐层通道数
        kernel_size: 每层 conv 的 kernel 大小
        dilation_rate: dilation 增长率 (每层 dilation = dilation_rate ** layer_idx)
        n_layers: 层数
        gin_channels: 全局条件通道数 (0 = 无条件)
    """

    def __init__(
        self,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        gin_channels: int = 0,
    ):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd"

        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels

        self.in_layers = nn.ModuleList()
        self.res_skips = nn.ModuleList()
        self.res_scale = 1.0 / n_layers ** 0.5

        if gin_channels > 0:
            self.cond_layer = nn.Conv1d(gin_channels, 2 * hidden_channels * n_layers, 1)

        for i in range(n_layers):
            dilation = dilation_rate ** i
            padding = (kernel_size - 1) * dilation // 2
            in_layer = nn.Conv1d(
                hidden_channels, 2 * hidden_channels,
                kernel_size, dilation=dilation, padding=padding,
            )
            in_layer = weight_norm(in_layer)
            self.in_layers.append(in_layer)

            res_skip = nn.Conv1d(hidden_channels, 2 * hidden_channels, 1)
            res_skip = weight_norm(res_skip)
            self.res_skips.append(res_skip)

        self.out_layer = nn.Conv1d(hidden_channels, 2 * hidden_channels, 1)

    def forward(
        self, x: torch.Tensor, x_mask: torch.Tensor, g: torch.Tensor | None = None
    ) -> torch.Tensor:
        """前向传播。

        Args:
            x: (B, C, T)
            x_mask: (B, 1, T), 二值掩码（0=padding位置）
            g: (B, gin_channels, T) or None, 全局条件

        Returns:
            (B, C, T)
        """
        B, C, T = x.shape
        assert C == self.hidden_channels

        x = x * x_mask

        if g is not None:
            g = self.cond_layer(g)

        output = torch.zeros_like(x)
        for i in range(self.n_layers):
            x_in = self.in_layers[i](x)
            if g is not None:
                cond = g[:, i * 2 * self.hidden_channels:(i + 1) * 2 * self.hidden_channels, :]
                x_in += cond

            # gated activation: split 2*hidden → gate * filter
            gate = x_in[:, :self.hidden_channels, :]
            filt = x_in[:, self.hidden_channels:, :]
            act = gate * torch.sigmoid(filt)
            act *= self.res_scale

            # res + skip
            res_skip = self.res_skips[i](act)
            res_skip = res_skip * x_mask
            x = x + res_skip[:, :self.hidden_channels, :]
            output = output + res_skip[:, self.hidden_channels:, :]

        output = self.out_layer(output) * x_mask
        output = output[:, :self.hidden_channels, :] + output[:, self.hidden_channels:, :]
        output = output * x_mask
        return output

    def remove_weight_norm(self):
        for layer in self.in_layers:
            remove_parametrizations(layer, 'weight')
        for layer in self.res_skips:
            remove_parametrizations(layer, 'weight')
        remove_parametrizations(self.out_layer, 'weight')
        if hasattr(self, 'cond_layer') and self.gin_channels > 0:
            remove_parametrizations(self.cond_layer, 'weight')


# ============================================================
# ConvReluNorm — Conv + LayerNorm 的 FFN 块
# ============================================================

class ConvReluNorm(nn.Module):
    """Conv1d → ReLU → LayerNorm 的两层 FFN 子块。"""

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int,
                 kernel_size: int, n_layers: int, p_dropout: float):
        super().__init__()
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(n_layers):
            in_ch = in_channels if i == 0 else hidden_channels
            out_ch = out_channels if i == n_layers - 1 else hidden_channels
            padding = kernel_size // 2
            self.convs.append(nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding))
            self.norms.append(nn.LayerNorm(out_ch))

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x * x_mask)
            x = x.permute(0, 2, 1)  # (B, T, C) for LayerNorm
            x = norm(x)
            x = x.permute(0, 2, 1)  # (B, C, T)
            x = F.relu(x)
            x = F.dropout(x, self.p_dropout, training=self.training)
        return x * x_mask


# ============================================================
# MultiHeadAttention
# ============================================================

class MultiHeadAttention(nn.Module):
    def __init__(self, channels: int, out_channels: int, n_heads: int,
                 p_dropout: float, window_size: int | None = None):
        super().__init__()
        assert channels % n_heads == 0
        self.channels = channels
        self.out_channels = out_channels
        self.n_heads = n_heads
        self.window_size = window_size

        self.k_proj = nn.Linear(channels, channels)
        self.v_proj = nn.Linear(channels, channels)
        self.q_proj = nn.Linear(channels, channels)
        self.out_proj = nn.Linear(channels, out_channels)
        self.dropout = nn.Dropout(p_dropout)

    def forward(
        self, x: torch.Tensor, x_mask: torch.Tensor,
        key_value: torch.Tensor | None = None,
        kv_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, C, T = x.shape
        kv = key_value if key_value is not None else x
        kv_mask = x_mask if kv_mask is None else kv_mask

        # Linear projections
        q = self.q_proj(x.permute(0, 2, 1))  # (B, T, C)
        k = self.k_proj(kv.permute(0, 2, 1))  # (B, T_kv, C)
        v = self.v_proj(kv.permute(0, 2, 1))

        # Split heads
        q = q.view(B, T, self.n_heads, self.channels // self.n_heads).permute(0, 2, 1, 3)
        k = k.view(B, -1, self.n_heads, self.channels // self.n_heads).permute(0, 2, 1, 3)
        v = v.view(B, -1, self.n_heads, self.channels // self.n_heads).permute(0, 2, 1, 3)

        # Scaled dot-product attention
        attn = torch.matmul(q, k.transpose(-2, -1)) / (self.channels // self.n_heads) ** 0.5

        # Window mask (for local attention)
        if self.window_size is not None:
            attn_mask = torch.zeros_like(attn)
            for i in range(T):
                start = max(0, i - self.window_size // 2)
                end = min(T, i + self.window_size // 2 + 1)
                attn_mask[:, :, i, start:end] = 1.0
            attn = attn.masked_fill(attn_mask == 0, float('-inf'))

        # Padding mask
        attn = attn + (x_mask[:, :, :, None] + kv_mask[:, :, None, :] - 1) * 1e9

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.permute(0, 2, 1, 3).contiguous().view(B, T, self.channels)
        out = self.out_proj(out)
        out = out.permute(0, 2, 1)  # (B, C, T)
        return out * x_mask


# ============================================================
# FFT Block (Feed-Forward Transformer)
# ============================================================

class FFTBlock(nn.Module):
    """FastSpeech FFN + MultiHeadAttention 块。"""

    def __init__(self, channels: int, kernel_size: int, n_layers: int,
                 n_heads: int, p_dropout: float):
        super().__init__()
        self.attn = MultiHeadAttention(channels, channels, n_heads, p_dropout)
        self.ffn = ConvReluNorm(channels, channels, channels, kernel_size, 2, p_dropout)
        self.norm_attn = nn.LayerNorm(channels)
        self.norm_ffn = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
        # Self-attention with residual + norm
        attn_out = self.attn(x, x_mask)
        x = x + attn_out
        x = x.permute(0, 2, 1)
        x = self.norm_attn(x)
        x = x.permute(0, 2, 1)

        # FFN with residual + norm
        ffn_out = self.ffn(x, x_mask)
        x = x + ffn_out
        x = x.permute(0, 2, 1)
        x = self.norm_ffn(x)
        x = x.permute(0, 2, 1)
        return x * x_mask


# ============================================================
# Affine Coupling Layer — Flow 核心
# ============================================================

class AffineCouplingLayer(nn.Module):
    """Affine Coupling Layer for Normalizing Flow。

    将 x 沿通道分成两半: x₁, x₂
    x₁ 直接复制
    x₂ 经过: x₂' = (x₂ + shift) * scale
    其中 shift, scale = NN(x₁)
    """

    def __init__(self, channels: int, hidden_channels: int, kernel_size: int,
                 dilation_rate: int, n_layers: int, gin_channels: int = 0,
                 mean_only: bool = False):
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.mean_only = mean_only

        self.net = WN(channels // 2, kernel_size, dilation_rate, n_layers, gin_channels)

        # mean_only=True: 只输出 shift (logdet 恒为 0, 官方 VITS 先验 flow 的做法)。
        # mean_only=False: 输出 shift+scale (full affine), logdet 非零, 须计入目标函数。
        out_channels = channels // 2 if mean_only else channels
        self.proj = nn.Conv1d(channels // 2, out_channels, 1)
        self.proj.weight.data.zero_()
        self.proj.bias.data.zero_()

    def forward(
        self, x: torch.Tensor, x_mask: torch.Tensor, g: torch.Tensor | None = None,
        reverse: bool = False, return_logdet: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """前向: x → z (变换). 反向: z → x (逆变换)。

        Args:
            x: (B, C, T)
            x_mask: (B, 1, T)
            g: (B, gin_channels, T) or None
            reverse: False=正向(flow), True=反向(inverse flow)
            return_logdet: True 时返回 (out, log_det), log_det 形状 (B, T)

        Returns:
            return_logdet=False: (B, C, T)
            return_logdet=True: ((B, C, T), (B, T))
        """
        B, C, T = x.shape
        assert C == self.channels
        half = C // 2

        x0 = x[:, :half, :]
        x1 = x[:, half:, :]

        h = self.net(x0, x_mask, g)
        params = self.proj(h)

        if self.mean_only:
            # shift-only: 体积保持, logdet=0
            shift = params
            if not reverse:
                x1_out = x1 + shift
            else:
                x1_out = x1 - shift
            log_det = torch.zeros(B, T, device=x.device, dtype=x.dtype)
        else:
            shift = params[:, :half, :]
            scale = params[:, half:, :]
            # clamp 防止 exp(scale) 在 fp16/fp32 溢出 → NaN
            scale = torch.clamp(scale, min=-8.0, max=8.0)
            if not reverse:
                x1_out = (x1 + shift) * torch.exp(scale)
                log_det = scale.sum(dim=1) * x_mask.squeeze(1)
            else:
                x1_out = x1 * torch.exp(-scale) - shift
                log_det = -scale.sum(dim=1) * x_mask.squeeze(1)

        out = torch.cat([x0, x1_out], dim=1)
        out = out * x_mask
        if return_logdet:
            return out, log_det
        return out

    def remove_weight_norm(self):
        self.net.remove_weight_norm()
        if hasattr(self, 'proj') and hasattr(self.proj, 'parametrizations'):
            remove_parametrizations(self.proj, 'weight')


# ============================================================
# ResBlock & MRF — 用于 HiFi-GAN Generator (Decoder)
# ============================================================

class ResBlock(nn.Module):
    """残差块: 3 层空洞卷积 + 残差连接。"""

    def __init__(self, channels: int, kernel_size: int, dilations: list[int]):
        super().__init__()
        self.convs = nn.ModuleList()
        for d in dilations:
            self.convs.append(
                weight_norm(nn.Conv1d(
                    channels, channels,
                    kernel_size=kernel_size,
                    padding=(kernel_size * d - d) // 2,
                    dilation=d,
                ))
            )
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

    3 路并行的 ResBlock (kernel=3,7,11)，每路 3 层空洞卷积。
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
# 辅助函数
# ============================================================

def rand_seed(seed: int = 123):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def test_modules():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"测试设备: {device}")

    B, C, T = 2, 192, 50
    x = torch.randn(B, C, T, device=device)
    x_mask = torch.ones(B, 1, T, device=device)

    # WN
    wn = WN(C, kernel_size=5, dilation_rate=2, n_layers=4).to(device)
    out = wn(x, x_mask)
    print(f"WN: {x.shape} → {out.shape}")
    assert out.shape == x.shape

    # FFT
    fft = FFTBlock(C, kernel_size=3, n_layers=2, n_heads=2, p_dropout=0.1).to(device)
    out = fft(x, x_mask)
    print(f"FFT: {x.shape} → {out.shape}")

    # Affine Coupling (half channels)
    half = C // 2
    ac = AffineCouplingLayer(half, hidden_channels=half, kernel_size=5,
                             dilation_rate=2, n_layers=4).to(device)
    z = torch.randn(B, half, T, device=device) * x_mask
    out_fwd = ac(z, x_mask, reverse=False)
    out_rev = ac(out_fwd, x_mask, reverse=True)
    print(f"AffineCoupling: {z.shape} → {out_fwd.shape}, rev diff: {(z - out_rev).abs().max().item():.6f}")

    print("✓ 模块测试通过")


if __name__ == "__main__":
    test_modules()