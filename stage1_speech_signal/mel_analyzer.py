"""
阶段 1 — Mel 频谱分析器
========================
语音信号处理基础实践项目。

功能:
  1. 加载任意 wav 音频
  2. 短时傅里叶变换 (STFT) → 频谱图
  3. Mel 滤波器组 → Mel 频谱图
  4. Griffin-Lim 算法从幅度谱重建音频
  5. 可视化对比原始音频与重建音频
  6. 分析不同参数对重建质量的影响

核心学习点:
  - 声源-滤波器模型 (source-filter model)
  - 时频分辨率权衡 (窗口长度 vs 频率分辨率)
  - Mel 尺度的感知动机
  - 相位信息的重要性 / Griffin-Lim 的局限
"""

import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
import torchaudio
import torchaudio.functional as F

warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================
# 1. 音频加载与基础信息
# ============================================================

def load_audio(filepath: str, target_sr: int | None = None) -> tuple[torch.Tensor, int]:
    """加载音频文件，返回 (waveform, sample_rate)。

    Args:
        filepath: wav 文件路径
        target_sr: 目标采样率，None 则保持原采样率
    """
    waveform, sr = torchaudio.load(filepath)
    if target_sr is not None and target_sr != sr:
        waveform = F.resample(waveform, sr, target_sr)
        sr = target_sr
    # 转为单声道
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform, sr


def print_audio_info(waveform: torch.Tensor, sr: int) -> None:
    """打印音频基本信息。"""
    duration = waveform.shape[-1] / sr
    print(f"  采样率: {sr} Hz")
    print(f"  声道数: {waveform.shape[0]}")
    print(f"  总采样点数: {waveform.shape[-1]}")
    print(f"  时长: {duration:.3f} 秒")
    print(f"  幅度范围: [{waveform.min():.4f}, {waveform.max():.4f}]")


# ============================================================
# 2. STFT & 频谱图
# ============================================================

def compute_stft(
    waveform: torch.Tensor,
    n_fft: int = 1024,
    hop_length: int = 256,
    win_length: int | None = None,
    window_fn: callable = torch.hann_window,
) -> torch.Tensor:
    """计算短时傅里叶变换。

    Args:
        waveform: 输入波形 (1, T)
        n_fft: FFT 点数
        hop_length: 帧移
        win_length: 窗长 (默认 = n_fft)
        window_fn: 窗函数

    Returns:
        复数频谱 (1, n_fft//2 + 1, num_frames)
    """
    if win_length is None:
        win_length = n_fft

    window = window_fn(win_length)
    stft = torch.stft(
        waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        return_complex=True,
    )
    return stft


def spectrogram_from_stft(stft: torch.Tensor) -> torch.Tensor:
    """从复数 STFT 计算功率谱 (幅度平方)。"""
    return stft.abs() ** 2


def amplitude_from_stft(stft: torch.Tensor) -> torch.Tensor:
    """从复数 STFT 计算幅度谱。"""
    return stft.abs()


# ============================================================
# 3. Mel 滤波器组
# ============================================================

def create_mel_filterbank(
    n_mels: int,
    n_fft: int,
    sample_rate: int,
    f_min: float = 0.0,
    f_max: float | None = None,
) -> torch.Tensor:
    """创建 Mel 滤波器组矩阵。

    公式: mel(f) = 2595 * log10(1 + f/700)

    Args:
        n_mels: Mel 滤波器数量
        n_fft: FFT 点数
        sample_rate: 采样率
        f_min: 最低频率 (Hz)
        f_max: 最高频率 (Hz)，默认 Nyquist 频率

    Returns:
        滤波器组矩阵 (n_mels, n_fft//2 + 1)
    """
    if f_max is None:
        f_max = sample_rate / 2.0

    # 在 Mel 尺度上均匀分布 n_mels+2 个点 (含 f_min, f_max 各留一个边界)
    mel_min = _hz_to_mel(f_min)
    mel_max = _hz_to_mel(f_max)
    mel_points = torch.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = _mel_to_hz(mel_points)

    # 对应到 FFT bin 索引
    n_freqs = n_fft // 2 + 1
    bin_indices = torch.floor((n_fft + 1) * hz_points / sample_rate).long()
    # 限制在有效范围内
    bin_indices = torch.clamp(bin_indices, 0, n_freqs - 1)

    filters = torch.zeros(n_mels, n_freqs)

    for i in range(1, n_mels + 1):
        left = bin_indices[i - 1]
        center = bin_indices[i]
        right = bin_indices[i + 1]

        # 左斜坡
        if center > left:
            filters[i - 1, left:center] = (
                torch.arange(left, center, dtype=torch.float32) - left
            ) / max(center - left, 1)

        # 右斜坡
        if right > center:
            filters[i - 1, center:right] = (
                right - torch.arange(center, right, dtype=torch.float32)
            ) / max(right - center, 1)

    return filters


def _hz_to_mel(f: float | torch.Tensor) -> torch.Tensor:
    """Hz → Mel 尺度。"""
    f = torch.as_tensor(f, dtype=torch.float32)
    return 2595.0 * torch.log10(1.0 + f / 700.0)


def _mel_to_hz(m: float | torch.Tensor) -> torch.Tensor:
    """Mel → Hz 尺度。"""
    m = torch.as_tensor(m, dtype=torch.float32)
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def compute_mel_spectrogram(
    spec: torch.Tensor,
    mel_filters: torch.Tensor,
) -> torch.Tensor:
    """将功率谱 / 幅度谱转换为 Mel 频谱。

    Args:
        spec: 功率谱或幅度谱 (..., n_freqs, n_frames)
        mel_filters: Mel 滤波器组 (n_mels, n_freqs)

    Returns:
        Mel 频谱 (..., n_mels, n_frames)
    """
    return torch.matmul(mel_filters, spec)


# ============================================================
# 4. Griffin-Lim 相位重建
# ============================================================

def griffin_lim(
    magnitude: torch.Tensor,
    n_fft: int,
    hop_length: int,
    win_length: int | None = None,
    n_iter: int = 60,
    window_fn: callable = torch.hann_window,
) -> torch.Tensor:
    """Griffin-Lim 算法：从幅度谱迭代重建相位。

    核心思想:
      1. 初始化随机相位 → 构造复数频谱
      2. iSTFT → 时域波形
      3. STFT → 提取相位，保留给定的幅度
      4. 重复 2-3 直到收敛

    Args:
        magnitude: 幅度谱 (1, n_freqs, n_frames)，不含 batch 维度
        n_fft: FFT 点数
        hop_length: 帧移
        win_length: 窗长
        n_iter: 迭代次数
        window_fn: 窗函数

    Returns:
        重建波形 (1, T)
    """
    if win_length is None:
        win_length = n_fft

    window = window_fn(win_length)
    # 确保 window 在正确设备上
    window = window.to(magnitude.device)

    # 初始随机相位
    angles = torch.randn_like(magnitude, dtype=torch.float32)
    angles = angles / (angles.norm(dim=-2, keepdim=True) + 1e-8) * np.pi

    for i in range(n_iter):
        # 构造复数频谱
        stft_complex = magnitude.to(torch.complex64) * torch.exp(1j * angles.to(torch.complex64))

        # iSTFT → 时域
        waveform = torch.istft(
            stft_complex,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
        )

        # STFT → 提取相位
        stft_new = torch.stft(
            waveform.unsqueeze(0),
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            return_complex=True,
        ).squeeze(0)

        angles = stft_new.angle()

    # 最后一次 iSTFT 得到最终波形
    stft_complex = magnitude.to(torch.complex64) * torch.exp(1j * angles.to(torch.complex64))
    waveform = torch.istft(
        stft_complex,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
    )
    return waveform.unsqueeze(0)  # (1, T)


# ============================================================
# 5. 可视化
# ============================================================

def plot_analysis(
    waveform: torch.Tensor,
    sr: int,
    stft_mag: torch.Tensor,
    mel_spec: torch.Tensor,
    n_fft: int,
    hop_length: int,
    n_mels: int,
    mel_filters: torch.Tensor,
    output_dir: str,
) -> None:
    """生成综合分析可视化。"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # --- 原始波形 ---
    t = np.arange(waveform.shape[-1]) / sr
    axes[0, 0].plot(t, waveform.squeeze().numpy(), linewidth=0.5)
    axes[0, 0].set_title("原始波形")
    axes[0, 0].set_xlabel("时间 (s)")
    axes[0, 0].set_ylabel("幅度")

    # --- 线性频谱图 ---
    spec_db = 20 * torch.log10(stft_mag.squeeze() + 1e-6)
    im = axes[0, 1].imshow(
        spec_db.numpy(),
        origin="lower",
        aspect="auto",
        cmap="magma",
        extent=[0, stft_mag.shape[-1] * hop_length / sr, 0, sr / 2],
    )
    axes[0, 1].set_title(f"线性频谱图 (n_fft={n_fft})")
    axes[0, 1].set_xlabel("时间 (s)")
    axes[0, 1].set_ylabel("频率 (Hz)")
    plt.colorbar(im, ax=axes[0, 1], label="dB")

    # --- Mel 频谱图 ---
    mel_db = 20 * torch.log10(mel_spec.squeeze() + 1e-6)
    im2 = axes[0, 2].imshow(
        mel_db.numpy(),
        origin="lower",
        aspect="auto",
        cmap="magma",
        extent=[0, mel_spec.shape[-1] * hop_length / sr, 0, n_mels - 1],
    )
    axes[0, 2].set_title(f"Mel 频谱图 (n_mels={n_mels})")
    axes[0, 2].set_xlabel("时间 (s)")
    axes[0, 2].set_ylabel("Mel bin")
    plt.colorbar(im2, ax=axes[0, 2], label="dB")

    # --- Mel 滤波器组可视化 ---
    freqs = np.linspace(0, sr / 2, mel_filters.shape[1])
    for i in range(0, n_mels, max(1, n_mels // 20)):
        axes[1, 0].plot(freqs, mel_filters[i].numpy(), linewidth=0.7)
    axes[1, 0].set_title(f"Mel 滤波器组 ({n_mels} filters)")
    axes[1, 0].set_xlabel("频率 (Hz)")
    axes[1, 0].set_ylabel("权重")
    axes[1, 0].set_xscale("log")
    axes[1, 0].set_xlim([20, sr / 2])

    # --- 某个时间帧的频谱对比 ---
    mid_frame = stft_mag.shape[-1] // 2
    freq_axis = np.linspace(0, sr / 2, stft_mag.shape[-2])
    axes[1, 1].plot(freq_axis, stft_mag[0, :, mid_frame].numpy(), linewidth=0.7, label="线性频谱")
    axes[1, 1].set_title(f"单帧频谱对比 (frame #{mid_frame})")
    axes[1, 1].set_xlabel("频率 (Hz)")
    axes[1, 1].set_ylabel("幅度")
    axes[1, 1].legend()

    # Mel 滤波后的该帧
    mel_axis = np.arange(n_mels)
    axes[1, 2].plot(mel_axis, mel_spec[0, :, mid_frame].numpy(), linewidth=0.7, color="C1")
    axes[1, 2].set_title(f"Mel 频谱 (frame #{mid_frame})")
    axes[1, 2].set_xlabel("Mel bin")
    axes[1, 2].set_ylabel("能量")

    plt.tight_layout()
    save_path = Path(output_dir) / "mel_analysis.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  可视化已保存: {save_path}")


def plot_reconstruction_comparison(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    sr: int,
    output_dir: str,
) -> None:
    """对比原始音频和 Griffin-Lim 重建音频。"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 8))

    t = np.arange(original.shape[-1]) / sr

    for row, (waveform, label) in enumerate(
        [(original, "原始"), (reconstructed, "Griffin-Lim 重建")]
    ):
        wav = waveform.squeeze().numpy()
        # 截断到最小长度（iSTFT 无法保证重建长度与原音频完全一致）
        min_len = min(wav.shape[-1], original.shape[-1], reconstructed.shape[-1])
        wav = wav[:min_len]
        t_use = t[:min_len]

        axes[row, 0].plot(t_use, wav, linewidth=0.5)
        axes[row, 0].set_title(f"{label}波形")
        axes[row, 0].set_xlabel("时间 (s)")
        axes[row, 0].set_ylabel("幅度")

        # 频谱对比
        spec = np.abs(np.fft.rfft(wav))
        freq = np.fft.rfftfreq(len(wav), 1 / sr)
        axes[row, 1].plot(freq[:500], 20 * np.log10(spec[:500] + 1e-10), linewidth=0.7)
        axes[row, 1].set_title(f"{label}频率分布")
        axes[row, 1].set_xlabel("频率 (Hz)")
        axes[row, 1].set_ylabel("幅度 (dB)")

    plt.tight_layout()
    save_path = Path(output_dir) / "reconstruction_comparison.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  重建对比已保存: {save_path}")


# ============================================================
# 6. 参数扫描实验
# ============================================================

def run_parameter_sweep(
    waveform: torch.Tensor,
    sr: int,
    output_dir: str,
) -> None:
    """分析不同 FFT 参数、Mel bin 数对重建质量的影响。"""
    print("\n[实验] 参数扫描：不同 n_fft 对重建质量的影响")
    print("-" * 50)

    # 不同 n_fft
    n_fft_list = [256, 512, 1024, 2048]

    # 不同 n_mels
    n_mels_list = [20, 40, 80, 128]

    # 不同 Griffin-Lim 迭代次数
    iter_list = [5, 15, 30, 60]

    results = {}

    # --- 实验 1: n_fft 变化 ---
    print("\n  n_fft  | hop_len | 重建 SNR (dB)")
    print("  ------|---------|-------------")
    for n_fft in n_fft_list:
        hop_len = n_fft // 4  # 保持 75% 重叠，iSTFT 需要 > 0 重叠
        stft = compute_stft(waveform, n_fft=n_fft, hop_length=hop_len)
        mag = amplitude_from_stft(stft)
        recon = griffin_lim(mag.squeeze(0), n_fft=n_fft, hop_length=hop_len, n_iter=30)
        snr = compute_snr(waveform, recon)
        results[f"n_fft={n_fft}"] = snr
        print(f"  {n_fft:5d}  |  {hop_len:5d}  |  {snr:.2f}")

    # --- 实验 2: Mel bin 数变化 ---
    print("\n  n_mels | 重建 SNR (dB)")
    print("  -------|-------------")
    n_fft = 1024
    hop = n_fft // 4
    for n_mels in n_mels_list:
        mel_filters = create_mel_filterbank(n_mels, n_fft, sr)
        stft = compute_stft(waveform, n_fft=n_fft, hop_length=hop)
        power_spec = spectrogram_from_stft(stft)
        mel_spec = compute_mel_spectrogram(power_spec.squeeze(0), mel_filters)
        # Mel → Linear (伪逆)
        mel_pinv = torch.linalg.pinv(mel_filters)
        spec_approx = torch.matmul(mel_pinv, mel_spec)
        mag_approx = torch.sqrt(torch.clamp(spec_approx, min=0))
        recon = griffin_lim(mag_approx, n_fft=n_fft, hop_length=hop, n_iter=30)
        snr = compute_snr(waveform, recon)
        results[f"n_mels={n_mels}"] = snr
        print(f"  {n_mels:6d} |  {snr:.2f}")

    # --- 实验 3: Griffin-Lim 迭代次数 ---
    print("\n  迭代次数 | 重建 SNR (dB)")
    print("  ---------|-------------")
    n_fft = 1024
    hop = n_fft // 4
    stft = compute_stft(waveform, n_fft=n_fft, hop_length=hop)
    mag = amplitude_from_stft(stft)
    for n_iter in iter_list:
        recon = griffin_lim(mag.squeeze(0), n_fft=n_fft, hop_length=hop, n_iter=n_iter)
        snr = compute_snr(waveform, recon)
        results[f"iter={n_iter}"] = snr
        print(f"  {n_iter:8d} |  {snr:.2f}")

    # 保存结果
    results_path = Path(output_dir) / "parameter_sweep_results.txt"
    with open(results_path, "w") as f:
        for key, val in results.items():
            f.write(f"{key}: {val:.4f} dB\n")
    print(f"\n  参数扫描结果已保存: {results_path}")


def compute_snr(original: torch.Tensor, reconstructed: torch.Tensor) -> float:
    """计算信噪比 (dB)。"""
    orig = original.squeeze()
    recon = reconstructed.squeeze()
    min_len = min(orig.shape[-1], recon.shape[-1])
    orig = orig[:min_len]
    recon = recon[:min_len]
    noise = orig - recon
    snr = 10 * torch.log10((orig ** 2).sum() / (noise ** 2).sum() + 1e-10)
    return snr.item()


# ============================================================
# 7. 主流程
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="阶段 1: Mel 频谱分析器")
    parser.add_argument("audio", nargs="?", default=None,
                        help="输入 wav 文件路径 (不提供则使用 torchaudio 内置测试音频)")
    parser.add_argument("--n-fft", type=int, default=1024, help="FFT 点数 (默认 1024)")
    parser.add_argument("--hop-length", type=int, default=256, help="帧移 (默认 256)")
    parser.add_argument("--n-mels", type=int, default=80, help="Mel 滤波器数量 (默认 80)")
    parser.add_argument("--griffin-lim-iters", type=int, default=60,
                        help="Griffin-Lim 迭代次数 (默认 60)")
    parser.add_argument("--output-dir", type=str,
                        default=str(Path(__file__).parent / "output"),
                        help="输出目录 (默认 ./output)")
    parser.add_argument("--sweep", action="store_true", help="运行参数扫描实验")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- 加载音频 ---
    if args.audio:
        print(f"加载音频: {args.audio}")
        waveform, sr = load_audio(args.audio)
    else:
        print("生成测试音频 (扫频信号 + 谐波)")
        sr = 22050
        duration = 3.0
        t = torch.linspace(0, duration, int(sr * duration))
        # 从 200Hz 到 4000Hz 的扫频信号
        freq_sweep = 200 + 2000 * (t / duration)
        tone1 = torch.sin(2 * torch.pi * freq_sweep * t)
        # 加两个谐波分量让频谱更丰富
        tone2 = 0.5 * torch.sin(2 * torch.pi * 2 * freq_sweep * t)
        tone3 = 0.25 * torch.sin(2 * torch.pi * 3 * freq_sweep * t)
        waveform = (tone1 + tone2 + tone3).unsqueeze(0).float()
        # 归一化
        waveform = waveform / waveform.abs().max()

    print_audio_info(waveform, sr)

    # --- STFT ---
    print(f"\n计算 STFT (n_fft={args.n_fft}, hop_length={args.hop_length})...")
    stft = compute_stft(waveform, n_fft=args.n_fft, hop_length=args.hop_length)
    stft_mag = amplitude_from_stft(stft)
    print(f"  STFT 形状: {stft.shape} (freq_bins x time_frames)")

    # --- Mel 滤波器组 ---
    print(f"\n创建 Mel 滤波器组 (n_mels={args.n_mels})...")
    mel_filters = create_mel_filterbank(args.n_mels, args.n_fft, sr)
    power_spec = spectrogram_from_stft(stft)
    mel_spec = compute_mel_spectrogram(power_spec.squeeze(0), mel_filters).unsqueeze(0)
    print(f"  Mel 频谱形状: {mel_spec.shape}")

    # --- Griffin-Lim 重建 ---
    print(f"\n运行 Griffin-Lim 重建 (iters={args.griffin_lim_iters})...")
    reconstructed = griffin_lim(
        stft_mag.squeeze(0),
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        n_iter=args.griffin_lim_iters,
    )
    snr = compute_snr(waveform, reconstructed)
    print(f"  重建 SNR: {snr:.2f} dB")

    # --- 保存重建音频 ---
    recon_path = output_dir / "reconstructed.wav"
    sf.write(str(recon_path), reconstructed.squeeze().numpy(), sr)
    print(f"  重建音频已保存: {recon_path}")

    # --- 可视化 ---
    print("\n生成可视化...")
    plot_analysis(waveform, sr, stft_mag, mel_spec,
                  args.n_fft, args.hop_length, args.n_mels,
                  mel_filters, str(output_dir))
    plot_reconstruction_comparison(waveform, reconstructed, sr, str(output_dir))

    # --- 参数扫描 ---
    if args.sweep:
        run_parameter_sweep(waveform, sr, str(output_dir))

    print("\n✓ 阶段 1 实践项目完成！")
    print(f"  输出目录: {output_dir.absolute()}")
    print(f"\n可尝试的操作:")
    print(f"  - 更换输入音频: python mel_analyzer.py /path/to/audio.wav")
    print(f"  - 调整 FFT 参数: python mel_analyzer.py --n-fft 2048 --hop-length 512")
    print(f"  - 调整 Mel 参数: python mel_analyzer.py --n-mels 128")
    print(f"  - 参数扫描实验: python mel_analyzer.py --sweep")


if __name__ == "__main__":
    main()
