"""
对比实验: Griffin-Lim vs HiFi-GAN
==================================
同一段音频，分别用两种方法从 Mel 频谱重建，对比 SNR 和可视化结果。

用法:
  python compare.py --input ../stage1_speech_signal/prompt_audio.wav
  python compare.py --input speech.wav --checkpoint output/g_best.pt
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
import torchaudio
import torchaudio.functional as F

# 从阶段 1 导入 Griffin-Lim
sys.path.insert(0, str(Path(__file__).parent.parent))
from stage1_speech_signal.mel_analyzer import griffin_lim, compute_snr

from model import HiFiGANGenerator


# ============================================================
# Mel 频谱提取
# ============================================================

class MelSpectrogram(torch.nn.Module):
    """固定参数的 Mel 频谱提取器 (与阶段 1 一致)。"""

    def __init__(self, sr=22050, n_fft=1024, n_mels=80, hop_length=256, f_min=0, f_max=8000):
        super().__init__()
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr,
            n_fft=n_fft,
            n_mels=n_mels,
            hop_length=hop_length,
            f_min=f_min,
            f_max=f_max,
            power=1.0,
        )
        self.n_fft = n_fft
        self.hop_length = hop_length

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        mel = self.mel_spec(wav)
        mel = torch.log(1 + mel)
        return mel


# ============================================================
# 对比实验
# ============================================================

def compare(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- 加载音频 ---
    print(f"加载音频: {args.input}")
    wav, sr = sf.read(args.input)
    if wav.ndim > 1:
        wav = wav.mean(-1)
    wav = torch.from_numpy(wav).float().unsqueeze(0)  # (1, T)
    if sr != args.sr:
        wav = F.resample(wav, sr, args.sr)
        sr = args.sr

    print(f"  时长: {wav.shape[-1] / sr:.2f}s, 采样率: {sr}Hz")

    # --- Mel 提取 ---
    mel_fn = MelSpectrogram(sr=args.sr, n_fft=args.n_fft, n_mels=args.n_mels,
                            hop_length=args.hop_length)
    mel = mel_fn(wav)  # (1, n_mels, T_mel)

    # ========== 方法 1: Griffin-Lim ==========
    print("\n[1/2] Griffin-Lim 重建...")
    # 从 Mel 频谱恢复线性幅度谱 (伪逆)
    from stage1_speech_signal.mel_analyzer import create_mel_filterbank
    mel_filters = create_mel_filterbank(args.n_mels, args.n_fft, sr)
    mel_pinv = torch.linalg.pinv(mel_filters)
    spec_approx = torch.matmul(mel_pinv, mel.squeeze(0))
    mag_approx = torch.sqrt(torch.clamp(spec_approx, min=0))

    gl_wav = griffin_lim(
        mag_approx,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        n_iter=args.gl_iters,
    )
    gl_snr = compute_snr(wav, gl_wav)
    print(f"  Griffin-Lim SNR: {gl_snr:.2f} dB")

    # 保存
    sf.write(str(output_dir / "gl_reconstructed.wav"),
             gl_wav.squeeze().numpy(), sr)

    # ========== 方法 2: HiFi-GAN ==========
    print("\n[2/2] HiFi-GAN 重建...")
    generator = HiFiGANGenerator(in_channels=args.n_mels).to(device)
    if Path(args.checkpoint).exists():
        state_dict = torch.load(args.checkpoint, map_location=device, weights_only=True)
        generator.load_state_dict(state_dict)
        print(f"  加载权重: {args.checkpoint}")
    else:
        print(f"  ⚠ 权重文件不存在: {args.checkpoint}")
        print(f"  使用未训练的随机初始化模型 (结果无意义)")

    generator.eval()
    with torch.no_grad():
        hifi_wav = generator(mel.unsqueeze(0).to(device)).cpu()

    # 对齐长度
    min_len = min(wav.shape[-1], hifi_wav.shape[-1])
    hifi_snr = compute_snr(wav[:, :min_len], hifi_wav[:, :min_len])
    print(f"  HiFi-GAN SNR: {hifi_snr:.2f} dB")

    # 保存
    sf.write(str(output_dir / "hifi_reconstructed.wav"),
             hifi_wav.squeeze().numpy(), sr)

    # ========== 结果汇总 ==========
    print("\n" + "=" * 50)
    print("对比结果")
    print("=" * 50)
    print(f"  Griffin-Lim (iters={args.gl_iters}):  {gl_snr:.2f} dB")
    print(f"  HiFi-GAN:                           {hifi_snr:.2f} dB")
    print(f"  差异:                                {hifi_snr - gl_snr:+.2f} dB")
    print("=" * 50)

    # ========== 可视化 ==========
    print("\n生成对比可视化...")
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))

    t = np.arange(min_len) / sr

    # 行 1: 波形对比
    axes[0, 0].plot(t, wav[0, :min_len].numpy(), linewidth=0.5)
    axes[0, 0].set_title("原始波形")
    axes[0, 0].set_xlabel("时间 (s)")

    axes[0, 1].plot(t, hifi_wav[0, :min_len].numpy(), linewidth=0.5, color="C1")
    axes[0, 1].set_title(f"HiFi-GAN 重建 (SNR={hifi_snr:.1f}dB)")
    axes[0, 1].set_xlabel("时间 (s)")

    # 行 2: 频谱对比
    def plot_spectrogram(ax, wav_tensor, title):
        spec = torch.stft(
            wav_tensor.unsqueeze(0),
            n_fft=args.n_fft, hop_length=args.hop_length,
            window=torch.hann_window(args.n_fft),
            return_complex=True,
        )
        spec_db = 20 * torch.log10(spec.abs().squeeze().numpy() + 1e-6)
        ax.imshow(spec_db, origin="lower", aspect="auto", cmap="magma",
                  extent=[0, wav_tensor.shape[-1] / sr, 0, sr / 2])
        ax.set_title(title)
        ax.set_xlabel("时间 (s)")
        ax.set_ylabel("频率 (Hz)")

    plot_spectrogram(axes[1, 0], wav[0, :min_len], "原始频谱")
    plot_spectrogram(axes[1, 1], hifi_wav[0, :min_len], "HiFi-GAN 频谱")

    # 行 3: 误差分析
    error = wav[0, :min_len] - hifi_wav[0, :min_len]
    axes[2, 0].plot(t, error.numpy(), linewidth=0.3, color="red")
    axes[2, 0].set_title(f"重建误差 (RMSE={error.pow(2).mean().sqrt():.4f})")
    axes[2, 0].set_xlabel("时间 (s)")

    # 误差频谱
    error_spec = torch.stft(
        error.unsqueeze(0),
        n_fft=args.n_fft, hop_length=args.hop_length,
        window=torch.hann_window(args.n_fft),
        return_complex=True,
    )
    error_db = 20 * torch.log10(error_spec.abs().squeeze().numpy() + 1e-10)
    axes[2, 1].imshow(error_db, origin="lower", aspect="auto", cmap="magma",
                      extent=[0, min_len / sr, 0, sr / 2])
    axes[2, 1].set_title("误差频谱")
    axes[2, 1].set_xlabel("时间 (s)")
    axes[2, 1].set_ylabel("频率 (Hz)")

    plt.tight_layout()
    save_path = output_dir / "comparison.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  对比图已保存: {save_path}")

    return gl_snr, hifi_snr


# ============================================================
# 主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Griffin-Lim vs HiFi-GAN 对比")
    parser.add_argument("--input", type=str, required=True, help="输入 wav 文件")
    parser.add_argument("--checkpoint", type=str,
                        default=str(Path(__file__).parent / "output" / "g_best.pt"),
                        help="HiFi-GAN 权重路径")
    parser.add_argument("--output-dir", type=str,
                        default=str(Path(__file__).parent / "output"),
                        help="输出目录")
    parser.add_argument("--sr", type=int, default=22050, help="采样率")
    parser.add_argument("--n-fft", type=int, default=1024, help="FFT 点数")
    parser.add_argument("--n-mels", type=int, default=80, help="Mel 通道数")
    parser.add_argument("--hop-length", type=int, default=256, help="帧移")
    parser.add_argument("--gl-iters", type=int, default=60, help="Griffin-Lim 迭代次数")
    args = parser.parse_args()

    compare(args)


if __name__ == "__main__":
    main()