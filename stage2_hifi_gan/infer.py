"""
HiFi-GAN 推理脚本
==================
加载预训练生成器权重，从 Mel 频谱重建波形。

用法:
  python infer.py --input speech.wav
  python infer.py --input speech.wav --checkpoint output/g_best.pt
"""

import argparse
from pathlib import Path

import torch
import torchaudio
import torchaudio.functional as F
import soundfile as sf
import numpy as np

from model import HiFiGANGenerator


# ============================================================
# Mel 频谱提取
# ============================================================

class MelSpectrogram(torch.nn.Module):
    """固定参数的 Mel 频谱提取器 (与训练一致)。"""

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

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        mel = self.mel_spec(wav)
        mel = torch.log(1 + mel)
        return mel


# ============================================================
# 推理
# ============================================================

def compute_snr(original: torch.Tensor, reconstructed: torch.Tensor) -> float:
    """信噪比 (dB)。"""
    orig = original.squeeze()
    recon = reconstructed.squeeze()
    min_len = min(orig.shape[-1], recon.shape[-1])
    orig = orig[:min_len]
    recon = recon[:min_len]
    noise = orig - recon
    snr = 10 * torch.log10((orig ** 2).sum() / (noise ** 2).sum() + 1e-10)
    return snr.item()


def infer(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    # --- 加载模型 ---
    print(f"加载生成器: {args.checkpoint}")
    generator = HiFiGANGenerator(in_channels=args.n_mels).to(device)
    state_dict = torch.load(args.checkpoint, map_location=device, weights_only=True)
    generator.load_state_dict(state_dict)
    generator.eval()
    print(f"  参数量: {sum(p.numel() for p in generator.parameters()):,}")

    # --- Mel 提取器 ---
    mel_fn = MelSpectrogram(sr=args.sr, n_fft=args.n_fft, n_mels=args.n_mels,
                            hop_length=args.hop_length).to(device)

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

    # --- 提取 Mel ---
    mel = mel_fn(wav).unsqueeze(0).to(device)  # (1, n_mels, T_mel)
    print(f"  Mel 形状: {mel.shape}")

    # --- 生成波形 ---
    with torch.no_grad():
        fake_wav = generator(mel).cpu()

    # --- 计算 SNR ---
    snr = compute_snr(wav, fake_wav.squeeze(0))
    print(f"  重建 SNR: {snr:.2f} dB")

    # --- 保存 ---
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "hifi_reconstructed.wav"
    sf.write(str(output_path), fake_wav.squeeze().numpy(), args.sr)
    print(f"  重建音频已保存: {output_path}")

    return fake_wav


# ============================================================
# 主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="HiFi-GAN 推理")
    parser.add_argument("--input", type=str, required=True, help="输入 wav 文件")
    parser.add_argument("--checkpoint", type=str,
                        default=str(Path(__file__).parent / "output" / "g_best.pt"),
                        help="生成器权重路径")
    parser.add_argument("--output-dir", type=str,
                        default=str(Path(__file__).parent / "output"),
                        help="输出目录")
    parser.add_argument("--sr", type=int, default=22050, help="采样率")
    parser.add_argument("--n-fft", type=int, default=1024, help="FFT 点数")
    parser.add_argument("--n-mels", type=int, default=80, help="Mel 通道数")
    parser.add_argument("--hop-length", type=int, default=256, help="帧移")
    args = parser.parse_args()

    infer(args)


if __name__ == "__main__":
    main()