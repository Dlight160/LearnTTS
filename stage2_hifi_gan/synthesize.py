"""
HiFi-GAN 合成脚本
==================
从训练好的 checkpoint 加载生成器，输入 Mel 频谱合成波形。
"""

import argparse
from pathlib import Path

import torch
import torchaudio
import numpy as np
import soundfile as sf

from model import HiFiGANGenerator
from train import MelSpectrogram


@torch.inference_mode()
def synthesize(mel: torch.Tensor, generator: HiFiGANGenerator, device: torch.device) -> torch.Tensor:
    """Mel (B, n_mels, T_mel) → 波形 (B, 1, T_wav)。"""
    return generator(mel.to(device)).cpu()


def main():
    parser = argparse.ArgumentParser(description="HiFi-GAN 合成")
    parser.add_argument("--wav", type=str, default=None, help="输入 wav 路径（提取 Mel 后重建）")
    parser.add_argument("--test", action="store_true", help="用随机 Mel 测试")
    parser.add_argument("--checkpoint", type=str, default=None, help="生成器 checkpoint（默认用 g_best.pt）")
    parser.add_argument("--output-dir", type=str, default=None, help="输出目录（默认 model 同级的 output）")
    parser.add_argument("--plot", action="store_true", help="保存 Mel 对比图")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    # --- 模型 ---
    generator = HiFiGANGenerator().to(device)
    generator.eval()

    output_dir = Path(args.output_dir or (Path(__file__).parent / "output"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- 加载 checkpoint ---
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
    else:
        ckpt_path = output_dir / "g_best.pt"
        if not ckpt_path.exists():
            ckpt_path = output_dir / "g_final.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"找不到 checkpoint: {ckpt_path}")

    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    generator.load_state_dict(state)
    step_tag = ckpt_path.stem  # g_best, g_final, g_010000
    print(f"加载 checkpoint: {ckpt_path}")

    # --- Mel 提取器 ---
    mel_fn = MelSpectrogram(sr=22050, n_fft=1024, n_mels=80, hop_length=256)

    if args.test:
        # 测试模式：用类似训练集的扫频信号生成 Mel → 重建
        print("测试模式：生成扫频信号 Mel → 重建")
        sr = 22050
        t = torch.linspace(0, 2, 2 * sr)
        freq = 200 + 300 * t  # 从 200Hz 扫到 800Hz
        test_wav = torch.sin(2 * torch.pi * freq * t)
        test_wav += 0.5 * torch.sin(2 * torch.pi * 2 * freq * t)
        test_wav = test_wav / test_wav.abs().max()
        test_wav = test_wav.unsqueeze(0)  # (1, T)

        mel = mel_fn(test_wav)  # (1, 80, T_mel)
        wav_out = synthesize(mel, generator, device)

        out_path = output_dir / f"synth_test_{step_tag}.wav"
        sf.write(str(out_path), wav_out.squeeze().numpy(), sr)
        print(f"合成输出: {out_path}")

        if args.plot:
            _save_plot(mel, mel_fn(wav_out), output_dir / f"synth_test_{step_tag}.png")

    elif args.wav:
        wav_path = Path(args.wav)
        print(f"输入 wav: {wav_path}")

        wav, sr = sf.read(str(wav_path))
        wav = torch.from_numpy(wav).float()
        if wav.ndim > 1:
            wav = wav.mean(-1)
        if sr != 22050:
            wav = torchaudio.functional.resample(wav, sr, 22050)
        wav = wav.unsqueeze(0)  # (1, T)

        mel = mel_fn(wav)
        wav_out = synthesize(mel, generator, device)

        out_path = output_dir / f"synth_{wav_path.stem}_{step_tag}.wav"
        sf.write(str(out_path), wav_out.squeeze().numpy(), 22050)
        print(f"合成输出: {out_path}")

        # 也保存原始 wav 的重采样版本，方便对比
        ref_path = output_dir / f"ref_{wav_path.stem}.wav"
        sf.write(str(ref_path), wav.squeeze().numpy(), 22050)
        print(f"参考音频: {ref_path}")

        if args.plot:
            _save_plot(mel, mel_fn(wav_out), output_dir / f"synth_{wav_path.stem}_{step_tag}.png")
    else:
        parser.print_help()
        print("\n请指定 --wav 或 --test")


def _save_plot(mel_real: torch.Tensor, mel_fake: torch.Tensor, save_path: Path):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    axes[0].imshow(mel_real.squeeze().numpy(), origin="lower", aspect="auto", cmap="magma")
    axes[0].set_title("Input Mel")
    axes[1].imshow(mel_fake.squeeze().numpy(), origin="lower", aspect="auto", cmap="magma")
    axes[1].set_title("Reconstructed Mel")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Mel 对比图保存: {save_path}")


if __name__ == "__main__":
    main()