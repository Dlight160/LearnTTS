"""
HiFi-GAN 训练脚本
==================
支持完整训练和 --fast 快速验证模式。

损失函数:
  - Mel-spectrogram Loss: L1(Mel(real), Mel(fake))
  - GAN Loss: LSGAN (MSE)
  - Feature Matching Loss: 判别器中间层特征的 L1
"""

import argparse
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torchaudio.datasets import LJSPEECH

import soundfile as sf
import numpy as np

import torchaudio
from model import HiFiGANGenerator, MultiScaleDiscriminator, MultiPeriodDiscriminator


# ============================================================
# Mel 频谱提取 (与阶段 1 参数一致)
# ============================================================

class MelSpectrogram(nn.Module):
    """固定参数的 Mel 频谱提取器。"""

    def __init__(self, sr=22050, n_fft=1024, n_mels=80, hop_length=256, f_min=0, f_max=8000):
        super().__init__()
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr,
            n_fft=n_fft,
            n_mels=n_mels,
            hop_length=hop_length,
            f_min=f_min,
            f_max=f_max,
            power=1.0,  # 幅度谱
        )
        self.hop_length = hop_length

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """wav: (B, T) → mel: (B, n_mels, T')"""
        mel = self.mel_spec(wav)
        # 对数压缩: log(1 + mel) 避免 log(0)
        mel = torch.log(1 + mel)
        return mel


# ============================================================
# 损失函数
# ============================================================

def mel_loss(real_mel: torch.Tensor, fake_wav: torch.Tensor, mel_fn: MelSpectrogram) -> torch.Tensor:
    """Mel 频谱 L1 损失 (自动对齐时间帧)。"""
    fake_mel = mel_fn(fake_wav.squeeze(1))
    min_len = min(real_mel.shape[-1], fake_mel.shape[-1])
    return F.l1_loss(real_mel[..., :min_len], fake_mel[..., :min_len])


def discriminator_loss(real_logits: list[torch.Tensor], fake_logits: list[torch.Tensor]) -> torch.Tensor:
    """LSGAN 判别器损失: MSE(D(real), 1) + MSE(D(fake), 0)。"""
    loss = 0
    for real, fake in zip(real_logits, fake_logits):
        loss += F.mse_loss(real, torch.ones_like(real))
        loss += F.mse_loss(fake, torch.zeros_like(fake))
    return loss


def generator_loss(fake_logits: list[torch.Tensor]) -> torch.Tensor:
    """LSGAN 生成器损失: MSE(D(fake), 1)。"""
    loss = 0
    for logit in fake_logits:
        loss += F.mse_loss(logit, torch.ones_like(logit))
    return loss


def feature_matching_loss(
    real_feats: list[list[torch.Tensor]],
    fake_feats: list[list[torch.Tensor]],
) -> torch.Tensor:
    """特征匹配损失: 判别器各层特征的 L1 (自动对齐时间维度)。"""
    loss = 0
    n_layers = 0
    for real_f, fake_f in zip(real_feats, fake_feats):
        for r, f in zip(real_f, fake_f):
            min_len = min(r.shape[-1], f.shape[-1])
            loss += F.l1_loss(r[..., :min_len], f[..., :min_len].detach())
            n_layers += 1
    return loss / max(n_layers, 1)


# ============================================================
# 数据集
# ============================================================

class AudioDataset(Dataset):
    """通用的音频数据集包装器。

    支持 LJSpeech 和任意 wav 文件目录。
    """

    def __init__(self, root: str | None = None, sr: int = 22050, segment_len: int = 8192,
                 synthetic: bool = False):
        self.sr = sr
        self.segment_len = segment_len
        self.synthetic = synthetic
        self.files = []

        if synthetic:
            self.len = 32
            print(f"  [合成数据] {self.len} 条随机正弦波")
        elif root and os.path.isfile(root):
            self.files = [root]
        elif root and os.path.isdir(root):
            self.files = sorted(Path(root).glob("*.wav"))
        else:
            try:
                self.dataset = LJSPEECH(root="./data", download=True)
                self.files = None
                print(f"  加载 LJSpeech: {len(self.dataset)} 条")
            except Exception as e:
                raise ValueError(f"无法加载音频 (请指定 --data-root): {e}")

    def __len__(self) -> int:
        if self.synthetic:
            return self.len
        if self.files is not None:
            return len(self.files)
        return len(self.dataset)

    def __getitem__(self, idx: int) -> torch.Tensor:
        if self.synthetic:
            duration = self.segment_len / self.sr
            t = torch.linspace(0, duration, self.segment_len)
            freq = 100 + 400 * (idx / self.len)
            wav = torch.sin(2 * torch.pi * freq * t)
            wav += 0.5 * torch.sin(2 * torch.pi * 2 * freq * t)
            wav += 0.25 * torch.sin(2 * torch.pi * 3 * freq * t)
            wav = wav / wav.abs().max()
            return wav.unsqueeze(0).float()

        if self.files is not None:
            wav, sr = sf.read(str(self.files[idx]))
            wav = torch.from_numpy(wav).float()
            if wav.ndim > 1:
                wav = wav.mean(-1)
            if sr != self.sr:
                wav = torchaudio.functional.resample(wav, sr, self.sr)
        else:
            wav, sr, _, _ = self.dataset[idx]  # (C, T)
            wav = wav.mean(dim=0)  # 混缩为单声道 (T,)
            if sr != self.sr:
                wav = torchaudio.functional.resample(wav, sr, self.sr)

        # 随机裁剪到固定长度
        if wav.shape[-1] > self.segment_len:
            start = torch.randint(0, wav.shape[-1] - self.segment_len, (1,)).item()
            wav = wav[start:start + self.segment_len]
        else:
            # 短于 segment_len 则补零
            wav = F.pad(wav, (0, self.segment_len - wav.shape[-1]))

        return wav.unsqueeze(0)  # (1, T)


# ============================================================
# 训练循环
# ============================================================

def train(rank, args, world_size=1):
    is_ddp = world_size > 1

    if is_ddp:
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
        dist.init_process_group(backend="nccl", init_method="env://",
                                rank=rank, world_size=world_size)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if rank == 0:
        print(f"设备: {device}" + (f" (world_size={world_size})" if is_ddp else ""))
        print(f"输出目录: {args.output_dir}")

    output_dir = Path(args.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)

    # --- 数据 ---
    if rank == 0:
        print("加载数据...")
    dataset = AudioDataset(args.data_root, sr=args.sr, segment_len=args.segment_len,
                           synthetic=args.fast)
    sampler = DistributedSampler(dataset) if is_ddp else None
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=sampler is None,
                        sampler=sampler, num_workers=0)

    # --- 模型 ---
    if rank == 0:
        print("初始化模型...")
    generator = HiFiGANGenerator(in_channels=args.n_mels).to(device)
    mpd = MultiPeriodDiscriminator().to(device)
    msd = MultiScaleDiscriminator().to(device)

    if is_ddp:
        generator = DDP(generator, device_ids=[rank])
        mpd = DDP(mpd, device_ids=[rank])
        msd = DDP(msd, device_ids=[rank])

    # --- Mel 提取器 ---
    mel_fn = MelSpectrogram(sr=args.sr, n_fft=args.n_fft, n_mels=args.n_mels,
                            hop_length=args.hop_length).to(device)

    # --- 优化器 ---
    optim_g = torch.optim.AdamW(generator.parameters(), lr=args.lr, betas=(0.8, 0.99))
    optim_d = torch.optim.AdamW(
        list(msd.parameters()) + list(mpd.parameters()),
        lr=args.lr, betas=(0.8, 0.99),
    )

    # --- 训练 ---
    if rank == 0:
        print(f"开始训练 (max_steps={args.max_steps})...")
        print(f"  batch_size={args.batch_size}, segment_len={args.segment_len}")
        print(f"  lr={args.lr}, 总参数量={sum(p.numel() for p in generator.parameters()):,}")
        print()

    def unwrap(m):
        return m.module if is_ddp else m

    step = 0
    epoch = 0
    best_loss = float("inf")

    while step < args.max_steps:
        epoch += 1
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch_idx, wav in enumerate(loader):
            if step >= args.max_steps:
                break

            wav = wav.to(device)  # (B, 1, T)
            B = wav.shape[0]

            # --- 提取 Mel ---
            mel = mel_fn(wav.squeeze(1))  # (B, n_mels, T_mel)

            # ========== 训练判别器 ==========
            for p in msd.parameters():
                p.requires_grad = True
            for p in mpd.parameters():
                p.requires_grad = True

            optim_d.zero_grad()

            fake_wav = generator(mel)

            # MPD
            mpd_real = mpd(wav)
            mpd_fake = mpd(fake_wav.detach())
            loss_mpd = discriminator_loss(
                [r[1] for r in mpd_real],
                [f[1] for f in mpd_fake],
            )

            # MSD
            msd_real = msd(wav)
            msd_fake = msd(fake_wav.detach())
            loss_msd = discriminator_loss(
                [r[1] for r in msd_real],
                [f[1] for f in msd_fake],
            )

            loss_d = loss_mpd + loss_msd
            loss_d.backward()
            optim_d.step()

            # ========== 训练生成器 ==========
            for p in msd.parameters():
                p.requires_grad = False
            for p in mpd.parameters():
                p.requires_grad = False

            optim_g.zero_grad()

            fake_wav = generator(mel)

            # GAN loss
            mpd_fake = mpd(fake_wav)
            msd_fake = msd(fake_wav)
            loss_gan = generator_loss([f[1] for f in mpd_fake]) + \
                       generator_loss([f[1] for f in msd_fake])

            # Mel loss
            loss_mel = mel_loss(mel, fake_wav, mel_fn)

            # Feature matching loss
            mpd_real = mpd(wav)
            msd_real = msd(wav)
            loss_fm = feature_matching_loss(
                [f[0] for f in mpd_real],
                [f[0] for f in mpd_fake],
            ) + feature_matching_loss(
                [f[0] for f in msd_real],
                [f[0] for f in msd_fake],
            )

            loss_g = loss_gan + args.lambda_mel * loss_mel + args.lambda_fm * loss_fm
            loss_g.backward()
            optim_g.step()

            # --- 日志 ---
            if rank == 0 and step % args.log_interval == 0:
                print(
                    f"  Step {step:6d}/{args.max_steps} | "
                    f"D: {loss_d.item():.4f} | "
                    f"G: {loss_g.item():.4f} | "
                    f"Mel: {loss_mel.item():.4f} | "
                    f"FM: {loss_fm.item():.4f} | "
                    f"GAN: {loss_gan.item():.4f}"
                )

            # --- 保存检查点 ---
            if rank == 0 and step % args.save_interval == 0 and step > 0:
                ckpt_path = output_dir / f"g_{step:06d}.pt"
                torch.save(unwrap(generator).state_dict(), ckpt_path)
                if loss_mel.item() < best_loss:
                    best_loss = loss_mel.item()
                    best_path = output_dir / "g_best.pt"
                    torch.save(unwrap(generator).state_dict(), best_path)
                    print(f"  ✓ 最佳模型已保存: {best_path}")

            step += 1

    # 最终保存
    if rank == 0:
        torch.save(unwrap(generator).state_dict(), output_dir / "g_final.pt")
        print(f"\n训练完成! 模型已保存到 {output_dir}")
        print(f"  最终 Mel Loss: {loss_mel.item():.4f}")

    if is_ddp:
        torch.cuda.synchronize(device)
        try:
            dist.destroy_process_group()
        except Exception as e:
            if rank == 0:
                print(f"  [DDP] 清理 NCCL 时发生非致命错误: {e}")


# ============================================================
# 主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="HiFi-GAN 训练")
    parser.add_argument("--data-root", type=str, default=None,
                        help="音频文件或目录 (默认: 自动下载 LJSpeech)")
    parser.add_argument("--output-dir", type=str,
                        default=str(Path(__file__).parent / "output"),
                        help="输出目录")
    parser.add_argument("--sr", type=int, default=22050, help="采样率")
    parser.add_argument("--n-fft", type=int, default=1024, help="FFT 点数")
    parser.add_argument("--n-mels", type=int, default=80, help="Mel 通道数")
    parser.add_argument("--hop-length", type=int, default=256, help="帧移")
    parser.add_argument("--segment-len", type=int, default=8192, help="训练音频片段长度")
    parser.add_argument("--batch-size", type=int, default=8, help="批次大小")
    parser.add_argument("--lr", type=float, default=2e-4, help="学习率")
    parser.add_argument("--max-steps", type=int, default=500000, help="最大训练步数")
    parser.add_argument("--lambda-mel", type=float, default=45.0, help="Mel loss 权重")
    parser.add_argument("--lambda-fm", type=float, default=10.0, help="Feature matching loss 权重")
    parser.add_argument("--log-interval", type=int, default=100, help="日志间隔")
    parser.add_argument("--save-interval", type=int, default=10000, help="保存间隔")
    parser.add_argument("--fast", action="store_true", help="快速验证模式 (100 步)")
    parser.add_argument("--multi-gpu", action="store_true", help="使用所有可用 GPU (DDP)")
    args = parser.parse_args()

    if args.fast:
        args.max_steps = 100
        args.log_interval = 10
        args.batch_size = 4
        args.segment_len = 4096
        print("[快速验证模式]")

    if args.multi_gpu:
        world_size = torch.cuda.device_count()
        if world_size < 2:
            print("只有 1 张卡可用，使用单卡模式")
            train(0, args)
        else:
            os.environ.setdefault("MASTER_ADDR", "localhost")
            os.environ.setdefault("MASTER_PORT", "29500")
            mp.spawn(train, args=(args, world_size), nprocs=world_size, join=True)
    else:
        train(0, args)


if __name__ == "__main__":
    main()