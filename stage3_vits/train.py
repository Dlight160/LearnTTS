#!/usr/bin/env python3
"""
VITS 训练脚本
=============
在 LJSpeech-1.1 上完整训练 VITS。

用法:
  # 快速验证 (100 步)
  ./conda_env/bin/python stage3_vits/train.py --fast

  # 完整训练
  ./conda_env/bin/python stage3_vits/train.py --data-root ./data/LJSpeech-1.1 --max-steps 500000

  # 多卡 DDP
  CUDA_VISIBLE_DEVICES=0,1,2,3 ./conda_env/bin/python stage3_vits/train.py --multi-gpu

损失组成:
  - KL Loss: KL(q(z|x) || p(z|c)), VAE 散度
  - Duration Loss: NLL of flow-based duration predictor
  - GAN Loss: LSGAN for MSD + MPD
  - Feature Matching Loss: 判别器中间层 L1
  - Mel Loss: 频谱 L1 (内容保真)
"""

import argparse
import math
import os
import sys
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torchaudio.datasets import LJSPEECH as _LJSPEECH
import soundfile as sf
import numpy as np

# Add parent dir to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from stage3_vits.model import SynthesizerTrn
from stage3_vits.text_symbols import text_to_sequence
from stage3_vits.discriminator import MultiScaleDiscriminator, MultiPeriodDiscriminator


# ============================================================
# Mel 频谱提取
# ============================================================

def get_mel_fn(device='cpu'):
    return torchaudio.transforms.MelSpectrogram(
        sample_rate=22050,
        n_fft=1024,
        n_mels=80,
        hop_length=256,
        f_min=0,
        f_max=8000,
        power=1.0,
    ).to(device)


def wav_to_mel(wav: torch.Tensor, mel_fn) -> torch.Tensor:
    """wav: (T,) → mel: (80, T_mel)"""
    # Ensure 1D
    if wav.dim() > 1:
        wav = wav.squeeze()
    mel = mel_fn(wav)
    # dB scale
    mel = 20 * torch.log10(torch.clamp(mel, min=1e-5))
    return mel


# ============================================================
# 数据集
# ============================================================

class LJSpeechDataset(Dataset):
    """LJSpeech 数据集: 文本 → 音素, 音频 → Mel。

    返回 (phoneme_ids, mel, wav)。
    """

    def __init__(self, root: str, hop_length: int = 256, win_length: int = 1024,
                 sr: int = 22050, segment_len: int = 16384):
        self.root = Path(root)
        self.hop_length = hop_length
        self.segment_len = segment_len
        self.sr = sr
        self.mel_fn = None

        # Load metadata
        self.items = []
        meta_path = self.root / "metadata.csv"
        with open(meta_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('|')
                if len(parts) >= 3:
                    self.items.append({
                        'id': parts[0],
                        'text': parts[2],
                    })

        # 音素缓存: gruut g2p ~1.2s/句, 是训练头号瓶颈。优先读预计算缓存,
        # miss 时回退实时 g2p。用 precompute_phonemes.py 生成。
        self.phoneme_cache = None
        cache_path = self.root / "phonemes_cache.pt"
        if cache_path.exists():
            self.phoneme_cache = torch.load(cache_path, weights_only=False)
            print(f"  LJSpeech: {len(self.items)} 条 (音素缓存 {len(self.phoneme_cache)} 条命中)")
        else:
            print(f"  LJSpeech: {len(self.items)} 条 (⚠ 无音素缓存, 将实时 g2p — "
                  f"请先跑 precompute_phonemes.py)")

    def _get_mel_fn(self, device):
        if self.mel_fn is None:
            self.mel_fn = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.sr,
                n_fft=1024,
                n_mels=80,
                hop_length=self.hop_length,
                f_min=0,
                f_max=8000,
                power=1.0,
            ).to(device)
        return self.mel_fn

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        item = self.items[idx]

        # Audio
        wav_path = self.root / "wavs" / f"{item['id']}.wav"
        wav, sr = sf.read(str(wav_path))
        wav = torch.from_numpy(wav).float()
        if wav.dim() > 1:
            wav = wav.mean(-1)
        if sr != self.sr:
            wav = torchaudio.functional.resample(wav, sr, self.sr)

        # Pad/slice to segment_len
        if wav.shape[-1] < self.segment_len:
            wav = F.pad(wav, (0, self.segment_len - wav.shape[-1]))
        else:
            start = torch.randint(0, wav.shape[-1] - self.segment_len, (1,)).item()
            wav = wav[start:start + self.segment_len]

        # Mel
        mel_fn = self._get_mel_fn(wav.device)
        mel = mel_fn(wav)
        mel = 20 * torch.log10(torch.clamp(mel, min=1e-5))

        # Text → phoneme IDs (优先缓存, miss 回退实时 g2p)
        if self.phoneme_cache is not None and item['id'] in self.phoneme_cache:
            phones = self.phoneme_cache[item['id']]
        else:
            phones = text_to_sequence(item['text'])

        return torch.tensor(phones, dtype=torch.long), mel, wav.unsqueeze(0)


class Collation:
    """批量 collate 函数: padding 变长序列。"""

    @staticmethod
    def collate(batch: list) -> tuple:
        phoneme_ids, mels, wavs = zip(*batch)

        # Phonemes: pad to max len
        text_lens = torch.tensor([p.shape[0] for p in phoneme_ids])
        max_text_len = text_lens.max().item()
        texts_padded = torch.zeros(len(phoneme_ids), max_text_len, dtype=torch.long)
        for i, p in enumerate(phoneme_ids):
            texts_padded[i, :p.shape[0]] = p

        # Mels: pad to max len
        mel_lens = torch.tensor([m.shape[-1] for m in mels])
        max_mel_len = mel_lens.max().item()
        mels_padded = torch.zeros(len(mels), mels[0].shape[0], max_mel_len)
        for i, m in enumerate(mels):
            mels_padded[i, :, :m.shape[-1]] = m

        # Wavs: pad
        wav_lens = torch.tensor([w.shape[-1] for w in wavs])
        max_wav_len = wav_lens.max().item()
        wavs_padded = torch.zeros(len(wavs), 1, max_wav_len)
        for i, w in enumerate(wavs):
            wavs_padded[i, :, :w.shape[-1]] = w

        # Masks
        text_mask = torch.zeros(len(phoneme_ids), 1, max_text_len)
        mel_mask = torch.zeros(len(phoneme_ids), 1, max_mel_len)
        for i in range(len(phoneme_ids)):
            text_mask[i, :, :text_lens[i]] = 1.0
            mel_mask[i, :, :mel_lens[i]] = 1.0

        return texts_padded, text_mask, mels_padded, mel_mask, wavs_padded


class FastDataset(Dataset):
    """快速验证用合成数据。"""

    def __init__(self, n_items=16):
        self.n_items = n_items
        # Short example text
        self.texts = [
            "Printing, in the only sense.",
            "We are concerned with.",
            "The invention of printing.",
            "A similar process was used.",
            "The block books were common.",
            "The art of printing.",
            "This is a test sentence.",
            "Hello world, this is LJSpeech.",
            "She sells sea shells.",
            "The quick brown fox jumps.",
            "How are you doing today?",
            "Machine learning is fun.",
            "Deep learning for speech.",
            "Neural networks are powerful.",
            "Text to speech synthesis.",
            "End to end modeling works.",
        ][:n_items]

    def __len__(self):
        return self.n_items

    def __getitem__(self, idx):
        phones = text_to_sequence(self.texts[idx])
        # Random mel and wav
        T_wav = 22050  # 1 second
        T_mel = T_wav // 256
        mel = torch.randn(80, T_mel)
        wav = torch.randn(1, T_wav)
        return torch.tensor(phones, dtype=torch.long), mel, wav


# ============================================================
# 损失函数
# ============================================================

def discriminator_loss(real_logits, fake_logits):
    """LSGAN discriminator loss."""
    loss = 0
    for real, fake in zip(real_logits, fake_logits):
        loss += F.mse_loss(real, torch.ones_like(real))
        loss += F.mse_loss(fake, torch.zeros_like(fake))
    return loss


def generator_loss(fake_logits):
    """LSGAN generator loss."""
    loss = 0
    for logit in fake_logits:
        loss += F.mse_loss(logit, torch.ones_like(logit))
    return loss


def feature_matching_loss(real_feats, fake_feats):
    """Feature matching loss (discriminator intermediate features L1)."""
    loss = 0
    n_layers = 0
    for real_f, fake_f in zip(real_feats, fake_feats):
        for r, f in zip(real_f, fake_f):
            loss += F.l1_loss(r, f.detach())
            n_layers += 1
    return loss / max(n_layers, 1)


# ============================================================
# 训练
# ============================================================

def _worker_init_fn(worker_id: int):
    """DataLoader worker: 每个 worker 只需单线程做 io + CPU mel,
    避免 (进程数 × worker 数 × 128) 线程把 CPU 压垮。"""
    torch.set_num_threads(1)


def train(rank, args, world_size=1):
    # 行缓冲: 重定向到文件时 print 默认块缓冲, step 日志会长时间不落盘
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except Exception:
        pass

    is_ddp = world_size > 1

    # torch 默认按全部核心 (这里 128) 开 intra-op 线程; 多进程 DDP 下 N×128 线程
    # 抢核会让 CPU 端小算子 (collate / CPU mel) 严重过度订阅。每进程压到 4 足够。
    torch.set_num_threads(4)

    if is_ddp:
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
        # 加大 timeout: 各 rank 数据长度不同导致同步等待时, 避免默认 watchdog 提前 abort
        dist.init_process_group(backend="nccl", init_method="env://",
                                rank=rank, world_size=world_size,
                                timeout=timedelta(minutes=30))
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
    if args.fast:
        dataset = FastDataset(n_items=16)
    else:
        dataset = LJSpeechDataset(root=args.data_root, segment_len=args.segment_len)

    sampler = torch.utils.data.distributed.DistributedSampler(dataset) if is_ddp else None
    # 音素已缓存后, worker 只做 sf.read + CPU mel (各 ~几 ms), 不再跑 gruut,
    # 原先 DDP + worker 死锁 (espeak 子进程 fork) 风险消失, DDP 也可开 worker。
    nw = 0 if args.fast else min(4, os.cpu_count())
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=sampler is None,
        sampler=sampler, num_workers=nw,
        collate_fn=Collation.collate, pin_memory=True,
        persistent_workers=nw > 0,
        prefetch_factor=2 if nw > 0 else None,
        worker_init_fn=_worker_init_fn if nw > 0 else None,
    )

    # --- 模型 ---
    if rank == 0:
        print("初始化模型...")

    model = SynthesizerTrn(
        n_vocab=88,
        spec_channels=80,
        segment_size=args.segment_len,
        inter_channels=192,
        hidden_channels=192,
        kernel_size=5,
        dilation_rate=1,
        n_layers=4,
        n_flows=4,
        n_heads=2,
        p_dropout=0.1,
        gin_channels=0,
    ).to(device)

    msd = MultiScaleDiscriminator().to(device)
    mpd = MultiPeriodDiscriminator().to(device)

    if is_ddp:
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)
        msd = DDP(msd, device_ids=[rank])
        mpd = DDP(mpd, device_ids=[rank])

    # --- Mel 提取器 ---
    mel_fn = torchaudio.transforms.MelSpectrogram(
        sample_rate=22050, n_fft=1024, n_mels=80,
        hop_length=256, f_min=0, f_max=8000, power=1.0,
    ).to(device)

    # --- 优化器 ---
    optim_g = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.8, 0.99))
    optim_d = torch.optim.AdamW(
        list(msd.parameters()) + list(mpd.parameters()),
        lr=args.lr, betas=(0.8, 0.99),
    )

    # --- AMP ---
    scaler_g = torch.amp.GradScaler(init_scale=2**10, growth_interval=200)
    scaler_d = torch.amp.GradScaler(init_scale=2**10, growth_interval=200)

    # --- Helper ---
    def unwrap(m):
        return m.module if is_ddp else m

    # --- 训练状态 ---
    step = 0
    epoch = 0
    best_kl = float('inf')
    nan_skips = 0       # 连续非有限 loss 计数
    abort = False       # 连续发散时终止训练

    def all_finite(loss: torch.Tensor) -> bool:
        """判断 loss 是否有限; DDP 下所有 rank 必须取得一致结论,
        否则部分 rank 跳过 backward 会使梯度 all-reduce 死锁。"""
        ok = torch.isfinite(loss).int()
        if is_ddp:
            dist.all_reduce(ok, op=dist.ReduceOp.MIN)
        return ok.item() > 0

    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  模型参数量: {total_params:,}")
        print(f"  MSD params: {sum(p.numel() for p in msd.parameters()):,}")
        print(f"  MPD params: {sum(p.numel() for p in mpd.parameters()):,}")
        print(f"开始训练 (max_steps={args.max_steps})...")
        print()

    while step < args.max_steps:
        epoch += 1
        if sampler is not None:
            sampler.set_epoch(epoch)
        if rank == 0 and step % max(1, args.log_interval) == 0 and step > 0:
            print(f"--- Epoch {epoch}, Step {step}/{args.max_steps} ---")

        for batch in loader:
            if step >= args.max_steps:
                break

            text_padded, text_mask, mel_padded, mel_mask, wav_padded = [
                t.to(device) if isinstance(t, torch.Tensor) else t for t in batch
            ]

            # ========== 训练判别器 (生成器 no_grad, 只存判别器计算图) ==========
            for p in msd.parameters():
                p.requires_grad = True
            for p in mpd.parameters():
                p.requires_grad = True

            optim_d.zero_grad()

            with torch.no_grad():
                with torch.amp.autocast('cuda'):
                    out = model(text_padded, text_mask, mel_padded, wav_padded, mel_mask)
                y_hat = out['y_hat']
                y_slice = out['y_slice']

            with torch.amp.autocast('cuda'):
                mpd_real = mpd(y_slice)
                mpd_fake = mpd(y_hat)
                msd_real = msd(y_slice)
                msd_fake = msd(y_hat)

                loss_mpd = discriminator_loss(
                    [r[1] for r in mpd_real], [f[1] for f in mpd_fake])
                loss_msd = discriminator_loss(
                    [r[1] for r in msd_real], [f[1] for f in msd_fake])
                loss_d = loss_mpd + loss_msd

            d_ok = all_finite(loss_d)
            if d_ok:
                scaler_d.scale(loss_d).backward()
                scaler_d.unscale_(optim_d)
                torch.nn.utils.clip_grad_norm_(
                    list(msd.parameters()) + list(mpd.parameters()), max_norm=5.0)
                scaler_d.step(optim_d)
                scaler_d.update()
                loss_d_log = loss_d.item()
            else:
                optim_d.zero_grad(set_to_none=True)
                loss_d_log = float('nan')

            del mpd_real, mpd_fake, msd_real, msd_fake, loss_d, y_hat, y_slice, out

            # ========== 训练生成器 (带梯度, 生成器+判别器图, backward 后释放) ==========
            for p in msd.parameters():
                p.requires_grad = False
            for p in mpd.parameters():
                p.requires_grad = False

            optim_g.zero_grad()

            with torch.amp.autocast('cuda'):
                out = model(text_padded, text_mask, mel_padded, wav_padded, mel_mask)
                y_hat = out['y_hat']
                y_slice = out['y_slice']
                kl_loss = out['kl_loss']
                dur_loss = out['dur_loss']

                mpd_fake = mpd(y_hat)
                msd_fake = msd(y_hat)
                with torch.no_grad():
                    mpd_real_feats = mpd(y_slice)
                    msd_real_feats = msd(y_slice)

                loss_gan = generator_loss([f[1] for f in mpd_fake]) + \
                           generator_loss([f[1] for f in msd_fake])

                fake_mel = mel_fn(y_hat.squeeze(1))
                real_mel = mel_fn(y_slice.squeeze(1))
                min_len = min(fake_mel.shape[-1], real_mel.shape[-1])
                loss_mel = F.l1_loss(fake_mel[..., :min_len], real_mel[..., :min_len])

                loss_fm = feature_matching_loss(
                    [f[0] for f in mpd_real_feats], [f[0] for f in mpd_fake]) + \
                    feature_matching_loss(
                        [f[0] for f in msd_real_feats], [f[0] for f in msd_fake])

                lambda_kl = 1.0
                lambda_dur = 1.0
                lambda_gan = 1.0
                lambda_fm = 10.0
                lambda_mel = 45.0

                loss_g = (lambda_kl * kl_loss +
                          lambda_dur * dur_loss +
                          lambda_gan * loss_gan +
                          lambda_fm * loss_fm +
                          lambda_mel * loss_mel)

            g_ok = all_finite(loss_g)
            if g_ok:
                scaler_g.scale(loss_g).backward()
                scaler_g.unscale_(optim_g)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler_g.step(optim_g)
                scaler_g.update()
            else:
                optim_g.zero_grad(set_to_none=True)

            # --- 非有限 loss 守卫: 跳过本步更新, 避免污染权重/检查点;
            #     连续发散则终止, 不再空跑写 NaN 检查点 ---
            if d_ok and g_ok:
                nan_skips = 0
            else:
                nan_skips += 1
                if rank == 0:
                    print(f"  ⚠ Step {step}: 非有限 loss (D_ok={d_ok}, G_ok={g_ok}), "
                          f"跳过更新 (连续 {nan_skips})")
                if nan_skips >= 50:
                    if rank == 0:
                        print("  ✗ 连续 50 步 loss 非有限, 判定发散, 终止训练。")
                    abort = True

            # --- 日志 (保存值, 然后释放计算图) ---
            kl_log = kl_loss.item()
            dur_log = dur_loss.item()
            loss_g_log = loss_g.item()
            loss_mel_log = loss_mel.item()
            loss_fm_log = loss_fm.item()
            loss_gan_log = loss_gan.item()

            del out, y_hat, y_slice, mpd_fake, msd_fake, mpd_real_feats, msd_real_feats, loss_g, loss_mel, loss_fm

            if rank == 0 and step % args.log_interval == 0:
                print(
                    f"  Step {step:6d}/{args.max_steps} | "
                    f"KL: {kl_log:.2f} | "
                    f"Dur: {dur_log:.2f} | "
                    f"G: {loss_g_log:.2f} | "
                    f"D: {loss_d_log:.2f} | "
                    f"Mel: {loss_mel_log:.4f} | "
                    f"FM: {loss_fm_log:.4f} | "
                    f"GAN: {loss_gan_log:.4f}"
                )

            # --- 保存检查点 ---
            if rank == 0 and step % args.save_interval == 0 and step > 0:
                ckpt = {
                    'model': unwrap(model).state_dict(),
                    'msd': unwrap(msd).state_dict(),
                    'mpd': unwrap(mpd).state_dict(),
                    'optim_g': optim_g.state_dict(),
                    'optim_d': optim_d.state_dict(),
                    'scaler_g': scaler_g.state_dict(),
                    'scaler_d': scaler_d.state_dict(),
                    'step': step,
                    'epoch': epoch,
                    'best_kl': best_kl,
                }

                torch.save(ckpt, output_dir / f'checkpoint_{step:06d}.pt')
                torch.save(unwrap(model).state_dict(), output_dir / f'model_{step:06d}.pt')

                if kl_log < best_kl:
                    best_kl = kl_log
                    torch.save(ckpt, output_dir / 'checkpoint_best.pt')
                    torch.save(unwrap(model).state_dict(), output_dir / 'model_best.pt')
                    print(f"  ✓ 最佳模型已保存 (KL={best_kl:.2f})")

                # Demo inference
                if step % (args.save_interval * 5) == 0 and step > 0:
                    _run_demo(unwrap(model), output_dir, step, device)

            step += 1

            if abort:
                break

        if abort:
            break

    # 最终保存
    if rank == 0:
        ckpt = {
            'model': unwrap(model).state_dict(),
            'step': step,
            'epoch': epoch,
        }
        torch.save(ckpt, output_dir / 'checkpoint_final.pt')
        torch.save(unwrap(model).state_dict(), output_dir / 'model_final.pt')
        print(f"\n训练完成! 模型已保存到 {output_dir}")

    if is_ddp:
        torch.cuda.synchronize(device)
        dist.destroy_process_group()


def _run_demo(model, output_dir, step, device):
    """运行推理 demo 并保存音频。"""
    import soundfile as sf

    model.eval()
    demo_texts = [
        "Printing, in the only sense with which we are at present concerned, differs from most if not from all the arts and crafts represented in the Exhibition.",
        "The invention of movable metal letters in the middle of the fifteenth century may justly be considered as the invention of the art of printing.",
    ]

    try:
        for i, text in enumerate(demo_texts):
            phones = text_to_sequence(text)
            x = torch.tensor(phones, dtype=torch.long, device=device).unsqueeze(0)
            x_mask = torch.ones(1, 1, x.shape[-1], device=device)

            wav = model.inference(x, x_mask, noise_scale=0.667, noise_scale_dur=0.8)
            wav = wav.squeeze().cpu().numpy()

            sf.write(str(output_dir / f'demo_{step:06d}_{i}.wav'), wav, 22050)
            print(f"  Demo {i}: 已保存 ({len(wav)/22050:.1f}s)")
    except Exception as e:
        print(f"  Demo 失败: {e}")

    model.train()


# ============================================================
# 主入口
# ============================================================

def main():
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except Exception:
        pass

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False  # 变长输入下 benchmark 会导致 workspace 缓存无限增长

    parser = argparse.ArgumentParser(description="VITS 训练")
    parser.add_argument("--data-root", type=str, default="./data/LJSpeech-1.1",
                        help="LJSpeech 数据根目录")
    parser.add_argument("--output-dir", type=str,
                        default=str(Path(__file__).parent / "output"),
                        help="输出目录")
    parser.add_argument("--segment-len", type=int, default=8192,
                        help="训练音频片段长度")
    parser.add_argument("--batch-size", type=int, default=64, help="每卡批次大小")
    parser.add_argument("--lr", type=float, default=2e-4, help="学习率")
    parser.add_argument("--max-steps", type=int, default=500000, help="最大步数")
    parser.add_argument("--log-interval", type=int, default=100, help="日志间隔")
    parser.add_argument("--save-interval", type=int, default=10000, help="保存间隔")
    parser.add_argument("--fast", action="store_true", help="快速验证模式 (100 步)")
    parser.add_argument("--multi-gpu", action="store_true", help="启用多卡 DDP")
    parser.add_argument("--resume", type=str, default=None, help="恢复检查点")
    args = parser.parse_args()

    if args.fast:
        args.max_steps = 200
        args.log_interval = 20
        args.batch_size = 4
        args.segment_len = 16384
        print("[快速验证模式]")

    world_size = torch.cuda.device_count() if args.multi_gpu else 1

    if world_size > 1 and args.multi_gpu:
        print(f"检测到 {world_size} 张 GPU, 使用 DDP 多卡训练")
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29501")
        mp.spawn(train, args=(args, world_size), nprocs=world_size, join=True)
    else:
        train(0, args)


if __name__ == "__main__":
    main()