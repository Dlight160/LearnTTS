#!/usr/bin/env python3
"""
音素缓存预计算
===============
gruut g2p (文本→IPA 音素) 在真实 LJSpeech 句子上 ~1.2s/句, 是训练时的头号瓶颈:
DDP 下 num_workers=0, 每步 batch 个 g2p 全堵在主进程, GPU 空等 → 利用率 0/100 抖动。

文本→音素是确定性的, 跑一次存盘即可。本脚本用多进程并行预计算, 缓存成
phonemes_cache.pt (dict: utterance_id -> list[int])。训练时 __getitem__ 直接查表。

用法:
  ./conda_env/bin/python stage3_vits/precompute_phonemes.py \
      --data-root ./data/LJSpeech-1.1 --workers 64
"""

import argparse
import time
from multiprocessing import Pool
from pathlib import Path

import torch


def _g2p_worker(args: tuple) -> tuple:
    """子进程: 单句 g2p。延迟 import 让每个 worker 各自加载 gruut。"""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from stage3_vits.text_symbols import text_to_sequence

    uid, text = args
    return uid, text_to_sequence(text)


def load_items(meta_path: Path) -> list:
    items = []
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("|")
            if len(parts) >= 3:
                items.append((parts[0], parts[2]))
    return items


def main():
    parser = argparse.ArgumentParser(description="预计算 LJSpeech 音素缓存")
    parser.add_argument("--data-root", type=str, default="./data/LJSpeech-1.1")
    parser.add_argument("--workers", type=int, default=64,
                        help="并行进程数 (默认 64)")
    parser.add_argument("--out", type=str, default=None,
                        help="输出缓存路径 (默认 <data-root>/phonemes_cache.pt)")
    args = parser.parse_args()

    root = Path(args.data_root)
    meta_path = root / "metadata.csv"
    out_path = Path(args.out) if args.out else root / "phonemes_cache.pt"

    items = load_items(meta_path)
    n = len(items)
    print(f"待处理: {n} 句  |  并行进程: {args.workers}  |  输出: {out_path}")

    cache = {}
    t0 = time.time()
    done = 0
    with Pool(processes=args.workers) as pool:
        for uid, phones in pool.imap_unordered(_g2p_worker, items, chunksize=8):
            cache[uid] = phones
            done += 1
            if done % 500 == 0 or done == n:
                elapsed = time.time() - t0
                rate = done / max(elapsed, 1e-6)
                eta = (n - done) / max(rate, 1e-6)
                print(f"  {done}/{n}  ({rate:.0f} 句/s, 已用 {elapsed:.0f}s, "
                      f"剩 {eta:.0f}s)", flush=True)

    torch.save(cache, out_path)
    lens = [len(v) for v in cache.values()]
    print(f"\n完成: {len(cache)} 条 → {out_path}")
    print(f"  音素序列长度: min={min(lens)} max={max(lens)} "
          f"mean={sum(lens)/len(lens):.1f}")
    print(f"  总耗时: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
