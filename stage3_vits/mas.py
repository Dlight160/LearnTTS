"""
Monotonic Alignment Search (MAS)
=================================
VITS 核心创新: 无监督单调对齐搜索算法。

通过动态规划在文本音素和波形的隐变量帧之间找到最可能的单调对齐。

MAS 的核心思想:
  给定音素序列的先验分布参数 (mu, sigma) 和后验隐变量序列 z，
  找到使 log p(z | mu, sigma) 最大的单调对齐路径。

  dp[i][j] = max(dp[i][j-1], dp[i-1][j-1]) + log p(z_j | mu_i, sigma_i)
    - dp[i][j-1]:   当前音素继续 (stay)
    - dp[i-1][j-1]: 切换到下一个音素 (switch)

性能说明
--------
官方 VITS 用 Cython 编译的 maximum_path 在 CPU 上跑 DP。这里用纯 numpy:
  - DP 只对 mel 帧 (j) 做一层 Python 循环, 在 (B, T_text) 维度上向量化;
  - 回溯同样向量化over batch, 全程零 GPU↔CPU 同步。
这避免了"在 GPU 张量上逐元素 Python 循环 + 标量比较触发同步"导致的
GPU 空转 (旧实现在全长 LJSpeech 上每次前向要发射数十万串行 micro-kernel)。
"""

import numpy as np
import torch


# ============================================================
# log p 矩阵 (GPU 向量化, 无循环)
# ============================================================

def mas_logp(
    z_p: torch.Tensor,
    m_p: torch.Tensor,
    logs_p: torch.Tensor,
    x_mask: torch.Tensor,
    y_mask: torch.Tensor,
) -> torch.Tensor:
    """计算 MAS 的 log p 矩阵。

    log p(z_j | mu_i, sigma_i) = -0.5 * (log(2pi) + 2*log_sigma + (z-mu)^2 / sigma^2)
    (只算到与 argmax 无关的常数项之外的部分)

    Args:
        z_p:    (B, C, T_mel),  后验隐变量
        m_p:    (B, C, T_text), 先验均值
        logs_p: (B, C, T_text), 先验对数标准差
        x_mask: (B, 1, T_text)
        y_mask: (B, 1, T_mel)

    Returns:
        (B, T_text, T_mel), 对数概率矩阵
    """
    z = z_p.unsqueeze(-1)        # (B, C, T_mel, 1)
    m = m_p.unsqueeze(-2)        # (B, C, 1, T_text)
    logs = logs_p.unsqueeze(-2)  # (B, C, 1, T_text)

    log_2pi = float(np.log(2 * np.pi))
    log_p = -0.5 * (log_2pi + 2 * logs + (z - m) ** 2 * torch.exp(-2 * logs))

    log_p = log_p.sum(dim=1)         # (B, T_mel, T_text)
    log_p = log_p.permute(0, 2, 1)   # (B, T_text, T_mel)
    return log_p


# ============================================================
# 向量化 numpy DP + 回溯
# ============================================================

NEG_INF = -1e9


def _build_band_mask(B: int, T: int, S: int,
                     t_x: np.ndarray, t_y: np.ndarray) -> np.ndarray:
    """单调对齐可行域 (band)。

    cell (i, j) 可行 <=> 存在一条从 (0,0) 到 (t_x-1, t_y-1) 的单调路径经过它:
        i <= j                    (前 i+1 个音素至少占 i+1 帧)
        j - i <= t_y - t_x        (剩余音素 <= 剩余帧, 保证每个音素 >=1 帧)
        i < t_x 且 j < t_y        (落在该样本的有效长度内)
    """
    ii = np.arange(T)[None, :, None]   # (1, T, 1)
    jj = np.arange(S)[None, None, :]   # (1, 1, S)
    tx = t_x[:, None, None]
    ty = t_y[:, None, None]
    band = (ii <= jj) & ((jj - ii) <= (ty - tx)) & (ii < tx) & (jj < ty)
    return band  # (B, T, S) bool


def maximum_path(log_p: torch.Tensor,
                 x_mask: torch.Tensor,
                 y_mask: torch.Tensor) -> torch.Tensor:
    """对每个样本求最优单调对齐 (硬对齐 0/1 矩阵)。

    Args:
        log_p:  (B, T_text, T_mel), 对数概率
        x_mask: (B, 1, T_text)
        y_mask: (B, 1, T_mel)

    Returns:
        attn: (B, T_text, T_mel) float, 二进制对齐矩阵, 与 log_p 同 device。
    """
    device = log_p.device
    B, T, S = log_p.shape

    t_x = x_mask.squeeze(1).sum(-1).long().cpu().numpy().astype(np.int64)  # (B,)
    t_y = y_mask.squeeze(1).sum(-1).long().cpu().numpy().astype(np.int64)  # (B,)

    value = log_p.detach().float().cpu().numpy().astype(np.float64)  # (B, T, S)

    # 可行域外置 -inf
    band = _build_band_mask(B, T, S, t_x, t_y)
    value = np.where(band, value, NEG_INF)

    # ---- DP: 只对 mel 帧 j 循环, 在 (B, T_text) 维度向量化 ----
    # dp[:, i, j] = max(dp[:, i, j-1], dp[:, i-1, j-1]) + logp[:, i, j]
    for j in range(1, S):
        v_stay = value[:, :, j - 1]                  # (B, T)  同音素继续
        v_switch = np.full((B, T), NEG_INF)          # (B, T)  切换到下一音素 (i-1)
        v_switch[:, 1:] = value[:, :-1, j - 1]
        value[:, :, j] = np.maximum(v_stay, v_switch) + value[:, :, j]

    # ---- 回溯: 从 (t_x-1, t_y-1) 沿最优前驱走回 (0, 0), 向量化over batch ----
    path = np.zeros((B, T, S), dtype=np.float32)
    arangeB = np.arange(B)
    index = (t_x - 1).clip(min=0)  # (B,) 当前音素索引

    for j in range(S - 1, -1, -1):
        active = j < t_y                       # (B,) 该样本是否仍在有效长度内
        b_active = np.nonzero(active)[0]
        if b_active.size:
            path[b_active, index[b_active], j] = 1.0

        if j > 0:
            stay = value[arangeB, index, j - 1]
            prev_i = np.maximum(index - 1, 0)
            switch = np.where(index > 0, value[arangeB, prev_i, j - 1], NEG_INF)
            move = (switch >= stay) & (index > 0) & active
            index = index - move.astype(np.int64)

    return torch.from_numpy(path).to(device)


# ============================================================
# 训练入口: log p 计算 + 路径搜索
# ============================================================

def mas_hardcoded(z_p: torch.Tensor, m_p: torch.Tensor, logs_p: torch.Tensor,
                  x_mask: torch.Tensor, y_mask: torch.Tensor) -> torch.Tensor:
    """完整 MAS: 计算 logp + 单调路径搜索 (带 mask)。

    Args:
        z_p:    (B, C, T_mel),  后验 z
        m_p:    (B, C, T_text), 先验均值
        logs_p: (B, C, T_text), 先验对数标准差
        x_mask: (B, 1, T_text)
        y_mask: (B, 1, T_mel)

    Returns:
        attn: (B, T_text, T_mel) float, 对齐矩阵
    """
    log_p = mas_logp(z_p, m_p, logs_p, x_mask, y_mask)
    return maximum_path(log_p, x_mask, y_mask)


def generate_attn(durations: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
    """从音素时长生成对齐矩阵 (用于由 duration 展开先验)。

    Args:
        durations: (B, T_text), 每个音素的帧数
        x_mask:    (B, 1, T_text)

    Returns:
        attn: (B, T_text, T_mel)
    """
    B, T_text = durations.shape
    T_mel = int(durations.sum(dim=1).max().item())

    attn = torch.zeros(B, T_text, T_mel, device=durations.device)
    for b in range(B):
        pos = 0
        for i in range(T_text):
            d = int(durations[b, i].item())
            if d > 0:
                attn[b, i, pos:pos + d] = 1.0
                pos += d

    attn = attn * x_mask.transpose(1, 2)
    return attn


# ============================================================
# 自测
# ============================================================

def test_mas():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"测试: {device}")

    B, C = 3, 80
    T_text, T_mel = 12, 70

    z_p = torch.randn(B, C, T_mel, device=device)
    m_p = torch.randn(B, C, T_text, device=device)
    logs_p = torch.randn(B, C, T_text, device=device) * 0.5 - 1

    # 变长: 每个样本不同的有效长度, 测 mask + band
    x_lens = torch.tensor([12, 9, 6], device=device)
    y_lens = torch.tensor([70, 55, 30], device=device)
    x_mask = (torch.arange(T_text, device=device)[None, :] < x_lens[:, None]).float().unsqueeze(1)
    y_mask = (torch.arange(T_mel, device=device)[None, :] < y_lens[:, None]).float().unsqueeze(1)

    attn = mas_hardcoded(z_p, m_p, logs_p, x_mask, y_mask)
    print(f"对齐矩阵: {tuple(attn.shape)}")

    for b in range(B):
        a = attn[b]
        tx, ty = int(x_lens[b]), int(y_lens[b])
        # 每个有效帧恰好对齐一个音素
        col = a.sum(dim=0)
        assert torch.all(col[:ty] == 1), f"样本{b}: 有效帧应恰好对齐一个音素"
        assert torch.all(col[ty:] == 0), f"样本{b}: padding 帧不应对齐"
        # 每个有效音素至少占一帧 (band 保证)
        row = a.sum(dim=1)
        assert torch.all(row[:tx] >= 1), f"样本{b}: 每个音素应 >=1 帧"
        assert torch.all(row[tx:] == 0), f"样本{b}: padding 音素不应对齐"
        # 时长之和 = 有效帧数
        assert int(row.sum()) == ty, f"样本{b}: 时长和={int(row.sum())} != {ty}"
        # 单调性: 音素索引随帧单调不减
        idx = a.argmax(dim=0)[:ty]
        assert torch.all(idx[1:] >= idx[:-1]), f"样本{b}: 对齐应单调不减"

    print("  ✓ mask / band / 单调性 / 满覆盖 全部通过")
    print("✓ MAS 测试通过")


if __name__ == "__main__":
    test_mas()
