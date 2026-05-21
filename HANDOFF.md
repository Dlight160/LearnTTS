# Handoff — 2026-05-15 21:00

## 上次进度（2026-05-11）
完成了 git 仓库初始化、权限白名单、QA_Log.md 和 /handoff 技能创建。

## 本阶段完成（2026-05-15）

### 已完成
- **阶段 1 学习完成**
  - 完整阅读 `stage1_speech_signal/mel_analyzer.py` 全部代码（~600 行）
  - `load_audio` 重构：从 `torchaudio.load` 改为 `soundfile.read`，避免 torchcodec 依赖问题
  - 参数扫描实验代码已就绪（`--sweep` 模式）：n_fft、n_mels、GL 迭代次数的正交对比
- **阶段 1 知识点考核**（6 道题，全部通过）：
  - ✅ Mel 尺度动机：人耳感知非线性压缩，低频敏感、高频不敏感
  - ⚠️ hop_length 权衡：时间分辨率与重建质量，75% 重叠是经验值
  - ✅ GL 随机相位：避免对称陷阱，但始终无法恢复原始相位（信息论局限）
  - ✅ SNR 与 n_mels：先升后 plateau，Mel 滤波本质是信息压缩
  - ✅ GL 迭代上限：~30-60 次收敛，1000 次也无法逼近原始 SNR
  - ✅ Mel bin ↔ Hz 映射：低频 bin 窄、高频 bin 宽的非线性关系

## 本阶段完成（2026-05-21）

### 已完成
- **阶段 2：HiFi-GAN 模型构建**
  - `stage2_hifi_gan/model.py`：完整实现 Generator（含 MRF）、MSD、MPD
  - `stage2_hifi_gan/train.py`：完整训练循环（Mel Loss + GAN Loss + FM Loss）
  - 单卡训练 + 多卡 DDP 支持（`--multi-gpu` 参数）
  - LJSpeech-1.1 数据手动下载到 `data/LJSpeech-1.1/`
  - 训练验证通过：单卡（GPU 2）和 4 卡 DDP（GPU 2-5）均正常跑通

- **阶段 2 理论学习（QA_Log.md）**
  - 卷积层 kernel / stride / dilation / padding / bias 机制
  - LeakyReLU 激活函数的作用
  - MRF 多感受野融合的设计动机
  - ResBlock 残差连接的原理
  - MSD（多尺度判别器）+ MPD（多周期判别器）架构详解
  - GAN 对抗训练的必要性（Mel Loss 局限性）

- **环境迁移**
  - 从 Windows 11 迁移到 Linux（RTX 3090 ×8）
  - conda 环境重建为本地 `conda_env/`
  - CLAUDE.md 命令更新为 Linux 路径

### 代码改动
| 文件 | 改动 |
|------|------|
| `stage1_speech_signal/mel_analyzer.py` | `load_audio` 改用 `sf.read`，多声道自动混缩 |
| `stage2_hifi_gan/model.py` | 新建：HiFiGANGenerator / MSD / MPD 完整实现 |
| `stage2_hifi_gan/train.py` | 新建：训练循环 + 多卡 DDP 支持 |
| `stage2_hifi_gan/infer.py` | 新建：推理脚本 |
| `stage2_hifi_gan/compare.py` | 新建：GL vs HiFi-GAN 对比 |
| `CLAUDE.md` | 更新为 Linux 环境配置和命令 |
| `.claude/CLAUDE.md` | 添加 conda 运行指令 |
| `QA_Log.md` | 新增 10 条深度学习/声码器概念问答 |
| `.gitignore` | 新增 `conda_env/`、`data/` 条目 |

### 存在问题的点
- 多声道自动混缩用 `data.mean(axis=1)` 代替了原来的 `waveform.mean(dim=0, keepdim=True)` 逻辑——混缩后的幅度略有不同
- `conda_env/` 和 `data/` 已加入 `.gitignore`，不纳入版本控制

### 代码改动
| 文件 | 改动 |
|------|------|
| `stage1_speech_signal/mel_analyzer.py` | `load_audio` 改用 `sf.read`，多声道自动混缩 |
| `conda_env/` | 未追踪的本地 conda 环境文件 |
| `stage1_speech_signal/prompt_audio.wav` | 未追踪的测试音频 |

### 存在问题的点
- 多声道自动混缩用 `data.mean(axis=1)` 代替了原来的 `waveform.mean(dim=0, keepdim=True)` 逻辑——混缩后的幅度略有不同
- `conda_env/` 和 `data/` 已加入 `.gitignore`，不纳入版本控制

## 下一步
- **阶段 2 继续：HiFi-GAN 训练与评估**
  - `infer.py` 推理脚本 + Mel Loss / PESQ / STOI 客观指标
  - `compare.py` Griffin-Lim vs HiFi-GAN 对比实验
  - 完整 500K steps 训练 + 中间 checkpoint 可视化

## 关键上下文（继承）
- 代理配置：git push 需走 `http://127.0.0.1:7897` 代理
- 技能文件路径：`.claude/skills/handoff/SKILL.md`
- 项目 CLAUDE.md 有两级：根目录（项目信息）和 `.claude/` 目录（指令）
- 多卡训练使用 `CUDA_VISIBLE_DEVICES` 指定 GPU + `--multi-gpu` 参数
