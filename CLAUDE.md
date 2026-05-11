7 从v吧UDE.md
/+
This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LearnTTS — 系统性 TTS 学习项目,从语音信号基础到前沿扩散/LLM-based TTS。目标是理解 CosyVoice、FishSpeech 等主流模型并能在开源模型上做优化。

## 运行环境

- **Python**: 3.10 (conda 环境 `learntts`)
- **conda 路径**: `C:/Users/L/miniconda3/envs/learntts/`
- **CUDA**: 可用
- **操作系统**: Windows 11 (Git Bash)

### 运行 Python 脚本

Windows 上 `conda run` 有 Unicode 编码问题,必须用以下方式运行:

```bash
PYTHONIOENCODING=utf-8 C:/Users/L/miniconda3/envs/learntts/python.exe <script.py> [args]
```

## 项目结构

| 路径 | 内容 |
|------|------|
| `TTS_LEARNING_ROADMAP.md` | 完整学习路线(6 阶段)、项目、论文清单 |
| `stage1_speech_signal/` | 阶段1: Mel 频谱分析器 |
| `stageN_*/` | 后续阶段逐步添加 |

## 阶段 1 命令参考

```bash
# 运行 Mel 频谱分析器(默认生成扫频测试信号)
PYTHONIOENCODING=utf-8 C:/Users/L/miniconda3/envs/learntts/python.exe stage1_speech_signal/mel_analyzer.py

# 指定音频文件
PYTHONIOENCODING=utf-8 C:/Users/L/miniconda3/envs/learntts/python.exe stage1_speech_signal/mel_analyzer.py /path/to/speech.wav

# 参数扫描实验
PYTHONIOENCODING=utf-8 C:/Users/L/miniconda3/envs/learntts/python.exe stage1_speech_signal/mel_analyzer.py --sweep

# 自定义参数
PYTHONIOENCODING=utf-8 C:/Users/L/miniconda3/envs/learntts/python.exe stage1_speech_signal/mel_analyzer.py --n-fft 2048 --n-mels 128
```

## 学习路线概览

1. **阶段1 ✅** — 语音信号基础: STFT、Mel 频谱、Griffin-Lim
2. **阶段2** — 神经声码器: HiFi-GAN
3. **阶段3** — 端到端 TTS: VITS
4. **阶段4** — 音频 Tokenization: VQ-VAE / RVQ
5. **阶段5** — LLM-based TTS: FishSpeech
6. **阶段6** — 扩散/Flow Matching: CosyVoice

## 开发约定

- 每个阶段一个独立目录 `stageN_name/`
- 数据文件(wav、图片、结果)统一放在 `stageN_name/output/`
- Mel 频谱图用 `magma` colormap,一致性标准化输出
