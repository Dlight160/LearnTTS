# CLAUDE.md
This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LearnTTS — 系统性 TTS 学习项目,从语音信号基础到前沿扩散/LLM-based TTS。目标是理解 CosyVoice、FishSpeech 等主流模型并能在开源模型上做优化。

## 运行环境

- **Python**: 3.12 (conda 本地环境 `conda_env/`)
- **Python 路径**: `<project_root>/conda_env/bin/python`
- **CUDA**: 可用 (NVIDIA GeForce RTX 3090 ×8)
- **操作系统**: Linux

### 运行 Python 脚本

```bash
# 项目根目录执行，使用本地 conda 环境
./conda_env/bin/python <script.py> [args]
```

## 项目结构

| 路径 | 内容 |
|------|------|
| `TTS_LEARNING_ROADMAP.md` | 完整学习路线(6 阶段)、项目、论文清单 |
| `stage1_speech_signal/` | 阶段1: Mel 频谱分析器 |
| `stage2_hifi_gan/` | 阶段2: HiFi-GAN 神经声码器 |
| `stage3_vits/` | 阶段3: VITS 端到端 TTS |

## 命令参考

所有脚本用 `./conda_env/bin/python` 运行，具体参数见各脚本的 argparse 或 `--help`。

<!-- 学习路线详见 LEARNING_ROADMAP.md（阶段 1~3 已完成，阶段 4 待开始） -->

## 开发约定

- 每个阶段一个独立目录 `stageN_name/`
- 数据文件(wav、图片、结果)统一放在 `stageN_name/output/`
- Mel 频谱图用 `magma` colormap,一致性标准化输出

## 代码风格

- 函数签名加完整类型注解（Python 3.12 语法）
- 用 f-string，不用 `%` 或 `.format()`
- import 顺序：标准库 → 第三方（PyTorch、soundfile 等）→ 本地模块
- 类名用 PascalCase，函数/变量用 snake_case
- Tensor 操作优先用 PyTorch 函数而非 NumPy

## 调试与常见问题

- **显存 OOM**：先减半 batch_size 重试，或切到单卡模式
- **训练日志**：输出到 `stageN_name/output/logs/`，用 TensorBoard 查看
- **状态检查**：`nvidia-smi` 查看 GPU 占用和温度
- **训练中断**：checkpoint 自动保存到 `stageN_name/output/ckpt/`
- **数据路径**：训练数据统一放在 `data/` 目录下（已 gitignore）
- **多卡训练**：用 `CUDA_VISIBLE_DEVICES` 选定 GPU，配合 `--multi-gpu`

## 工作流

- **HANDOFF.md** — 阶段进度记录。每次对话开始时读取，了解上次完成的内容、遗留问题和下一步计划
- **QA_Log.md** — 学习问答积累。我提概念/学习性问题时，主动追加记录（日期 / 原始问题 / 答案核心 / 引用），从新到旧排序
