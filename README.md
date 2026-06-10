# RSV — Rust Speech/Video Toolkit 🦀🎤🔊

**R**ust **S**peech **V**ideo — 用 Rust 写的语音识别 + 翻译 + 语音克隆工具箱。

## 架构

```
rsv/
├── rsv-core/         # 核心类型、配置
├── rsv-asr/          # 语音识别 (Faster-Whisper)
├── rsv-translate/    # 翻译 (M2M100)
├── rsv-tts/          # 语音合成/克隆 (Qwen3-TTS)
├── rsv-cli/          # CLI 入口
└── scripts/          # Python bridge 脚本
```

每个 crate 独立，可单独测试。

## 安装

### 1. 系统依赖

```bash
# Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Python + FFmpeg
sudo pacman -S python3 python-pip ffmpeg   # Arch
# 或 brew install python3 ffmpeg           # macOS
# 或 apt install python3 python3-pip ffmpeg # Ubuntu/Debian
```

### 2. Python 虚拟环境

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. 安装 Python 依赖

```bash
pip install faster-whisper
pip install transformers torch sentencepiece soundfile pydub
pip install qwen-tts
```

### 4. 编译 Rust

```bash
cargo build
```

## 快速开始

```bash
# 激活虚拟环境（每次新终端都需要）
source venv/bin/activate

# 一键管线（ASR → 翻译 → TTS → 合成视频）
export LD_LIBRARY_PATH="/tmp/cublas_fix:${LD_LIBRARY_PATH:-}"
bash scripts/quick_run.sh input.mp4

# 或分步使用：
# 语音识别
cargo run -- asr input.mp4 -m large-v3-turbo

# 翻译
cargo run -- translate "Hello world" -s en -t zh-cn

# TTS / 语音克隆
cargo run -- tts "你好世界" -o output/tts
```

## 依赖

- **Rust** (edition 2021)
- **Python 3.8+** — 模型运行依赖 Python
- **FFmpeg** — 音视频处理

## CLI 用法

```bash
rsv asr <音频文件>           # 语音识别 → 字幕
rsv translate <文本>         # 翻译
rsv tts <文本>               # 语音合成/克隆
```

每个命令都支持 `--help` 查看详细参数。
