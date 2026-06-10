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

## 快速开始

```bash
# 编译
cargo build

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

### Python 依赖

```bash
# ASR
pip install faster-whisper

# 翻译
pip install transformers torch sentencepiece

# TTS (至少选一个)
pip install edge-tts              # 免费在线，无需 GPU
# 或 Qwen3-TTS (根据具体安装方式)
```

## CLI 用法

```bash
rsv asr <音频文件>           # 语音识别 → 字幕
rsv translate <文本>         # 翻译
rsv tts <文本>               # 语音合成/克隆
```

每个命令都支持 `--help` 查看详细参数。
