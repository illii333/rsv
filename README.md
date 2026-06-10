# RSV — 视频翻译配音工具 🎬🔊

把一个英文视频，自动转成**中文配音 + 中文字幕**的视频。

也可以翻译成日文、韩文、法文等多种语言。

---

## 📦 安装教程（小白版）

### 你需要准备什么

- 一台有 **NVIDIA 显卡** 的电脑（推荐 6GB 以上显存）
- 已经装好了系统（Windows 需要 WSL2，Linux/macOS 直接装）

### 第一步：下载代码

```bash
# 如果已经装了 git
git clone https://github.com/illii333/rsv.git
cd rsv

# 如果没有 git，去 https://github.com/illii333/rsv 点绿色按钮 Download ZIP
# 解压后打开终端，进入那个文件夹
```

### 第二步：安装系统依赖

打开终端（命令行），根据你的系统选一个跑：

<details>
<summary><b>🟦 Arch Linux</b></summary>

```bash
sudo pacman -S python3 python-pip ffmpeg rust
```
</details>

<details>
<summary><b>🟧 Ubuntu / Debian</b></summary>

```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv ffmpeg curl
# Rust 用这个装
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
# 按 1 回车，装完重启终端
```
</details>

<details>
<summary><b>🍎 macOS</b></summary>

```bash
# 先装 Homebrew（如果没有的话）
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 再装依赖
brew install python3 ffmpeg rust
```
</details>

<details>
<summary><b>🪟 Windows（WSL2）</b></summary>

```bash
# 先装 WSL2（百度搜教程），打开 Ubuntu 终端
sudo apt update
sudo apt install python3 python3-pip python3-venv ffmpeg curl
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
# 按 1 回车，装完重启终端
```
</details>

### 第三步：创建虚拟环境（重要！）

```bash
# 在 rsv 文件夹里运行
python3 -m venv venv
```

#### 每次打开新终端，都要先激活：

```bash
# Linux / Mac
source venv/bin/activate

# Windows (WSL2) 也是一样
source venv/bin/activate
```

激活成功后，终端前面会出现 `(venv)` 字样。

### 第四步：安装 Python 包

```bash
# 确保你已经激活了虚拟环境（前面有 (venv)）
# 然后一条一条跑：

pip install faster-whisper          # 语音识别
pip install transformers torch sentencepiece soundfile pydub  # 翻译 + 音频处理
pip install qwen-tts                # 语音克隆
```

> ⏳ 下载时间可能比较长（几个 G 的模型文件），耐心等待。

### 第五步：编译 Rust 程序

```bash
cargo build
```

> 第一次编译会比较慢，之后就好了。

---

## 🎬 使用教程

### 准备一个视频

把你要翻译的英文视频放到某个位置，比如 `/home/你的名字/视频.mp4`

### 一键运行

```bash
# 1. 激活虚拟环境（每次新终端都要）
source venv/bin/activate

# 2. 设置环境变量（解决 CUDA 兼容问题）
export LD_LIBRARY_PATH="/tmp/cublas_fix:${LD_LIBRARY_PATH:-}"

# 3. 开始处理！（把路径换成你的视频）
bash scripts/quick_run.sh /home/你的名字/视频.mp4
```

### 运行过程中

脚本会一步步问你：

1. **语音识别** — 自动跑，不用管
2. **选语言** — 输入数字：`1`=中文 `2`=英文 `3`=日文...
3. **翻译完成** — 会显示字幕内容
4. **审查字幕** — 打开提示的文件，修改错别字，改完按 `Y` 继续
5. **语音合成** — 自动跑
6. **合成视频** — 自动跑

最后生成的文件在 `output_xxxx/` 文件夹里：
- `final.mp4` — 🎉 最终成品！
- `2_translated.srt` — 字幕文件

---

## ❓ 常见问题

### 跑的时候报错 "libcublas.so.12 not found"

```bash
# 执行这一行
export LD_LIBRARY_PATH="/tmp/cublas_fix:${LD_LIBRARY_PATH:-}"
```

### 报错 "No module named xxx"

```bash
# 忘记激活虚拟环境了
source venv/bin/activate
pip install xxx
```

### 语音识别很慢

检查是不是在用 GPU：
```
[ASR] Device auto-detected: cuda    ← ✅ GPU 模式
[ASR] Device auto-detected: cpu     ← ❌ CPU 模式，很慢
```
如果是 CPU 模式，检查 CUDA 安装。

### 生成的视频有杂音/爆音

已经处理过了，如果还有，可以调大 `assemble.py` 里的淡入淡出值。

---

## 📁 项目结构

```
rsv/
├── scripts/              # Python 脚本（主要逻辑）
│   ├── quick_run.sh      # 一键交互式运行
│   ├── asr_faster_whisper.py   # 语音识别
│   ├── translate_m2m100.py     # 翻译
│   ├── tts_segment.py          # 语音克隆
│   └── assemble.py             # 合成视频
├── rsv-cli/              # Rust CLI 入口
├── rsv-core/             # 核心配置
├── rsv-asr/              # ASR 引擎
├── rsv-translate/        # 翻译引擎
├── rsv-tts/              # TTS 引擎
├── venv/                 # Python 虚拟环境（装完才有）
└── Cargo.toml            # Rust 配置
```

---

## 技术说明

- **ASR**: Faster-Whisper large-v3-turbo（GPU）
- **翻译**: Facebook M2M100 418M
- **TTS**: Qwen3-TTS-12Hz-1.7B-Base（语音克隆）
- **显存**: 约 4.5GB（一次只跑一个模型）
- **所有模型在第一次运行时自动下载**，之后可以离线使用
