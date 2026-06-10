#!/usr/bin/env python3
"""
RSV - 单条语音克隆合成
每次处理指定范围的字幕段，用完释放显存。

Usage:
  # 处理第 0 条
  python3 tts_segment.py '<json>'

  # 处理第 0-4 条
  python3 tts_segment.py '{"input_dir": "...", "output_dir": "...", "start": 0, "end": 5}'
"""

import json
import sys
import os
import subprocess
import time
import gc
import numpy as np

_tts_model = None
_tts_prompt = None
_tts_device = None
_processor = None

def step(msg: str):
    print(f"[TTS] {msg}", file=sys.stderr)

def free_vram(msg=""):
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
        if msg:
            free_mb, total_mb = torch.cuda.mem_get_info()
            used_gb = (total_mb - free_mb) / 1024**3
            total_gb = total_mb / 1024**3
            step(f"VRAM {msg}: {used_gb:.1f}/{total_gb:.1f} GB")
    except:
        pass

def cleanup_model():
    """彻底释放 TTS 模型"""
    global _tts_model, _tts_prompt, _processor
    if _tts_model is not None:
        del _tts_model
        _tts_model = None
    if _tts_prompt is not None:
        del _tts_prompt
        _tts_prompt = None
    if _processor is not None:
        del _processor
        _processor = None
    free_vram("model cleaned")

def load_model():
    """加载 Qwen3-TTS 模型（bfloat16，省显存）"""
    global _tts_model, _processor
    if _tts_model is not None:
        return _tts_model

    import torch
    from qwen_tts import Qwen3TTSModel

    # 使用本地缓存路径（没网络）
    cache_home = os.path.expanduser('~/.cache/huggingface/hub')
    model_path = os.path.join(cache_home,
        'models--Qwen--Qwen3-TTS-12Hz-1.7B-Base',
        'snapshots', 'fd4b254389122332181a7c3db7f27e918eec64e3')
    step(f"Loading 1.7B model from local cache (bfloat16)...")
    t0 = time.time()

    _tts_model = Qwen3TTSModel.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )

    step(f"Loaded in {time.time()-t0:.1f}s, device={_tts_model.device}")
    free_vram("model loaded")
    return _tts_model

def create_prompt(ref_audio: str, ref_text: str):
    """创建语音克隆 prompt（只做一次）"""
    global _tts_prompt
    if _tts_prompt is not None:
        return _tts_prompt

    model = load_model()
    step("Creating voice clone prompt...")
    _tts_prompt = model.create_voice_clone_prompt(
        ref_audio=ref_audio, ref_text=ref_text,
    )
    step("Voice clone prompt ready")
    return _tts_prompt

def gen_segment(model, prompt, text: str, output_path: str, orig_duration: float,
                tts_language: str = 'chinese') -> dict:
    """生成一条语音克隆音频"""
    import soundfile as sf

    if not text.strip():
        step(f"  Empty text, generating silence")
        audio = np.zeros((int(orig_duration * 24000),), dtype=np.float32)
        sf.write(output_path, audio, 24000)
        return {"duration": orig_duration, "sr": 24000}

    step(f"  Generating: {text[:50]}...")
    try:
        audio_chunks, sr = model.generate_voice_clone(
            text=text,
            language=tts_language,
            voice_clone_prompt=prompt,
            non_streaming_mode=True,
        )
        audio = np.concatenate(audio_chunks)
        tts_dur = len(audio) / sr
    except Exception as e:
        step(f"  Failed: {e}, using silence")
        audio = np.zeros((int(orig_duration * 24000),), dtype=np.float32)
        sr = 24000
        tts_dur = orig_duration

    # 速度调整到匹配原始时长
    speed = tts_dur / orig_duration if orig_duration > 0 else 1.0
    speed = max(0.5, min(2.0, speed))

    if abs(speed - 1.0) > 0.05:
        raw_path = output_path.replace(".wav", "_raw.wav")
        adj_path = output_path.replace(".wav", "_adj.wav")
        sf.write(raw_path, audio, sr)
        subprocess.run([
            "ffmpeg", "-y", "-i", raw_path,
            "-filter:a", f"atempo={speed}",
            "-ac", "1", "-ar", "24000", adj_path,
        ], capture_output=True, check=True)

        adj_audio, adj_sr = sf.read(adj_path)
        target_len = int(orig_duration * adj_sr)
        if len(adj_audio) > target_len:
            adj_audio = adj_audio[:target_len]
        elif len(adj_audio) < target_len:
            adj_audio = np.pad(adj_audio, (0, target_len - len(adj_audio)))
        sf.write(output_path, adj_audio, adj_sr)
        for f in [raw_path, adj_path]:
            if os.path.exists(f): os.remove(f)
    else:
        sf.write(output_path, audio, sr)

    return {"duration": orig_duration, "sr": 24000, "text": text}

def run(params: dict) -> dict:
    input_dir = params.get("input_dir", "/mnt/Study/rsv-output")
    output_dir = params.get("output_dir", "/mnt/Study/rsv-output")
    ref_audio = params.get("ref_audio", os.path.join(input_dir, "ref_audio.wav"))
    ref_text_file = params.get("ref_text", os.path.join(input_dir, "ref_text.txt"))
    start_idx = params.get("start", 0)
    end_idx = params.get("end", 999)

    # 读取翻译结果
    trans_file = os.path.join(input_dir, "2_translated.json")
    with open(trans_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    segments = data.get("segments", data if isinstance(data, list) else [])

    if not segments:
        return {"error": "No segments found"}
    if start_idx >= len(segments):
        return {"error": f"Start index {start_idx} >= {len(segments)}"}

    end_idx = min(end_idx, len(segments))
    target_segments = segments[start_idx:end_idx]

    # 读取参考文本
    ref_text = ""
    if os.path.exists(ref_text_file):
        with open(ref_text_file, "r", encoding="utf-8") as f:
            ref_text = f.read().strip()

    step(f"Processing segments [{start_idx}-{end_idx}) ({len(target_segments)} items)")
    step(f"Reference: {os.path.basename(ref_audio)}")

    # 创建输出目录
    tts_dir = os.path.join(output_dir, "tts_audio")
    os.makedirs(tts_dir, exist_ok=True)

    # 加载模型 + prompt
    model = load_model()
    prompt = create_prompt(ref_audio, ref_text)

    # TTS 语言映射
    lang_map = {
        'zh-cn': 'chinese', 'zh-tw': 'chinese', 'zh': 'chinese',
        'en': 'english', 'ja': 'japanese', 'ko': 'korean',
        'fr': 'french', 'de': 'german', 'ru': 'russian', 'es': 'spanish',
    }
    tts_lang = lang_map.get(params.get("tts_language", "zh-cn"), 'chinese')
    step(f"TTS language: {tts_lang}")

    results = []
    for i, seg in enumerate(target_segments):
        idx = start_idx + i
        text = seg["translated_text"]
        start_time = seg["start"]
        end_time = seg["end"]
        duration = end_time - start_time

        # 前面加空格，让 TTS 初始化的"吱"声落在空格时段
        spaced_text = "       " + text
        step(f"\n[{idx+1}/{len(segments)}] {start_time:.1f}s-{end_time:.1f}s ({duration:.1f}s)")
        audio_path = os.path.join(tts_dir, f"seg_{idx:04d}.wav")
        gen_segment(model, prompt, spaced_text, audio_path, duration, tts_lang)

        results.append({
            "index": idx, "start": start_time, "end": end_time,
            "text": text, "audio_path": audio_path,
        })

        step(f"  ✅ Saved: {os.path.basename(audio_path)}")

    # 清理显存
    cleanup_model()
    step(f"\n✅ Done segments [{start_idx}-{end_idx})")

    return {
        "segments": results,
        "count": len(results),
        "range": [start_idx, end_idx],
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Missing JSON input"}))
        sys.exit(1)

    params = json.loads(sys.argv[1])
    result = run(params)
    print(json.dumps(result, ensure_ascii=False))
