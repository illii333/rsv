#!/usr/bin/env python3
"""
RSV - Resume Pipeline from TTS Step
续跑脚本：跳过 ASR + 翻译，直接从语音合成(TTS)开始做。

显存优化：
  - 每个模型加载前释放上一个模型
  - torch.bfloat16 半精度加载
  - 每 N 段清理一次 CUDA 缓存
  - 全部释放后才做音频/视频合成

Usage: python3 resume_tts.py '<json_input>'
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


def step(msg: str):
    print(f"[Pipeline] {msg}", file=sys.stderr)


def free_vram(msg=""):
    """释放 GPU 显存的通用函数"""
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
        if msg:
            free, total = torch.cuda.mem_get_info()
            step(f"VRAM after {msg}: {(total-free)/1024**3:.1f}/{total/1024**3:.1f} GB used")
    except:
        pass


# ============================================================
# TTS with VRAM optimization
# ============================================================

def init_tts_model(use_small_model=False):
    """Load Qwen3-TTS model with VRAM optimizations."""
    global _tts_model
    if _tts_model is not None:
        return _tts_model

    import torch

    model_id = 'Qwen/Qwen3-TTS-12Hz-0.6B-Base' if use_small_model else 'Qwen/Qwen3-TTS-12Hz-1.7B-Base'
    step(f"Loading TTS model '{model_id}' with bfloat16...")

    # Check if small model exists, fallback if not
    if use_small_model:
        import huggingface_hub
        try:
            huggingface_hub.snapshot_download(model_id, allow_patterns="*.safetensors")
        except:
            step("Small model not cached, falling back to 1.7B...")
            model_id = 'Qwen/Qwen3-TTS-12Hz-1.7B-Base'

    t0 = time.time()
    from qwen_tts import Qwen3TTSModel

    _tts_model = Qwen3TTSModel.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )

    step(f"TTS model loaded in {time.time()-t0:.1f}s on device: {_tts_model.device}")
    free_vram("TTS loaded")
    return _tts_model


def init_voice_prompt(model, ref_audio: str, ref_text: str):
    """Create voice clone prompt once."""
    global _tts_prompt
    if _tts_prompt is not None:
        return _tts_prompt

    step("Creating voice clone prompt from reference audio...")
    _tts_prompt = model.create_voice_clone_prompt(
        ref_audio=ref_audio, ref_text=ref_text,
    )
    step("Voice clone prompt ready")
    return _tts_prompt


def run_tts(segments: list, output_dir: str, ref_audio: str, ref_text: str,
            use_small_model=False) -> list:
    """Generate TTS with VRAM-optimized per-segment processing."""
    os.makedirs(output_dir, exist_ok=True)
    import soundfile as sf

    model = init_tts_model(use_small_model)
    prompt = init_voice_prompt(model, ref_audio, ref_text)

    result_segments = []
    total = len(segments)

    for i, seg in enumerate(segments):
        text = seg["translated_text"]
        orig_start = seg["start"]
        orig_end = seg["end"]
        orig_duration = orig_end - orig_start

        if not text.strip():
            silence = np.zeros((int(orig_duration * 24000),), dtype=np.float32)
            audio_path = os.path.join(output_dir, f"seg_{i:04d}.wav")
            sf.write(audio_path, silence, 24000)
            result_segments.append({
                "index": i, "start": orig_start, "end": orig_end,
                "text": text, "audio_path": audio_path, "duration": orig_duration,
            })
            continue

        step(f"  TTS [{i+1}/{total}]: {text[:50]}...")

        try:
            audio_chunks, sr = model.generate_voice_clone(
                text=text,
                language='chinese',
                voice_clone_prompt=prompt,
                non_streaming_mode=True,
            )
            audio = np.concatenate(audio_chunks)
            tts_duration = len(audio) / sr
        except Exception as e:
            step(f"  TTS failed: {e}, using silence")
            audio = np.zeros((int(orig_duration * 24000),), dtype=np.float32)
            sr = 24000
            tts_duration = orig_duration

        # Speed adjustment
        raw_path = os.path.join(output_dir, f"seg_{i:04d}_raw.wav")
        audio_path = os.path.join(output_dir, f"seg_{i:04d}.wav")

        speed_factor = tts_duration / orig_duration if orig_duration > 0 else 1.0
        speed_factor = max(0.5, min(2.0, speed_factor))

        if abs(speed_factor - 1.0) > 0.05:
            sf.write(raw_path, audio, sr)
            adjusted_path = os.path.join(output_dir, f"seg_{i:04d}_adj.wav")
            subprocess.run([
                "ffmpeg", "-y", "-i", raw_path,
                "-filter:a", f"atempo={speed_factor}",
                "-ac", "1", "-ar", "24000",
                adjusted_path,
            ], capture_output=True, check=True)

            adj_audio, adj_sr = sf.read(adjusted_path)
            target_len = int(orig_duration * adj_sr)
            if len(adj_audio) > target_len:
                adj_audio = adj_audio[:target_len]
            elif len(adj_audio) < target_len:
                adj_audio = np.pad(adj_audio, (0, target_len - len(adj_audio)))
            sf.write(audio_path, adj_audio, adj_sr)

            for f in [raw_path, adjusted_path]:
                if os.path.exists(f):
                    os.remove(f)
        else:
            sf.write(audio_path, audio, sr)

        result_segments.append({
            "index": i, "start": orig_start, "end": orig_end,
            "text": text, "audio_path": audio_path, "duration": orig_duration,
        })

        # VRAM cleanup every 5 segments
        if (i + 1) % 5 == 0:
            free_vram(f"TTS segment {i+1}/{total}")

    return result_segments


# ============================================================
# Audio/Video Assembly
# ============================================================

def write_srt(segments: list, srt_path: str):
    def fmt_time(sec: float) -> str:
        h = int(sec // 3600)
        m = int((sec % 3600) // 60)
        s = int(sec % 60)
        ms = int((sec % 1) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    with open(srt_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments):
            text = seg.get("translated_text", seg["text"])
            if not text.strip():
                continue
            f.write(f"{i + 1}\n")
            f.write(f"{fmt_time(seg['start'])} --> {fmt_time(seg['end'])}\n")
            f.write(f"{text}\n\n")


def get_duration(video_path: str) -> float:
    r = subprocess.run([
        "ffprobe", "-v", "quiet",
        "-show_entries", "format=duration",
        "-of", "csv=p=0", video_path,
    ], capture_output=True, text=True, check=True)
    return float(r.stdout.strip())


def assemble_audio(segments: list, video_duration: float, output_path: str, sr: int = 24000):
    """Mix all TTS segments into one aligned track."""
    from pydub import AudioSegment
    step(f"Assembling audio track ({video_duration:.1f}s)...")
    track = AudioSegment.silent(duration=int(video_duration * 1000), frame_rate=sr)

    for i, seg in enumerate(segments):
        if (i + 1) % 10 == 0:
            step(f"  Assembling audio: {i+1}/{len(segments)}")
        audio = AudioSegment.from_wav(seg["audio_path"])
        position = int(seg["start"] * 1000)
        track = track.overlay(audio, position=position)

    track.export(output_path, format="wav")
    step(f"Audio track saved: {output_path}")


def assemble_video(video_path: str, audio_path: str, srt_path: str, output_path: str):
    """Replace audio + burn hard subtitles."""
    step("Assembling final video...")
    temp_video = output_path.replace(".mp4", "_no_subs.mp4")

    subprocess.run([
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", audio_path,
        "-c:v", "libx264", "-crf", "23", "-preset", "fast",
        "-c:a", "aac", "-b:a", "128k",
        "-map", "0:v:0", "-map", "1:a:0",
        "-shortest",
        temp_video,
    ], capture_output=True, check=True)

    subprocess.run([
        "ffmpeg", "-y",
        "-i", temp_video,
        "-vf", f"subtitles={srt_path}:force_style='FontName=Noto Sans CJK SC,FontSize=16,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,BorderStyle=1,Outline=1,Shadow=0,MarginV=10'",
        "-c:a", "copy",
        output_path,
    ], capture_output=True, check=True)

    os.remove(temp_video)
    step(f"✅ Final video: {output_path}")


# ============================================================
# Main
# ============================================================

def run(params: dict) -> dict:
    video_path = params["video_path"]
    input_dir = params.get("input_dir", "")
    output_dir = params.get("output_dir", "output")
    use_small_model = params.get("use_small_model", False)

    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(video_path):
        return {"error": f"Video not found: {video_path}"}

    # --- Load existing ASR + Translation data ---
    trans_file = os.path.join(input_dir, "2_translated.json")
    if not os.path.exists(trans_file):
        trans_file = os.path.join(output_dir, "2_translated.json")
    if not os.path.exists(trans_file):
        return {"error": f"Translated data not found: {trans_file}"}

    with open(trans_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Load reference audio info
    ref_audio = params.get("ref_audio", os.path.join(input_dir, "ref_audio.wav"))
    ref_text_file = params.get("ref_text", os.path.join(input_dir, "ref_text.txt"))

    if not os.path.exists(ref_audio):
        return {"error": f"Reference audio not found: {ref_audio}"}

    ref_text = ""
    if os.path.exists(ref_text_file):
        with open(ref_text_file, "r", encoding="utf-8") as f:
            ref_text = f.read().strip()

    segments = data.get("segments", data if isinstance(data, list) else [])
    if not segments:
        return {"error": "No segments found in translated data"}

    step(f"Loaded {len(segments)} translated segments")
    step(f"Reference: {os.path.basename(ref_audio)}, text: {ref_text[:60]}...")

    # --- Step 3: TTS (Voice Cloning) ---
    step("=" * 50)
    step(f"STEP 3/3: Voice Cloning TTS ({'0.6B' if use_small_model else '1.7B'})")
    step("=" * 50)

    free_vram("before TTS")
    tts_dir = os.path.join(output_dir, "tts_audio")
    os.makedirs(tts_dir, exist_ok=True)

    # Run TTS with VRAM optimization
    tts_segments = run_tts(segments, tts_dir, ref_audio, ref_text, use_small_model)

    # --- Unload TTS model to free VRAM ---
    global _tts_model, _tts_prompt
    step("Releasing TTS model from GPU...")
    del _tts_model
    del _tts_prompt
    _tts_model = None
    _tts_prompt = None
    free_vram("TTS model released")

    # --- Step 4: Audio Assembly ---
    step("=" * 50)
    step("STEP 4/3: Audio Assembly")
    step("=" * 50)
    video_duration = get_duration(video_path)
    assembled_audio = os.path.join(output_dir, "3_chinese_audio.wav")
    assemble_audio(tts_segments, video_duration, assembled_audio)

    # --- Step 5: Video Assembly ---
    step("=" * 50)
    step("STEP 5/3: Video Assembly")
    step("=" * 50)
    srt_path = os.path.join(output_dir, "2_translated.srt")
    if not os.path.exists(srt_path):
        srt_path = os.path.join(output_dir, "subtitles.srt")
        write_srt(segments, srt_path)

    final_video = os.path.join(output_dir, "final.mp4")
    assemble_video(video_path, assembled_audio, srt_path, final_video)

    return {
        "output_video": final_video,
        "subtitle_file": srt_path,
        "segments_count": len(segments),
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Missing JSON input"}))
        sys.exit(1)

    try:
        params = json.loads(sys.argv[1])
    except json.JSONDecodeError as e:
        print(json.dumps({"error": f"Invalid JSON: {e}"}))
        sys.exit(1)

    result = run(params)
    print(json.dumps(result, ensure_ascii=False))
