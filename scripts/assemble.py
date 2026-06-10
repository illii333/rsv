#!/usr/bin/env python3
"""
RSV - 合成最终视频
步骤：
  1. 按顺序拼接所有 TTS 音频 → 一个大 WAV
  2. 替换视频音轨 + 烧录字幕

Usage: python3 assemble.py '<json>'
"""

import json
import sys
import os
import subprocess
import numpy as np
from pydub import AudioSegment


def step(msg):
    print(f"[Assemble] {msg}", file=sys.stderr)


def load_segments(translated_json: str) -> list:
    with open(translated_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("segments", data if isinstance(data, list) else [])


def trim_leading_noise(audio_seg, sample_rate=24000, max_trim_ms=300):
    """自动检测并切掉开头噪音，最多切 max_trim_ms 毫秒"""
    # 转 numpy 分析
    raw = np.array(audio_seg.get_array_of_samples()).astype(np.float32)
    if audio_seg.channels > 1:
        raw = raw.reshape(-1, audio_seg.channels).mean(axis=1)

    # 计算短时能量，找第一个"有声音"的位置
    win_ms = 20                     # 20ms 窗口
    win_samples = int(win_ms * sample_rate / 1000)
    threshold = 0.015               # 振幅阈值

    max_samples = int(max_trim_ms * sample_rate / 1000)

    # 从开头逐步滑动窗口，直到窗口内最大振幅超过阈值
    for start in range(0, min(len(raw), max_samples), win_samples):
        end = min(start + win_samples, len(raw))
        if np.max(np.abs(raw[start:end])) > threshold:
            trim_pos = start
            break
    else:
        trim_pos = 0

    # 再额外多切 15ms 确保开头突变更稳
    extra = int(15 * sample_rate / 1000)
    trim_pos = max(0, trim_pos - extra)

    # 转成毫秒（pydub 的 slice 用 ms）
    trim_ms = int(trim_pos / sample_rate * 1000)
    if trim_ms > 20:  # 至少切 20ms，否则没必要
        return audio_seg[trim_ms:], trim_ms
    return audio_seg, 0


def assemble_audio(segments: list, tts_dir: str, video_duration: float,
                   output_path: str, sample_rate: int = 24000) -> str:
    """拼接所有 TTS 音频到一条音轨（自动切静音 + 淡入淡出）"""

    step(f"Creating silent track ({video_duration:.1f}s, {sample_rate}Hz)...")
    track = AudioSegment.silent(duration=int(video_duration * 1000),
                                frame_rate=sample_rate)

    FADE_IN = 20
    FADE_OUT = 20

    for i, seg in enumerate(segments):
        idx = seg.get("index", i)
        audio_path = os.path.join(tts_dir, f"seg_{idx:04d}.wav")
        if not os.path.exists(audio_path):
            step(f"  ⚠️ Missing: {audio_path}, skipping")
            continue

        audio = AudioSegment.from_wav(audio_path)

        # 自动检测开头静音并切掉（只切噪音/空格停顿，不切到说话）
        trimmed_audio, trimmed_ms = trim_leading_noise(audio, sample_rate)
        audio = trimmed_audio

        # 淡入淡出
        audio = audio.fade_in(FADE_IN).fade_out(FADE_OUT)

        position = int(seg["start"] * 1000)
        track = track.overlay(audio, position=position)

        if (i + 1) % 10 == 0:
            step(f"  Mixed {i+1}/{len(segments)}")

    step(f"Exporting mixed audio: {output_path}")
    track.export(output_path, format="wav")
    step(f"✅ Audio track: {output_path} ({os.path.getsize(output_path)/1024**2:.1f} MB)")
    return output_path


def get_duration(video_path: str) -> float:
    r = subprocess.run([
        "ffprobe", "-v", "quiet",
        "-show_entries", "format=duration",
        "-of", "csv=p=0", video_path,
    ], capture_output=True, text=True, check=True)
    return float(r.stdout.strip())


def assemble_video(video_path: str, audio_path: str, srt_path: str,
                   output_path: str):
    """替换视频音轨 + 烧录字幕"""
    step("Replacing audio track...")
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

    step("Burning subtitles...")
    style = ("FontName=Noto Sans CJK SC,FontSize=16,"
             "PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,"
             "BorderStyle=1,Outline=1,Shadow=0,MarginV=10")
    subprocess.run([
        "ffmpeg", "-y",
        "-i", temp_video,
        "-vf", f"subtitles={srt_path}:force_style='{style}'",
        "-c:a", "copy",
        output_path,
    ], capture_output=True, check=True)

    os.remove(temp_video)
    step(f"✅ Final video: {output_path}")


def run(params: dict) -> dict:
    input_dir = params.get("input_dir", "/mnt/Study/rsv-output")
    output_dir = params.get("output_dir", "/mnt/Study/rsv-output")
    video_path = params.get("video_path", "/mnt/Study/HelixEditor.mp4")
    tts_dir = params.get("tts_dir", os.path.join(output_dir, "tts_audio"))

    # Load segments
    trans_file = os.path.join(output_dir, "2_translated.json")
    segments = load_segments(trans_file)
    step(f"Loaded {len(segments)} segments from {trans_file}")

    # Get video duration
    video_duration = get_duration(video_path)
    step(f"Video: {video_path} ({video_duration:.1f}s)")

    # Step 1: Assemble audio
    audio_file = os.path.join(output_dir, "3_chinese_audio.wav")
    assemble_audio(segments, tts_dir, video_duration, audio_file)

    # Step 2: Assemble video
    srt_file = os.path.join(output_dir, "2_translated.srt")
    final_video = os.path.join(output_dir, "final.mp4")
    assemble_video(video_path, audio_file, srt_file, final_video)

    return {
        "output_video": final_video,
        "audio_track": audio_file,
        "segments": len(segments),
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        params = {}
    else:
        params = json.loads(sys.argv[1])

    result = run(params)
    print(json.dumps(result, ensure_ascii=False))
