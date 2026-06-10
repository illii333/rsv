#!/usr/bin/env python3
"""
RSV - Full Pipeline: ASR → Translate → Qwen3-TTS Voice Clone → Video Assembly

Usage: python3 pipeline.py '<json_input>'
"""

import json
import sys
import os
import subprocess
import time
import numpy as np
import soundfile as sf

_tts_model = None
_tts_prompt = None


def step(msg: str):
    print(f"[Pipeline] {msg}", file=sys.stderr)


# ============================================================
# Step 0: Extract reference audio for voice cloning
# ============================================================

def extract_reference_audio(video_path: str, output_dir: str, duration: float = 10.0) -> dict:
    """Extract first N seconds as voice clone reference."""
    step(f"Extracting reference audio (first {duration}s) for voice cloning...")
    ref_audio = os.path.join(output_dir, "ref_audio.wav")
    ref_text_file = os.path.join(output_dir, "ref_text.txt")

    subprocess.run([
        "ffmpeg", "-y",
        "-i", video_path,
        "-ss", "0",
        "-t", str(duration),
        "-vn", "-ac", "1", "-ar", "16000",
        ref_audio,
    ], capture_output=True, check=True)

    # We'll do ASR on this reference to get the text
    step(f"Reference audio saved: {ref_audio}")
    return {"ref_audio": ref_audio, "ref_text_file": ref_text_file}


def transcribe_reference(ref_audio: str) -> str:
    """Use Faster-Whisper to get the transcript of the reference audio."""
    from faster_whisper import WhisperModel
    step("Transcribing reference audio...")
    model = WhisperModel("base", device="cpu", compute_type="int8")
    segments, info = model.transcribe(ref_audio, beam_size=5, language="en")
    text = " ".join(seg.text.strip() for seg in segments)
    step(f"Reference text: '{text[:80]}...'")
    return text


# ============================================================
# Step 1: ASR
# ============================================================

def run_asr(params: dict) -> dict:
    from faster_whisper import WhisperModel
    video_path = params["video_path"]
    model_name = params.get("asr_model", "base")

    device = "cpu"
    try:
        import ctranslate2
        ctranslate2.can_use_cuda(raise_exception=True)
        device = "cuda"
        compute = "float16"
    except Exception:
        device = "cpu"
        compute = "int8"
    step(f"ASR device: {device}")

    step(f"Loading ASR model '{model_name}'...")
    model = WhisperModel(model_name, device=device, compute_type=compute)

    step(f"Transcribing...")
    segments, info = model.transcribe(
        video_path, beam_size=5,
        language=params.get("source_lang", "en"),
        vad_filter=True,
    )

    result = []
    for seg in segments:
        result.append({
            "start": round(seg.start, 3),
            "end": round(seg.end, 3),
            "text": seg.text.strip(),
            "confidence": round(seg.avg_logprob if hasattr(seg, 'avg_logprob') else 0.0, 4),
        })

    lang = info.language if info else params.get("source_lang", "en")
    step(f"ASR done: {len(result)} segments, language={lang}")
    return {"segments": result, "language": lang}


# ============================================================
# Step 2: Translation (M2M100)
# ============================================================

def run_translate(segments: list, source_lang: str, target_lang: str, model_name: str) -> list:
    from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
    import torch

    lang_map = {"en": "en", "zh-cn": "zh", "zh-tw": "zh", "ja": "ja", "ko": "ko",
                "fr": "fr", "de": "de", "ru": "ru", "es": "es", "th": "th"}
    src_code = lang_map.get(source_lang, source_lang)
    tgt_code = lang_map.get(target_lang, target_lang)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    step(f"Loading translation model on {device}...")
    model = M2M100ForConditionalGeneration.from_pretrained(model_name)
    tokenizer = M2M100Tokenizer.from_pretrained(model_name)
    model = model.to(device)
    tokenizer.src_lang = src_code

    texts = [seg["text"] for seg in segments]
    translations = []
    step(f"Translating {len(texts)} segments: {source_lang} → {target_lang}...")

    batch_size = 8
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        encoded = tokenizer(batch, return_tensors="pt", padding=True).to(device)
        generated = model.generate(
            **encoded,
            forced_bos_token_id=tokenizer.get_lang_id(tgt_code),
            max_length=256, num_beams=5,
        )
        batch_trans = tokenizer.batch_decode(generated, skip_special_tokens=True)
        translations.extend(batch_trans)

    step(f"Translation done: {len(translations)} segments")
    result = []
    for seg, trans in zip(segments, translations):
        result.append({**seg, "translated_text": trans})
    return result


# ============================================================
# Step 3: Qwen3-TTS Voice Cloning
# ============================================================

def init_tts_model():
    """Load Qwen3-TTS Base model once (singleton)."""
    global _tts_model
    if _tts_model is None:
        from qwen_tts import Qwen3TTSModel
        step("Loading Qwen3-TTS-12Hz-1.7B-Base for voice cloning...")
        t0 = time.time()
        _tts_model = Qwen3TTSModel.from_pretrained('Qwen/Qwen3-TTS-12Hz-1.7B-Base')
        step(f"TTS model loaded in {time.time()-t0:.1f}s")
    return _tts_model


def init_voice_prompt(model, ref_audio: str, ref_text: str):
    """Create voice clone prompt once."""
    global _tts_prompt
    if _tts_prompt is None:
        step("Creating voice clone prompt from reference audio...")
        _tts_prompt = model.create_voice_clone_prompt(
            ref_audio=ref_audio, ref_text=ref_text,
        )
        step("Voice clone prompt ready")
    return _tts_prompt


def run_vc_tts(segments: list, output_dir: str, ref_audio: str, ref_text: str) -> list:
    """Generate TTS with voice cloning using Qwen3-TTS."""
    os.makedirs(output_dir, exist_ok=True)
    model = init_tts_model()
    prompt = init_voice_prompt(model, ref_audio, ref_text)

    result_segments = []

    for i, seg in enumerate(segments):
        text = seg["translated_text"]
        orig_start = seg["start"]
        orig_end = seg["end"]
        orig_duration = orig_end - orig_start

        if not text.strip():
            # Silence for empty segments
            silence = np.zeros((int(orig_duration * 24000),), dtype=np.float32)
            audio_path = os.path.join(output_dir, f"seg_{i:04d}.wav")
            sf.write(audio_path, silence, 24000)
            result_segments.append({
                "index": i, "start": orig_start, "end": orig_end,
                "text": text, "audio_path": audio_path, "duration": orig_duration,
            })
            continue

        step(f"  TTS [{i+1}/{len(segments)}]: {text[:40]}...")

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
            step(f"  TTS failed for segment {i}: {e}, using silence")
            audio = np.zeros((int(orig_duration * 24000),), dtype=np.float32)
            sr = 24000
            tts_duration = orig_duration

        # Speed adjustment to match original duration
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

        if i >= 1 and i % 5 == 0:
            step(f"  TTS progress: {i}/{len(segments)}")

        result_segments.append({
            "index": i, "start": orig_start, "end": orig_end,
            "text": text, "audio_path": audio_path, "duration": orig_duration,
        })

    return result_segments


# ============================================================
# Step 4: Audio Assembly
# ============================================================

def assemble_audio(segments: list, video_duration: float, output_path: str, sr: int = 24000):
    """Mix all TTS segments into one aligned track."""
    from pydub import AudioSegment
    step(f"Assembling audio track ({video_duration:.1f}s)...")
    track = AudioSegment.silent(duration=int(video_duration * 1000), frame_rate=sr)

    for seg in segments:
        audio = AudioSegment.from_wav(seg["audio_path"])
        position = int(seg["start"] * 1000)
        track = track.overlay(audio, position=position)

    track.export(output_path, format="wav")
    step(f"Audio track saved: {output_path}")


# ============================================================
# Step 5: Video Assembly
# ============================================================

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
# SRT Writer
# ============================================================

def write_srt(segments: list, srt_path: str, use_translated: bool = True):
    def fmt_time(sec: float) -> str:
        h = int(sec // 3600)
        m = int((sec % 3600) // 60)
        s = int(sec % 60)
        ms = int((sec % 1) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    with open(srt_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments):
            text = seg.get("translated_text", seg["text"]) if use_translated else seg["text"]
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


# ============================================================
# Main
# ============================================================

def run_pipeline(params: dict) -> dict:
    video_path = params["video_path"]
    output_dir = params.get("output_dir", "output")
    source_lang = params.get("source_lang", "en")
    target_lang = params.get("target_lang", "zh-cn")
    asr_model = params.get("asr_model", "base")
    translate_model = params.get("translate_model", "facebook/m2m100_418M")

    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(video_path):
        return {"error": f"Video not found: {video_path}"}

    # ---- Step 0: Extract reference audio ----
    step("=" * 50)
    step("STEP 0/5: Extract Reference Audio for Voice Cloning")
    step("=" * 50)
    ref_info = extract_reference_audio(video_path, output_dir)
    ref_text = transcribe_reference(ref_info["ref_audio"])

    with open(ref_info["ref_text_file"], "w", encoding="utf-8") as f:
        f.write(ref_text)

    # ---- Step 1: ASR ----
    step("=" * 50)
    step("STEP 1/5: Speech Recognition (ASR)")
    step("=" * 50)
    asr_result = run_asr(params)
    if "error" in asr_result:
        return asr_result
    segments = asr_result["segments"]
    if not segments:
        return {"error": "No speech detected"}

    with open(os.path.join(output_dir, "1_asr_raw.json"), "w", encoding="utf-8") as f:
        json.dump({"segments": segments}, f, ensure_ascii=False, indent=2)
    write_srt(segments, os.path.join(output_dir, "1_asr_original.srt"), use_translated=False)

    # ---- Step 2: Translate ----
    step("=" * 50)
    step("STEP 2/5: Translation")
    step("=" * 50)
    valid_segments = [s for s in segments if s["text"].strip()]
    translated = run_translate(valid_segments, source_lang, target_lang, translate_model)

    # Merge translations
    trans_map = {}
    for seg in translated:
        trans_map[seg["start"]] = seg["translated_text"]
    completed_segments = []
    for seg in segments:
        seg["translated_text"] = trans_map.get(seg["start"], seg["text"])
        completed_segments.append(seg)

    write_srt(completed_segments, os.path.join(output_dir, "2_translated.srt"))
    with open(os.path.join(output_dir, "2_translated.json"), "w", encoding="utf-8") as f:
        json.dump({"segments": completed_segments}, f, ensure_ascii=False, indent=2)

    # ---- Step 3: Qwen3-TTS Voice Cloning ----
    step("=" * 50)
    step("STEP 3/5: Voice Cloning TTS (Qwen3-TTS)")
    step("=" * 50)
    tts_dir = os.path.join(output_dir, "tts_audio")
    tts_segments = run_vc_tts(completed_segments, tts_dir, ref_info["ref_audio"], ref_text)

    # ---- Step 4: Audio Assembly ----
    step("=" * 50)
    step("STEP 4/5: Audio Assembly")
    step("=" * 50)
    video_duration = get_duration(video_path)
    assembled_audio = os.path.join(output_dir, "3_chinese_audio.wav")
    assemble_audio(tts_segments, video_duration, assembled_audio)

    # ---- Step 5: Video Assembly ----
    step("=" * 50)
    step("STEP 5/5: Video Assembly")
    step("=" * 50)
    srt_path = os.path.join(output_dir, "2_translated.srt")
    final_video = os.path.join(output_dir, "final.mp4")
    assemble_video(video_path, assembled_audio, srt_path, final_video)

    return {
        "output_video": final_video,
        "subtitle_file": srt_path,
        "segments_count": len(completed_segments),
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

    result = run_pipeline(params)
    print(json.dumps(result, ensure_ascii=False))
