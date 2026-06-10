#!/usr/bin/env python3
"""
RSV - ASR Bridge: Faster-Whisper 语音识别

支持音频/视频输入，自动提取音频轨道。

Usage: python3 asr_faster_whisper.py '<json_input>'

JSON input:
{
    "input_path": "/path/to/audio.wav 或 video.mp4",
    "model": "large-v3-turbo",
    "device": "auto",
    "compute_type": "float16",
    "language": "auto"
}

JSON output (stdout):
{
    "segments": [
        {"start": 0.0, "end": 2.5, "text": "Hello", "confidence": 0.95, "speaker": null}
    ],
    "language": "en"
}
"""

import os, ctypes

# 修复 libcublas.so.12 找不到的问题（CUDA 13 兼容）
_cublas_fix = '/tmp/cublas_fix/libcublas.so.12'
if not os.path.exists(_cublas_fix):
    try:
        os.makedirs('/tmp/cublas_fix', exist_ok=True)
        _src = '/opt/cuda/lib64/libcublas.so.13'
        if os.path.exists(_src):
            os.symlink(_src, _cublas_fix)
    except Exception:
        pass
# 预加载 libcublas（让 ctranslate2 能找到）
try:
    ctypes.CDLL(_cublas_fix, ctypes.RTLD_GLOBAL)
except Exception:
    pass

import json
import sys
import tempfile


VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".avi", ".flv", ".webm", ".m4v", ".ts", ".mts"}
AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".aac", ".ogg", ".opus", ".wma"}


def extract_audio(input_path: str, output_wav: str):
    """Extract audio from video file using PyAV (installed as faster-whisper dependency)."""
    try:
        import av
    except ImportError:
        import subprocess
        print(f"[ASR] Using ffmpeg to extract audio from video...", file=sys.stderr)
        cmd = [
            "ffmpeg", "-y",
            "-i", input_path,
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            output_wav,
        ]
        subprocess.run(cmd, capture_output=True, check=True)
        return

    print(f"[ASR] Extracting audio from video using PyAV...", file=sys.stderr)
    container = av.open(input_path)
    stream = next(s for s in container.streams if s.type == "audio")

    from av.audio.resampler import AudioResampler
    resampler = AudioResampler(format="s16", layout="mono", rate=16000)

    output_container = av.open(output_wav, mode="w")
    output_stream = output_container.add_stream("pcm_s16le", rate=16000, layout="mono")

    for frame in container.decode(audio=0):
        if frame is None:
            continue
        resampled = resampler.resample(frame)
        for rframe in resampled:
            for packet in output_stream.encode(rframe):
                output_container.mux(packet)

    for packet in output_stream.encode(None):
        output_container.mux(packet)

    output_container.close()
    container.close()
    print(f"[ASR] Audio extracted: {output_wav}", file=sys.stderr)


def run_faster_whisper(params: dict) -> dict:
    """
    Run Faster-Whisper transcription.

    Requires: pip install faster-whisper
    """
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        return {
            "error": "faster-whisper not installed. Run: pip install faster-whisper",
            "segments": [],
            "language": None,
        }

    input_path = params["input_path"]
    model_name = params.get("model", "large-v3-turbo")
    device = params.get("device", "auto")
    compute_type = params.get("compute_type", "float16")
    language = params.get("language", "auto")

    if not os.path.exists(input_path):
        return {
            "error": f"File not found: {input_path}",
            "segments": [],
            "language": None,
        }

    # Resolve device — try CUDA, fall back to CPU gracefully
    if device == "auto":
        try:
            import ctranslate2
            if ctranslate2.get_cuda_device_count() > 0:
                device = "cuda"
            else:
                raise RuntimeError("No CUDA devices")
        except Exception:
            device = "cpu"
            compute_type = "int8"
        print(f"[ASR] Device auto-detected: {device}", file=sys.stderr)

    # Check if input is video — extract audio if so
    ext = os.path.splitext(input_path)[1].lower()
    audio_path = input_path
    cleanup_wav = False
    if ext in VIDEO_EXTENSIONS:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        audio_path = tmp.name
        try:
            extract_audio(input_path, audio_path)
        except Exception as e:
            os.unlink(audio_path)
            return {
                "error": f"Failed to extract audio from video: {e}",
                "segments": [],
                "language": None,
            }
        cleanup_wav = True

    try:
        print(f"[ASR] Loading model '{model_name}' on {device} ({compute_type})", file=sys.stderr)
        model = WhisperModel(model_name, device=device, compute_type=compute_type)

        print(f"[ASR] Transcribing: {input_path}", file=sys.stderr)
        segments, info = model.transcribe(
            audio_path,
            beam_size=5,
            language=language if language != "auto" else None,
        )

        result_segments = []
        for seg in segments:
            result_segments.append({
                "start": round(seg.start, 3),
                "end": round(seg.end, 3),
                "text": seg.text.strip(),
                "confidence": round(seg.avg_logprob if hasattr(seg, 'avg_logprob') else 0.0, 4),
                "speaker": None,
                "no_speech_prob": round(seg.no_speech_prob if hasattr(seg, 'no_speech_prob') else 0.0, 4),
            })

        detected_language = info.language if info else None
        print(f"[ASR] Done: {len(result_segments)} segments, language={detected_language}", file=sys.stderr)

        return {
            "segments": result_segments,
            "language": detected_language,
        }
    finally:
        if cleanup_wav and os.path.exists(audio_path):
            os.unlink(audio_path)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Missing JSON input argument", "segments": [], "language": None}))
        sys.exit(1)

    try:
        params = json.loads(sys.argv[1])
    except json.JSONDecodeError as e:
        print(json.dumps({"error": f"Invalid JSON: {e}", "segments": [], "language": None}))
        sys.exit(1)

    result = run_faster_whisper(params)
    print(json.dumps(result, ensure_ascii=False))
