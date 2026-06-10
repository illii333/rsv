#!/usr/bin/env python3
"""Test Qwen3-TTS voice cloning with Base model."""
from qwen_tts import Qwen3TTSModel
import numpy as np
import soundfile as sf

model = Qwen3TTSModel.from_pretrained('Qwen/Qwen3-TTS-12Hz-1.7B-Base')

ref_audio = '/mnt/Study/rsv-output/ref_audio/reference.wav'
ref_text = "What's up guys this video is going to be about command mode in helix"

print('Creating voice clone prompt...')
prompt = model.create_voice_clone_prompt(
    ref_audio=ref_audio,
    ref_text=ref_text,
)

print('Generating with cloned voice...')
audio_chunks, sample_rate = model.generate_voice_clone(
    text='大家好，欢迎来到本教程。今天我们学习 Helix 编辑器的命令模式。',
    language='chinese',
    voice_clone_prompt=prompt,
    non_streaming_mode=True,
)

audio = np.concatenate(audio_chunks)
output_path = '/mnt/Study/rsv-output/ref_audio/clone_test.wav'
sf.write(output_path, audio, sample_rate)
print(f'Saved: {output_path} ({len(audio)/sample_rate:.1f}s, {sample_rate}Hz)')
print('DONE')
