#!/bin/bash
# RSV 视频翻译管线
# 用法: bash /mnt/Study/rsv/scripts/quick_run.sh <视频文件>
set -e

# CUDA 13 兼容（libcublas.so.12 -> libcublas.so.13）
export LD_LIBRARY_PATH="/tmp/cublas_fix:${LD_LIBRARY_PATH:-}"

VIDEO="$1"
RSV="/mnt/Study/rsv"

if [ -z "$VIDEO" ]; then
    echo "用法: bash $0 <视频文件>"
    exit 1
fi

cd "$RSV"
OUT="output_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"
LANG_FILE="/tmp/rsv_lang_$$"

echo ""
echo "=========================================="
echo "  🎬 RSV 视频翻译"
echo "  视频: $VIDEO"
echo "=========================================="

# ======== ASR ========
echo ""
echo "🔊 [1/4] 语音识别..."
python3 scripts/asr_faster_whisper.py '{"input_path": "'"$VIDEO"'", "model": "large-v3-turbo"}' 2>&1 \
  | tee "$OUT/1_asr.log" | grep "^\[ASR\]"
grep -v "^\[" "$OUT/1_asr.log" | tail -1 > "$OUT/1_asr_raw.json"
echo "✅ ASR 完成"

# ======== 提取参考音频 ========
echo "📝 提取参考音频..."
ffmpeg -y -i "$VIDEO" -ss 0 -t 10 -vn -ac 1 -ar 16000 "$OUT/ref_audio.wav" 2>/dev/null
echo '{"input_path": "'"$OUT"'/ref_audio.wav", "model": "base", "device": "cpu", "compute_type": "int8"}' \
  | python3 -c "
import sys, json, subprocess
p = json.load(sys.stdin)
r = subprocess.run(['python3', 'scripts/asr_faster_whisper.py', json.dumps(p)], capture_output=True, text=True)
d = json.loads(r.stdout.strip())
t = ' '.join(s['text'].strip() for s in d['segments'])
with open('$OUT/ref_text.txt', 'w') as f: f.write(t)
" 2>/dev/null

# ======== 选择语言 ========
flush_stdin() { read -t 0.1 -n 10000 2>/dev/null || true; }
flush_stdin
echo ""
echo "=========================================="
echo "  翻译成什么语言？"
echo "=========================================="
echo "  1) 中文"
echo "  2) 英文"
echo "  3) 日文"
echo "  4) 韩文"
echo "  5) 法文"
echo "  6) 德文"
echo "  7) 俄文"
echo "  8) 西班牙文"
echo "=========================================="
read -p "  请选择 [1]: " LANG_CHOICE
case "${LANG_CHOICE:-1}" in
    2) TGT_LANG="en" ;;
    3) TGT_LANG="ja" ;;
    4) TGT_LANG="ko" ;;
    5) TGT_LANG="fr" ;;
    6) TGT_LANG="de" ;;
    7) TGT_LANG="ru" ;;
    8) TGT_LANG="es" ;;
    *) TGT_LANG="zh-cn" ;;
esac
echo "  → 翻译成: $TGT_LANG"

# ======== 翻译 ========
echo ""
echo "🌐 [2/4] 翻译..."
python3 -c "
import json, subprocess
with open('$OUT/1_asr_raw.json') as f: d = json.load(f)
texts = [s['text'] for s in d['segments']]
r = subprocess.run(['python3', 'scripts/translate_m2m100.py', json.dumps({'texts': texts, 'source_lang': 'en', 'target_lang': '$TGT_LANG', 'device': 'auto'})], capture_output=True, text=True)
trans = json.loads(r.stdout.strip())['translations']
for i, s in enumerate(d['segments']):
    s['translated_text'] = trans[i] if i < len(trans) else s['text']
with open('$OUT/2_translated.json', 'w') as f: json.dump({'segments': d['segments']}, f, ensure_ascii=False, indent=2)
print(f'翻译完成: {len(trans)} 条')
"
# 写 SRT
python3 -c "
import json
def fmt(t):
    h=int(t//3600); m=int((t%3600)//60); s=int(t%60); ms=int((t%1)*1000)
    return f'{h:02d}:{m:02d}:{s:02d},{ms:03d}'
with open('$OUT/2_translated.json') as f: d = json.load(f)
with open('$OUT/2_translated.srt', 'w') as f:
    for i, s in enumerate(d['segments']):
        t = s.get('translated_text', s['text'])
        if not t.strip(): continue
        f.write(f'{i+1}\n{fmt(s[\"start\"])} --> {fmt(s[\"end\"])}\n{t}\n\n')
print('SRT 已生成')
"

# ======== 审查翻译结果 ========
echo ""
echo "=========================================="
echo "  翻译结果 — 修改 2_translated.srt 后按 Y 继续"
echo "=========================================="
python3 -c "
import json
with open('$OUT/2_translated.json') as f: d = json.load(f)
for i, s in enumerate(d['segments']):
    print(f'  [{i+1}] {s[\"start\"]:5.1f}s: {s.get(\"translated_text\", s[\"text\"])}')
"
echo "=========================================="
echo "  改这个文件: $RSV/$OUT/2_translated.srt"
echo "  改完之后按 Y，脚本自动同步到语音合成"
echo "=========================================="
flush_stdin
read -p "  ⏸️  继续语音合成？(Y/n) " cont
if [[ "${cont:-y}" =~ ^[Nn] ]]; then
    echo "已暂停。改完 .srt 后重新运行即可。"
    exit 0
fi

# ======== 把 .srt 同步回 .json ========
echo "🔄 同步字幕到语音..."
python3 -c "
import json, re
# 读 SRT
with open('$OUT/2_translated.srt', 'r') as f: srt_text = f.read()
# 解析 SRT 提取文本
lines = srt_text.strip().split(chr(10))
texts = []
for line in lines:
    if re.match(r'^\d+$', line.strip()): continue    # 序号
    if re.match(r'^\d+:\d+:\d+', line.strip()): continue  # 时间轴
    if line.strip() == '': continue
    texts.append(line.strip())
# 更新 JSON
with open('$OUT/2_translated.json', 'r') as f: d = json.load(f)
for i, s in enumerate(d['segments']):
    if i < len(texts) and texts[i]:
        s['translated_text'] = texts[i]
with open('$OUT/2_translated.json', 'w') as f: json.dump(d, f, ensure_ascii=False, indent=2)
print(f'同步完成: {len(texts)} 条')
"

# ======== TTS（自动用翻译的语言） ========
echo ""
echo "🔊 [3/4] 语音合成 ($TGT_LANG)..."
python3 scripts/tts_segment.py '{"input_dir": "'"$OUT"'", "output_dir": "'"$OUT"'", "start": 0, "end": 999, "tts_language": "'"$TGT_LANG"'"}' 2>&1 | grep "^\[TTS\]" || true
echo "✅ 语音合成完成"

# ======== 合成视频 ========
echo ""
echo "🎬 [4/4] 合成视频..."
python3 scripts/assemble.py '{"input_dir": "'"$OUT"'", "output_dir": "'"$OUT"'", "video_path": "'"$VIDEO"'"}' 2>&1 | grep "^\[Assemble\]" || true

echo ""
echo "=========================================="
echo "  🎉 完成！"
echo "=========================================="
echo "  视频: $RSV/$OUT/final.mp4"
echo "  字幕: $RSV/$OUT/2_translated.srt"
echo "=========================================="
