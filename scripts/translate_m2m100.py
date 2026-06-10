#!/usr/bin/env python3
"""
RSV - Translation Bridge: M2M100 多语言翻译

Usage: python3 translate_m2m100.py '<json_input>'

JSON input:
{
    "texts": ["Hello world", "How are you?"],
    "source_lang": "en",
    "target_lang": "zh-cn",
    "model": "m2m100_418M",
    "device": "auto",
    "beam_size": 5,
    "max_length": 256
}

JSON output (stdout):
{
    "translations": ["你好世界", "你好吗？"]
}
"""

import json
import sys
import os


def run_m2m100(params: dict) -> dict:
    """
    Run M2M100 translation.

    Requires: pip install transformers torch sentencepiece
    """
    try:
        from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
    except ImportError:
        return {
            "error": (
                "transformers not installed. Run:\n"
                "  pip install transformers torch sentencepiece"
            ),
            "translations": [],
        }

    texts = params["texts"]
    source_lang = params["source_lang"]
    target_lang = params["target_lang"]
    model_name = params.get("model", "m2m100_418M")
    device = params.get("device", "auto")
    beam_size = params.get("beam_size", 5)
    max_length = params.get("max_length", 256)

    # Map language codes to M2M100 codes
    lang_map = {
        "zh-cn": "zh",
        "zh-tw": "zh",
        "en": "en",
        "ja": "ja",
        "ko": "ko",
        "fr": "fr",
        "de": "de",
        "ru": "ru",
        "es": "es",
        "th": "th",
    }
    src_code = lang_map.get(source_lang, source_lang)
    tgt_code = lang_map.get(target_lang, target_lang)

    if device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[Translate] Loading model '{model_name}' on {device}", file=sys.stderr)

    # 使用本地缓存路径（没网络时）
    cache_home = os.path.expanduser('~/.cache/huggingface/hub')
    model_path = os.path.join(cache_home,
        'models--facebook--m2m100_418M',
        'snapshots', '55c2e61bbf05dfb8d7abccdc3fae6fc8512fd636')
    if os.path.exists(model_path):
        model = M2M100ForConditionalGeneration.from_pretrained(model_path)
        tokenizer = M2M100Tokenizer.from_pretrained(model_path)
    else:
        model = M2M100ForConditionalGeneration.from_pretrained(model_name)
        tokenizer = M2M100Tokenizer.from_pretrained(model_name)
    model = model.to(device)
    tokenizer.src_lang = src_code

    print(f"[Translate] Translating {len(texts)} texts: {src_code} → {tgt_code}", file=sys.stderr)

    translations = []
    # Process in batches to avoid OOM
    batch_size = 16
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        encoded = tokenizer(batch, return_tensors="pt", padding=True).to(device)
        generated_tokens = model.generate(
            **encoded,
            forced_bos_token_id=tokenizer.get_lang_id(tgt_code),
            max_length=max_length,
            num_beams=beam_size,
        )
        batch_translations = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        translations.extend(batch_translations)

    print(f"[Translate] Done: {len(translations)} translations", file=sys.stderr)

    return {"translations": translations}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Missing JSON input argument", "translations": []}))
        sys.exit(1)

    try:
        params = json.loads(sys.argv[1])
    except json.JSONDecodeError as e:
        print(json.dumps({"error": f"Invalid JSON: {e}", "translations": []}))
        sys.exit(1)

    result = run_m2m100(params)
    print(json.dumps(result, ensure_ascii=False))
