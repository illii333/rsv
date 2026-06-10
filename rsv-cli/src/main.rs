use clap::{Parser, Subcommand};
use rsv_core::config::RsvConfig;
use std::path::PathBuf;

/// RSV — Rust Speech/Video Toolkit
///
/// 语音识别 (ASR) + 翻译 (Translate) + 语音克隆 (TTS)
/// 基于 Faster-Whisper / M2M100 / Qwen3-TTS
#[derive(Parser)]
#[command(name = "rsv")]
#[command(version, about, long_about)]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Config file path
    #[arg(short, long, default_value = "rsv.config.json")]
    config: PathBuf,
}

#[derive(Subcommand)]
enum Commands {
    /// 语音识别 — 将音频/视频转写为字幕
    Asr {
        /// 输入文件 (wav/mp4/mkv/mp3...)
        input: PathBuf,

        /// 输出字幕文件 (srt)
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// 模型大小: tiny/base/small/medium/large-v3/large-v3-turbo
        #[arg(short, long, default_value = "large-v3-turbo")]
        model: String,
    },

    /// 翻译 — 将字幕文本翻译为目标语言
    Translate {
        /// 输入文本或字幕文件
        input: String,

        /// 源语言代码 (zh-cn/en/ja/ko/fr/de/ru/es/th)
        #[arg(short, long)]
        source: String,

        /// 目标语言代码
        #[arg(short, long)]
        target: String,

        /// 输入是文件路径（SRT 或 JSON）
        #[arg(long)]
        file: bool,
    },

    /// 语音合成/克隆 — 文本转语音
    Tts {
        /// 输入文本或字幕文件
        input: String,

        /// 输出目录
        #[arg(short, long, default_value = "output/tts")]
        output: PathBuf,

        /// 参考音频（用于语音克隆）
        #[arg(short, long)]
        reference_audio: Option<PathBuf>,

        /// 参考音频对应的文本
        #[arg(short = 't', long)]
        reference_text: Option<String>,
    },

    /// 🎬 完整视频翻译管线: ASR → 翻译 → 配音 → 输出视频
    Run {
        /// 输入视频文件
        input: PathBuf,

        /// 输出目录
        #[arg(short, long, default_value = "output")]
        output: PathBuf,

        /// 源语言代码
        #[arg(short = 's', long, default_value = "en")]
        source: String,

        /// 目标语言代码
        #[arg(short = 't', long, default_value = "zh-cn")]
        target: String,

        /// ASR 模型
        #[arg(short = 'm', long, default_value = "base")]
        asr_model: String,

        /// TTS 语音
        #[arg(long, default_value = "zh-CN-XiaoxiaoNeural")]
        tts_voice: String,
    },
}

fn main() -> anyhow::Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        )
        .init();

    let cli = Cli::parse();

    // Load config
    let config = if cli.config.exists() {
        RsvConfig::from_file(&cli.config)?
    } else {
        tracing::warn!("Config file not found, using defaults: {}", cli.config.display());
        let mut cfg = RsvConfig::default();
        cfg.scripts_dir = PathBuf::from("scripts");
        cfg
    };

    match &cli.command {
        Commands::Asr { input, output, model } => {
            run_asr(input, output, model, &config)?;
        }
        Commands::Translate { input, source, target, file } => {
            run_translate(input, source, target, *file, &config)?;
        }
        Commands::Tts { input, output, reference_audio, reference_text } => {
            run_tts(input, output, reference_audio.as_ref(), reference_text.as_ref(), &config)?;
        }
        Commands::Run { input, output, source, target, asr_model, tts_voice } => {
            run_pipeline(input, output, source, target, asr_model, tts_voice, &config)?;
        }
    }

    Ok(())
}

// ---- ASR ----

fn run_asr(
    input: &PathBuf,
    output: &Option<PathBuf>,
    model: &str,
    config: &RsvConfig,
) -> anyhow::Result<()> {
    let mut cfg = config.clone();
    cfg.asr.model = model.to_string();

    let engine = rsv_asr::create_faster_whisper();
    tracing::info!("🎤 Using ASR engine: {}", engine.name());

    let result = engine.transcribe(
        input.to_str().unwrap(),
        &cfg,
    )?;

    println!("\n✅ Transcribed {} segments\n", result.segments.len());
    for seg in &result.segments {
        println!(
            "  [{:>6.1}s -> {:>6.1}s] {:>5.1}%  {}",
            seg.start,
            seg.end,
            seg.confidence * 100.0,
            seg.text,
        );
    }

    // Save output
    let output_path = output
        .clone()
        .unwrap_or_else(|| PathBuf::from("output.srt"));

    let mut srt = String::new();
    for entry in &result.subtitles.entries {
        srt.push_str(&format!("{}\n", entry.index));
        srt.push_str(&format!(
            "{:02}:{:02}:{:02},{:03} --> {:02}:{:02}:{:02},{:03}\n",
            (entry.start as u32) / 3600,
            ((entry.start as u32) % 3600) / 60,
            (entry.start as u32) % 60,
            (entry.start.fract() * 1000.0) as u32,
            (entry.end as u32) / 3600,
            ((entry.end as u32) % 3600) / 60,
            (entry.end as u32) % 60,
            (entry.end.fract() * 1000.0) as u32,
        ));
        srt.push_str(&format!("{}\n\n", entry.text));
    }
    std::fs::write(&output_path, srt)?;
    tracing::info!("💾 Saved subtitles to: {}", output_path.display());

    Ok(())
}

// ---- Translate ----

fn run_translate(
    input: &str,
    source: &str,
    target: &str,
    is_file: bool,
    config: &RsvConfig,
) -> anyhow::Result<()> {
    let texts: Vec<String> = if is_file {
        let path = PathBuf::from(input);
        let content = std::fs::read_to_string(&path)?;
        content.lines().map(|l| l.to_string()).collect()
    } else {
        vec![input.to_string()]
    };

    let engine = rsv_translate::create_m2m100();
    tracing::info!("🌐 Using translation engine: {}", engine.name());

    let result = engine.translate(&texts, source, target, config)?;

    println!("\n✅ Translated {} → {}", source, target);
    println!();
    for entry in &result.entries {
        println!("  [{}]", entry.index);
        println!("    SRC: {}", entry.source_text);
        println!("    TGT: {}", entry.translated_text);
        println!();
    }

    Ok(())
}

// ---- TTS ----

fn run_tts(
    input: &str,
    output: &PathBuf,
    reference_audio: Option<&PathBuf>,
    reference_text: Option<&String>,
    config: &RsvConfig,
) -> anyhow::Result<()> {
    let mut cfg = config.clone();
    cfg.tts.reference_audio = reference_audio.map(|p| p.to_string_lossy().to_string());
    cfg.tts.reference_text = reference_text.cloned();
    cfg.tts.voice_clone = reference_audio.is_some();

    let text_segments: Vec<String> = if PathBuf::from(input).exists() {
        let content = std::fs::read_to_string(input)?;
        content.lines().map(|l| l.to_string()).collect()
    } else {
        vec![input.to_string()]
    };

    let segments: Vec<(String, Option<String>)> = text_segments
        .into_iter()
        .map(|t| (t, None))
        .collect();

    let engine = rsv_tts::create_qwen3_tts();
    tracing::info!("🔊 Using TTS engine: {}", engine.name());

    std::fs::create_dir_all(output)?;

    let result = engine.synthesize(
        &segments,
        output.to_str().unwrap(),
        &cfg,
    )?;

    println!("\n✅ Generated {} audio segments\n", result.segments.len());
    for seg in &result.segments {
        println!(
            "  [{}] {:.1}s → {}",
            seg.index,
            seg.duration,
            seg.audio_path,
        );
    }

    Ok(())
}

// ---- Pipeline ----

fn run_pipeline(
    input: &PathBuf,
    output: &PathBuf,
    source: &str,
    target: &str,
    asr_model: &str,
    tts_voice: &str,
    config: &RsvConfig,
) -> anyhow::Result<()> {
    let script = config.scripts_dir.join("pipeline.py");
    if !script.exists() {
        anyhow::bail!("Pipeline script not found: {}", script.display());
    }

    tracing::info!("🎬 Starting full video translation pipeline");
    tracing::info!("   Input: {}", input.display());
    tracing::info!("   Output: {}", output.display());
    tracing::info!("   {} → {}", source, target);
    tracing::info!("   ASR model: {}, TTS voice: {}", asr_model, tts_voice);

    let params = serde_json::json!({
        "video_path": input.to_str().unwrap(),
        "output_dir": output.to_str().unwrap(),
        "source_lang": source,
        "target_lang": target,
        "asr_model": asr_model,
        "tts_voice": tts_voice,
        "translate_model": config.translate.model,
        "asr_device": config.asr.device,
    });

    let cmd = std::process::Command::new("python3")
        .arg(script.to_str().unwrap())
        .arg(params.to_string())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::inherit())  // Show progress from Python
        .spawn()
        .map_err(|e| anyhow::anyhow!("Failed to start pipeline: {}", e))?;

    let output_result = cmd.wait_with_output()
        .map_err(|e| anyhow::anyhow!("Pipeline failed: {}", e))?;

    if !output_result.status.success() {
        anyhow::bail!("Pipeline exited with code: {:?}", output_result.status.code());
    }

    let stdout = String::from_utf8_lossy(&output_result.stdout);
    let result: serde_json::Value = serde_json::from_str(&stdout)
        .map_err(|e| anyhow::anyhow!("Failed to parse pipeline result: {}", e))?;

    if let Some(error) = result.get("error") {
        anyhow::bail!("Pipeline error: {}", error);
    }

    println!("\n🎉 Pipeline complete!");
    println!("   📹 Final video: {}", result["output_video"].as_str().unwrap_or("?"));
    println!("   📝 Subtitles: {}", result["subtitle_file"].as_str().unwrap_or("?"));
    println!("   🎯 Segments: {}", result["segments_count"].as_u64().unwrap_or(0));

    Ok(())
}
