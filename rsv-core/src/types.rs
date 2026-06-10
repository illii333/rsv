use serde::{Deserialize, Serialize};

/// Supported languages
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum Language {
    ZhCn, // 简体中文
    ZhTw, // 繁体中文
    En,   // English
    Ja,   // 日本語
    Ko,   // 한국어
    Fr,   // Français
    De,   // Deutsch
    Ru,   // Русский
    Es,   // Español
    Th,   // ภาษาไทย
    Other(String),
}

impl Language {
    pub fn code(&self) -> &str {
        match self {
            Language::ZhCn => "zh-cn",
            Language::ZhTw => "zh-tw",
            Language::En => "en",
            Language::Ja => "ja",
            Language::Ko => "ko",
            Language::Fr => "fr",
            Language::De => "de",
            Language::Ru => "ru",
            Language::Es => "es",
            Language::Th => "th",
            Language::Other(s) => s.as_str(),
        }
    }

    pub fn from_code(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "zh-cn" | "zh" | "chi" => Language::ZhCn,
            "zh-tw" | "zh_tw" => Language::ZhTw,
            "en" | "eng" => Language::En,
            "ja" | "jpn" => Language::Ja,
            "ko" | "kor" => Language::Ko,
            "fr" | "fra" => Language::Fr,
            "de" | "deu" => Language::De,
            "ru" | "rus" => Language::Ru,
            "es" | "spa" => Language::Es,
            "th" | "tha" => Language::Th,
            _ => Language::Other(s.to_string()),
        }
    }
}

/// A single subtitle entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubtitleEntry {
    pub index: u32,
    pub start: f64,  // seconds
    pub end: f64,    // seconds
    pub text: String,
}

/// Full subtitle track (SRT-like)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Subtitles {
    pub entries: Vec<SubtitleEntry>,
    pub language: String,
}

/// ASR result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AsrResult {
    pub subtitles: Subtitles,
    pub segments: Vec<AsrSegment>,
}

/// A single ASR segment with timing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AsrSegment {
    pub start: f64,
    pub end: f64,
    pub text: String,
    pub confidence: f64,
    pub speaker: Option<String>,
}

/// Translation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranslationResult {
    pub source_language: String,
    pub target_language: String,
    pub entries: Vec<TranslatedEntry>,
}

/// A single translated entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranslatedEntry {
    pub index: u32,
    pub source_text: String,
    pub translated_text: String,
}

/// TTS / voice cloning result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TtsResult {
    pub audio_path: String,
    pub segments: Vec<TtsSegment>,
}

/// A single TTS segment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TtsSegment {
    pub index: u32,
    pub text: String,
    pub audio_path: String,
    pub duration: f64,
    pub speaker: Option<String>,
}

/// Video translation task
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VtvTask {
    pub input_video: String,
    pub output_dir: String,
    pub source_language: String,
    pub target_language: String,
    pub asr_config: AsrConfig,
    pub translate_config: TranslateConfig,
    pub tts_config: TtsConfig,
}

/// ASR configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AsrConfig {
    pub model: String,      // e.g. "small", "medium", "large-v3", "large-v3-turbo"
    pub device: String,     // "cpu", "cuda"
    pub compute_type: String, // "float16", "int8", "float32"
    pub num_speakers: Option<usize>,
}

impl Default for AsrConfig {
    fn default() -> Self {
        Self {
            model: "large-v3-turbo".to_string(),
            device: "auto".to_string(),
            compute_type: "float16".to_string(),
            num_speakers: None,
        }
    }
}

/// Translation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranslateConfig {
    pub model: String,
    pub device: String,
    pub beam_size: usize,
    pub max_length: usize,
}

impl Default for TranslateConfig {
    fn default() -> Self {
        Self {
            model: "facebook/m2m100_418M".to_string(),
            device: "auto".to_string(),
            beam_size: 5,
            max_length: 256,
        }
    }
}

/// TTS configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TtsConfig {
    pub model: String,
    pub device: String,
    pub voice_clone: bool,
    pub reference_audio: Option<String>,
    pub reference_text: Option<String>,
}

impl Default for TtsConfig {
    fn default() -> Self {
        Self {
            model: "Qwen3-TTS".to_string(),
            device: "auto".to_string(),
            voice_clone: false,
            reference_audio: None,
            reference_text: None,
        }
    }
}
