use crate::types::{AsrConfig, TtsConfig, TranslateConfig};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Global RSV configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RsvConfig {
    /// Path to store downloaded models
    pub models_dir: PathBuf,

    /// Path to store temporary files
    pub temp_dir: PathBuf,

    /// FFmpeg path (auto-detect if None)
    pub ffmpeg_path: Option<PathBuf>,

    /// ASR (speech recognition) config
    pub asr: AsrConfig,

    /// Translation config
    pub translate: TranslateConfig,

    /// TTS (voice synthesis/cloning) config
    pub tts: TtsConfig,

    /// Python interpreter path (for calling model scripts)
    pub python_path: String,

    /// Path to Python scripts directory
    pub scripts_dir: PathBuf,
}

impl Default for RsvConfig {
    fn default() -> Self {
        Self {
            models_dir: PathBuf::from("models"),
            temp_dir: PathBuf::from("/tmp/rsv"),
            ffmpeg_path: None,
            asr: AsrConfig::default(),
            translate: TranslateConfig::default(),
            tts: TtsConfig::default(),
            python_path: "python3".to_string(),
            scripts_dir: PathBuf::from("scripts"),
        }
    }
}

impl RsvConfig {
    /// Load config from a JSON file
    pub fn from_file(path: impl Into<PathBuf>) -> Result<Self> {
        let content = std::fs::read_to_string(path.into())?;
        let config: RsvConfig = serde_json::from_str(&content)?;
        Ok(config)
    }

    /// Save config to a JSON file
    pub fn save(&self, path: impl Into<PathBuf>) -> Result<()> {
        let content = serde_json::to_string_pretty(self)?;
        std::fs::write(path.into(), content)?;
        Ok(())
    }

    /// Get absolute path to a model file
    pub fn model_path(&self, name: &str) -> PathBuf {
        self.models_dir.join(name)
    }
}
