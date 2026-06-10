pub mod engine;
pub mod bridge;
pub mod error;

use rsv_core::types::AsrResult;
use rsv_core::config::RsvConfig;

/// The ASR engine trait — implement this for different backends
pub trait AsrEngine: Send {
    /// Transcribe an audio file and return subtitles
    fn transcribe(&self, audio_path: &str, config: &RsvConfig) -> anyhow::Result<AsrResult>;

    /// Get the engine name
    fn name(&self) -> &'static str;
}

/// Create a Faster-Whisper ASR engine
pub fn create_faster_whisper() -> Box<dyn AsrEngine> {
    Box::new(bridge::FasterWhisperEngine::new())
}
