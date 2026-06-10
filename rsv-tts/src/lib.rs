pub mod engine;
pub mod bridge;
pub mod error;

use rsv_core::types::TtsResult;
use rsv_core::config::RsvConfig;

/// The TTS engine trait
pub trait TtsEngine: Send {
    /// Synthesize speech from text, optionally clone a voice
    fn synthesize(
        &self,
        text_segments: &[(String, Option<String>)],  // (text, speaker_label)
        output_dir: &str,
        config: &RsvConfig,
    ) -> anyhow::Result<TtsResult>;

    fn name(&self) -> &'static str;
}

/// Create a Qwen3-TTS engine
pub fn create_qwen3_tts() -> Box<dyn TtsEngine> {
    Box::new(bridge::Qwen3TtsEngine::new())
}
