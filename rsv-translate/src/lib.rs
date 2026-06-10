pub mod engine;
pub mod bridge;
pub mod error;

use rsv_core::types::TranslationResult;
use rsv_core::config::RsvConfig;

/// The translation engine trait
pub trait TranslateEngine: Send {
    /// Translate subtitles from source to target language
    fn translate(
        &self,
        text: &[String],
        source_lang: &str,
        target_lang: &str,
        config: &RsvConfig,
    ) -> anyhow::Result<TranslationResult>;

    fn name(&self) -> &'static str;
}

/// Create an M2M100 translation engine
pub fn create_m2m100() -> Box<dyn TranslateEngine> {
    Box::new(bridge::M2M100Engine::new())
}
