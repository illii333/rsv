/// Core TTS algorithm
pub trait TtsAlgorithm {
    /// Generate audio for a single text segment
    fn synthesize_segment(
        &self,
        text: &str,
        reference_audio: Option<&str>,
        reference_text: Option<&str>,
        output_path: &str,
    ) -> anyhow::Result<f64>;  // returns duration in seconds
}
