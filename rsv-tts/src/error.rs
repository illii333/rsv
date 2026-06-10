use thiserror::Error;

#[derive(Error, Debug)]
pub enum TtsError {
    #[error("TTS model error: {0}")]
    ModelError(String),

    #[error("Python execution failed: {0}")]
    Execution(String),

    #[error("Failed to parse TTS output: {0}")]
    Parse(String),

    #[error("Voice cloning failed: {0}")]
    VoiceCloneError(String),
}
