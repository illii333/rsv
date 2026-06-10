use thiserror::Error;

#[derive(Error, Debug)]
pub enum AsrError {
    #[error("No speech detected")]
    NoSpeech,

    #[error("Model not found: {0}")]
    ModelNotFound(String),

    #[error("Python execution failed: {0}")]
    PythonExecution(String),

    #[error("Failed to parse ASR output: {0}")]
    ParseError(String),

    #[error("Audio processing failed: {0}")]
    AudioError(String),
}
