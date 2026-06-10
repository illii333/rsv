use thiserror::Error;

#[derive(Error, Debug)]
pub enum RsvError {
    #[error("Model error: {0}")]
    ModelError(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("FFmpeg error: {0}")]
    FfmpegError(String),

    #[error("Python runtime error: {0}")]
    PythonError(String),

    #[error("Serialization error: {0}")]
    SerdeError(#[from] serde_json::Error),

    #[error("Invalid configuration: {0}")]
    ConfigError(String),

    #[error("Task cancelled")]
    Cancelled,

    #[error("{0}")]
    Other(String),
}

pub type RsvResult<T> = std::result::Result<T, RsvError>;
