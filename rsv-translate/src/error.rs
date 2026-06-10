use thiserror::Error;

#[derive(Error, Debug)]
pub enum TranslateError {
    #[error("Translation model error: {0}")]
    ModelError(String),

    #[error("Python execution failed: {0}")]
    Execution(String),

    #[error("Failed to parse translation output: {0}")]
    Parse(String),
}
