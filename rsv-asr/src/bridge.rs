use crate::AsrEngine;
use crate::error::AsrError;
use rsv_core::types::*;
use rsv_core::config::RsvConfig;
use std::process::Command;
use serde::Deserialize;

/// Python bridge: calls a Python script that runs Faster-Whisper
pub struct FasterWhisperEngine;

impl FasterWhisperEngine {
    pub fn new() -> Self {
        Self
    }

    /// Path to the Python bridge script
    fn script_path(&self, config: &RsvConfig) -> std::path::PathBuf {
        config.scripts_dir.join("asr_faster_whisper.py")
    }

    fn run_python(&self, script: &str, input_json: &str) -> Result<PythonOutput, AsrError> {
        let output = Command::new("python3")
            .arg(script)
            .arg(input_json)
            .output()
            .map_err(|e| AsrError::PythonExecution(format!("Failed to run Python: {e}")))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(AsrError::PythonExecution(format!(
                "Python script failed:\n{stderr}"
            )));
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        serde_json::from_str(&stdout)
            .map_err(|e| AsrError::ParseError(format!("Failed to parse ASR output: {e}")))
    }
}

impl AsrEngine for FasterWhisperEngine {
    fn transcribe(&self, audio_path: &str, config: &RsvConfig) -> anyhow::Result<AsrResult> {
        let script = self.script_path(config);
        if !script.exists() {
            anyhow::bail!("ASR Python script not found: {}", script.display());
        }

        tracing::info!("Transcribing audio: {} (model: {})", audio_path, config.asr.model);

        let input = serde_json::json!({
            "input_path": audio_path,
            "model": config.asr.model,
            "device": config.asr.device,
            "compute_type": config.asr.compute_type,
            "language": "auto",
        });

        let output = self.run_python(
            script.to_str().unwrap(),
            &input.to_string(),
        )?;

        let segments: Vec<AsrSegment> = output.segments
            .into_iter()
            .map(|s| AsrSegment {
                start: s.start,
                end: s.end,
                text: s.text,
                confidence: s.confidence,
                speaker: s.speaker,
            })
            .collect();

        let mut entries = Vec::new();
        for (i, seg) in segments.iter().enumerate() {
            entries.push(SubtitleEntry {
                index: (i + 1) as u32,
                start: seg.start,
                end: seg.end,
                text: seg.text.clone(),
            });
        }

        Ok(AsrResult {
            subtitles: Subtitles {
                entries,
                language: output.language.unwrap_or_else(|| "unknown".into()),
            },
            segments,
        })
    }

    fn name(&self) -> &'static str {
        "faster-whisper"
    }
}

// ---- Python bridge protocol ----

#[derive(Debug, Deserialize)]
struct PythonOutput {
    segments: Vec<PythonSegment>,
    language: Option<String>,
}

#[derive(Debug, Deserialize)]
struct PythonSegment {
    start: f64,
    end: f64,
    text: String,
    confidence: f64,
    speaker: Option<String>,
}
