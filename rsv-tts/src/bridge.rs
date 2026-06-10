use crate::TtsEngine;
use crate::error::TtsError;
use rsv_core::types::TtsSegment;
use rsv_core::config::RsvConfig;
use std::process::Command;
use serde::Deserialize;

/// Python bridge: calls Qwen3-TTS for voice synthesis/cloning
pub struct Qwen3TtsEngine;

impl Qwen3TtsEngine {
    pub fn new() -> Self {
        Self
    }

    fn script_path(&self, config: &RsvConfig) -> std::path::PathBuf {
        config.scripts_dir.join("tts_qwen3.py")
    }
}

impl TtsEngine for Qwen3TtsEngine {
    fn synthesize(
        &self,
        text_segments: &[(String, Option<String>)],
        output_dir: &str,
        config: &RsvConfig,
    ) -> anyhow::Result<rsv_core::types::TtsResult> {
        let script = self.script_path(config);
        if !script.exists() {
            anyhow::bail!("TTS Python script not found: {}", script.display());
        }

        tracing::info!(
            "Synthesizing {} segments with Qwen3-TTS (voice_clone: {})",
            text_segments.len(),
            config.tts.voice_clone,
        );

        // Build segments input
        let segments_json: Vec<serde_json::Value> = text_segments
            .iter()
            .map(|(text, speaker)| serde_json::json!({
                "text": text,
                "speaker": speaker,
            }))
            .collect();

        let input = serde_json::json!({
            "segments": segments_json,
            "output_dir": output_dir,
            "model": config.tts.model,
            "device": config.tts.device,
            "voice_clone": config.tts.voice_clone,
            "reference_audio": config.tts.reference_audio,
            "reference_text": config.tts.reference_text,
        });

        let output = Command::new("python3")
            .arg(script.to_str().unwrap())
            .arg(&input.to_string())
            .output()
            .map_err(|e| TtsError::Execution(format!("Failed to run Python: {e}")))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(TtsError::Execution(format!(
                "Python script failed:\n{stderr}"
            )).into());
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        let result: PythonTtsOutput = serde_json::from_str(&stdout)
            .map_err(|e| TtsError::Parse(format!("Failed to parse TTS output: {e}")))?;

        let segments: Vec<TtsSegment> = result.segments
            .into_iter()
            .map(|s| TtsSegment {
                index: s.index,
                text: s.text,
                audio_path: s.audio_path,
                duration: s.duration,
                speaker: s.speaker,
            })
            .collect();

        Ok(rsv_core::types::TtsResult {
            audio_path: result.audio_path,
            segments,
        })
    }

    fn name(&self) -> &'static str {
        "qwen3-tts"
    }
}

#[derive(Debug, Deserialize)]
struct PythonTtsOutput {
    audio_path: String,
    segments: Vec<PythonTtsSegment>,
}

#[derive(Debug, Deserialize)]
struct PythonTtsSegment {
    index: u32,
    text: String,
    audio_path: String,
    duration: f64,
    speaker: Option<String>,
}
