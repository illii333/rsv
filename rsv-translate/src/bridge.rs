use crate::TranslateEngine;
use crate::error::TranslateError;
use rsv_core::types::TranslatedEntry;
use rsv_core::config::RsvConfig;
use std::process::Command;
use serde::Deserialize;

/// Python bridge: calls a Python script that runs M2M100
pub struct M2M100Engine;

impl M2M100Engine {
    pub fn new() -> Self {
        Self
    }

    fn script_path(&self, config: &RsvConfig) -> std::path::PathBuf {
        config.scripts_dir.join("translate_m2m100.py")
    }
}

impl TranslateEngine for M2M100Engine {
    fn translate(
        &self,
        text: &[String],
        source_lang: &str,
        target_lang: &str,
        config: &RsvConfig,
    ) -> anyhow::Result<rsv_core::types::TranslationResult> {
        let script = self.script_path(config);
        if !script.exists() {
            anyhow::bail!("Translation Python script not found: {}", script.display());
        }

        tracing::info!(
            "Translating {} segments: {} → {}",
            text.len(),
            source_lang,
            target_lang
        );

        let input = serde_json::json!({
            "texts": text,
            "source_lang": source_lang,
            "target_lang": target_lang,
            "model": config.translate.model,
            "device": config.translate.device,
            "beam_size": config.translate.beam_size,
            "max_length": config.translate.max_length,
        });

        let output = Command::new("python3")
            .arg(script.to_str().unwrap())
            .arg(&input.to_string())
            .output()
            .map_err(|e| TranslateError::Execution(format!("Failed to run Python: {e}")))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(TranslateError::Execution(format!(
                "Python script failed:\n{stderr}"
            )).into());
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        let result: PythonTranslateOutput = serde_json::from_str(&stdout)
            .map_err(|e| TranslateError::Parse(format!("Failed to parse translation: {e}")))?;

        let entries: Vec<TranslatedEntry> = result.translations
            .into_iter()
            .enumerate()
            .map(|(i, t)| TranslatedEntry {
                index: (i + 1) as u32,
                source_text: text.get(i).cloned().unwrap_or_default(),
                translated_text: t,
            })
            .collect();

        Ok(rsv_core::types::TranslationResult {
            source_language: source_lang.to_string(),
            target_language: target_lang.to_string(),
            entries,
        })
    }

    fn name(&self) -> &'static str {
        "m2m100"
    }
}

#[derive(Debug, Deserialize)]
struct PythonTranslateOutput {
    translations: Vec<String>,
}
