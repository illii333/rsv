use rsv_core::types::*;
use rsv_core::config::RsvConfig;

/// Core ASR algorithm — describes how recognition works conceptually
pub trait AsrAlgorithm {
    /// Run recognition on 16kHz mono WAV audio data
    fn recognize(&self, audio_data: &[u8], config: &RsvConfig) -> anyhow::Result<Vec<AsrSegment>>;

    /// Post-process: convert raw segments to subtitles
    fn segments_to_subtitles(&self, segments: Vec<AsrSegment>, language: &str) -> Subtitles {
        let mut entries = Vec::new();
        for (i, seg) in segments.iter().enumerate() {
            entries.push(SubtitleEntry {
                index: (i + 1) as u32,
                start: seg.start,
                end: seg.end,
                text: seg.text.clone(),
            });
        }
        Subtitles {
            entries,
            language: language.to_string(),
        }
    }
}
