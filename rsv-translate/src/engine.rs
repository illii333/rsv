/// Core translation algorithm
pub trait TranslateAlgorithm {
    /// Translate a batch of texts
    fn translate_batch(
        &self,
        texts: &[String],
        source_lang: &str,
        target_lang: &str,
    ) -> anyhow::Result<Vec<String>>;
}
