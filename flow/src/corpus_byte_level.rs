use std::{fs, io, path::Path};

pub const DEFAULT_CORPUS_PATH: &str =
    "/home/alcss/Documentos/Autor/Programación/Laboratorio/H-Net/Corpus/obsidian_corpus.txt";

#[derive(Debug, Clone)]
pub struct ByteCorpus {
    bytes: Vec<u8>,
}

impl ByteCorpus {
    pub fn from_file(path: impl AsRef<Path>) -> io::Result<Self> {
        let bytes = fs::read(path)?;
        Ok(Self { bytes })
    }

    pub fn len(&self) -> usize {
        self.bytes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }

    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub fn token_chunks(&self, seq_len: usize) -> impl Iterator<Item = &[u8]> {
        self.bytes.chunks_exact(seq_len)
    }

    pub fn batch_tokens(&self, batch_size: usize, seq_len: usize) -> Option<Vec<i64>> {
        let token_count = batch_size.checked_mul(seq_len)?;
        if self.bytes.len() < token_count {
            return None;
        }

        Some(
            self.bytes[..token_count]
                .iter()
                .map(|byte| i64::from(*byte))
                .collect(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_chunks_returns_exact_chunks() {
        let corpus = ByteCorpus {
            bytes: vec![0, 1, 2, 3, 4],
        };

        let chunks: Vec<&[u8]> = corpus.token_chunks(2).collect();

        assert_eq!(chunks, vec![&[0, 1][..], &[2, 3][..]]);
    }

    #[test]
    fn batch_tokens_returns_byte_ids() {
        let corpus = ByteCorpus {
            bytes: vec![0, 127, 255, 42],
        };

        assert_eq!(corpus.batch_tokens(2, 2), Some(vec![0, 127, 255, 42]));
    }

    #[test]
    fn batch_tokens_returns_none_when_corpus_is_too_short() {
        let corpus = ByteCorpus { bytes: vec![1, 2] };

        assert_eq!(corpus.batch_tokens(1, 3), None);
    }
}
