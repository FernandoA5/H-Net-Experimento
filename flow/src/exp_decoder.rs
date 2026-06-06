#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DecodeError {
    TokenOutOfByteRange(i64),
}

impl std::fmt::Display for DecodeError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TokenOutOfByteRange(token) => {
                write!(formatter, "token {token} is outside byte range 0..=255")
            }
        }
    }
}

impl std::error::Error for DecodeError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodedTokens {
    pub text: String,
    pub words: Vec<String>,
}

pub fn decode_byte_tokens(tokens: &[i64]) -> Result<DecodedTokens, DecodeError> {
    let bytes = tokens_to_bytes(tokens)?;
    let text = String::from_utf8_lossy(&bytes).into_owned();
    let words = text
        .split(|character: char| !character.is_alphanumeric())
        .filter(|word| !word.is_empty())
        .map(str::to_owned)
        .collect();

    Ok(DecodedTokens { text, words })
}

fn tokens_to_bytes(tokens: &[i64]) -> Result<Vec<u8>, DecodeError> {
    tokens
        .iter()
        .map(|token| u8::try_from(*token).map_err(|_| DecodeError::TokenOutOfByteRange(*token)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_byte_tokens_restores_utf8_text_and_words() {
        let tokens = "Hola, mundo real. á"
            .bytes()
            .map(i64::from)
            .collect::<Vec<_>>();

        let decoded = decode_byte_tokens(&tokens).unwrap();

        assert_eq!(decoded.text, "Hola, mundo real. á");
        assert_eq!(decoded.words, vec!["Hola", "mundo", "real", "á"]);
    }

    #[test]
    fn decode_byte_tokens_rejects_tokens_outside_byte_range() {
        let error = decode_byte_tokens(&[256]).unwrap_err();

        assert_eq!(error, DecodeError::TokenOutOfByteRange(256));
    }
}
