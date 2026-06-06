use burn::{
    config::Config,
    module::Module,
    nn::{Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig},
    tensor::{activation::gelu, backend::Backend, Device, Int, Tensor},
};

use crate::attention::{CausalSelfAttention, CausalSelfAttentionConfig};

#[derive(Config, Debug)]
pub struct FeedForwardConfig {
    pub d_model: usize,
    #[config(default = 4)]
    pub expansion: usize,
    #[config(default = false)]
    pub bias: bool,
}

impl FeedForwardConfig {
    pub fn init<B: Backend>(&self, device: &Device<B>) -> FeedForward<B> {
        let hidden_dim = self.d_model * self.expansion;
        let fc_in = LinearConfig::new(self.d_model, hidden_dim)
            .with_bias(self.bias)
            .init(device);
        let fc_out = LinearConfig::new(hidden_dim, self.d_model)
            .with_bias(self.bias)
            .init(device);

        FeedForward {
            fc_in,
            fc_out,
            d_model: self.d_model,
            hidden_dim,
        }
    }
}

#[derive(Module, Debug)]
pub struct FeedForward<B: Backend> {
    pub fc_in: Linear<B>,
    pub fc_out: Linear<B>,
    pub d_model: usize,
    pub hidden_dim: usize,
}

impl<B: Backend> FeedForward<B> {
    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        self.fc_out.forward(gelu(self.fc_in.forward(input)))
    }
}

#[derive(Config, Debug)]
pub struct TransformerBlockConfig {
    pub d_model: usize,
    pub n_heads: usize,
    #[config(default = 4)]
    pub ffn_expansion: usize,
    #[config(default = 1e-5)]
    pub eps: f64,
    #[config(default = false)]
    pub bias: bool,
}

impl TransformerBlockConfig {
    pub fn init<B: Backend>(&self, device: &Device<B>) -> TransformerBlock<B> {
        let attn_norm = LayerNormConfig::new(self.d_model)
            .with_epsilon(self.eps)
            .init(device);
        let attention = CausalSelfAttentionConfig::new(self.d_model, self.n_heads)
            .with_bias(self.bias)
            .init(device);
        let ffn_norm = LayerNormConfig::new(self.d_model)
            .with_epsilon(self.eps)
            .init(device);
        let ffn = FeedForwardConfig::new(self.d_model)
            .with_expansion(self.ffn_expansion)
            .with_bias(self.bias)
            .init(device);

        TransformerBlock {
            attn_norm,
            attention,
            ffn_norm,
            ffn,
            d_model: self.d_model,
        }
    }
}

#[derive(Module, Debug)]
pub struct TransformerBlock<B: Backend> {
    pub attn_norm: LayerNorm<B>,
    pub attention: CausalSelfAttention<B>,
    pub ffn_norm: LayerNorm<B>,
    pub ffn: FeedForward<B>,
    pub d_model: usize,
}

impl<B: Backend> TransformerBlock<B> {
    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let attn_input = self.attn_norm.forward(input.clone());
        let x = input + self.attention.forward(attn_input);
        let ffn_input = self.ffn_norm.forward(x.clone());
        x + self.ffn.forward(ffn_input)
    }
}

#[derive(Config, Debug)]
pub struct TransformerLmConfig {
    pub vocab_size: usize,
    pub d_model: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub max_seq_len: usize,
    #[config(default = 4)]
    pub ffn_expansion: usize,
    #[config(default = 1e-5)]
    pub eps: f64,
    #[config(default = false)]
    pub bias: bool,
}

impl TransformerLmConfig {
    pub fn init<B: Backend>(&self, device: &Device<B>) -> TransformerLm<B> {
        let token_embedding = EmbeddingConfig::new(self.vocab_size, self.d_model).init(device);
        let position_embedding = EmbeddingConfig::new(self.max_seq_len, self.d_model).init(device);
        let mut layers = Vec::with_capacity(self.n_layers);

        for _ in 0..self.n_layers {
            layers.push(
                TransformerBlockConfig::new(self.d_model, self.n_heads)
                    .with_ffn_expansion(self.ffn_expansion)
                    .with_eps(self.eps)
                    .with_bias(self.bias)
                    .init(device),
            );
        }

        let norm_f = LayerNormConfig::new(self.d_model)
            .with_epsilon(self.eps)
            .init(device);
        let lm_head = LinearConfig::new(self.d_model, self.vocab_size)
            .with_bias(false)
            .init(device);

        TransformerLm {
            token_embedding,
            position_embedding,
            layers,
            norm_f,
            lm_head,
            vocab_size: self.vocab_size,
            d_model: self.d_model,
            n_layers: self.n_layers,
            n_heads: self.n_heads,
            max_seq_len: self.max_seq_len,
        }
    }
}

#[derive(Module, Debug)]
pub struct TransformerLm<B: Backend> {
    pub token_embedding: Embedding<B>,
    pub position_embedding: Embedding<B>,
    pub layers: Vec<TransformerBlock<B>>,
    pub norm_f: LayerNorm<B>,
    pub lm_head: Linear<B>,
    pub vocab_size: usize,
    pub d_model: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub max_seq_len: usize,
}

impl<B: Backend> TransformerLm<B> {
    pub fn forward(&self, input: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let [batch, seq_len] = input.dims();
        assert!(
            seq_len <= self.max_seq_len,
            "sequence length exceeds max_seq_len"
        );

        let token_embeddings = self.token_embedding.forward(input);
        let positions = Tensor::<B, 1, Int>::arange(0..seq_len as i64, &token_embeddings.device())
            .unsqueeze_dim::<2>(0)
            .repeat(&[batch, 1]);
        let position_embeddings = self.position_embedding.forward(positions);
        let mut hidden = token_embeddings + position_embeddings;

        for layer in &self.layers {
            hidden = layer.forward(hidden);
        }

        self.lm_head.forward(self.norm_f.forward(hidden))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::Distribution;

    #[test]
    fn feed_forward_forward_has_expected_shape() {
        let device = Default::default();
        let ffn = FeedForwardConfig::new(64).init::<burn::backend::NdArray>(&device);
        let input =
            Tensor::<burn::backend::NdArray, 3>::random([2, 8, 64], Distribution::Default, &device);

        let output = ffn.forward(input);

        assert_eq!(output.dims(), [2, 8, 64]);
    }

    #[test]
    fn transformer_block_forward_has_expected_shape() {
        let device = Default::default();
        let block = TransformerBlockConfig::new(64, 4).init::<burn::backend::NdArray>(&device);
        let input =
            Tensor::<burn::backend::NdArray, 3>::random([2, 8, 64], Distribution::Default, &device);

        let output = block.forward(input);

        assert_eq!(output.dims(), [2, 8, 64]);
    }

    #[test]
    fn transformer_lm_forward_has_expected_shape() {
        let device = Default::default();
        let model =
            TransformerLmConfig::new(256, 64, 2, 4, 128).init::<burn::backend::NdArray>(&device);
        let input = Tensor::<burn::backend::NdArray, 2, Int>::random(
            [2, 16],
            Distribution::Uniform(0.0, 255.0),
            &device,
        );

        let output = model.forward(input);

        assert_eq!(output.dims(), [2, 16, 256]);
    }
}
