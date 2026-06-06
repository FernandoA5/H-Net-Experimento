use burn::{
    config::Config,
    module::Module,
    nn::{Linear, LinearConfig},
    tensor::{backend::Backend, module::attention, ops::AttentionModuleOptions, Device, Tensor},
};

#[derive(Config, Debug)]
pub struct CausalSelfAttentionConfig {
    pub d_model: usize,
    pub n_heads: usize,
    #[config(default = false)]
    pub bias: bool,
}

impl CausalSelfAttentionConfig {
    pub fn init<B: Backend>(&self, device: &Device<B>) -> CausalSelfAttention<B> {
        assert!(self.n_heads > 0, "n_heads must be greater than zero");
        assert_eq!(
            self.d_model % self.n_heads,
            0,
            "d_model must be divisible by n_heads"
        );

        let q_proj = LinearConfig::new(self.d_model, self.d_model)
            .with_bias(self.bias)
            .init(device);
        let k_proj = LinearConfig::new(self.d_model, self.d_model)
            .with_bias(self.bias)
            .init(device);
        let v_proj = LinearConfig::new(self.d_model, self.d_model)
            .with_bias(self.bias)
            .init(device);
        let out_proj = LinearConfig::new(self.d_model, self.d_model)
            .with_bias(self.bias)
            .init(device);

        CausalSelfAttention {
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            d_model: self.d_model,
            n_heads: self.n_heads,
            head_dim: self.d_model / self.n_heads,
        }
    }
}

#[derive(Module, Debug)]
pub struct CausalSelfAttention<B: Backend> {
    pub q_proj: Linear<B>,
    pub k_proj: Linear<B>,
    pub v_proj: Linear<B>,
    pub out_proj: Linear<B>,
    pub d_model: usize,
    pub n_heads: usize,
    pub head_dim: usize,
}

impl<B: Backend> CausalSelfAttention<B> {
    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, seq_len, _] = input.dims();

        let query = self.project_heads(self.q_proj.forward(input.clone()), batch, seq_len);
        let key = self.project_heads(self.k_proj.forward(input.clone()), batch, seq_len);
        let value = self.project_heads(self.v_proj.forward(input), batch, seq_len);

        let context = attention(
            query,
            key,
            value,
            None,
            None,
            AttentionModuleOptions {
                scale: None,
                softcap: None,
                is_causal: true,
            },
        );

        let merged = context
            .swap_dims(1, 2)
            .reshape([batch, seq_len, self.d_model]);

        self.out_proj.forward(merged)
    }

    fn project_heads(&self, tensor: Tensor<B, 3>, batch: usize, seq_len: usize) -> Tensor<B, 4> {
        tensor
            .reshape([batch, seq_len, self.n_heads, self.head_dim])
            .swap_dims(1, 2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::Distribution;

    #[test]
    fn causal_self_attention_forward_has_expected_shape() {
        let device = Default::default();
        let attention =
            CausalSelfAttentionConfig::new(64, 4).init::<burn::backend::NdArray>(&device);
        let input = Tensor::<burn::backend::NdArray, 3>::random(
            [2, 16, 64],
            Distribution::Default,
            &device,
        );

        let output = attention.forward(input);

        assert_eq!(output.dims(), [2, 16, 64]);
    }

    #[test]
    fn causal_self_attention_is_prefix_invariant() {
        let device = Default::default();
        let attention =
            CausalSelfAttentionConfig::new(32, 4).init::<burn::backend::NdArray>(&device);
        let input =
            Tensor::<burn::backend::NdArray, 3>::random([1, 6, 32], Distribution::Default, &device);
        let prefix = input.clone().slice([0..1, 0..3, 0..32]);

        let full_output = attention.forward(input).slice([0..1, 0..3, 0..32]);
        let prefix_output = attention.forward(prefix);

        let full_values = full_output.into_data().to_vec::<f32>().unwrap();
        let prefix_values = prefix_output.into_data().to_vec::<f32>().unwrap();

        for (full, prefix) in full_values.iter().zip(prefix_values.iter()) {
            assert!((full - prefix).abs() < 1e-4);
        }
    }
}
