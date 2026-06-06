use burn::{
    config::Config,
    module::Module,
    nn::{LayerNorm, LayerNormConfig},
    tensor::{backend::Backend, Device, Tensor},
};

use crate::{Expert, ExpertConfig, TopKRouter, TopKRouterConfig};

#[derive(Config, Debug)]
pub struct SparseMoEConfig {
    pub d_model: usize,
    pub num_experts: usize,
    pub top_k: usize,
    #[config(default = 4)]
    pub expert_expansion: usize,
    #[config(default = 1e-5)]
    pub eps: f64,
    #[config(default = false)]
    pub bias: bool,
}

impl SparseMoEConfig {
    pub fn init<B: Backend>(&self, device: &Device<B>) -> SparseMoE<B> {
        let router = TopKRouterConfig::new(self.d_model, self.num_experts, self.top_k)
            .with_bias(self.bias)
            .init(device);
        let norm = LayerNormConfig::new(self.d_model)
            .with_epsilon(self.eps)
            .init(device);
        let mut experts = Vec::with_capacity(self.num_experts);

        for _ in 0..self.num_experts {
            experts.push(
                ExpertConfig::new(self.d_model)
                    .with_expansion(self.expert_expansion)
                    .with_bias(self.bias)
                    .init(device),
            );
        }

        SparseMoE {
            router,
            norm,
            experts,
            d_model: self.d_model,
            num_experts: self.num_experts,
            top_k: self.top_k,
        }
    }
}

#[derive(Module, Debug)]
pub struct SparseMoE<B: Backend> {
    pub router: TopKRouter<B>,
    pub norm: LayerNorm<B>,
    pub experts: Vec<Expert<B>>,
    pub d_model: usize,
    pub num_experts: usize,
    pub top_k: usize,
}

#[derive(Debug)]
pub struct MoEOutput<B: Backend> {
    pub hidden: Tensor<B, 3>,
    pub router_logits: Tensor<B, 3>,
    pub router_probs: Tensor<B, 3>,
    pub expert_mask: Tensor<B, 3>,
    pub load_balance_loss: Tensor<B, 1>,
}

impl<B: Backend> SparseMoE<B> {
    pub fn forward(&self, input: Tensor<B, 3>) -> MoEOutput<B> {
        let normalized = self.norm.forward(input);
        let route = self.router.forward(normalized.clone());
        let mut expert_outputs = Vec::with_capacity(self.num_experts);

        for expert in &self.experts {
            expert_outputs.push(expert.forward(normalized.clone()));
        }

        let stacked_experts: Tensor<B, 4> = Tensor::stack(expert_outputs, 2);
        let weights = route.probs.clone().unsqueeze_dim::<4>(3);
        let hidden = (stacked_experts * weights).sum_dim(2).squeeze_dim::<3>(2);
        let load_balance_loss =
            self.load_balance_loss(route.probs.clone(), route.expert_mask.clone());

        MoEOutput {
            hidden,
            router_logits: route.logits,
            router_probs: route.probs,
            expert_mask: route.expert_mask,
            load_balance_loss,
        }
    }

    fn load_balance_loss(&self, probs: Tensor<B, 3>, expert_mask: Tensor<B, 3>) -> Tensor<B, 1> {
        let importance = probs
            .mean_dim(0)
            .mean_dim(1)
            .squeeze_dim::<2>(0)
            .squeeze_dim::<1>(0);
        let load = expert_mask
            .mean_dim(0)
            .mean_dim(1)
            .squeeze_dim::<2>(0)
            .squeeze_dim::<1>(0);

        (importance * load).sum() * self.num_experts as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::Distribution;

    #[test]
    fn sparse_moe_forward_has_expected_shapes() {
        let device = Default::default();
        let moe = SparseMoEConfig::new(64, 4, 2).init::<burn::backend::NdArray>(&device);
        let input =
            Tensor::<burn::backend::NdArray, 3>::random([2, 8, 64], Distribution::Default, &device);

        let output = moe.forward(input);

        assert_eq!(output.hidden.dims(), [2, 8, 64]);
        assert_eq!(output.router_logits.dims(), [2, 8, 4]);
        assert_eq!(output.router_probs.dims(), [2, 8, 4]);
        assert_eq!(output.expert_mask.dims(), [2, 8, 4]);
        assert_eq!(output.load_balance_loss.dims(), [1]);
    }

    #[test]
    fn router_probabilities_sum_to_one_per_token() {
        let device = Default::default();
        let moe = SparseMoEConfig::new(32, 4, 2).init::<burn::backend::NdArray>(&device);
        let input =
            Tensor::<burn::backend::NdArray, 3>::random([1, 6, 32], Distribution::Default, &device);

        let output = moe.forward(input);
        let sums = output.router_probs.sum_dim(2).squeeze_dim::<2>(2);
        let values = sums.into_data().to_vec::<f32>().unwrap();

        for value in values {
            assert!((value - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn router_selects_top_k_experts_per_token() {
        let device = Default::default();
        let moe = SparseMoEConfig::new(32, 6, 2).init::<burn::backend::NdArray>(&device);
        let input =
            Tensor::<burn::backend::NdArray, 3>::random([1, 5, 32], Distribution::Default, &device);

        let output = moe.forward(input);
        let selected = output.expert_mask.sum_dim(2).squeeze_dim::<2>(2);
        let values = selected.into_data().to_vec::<f32>().unwrap();

        for value in values {
            assert!((value - 2.0).abs() < 1e-5);
        }
    }
}
