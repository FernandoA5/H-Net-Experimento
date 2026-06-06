use burn::{
    config::Config,
    module::Module,
    nn::{Linear, LinearConfig},
    tensor::{activation::softmax, backend::Backend, Device, IndexingUpdateOp, Tensor},
};

#[derive(Config, Debug)]
pub struct TopKRouterConfig {
    pub d_model: usize,
    pub num_experts: usize,
    pub top_k: usize,
    #[config(default = false)]
    pub bias: bool,
}

impl TopKRouterConfig {
    pub fn init<B: Backend>(&self, device: &Device<B>) -> TopKRouter<B> {
        assert!(
            self.num_experts > 0,
            "num_experts must be greater than zero"
        );
        assert!(self.top_k > 0, "top_k must be greater than zero");
        assert!(
            self.top_k <= self.num_experts,
            "top_k must be less than or equal to num_experts"
        );

        let gate = LinearConfig::new(self.d_model, self.num_experts)
            .with_bias(self.bias)
            .init(device);

        TopKRouter {
            gate,
            d_model: self.d_model,
            num_experts: self.num_experts,
            top_k: self.top_k,
        }
    }
}

#[derive(Module, Debug)]
pub struct TopKRouter<B: Backend> {
    pub gate: Linear<B>,
    pub d_model: usize,
    pub num_experts: usize,
    pub top_k: usize,
}

#[derive(Debug)]
pub struct RouterOutput<B: Backend> {
    pub logits: Tensor<B, 3>,
    pub probs: Tensor<B, 3>,
    pub expert_mask: Tensor<B, 3>,
}

impl<B: Backend> TopKRouter<B> {
    pub fn forward(&self, input: Tensor<B, 3>) -> RouterOutput<B> {
        let logits = self.gate.forward(input);
        let [batch, seq_len, _] = logits.dims();
        let (topk_logits, topk_indices) = logits.clone().topk_with_indices(self.top_k, 2);
        let topk_probs = softmax(topk_logits, 2);
        let zeros = Tensor::<B, 3>::zeros(&[batch, seq_len, self.num_experts], &logits.device());
        let ones = Tensor::<B, 3>::ones(&[batch, seq_len, self.top_k], &logits.device());
        let probs =
            zeros
                .clone()
                .scatter(2, topk_indices.clone(), topk_probs, IndexingUpdateOp::Add);
        let expert_mask = zeros.scatter(2, topk_indices, ones, IndexingUpdateOp::Add);

        RouterOutput {
            logits,
            probs,
            expert_mask,
        }
    }
}
