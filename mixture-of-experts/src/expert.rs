use burn::{
    config::Config,
    module::Module,
    nn::{Linear, LinearConfig},
    tensor::{activation::gelu, backend::Backend, Device, Tensor},
};

#[derive(Config, Debug)]
pub struct ExpertConfig {
    pub d_model: usize,
    #[config(default = 4)]
    pub expansion: usize,
    #[config(default = false)]
    pub bias: bool,
}

impl ExpertConfig {
    pub fn init<B: Backend>(&self, device: &Device<B>) -> Expert<B> {
        let hidden_dim = self.d_model * self.expansion;
        let fc_in = LinearConfig::new(self.d_model, hidden_dim)
            .with_bias(self.bias)
            .init(device);
        let fc_out = LinearConfig::new(hidden_dim, self.d_model)
            .with_bias(self.bias)
            .init(device);

        Expert {
            fc_in,
            fc_out,
            d_model: self.d_model,
            hidden_dim,
        }
    }
}

#[derive(Module, Debug)]
pub struct Expert<B: Backend> {
    pub fc_in: Linear<B>,
    pub fc_out: Linear<B>,
    pub d_model: usize,
    pub hidden_dim: usize,
}

impl<B: Backend> Expert<B> {
    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        self.fc_out.forward(gelu(self.fc_in.forward(input)))
    }
}
