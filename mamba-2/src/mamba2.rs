use burn::{
    config::Config,
    module::{Module, Param, ParamId},
    nn::{Linear, LinearConfig, PaddingConfig1d, RmsNorm, RmsNormConfig},
    tensor::{
        activation::{silu, softplus},
        backend::Backend,
        Device, Distribution, Tensor,
    },
};

use crate::ssd;

/// Configuration for a single Mamba-2 block.
#[derive(Config, Debug)]
pub struct Mamba2BlockConfig {
    pub d_model: usize,
    #[config(default = 64)]
    pub d_state: usize,
    #[config(default = 4)]
    pub d_conv: usize,
    #[config(default = 2)]
    pub expand: usize,
    #[config(default = 64)]
    pub headdim: usize,
    #[config(default = 1)]
    pub ngroups: usize,
    #[config(default = 256)]
    pub chunk_size: usize,
    #[config(default = 0.001)]
    pub dt_min: f64,
    #[config(default = 0.1)]
    pub dt_max: f64,
    #[config(default = 1e-4)]
    pub dt_init_floor: f64,
    #[config(default = 1e-5)]
    pub eps: f64,
    #[config(default = false)]
    pub bias: bool,
    #[config(default = true)]
    pub conv_bias: bool,
    #[config(default = false)]
    pub norm_before_gate: bool,
}

impl Mamba2BlockConfig {
    pub fn init<B: Backend>(&self, device: &Device<B>) -> Mamba2Block<B> {
        let d_inner = self.d_model * self.expand;
        let nheads = d_inner / self.headdim;
        let d_in_proj = 2 * d_inner + 2 * self.ngroups * self.d_state + nheads;
        let conv_dim = d_inner + 2 * self.ngroups * self.d_state;

        let in_proj = LinearConfig::new(self.d_model, d_in_proj)
            .with_bias(self.bias)
            .init(device);

        let conv1d = burn::nn::conv::Conv1dConfig::new(conv_dim, conv_dim, self.d_conv)
            .with_groups(conv_dim)
            .with_bias(self.conv_bias)
            .with_padding(PaddingConfig1d::Explicit(self.d_conv - 1, self.d_conv - 1))
            .init(device);

        let a = Tensor::<B, 1>::random(
            [nheads],
            Distribution::Uniform(self.dt_min, self.dt_max),
            device,
        );
        let a_log = a.log();

        let raw = Tensor::<B, 1>::random([nheads], Distribution::Uniform(0.0, 1.0), device);
        let dt = (raw * (self.dt_max.ln() - self.dt_min.ln()) + self.dt_min.ln()).exp();
        let dt = dt.clamp(self.dt_init_floor, f64::INFINITY);
        let ones_n: Tensor<B, 1> = Tensor::ones(&[nheads], device);
        let dt_shift: Tensor<B, 1> = (-dt.clone()).exp();
        let inv_dt = dt.clone() + (ones_n - dt_shift).log();

        let d_param = Tensor::<B, 1>::ones(&[nheads], device);

        let norm = RmsNormConfig::new(d_inner)
            .with_epsilon(self.eps)
            .init(device);

        let out_proj = LinearConfig::new(d_inner, self.d_model)
            .with_bias(self.bias)
            .init(device);

        Mamba2Block {
            in_proj,
            conv1d,
            a_log: Param::initialized(ParamId::new(), a_log),
            dt_bias: Param::initialized(ParamId::new(), inv_dt),
            d_param: Param::initialized(ParamId::new(), d_param),
            norm,
            out_proj,
            d_model: self.d_model,
            d_state: self.d_state,
            d_conv: self.d_conv,
            expand: self.expand,
            d_inner,
            headdim: self.headdim,
            nheads,
            ngroups: self.ngroups,
            chunk_size: self.chunk_size,
            eps: self.eps,
            norm_before_gate: self.norm_before_gate,
        }
    }
}

/// A single Mamba-2 block.
///
/// Architecture:
/// 1. Input projection: `d_model -> [z, x, B, C, dt]`
/// 2. Depthwise causal conv1d on `[x, B, C]`
/// 3. SiLU activation
/// 4. Multi-head SSD selective scan
/// 5. Gated RMSNorm
/// 6. Output projection: `d_inner -> d_model`
#[derive(Module, Debug)]
pub struct Mamba2Block<B: Backend> {
    pub in_proj: Linear<B>,
    pub conv1d: burn::nn::conv::Conv1d<B>,
    pub a_log: Param<Tensor<B, 1>>,
    pub dt_bias: Param<Tensor<B, 1>>,
    pub d_param: Param<Tensor<B, 1>>,
    pub norm: RmsNorm<B>,
    pub out_proj: Linear<B>,

    pub d_model: usize,
    pub d_state: usize,
    pub d_conv: usize,
    pub expand: usize,
    pub d_inner: usize,
    pub headdim: usize,
    pub nheads: usize,
    pub ngroups: usize,
    pub chunk_size: usize,
    pub eps: f64,
    pub norm_before_gate: bool,
}

impl<B: Backend> Mamba2Block<B> {
    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let dims = input.dims();
        let batch = dims[0];
        let seqlen = dims[1];

        let zxbcdt = self.in_proj.forward(input);

        let total_xbc = self.d_inner + 2 * self.ngroups * self.d_state;
        let dt_offset = 2 * self.d_inner + 2 * self.ngroups * self.d_state;
        let dt_end = dt_offset + self.nheads;

        let z = zxbcdt.clone().slice([0..batch, 0..seqlen, 0..self.d_inner]);
        let xbc = zxbcdt
            .clone()
            .slice([0..batch, 0..seqlen, self.d_inner..dt_offset]);
        let dt_raw = zxbcdt
            .clone()
            .slice([0..batch, 0..seqlen, dt_offset..dt_end]);

        let xbc_t = xbc.swap_dims(1, 2);
        let conv_out = self.conv1d.forward(xbc_t);
        let xbc_act = silu(
            conv_out
                .swap_dims(1, 2)
                .slice([0..batch, 0..seqlen, 0..total_xbc]),
        );

        let x = xbc_act
            .clone()
            .slice([0..batch, 0..seqlen, 0..self.d_inner]);
        let b = xbc_act.clone().slice([
            0..batch,
            0..seqlen,
            self.d_inner..self.d_inner + self.ngroups * self.d_state,
        ]);
        let c = xbc_act.clone().slice([
            0..batch,
            0..seqlen,
            self.d_inner + self.ngroups * self.d_state
                ..self.d_inner + 2 * self.ngroups * self.d_state,
        ]);

        let x_4d = x.reshape([batch, seqlen, self.nheads, self.headdim]);
        let b_4d = b.reshape([batch, seqlen, self.ngroups, self.d_state]);
        let c_4d = c.reshape([batch, seqlen, self.ngroups, self.d_state]);

        let a_log_val = self.a_log.val().clone();
        let a_neg: Tensor<B, 1> = -a_log_val.exp();
        let dt_bias_3d: Tensor<B, 3> = self
            .dt_bias
            .val()
            .clone()
            .unsqueeze_dim::<2>(0)
            .unsqueeze_dim::<3>(0);
        let dt = dt_raw + dt_bias_3d;
        let d = self.d_param.val().clone();

        let (y_ssm, _final_state) = ssd::selective_scan(x_4d, dt, a_neg, b_4d, c_4d, d);

        let y_ssm_3d: Tensor<B, 3> = y_ssm.reshape([batch, seqlen, self.d_inner]);
        let y_gated = self.apply_gated_norm(y_ssm_3d, z);

        self.out_proj.forward(y_gated)
    }

    pub fn step(
        &self,
        input: Tensor<B, 2>,
        _conv_state: &mut Tensor<B, 3>,
        ssm_state: &mut Tensor<B, 4>,
    ) -> Tensor<B, 2> {
        let batch = input.dims()[0];

        let zxbcdt = self.in_proj.forward(input.unsqueeze_dim::<3>(1));

        let total_xbc = self.d_inner + 2 * self.ngroups * self.d_state;
        let dt_offset = 2 * self.d_inner + 2 * self.ngroups * self.d_state;
        let dt_end = dt_offset + self.nheads;

        let z = zxbcdt.clone().slice([0..batch, 0..1, 0..self.d_inner]);
        let xbc = zxbcdt
            .clone()
            .slice([0..batch, 0..1, self.d_inner..dt_offset]);
        let dt_raw = zxbcdt.clone().slice([0..batch, 0..1, dt_offset..dt_end]);

        let xbc_t = xbc.swap_dims(1, 2);
        let conv_out = self.conv1d.forward(xbc_t);
        let xbc_act = silu(
            conv_out
                .swap_dims(1, 2)
                .slice([0..batch, 0..1, 0..total_xbc]),
        );

        let x = xbc_act.clone().slice([0..batch, 0..1, 0..self.d_inner]);
        let b = xbc_act.clone().slice([
            0..batch,
            0..1,
            self.d_inner..self.d_inner + self.ngroups * self.d_state,
        ]);
        let c = xbc_act.clone().slice([
            0..batch,
            0..1,
            self.d_inner + self.ngroups * self.d_state
                ..self.d_inner + 2 * self.ngroups * self.d_state,
        ]);

        let x_4d = x.reshape([batch, 1, self.nheads, self.headdim]);
        let b_4d = b.reshape([batch, 1, self.ngroups, self.d_state]);
        let c_4d = c.reshape([batch, 1, self.ngroups, self.d_state]);

        let a_log_val = self.a_log.val().clone();
        let a_neg: Tensor<B, 1> = -a_log_val.exp();
        let dt_bias_3d: Tensor<B, 3> = self
            .dt_bias
            .val()
            .clone()
            .unsqueeze_dim::<2>(0)
            .unsqueeze_dim::<3>(0);
        let dt = dt_raw + dt_bias_3d;
        let d = self.d_param.val().clone();

        let dt_soft = softplus(dt, 1.0);

        let heads_per_group = self.nheads / self.ngroups;

        let x_3d: Tensor<B, 3> = x_4d.squeeze_dim::<3>(1);
        let dt_2d: Tensor<B, 2> = dt_soft.squeeze_dim::<2>(1);

        let b_g: Tensor<B, 3> = b_4d.squeeze_dim::<3>(1);
        let c_g: Tensor<B, 3> = c_4d.squeeze_dim::<3>(1);

        let b_h: Tensor<B, 3> = if self.ngroups == self.nheads {
            b_g
        } else {
            let expanded = b_g
                .unsqueeze_dim::<4>(1)
                .repeat(&[1, heads_per_group, 1, 1]);
            expanded.reshape([batch, self.nheads, self.d_state])
        };
        let c_h: Tensor<B, 3> = if self.ngroups == self.nheads {
            c_g
        } else {
            let expanded = c_g
                .unsqueeze_dim::<4>(1)
                .repeat(&[1, heads_per_group, 1, 1]);
            expanded.reshape([batch, self.nheads, self.d_state])
        };

        let a_2d: Tensor<B, 2> = a_neg.clone().unsqueeze_dim::<2>(0);
        let da_2d = (dt_2d.clone() * a_2d).exp();
        let da_3d: Tensor<B, 3> = da_2d.unsqueeze_dim::<3>(2);
        let da_4d: Tensor<B, 4> = da_3d.unsqueeze_dim::<4>(3);

        let dt_3d: Tensor<B, 3> = dt_2d.clone().unsqueeze_dim::<3>(2);
        let dt_4d: Tensor<B, 4> = dt_3d.unsqueeze_dim::<4>(3);
        let b_4d_expanded: Tensor<B, 4> = b_h.clone().unsqueeze_dim::<4>(2);
        let x_4d_expanded: Tensor<B, 4> = x_3d.clone().unsqueeze_dim::<4>(3);

        let d_bx = dt_4d * b_4d_expanded * x_4d_expanded;

        let h_new: Tensor<B, 4> = da_4d * ssm_state.clone() + d_bx;
        *ssm_state = h_new.clone();

        let c_4d_expanded: Tensor<B, 4> = c_h.clone().unsqueeze_dim::<4>(2);
        let y_ssm_4d = h_new * c_4d_expanded;
        let y_ssm_3d: Tensor<B, 3> = y_ssm_4d.sum_dim(3).squeeze_dim::<3>(3);

        let d_2d: Tensor<B, 2> = d.clone().unsqueeze_dim::<2>(0);
        let d_3d: Tensor<B, 3> = d_2d.unsqueeze_dim::<3>(2);
        let y_step: Tensor<B, 3> = y_ssm_3d + d_3d * x_3d;

        let y_2d: Tensor<B, 2> = y_step.reshape([batch, self.d_inner]);
        let z_sq: Tensor<B, 2> = z.clone().squeeze_dim::<2>(1).reshape([batch, self.d_inner]);
        let y_gated: Tensor<B, 2> = self.apply_gated_norm_step(y_2d, z_sq);

        let out: Tensor<B, 3> = self.out_proj.forward(y_gated.unsqueeze_dim::<3>(1));
        out.squeeze_dim::<2>(1)
    }

    fn apply_gated_norm(&self, x: Tensor<B, 3>, gate: Tensor<B, 3>) -> Tensor<B, 3> {
        if self.norm_before_gate {
            let normalized = self.norm.forward(x);
            normalized * silu(gate)
        } else {
            let activated = x * silu(gate);
            self.norm.forward(activated)
        }
    }

    fn apply_gated_norm_step(&self, x: Tensor<B, 2>, gate: Tensor<B, 2>) -> Tensor<B, 2> {
        if self.norm_before_gate {
            let x_3d: Tensor<B, 3> = x.unsqueeze_dim::<3>(1);
            let gate_3d: Tensor<B, 3> = gate.unsqueeze_dim::<3>(1);
            let normalized = self.norm.forward(x_3d);
            let gated = normalized * silu(gate_3d);
            gated.squeeze_dim::<2>(1)
        } else {
            let activated: Tensor<B, 2> = x * silu(gate);
            let activated_3d: Tensor<B, 3> = activated.unsqueeze_dim::<3>(1);
            let normalized = self.norm.forward(activated_3d);
            normalized.squeeze_dim::<2>(1)
        }
    }

    pub fn allocate_states(
        &self,
        batch_size: usize,
        device: &Device<B>,
    ) -> (Tensor<B, 3>, Tensor<B, 4>) {
        let conv_dim = self.d_inner + 2 * self.ngroups * self.d_state;
        let conv_state: Tensor<B, 3> =
            Tensor::zeros([batch_size, self.d_conv, conv_dim], device).swap_dims(1, 2);
        let ssm_state: Tensor<B, 4> = Tensor::zeros(
            [batch_size, self.nheads, self.headdim, self.d_state],
            device,
        );
        (conv_state, ssm_state)
    }
}

/// Configuration for the complete Mamba-2 language model.
#[derive(Config, Debug)]
pub struct Mamba2ModelConfig {
    pub vocab_size: usize,
    pub d_model: usize,
    pub n_layers: usize,
    #[config(default = 64)]
    pub d_state: usize,
    #[config(default = 4)]
    pub d_conv: usize,
    #[config(default = 2)]
    pub expand: usize,
    #[config(default = 64)]
    pub headdim: usize,
    #[config(default = 1)]
    pub ngroups: usize,
    #[config(default = 1e-5)]
    pub eps: f64,
    #[config(default = false)]
    pub bias: bool,
    #[config(default = true)]
    pub conv_bias: bool,
}

impl Mamba2ModelConfig {
    pub fn init<B: Backend>(&self, device: &Device<B>) -> Mamba2Model<B> {
        let embedding = burn::nn::EmbeddingConfig::new(self.vocab_size, self.d_model).init(device);

        let mut layers = Vec::with_capacity(self.n_layers);
        for _ in 0..self.n_layers {
            let layer = Mamba2BlockConfig::new(self.d_model)
                .with_d_state(self.d_state)
                .with_d_conv(self.d_conv)
                .with_expand(self.expand)
                .with_headdim(self.headdim)
                .with_ngroups(self.ngroups)
                .with_eps(self.eps)
                .with_bias(self.bias)
                .with_conv_bias(self.conv_bias)
                .init(device);
            layers.push(layer);
        }

        let norm_f = RmsNormConfig::new(self.d_model)
            .with_epsilon(self.eps)
            .init(device);

        let lm_head = LinearConfig::new(self.d_model, self.vocab_size)
            .with_bias(false)
            .init(device);

        Mamba2Model {
            embedding,
            layers,
            norm_f,
            lm_head,
            d_model: self.d_model,
            n_layers: self.n_layers,
        }
    }
}

/// The complete Mamba-2 language model.
#[derive(Module, Debug)]
pub struct Mamba2Model<B: Backend> {
    pub embedding: burn::nn::Embedding<B>,
    pub layers: Vec<Mamba2Block<B>>,
    pub norm_f: RmsNorm<B>,
    pub lm_head: Linear<B>,
    pub d_model: usize,
    pub n_layers: usize,
}

impl<B: Backend> Mamba2Model<B> {
    pub fn forward(&self, input: Tensor<B, 2, burn::tensor::Int>) -> Tensor<B, 3> {
        let x = self.embedding.forward(input);
        let mut h = x;
        for layer in &self.layers {
            h = layer.forward(h);
        }
        h = self.norm_f.forward(h);
        self.lm_head.forward(h)
    }

    pub fn allocate_inference_states(
        &self,
        batch_size: usize,
        device: &Device<B>,
    ) -> Vec<(Tensor<B, 3>, Tensor<B, 4>)> {
        self.layers
            .iter()
            .map(|layer| layer.allocate_states(batch_size, device))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::Distribution;

    #[test]
    fn test_mamba2_block_forward() {
        let device = Default::default();
        let config = Mamba2BlockConfig::new(64)
            .with_d_state(16)
            .with_expand(2)
            .with_headdim(32);
        let block = config.init::<burn::backend::NdArray>(&device);

        let input = Tensor::<burn::backend::NdArray, 3>::random(
            [2, 16, 64],
            Distribution::Default,
            &device,
        );
        let output = block.forward(input);
        assert_eq!(output.dims(), [2, 16, 64]);
    }

    #[test]
    fn test_mamba2_block_step() {
        let device = Default::default();
        let config = Mamba2BlockConfig::new(64)
            .with_d_state(16)
            .with_expand(2)
            .with_headdim(32);
        let block = config.init::<burn::backend::NdArray>(&device);

        let input =
            Tensor::<burn::backend::NdArray, 2>::random([2, 64], Distribution::Default, &device);
        let (mut conv_state, mut ssm_state) = block.allocate_states(2, &device);
        let output = block.step(input, &mut conv_state, &mut ssm_state);
        assert_eq!(output.dims(), [2, 64]);
    }

    #[test]
    fn test_mamba2_model_forward() {
        let device = Default::default();
        let config = Mamba2ModelConfig::new(100, 64, 2)
            .with_d_state(16)
            .with_expand(2)
            .with_headdim(32);
        let model = config.init::<burn::backend::NdArray>(&device);

        let input = Tensor::<burn::backend::NdArray, 2, burn::tensor::Int>::from_data(
            [[1, 2, 3, 4, 5, 6, 7, 8]],
            &device,
        );
        let output = model.forward(input);
        assert_eq!(output.dims(), [1, 8, 100]);
    }
}
