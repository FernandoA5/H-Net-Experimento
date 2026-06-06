use burn::tensor::{activation::softplus, backend::Backend, Tensor};

/// Simplified sequential selective scan for Mamba-2 SSD.
///
/// This is the naive O(L) sequential implementation, correct but not
/// as efficient as the chunked scan. Used as reference and fallback.
///
/// Shapes:
/// - x:    `[B, L, H, P]`
/// - dt:   `[B, L, H]`    (raw dt, before softplus)
/// - a:    `[H]`          (A = -exp(A_log), already negative)
/// - b:    `[B, L, G, N]`
/// - c:    `[B, L, G, N]`
/// - d:    `[H]`
pub fn selective_scan<B: Backend>(
    x: Tensor<B, 4>,
    dt: Tensor<B, 3>,
    a: Tensor<B, 1>,
    b: Tensor<B, 4>,
    c: Tensor<B, 4>,
    d_skip: Tensor<B, 1>,
) -> (Tensor<B, 4>, Tensor<B, 4>) {
    let dims_x = x.dims();
    let batch = dims_x[0];
    let seq_len = dims_x[1];
    let n_heads = dims_x[2];
    let head_dim = dims_x[3];
    let d_state = b.dims()[3];
    let n_groups = b.dims()[2];
    let device = x.device();

    let dt_soft = softplus(dt, 1.0);
    let heads_per_group = n_heads / n_groups;

    let mut h: Tensor<B, 4> = Tensor::zeros([batch, n_heads, head_dim, d_state], &device);
    let mut outputs: Vec<Tensor<B, 4>> = Vec::with_capacity(seq_len);

    for t in 0..seq_len {
        let x_t = x
            .clone()
            .slice([0..batch, t..t + 1, 0..n_heads, 0..head_dim]);
        let dt_t_raw = dt_soft.clone().slice([0..batch, t..t + 1, 0..n_heads]);
        let b_t = b
            .clone()
            .slice([0..batch, t..t + 1, 0..n_groups, 0..d_state]);
        let c_t = c
            .clone()
            .slice([0..batch, t..t + 1, 0..n_groups, 0..d_state]);

        let x_3d: Tensor<B, 3> = x_t.squeeze_dim::<3>(1);
        let dt_2d: Tensor<B, 2> = dt_t_raw.squeeze_dim::<2>(1);
        let b_3d: Tensor<B, 3> = b_t.squeeze_dim::<3>(1);
        let c_3d: Tensor<B, 3> = c_t.squeeze_dim::<3>(1);

        let b_h: Tensor<B, 3> = if n_groups == n_heads {
            b_3d
        } else {
            let expanded = b_3d
                .unsqueeze_dim::<4>(1)
                .repeat(&[1, heads_per_group, 1, 1]);
            expanded.reshape([batch, n_heads, d_state])
        };
        let c_h: Tensor<B, 3> = if n_groups == n_heads {
            c_3d
        } else {
            let expanded = c_3d
                .unsqueeze_dim::<4>(1)
                .repeat(&[1, heads_per_group, 1, 1]);
            expanded.reshape([batch, n_heads, d_state])
        };

        let a_2d: Tensor<B, 2> = a.clone().unsqueeze_dim::<2>(0);
        let da_2d = (dt_2d.clone() * a_2d).exp();
        let da_3d: Tensor<B, 3> = da_2d.unsqueeze_dim::<3>(2);
        let da_4d: Tensor<B, 4> = da_3d.unsqueeze_dim::<4>(3);

        let dt_3d: Tensor<B, 3> = dt_2d.clone().unsqueeze_dim::<3>(2);
        let dt_4d: Tensor<B, 4> = dt_3d.unsqueeze_dim::<4>(3);
        let b_4d: Tensor<B, 4> = b_h.clone().unsqueeze_dim::<4>(2);
        let x_4d: Tensor<B, 4> = x_3d.clone().unsqueeze_dim::<4>(3);

        let d_bx = dt_4d * b_4d * x_4d;

        h = da_4d * h.clone() + d_bx;

        let c_4d: Tensor<B, 4> = c_h.clone().unsqueeze_dim::<4>(2);
        let y_ssm_4d = h.clone() * c_4d;
        let y_ssm_3d: Tensor<B, 3> = y_ssm_4d.sum_dim(3).squeeze_dim::<3>(3);

        let d_2d: Tensor<B, 2> = d_skip.clone().unsqueeze_dim::<2>(0);
        let d_3d: Tensor<B, 3> = d_2d.unsqueeze_dim::<3>(2);
        let y = y_ssm_3d + d_3d * x_3d;

        outputs.push(y.unsqueeze_dim::<4>(1));
    }

    let y = Tensor::cat(outputs, 1);
    (y, h)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::Distribution;

    #[test]
    fn test_selective_scan() {
        let device = Default::default();
        let batch = 2;
        let seq_len = 8;
        let n_heads = 4;
        let head_dim = 16;
        let d_state = 8;
        let n_groups = 1;

        let x = Tensor::<burn::backend::NdArray, 4>::random(
            [batch, seq_len, n_heads, head_dim],
            Distribution::Default,
            &device,
        );
        let dt = Tensor::<burn::backend::NdArray, 3>::random(
            [batch, seq_len, n_heads],
            Distribution::Default,
            &device,
        );
        let a = Tensor::<burn::backend::NdArray, 1>::from_floats([-1.0, -2.0, -3.0, -4.0], &device);
        let b = Tensor::<burn::backend::NdArray, 4>::random(
            [batch, seq_len, n_groups, d_state],
            Distribution::Default,
            &device,
        );
        let c = Tensor::<burn::backend::NdArray, 4>::random(
            [batch, seq_len, n_groups, d_state],
            Distribution::Default,
            &device,
        );
        let d_skip = Tensor::<burn::backend::NdArray, 1>::ones(&[n_heads], &device);

        let (y, _state) = selective_scan(x, dt, a, b, c, d_skip);
        assert_eq!(y.dims(), [batch, seq_len, n_heads, head_dim]);
    }
}
