#[allow(dead_code)]
mod corpus_byte_level;
#[allow(dead_code)]
mod exp_decoder;

use std::{error::Error, io};

use burn::{
    backend::NdArray,
    tensor::{Int, Tensor, TensorData},
};
use corpus_byte_level::{ByteCorpus, DEFAULT_CORPUS_PATH};

use mamba_2::Mamba2ModelConfig;
use mixture_of_experts::SparseMoEConfig;
use self_attention::TransformerLmConfig;

fn main() -> Result<(), Box<dyn Error>> {
    let corpus = ByteCorpus::from_file(DEFAULT_CORPUS_PATH)?;
    let device = Default::default();
    let batch_size = 1;
    let seq_len = 1024;
    let token_ids = corpus
        .batch_tokens(batch_size, seq_len)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "corpus is too short"))?;
    let input = Tensor::<NdArray, 2, Int>::from_data(
        TensorData::new(token_ids.clone(), [batch_size, seq_len]),
        &device,
    );
    let mamba_model = Mamba2ModelConfig::new(256, 64, 1)
        .with_d_state(16)
        .with_expand(2)
        .with_headdim(32)
        .init::<NdArray>(&device);
    let transformer_model = TransformerLmConfig::new(256, 64, 1, 4, seq_len)
        .with_ffn_expansion(4)
        .init::<NdArray>(&device);
    let moe = SparseMoEConfig::new(64, 4, 2).init::<NdArray>(&device);

    let mut mamba_hidden = mamba_model.embedding.forward(input.clone());
    for layer in &mamba_model.layers {
        mamba_hidden = layer.forward(mamba_hidden);
    }
    mamba_hidden = mamba_model.norm_f.forward(mamba_hidden);
    let mamba_hidden_input_nan_count = nan_count(mamba_hidden.clone());
    let mamba_logits = mamba_model.lm_head.forward(mamba_hidden.clone());
    let mamba_logits_nan_count = nan_count(mamba_logits.clone());
    let mamba_moe = moe.forward(mamba_hidden);

    let mut transformer_hidden = transformer_model.token_embedding.forward(input.clone());
    let positions = Tensor::<NdArray, 1, Int>::arange(0..seq_len as i64, &device)
        .unsqueeze_dim::<2>(0)
        .repeat(&[batch_size, 1]);
    transformer_hidden =
        transformer_hidden + transformer_model.position_embedding.forward(positions);
    for layer in &transformer_model.layers {
        transformer_hidden = layer.forward(transformer_hidden);
    }
    transformer_hidden = transformer_model.norm_f.forward(transformer_hidden);
    let transformer_hidden_input_nan_count = nan_count(transformer_hidden.clone());
    let transformer_logits = transformer_model
        .lm_head
        .forward(transformer_hidden.clone());
    let transformer_logits_nan_count = nan_count(transformer_logits.clone());
    let transformer_moe = moe.forward(transformer_hidden);

    let mamba_moe_load = expert_load(mamba_moe.expert_mask.clone());
    let transformer_moe_load = expert_load(transformer_moe.expert_mask.clone());
    let mamba_balance_loss = scalar_value(mamba_moe.load_balance_loss.clone());
    let transformer_balance_loss = scalar_value(transformer_moe.load_balance_loss.clone());
    let mamba_hidden_nan_count = nan_count(mamba_moe.hidden.clone());
    let transformer_hidden_nan_count = nan_count(transformer_moe.hidden.clone());
    let mamba_router_nan_count = nan_count(mamba_moe.router_logits.clone());
    let transformer_router_nan_count = nan_count(transformer_moe.router_logits.clone());

    // ------ analysis ------
    let bytes: Vec<u8> = token_ids.iter().map(|t| *t as u8).collect();
    let text = String::from_utf8_lossy(&bytes);
    let chars: Vec<char> = text.chars().collect();
    let word_chars: Vec<&str> = text
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .collect();

    let mut byte_hist = [0u64; 256];
    for b in &bytes {
        byte_hist[*b as usize] += 1;
    }
    let mut hist_pairs: Vec<(usize, u64)> = byte_hist.iter().copied().enumerate().collect();
    hist_pairs.sort_by(|a, b| b.1.cmp(&a.1));
    let top_bytes: Vec<(usize, u64)> = hist_pairs.into_iter().take(10).collect();
    let unused_bytes: Vec<usize> = byte_hist
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, count)| *count == 0)
        .map(|(byte, _)| byte)
        .collect();

    let avg_word_len: f64 = if word_chars.is_empty() {
        0.0
    } else {
        word_chars.iter().map(|w| w.len()).sum::<usize>() as f64 / word_chars.len() as f64
    };

    println!("=== Corpus ===");
    println!("path:  {}", DEFAULT_CORPUS_PATH);
    println!("bytes: {}", corpus.len());
    println!();
    println!("=== Batch (first {seq_len} bytes) ===");
    println!("tokens: {}", token_ids.len());
    println!(
        "text preview: {:?}",
        text.chars().take(160).collect::<String>()
    );
    println!(
        "words: {} (preview: {:?})",
        word_chars.len(),
        &word_chars[..10.min(word_chars.len())]
    );
    println!("avg word length: {avg_word_len:.1}");
    println!(
        "characters: {} (multi-byte ratio: {:.3})",
        chars.len(),
        token_ids.len() as f64 / chars.len().max(1) as f64
    );
    println!();
    println!("=== Byte distribution (top 5) ===");
    for (b, count) in &top_bytes {
        let byte_val = *b as u8;
        let label = match byte_val {
            10 => "\\n".to_string(),
            32 => "␣".to_string(),
            9 => "⇥".to_string(),
            _ if byte_val.is_ascii_graphic() => (byte_val as char).to_string(),
            _ => format!("{byte_val}"),
        };
        println!(
            "  {b:3} ({label:>2}) : {count:5} ({:.2}%)",
            *count as f64 / token_ids.len() as f64 * 100.0
        );
    }
    println!("unused byte values: {} (out of 256)", unused_bytes.len());
    println!(
        "vocab coverage: {:.1}%",
        (256 - unused_bytes.len()) as f64 / 256.0 * 100.0
    );
    println!();
    println!("=== Forward ===");
    println!("mamba logits dims: {:?}", mamba_logits.dims());
    println!("transformer logits dims: {:?}", transformer_logits.dims());
    println!("mamba moe hidden dims: {:?}", mamba_moe.hidden.dims());
    println!(
        "transformer moe hidden dims: {:?}",
        transformer_moe.hidden.dims()
    );
    println!(
        "mamba moe router dims: {:?}",
        mamba_moe.router_logits.dims()
    );
    println!(
        "transformer moe router dims: {:?}",
        transformer_moe.router_logits.dims()
    );
    println!("mamba moe expert load: {:?}", mamba_moe_load);
    println!("transformer moe expert load: {:?}", transformer_moe_load);
    println!("mamba moe load balance loss: {mamba_balance_loss:.6}");
    println!("transformer moe load balance loss: {transformer_balance_loss:.6}");
    println!("mamba hidden pre-moe NaNs: {mamba_hidden_input_nan_count}");
    println!("transformer hidden pre-moe NaNs: {transformer_hidden_input_nan_count}");
    println!("mamba logits NaNs: {mamba_logits_nan_count}");
    println!("transformer logits NaNs: {transformer_logits_nan_count}");
    println!("mamba moe hidden NaNs: {mamba_hidden_nan_count}");
    println!("transformer moe hidden NaNs: {transformer_hidden_nan_count}");
    println!("mamba moe router NaNs: {mamba_router_nan_count}");
    println!("transformer moe router NaNs: {transformer_router_nan_count}");

    Ok(())
}

fn expert_load(mask: Tensor<NdArray, 3>) -> Vec<f32> {
    mask.mean_dim(0)
        .mean_dim(1)
        .squeeze_dim::<2>(0)
        .squeeze_dim::<1>(0)
        .into_data()
        .to_vec::<f32>()
        .unwrap()
}

fn scalar_value(tensor: Tensor<NdArray, 1>) -> f32 {
    tensor.into_data().to_vec::<f32>().unwrap()[0]
}

fn nan_count<const D: usize>(tensor: Tensor<NdArray, D>) -> f32 {
    tensor
        .is_nan()
        .float()
        .sum()
        .into_data()
        .to_vec::<f32>()
        .unwrap()[0]
}
