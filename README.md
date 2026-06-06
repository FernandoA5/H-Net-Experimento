# H-Net Laboratory

Byte-level sequence modeling stack in Rust with [Burn 0.21](https://burn.dev). Explores Mamba-2, causal Transformer, and sparse Mixture of Experts as primitives toward H-Net-style adaptive tokenization (learned dynamic chunking over raw bytes).

## Crates

| Crate | Description |
|---|---|
| `mamba-2/` | Mamba-2 SSD recurrent block and language model |
| `self-attention/` | Causal Transformer decoder with native attention |
| `mixture-of-experts/` | Sparse top-k MoE with load-balance loss |
| `flow/` | Integration: corpus ingestion, forward pass, diagnostics |

## Quick Start

```sh
cargo test            # all crates
cargo run -p flow     # end-to-end byte-level forward pass
```

## Data Flow

```
Corpus/obsidian_corpus.txt
  → ByteCorpus
  → Tensor<Int>[B, seq_len]
  → Mamba2Model / TransformerLm
  → SparseMoE
  → diagnostics
```

## Status

- All crates pass tests and produce zero NaNs in end-to-end forward passes.
- Not yet an H-Net adaptive tokenizer — see `docs/h-net-implementation-plan.md`.

## License

MIT
