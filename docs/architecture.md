# H-Net Laboratory Architecture

## Goal

This workspace explores an end-to-end byte-level sequence modeling stack in Rust using Burn 0.21. The current implementation is not yet an H-Net, but it provides the validated low-level components required to build one: byte corpus ingestion, Mamba-2, causal Transformer attention, sparse Mixture of Experts, and an integration flow that runs them over the same byte batch.

## Stack

- Language: Rust 2021 for reusable model crates; `flow` currently uses Rust 2024.
- ML framework: Burn 0.21 with `train` and `ndarray` features.
- Backend used for validation: `burn::backend::NdArray`.
- Input representation: raw bytes as integer token IDs with `vocab_size = 256`.

## Workspace Layout

- `Corpus/`: corpus construction utilities and generated `obsidian_corpus.txt`.
- `mamba-2/`: canonical Mamba-2 style block and language model implementation.
- `self-attention/`: causal Transformer decoder/language model implementation.
- `mixture-of-experts/`: sparse top-k MoE layer with router, independent experts, exact top-k scatter mask, and load-balancing loss.
- `flow/`: integration executable that loads the corpus, creates byte batches, runs Mamba, Transformer, and MoE, and prints structural/numerical diagnostics.
- `docs/`: architecture, implementation plans, and structural decisions.

## Current Data Flow

```text
Corpus/obsidian_corpus.txt
-> ByteCorpus
-> Tensor<Int>[batch, seq_len]
-> Mamba2Model hidden/logits
-> TransformerLm hidden/logits
-> SparseMoE over hidden states
-> diagnostics
```

The current flow validates that Mamba, Transformer, and MoE share compatible hidden dimensions:

```text
input:        [1, 1024]
hidden:       [1, 1024, 64]
lm logits:    [1, 1024, 256]
moe router:   [1, 1024, 4]
moe hidden:   [1, 1024, 64]
```

## Current Component Contracts

### Byte Corpus

- Source: `Corpus/obsidian_corpus.txt`.
- Tokenization: byte-level, `u8 -> i64`.
- Batch shape: `Tensor<B, 2, Int>` with `[batch, seq_len]`.

### Mamba-2

- Crate: `mamba-2`.
- LM input: `Tensor<B, 2, Int>`.
- LM output: `Tensor<B, 3>` with `[batch, seq_len, vocab_size]`.
- Block input/output: `Tensor<B, 3>` with `[batch, seq_len, d_model]`.
- Important invariant: SSD transition parameter `A` must be negative: `A = -exp(A_log)`.

### Causal Transformer

- Crate: `self-attention`.
- LM input: `Tensor<B, 2, Int>`.
- LM output: `Tensor<B, 3>` with `[batch, seq_len, vocab_size]`.
- Block input/output: `Tensor<B, 3>` with `[batch, seq_len, d_model]`.
- Uses causal attention via Burn's native attention kernel.

### Sparse MoE

- Crate: `mixture-of-experts`.
- Input/output hidden contract: `Tensor<B, 3>` with `[batch, seq_len, d_model]`.
- Router output: `[batch, seq_len, num_experts]`.
- Routing: exact top-k via `topk_with_indices` and `scatter`.
- Auxiliary signal: load-balance loss.

## Current Validation Status

- `mamba-2`: tests pass after fixing the SSD transition sign.
- `self-attention`: tests pass.
- `mixture-of-experts`: tests pass.
- `flow`: tests pass and end-to-end run produces no NaNs in Mamba, Transformer, logits, MoE hidden, or MoE router outputs.

## Known Corpus Characteristics

- Corpus size: approximately 5.19 MB from 1608 Markdown files.
- Current batch length: 1024 bytes.
- Current batch is dominated by artificial file headers and Markdown/YAML structure.
- Current batch vocabulary coverage is low: 62 used byte values out of 256.
- Space byte dominates the sample, indicating strong formatting bias in the first batch.

## Gap To H-Net

The current architecture is byte-level and end-to-end over fixed byte positions, but it does not yet implement adaptive tokenization. A real H-Net style architecture requires learned dynamic chunking and explicit hierarchy:

```text
bytes
-> local byte encoder
-> learned boundary/chunking module
-> chunk-level compression
-> high-level sequence model
-> expansion/alignment back to bytes
-> byte-level prediction
```

The implementation plan for this transition is persisted in `docs/h-net-implementation-plan.md`.
