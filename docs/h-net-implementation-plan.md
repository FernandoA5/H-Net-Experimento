# H-Net Implementation Plan

## Objective

Build an H-Net style end-to-end hierarchical sequence model over raw bytes, inspired by dynamic chunking for learned adaptive tokenization. Each phase below is a concrete component. Components should be implemented as independent Rust/Burn crates or modules with explicit tensor contracts, tests, and integration into `flow` before moving to the next phase.

## Phase 1: Corpus Sampler Component

### Purpose

Replace the current first-1024-bytes batch with reusable corpus sampling that can produce deterministic sequential batches and randomized offset batches.

### Contract

```text
input: corpus bytes
output: Tensor<B, 2, Int> [batch, seq_len]
```

### Requirements

- Preserve byte-level vocabulary `0..=255`.
- Support deterministic offset for reproducible probes.
- Support random offsets for training.
- Optionally skip or down-weight generated `--- FILE:` headers.
- Report byte histogram, UTF-8 ratio, and effective byte coverage per batch.

### Acceptance Criteria

- Existing `flow` can select first-batch or random-batch mode.
- Batch diagnostics show whether the sample is header-heavy or content-heavy.

## Phase 2: Local Byte Encoder Component

### Purpose

Produce contextual byte representations before chunk boundary prediction.

### Contract

```text
input:  Tensor<B, 2, Int> [batch, seq_len]
output: Tensor<B, 3>      [batch, seq_len, d_model]
```

### Candidate Implementation

- Byte embedding.
- One or more local Mamba-2 blocks or causal convolutional blocks.
- Optional causal Transformer block for short local context.

### Requirements

- Must be causal.
- Must preserve one hidden vector per byte.
- Must remain numerically finite on current corpus batch.

### Acceptance Criteria

- Hidden shape `[batch, seq_len, d_model]`.
- NaN count is zero.
- Can be used by both the boundary predictor and byte-level decoder.

## Phase 3: Boundary Predictor Component

### Purpose

Predict content/context-dependent chunk boundaries over byte positions.

### Contract

```text
input:  Tensor<B, 3> [batch, seq_len, d_model]
output: boundary_logits Tensor<B, 3> [batch, seq_len, 2]
output: boundary_probs  Tensor<B, 2> [batch, seq_len]
output: boundary_mask   Tensor<B, 2> [batch, seq_len]
```

### Requirements

- Predict whether each byte ends a chunk.
- Enforce causal prediction: boundary at position `t` cannot depend on bytes after `t`.
- Provide a differentiable path during training. If hard masks are used for routing/compression, keep soft probabilities for losses and diagnostics.
- Include constraints for minimum/maximum chunk size to avoid degenerate all-one-byte or all-sequence chunks.

### Acceptance Criteria

- Produces valid boundary probabilities in `[0, 1]`.
- Produces at least one chunk per sequence.
- Diagnostics report chunk count, mean chunk length, min/max chunk length, and boundary entropy.

## Phase 4: Dynamic Chunker Component

### Purpose

Convert boundary predictions into chunk IDs and chunk masks that define adaptive tokens.

### Contract

```text
input:  boundary_mask Tensor<B, 2> [batch, seq_len]
output: chunk_ids     Tensor<B, 2, Int> [batch, seq_len]
output: chunk_mask    Tensor<B, 3>      [batch, seq_len, max_chunks]
output: chunk_lengths Tensor<B, 2>      [batch, max_chunks]
```

### Requirements

- Assign every byte to exactly one chunk.
- Preserve byte order.
- Support variable number of chunks per sequence through padding/masks.
- Keep a clear distinction between hard chunk IDs for structure and soft boundary probabilities for training signals.

### Acceptance Criteria

- Sum over chunks for each byte equals one.
- Chunk lengths match the boundary-derived segmentation.
- `flow` can print human-readable chunks from the current corpus batch.

## Phase 5: Chunk Compression Component

### Purpose

Compress byte-level hidden states into chunk-level representations.

### Contract

```text
input:  byte_hidden Tensor<B, 3> [batch, seq_len, d_model]
input:  chunk_mask  Tensor<B, 3> [batch, seq_len, max_chunks]
output: chunk_hidden Tensor<B, 3> [batch, max_chunks, d_model]
```

### Candidate Implementations

- Masked mean pooling baseline.
- Boundary-state pooling: use hidden state at chunk-ending byte.
- Attention pooling within chunk.

### Requirements

- Start with masked mean or boundary-state pooling as the reference implementation.
- Preserve gradient path from chunk-level model to byte encoder.
- Avoid leaking future bytes into earlier chunk representations beyond causal boundary semantics.

### Acceptance Criteria

- Chunk hidden shape `[batch, max_chunks, d_model]`.
- Padded chunks are masked out.
- Diagnostics show compression ratio `seq_len / num_chunks`.

## Phase 6: Chunk-Level Sequence Model Component

### Purpose

Model the adaptive token sequence created by the chunker.

### Contract

```text
input:  chunk_hidden Tensor<B, 3> [batch, max_chunks, d_model]
input:  chunk_valid_mask
output: chunk_context Tensor<B, 3> [batch, max_chunks, d_model]
```

### Candidate Implementation

- Mamba-2 blocks over chunks.
- Causal Transformer blocks over chunks.
- Optional SparseMoE after chunk-level blocks.

### Requirements

- Must be causal over chunks.
- Must respect padded chunk mask.
- Must support later stacking into multiple hierarchy stages.

### Acceptance Criteria

- Chunk-level output remains finite.
- Same `d_model` contract as existing blocks unless a projection component is introduced.

## Phase 7: Chunk Expansion Component

### Purpose

Broadcast or decode chunk-level context back to byte positions.

### Contract

```text
input:  chunk_context Tensor<B, 3> [batch, max_chunks, d_model]
input:  chunk_mask    Tensor<B, 3> [batch, seq_len, max_chunks]
output: byte_context  Tensor<B, 3> [batch, seq_len, d_model]
```

### Requirements

- Assign each byte the context vector of its chunk.
- Preserve byte order and sequence length.
- Support combining byte-local hidden and chunk-context hidden through residual, gated fusion, or projection.

### Acceptance Criteria

- Expanded output shape `[batch, seq_len, d_model]`.
- Every byte receives exactly one chunk context.

## Phase 8: Byte-Level Decoder Component

### Purpose

Predict next byte from fused byte-local and chunk-level representations.

### Contract

```text
input:  byte_hidden/context Tensor<B, 3> [batch, seq_len, d_model]
output: logits              Tensor<B, 3> [batch, seq_len, 256]
```

### Requirements

- Use causal next-byte LM objective.
- Preserve compatibility with current `flow` diagnostics.
- Keep logits byte-level; do not introduce BPE or fixed external tokenization.

### Acceptance Criteria

- Produces `[batch, seq_len, 256]` logits.
- Cross-entropy loss can be computed against shifted byte targets.

## Phase 9: Training Losses Component

### Purpose

Train the H-Net end-to-end and prevent degenerate chunking.

### Required Losses

- Next-byte cross entropy.
- Boundary entropy/regularization signal.
- Chunk length regularization or constraint loss.
- Optional MoE load-balancing loss when MoE is active.

### Acceptance Criteria

- Loss object reports all scalar components separately.
- Training can run a forward/backward/update step with Burn autodiff backend.

## Phase 10: Multi-Stage Hierarchy Component

### Purpose

Generalize from one hierarchy level to multiple levels of learned abstraction.

### Contract

```text
bytes -> level_1_chunks -> level_2_chunks -> ... -> byte logits
```

### Requirements

- Reuse Boundary Predictor, Dynamic Chunker, Compression, Chunk Model, and Expansion components recursively.
- Keep explicit diagnostics per level.
- Prevent silent shape ambiguity by naming every level's sequence length and mask.

### Acceptance Criteria

- One-stage H-Net remains the default baseline.
- Two-stage H-Net can run over a short batch and report per-level compression ratios.

## Phase 11: Evaluation And Diagnostics Component

### Purpose

Measure whether adaptive tokenization is actually emerging.

### Required Diagnostics

- Chunk length distribution.
- Boundary positions visualized over decoded text.
- Compression ratio.
- Boundary entropy.
- Per-byte CE grouped by chunk length.
- Robustness tests on accent changes, spacing changes, byte corruption, code, and Markdown headers.

### Acceptance Criteria

- `flow` can print chunked text for a batch.
- Diagnostics distinguish artificial corpus headers from content chunks.

## Implementation Order

1. Corpus Sampler Component.
2. Local Byte Encoder Component.
3. Boundary Predictor Component.
4. Dynamic Chunker Component.
5. Chunk Compression Component.
6. Chunk-Level Sequence Model Component.
7. Chunk Expansion Component.
8. Byte-Level Decoder Component.
9. Training Losses Component.
10. Multi-Stage Hierarchy Component.
11. Evaluation And Diagnostics Component.

## First Concrete Milestone

Build a one-stage H-Net prototype with:

```text
ByteCorpus
-> LocalByteEncoder
-> BoundaryPredictor
-> DynamicChunker
-> MaskedMeanChunkCompressor
-> ChunkMambaOrTransformer
-> ChunkExpander
-> ByteDecoder
```

The milestone is complete when `flow` prints byte logits, chunk boundaries, compression ratio, and zero NaNs for a 1024-byte batch.
