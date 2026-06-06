# Decision 0001: Build H-Net As A Componentized Byte Hierarchy

## Status

Accepted

## Context

The current project has validated independent Rust/Burn crates for Mamba-2, causal Transformer attention, and sparse Mixture of Experts. The `flow` crate confirms these components can consume the same byte-level corpus batch and produce compatible hidden states and logits without NaNs.

However, this is not yet an H-Net style adaptive tokenizer. The missing architectural feature is learned dynamic chunking and explicit hierarchy over raw bytes.

## Decision

H-Net functionality will be implemented as a sequence of explicit components rather than by modifying the existing Mamba, Transformer, or MoE crates directly.

The planned component boundary is:

```text
ByteCorpus
-> LocalByteEncoder
-> BoundaryPredictor
-> DynamicChunker
-> ChunkCompressor
-> ChunkSequenceModel
-> ChunkExpander
-> ByteDecoder
```

Existing model crates remain reusable primitives:

- `mamba-2` can serve as local byte encoder or chunk-level model.
- `self-attention` can serve as local byte encoder or chunk-level model.
- `mixture-of-experts` can serve as an optional block over byte or chunk hidden states.

## Consequences

- The adaptive tokenization logic remains isolated and testable.
- The project can validate one hierarchy stage before adding recursive multi-stage hierarchy.
- The current `flow` crate remains the integration and diagnostics surface.
- New components must expose explicit tensor contracts and diagnostics, especially for chunk masks, chunk lengths, boundary entropy, compression ratio, and NaN counts.

## Non-Goals

- Do not introduce BPE or a fixed tokenizer.
- Do not hide chunking inside Mamba or Transformer internals.
- Do not train a multi-stage H-Net before a one-stage H-Net produces valid adaptive chunks and byte logits.
