# MSA: Memory Sparse Attention — Analysis

*Generated 2026-03-21 by Opus agent reading README + figures (PDF unrenderable on Windows)*

---

## Paper Overview

**Repo**: https://github.com/EverMind-AI/MSA
**Paper**: Zenodo DOI 10.5281/zenodo.19103670 (March 2026)
**License**: MIT
**Authors**: Chen et al. (EverMind AI)

**Problem addressed**: Full attention scales quadratically, limiting effective context to 128K–1M tokens. Existing approaches (hybrid linear attention, fixed-size state memory, RAG/agents) either degrade at extreme scale, lack end-to-end differentiability, or require complex pipelines.

**Core approach**: An end-to-end trainable sparse attention layer that replaces full attention in upper transformer layers. Documents are encoded offline into compressed KV states via chunk-mean pooling. A learned router (cosine similarity on projected routing keys) selects top-k relevant document blocks per query. Selected compressed KV is concatenated with local context for autoregressive generation. Document-wise RoPE prevents position drift between short training (64K) and long inference (100M).

**Key results**:
- <9% degradation from 16K→100M tokens on MS MARCO QA
- 94.84% NIAH accuracy at 1M tokens (backbone collapses to 24.69%)
- Beats same-backbone RAG (+16%), RAG+rerank (+11.5%), and HippoRAG2 (+14.8%) on 9-dataset QA average
- Competitive with best-of-breed RAG stacks using 60x larger generators (KaLMv2 + Qwen3-235B)
- Runs on 2×A800 GPUs via Memory Parallel (routing keys on GPU, content KV in host DRAM, async fetch)

**Training**: 158.95B-token continuous pretraining with auxiliary routing loss on Qwen3-4B-Instruct-2507, followed by two-stage SFT (8K→64K curriculum).

---

## Architecture

### MSA Layer

Replaces standard attention in upper transformer layers (lower layers keep independent document processing for hierarchical alignment):

1. **Offline encoding**: Forward pass over corpus produces chunk-mean-pooled K̄, V̄, K̄ᵣ (routing keys) per document
2. **Online routing**: Query projected to Qᵣ, cosine similarity against K̄ᵣ, top-k selection
3. **Sparse generation**: Selected K̄/V̄ concatenated with local KV, standard attention over sparse context

### Key Techniques

**Document-wise RoPE**: Each document resets position indices from 0, preventing train-short/infer-long position drift. Global RoPE on the active context preserves causal ordering (background → query → generation).

**KV cache compression**: Token-wise mean pooling compresses document KV states. Routing keys (K̄ᵣ) are GPU-resident; content K̄/V̄ stays in host DRAM and is fetched asynchronously on selection.

**Memory Interleave**: Multi-round "generative retrieval → context expansion → generation" for multi-hop reasoning across scattered memory segments.

**Memory Parallel**: Routing keys sharded across GPUs; query broadcast → local scoring → global reduce. Enables 100M-token deployment on 2 GPUs.

---

## Relevance to Somnigraph

**None.** MSA operates at the model architecture level (transformer attention patterns, KV cache management, position encoding). Somnigraph operates at the application level (discrete memory storage, retrieval ranking, feedback loops). The problems are related (long-term memory for AI) but the layers are completely disjoint. No techniques are transferable in either direction.

The paper is cataloged for completeness as part of the memory systems landscape. The scaling results (16K→100M with <9% degradation) and the comparison showing a 4B model with MSA beating RAG pipelines using 235B generators are noteworthy results for the field, even though they don't inform Somnigraph's design.
