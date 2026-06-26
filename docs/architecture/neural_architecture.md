# Neural Architecture

## Overview

The Artificial Cognitive Brain employs a heterogeneous ensemble of neural architectures, each selected to match the computational demands of its host cognitive module. Rather than forcing a single model type across all functions, ACB leverages domain-specific architectures — transformers for perception, memory-augmented networks for recall, graph neural networks for relational reasoning, and attention-based planners for action selection. This section details each architecture, its role, and key design trade-offs.

## Transformer Perception

### Architecture

The perception module uses a **multi-modal transformer encoder** inspired by architectures like Flamingo and LLaVA. It consists of:

- **Text Encoder**: A 12-layer BERT-scale transformer (110 M parameters) that produces contextual token embeddings for textual input.
- **Vision Encoder**: A Vision Transformer (ViT-B/16) pre-trained on large-scale image-text corpora, producing patch-level visual tokens.
- **Cross-Attention Fusion Layer**: A set of cross-attention blocks that align visual and textual token streams, producing a unified multi-modal representation.

### Design Choices

| Decision                  | Rationale                                                                  |
|---------------------------|----------------------------------------------------------------------------|
| Separate encoders         | Allows independent pre-training on modality-specific corpora, improving zero-shot transfer. |
| Cross-attention fusion    | More parameter-efficient than joint training; late fusion avoids interference between modalities. |
| 110 M parameter budget    | Balances expressiveness with inference latency; perception must complete in ≤ 50 ms. |

## Memory-Augmented Neural Networks

### Architecture

Memory retrieval and management is powered by **Memory-Augmented Neural Networks (MANNs)**, combining:

- **Differentiable Neural Computer (DNC)**: An external memory matrix with read/write heads controlled by an LSTM controller. Used for working-memory-style scratchpad operations during complex reasoning.
- **Approximate Nearest Neighbour (ANN) Index**: A Qdrant / Milvus-hosted HNSW index storing long-term memory embeddings. Serves fast (sub-5 ms) semantic retrieval at scale (millions of vectors).

### Interaction Pattern

```
Query ──► ANN Index ──► Top-K Embeddings ──► DNC Read Heads ──► Enriched Representation
```

The ANN index provides coarse retrieval, while the DNC performs fine-grained relational reasoning over the retrieved set, enabling compositional memory operations not possible with vector search alone.

## Graph Neural Networks

### Architecture

Relational and causal reasoning benefits from **Graph Neural Networks (GNNs)**:

- **Relational Graph Convolutional Network (R-GCN)**: Models entity-relationship graphs where entities are knowledge-base concepts and edges are typed relations (e.g., *causes*, *inhibits*, *requires*).
- **Temporal Graph Network (TGN)**: Extends the relational graph with temporal edge attributes, enabling reasoning about event sequences and temporal causality.

### Use Cases

- **Knowledge Base Reasoning**: Answering multi-hop questions by traversing the relational graph.
- **Causal Discovery**: Constructing and querying directed acyclic graphs (DAGs) to support planning under causal models.
- **Explainability**: GNN attention weights provide interpretable paths showing which entities and relations contributed to a conclusion.

## Attention Mechanisms

ACB deploys attention at multiple levels of abstraction:

| Mechanism                     | Location                 | Purpose                                                            |
|-------------------------------|--------------------------|--------------------------------------------------------------------|
| **Multi-Head Self-Attention** | Perception Transformer   | Captures long-range dependencies within and across modalities.     |
| **Cross-Modal Attention**     | Fusion Layer             | Aligns features from different input streams (text ↔ image).       |
| **Retrieval Attention**       | Memory Module            | Weights retrieved memories by relevance to the current query.     |
| **Graph Attention**          | GNN Layers              | Learns edge-type-specific importance weights for relational reasoning. |
| **Workspace Attention**       | Global Workspace         | Competitive softmax over broadcast chunks; implements the attentional bottleneck. |

## Architecture Trade-Offs

| Trade-Off                          | Option A                          | Option B                          | ACB Choice | Rationale                                                |
|------------------------------------|------------------------------------|------------------------------------|------------|----------------------------------------------------------|
| **Perception: Joint vs. Separate** | Joint ViT (single model)          | Separate encoders + cross-attention | B         | Better modality transfer; lower training cost per encoder. |
| **Memory: ANN vs. Full-Attention** | Pure ANN retrieval                | DNC + ANN hybrid                   | B         | Hybrid enables compositional reasoning over retrieved sets. |
| **Reasoning: LLM vs. GNN**         | Large autoregressive LLM          | GNN over knowledge graph           | Both      | LLM for linguistic reasoning, GNN for relational/causal reasoning. |
| **Planning: MCTS vs. Learned**     | Search-based (MCTS)               | Learned policy network             | Hybrid    | MCTS for reliability, learned policy for speed in familiar contexts. |

## Parameter & Compute Budget

| Module        | Architecture              | Parameters | GPU Memory (FP16) | Typical Batch Size |
|---------------|---------------------------|------------|--------------------|--------------------|
| Perception    | Multi-modal Transformer   | 300 M      | 2.4 GB             | 8                  |
| Memory (DNC)  | LSTM + External Memory    | 50 M       | 0.8 GB             | 16                 |
| Reasoning     | LLM (7 B) + R-GCN         | 7.1 B      | 14 GB              | 1                  |
| Planning      | Policy Network (500 M)    | 500 M      | 1.6 GB             | 32                 |
| Learning      | Fine-tuning Head          | 200 M      | 0.6 GB             | 8                  |
| **Total**     |                           | ~8.2 B     | ~19.4 GB           | —                  |

The total footprint fits comfortably on a single NVIDIA A100 (40 GB), enabling cost-effective deployment while leaving headroom for batching and caching.
