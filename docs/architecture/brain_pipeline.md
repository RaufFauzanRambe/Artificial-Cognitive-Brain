# Processing Pipeline

## Pipeline Overview

The Artificial Cognitive Brain processes every user interaction through a deterministic, staged pipeline. Each stage transforms, enriches, or routes data before passing it downstream. The pipeline is designed for **sub-second end-to-end latency** in typical conversational scenarios while supporting long-running deliberative reasoning when complexity demands it.

```
┌──────┐   ┌───────────┐   ┌─────────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌───────────┐
│Input │──►│  Pre-      │──►│ Perception  │──►│ Memory   │──►│Reasoning │──►│ Learning │──►│ Planning  │──► Action
│Buffer│   │ Processing │   │ Engine      │   │ Retrieval│   │  Chain    │   │  Update  │   │  Engine   │   │Executor
└──────┘   └───────────┘   └─────────────┘   └──────────┘   └──────────┘   └──────────┘   └──────────┘   └───────────┘
```

## Stage 1 — Input Processing

The input processing layer is the system's front door. It is responsible for:

- **Normalisation**: Lowercasing, tokenisation, and schema validation of incoming payloads.
- **Modality Routing**: Classifying whether the input is text, image, audio, or structured data, and dispatching to the appropriate perception encoder.
- **Session Binding**: Associating the input with a user session and conversation history from working memory.
- **Rate Limiting & Deduplication**: Preventing duplicate processing of repeated or malicious payloads.

Outputs from this stage are **raw tensors** or **token sequences** tagged with session metadata.

## Stage 2 — Perception

The perception engine converts raw input into semantically rich representations:

- **Text Perception**: A transformer-based encoder (e.g., a fine-tuned BERT or similar architecture) produces contextual embeddings and named-entity annotations.
- **Vision Perception**: A vision-language model extracts object detections, spatial relationships, and scene captions from image inputs.
- **Fusion**: Multi-modal inputs undergo late fusion, concatenating modality-specific embeddings before a shared projection layer.

The perception stage outputs **perceptual chunks** — structured objects containing the embedding vector, confidence score, source modality, and timestamp.

## Stage 3 — Memory Retrieval

Before reasoning begins, the system enriches the workspace with relevant memories:

1. **Query Formulation**: The perceptual chunk is used to construct a retrieval query, optionally expanded via query rewriting using a small LLM.
2. **Vector Search**: Approximate nearest-neighbour (ANN) search is performed over the long-term memory index (Qdrant / Milvus) to find the top-k semantically similar declarative and episodic memories.
3. **Activation Scoring**: Retrieved memories are assigned activation scores based on recency, frequency, and semantic relevance (analogous to ACT-R's base-level activation).
4. **Workspace Injection**: The top-N activated memories are injected into the Global Workspace alongside the current perceptual chunk.

## Stage 4 — Reasoning Chain

The reasoning module operates on the full workspace state:

- **Chain-of-Thought (CoT)**: A step-by-step reasoning trace is generated, with each intermediate conclusion added back to the workspace for verification.
- **Self-Consistency**: Multiple reasoning paths are sampled in parallel; the most consistent conclusion across paths is selected.
- **Abduction & Hypothesis Generation**: When evidence is incomplete, the module generates plausible hypotheses ranked by explanatory power.
- **Causal Analysis**: For planning-critical scenarios, a lightweight causal graph is constructed to identify confounders and estimate intervention effects.

Output: A **reasoned conclusion chunk** containing the final answer, confidence interval, reasoning trace (for auditability), and any remaining hypotheses.

## Stage 5 — Learning Update

Every completed cycle is a learning opportunity:

- **Experience Logging**: The full cycle (input → perception → memory → reasoning → output) is stored as an episodic trace in the experience replay buffer.
- **Declarative Consolidation**: High-confidence facts extracted from reasoning are written to the long-term declarative store.
- **Policy Gradient Update**: If the cycle resulted in environmental feedback (reward signal), the planning module's policy network is updated via proximal policy optimisation (PPO).
- **Synaptic Consolidation**: Periodically (during idle periods), the learning module replays buffered experiences and performs model fine-tuning to prevent catastrophic forgetting.

## Stage 6 — Planning & Action Execution

The planning engine converts conclusions into concrete actions:

- **Goal Decomposition**: High-level goals are broken into sub-goals using hierarchical task networks (HTNs).
- **Monte Carlo Tree Search (MCTS)**: Each sub-goal is evaluated through simulated rollouts to estimate expected utility.
- **Execution Dispatch**: Chosen actions are serialised into API calls or tool invocations and dispatched to the Action Executor.
- **Feedback Loop**: Action results return to the Input Buffer, initiating a new cognitive cycle.

## Data Flow & Latency Optimisation

| Optimisation Strategy               | Description                                                                 |
|-------------------------------------|-----------------------------------------------------------------------------|
| **Streaming Perception**            | Perceptual embeddings are streamed to memory retrieval before full encoding completes. |
| **Speculative Reasoning**           | The reasoning module begins inference with partial memory results, revising once full retrieval completes. |
| **Async Learning**                  | Learning updates are decoupled from the critical path via a separate worker pool. |
| **Batched MCTS**                    | Multiple planning rollouts execute in parallel on GPU for sub-100ms plan generation. |
| **Cached Workspace Snapshots**      | Frequently accessed workspace states are cached in Redis to avoid recomputation. |

Target latencies: **Perception** ≤ 50 ms, **Memory Retrieval** ≤ 20 ms, **Reasoning** ≤ 200 ms, **Planning** ≤ 100 ms, **End-to-End** ≤ 500 ms (conversational path).
