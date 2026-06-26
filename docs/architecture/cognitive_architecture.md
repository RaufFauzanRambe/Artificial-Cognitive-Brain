# Cognitive Architecture

## Theoretical Foundations

The Artificial Cognitive Brain draws on two principal theories from cognitive science:

### Global Workspace Theory (GWT)

GWT, proposed by Bernard Baars, posits that consciousness arises from a **global broadcast** mechanism. Multiple specialised, unconscious cognitive processors compete for access to a shared workspace. Winning coalitions broadcast their information across the entire system, temporarily making it available to all modules. In ACB, the **Global Workspace Bus** realises this broadcast: modules publish structured messages (chunks) that are simultaneously visible to every subscriber, enabling coherent, system-wide "attentional" episodes.

### ACT-R Inspired Design

ACT-R (Adaptive Control of Thought — Rational) provides a production-rule framework where cognition emerges from the interaction of declarative memory chunks and procedural production rules. ACB adopts ACT-R's separation of:

- **Declarative Memory** — Factual knowledge stored as key-value chunks with activation strengths.
- **Procedural Memory** — Condition-action rules encoded as learned policy modules.
- **Goal Stack** — A hierarchical stack of current intentions that guides production selection.

While ACB does not implement symbolic production rules directly, it mirrors ACT-R's architectural separation through neural module boundaries and a goal-directed planning system.

## The Cognitive Cycle

Every interaction with the environment follows a repeatable cognitive cycle:

```
   ┌──────────────────────────────────────────────────┐
   │              Cognitive Cycle                     │
   │                                                  │
   │   Sense ──► Perceive ──► Reason ──► Plan ──► Act│
   │     ▲                                      │    │
   │     └──────────────────────────────────────┘    │
   │              (environment feedback)             │
   └──────────────────────────────────────────────────┘
```

### 1. Sense

Raw signals arrive from sensors (text input, image frames, API payloads, audio streams). The Sense stage buffers and time-stamps incoming data without interpretation.

### 2. Perceive

The Perception module encodes raw signals into semantic embeddings and identifies salient features. Relevant perceptual chunks are published to the Global Workspace.

### 3. Reason

The Reasoning module consumes workspace contents, retrieves supporting memories, constructs inference chains, and generates hypotheses. Reasoned conclusions are broadcast back to the workspace.

### 4. Plan

The Planning module evaluates hypotheses against current goals, constructs multi-step action sequences, and performs risk/benefit analysis. Chosen plans enter the workspace as intentions.

### 5. Act

The Action Executor translates plans into concrete API calls, tool invocations, or motor commands. Results feed back into the Sense stage, closing the loop.

## Module Interactions

```
Perception ──────► Workspace ◄────── Reasoning
    │                   │                  ▲
    │                   ▼                  │
    └──────────► Memory ◄──────────────────┘
                        ▲
                        │
                   Learning ◄──── Planning
                        │
                        ▼
                   Action Executor ────► Environment
```

- **Perception → Workspace**: Publishes perceptual chunks.
- **Workspace → Reasoning**: Broadcasts current coalition for inferential analysis.
- **Reasoning → Memory**: Queries for relevant declarative / episodic traces.
- **Memory → Reasoning**: Returns ranked retrieval results.
- **Reasoning → Workspace**: Publishes conclusions and confidence scores.
- **Workspace → Planning**: Exposes reasoned state for goal-driven planning.
- **Planning → Memory**: Stores generated plans for future replay.
- **Learning → Memory**: Consolidates experiences during rest periods.
- **Planning → Action**: Dispatches executable steps.

## Attention Mechanisms

Attention in ACB operates at three levels:

| Level           | Scope               | Mechanism                                                        |
|-----------------|---------------------|------------------------------------------------------------------|
| **Module-Level**| Within a single module | Softmax-based self-attention over internal representations.      |
| **Workspace-Level** | Cross-module    | Competitive activation scoring; highest-scoring chunks occupy the broadcast slot. |
| **Goal-Level**  | Planning horizon   | Gated attention that biases reasoning and retrieval toward goal-relevant content. |

The attention system ensures that computational resources are allocated efficiently, preventing combinatorial explosion in large state spaces while maintaining the flexibility required for open-ended reasoning.
