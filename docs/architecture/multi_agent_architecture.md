# Multi-Agent Architecture

## Overview

The Artificial Cognitive Brain is inherently a multi-agent system. Each cognitive module (Perception, Memory, Reasoning, Learning, Planning) is realised as one or more autonomous agents that communicate through the Global Workspace. This section details the agent hierarchy, inter-agent protocols, task delegation strategies, and lifecycle management that enable cooperative cognition at scale.

## Agent Hierarchy

```
                    ┌──────────────────┐
                    │  Orchestrator     │  ◄── Root agent; manages lifecycle & system goals
                    └────────┬─────────┘
                             │
            ┌────────────────┼────────────────┐
            │                │                │
     ┌──────▼──────┐  ┌─────▼──────┐  ┌──────▼──────┐
     │ Cognitive   │  │ Adaptive   │  │  Executive   │  ◄── Mid-level coordinators
     │ Cluster     │  │ Cluster    │  │  Cluster     │
     └──┬──┬──┬───┘  └──┬──┬──┬───┘  └──┬──┬──┬───┘
        │  │  │         │  │  │         │  │  │
       P  M  R        L  F  E       Pl A  M        ◄── Leaf agents
                                              (perception, memory,
                                               reasoning, learning,
                                               feedback, evaluation,
                                               planning, action,
                                               monitoring)
```

### Agent Roles

| Agent Type      | Role                                                                 |
|-----------------|----------------------------------------------------------------------|
| **Orchestrator**| Bootstraps the system, allocates resources, sets global goals, and handles shutdown gracefully. |
| **Cognitive Agents** | Perception, Memory, and Reasoning agents form the core cognitive loop. |
| **Adaptive Agents**  | Learning, Feedback, and Evaluation agents handle continual improvement and self-assessment. |
| **Executive Agents** | Planning, Action, and Monitoring agents translate cognition into behaviour. |

## Communication Protocols

All inter-agent communication uses the **Global Workspace Message Protocol (GWMP)**, a JSON-based schema running over Apache Kafka.

### Message Types

| Message Type    | Direction              | Payload                                                 |
|-----------------|-------------------------|---------------------------------------------------------|
| `PERCEPT_CHUNK` | Perception → Workspace  | Embedding, modality tag, confidence, timestamp.         |
| `MEMORY_QUERY`  | Reasoning → Memory      | Query embedding, top-k, filters.                        |
| `MEMORY_RESULT` | Memory → Reasoning      | Ranked memory chunks with activation scores.            |
| `REASON_OUTPUT` | Reasoning → Workspace   | Conclusion, confidence, reasoning trace, hypotheses.    |
| `PLAN_PROPOSAL` | Planning → Workspace     | Action sequence, expected utility, risk assessment.     |
| `ACTION_DISPATCH`| Planning → Action       | Serialised API call or tool invocation.                 |
| `FEEDBACK`      | Environment → Workspace  | Reward signal, task completion status, error report.    |
| `HEARTBEAT`     | Any → Orchestrator       | Health status, resource utilisation, queue depth.      |

### Communication Patterns

- **Publish-Subscribe**: The Global Workspace acts as a topic-based pub/sub bus. Agents subscribe to message types relevant to their function.
- **Request-Reply**: For synchronous queries (e.g., memory retrieval), agents use Kafka's request-reply pattern with configurable timeouts.
- **Broadcast**: Critical system events (shutdown, configuration change) are broadcast to all agents via a dedicated control topic.

## Task Delegation

The Orchestrator and Planning agents collaboratively decompose high-level goals into agent-specific tasks:

1. **Goal Receipt**: The Orchestrator receives a goal from the user or an external system.
2. **Capability Discovery**: The Orchestrator queries the agent registry to identify which agents possess the required capabilities.
3. **Task Graph Construction**: A directed acyclic graph (DAG) of sub-tasks is constructed, specifying dependencies and data flow.
4. **Task Assignment**: Sub-tasks are assigned to agents via the Global Workspace. Agents acknowledge receipt and report completion.
5. **Aggregation**: The Planning agent aggregates sub-task results and synthesises a final output.

## Cooperative Reasoning

When a single reasoning pass is insufficient, agents engage in **cooperative reasoning**:

- **Debate Protocol**: Two or more Reasoning agents propose competing hypotheses. Each critiques the other's argument. A judge agent (Evaluation) selects the strongest position.
- **Divide-and-Conquer**: Complex reasoning tasks are split into independent sub-problems, each handled by a separate Reasoning agent, with results merged by the Memory agent.
- **Committee of Experts**: Multiple Reasoning agents with different system prompts or temperature settings vote on the final answer, improving robustness.

## Conflict Resolution

Conflicts between agents (e.g., contradictory conclusions, resource contention) are resolved through a priority-based arbitration scheme:

| Conflict Type           | Resolution Strategy                                             |
|-------------------------|------------------------------------------------------------------|
| **Contradictory Facts** | Evaluation agent scores each claim by evidence strength; higher-scored claim wins. |
| **Resource Contention**  | Orchestrator applies weighted priority queues (critical > normal > background). |
| **Plan Deadlock**       | Planning agent escalates to Orchestrator, which may replan with relaxed constraints. |
| **Stale State**         | Workspace versioning ensures agents operate on the latest snapshot; stale messages are discarded. |

## Agent Lifecycle

```
  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
  │ Inactive │───►│ Starting │───►│  Active  │───►│Draining  │───►│ Stopped  │
  └──────────┘    └──────────┘    └────┬─────┘    └──────────┘    └──────────┘
                                      │
                                      ▼
                                ┌──────────┐
                                │  Error   │───► Orchestrator initiates restart
                                └──────────┘
```

- **Inactive**: Agent container exists but is not consuming messages.
- **Starting**: Agent performs health checks, loads models, registers with the agent registry.
- **Active**: Agent processes messages, publishes results, and sends heartbeats.
- **Draining**: On shutdown signal, agent finishes in-flight tasks, publishes final state, and deregisters.
- **Error**: On unrecoverable failure, the Orchestrator triggers a restart with exponential backoff.

The Orchestrator monitors agent health via heartbeat messages. An agent missing three consecutive heartbeats (default: 30 seconds) is marked as degraded and its tasks are redistributed to backup instances.
