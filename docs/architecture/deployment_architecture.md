# Deployment Architecture

## Overview

The Artificial Cognitive Brain is designed for production-grade deployment on container-orchestrated infrastructure. This section covers the full deployment stack — from container orchestration and GPU management to model serving, observability, CI/CD, and high-availability strategies.

## Container Orchestration

ACB is deployed as a collection of Kubernetes workloads:

| Workload Type | Components                                          | Purpose                                       |
|---------------|-----------------------------------------------------|-----------------------------------------------|
| **Deployments** | Perception, Memory, Reasoning, Learning, Planning, Action, Orchestrator agents | Long-running stateless agent services.         |
| **StatefulSets** | Qdrant / Milvus (vector store), Kafka (message bus), Redis (cache) | Stateful services requiring stable network identity and persistent volumes. |
| **Jobs / CronJobs** | Learning consolidation, model fine-tuning, data ingestion | Batch workloads triggered on schedule or event. |
| **DaemonSets** | Node-level GPU monitoring, log collection agents    | Infrastructure services on every node.        |

Each agent is packaged as a Docker image built from a multi-stage Dockerfile:

```
Stage 1 (build):  Install Python dependencies, compile extensions
Stage 2 (runtime): Copy artifacts, set non-root user, expose health port
```

### Helm Chart Structure

```
charts/artificial-cognitive-brain/
├── Chart.yaml
├── values.yaml              # Default configuration
├── values-production.yaml   # Production overrides
├── values-staging.yaml      # Staging overrides
├── templates/
│   ├── agents/              # One deployment template per agent
│   ├── infrastructure/      # Kafka, Qdrant, Redis StatefulSets
│   ├── configmaps/          # Environment-specific configuration
│   ├── secrets/             # API keys, model credentials (sealed)
│   └── monitoring/          # Prometheus rules, Grafana dashboards
```

## GPU Management

### Resource Allocation

ACB uses Kubernetes' **device plugin for NVIDIA GPUs** to allocate GPU resources per-agent:

```yaml
resources:
  limits:
    nvidia.com/gpu: 1        # One GPU for Reasoning agent
    memory: "32Gi"
  requests:
    cpu: "4"
    memory: "16Gi"
```

### GPU Sharing Strategy

| Agent        | GPU Requirement | Sharing Mode           | Rationale                                         |
|---------------|-----------------|------------------------|----------------------------------------------------|
| Perception    | 1 GPU           | Exclusive (MIG slice)  | Low latency required; interference is unacceptable. |
| Reasoning     | 1 GPU           | Exclusive              | Large model; needs full memory bandwidth.          |
| Learning      | 1 GPU           | Time-sliced (MPS)      | Batch fine-tuning tolerates shared access.         |
| Planning      | None (CPU)      | —                      | MCTS rollouts are CPU-bound.                       |
| Memory        | None (CPU)      | —                      | Vector search runs on CPU-optimised HNSW.         |

### Multi-Instance GPU (MIG)

On A100 GPUs, MIG partitions are used to isolate Perception and Learning agents on the same physical GPU, reducing hardware costs while maintaining isolation.

## Model Serving

Models are served via **NVIDIA Triton Inference Server**, providing:

- **Dynamic Batching**: Automatic batching of concurrent inference requests to maximise GPU throughput.
- **Model Versioning**: Multiple model versions coexist; traffic is gradually shifted via canary deployments.
- **Multi-Framework Support**: PyTorch, ONNX Runtime, and TensorRT backends for optimal inference speed.
- **gRPC & HTTP**: Both protocols are exposed; gRPC is used for internal inter-agent calls, HTTP for external API access.

```
Agent ──gRPC──► Triton Server ──► GPU (model A)
                                    GPU (model B)
```

Model weights are stored in a persistent volume backed by cloud object storage (S3 / GCS) and pulled at startup. Hot model swapping (without server restart) is supported via Triton's model repository polling.

## Monitoring & Observability

### Metrics Pipeline

```
Agents ──► OpenTelemetry Collector ──► Prometheus ──► Grafana
                │
                ▼
           Jaeger (traces)
```

### Key Metrics

| Metric Category    | Examples                                                                 |
|--------------------|--------------------------------------------------------------------------|
| **Agent Health**   | Heartbeat interval, message processing rate, error rate, queue depth.    |
| **Inference**      | Latency (p50, p95, p99), throughput (req/s), batch utilisation.         |
| **Memory**         | Vector store size, retrieval latency, cache hit rate.                    |
| **GPU**            | Utilisation %, memory used, temperature, power draw.                     |
| **System**         | Kafka consumer lag, pod restarts, CPU / memory pressure.                |

### Alerting Rules

- **Critical**: Agent down > 60 s, GPU OOM, Kafka partition offline.
- **Warning**: p99 latency > 2× SLO, memory utilisation > 85%, consumer lag > 1000 messages.
- **Info**: Model version rolled back, learning job completed.

### Tracing

Every cognitive cycle is assigned a **trace ID** that propagates through all agents via Kafka message headers. This enables end-to-end request tracing in Jaeger, making it possible to identify latency bottlenecks across the full cognitive pipeline.

## CI/CD Pipeline

```
Push to main ──► GitHub Actions
                    ├── Lint & Type Check (ruff, mypy)
                    ├── Unit Tests (pytest, coverage ≥ 90%)
                    ├── Integration Tests (kind cluster, agent-to-agent)
                    ├── Build Docker Images (multi-arch)
                    ├── Push to Container Registry
                    ├── Run Helm Lint & Template Validation
                    └── Deploy to Staging (auto)
                          │
                          ▼
                    Manual Approval ──► Deploy to Production (canary → full rollout)
```

### Infrastructure as Code

- **Kubernetes Manifests**: Generated from Helm charts; versioned in Git.
- **Secrets**: Managed with HashiCorp Vault or Sealed Secrets; never committed to source control.
- **Environment Promotion**: Staging and production environments share the same Helm chart with different `values-*.yaml` overrides.

## High Availability

### Redundancy

| Component    | HA Strategy                                           |
|--------------|-------------------------------------------------------|
| **Agents**   | Horizontal Pod Autoscaler (HPA) scales replicas based on queue depth. Minimum 2 replicas per critical agent. |
| **Kafka**    | 3-broker cluster with replication factor 3.           |
| **Vector Store** | Qdrant cluster with 3 replicas and automatic shard rebalancing. |
| **Redis**    | Sentinel-managed master-replica topology.             |

### Failure Scenarios

| Failure                         | Recovery                                                    |
|----------------------------------|--------------------------------------------------------------|
| Agent pod crash                 | Kubernetes restarts pod; Orchestrator reassigns in-flight tasks. |
| GPU node failure                | Pods are rescheduled to healthy GPU nodes via pod anti-affinity. |
| Kafka broker down               | Remaining brokers continue serving; ISR catches up on recovery. |
| Data centre outage              | Multi-zone deployment ensures at least one zone remains operational. |

### Backup & Disaster Recovery

- **Vector Store**: Daily snapshots to object storage; point-in-time recovery supported.
- **Kafka**: Topic data retained for 7 days; mirror-maker replicates to a DR cluster in a separate region.
- **Model Registry**: All model versions are immutable in object storage; rollback is a single config change.

## Deployment Topology (Production)

```
Zone A                    Zone B
┌─────────────────┐      ┌─────────────────┐
│ Ingress (NLB)   │◄────►│ Ingress (NLB)   │
│                 │      │                 │
│ ┌─────────────┐ │      │ ┌─────────────┐ │
│ │ Agent Pods  │ │      │ │ Agent Pods  │ │
│ │ (GPU + CPU) │ │      │ │ (GPU + CPU) │ │
│ └─────────────┘ │      │ └─────────────┘ │
│                 │      │                 │
│ ┌─────────────┐ │      │ ┌─────────────┐ │
│ │ Kafka (3/3) │ │◄────►│ Kafka (3/3) │ │
│ │ Qdrant (2/3)│ │      │ Qdrant (1/3) │ │
│ │ Redis       │ │      │ Redis        │ │
│ └─────────────┘ │      │ └─────────────┘ │
└─────────────────┘      └─────────────────┘
         │                        │
         └────────┬───────────────┘
                  ▼
          Object Storage (S3/GCS)
          Model Registry & Backups
```
