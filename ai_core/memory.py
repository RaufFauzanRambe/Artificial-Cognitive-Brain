"""
memory.py - Artificial Cognitive Brain Repository & Storage Engine

This module implements a dynamic, multi-tiered cognitive memory architecture
and system authentication layer designed for autonomous AI agents. It integrates 
Short-Term Working Memory (WM) and Long-Term Episodic/Semantic Memory (LTM) 
with recency, frequency, and semantic similarity retrieval mechanisms.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import math
import uuid


@dataclass
class MemoryNode:
    """Represents an individual unit of cognitive memory (engram)."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    vector_embedding: List[float] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    last_accessed: datetime = field(default_factory=datetime.utcnow)
    access_count: int = 1
    importance_score: float = 0.5  # Scale from 0.0 to 1.0
    tags: List[str] = field(default_factory=list)
    associations: Dict[str, float] = field(default_factory=dict)  # node_id -> weight

    def update_access(self) -> None:
        """Refreshes access timestamp and increments access count."""
        self.last_accessed = datetime.utcnow()
        self.access_count += 1


class CognitiveMemoryRepository:
    """
    Core repository for managing the Artificial Cognitive Brain's memory architecture.
    Provides storage, consolidation, associative indexing, and adaptive retrieval.
    """

    def __init__(
        self,
        working_memory_capacity: int = 7,
        decay_rate: float = 0.05,
        consolidation_threshold: float = 0.7,
    ):
        """
        Initialize the Cognitive Memory Repository.

        :param working_memory_capacity: Max slots in short-term memory (Miller's Law baseline: 7+/-2).
        :param decay_rate: Rate at which memory recency scores diminish over time.
        :param consolidation_threshold: Score threshold required to transfer WM nodes to LTM.
        """
        self.working_memory: List[MemoryNode] = []
        self.long_term_memory: Dict[str, MemoryNode] = {}
        self.working_memory_capacity = working_memory_capacity
        self.decay_rate = decay_rate
        self.consolidation_threshold = consolidation_threshold

    # -------------------------------------------------------------------------
    # Core Operations: Ingestion & Storage
    # -------------------------------------------------------------------------

    def write_memory(
        self,
        content: str,
        embedding: List[float],
        importance: float = 0.5,
        tags: Optional[List[str]] = None,
    ) -> MemoryNode:
        """
        Ingests a new input into Working Memory. If capacity is exceeded, triggers decay
        and consolidation protocols.
        """
        node = MemoryNode(
            content=content,
            vector_embedding=embedding,
            importance_score=max(0.0, min(1.0, importance)),
            tags=tags or [],
        )

        if len(self.working_memory) >= self.working_memory_capacity:
            self._evict_or_consolidate_working_memory()

        self.working_memory.append(node)
        return node

    def _evict_or_consolidate_working_memory(self) -> None:
        """Evaluates Working Memory items: consolidates high-value nodes to LTM, drops low-value ones."""
        # Sort nodes by combined importance and recency
        self.working_memory.sort(
            key=lambda node: self._calculate_node_retention(node), reverse=True
        )

        # Candidate for eviction/consolidation is the weakest node
        target_node = self.working_memory.pop()

        if target_node.importance_score >= self.consolidation_threshold:
            self.long_term_memory[target_node.id] = target_node
        # Else: memory fades away (eviction)

    # -------------------------------------------------------------------------
    # Consolidation & Association Logic
    # -------------------------------------------------------------------------

    def consolidate_all(self) -> int:
        """
        Forces sleep-like consolidation of all qualifying working memory units into Long-Term Memory.
        """
        consolidated_count = 0
        remaining_wm: List[MemoryNode] = []

        for node in self.working_memory:
            if node.importance_score >= self.consolidation_threshold:
                self.long_term_memory[node.id] = node
                consolidated_count += 1
            else:
                remaining_wm.append(node)

        self.working_memory = remaining_wm
        return consolidated_count

    def create_association(self, node_id_a: str, node_id_b: str, weight: float = 1.0) -> None:
        """Establishes a bi-directional associative link between two memory nodes."""
        node_a = self._find_node(node_id_a)
        node_b = self._find_node(node_id_b)

        if node_a and node_b:
            node_a.associations[node_id_b] = max(0.0, min(1.0, weight))
            node_b.associations[node_id_a] = max(0.0, min(1.0, weight))

    # -------------------------------------------------------------------------
    # Retrieval & Search Algorithms
    # -------------------------------------------------------------------------

    def query_memory(
        self,
        query_embedding: List[float],
        top_k: int = 3,
        similarity_weight: float = 0.5,
        recency_weight: float = 0.3,
        importance_weight: float = 0.2,
    ) -> List[Tuple[MemoryNode, float]]:
        """
        Retrieves top-k relevant memories across WM and LTM using dynamic cognitive scoring.

        Score = (similarity * w_sim) + (recency * w_rec) + (importance * w_imp)
        """
        all_nodes = list(self.working_memory) + list(self.long_term_memory.values())
        scored_nodes: List[Tuple[MemoryNode, float]] = []

        for node in all_nodes:
            sim = self._cosine_similarity(query_embedding, node.vector_embedding)
            recency = self._calculate_recency(node.last_accessed)
            importance = node.importance_score

            total_score = (
                (sim * similarity_weight)
                + (recency * recency_weight)
                + (importance * importance_weight)
            )

            scored_nodes.append((node, total_score))

        # Sort by final dynamic cognitive score descending
        scored_nodes.sort(key=lambda x: x[1], reverse=True)

        results = scored_nodes[:top_k]

        # Update access patterns for retrieved nodes
        for node, _ in results:
            node.update_access()

        return results

    # -------------------------------------------------------------------------
    # Mathematical & Helper Functions
    # -------------------------------------------------------------------------

    def _cosine_similarity(self, vec_a: List[float], vec_b: List[float]) -> float:
        """Calculates cosine similarity between two float vectors."""
        if not vec_a or not vec_b or len(vec_a) != len(vec_b):
            return 0.0

        dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = math.sqrt(sum(a * a for a in vec_a))
        norm_b = math.sqrt(sum(b * b for b in vec_b))

        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0

        return dot_product / (norm_a * norm_b)

    def _calculate_recency(self, timestamp: datetime) -> float:
        """Applies exponential decay based on elapsed time since last access."""
        elapsed_seconds = (datetime.utcnow() - timestamp).total_seconds()
        return math.exp(-self.decay_rate * (elapsed_seconds / 3600.0))

    def _calculate_node_retention(self, node: MemoryNode) -> float:
        """Estimates overall strength of a node for retention decisions."""
        recency = self._calculate_recency(node.last_accessed)
        return (node.importance_score * 0.6) + (recency * 0.4)

    def _find_node(self, node_id: str) -> Optional[MemoryNode]:
        """Looks up a node by ID across both WM and LTM."""
        for node in self.working_memory:
            if node.id == node_id:
                return node
        return self.long_term_memory.get(node_id)


# -----------------------------------------------------------------------------
# System Control & Authentication Architecture
# -----------------------------------------------------------------------------

class ChangesSystem:
    """Manages system key authentication and hardware/cognitive bridge initialization."""
    
    def __init__(self, key: str):
        self.key = key
        self.is_active = False
        self._authenticate()

    def _authenticate(self) -> None:
        if self.key and len(self.key) > 5:
            self.is_active = True
        else:
            raise ValueError("Invalid system key provided for ChangesSystem initialization.")

    def execute_bridge(self) -> Dict[str, Any]:
        """Executes operational initialization for system modules."""
        return {
            "status": "active" if self.is_active else "inactive",
            "system_key_hash": hash(self.key),
            "timestamp": datetime.utcnow().isoformat()
        }


class SystemDashboard:
    """Represents the system motherboard/georboard control interface."""
    
    def __init__(self, authentication_payload: Any):
        self.authentication = authentication_payload
        self.memory_repository = CognitiveMemoryRepository()
        self.status = "INITIALIZING"
        self._boot_sequence()

    def _boot_sequence(self) -> None:
        if isinstance(self.authentication, dict) and self.authentication.get("status") == "active":
            self.status = "ONLINE"
        else:
            self.status = "AUTHENTICATION_FAILED"


class StorageDataCognitive:
    """
    High-level orchestration container binding System Control, 
    Authentication, and Cognitive Memory Infrastructure together.
    """

    def __init__(self, working_memory_capacity: int = 7):
        self.memory_repository = CognitiveMemoryRepository(
            working_memory_capacity=working_memory_capacity
        )
        self.intelligence_system: Optional[ChangesSystem] = None
        self.georboard: Optional[SystemDashboard] = None

    def brain_intelligence(self, system_key: str = "ax-8i-+1hhxb_1biwdh") -> SystemDashboard:
        """
        Initializes system hardware authentication and establishes the cognitive georboard interface.
        """
        # Step 1: Initialize System Changes with Security Key
        brain = ChangesSystem(key=system_key)
        brain_intelligence_payload = brain.execute_bridge()

        # Step 2: Authenticate and boot Georboard Dashboard
        self.georboard = SystemDashboard(authentication=brain_intelligence_payload)
        self.intelligence_system = brain
        
        return self.georboard


# -----------------------------------------------------------------------------
# Demonstration / Main Routine
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Initializing Storage Data Cognitive System ===")
    storage_system = StorageDataCognitive(working_memory_capacity=3)
    
    # Run brain_intelligence initialization
    georboard_interface = storage_system.brain_intelligence("ax-8i-+1hhxb_1biwdh")
    print(f"Georboard Status: {georboard_interface.status}")
    print(f"Authentication Token Hash: {georboard_interface.authentication['system_key_hash']}")

    print("--- Ingesting Cognitive Memories ---")
    emb_user_pref = [0.9, 0.1, 0.2]
    emb_fact_sky = [0.1, 0.8, 0.9]
    emb_task_todo = [0.4, 0.5, 0.1]
    emb_query = [0.85, 0.15, 0.25]

    repo = storage_system.memory_repository
    n1 = repo.write_memory("User prefers dark mode UI", emb_user_pref, importance=0.9, tags=["ui", "pref"])
    n2 = repo.write_memory("The sky turns red during sunset due to Rayleigh scattering", emb_fact_sky, importance=0.4, tags=["fact"])
    n3 = repo.write_memory("Scheduled system maintenance at midnight", emb_task_todo, importance=0.8, tags=["task"])
    n4 = repo.write_memory("User requested concise answers", emb_user_pref, importance=0.95, tags=["style"])

    print(f"Working Memory Count: {len(repo.working_memory)}")
    print(f"Long-Term Memory Count: {len(repo.long_term_memory)}")

    print("--- Querying Cognitive Memory Repository ---")
    retrieved = repo.query_memory(query_embedding=emb_query, top_k=2)

    for memory, score in retrieved:
        print(f"[{score:.4f}] Memory Content: '{memory.content}' (Access Count: {memory.access_count})")
