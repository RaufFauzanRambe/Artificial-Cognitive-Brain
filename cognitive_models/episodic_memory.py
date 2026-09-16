"""
Artificial Cognitive Brain - Episodic Memory Module.

This module implements the Episodic Memory Architecture for the Artificial Cognitive Brain.
Modeled after human autobiographical memory theories (Tulving's Episodic System, Complementary
Learning Systems Theory, and Temporal Context Models), this system enables the agent to store,
consolidate, index, reconstruct, and recall time-stamped experiential events (episodes) with 
associated perceptual, emotional, and causal metadata.

Key Features:
    - EpisodicTrace: Structured data class capturing multi-modal experience snapshots (Who, What, Where, When, Why, Affect).
    - Temporal & Spatial Indexing: Time-bucketed and location-bound retrieval trees.
    - Vector Semantic Search: Cosine similarity-based pattern matching over high-dimensional feature embeddings.
    - Memory Consolidation & Decay: Hippocampal-to-cortical consolidation modeling with emotional modulation.
    - Reconstructive Retrieval: Pattern completion mechanism for partial or noisy queries.
    - Counterfactual & Causal Querying: Sequence search and causal link tracing between consecutive episodes.

Author: Artificial Cognitive Brain Team
License: MIT
Version: 0.1.0
"""

import time
import math
import uuid
import logging
from typing import Dict, Any, List, Optional, Tuple, Set, Union
from dataclasses import dataclass, field

from cognitive_models import BaseCognitiveModel

# Configure module logger
logger = logging.getLogger(__name__)


@dataclass
class EpisodicTrace:
    """
    Represents an autobiographical memory trace (an episode) stored in episodic memory.

    Attributes:
        episode_id (str): Unique UUID identifying the episode trace.
        timestamp (float): UNIX timestamp when the episode occurred.
        location (Dict[str, Any]): Spatial context (e.g., coordinates, room_id, environment name).
        agent_id (str): Identifier of the agent experiencing the episode.
        sensory_payload (Dict[str, Any]): Raw or preprocessed sensory inputs (visual, audio, text).
        semantic_concepts (List[str]): Extracted concept tags, keywords, or entities.
        embedding (List[float]): Dense vector representation of the event for similarity search.
        emotional_state (Dict[str, float]): Affective dimensions (valence: [-1.0, 1.0], arousal: [0.0, 1.0]).
        causal_antecedents (List[str]): List of prior episode IDs that led to this event.
        causal_consequences (List[str]): List of subsequent episode IDs triggered by this event.
        rehearsal_count (int): Number of times this episode has been retrieved or consolidated.
        consolidation_score (float): Stability degree of the trace [0.0, 1.0] (0 = hippocampal, 1 = neocortical).
        last_retrieved_timestamp (float): UNIX timestamp of the most recent retrieval.
        metadata (Dict[str, Any]): Additional operational or diagnostic key-value pairs.
    """
    episode_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    location: Dict[str, Any] = field(default_factory=dict)
    agent_id: str = "self"
    sensory_payload: Dict[str, Any] = field(default_factory=dict)
    semantic_concepts: List[str] = field(default_factory=list)
    embedding: List[float] = field(default_factory=list)
    emotional_state: Dict[str, float] = field(default_factory=lambda: {"valence": 0.0, "arousal": 0.0})
    causal_antecedents: List[str] = field(default_factory=list)
    causal_consequences: List[str] = field(default_factory=list)
    rehearsal_count: int = 0
    consolidation_score: float = 0.0
    last_retrieved_timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def calculate_emotional_salience(self) -> float:
        """
        Computes the emotional intensity/salience of the episode.
        
        High arousal and extreme valence increase emotional salience, making the
        memory trace resistant to temporal decay.

        Returns:
            float: Emotional salience score in range [0.0, 1.0].
        """
        valence = self.emotional_state.get("valence", 0.0)
        arousal = self.emotional_state.get("arousal", 0.0)
        # Salience increases with higher arousal and strong valence bias (positive or negative)
        salience = (abs(valence) * 0.4) + (arousal * 0.6)
        return max(0.0, min(1.0, salience))

    def compute_retention_strength(self, current_time: Optional[float] = None, base_decay_rate: float = 0.01) -> float:
        """
        Calculates current trace strength using a modified Ebbinghaus forgetting equation
        modulated by consolidation status and emotional salience.

        Formula:
            R(t) = exp(- (lambda * delta_t) / ((1 + consolidation) * (1 + salience * 2) * (1 + 0.5 * log(1 + rehearsals))))

        Args:
            current_time (Optional[float]): Reference time for decay calculation.
            base_decay_rate (float): Standard baseline decay rate per hour.

        Returns:
            float: Current memory strength score in range [0.0, 1.0].
        """
        now = current_time or time.time()
        elapsed_seconds = max(0.0, now - self.timestamp)
        elapsed_hours = elapsed_seconds / 3600.0

        salience = self.calculate_emotional_salience()
        rehearsal_factor = 1.0 + 0.5 * math.log(1 + self.rehearsal_count)
        protection_factor = (1.0 + self.consolidation_score) * (1.0 + salience * 2.0) * rehearsal_factor

        effective_decay = base_decay_rate / protection_factor
        retention = math.exp(-effective_decay * elapsed_hours)
        return max(0.0, min(1.0, retention))

    def touch(self) -> None:
        """Updates retrieval statistics when the trace is accessed."""
        self.rehearsal_count += 1
        self.last_retrieved_timestamp = time.time()


class EpisodicMemoryStore:
    """
    Primary storage unit and index manager for episodic memory traces.
    
    Provides multi-indexed lookup capabilities including temporal range queries,
    spatial clustering, concept tags, and vector similarity search.
    """

    def __init__(self, max_capacity: int = 10000, base_decay_rate: float = 0.005) -> None:
        """
        Initializes the episodic store.

        Args:
            max_capacity (int): Maximum number of episodic traces stored before auto-archiving/pruning.
            base_decay_rate (float): Rate at which un-consolidated memories decay over time.
        """
        self.max_capacity = max_capacity
        self.base_decay_rate = base_decay_rate

        # Core primary storage
        self.traces: Dict[str, EpisodicTrace] = {}

        # Indices for efficient retrieval
        self.temporal_index: List[Tuple[float, str]] = []  # Sorted list of (timestamp, episode_id)
        self.concept_index: Dict[str, Set[str]] = {}       # concept -> set of episode_ids
        self.location_index: Dict[str, Set[str]] = {}      # location_tag -> set of episode_ids

    def insert(self, trace: EpisodicTrace) -> str:
        """
        Stores an episodic trace and updates all structural indices.

        Args:
            trace (EpisodicTrace): The episode instance to be inserted.

        Returns:
            str: The unique episode_id of the stored trace.
        """
        if len(self.traces) >= self.max_capacity:
            self._prune_weakest_trace()

        self.traces[trace.episode_id] = trace

        # Update Temporal Index
        self.temporal_index.append((trace.timestamp, trace.episode_id))
        self.temporal_index.sort(key=lambda x: x[0])

        # Update Concept Index
        for concept in trace.semantic_concepts:
            concept_key = concept.lower().strip()
            if concept_key not in self.concept_index:
                self.concept_index[concept_key] = set()
            self.concept_index[concept_key].add(trace.episode_id)

        # Update Location Index
        loc_tag = str(trace.location.get("name", trace.location.get("region", "unknown"))).lower()
        if loc_tag not in self.location_index:
            self.location_index[loc_tag] = set()
        self.location_index[loc_tag].add(trace.episode_id)

        logger.info(f"Episodic trace '{trace.episode_id}' inserted into memory store.")
        return trace.episode_id

    def get_by_id(self, episode_id: str) -> Optional[EpisodicTrace]:
        """Retrieves a single trace by its unique ID and updates access statistics."""
        trace = self.traces.get(episode_id)
        if trace:
            trace.touch()
        return trace

    def query_temporal_range(self, start_time: float, end_time: float) -> List[EpisodicTrace]:
        """
        Retrieves all episodes that occurred within a specified window of time.

        Args:
            start_time (float): Lower temporal bound (UNIX timestamp).
            end_time (float): Upper temporal bound (UNIX timestamp).

        Returns:
            List[EpisodicTrace]: Chronologically sorted list of matching episodes.
        """
        matching_ids = [
            ep_id for ts, ep_id in self.temporal_index
            if start_time <= ts <= end_time
        ]
        results = [self.traces[ep_id] for ep_id in matching_ids if ep_id in self.traces]
        for tr in results:
            tr.touch()
        return results

    def query_by_concepts(self, concepts: List[str], match_all: bool = False) -> List[EpisodicTrace]:
        """
        Retrieves episodes associated with specific semantic concepts/tags.

        Args:
            concepts (List[str]): Search concept terms.
            match_all (bool): If True, requires episode to match ALL concepts; otherwise ANY concept.

        Returns:
            List[EpisodicTrace]: List of matching episode traces.
        """
        if not concepts:
            return []

        matched_id_sets = []
        for c in concepts:
            ckey = c.lower().strip()
            matched_id_sets.append(self.concept_index.get(ckey, set()))

        if match_all:
            final_ids = set.intersection(*matched_id_sets) if matched_id_sets else set()
        else:
            final_ids = set.union(*matched_id_sets) if matched_id_sets else set()

        results = [self.traces[ep_id] for ep_id in final_ids if ep_id in self.traces]
        for tr in results:
            tr.touch()
        return results

    def query_vector_similarity(
        self, 
        query_embedding: List[float], 
        top_k: int = 5, 
        min_similarity: float = 0.0
    ) -> List[Tuple[EpisodicTrace, float]]:
        """
        Performs vector cosine similarity search over all episodic embeddings.

        Args:
            query_embedding (List[float]): Dense numerical vector representing query context.
            top_k (int): Maximum number of top matching traces to return.
            min_similarity (float): Minimum similarity score threshold [0.0, 1.0].

        Returns:
            List[Tuple[EpisodicTrace, float]]: List of tuples containing (EpisodicTrace, similarity_score).
        """
        if not query_embedding or not self.traces:
            return []

        scored_traces: List[Tuple[EpisodicTrace, float]] = []

        for trace in self.traces.values():
            if not trace.embedding:
                continue
            
            sim = self._cosine_similarity(query_embedding, trace.embedding)
            if sim >= min_similarity:
                scored_traces.append((trace, sim))

        # Sort by similarity score descending
        scored_traces.sort(key=lambda x: x[1], reverse=True)
        top_results = scored_traces[:top_k]

        for tr, _ in top_results:
            tr.touch()

        return top_results

    def _cosine_similarity(self, vec_a: List[float], vec_b: List[float]) -> float:
        """Computes cosine similarity between two equal-length numeric vectors."""
        if len(vec_a) != len(vec_b) or not vec_a:
            return 0.0
        
        dot = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = math.sqrt(sum(a * a for a in vec_a))
        norm_b = math.sqrt(sum(b * b for b in vec_b))

        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0

        return dot / (norm_a * norm_b)

    def _prune_weakest_trace(self) -> Optional[str]:
        """Identifies and purges the lowest-retention trace when capacity limits are hit."""
        if not self.traces:
            return None

        now = time.time()
        weakest_id = min(
            self.traces.keys(),
            key=lambda k: self.traces[k].compute_retention_strength(now, self.base_decay_rate)
        )
        
        pruned_trace = self.traces.pop(weakest_id)
        
        # Clean up indices
        self.temporal_index = [tup for tup in self.temporal_index if tup[1] != weakest_id]
        for cset in self.concept_index.values():
            cset.discard(weakest_id)
        for lset in self.location_index.values():
            lset.discard(weakest_id)

        logger.warning(f"Capacity exceeded. Pruned weakest episodic trace '{weakest_id}'.")
        return weakest_id


class EpisodicMemorySystem(BaseCognitiveModel):
    """
    High-level Episodic Memory Cognitive Module for the Artificial Cognitive Brain.
    
    Coordinates experience recording, memory recall, offline consolidation,
    causal sequence reconstruction, and integration with the Global Workspace.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initializes the Episodic Memory System.

        Args:
            config (Optional[Dict[str, Any]]): System configuration parameters including
                                               capacity, decay rate, and consolidation rates.
        """
        super().__init__(model_name="EpisodicMemorySystem", config=config)

        cfg = self.config or {}
        max_capacity = cfg.get("max_capacity", 5000)
        base_decay = cfg.get("base_decay_rate", 0.005)
        self.consolidation_rate = cfg.get("consolidation_rate", 0.1)

        self.store = EpisodicMemoryStore(max_capacity=max_capacity, base_decay_rate=base_decay)
        self.last_episode_id: Optional[str] = None

    def record_experience(
        self,
        sensory_payload: Dict[str, Any],
        semantic_concepts: Optional[List[str]] = None,
        embedding: Optional[List[float]] = None,
        location: Optional[Dict[str, Any]] = None,
        emotional_state: Optional[Dict[str, float]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Encodes and records a new episodic event into long-term episodic memory.

        Automatically links the new episode to the immediately preceding episode as a
        causal antecedent.

        Args:
            sensory_payload (Dict[str, Any]): Raw/structured sensory state.
            semantic_concepts (Optional[List[str]]): Keywords or concept representations.
            embedding (Optional[List[float]]): Dense feature vector.
            location (Optional[Dict[str, Any]]): Spatial context data.
            emotional_state (Optional[Dict[str, float]]): Affective state (valence, arousal).
            metadata (Optional[Dict[str, Any]]): Arbitrary additional tags.

        Returns:
            str: Assigned episode_id.
        """
        antecedents = [self.last_episode_id] if self.last_episode_id else []

        trace = EpisodicTrace(
            timestamp=time.time(),
            location=location or {"name": "internal_workspace"},
            sensory_payload=sensory_payload,
            semantic_concepts=semantic_concepts or [],
            embedding=embedding or [],
            emotional_state=emotional_state or {"valence": 0.0, "arousal": 0.0},
            causal_antecedents=antecedents,
            metadata=metadata or {}
        )

        episode_id = self.store.insert(trace)

        # Link forward causal connection from preceding episode
        if self.last_episode_id:
            prev_trace = self.store.get_by_id(self.last_episode_id)
            if prev_trace:
                prev_trace.causal_consequences.append(episode_id)

        self.last_episode_id = episode_id
        return episode_id

    def recall_recent(self, count: int = 5) -> List[EpisodicTrace]:
        """Retrieves the N most recently recorded episodes chronologically."""
        recent_tuples = self.store.temporal_index[-count:]
        traces = [self.store.traces[ep_id] for _, ep_id in recent_tuples if ep_id in self.store.traces]
        for tr in traces:
            tr.touch()
        return traces

    def recall_by_query(
        self,
        query_text_concepts: Optional[List[str]] = None,
        query_embedding: Optional[List[float]] = None,
        time_range: Optional[Tuple[float, float]] = None,
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Flexible multi-modal reconstructive recall engine.
        
        Combines concept matching, vector similarity, and temporal bounds into a unified 
        retrieval score.

        Args:
            query_text_concepts (Optional[List[str]]): List of target concepts.
            query_embedding (Optional[List[float]]): Target feature vector.
            time_range (Optional[Tuple[float, float]]): (start_timestamp, end_timestamp)
            top_k (int): Number of top matches to return.

        Returns:
            List[Dict[str, Any]]: Diagnostic list of retrieved traces with match confidence scores.
        """
        candidates: Dict[str, float] = {}

        # 1. Vector Search
        if query_embedding:
            vec_results = self.store.query_vector_similarity(query_embedding, top_k=top_k * 2)
            for trace, sim in vec_results:
                candidates[trace.episode_id] = candidates.get(trace.episode_id, 0.0) + (sim * 0.5)

        # 2. Concept Search
        if query_text_concepts:
            concept_traces = self.store.query_by_concepts(query_text_concepts, match_all=False)
            for trace in concept_traces:
                candidates[trace.episode_id] = candidates.get(trace.episode_id, 0.0) + 0.3

        # 3. Temporal Filter & Retention Adjustment
        final_scored: List[Tuple[EpisodicTrace, float]] = []
        now = time.time()

        for ep_id, base_score in candidates.items():
            trace = self.store.traces.get(ep_id)
            if not trace:
                continue

            if time_range:
                start_t, end_t = time_range
                if not (start_t <= trace.timestamp <= end_t):
                    continue

            # Weight score by trace's current retention strength
            retention = trace.compute_retention_strength(now, self.store.base_decay_rate)
            final_score = base_score * (0.5 + 0.5 * retention)
            final_scored.append((trace, final_score))

        final_scored.sort(key=lambda x: x[1], reverse=True)
        top_matches = final_scored[:top_k]

        output = []
        for tr, score in top_matches:
            tr.touch()
            output.append({
                "episode_id": tr.episode_id,
                "timestamp": tr.timestamp,
                "concepts": tr.semantic_concepts,
                "emotional_state": tr.emotional_state,
                "retrieval_confidence": round(score, 4),
                "payload": tr.sensory_payload
            })

        return output

    def consolidate_memories(self, cycles: int = 1) -> Dict[str, Any]:
        """
        Simulates sleep or offline hippocampal-to-neocortical memory consolidation.

        Increases the consolidation score of frequently rehearsed or emotionally salient 
        memories, rendering them durable against temporal decay.

        Args:
            cycles (int): Number of consolidation iterations to run.

        Returns:
            Dict[str, Any]: Summary metrics of the consolidation process.
        """
        consolidated_count = 0

        for _ in range(cycles):
            for trace in self.store.traces.values():
                salience = trace.calculate_emotional_salience()
                if trace.rehearsal_count > 2 or salience > 0.6:
                    boost = self.consolidation_rate * (1.0 + salience)
                    trace.consolidation_score = min(1.0, trace.consolidation_score + boost)
                    consolidated_count += 1

        logger.info(f"Consolidation complete. Updated {consolidated_count} episode traces.")
        return {
            "consolidated_episodes": consolidated_count,
            "total_episodes_in_store": len(self.store.traces)
        }

    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Cognitive Architecture step handler.

        Listens for 'record_episode', 'recall_episode', or 'consolidate' commands in workspace.

        Args:
            inputs (Dict[str, Any]): Global workspace state.

        Returns:
            Dict[str, Any]: Episodic memory response and state summary.
        """
        action_payload = inputs.get("episodic_command", {})
        response = {}

        if isinstance(action_payload, dict) and "action" in action_payload:
            act = action_payload["action"]
            
            if act == "record":
                ep_id = self.record_experience(
                    sensory_payload=action_payload.get("sensory_payload", {}),
                    semantic_concepts=action_payload.get("semantic_concepts"),
                    embedding=action_payload.get("embedding"),
                    location=action_payload.get("location"),
                    emotional_state=action_payload.get("emotional_state"),
                    metadata=action_payload.get("metadata")
                )
                response["recorded_episode_id"] = ep_id

            elif act == "recall":
                matches = self.recall_by_query(
                    query_text_concepts=action_payload.get("concepts"),
                    query_embedding=action_payload.get("embedding"),
                    top_k=action_payload.get("top_k", 3)
                )
                response["recalled_episodes"] = matches

            elif act == "consolidate":
                res = self.consolidate_memories(cycles=action_payload.get("cycles", 1))
                response["consolidation_results"] = res

        return {
            "last_episode_id": self.last_episode_id,
            "total_stored_episodes": len(self.store.traces),
            "command_response": response
        }

    def reset_state(self) -> None:
        """Clears all stored episodic traces and resets indices."""
        self.store = EpisodicMemoryStore(
            max_capacity=self.store.max_capacity,
            base_decay_rate=self.store.base_decay_rate
        )
        self.last_episode_id = None
        super().reset_state()
