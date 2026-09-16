"""
Artificial Cognitive Brain - Semantic Memory Module.

This module implements the Semantic Memory Architecture for the Artificial Cognitive Brain.
Modeled after cognitive science paradigms (Collins & Quillian's Network Model, Collins & Loftus'
Spreading Activation Theory), this system represents general world knowledge, facts, concepts, 
and relationships as a dynamic knowledge graph with spreading activation and vector groundings.

Key Components:
    - ConceptNode: Representation of a semantic concept, entity, or category.
    - SemanticRelation: Directed and weighted relationship link between two concepts.
    - SemanticKnowledgeGraph: Graph engine supporting entity-relation-attribute (ERA) queries and spreading activation.
    - SemanticMemorySystem: Core cognitive module interfacing with the Global Workspace.

Author: Artificial Cognitive Brain Team
License: MIT
Version: 0.1.0
"""

import math
import logging
from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass, field

from cognitive_models import BaseCognitiveModel

# Configure module logger
logger = logging.getLogger(__name__)


@dataclass
class SemanticRelation:
    """
    Represents a directed semantic link between a subject node and an object node.

    Attributes:
        relation_type (str): Type of predicate/relation (e.g., 'is_a', 'has_property', 'part_of', 'causes').
        target_concept_id (str): ID of the connected target concept.
        weight (float): Strength/confidence of the connection in range [0.0, 1.0].
        metadata (Dict[str, Any]): Additional contextual or provenance tags.
    """
    relation_type: str
    target_concept_id: str
    weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConceptNode:
    """
    Represents an individual concept or entity node within the Semantic Network.

    Attributes:
        concept_id (str): Unique identifier for the concept (e.g., 'bird', 'canary', 'fly').
        label (str): Human-readable name/label.
        category (str): Semantic category or ontological taxonomy (e.g., 'animal', 'action', 'attribute').
        attributes (Dict[str, Any]): Key-value property attributes (e.g., {'color': 'yellow'}).
        relations (List[SemanticRelation]): List of outgoing semantic links.
        activation_level (float): Current transient activation level [0.0, 1.0].
        embedding (List[float]): Vector representation for semantic similarity matching.
    """
    concept_id: str
    label: str
    category: str = "general"
    attributes: Dict[str, Any] = field(default_factory=dict)
    relations: List[SemanticRelation] = field(default_factory=list)
    activation_level: float = 0.0
    embedding: List[float] = field(default_factory=list)

    def add_relation(self, relation_type: str, target_id: str, weight: float = 1.0) -> None:
        """Adds or updates an outgoing directed relation to another concept."""
        for rel in self.relations:
            if rel.relation_type == relation_type and rel.target_concept_id == target_id:
                rel.weight = weight
                return
        self.relations.append(SemanticRelation(relation_type=relation_type, target_concept_id=target_id, weight=weight))


class SemanticKnowledgeGraph:
    """
    Knowledge Graph engine managing nodes, relations, and spreading activation algorithms.
    """

    def __init__(self, decay_factor: float = 0.8) -> None:
        """
        Initializes the knowledge graph.

        Args:
            decay_factor (float): Attenuation factor applied during spreading activation steps.
        """
        self.nodes: Dict[str, ConceptNode] = {}
        self.decay_factor = decay_factor

    def add_concept(
        self, 
        concept_id: str, 
        label: str, 
        category: str = "general", 
        attributes: Optional[Dict[str, Any]] = None,
        embedding: Optional[List[float]] = None
    ) -> ConceptNode:
        """Inserts or updates a concept node in the graph."""
        concept_key = concept_id.lower().strip()
        if concept_key in self.nodes:
            node = self.nodes[concept_key]
            node.label = label
            node.category = category
            if attributes:
                node.attributes.update(attributes)
            if embedding:
                node.embedding = embedding
        else:
            node = ConceptNode(
                concept_id=concept_key,
                label=label,
                category=category,
                attributes=attributes or {},
                embedding=embedding or []
            )
            self.nodes[concept_key] = node
        return node

    def add_triple(self, subject_id: str, predicate: str, object_id: str, weight: float = 1.0) -> None:
        """
        Inserts a subject-predicate-object triple into the knowledge graph.
        
        Creates placeholder nodes if subject or object do not exist yet.
        """
        subj_key = subject_id.lower().strip()
        obj_key = object_id.lower().strip()

        if subj_key not in self.nodes:
            self.add_concept(subj_key, label=subject_id)
        if obj_key not in self.nodes:
            self.add_concept(obj_key, label=object_id)

        self.nodes[subj_key].add_relation(predicate, obj_key, weight)

    def spreading_activation(
        self, 
        seed_concept_ids: List[str], 
        steps: int = 2, 
        initial_activation: float = 1.0,
        threshold: float = 0.1
    ) -> Dict[str, float]:
        """
        Executes Collins & Loftus spreading activation algorithm over the graph network.

        Args:
            seed_concept_ids (List[str]): Starting concept nodes for activation injection.
            steps (int): Number of spreading distance hops.
            initial_activation (float): Initial energy injected into seed nodes.
            threshold (float): Minimum activation level required to continue propagation.

        Returns:
            Dict[str, float]: Dictionary mapping concept_ids to their resulting activation levels.
        """
        # Reset current graph activations
        for node in self.nodes.values():
            node.activation_level = 0.0

        # Inject initial activation
        current_active: Dict[str, float] = {}
        for seed in seed_concept_ids:
            key = seed.lower().strip()
            if key in self.nodes:
                self.nodes[key].activation_level = initial_activation
                current_active[key] = initial_activation

        # Iterative propagation
        for step in range(steps):
            next_active: Dict[str, float] = {}
            for source_id, act_val in current_active.items():
                if act_val < threshold:
                    continue
                source_node = self.nodes[source_id]

                for rel in source_node.relations:
                    target_id = rel.target_concept_id
                    if target_id in self.nodes:
                        propagated_energy = act_val * rel.weight * self.decay_factor
                        target_node = self.nodes[target_id]
                        target_node.activation_level = min(1.0, target_node.activation_level + propagated_energy)
                        
                        if target_node.activation_level >= threshold:
                            next_active[target_id] = target_node.activation_level

            current_active = next_active

        return {k: round(v.activation_level, 3) for k, v in self.nodes.items() if v.activation_level >= threshold}

    def query_relations(self, subject_id: str, predicate: Optional[str] = None) -> List[Tuple[str, str, float]]:
        """Queries outgoing relations from a specified subject concept."""
        subj_key = subject_id.lower().strip()
        if subj_key not in self.nodes:
            return []

        results = []
        for rel in self.nodes[subj_key].relations:
            if predicate is None or rel.relation_type.lower() == predicate.lower():
                results.append((rel.relation_type, rel.target_concept_id, rel.weight))
        return results


class SemanticMemorySystem(BaseCognitiveModel):
    """
    Semantic Memory Cognitive System interfacing with the Global Workspace.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initializes the Semantic Memory System.

        Args:
            config (Optional[Dict[str, Any]]): Module configuration dict.
        """
        super().__init__(model_name="SemanticMemorySystem", config=config)

        cfg = self.config or {}
        decay = cfg.get("spreading_decay", 0.75)
        self.graph = SemanticKnowledgeGraph(decay_factor=decay)
        self._populate_default_ontology()

    def _populate_default_ontology(self) -> None:
        """Bootstraps basic semantic knowledge triples into the network."""
        default_triples = [
            ("canary", "is_a", "bird", 1.0),
            ("bird", "is_a", "animal", 1.0),
            ("bird", "has_capability", "fly", 0.9),
            ("bird", "has_part", "wings", 1.0),
            ("canary", "has_color", "yellow", 1.0),
            ("canary", "has_capability", "sing", 0.8),
            ("cat", "is_a", "animal", 1.0),
            ("cat", "has_part", "fur", 0.9)
        ]
        for subj, pred, obj, w in default_triples:
            self.graph.add_triple(subj, pred, obj, w)

    def learn_fact(self, subject: str, predicate: str, object_concept: str, weight: float = 1.0) -> None:
        """Stores a new factual semantic triple into long-term knowledge storage."""
        self.graph.add_triple(subject, predicate, object_concept, weight)
        logger.info(f"Learned new semantic triple: ({subject}) --[{predicate}]--> ({object_concept})")

    def query_concept(self, concept_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves complete structural properties and relations of a concept node."""
        key = concept_id.lower().strip()
        node = self.graph.nodes.get(key)
        if not node:
            return None

        return {
            "concept_id": node.concept_id,
            "label": node.label,
            "category": node.category,
            "attributes": node.attributes,
            "relations": [{"predicate": r.relation_type, "target": r.target_concept_id, "weight": r.weight} for r in node.relations]
        }

    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Cognitive loop step execution. Processes semantic commands from the workspace.

        Args:
            inputs (Dict[str, Any]): Global workspace state.

        Returns:
            Dict[str, Any]: Semantic memory search and activation results.
        """
        cmd = inputs.get("semantic_command", {})
        response = {}

        if isinstance(cmd, dict) and "action" in cmd:
            action = cmd["action"]

            if action == "learn":
                self.learn_fact(
                    subject=cmd["subject"],
                    predicate=cmd["predicate"],
                    object_concept=cmd["object"],
                    weight=cmd.get("weight", 1.0)
                )
                response["status"] = "fact_learned"

            elif action == "query_concept":
                response["concept_info"] = self.query_concept(cmd["concept_id"])

            elif action == "spreading_activation":
                seeds = cmd.get("seed_concepts", [])
                steps = cmd.get("steps", 2)
                response["active_network"] = self.graph.spreading_activation(seed_concept_ids=seeds, steps=steps)

        return {
            "total_concepts": len(self.graph.nodes),
            "command_response": response
        }

    def reset_state(self) -> None:
        """Resets network node activations."""
        for node in self.graph.nodes.values():
            node.activation_level = 0.0
        super().reset_state()
