"""
Artificial Cognitive Brain - Working Memory Module.

This module implements the Working Memory Architecture for the Artificial Cognitive Brain.
It is modeled after classic cognitive science paradigms (Baddeley & Hitch Model, Cowan's N-item
Capacity Model) to manage short-term information retention, attention focus, decay dynamic mechanisms,
and working memory updates for real-time cognitive processing.

Key Components:
    - WorkingMemoryItem: Data container for individual items in working memory with activation levels.
    - CentralExecutiveInterface: Sub-component responsible for allocating attentional resources.
    - PhonologicalLoop: Sub-buffer for handling verbal and textual memory buffers.
    - VisuospatialSketchpad: Sub-buffer for visual and spatial state representations.
    - WorkingMemorySystem: The core working memory system coordinating buffers and decay loops.

Author: Artificial Cognitive Brain Team
License: MIT
Version: 0.1.0
"""

import time
import logging
import math
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field

from cognitive_models import BaseCognitiveModel

# Configure module logger
logger = logging.getLogger(__name__)


@dataclass
class WorkingMemoryItem:
    """
    Data structure representing a single item stored in Working Memory.

    Attributes:
        item_id (str): Unique identifier for the memory item.
        content (Any): The payload data stored in memory (e.g., text, features, vector).
        modality (str): Modality type ('phonological', 'visuospatial', or 'abstract').
        activation_level (float): Current activation value in range [0.0, 1.0].
        base_activation (float): Initial baseline activation assigned upon encoding.
        creation_timestamp (float): UNIX timestamp when item was encoded.
        last_accessed_timestamp (float): UNIX timestamp of the last access/rehearsal.
        access_count (int): Number of times this item has been retrieved or rehearsed.
        metadata (Dict[str, Any]): Additional metadata or contextual tags.
    """
    item_id: str
    content: Any
    modality: str = "abstract"
    activation_level: float = 1.0
    base_activation: float = 1.0
    creation_timestamp: float = field(default_factory=time.time)
    last_accessed_timestamp: float = field(default_factory=time.time)
    access_count: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)

    def rehearse(self, boost_amount: float = 0.2) -> None:
        """
        Rehearses the item to boost its activation level and reset its access time.

        Args:
            boost_amount (float): Amount of activation incremented upon rehearsal.
        """
        self.last_accessed_timestamp = time.time()
        self.access_count += 1
        self.activation_level = min(1.0, self.activation_level + boost_amount)
        logger.debug(f"Item '{self.item_id}' rehearsed. New activation: {self.activation_level:.3f}")

    def compute_decay(self, decay_rate: float, current_time: Optional[float] = None) -> float:
        """
        Calculates and updates memory activation decay over time using exponential decay.

        Args:
            decay_rate (float): Rate at which memory decays per second.
            current_time (Optional[float]): Optional reference time; defaults to system time.

        Returns:
            float: Updated activation level.
        """
        now = current_time or time.time()
        time_elapsed = max(0.0, now - self.last_accessed_timestamp)
        
        # Exponential activation decay formula: A(t) = A0 * e^(-lambda * t)
        decay_factor = math.exp(-decay_rate * time_elapsed)
        self.activation_level *= decay_factor
        return self.activation_level


class WorkingMemoryBuffer:
    """
    Generic buffer storage implementing capacity bounds and decay eviction policies.
    """

    def __init__(self, buffer_name: str, max_capacity: int = 4, decay_rate: float = 0.05) -> None:
        """
        Initializes a memory buffer.

        Args:
            buffer_name (str): Identifier for this sub-buffer.
            max_capacity (int): Maximum number of items the buffer can hold concurrently.
            decay_rate (float): Temporal decay rate per second for items in this buffer.
        """
        self.buffer_name = buffer_name
        self.max_capacity = max_capacity
        self.decay_rate = decay_rate
        self.items: Dict[str, WorkingMemoryItem] = {}

    def add_item(self, item: WorkingMemoryItem) -> Optional[WorkingMemoryItem]:
        """
        Adds an item to the buffer. If capacity is exceeded, evicts the item with lowest activation.

        Args:
            item (WorkingMemoryItem): The item to store.

        Returns:
            Optional[WorkingMemoryItem]: The evicted item if capacity was exceeded, else None.
        """
        self.apply_decay()
        evicted_item = None

        if len(self.items) >= self.max_capacity and item.item_id not in self.items:
            evicted_item = self._evict_weakest_item()

        self.items[item.item_id] = item
        logger.info(f"Added item '{item.item_id}' to buffer '{self.buffer_name}'.")
        return evicted_item

    def get_item(self, item_id: str) -> Optional[WorkingMemoryItem]:
        """Retrieves and automatically rehearses an item if present."""
        self.apply_decay()
        item = self.items.get(item_id)
        if item:
            item.rehearse()
        return item

    def apply_decay(self, purge_threshold: float = 0.05) -> List[str]:
        """
        Applies temporal decay to all items in buffer and purges items below threshold.

        Args:
            purge_threshold (float): Minimum activation level before an item is forgotten.

        Returns:
            List[str]: Identifiers of purged (forgotten) items.
        """
        now = time.time()
        purged_ids = []

        for item_id, item in list(self.items.items()):
            current_act = item.compute_decay(self.decay_rate, current_time=now)
            if current_act < purge_threshold:
                purged_ids.append(item_id)
                del self.items[item_id]

        if purged_ids:
            logger.info(f"Purged forgotten items from '{self.buffer_name}': {purged_ids}")
        return purged_ids

    def _evict_weakest_item(self) -> Optional[WorkingMemoryItem]:
        """Finds and removes the item with the lowest activation level."""
        if not self.items:
            return None
        
        weakest_id = min(self.items, key=lambda k: self.items[k].activation_level)
        evicted = self.items.pop(weakest_id)
        logger.warning(
            f"Capacity reached in buffer '{self.buffer_name}'. Evicted weakest item '{weakest_id}' "
            f"(activation: {evicted.activation_level:.3f})."
        )
        return evicted

    def clear(self) -> None:
        """Clears all stored items in this buffer."""
        self.items.clear()


class WorkingMemorySystem(BaseCognitiveModel):
    """
    Central Working Memory Manager coordinating specialized buffers, attentional focus,
    and executive control integration within the Artificial Cognitive Brain.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initializes the complete Working Memory System.

        Args:
            config (Optional[Dict[str, Any]]): Configuration dictionary specifying capacity limits,
                                               decay rates, and cognitive thresholds.
        """
        super().__init__(model_name="WorkingMemorySystem", config=config)
        
        # Extracted parameters from config with cognitive defaults
        cfg = self.config or {}
        self.phonological_capacity = cfg.get("phonological_capacity", 4)
        self.visuospatial_capacity = cfg.get("visuospatial_capacity", 4)
        self.abstract_capacity = cfg.get("abstract_capacity", 7)  # Miller's Magic Number 7+-2
        self.decay_rate = cfg.get("decay_rate", 0.02)
        self.forgetting_threshold = cfg.get("forgetting_threshold", 0.1)

        # Specialized Memory Sub-buffers
        self.phonological_loop = WorkingMemoryBuffer(
            buffer_name="PhonologicalLoop", 
            max_capacity=self.phonological_capacity, 
            decay_rate=self.decay_rate
        )
        self.visuospatial_sketchpad = WorkingMemoryBuffer(
            buffer_name="VisuospatialSketchpad", 
            max_capacity=self.visuospatial_capacity, 
            decay_rate=self.decay_rate
        )
        self.central_buffer = WorkingMemoryBuffer(
            buffer_name="CentralExecutiveBuffer", 
            max_capacity=self.abstract_capacity, 
            decay_rate=self.decay_rate
        )

        # Focus of Attention (Cowan's Model: single active target)
        self.focus_of_attention_id: Optional[str] = None

    def store(
        self, 
        item_id: str, 
        content: Any, 
        modality: str = "abstract", 
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Encodes and inserts a new memory item into the appropriate working memory buffer.

        Args:
            item_id (str): Unique label for the memory item.
            content (Any): Object, text, or vector content.
            modality (str): Modality type ('phonological', 'visuospatial', or 'abstract').
            metadata (Optional[Dict[str, Any]]): Contextual tags.

        Returns:
            Dict[str, Any]: Result status including any evicted items.
        """
        item = WorkingMemoryItem(
            item_id=item_id,
            content=content,
            modality=modality,
            metadata=metadata or {}
        )

        target_buffer = self._route_buffer(modality)
        evicted = target_buffer.add_item(item)

        # Automatically focus attention on newly encoded item
        self.set_focus_of_attention(item_id)

        return {
            "status": "stored",
            "item_id": item_id,
            "modality": modality,
            "evicted_item": evicted.item_id if evicted else None
        }

    def retrieve(self, item_id: str) -> Optional[WorkingMemoryItem]:
        """
        Searches across all sub-buffers to retrieve an item and refresh its activation.

        Args:
            item_id (str): Identifier of the item to retrieve.

        Returns:
            Optional[WorkingMemoryItem]: The item object if found, else None.
        """
        for buffer in [self.phonological_loop, self.visuospatial_sketchpad, self.central_buffer]:
            item = buffer.get_item(item_id)
            if item:
                self.set_focus_of_attention(item_id)
                return item
        
        logger.warning(f"Item '{item_id}' not found in Working Memory.")
        return None

    def set_focus_of_attention(self, item_id: str) -> bool:
        """
        Directs top-down attentional focus to a specific item in working memory, boosting its activation.

        Args:
            item_id (str): Item to focus on.

        Returns:
            bool: True if item was found and focused, False otherwise.
        """
        for buffer in [self.phonological_loop, self.visuospatial_sketchpad, self.central_buffer]:
            if item_id in buffer.items:
                self.focus_of_attention_id = item_id
                buffer.items[item_id].rehearse(boost_amount=0.3)
                logger.info(f"Focus of Attention set to item: '{item_id}'")
                return True
        return False

    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Cognitive cycle integration process step.
        Performs periodic decay updates, processes sensory/executive inputs, and returns state summary.

        Args:
            inputs (Dict[str, Any]): Global workspace state or specific working memory instructions.

        Returns:
            Dict[str, Any]: Current snapshot of Working Memory state and attention status.
        """
        # Purge decayed items across all buffers
        purged_phonological = self.phonological_loop.apply_decay(self.forgetting_threshold)
        purged_visuospatial = self.visuospatial_sketchpad.apply_decay(self.forgetting_threshold)
        purged_central = self.central_buffer.apply_decay(self.forgetting_threshold)

        # Handle incoming direct memory commands from executive system if present
        wm_command = inputs.get("wm_command")
        if isinstance(wm_command, dict):
            action = wm_command.get("action")
            if action == "store":
                self.store(
                    item_id=wm_command["id"],
                    content=wm_command["content"],
                    modality=wm_command.get("modality", "abstract"),
                    metadata=wm_command.get("metadata")
                )
            elif action == "focus":
                self.set_focus_of_attention(wm_command["id"])

        return {
            "working_memory_state": self.get_summary(),
            "purged_items": purged_phonological + purged_visuospatial + purged_central,
            "focus_of_attention": self.focus_of_attention_id
        }

    def get_summary(self) -> Dict[str, Any]:
        """
        Generates a diagnostic overview of working memory contents and activation levels.

        Returns:
            Dict[str, Any]: Diagnostic summary of items stored in each buffer.
        """
        def format_buffer(buf: WorkingMemoryBuffer) -> Dict[str, float]:
            return {k: round(v.activation_level, 3) for k, v in buf.items.items()}

        return {
            "phonological_loop": format_buffer(self.phonological_loop),
            "visuospatial_sketchpad": format_buffer(self.visuospatial_sketchpad),
            "central_buffer": format_buffer(self.central_buffer),
            "total_items": (
                len(self.phonological_loop.items) + 
                len(self.visuospatial_sketchpad.items) + 
                len(self.central_buffer.items)
            )
        }

    def reset_state(self) -> None:
        """Clears all working memory sub-buffers and resets focus of attention."""
        self.phonological_loop.clear()
        self.visuospatial_sketchpad.clear()
        self.central_buffer.clear()
        self.focus_of_attention_id = None
        super().reset_state()

    def _route_buffer(self, modality: str) -> WorkingMemoryBuffer:
        """Routes item to appropriate sub-buffer based on its sensory modality."""
        if modality == "phonological" or modality == "text":
            return self.phonological_loop
        elif modality == "visuospatial" or modality == "image":
            return self.visuospatial_sketchpad
        else:
            return self.central_buffer
