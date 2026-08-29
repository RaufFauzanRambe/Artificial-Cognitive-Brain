"""
learning.py - Artificial Cognitive Brain Learning System & Speech Automation

Handles adaptive learning, engram reinforcement, dynamic weight updates, 
association plasticity, cognitive pattern abstraction, and audio feedback 
synthesis based on task execution rewards.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
import math
import os
import time
import uuid
import pyttsx3

# Fallback & integration setup for memory modules
try:
    from memory import (
        CognitiveMemoryRepository,
        MemoryNode,
        StorageDataCognitive,
        SystemDashboard,
    )
except ImportError:
    # Safe mock structures for standalone syntax verification
    pass


class LearningMode(Enum):
    """Defines the cognitive learning strategy applied during feedback processing."""
    HEBBIAN_REINFORCEMENT = "HEBBIAN_REINFORCEMENT"
    PENALTY_DECAY = "PENALTY_DECAY"
    PATTERN_ABSTRACT = "PATTERN_ABSTRACT"
    STRICT_CONSOLIDATION = "STRICT_CONSOLIDATION"


@dataclass
class LearningFeedback:
    """Structure for memory reinforcement feedback."""
    task_id: str
    target_memory_ids: List[str]
    reward_signal: float  # Range: -1.0 (Punishment) to +1.0 (Reward)
    feedback_notes: str
    learning_mode: LearningMode = LearningMode.HEBBIAN_REINFORCEMENT
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class LearningMetrics:
    """Tracking statistics for the cognitive learning lifecycle."""
    total_cycles: int = 0
    successful_reinforcements: int = 0
    penalized_memories: int = 0
    promoted_to_long_term: int = 0
    average_reward: float = 0.0
    last_learning_time: Optional[datetime] = None


class TextToSpeechManager:
    """Encapsulates TTS configuration, voice fallback, and audio rendering engine."""

    def __init__(self, speech_rate: int = 160, volume: float = 0.9):
        self.speech_rate = speech_rate
        self.volume = volume
        self.tts_engine: Optional[pyttsx3.Engine] = None
        self._initialize_engine()

    def _initialize_engine(self) -> None:
        """Configures default speech output properties with robust exception handling."""
        try:
            self.tts_engine = pyttsx3.init()
            if self.tts_engine:
                self.tts_engine.setProperty('rate', self.speech_rate)
                self.tts_engine.setProperty('volume', self.volume)
                
                # Try selecting a natural sounding voice if available
                voices = self.tts_engine.getProperty('voices')
                for voice in voices:
                    if "english" in voice.name.lower() or "indic" in voice.name.lower():
                        self.tts_engine.setProperty('voice', voice.id)
                        break
        except Exception as e:
            print(f"⚠️ [Speech Engine Init Warning]: Failed to bind TTS hardware. Audio will be muted. Details: {e}")
            self.tts_engine = None

    def speak(self, text: str) -> bool:
        """Synthesizes voice feedback from learning events."""
        print(f"🔊 [AI Voice Learning Output]: {text}")
        if not self.tts_engine:
            return False

        try:
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()
            return True
        except Exception as e:
            print(f"❌ [TTS Execution Error]: {e}")
            return False


class CognitiveLearningEngine:
    """
    Manages learning cycles, memory adjustments, pattern extraction, 
    associative link plasticity, and voice notification of learning progress.
    """

    def __init__(
        self, 
        storage_engine: StorageDataCognitive, 
        learning_rate: float = 0.1,
        hebbian_threshold: float = 0.75,
        enable_speech: bool = True
    ):
        """
        Initialize the Cognitive Learning Engine.

        :param storage_engine: The active StorageDataCognitive instance.
        :param learning_rate: Coefficient controlling magnitude of weight updates (0.0 to 1.0).
        :param hebbian_threshold: Threshold above which co-active nodes form strong links.
        :param enable_speech: Toggle automated voice synthesis output.
        """
        self.storage_engine = storage_engine
        self.memory_repo: CognitiveMemoryRepository = storage_engine.memory_repository[cite: 2]
        self.learning_rate = learning_rate
        self.hebbian_threshold = hebbian_threshold
        self.enable_speech = enable_speech
        
        self.tts_manager = TextToSpeechManager()
        self.metrics = LearningMetrics()
        self.feedback_history: List[LearningFeedback] = []

    # -------------------------------------------------------------------------
    # Core Learning & Reinforcement Methods
    # -------------------------------------------------------------------------

    def process_feedback(
        self, 
        feedback: LearningFeedback, 
        speak_output: Optional[bool] = None
    ) -> LearningMetrics:
        """
        Adjusts importance, associative weights, and vector spaces of memory nodes 
        based on task performance rewards.
        """
        should_speak = self.enable_speech if speak_output is None else speak_output
        start_time = time.time()
        
        # 1. Validate Target Memory Nodes
        target_nodes: List[MemoryNode] = []
        for mem_id in feedback.target_memory_ids:
            node = self.memory_repo._find_node(mem_id)
            if node:
                target_nodes.append(node)

        if not target_nodes:
            print(f"⚠️ [Learning Engine]: No valid memory nodes found for task '{feedback.task_id}'.")
            return self.metrics

        # 2. Update Individual Engram Importance & Decay Profiles
        promoted_count = 0
        delta = feedback.reward_signal * self.learning_rate

        for node in target_nodes:
            # Update importance score within bound [0.0, 1.0]
            old_importance = node.importance_score
            new_importance = max(0.0, min(1.0, old_importance + delta))
            node.importance_score = round(new_importance, 4)

            # Update access counts and temporal timestamps
            node.access_count += 1
            node.last_accessed = datetime.utcnow()

            # Dynamic Memory Consolidation into Long-Term Memory
            if new_importance >= self.memory_repo.consolidation_threshold:
                if node.id not in self.memory_repo.long_term_memory:
                    self.memory_repo.long_term_memory[node.id] = node
                    promoted_count += 1

            # Adjust underlying embedding towards positive reward vectors
            if feedback.reward_signal > 0:
                node.embedding = self._nudge_embedding(node.embedding, delta_factor=delta * 0.05)

        # 3. Apply Hebbian Learning Across Co-Active Memory Nodes
        if len(target_nodes) > 1 and feedback.reward_signal > 0.3:
            self._apply_hebbian_association(target_nodes, feedback.reward_signal)

        # 4. Extract Cognitive Abstract Patterns if Signal is Strongly Positive
        if feedback.reward_signal >= 0.85 and len(target_nodes) >= 2:
            self._abstract_pattern(target_nodes, feedback)

        # 5. Update Historical Metrics
        self._update_metrics(feedback, promoted_count)

        # 6. Generate Voice Synthesis Announcement
        announcement = self._construct_announcement(feedback, len(target_nodes), promoted_count)
        if should_speak:
            self.tts_manager.speak(announcement)

        return self.metrics

    # -------------------------------------------------------------------------
    # Advanced Cognitive Plasticity Mechanics
    # -------------------------------------------------------------------------

    def _apply_hebbian_association(self, nodes: List[MemoryNode], reward: float) -> None:
        """
        Applies Hebbian Principle ('Neurons that fire together, wire together') 
        by creating/strengthening graph edges between co-active memory nodes.
        """
        weight_boost = round(reward * self.learning_rate * 0.5, 4)
        
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                node_a = nodes[i]
                node_b = nodes[j]
                
                # Check if association already exists in graph
                existing_weight = node_a.associations.get(node_b.id, 0.0)
                new_weight = min(1.0, existing_weight + weight_boost)
                
                # Update bidirectional association links[cite: 1, 2]
                self.memory_repo.create_association(node_a.id, node_b.id, weight=new_weight)[cite: 1, 2]

    def _abstract_pattern(self, nodes: List[MemoryNode], feedback: LearningFeedback) -> Optional[MemoryNode]:
        """
        Creates a higher-level abstract concept node in long-term memory 
        when multiple memories produce strong positive outcomes together.
        """
        combined_tags = list(set(tag for n in nodes for tag in n.tags))
        combined_tags.append("abstract_pattern")
        
        # Calculate centroid embedding
        vec_len = len(nodes[0].embedding)
        centroid_emb = [0.0] * vec_len
        for n in nodes:
            for idx in range(vec_len):
                centroid_emb[idx] += n.embedding[idx]
        centroid_emb = [round(v / len(nodes), 4) for v in centroid_emb]

        pattern_content = (
            f"Synthesized Pattern from task '{feedback.task_id[:8]}': "
            f"Derived from nodes [{', '.join([n.id[:6] for n in nodes])}]. "
            f"Context: {feedback.feedback_notes}"
        )

        abstract_node = self.memory_repo.write_memory([cite: 1, 2]
            content=pattern_content,
            embedding=centroid_emb,
            importance=0.90,
            tags=combined_tags
        )
        print(f"🧩 [Pattern Abstraction]: Generated high-level engram node {abstract_node.id[:8]}")
        return abstract_node

    def _nudge_embedding(self, embedding: List[float], delta_factor: float) -> List[float]:
        """Slightly adjusts embedding space vectors to reflect reinforced learning state."""
        return [round(v + (v * delta_factor), 4) for v in embedding]

    # -------------------------------------------------------------------------
    # Helper & Announcement Formatting Routines
    # -------------------------------------------------------------------------

    def _update_metrics(self, feedback: LearningFeedback, promoted_count: int) -> None:
        """Updates internal telemetry metrics for the learning system."""
        self.feedback_history.append(feedback)
        self.metrics.total_cycles += 1
        
        if feedback.reward_signal > 0:
            self.metrics.successful_reinforcements += 1
        else:
            self.metrics.penalized_memories += 1

        self.metrics.promoted_to_long_term += promoted_count
        
        # Recalculate running mean average of reward signals
        all_rewards = [f.reward_signal for f in self.feedback_history]
        self.metrics.average_reward = round(sum(all_rewards) / len(all_rewards), 4)
        self.metrics.last_learning_time = datetime.utcnow()

    def _construct_announcement(self, feedback: LearningFeedback, node_count: int, promoted_count: int) -> str:
        """Constructs human-readable voice announcements for TTS execution."""
        task_short = feedback.task_id[:6]
        if feedback.reward_signal > 0:
            status = f"reinforced {node_count} memory engrams with a positive signal of {feedback.reward_signal:.1f}."
        else:
            status = f"penalized {node_count} memory engrams with a negative signal of {feedback.reward_signal:.1f}."

        announcement = f"Learning complete for task {task_short}. System has {status}"
        if promoted_count > 0:
            announcement += f" {promoted_count} engram was consolidated into long term memory."
            
        return announcement


# -----------------------------------------------------------------------------
# Comprehensive Test & Demonstration Routine
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("==========================================================")
    print("   Artificial Cognitive Brain - Learning System Execution")
    print("==========================================================")

    # 1. Initialize Storage Infrastructure & Memory Engine
    storage_system = StorageDataCognitive(working_memory_capacity=10)[cite: 1, 2]
    
    # Ensure motherboard board connection is active
    if not storage_system.georboard:
        storage_system.brain_intelligence("ax-8i-+1hhxb_1biwdh")[cite: 1, 2]

    # 2. Seed Storage with Initial Engram Nodes
    print("\n[Step 1]: Seeding Initial Memory Engrams...")
    node1 = storage_system.memory_repository.write_memory([cite: 1, 2]
        content="Voice automation protocol initialized successfully", 
        embedding=[0.12, 0.45, 0.88], 
        importance=0.65,
        tags=["speech", "init"]
    )
    
    node2 = storage_system.memory_repository.write_memory([cite: 1, 2]
        content="Audio feedback loop validated with zero speech errors", 
        embedding=[0.15, 0.49, 0.91], 
        importance=0.68,
        tags=["speech", "validation"]
    )

    print(f"Created Node 1: ID={node1.id[:8]} | Importance={node1.importance_score}")
    print(f"Created Node 2: ID={node2.id[:8]} | Importance={node2.importance_score}")

    # 3. Instantiate Learning Engine with Speech Enabled
    learner = CognitiveLearningEngine(
        storage_engine=storage_system,
        learning_rate=0.15,
        hebbian_threshold=0.70,
        enable_speech=True
    )

    # 4. Trigger Positive Reinforcement Feedback Cycle
    print("\n[Step 2]: Processing Positive Learning Reinforcement Cycle...")
    positive_feedback = LearningFeedback(
        task_id=f"task-audio-{str(uuid.uuid4())[:4]}",
        target_memory_ids=[node1.id, node2.id],
        reward_signal=0.90,
        feedback_notes="Successful audio boot and voice synthesis validation.",
        learning_mode=LearningMode.HEBBIAN_REINFORCEMENT
    )
    
    updated_metrics = learner.process_feedback(positive_feedback, speak_output=True)

    # 5. Output Learning Verification Details
    print("\n--- Post-Learning State Verification ---")
    print(f"Node 1 Updated Importance : {node1.importance_score}")
    print(f"Node 2 Updated Importance : {node2.importance_score}")
    print(f"Associative Weight (N1->N2): {node1.associations.get(node2.id, 0.0)}")
    print(f"Long-Term Memory Count   : {len(storage_system.memory_repository.long_term_memory)}")[cite: 1, 2]
    
    print("\n--- System Learning Metrics ---")
    print(f"Total Cycles              : {updated_metrics.total_cycles}")
    print(f"Successful Reinforcements: {updated_metrics.successful_reinforcements}")
    print(f"Average Reward            : {updated_metrics.average_reward}")
    print(f"Promoted Engrams          : {updated_metrics.promoted_to_long_term}")
    print("==========================================================")
