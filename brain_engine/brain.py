"""
Artificial Cognitive Brain - Central Brain Module (brain.py)
============================================================
This module acts as the primary cognitive coordinator, integrating 
perception, working memory, long-term memory, reasoning, and executive 
action planning into a unified operational pipeline.
"""

import logging
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

# Configure module logger
logger = logging.getLogger(__name__)


@dataclass
class SensoryInput:
    """Structure for incoming multi-modal raw data."""
    source: str
    payload: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)


@dataclass
class CognitiveState:
    """Represents the internal mental state of the Cognitive Brain."""
    working_memory: Dict[str, Any] = field(default_factory=dict)
    current_focus: Optional[str] = None
    emotional_valence: float = 0.0  # Range: -1.0 (negative) to 1.0 (positive)
    arousal_level: float = 0.5      # Range: 0.0 (passive) to 1.0 (alert)


@dataclass
class ActionOutput:
    """Structure for generated actions and decision parameters."""
    action_type: str
    parameters: Dict[str, Any]
    confidence_score: float
    timestamp: float = field(default_factory=time.time)


class PerceptionSubsystem:
    """Processes sensory input and filters irrelevant noise."""
    def process(self, sensory_input: SensoryInput) -> Dict[str, Any]:
        logger.debug(f"Processing sensory data from source: {sensory_input.source}")
        # Extract features and contextual signal
        return {
            "processed_feature": sensory_input.payload,
            "source_type": sensory_input.source,
            "processed_at": time.time()
        }


class MemorySubsystem:
    """Handles short-term working memory and long-term storage retrieval."""
    def __init__(self):
        self._episodic_memory: List[Dict[str, Any]] = []

    def store_episode(self, episode_data: Dict[str, Any]) -> None:
        self._episodic_memory.append(episode_data)
        logger.debug("Episode stored in episodic memory.")

    def recall_context(self, query_key: str) -> Optional[Dict[str, Any]]:
        # Mock retrieval mechanism
        for episode in reversed(self._episodic_memory):
            if query_key in episode:
                return episode[query_key]
        return None


class ReasoningSubsystem:
    """Executes logical inference, planning, and goal evaluation."""
    def evaluate(self, state: CognitiveState, perceived_data: Dict[str, Any]) -> ActionOutput:
        logger.debug("Evaluating state and selecting optimal action.")
        
        # Core decision logic based on current state and perception
        feature = perceived_data.get("processed_feature", {})
        intent = feature.get("intent", "idle")

        if intent == "query":
            return ActionOutput(
                action_type="GENERATE_RESPONSE",
                parameters={"text": "Processing query based on knowledge context."},
                confidence_score=0.95
            )
        elif intent == "alert":
            return ActionOutput(
                action_type="TRIGGER_ALARM",
                parameters={"urgency": "high"},
                confidence_score=0.99
            )
        
        return ActionOutput(
            action_type="NO_ACTION",
            parameters={},
            confidence_score=1.0
        )


class CognitiveBrain:
    """
    Main Artificial Cognitive Brain instance. Integrates perception, memory, 
    and reasoning components to execute cognitive cycles.
    """
    def __init__(self, brain_id: str = "ACB-Node-01"):
        self.brain_id = brain_id
        self.state = CognitiveState()
        self.perception = PerceptionSubsystem()
        self.memory = MemorySubsystem()
        self.reasoning = ReasoningSubsystem()
        
        logger.info(f"Artificial Cognitive Brain [{self.brain_id}] initialized successfully.")

    def cognitive_cycle(self, raw_input: SensoryInput) -> ActionOutput:
        """
        Executes a single perception-thought-action loop.
        
        1. Perception: Parse and feature-extract sensory inputs.
        2. Memory Update: Merge new information with internal state.
        3. Reasoning: Evaluate decisions and select the best action output.
        """
        logger.info(f"Initiating cognitive cycle for input: {raw_input.source}")

        # Step 1: Perception
        perceived_data = self.perception.process(raw_input)

        # Step 2: Memory Integration
        self.state.working_memory["latest_perception"] = perceived_data
        self.memory.store_episode(perceived_data)

        # Step 3: Reasoning & Action Selection
        action = self.reasoning.evaluate(self.state, perceived_data)

        logger.info(f"Cognitive cycle complete. Selected action: {action.action_type}")
        return action


if __name__ == "__main__":
    # Example execution flow
    logging.basicConfig(level=logging.INFO)

    # Initialize the brain
    brain = CognitiveBrain(brain_id="Cognitive Core Alpha")

    # Simulate incoming sensory data
    sample_input = SensoryInput(
        source="camera_sensor_01",
        payload={"intent": "query", "payload_data": "What is the status?"}
    )

    # Execute cognitive processing cycle
    output_action = brain.cognitive_cycle(sample_input)
    print(f"\nExecution Result: {output_action}")
