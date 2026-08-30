"""
Artificial Cognitive Brain - Cognitive Cycle Module
===================================================
This module defines the execution pipeline for a single cognitive cycle,
simulating the sequence of perception, memory retrieval, reasoning, 
planning, and action execution.
"""

import logging
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

# Configure logger for cognitive cycle events
logger = logging.getLogger(__name__)


@dataclass
class CognitiveState:
    """Represents the internal state of the agent during one cognitive cycle."""
    cycle_id: int
    raw_inputs: Dict[str, Any] = field(default_factory=dict)
    perceptual_data: Dict[str, Any] = field(default_factory=dict)
    retrieved_memories: List[Dict[str, Any]] = field(default_factory=list)
    reasoning_output: Dict[str, Any] = field(default_factory=dict)
    selected_action: Optional[Dict[str, Any]] = None
    timestamp: float = field(default_factory=time.time)


class CognitiveCycle:
    """
    Manages the sequential processing steps of the artificial brain's 
    perception-action loop.
    """

    def __init__(self, cycle_id: int):
        self.cycle_id = cycle_id
        self.state = CognitiveState(cycle_id=cycle_id)

    def perceive(self, raw_sensory_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Step 1: Perception Phase
        Processes raw sensory environmental input into structured perceptual features.
        """
        logger.info(f"[Cycle {self.cycle_id}] Phase 1: Perceiving sensory data...")
        self.state.raw_inputs = raw_sensory_input
        
        # Feature extraction and filter simulation
        perceptual_features = {
            "processed_at": time.time(),
            "filtered_signals": raw_sensory_input.get("signals", []),
            "environmental_context": raw_sensory_input.get("context", "default")
        }
        
        self.state.perceptual_data = perceptual_features
        return perceptual_features

    def recall_memory(self) -> List[Dict[str, Any]]:
        """
        Step 2: Memory Retrieval Phase
        Queries episodic and semantic memory components using perceptual features.
        """
        logger.info(f"[Cycle {self.cycle_id}] Phase 2: Accessing memory systems...")
        
        # Simulating context-driven memory access
        context = self.state.perceptual_data.get("environmental_context", "")
        memories = [
            {"memory_id": "mem_001", "relevance": 0.95, "content": f"Prior experience in {context}"},
            {"memory_id": "mem_002", "relevance": 0.82, "content": "General domain knowledge"}
        ]
        
        self.state.retrieved_memories = memories
        return memories

    def reason_and_plan(self) -> Dict[str, Any]:
        """
        Step 3: Reasoning & Planning Phase
        Combines current perception and retrieved memories to generate decisions.
        """
        logger.info(f"[Cycle {self.cycle_id}] Phase 3: Synthesizing reasoning & strategy...")
        
        # Evaluates goals, constraints, and memory context
        deductions = {
            "primary_goal": "optimize_task_execution",
            "evaluated_options": ["action_A", "action_B"],
            "chosen_strategy": "action_A",
            "confidence_score": 0.91
        }
        
        self.state.reasoning_output = deductions
        return deductions

    def select_action(self) -> Dict[str, Any]:
        """
        Step 4: Action Selection Phase
        Formulates an executable command based on the reasoning output.
        """
        logger.info(f"[Cycle {self.cycle_id}] Phase 4: Selecting motor/digital action...")
        
        strategy = self.state.reasoning_output.get("chosen_strategy", "idle")
        action = {
            "command": strategy,
            "parameters": {"intensity": 1.0, "duration": 250},
            "status": "ready_for_execution"
        }
        
        self.state.selected_action = action
        return action

    def execute_cycle(self, raw_sensory_input: Dict[str, Any]) -> CognitiveState:
        """
        Executes the full perception-action cognitive loop in sequence.
        """
        start_time = time.time()
        logger.info(f"--- Starting Cognitive Cycle #{self.cycle_id} ---")

        self.perceive(raw_sensory_input)
        self.recall_memory()
        self.reason_and_plan()
        self.select_action()

        elapsed = time.time() - start_time
        logger.info(f"--- Cycle #{self.cycle_id} completed in {elapsed:.4f}s ---")
        return self.state


if __name__ == "__main__":
    # Example execution of a single cognitive loop
    logging.basicConfig(level=logging.INFO)
    
    sample_input = {
        "signals": [0.12, 0.85, 0.43],
        "context": "high_priority_task"
    }
    
    engine_cycle = CognitiveCycle(cycle_id=1)
    final_state = engine_cycle.execute_cycle(sample_input)
    
    print("\nFinal Cognitive State Output:")
    print(f"Action Command: {final_state.selected_action['command']}")
    print(f"Confidence Score: {final_state.reasoning_output['confidence_score']}")
