"""
Neural_controller.py
--------------------
Core control unit for the Artificial Cognitive Brain repository.
Manages input processing, cognitive state integration, memory retrieval,
and action selection using dynamic neural routing.
"""

import logging
import numpy as np
from typing import Dict, Any, List, Optional

logging.basicConfi(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class NeuralController:
    """
    The NeuralController coordinates interactions between sensory processing,
    working memory, long-term memory access, and executive action selection.
    """

    def __init__(self, input_dim: int = 128, hidden_dim: int = 256, action_dim: int = 64):
        """
        Initialize the Cognitive Neural Controller.

        :param input_dim: Dimensionality of incoming sensory features.
        :param hidden_dim: Internal state/working memory representation size.
        :param action_dim: Dimensionality of the output action space.
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim

        # Simulated neural weights for cognitive state transformations
        self.weights_perception = np.random.randn(input_dim, hidden_dim) * 0.01
        self.weights_recurrence = np.random.randn(hidden_dim, hidden_dim) * 0.01
        self.weights_action = np.random.randn(hidden_dim, action_dim) * 0.01

        # Current internal cognitive state (Working Memory Representation)
        self.cognitive_state = np.zeros((1, hidden_dim))
        
        # Episodic memory buffer simulation
        self.episodic_memory: List[Dict[str, Any]] = []

        logging.info("Neural Controller initialized successfully.")

    def _activate(self, x: np.ndarray) -> np.ndarray:
        """Hyperbolic tangent activation function for cognitive state transitions."""
        return np.tanh(x)

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Softmax activation for action distribution mapping."""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def process_sensory_input(self, sensory_vector: np.ndarray) -> np.ndarray:
        """
        Processes raw sensory inputs and integrates them with the current cognitive state.

        :param sensory_vector: Numpy array representing encoded sensory data.
        :return: Updated internal cognitive state.
        """
        if sensory_vector.shape[-1] != self.input_dim:
            raise ValueError(f"Expected input dimension {self.input_dim}, got {sensory_vector.shape[-1]}")

        # Compute updated state: Current Input + Previous Internal State
        input_projection = np.dot(sensory_vector, self.weights_perception)
        state_projection = np.dot(self.cognitive_state, self.weights_recurrence)
        
        self.cognitive_state = self._activate(input_projection + state_projection)
        logging.debug("Cognitive state updated via sensory integration.")
        
        return self.cognitive_state

    def query_memory_bank(self, query_key: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Retrieves relevant memory traces based on cosine similarity with current cognitive state.

        :param query_key: Vector representation of the memory query.
        :return: Best matching memory dict or None.
        """
        if not self.episodic_memory:
            return None

        best_match = None
        highest_similarity = -1.0

        for memory in self.episodic_memory:
            mem_vector = memory["state"]
            similarity = np.dot(query_key, mem_vector.T) / (
                np.linalg.norm(query_key) * np.linalg.norm(mem_vector) + 1e-8
            )
            if similarity > highest_similarity:
                highest_similarity = float(similarity)
                best_match = memory

        logging.info(f"Memory queried. Best match confidence: {highest_similarity:.4f}")
        return best_match

    def consolidate_memory(self, context_tag: str, payload: Any) -> None:
        """
        Stores current cognitive state and context payload into long-term memory.

        :param context_tag: Descriptive tag for the cognitive memory.
        :param payload: Associated metadata or execution log.
        """
        memory_entry = {
            "tag": context_tag,
            "state": self.cognitive_state.copy(),
            "payload": payload
        }
        self.episodic_memory.append(memory_entry)
        logging.info(f"Consolidated new memory trace with tag: '{context_tag}'.")

    def execute_action(self) -> np.ndarray:
        """
        Generates an executive action payload based on current cognitive internal state.

        :return: Probability distribution over potential cognitive actions.
        """
        raw_action_logits = np.dot(self.cognitive_state, self.weights_action)
        action_probabilities = self._softmax(raw_action_logits)
        
        logging.info("Executive action selection executed.")
        return action_probabilities

    def cognitive_step(self, sensory_input: np.ndarray, context_tag: str = "general_step") -> Dict[str, Any]:
        """
        Full cognitive cycle loop: Perception -> Memory Processing -> Action Generation.

        :param sensory_input: Input vector from environment/sensors.
        :param context_tag: Optional metadata tag for memory consolidation.
        :return: Dictionary containing the updated state, action, and retrieved context.
        """
        # Step 1: Perception & State Update
        current_state = self.process_sensory_input(sensory_input)

        # Step 2: Memory Retrieval
        retrieved_memory = self.query_memory_bank(current_state)

        # Step 3: Decision Making / Action Execution
        action = self.execute_action()

        # Step 4: Consolidation
        self.consolidate_memory(context_tag=context_tag, payload={"action_taken": np.argmax(action)})

        return {
            "updated_state": current_state,
            "action_distribution": action,
            "selected_action_index": int(np.argmax(action)),
            "retrieved_memory": retrieved_memory
        }

class PowerManagement:
    def __init__(self, input_dim: int, x: float, y: float):
        self.input_dim = input_dim
        self.x = float(x)
        self.y = float(y)
        
        self.playground_movement = int(self.x)
        self.powerful_confrontations = int(self.y)
        
        self.weights = np.random.randn(self.input_dim, 4) * 0.01
        self.bias = np.zeros((1, 4))

    def step_controller(self, x: float, y: float, z: float, w: float) -> np.ndarray:
        real_action_input = np.array([[x, y, z, w]], dtype=np.float32)
        
        if real_action_input.shape[1] != self.input_dim:
            raise ValueError(f"Input dimension mismatch. Expected {self.input_dim}, got {real_action_input.shape[1]}")
            
        action_logits = np.dot(real_action_input, self.weights) + self.bias
        action_probabilities = np.exp(action_logits - np.max(action_logits, axis=-1, keepdims=True))
        action_probabilities /= np.sum(action_probabilities, axis=-1, keepdims=True)
        
        return action_probabilities


class NeuralController:
    def __init__(self, input_dim: int = 128, hidden_dim: int = 256, action_dim: int = 10):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim

        self.power_manager = PowerManagement(input_dim=4, x=1.0, y=2.0)

        self.weights_perception = np.random.randn(input_dim, hidden_dim) * 0.01
        self.weights_recurrence = np.random.randn(hidden_dim, hidden_dim) * 0.01
        self.weights_action = np.random.randn(hidden_dim, action_dim) * 0.01

        self.cognitive_state = np.zeros((1, hidden_dim))
        self.episodic_memory = []

    def _activate(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x)

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def process_sensory_input(self, sensory_vector: np.ndarray) -> np.ndarray:
        if sensory_vector.shape[-1] != self.input_dim:
            raise ValueError(f"Expected input dimension {self.input_dim}, got {sensory_vector.shape[-1]}")

        input_projection = np.dot(sensory_vector, self.weights_perception)
        state_projection = np.dot(self.cognitive_state, self.weights_recurrence)
        
        self.cognitive_state = self._activate(input_projection + state_projection)
        return self.cognitive_state

    def query_memory_bank(self, query_key: np.ndarray) -> Optional[Dict[str, Any]]:
        if not self.episodic_memory:
            return None

        best_match = None
        highest_similarity = -1.0

        for memory in self.episodic_memory:
            mem_vector = memory["state"]
            similarity = np.dot(query_key, mem_vector.T) / (
                np.linalg.norm(query_key) * np.linalg.norm(mem_vector) + 1e-8
            )
            if similarity > highest_similarity:
                highest_similarity = float(similarity)
                best_match = memory

        return best_match

    def consolidate_memory(self, context_tag: str, payload: Any) -> None:
        memory_entry = {
            "tag": context_tag,
            "state": self.cognitive_state.copy(),
            "payload": payload
        }
        self.episodic_memory.append(memory_entry)

    def execute_action(self) -> np.ndarray:
        raw_action_logits = np.dot(self.cognitive_state, self.weights_action)
        return self._softmax(raw_action_logits)

    def cognitive_step(self, sensory_input: np.ndarray, context_tag: str = "general_step") -> Dict[str, Any]:
        current_state = self.process_sensory_input(sensory_input)
        
        power_signals = self.power_manager.step_controller(
            x=float(sensory_input[0, 0]),
            y=float(sensory_input[0, 1]),
            z=float(sensory_input[0, 2]),
            w=float(sensory_input[0, 3])
        )

        retrieved_memory = self.query_memory_bank(current_state)
        action = self.execute_action()

        self.consolidate_memory(
            context_tag=context_tag, 
            payload={
                "action_taken": int(np.argmax(action)),
                "power_distribution": power_signals
            }
        )

        return {
            "updated_state": current_state,
            "power_signals": power_signals,
            "action_distribution": action,
            "selected_action_index": int(np.argmax(action)),
            "retrieved_memory": retrieved_memory
        }


if __name__ == "__main__":
    controller = NeuralController(input_dim=128, hidden_dim=256, action_dim=10)
    dummy_sensory_data = np.random.randn(1, 128)

    output = controller.cognitive_step(dummy_sensory_data, context_tag="initial_observation")
    
    print("\n--- Cognitive Step Results ---")
    print(f"Action Index Selected: {output['selected_action_index']}")
    print(f"Action Probability Vector Shape: {output['action_distribution'].shape}")
    print(f"Power Signals Shape: {output['power_signals'].shape}")
    print(f"Memory Buffer Size: {len(controller.episodic_memory)}")
