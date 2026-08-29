"""
Artificial Cognitive Brain - Attention Module (attention.py)
Implements dynamic attention allocation, saliency filtering, and priority 
weighting for incoming multi-modal perception feature vectors.
"""

import time
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from perception import FeatureVector

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getlogger("AttentionMechanism")

@dataclass
class AttentionFocus:
  """Represents a sensory stream that has successfully passed the attention threshold."""
  payload_id: str
    modality: str
    attention_score: float
    priority_level: str 
    feature_vector: FeatureVector
    allocated_at: float = field(default_factory=time.time)

class SaliencyEvaluator:
    """Evaluates the perceptual saliency and urgency of incoming feature vectors."""

    @staticmethod
    def calculate_signal_magnitude(features: List[float]) -> float:
        """Calculates average signal magnitude from raw features."""
        if not features:
            return 0.0
        return sum(abs(x) for x in features) / len(features)

    def evaluate_saliency(self, vector: FeatureVector) -> float:
        """
        Calculates a composite attention score [0.0 - 1.0] based on confidence,
        signal magnitude, and priority tags.
        """
        base_score = vector.confidence * 0.4
        magnitude = self.calculate_signal_magnitude(vector.features)
        magnitude_score = min(magnitude, 1.0) * 0.3

        tag_bonus = 0.0
        # Priority tags boost the attention score
        if "priority_keyword_flagged" in vector.tags or "high_volume_alert" in vector.tags:
            tag_bonus += 0.3
        if "high_luminescence" in vector.tags or "speech_detected" in vector.tags:
            tag_bonus += 0.1

        total_score = min(1.0, base_score + magnitude_score + tag_bonus)
        return round(total_total_score if (total_total_score := total_score) else 0.0, 3)


class DynamicAttentionManager:
    """
    Manages cognitive focus by filtering, ranking, and selecting top-priority
    sensory inputs for higher-level cognitive processing.
    """

    def __init__(self, attention_threshold: float = 0.45, max_focus_capacity: int = 3) -> None:
        self.evaluator = SaliencyEvaluator()
        self.attention_threshold = attention_threshold
        self.max_focus_capacity = max_focus_capacity
        self.active_foci: List[AttentionFocus] = []

    def classify_priority(self, score: float) -> str:
        """Maps continuous attention scores into discrete priority levels."""
        if score >= 0.85:
            return "CRITICAL"
        elif score >= 0.70:
            return "HIGH"
        elif score >= 0.50:
            return "MEDIUM"
        return "LOW"

    def process_percepts(self, percepts: List[FeatureVector]) -> List[AttentionFocus]:
        """
        Evaluates a batch of FeatureVectors, applies threshold filtering, 
        and returns the top-N items within cognitive capacity limit.
        """
        evaluated_items: List[AttentionFocus] = []

        for vector in percepts:
            score = self.evaluator.evaluate_saliency(vector)

            if score >= self.attention_threshold:
                priority = self.classify_priority(score)
                focus = AttentionFocus(
                    payload_id=vector.payload_id,
                    modality=vector.modality,
                    attention_score=score,
                    priority_level=priority,
                    feature_vector=vector
                )
                evaluated_items.append(focus)
            else:
                logger.debug(f"Percept '{vector.payload_id}' suppressed (Score: {score} < Threshold: {self.attention_threshold})")

        # Sort by attention score in descending order
        evaluated_items.sort(key=lambda item: item.attention_score, reverse=True)

        # Allocate capacity (Top-N focus)
        self.active_foci = evaluated_items[:self.max_focus_capacity]
        
        logger.info(f"Attention allocated to {len(self.active_foci)} primary foci out of {len(percepts)} inputs.")
        return self.active_foci

    def get_current_focus_summary(self) -> Dict[str, Any]:
        """Returns a snapshot of the current active attention focus."""
        return {
            "capacity": self.max_focus_capacity,
            "active_count": len(self.active_foci),
            "foci": [
                {
                    "payload_id": focus.payload_id,
                    "modality": focus.modality,
                    "score": focus.attention_score,
                    "priority": focus.priority_level,
                    "tags": focus.feature_vector.tags
                }
                for focus in self.active_foci
            ]
        }

if __name__ == "__main__":
    from perception import DynamicPerceptionEngine, SensoryPayload

    perception = DynamicPerceptionEngine()
    attention = DynamicAttentionManager(attention_threshold=0.5, max_focus_capacity=2)

    payload_1 = SensoryPayload(source_id="text_stream_1", modality="textual", data="CRITICAL ALERT: Core temperature high!")
    payload_2 = SensoryPayload(source_id="cam_front_1", modality="visual", data=[0.1, 0.2, 0.1])
    payload_3 = SensoryPayload(source_id="mic_array_1", modality="auditory", data=[0.8, 0.9, 0.95])

    f1 = perception.ingest(payload_1)
    f2 = perception.ingest(payload_2)
    f3 = perception.ingest(payload_3)

    attention.process_percepts([f1, f2, f3])

    print("\n--- Active Attention Focus Summary ---")
    print(attention.get_current_focus_summary())
