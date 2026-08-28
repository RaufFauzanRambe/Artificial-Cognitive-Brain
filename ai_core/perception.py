"""
Artificial Cognitive Brain - Perception Module (perception.py)
Handles multi-modal sensory input ingestion, filtering, feature extraction,
and translation into unified cognitive representations.
"""

import time
import logging
from typing import Dict, Any, List, Union, Optional
from dataclasses import dataclass, field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PerceptionEngine")


@dataclass
class SensoryPayload:
    """Encapsulates raw multi-modal sensory data."""
    source_id: str
    modality: str  # e.g., 'visual', 'auditory', 'textual', 'telemetry'
    timestamp: float = field(default_factory=time.time)
    data: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FeatureVector:
    """Represents processed numerical/symbolic features ready for cognitive integration."""
    payload_id: str
    modality: str
    confidence: float
    features: List[float]
    tags: List[str]
    processed_at: float = field(default_factory=time.time)


class NoiseFilter:
    """Pre-processes and sanitizes incoming sensory signals to reduce latency and data noise."""

    @staticmethod
    def sanitize_text(text: str) -> str:
        """Removes extraneous whitespace and trivial noise from textual input."""
        return " ".join(text.strip().split())

    @staticmethod
    def normalize_signals(signal_vector: List[float]) -> List[float]:
        """Normalizes numerical signal arrays into a standard range [0, 1]."""
        if not signal_vector:
            return []
        min_val, max_val = min(signal_vector), max(signal_vector)
        if max_val == min_val:
            return [0.5 for _ in signal_vector]
        return [(x - min_val) / (max_val - min_val) for x in signal_vector]


class VisualProcessor:
    """Sub-system for visual data analysis and feature extraction."""

    def process(self, payload: SensoryPayload) -> FeatureVector:
        logger.info(f"Processing visual stream from source: {payload.source_id}")
        # Simulated visual feature extraction (e.g., bounding boxes, embeddings)
        raw_signals = payload.data if isinstance(payload.data, list) else [0.2, 0.8, 0.4, 0.9]
        normalized = NoiseFilter.normalize_signals(raw_signals)
        
        return FeatureVector(
            payload_id=payload.source_id,
            modality="visual",
            confidence=0.92,
            features=normalized,
            tags=["object_detected", "spatial_depth_mapped"]
        )


class AuditoryProcessor:
    """Sub-system for acoustic signal processing and sound classification."""

    def process(self, payload: SensoryPayload) -> FeatureVector:
        logger.info(f"Processing auditory signal from source: {payload.source_id}")
        # Simulated spectral analysis and pitch detection
        raw_signals = payload.data if isinstance(payload.data, list) else [0.1, 0.3, 0.7]
        normalized = NoiseFilter.normalize_signals(raw_signals)

        return FeatureVector(
            payload_id=payload.source_id,
            modality="auditory",
            confidence=0.88,
            features=normalized,
            tags=["frequency_extracted", "speech_detected"]
        )


class TextualProcessor:
    """Sub-system for natural language understanding and semantic parsing."""

    def process(self, payload: SensoryPayload) -> FeatureVector:
        logger.info(f"Processing textual input from source: {payload.source_id}")
        raw_text = str(payload.data)
        clean_text = NoiseFilter.sanitize_text(raw_text)
        
        # Simulated semantic vector generation (embedding proxy)
        text_length = len(clean_text)
        word_count = len(clean_text.split())
        features = [float(text_length), float(word_count), 0.75]

        return FeatureVector(
            payload_id=payload.source_id,
            modality="textual",
            confidence=0.95,
            features=features,
            tags=["semantic_parsed", f"word_count:{word_count}"]
        )


class DynamicPerceptionEngine:
    """
    Main Perception Engine for the Artificial Cognitive Brain.
    Orchestrates multi-sensory processors and unifies perception results into a single frame.
    """

    def __init__(self) -> None:
        self.visual_processor = VisualProcessor()
        self.auditory_processor = AuditoryProcessor()
        self.textual_processor = TextualProcessor()
        self.perception_history: List[FeatureVector] = []

    def ingest(self, payload: SensoryPayload) -> FeatureVector:
        """Routes payload to the appropriate sensory processor based on modality."""
        modality = payload.modality.lower()
        
        if modality == "visual":
            vector = self.visual_processor.process(payload)
        elif modality == "auditory":
            vector = self.auditory_processor.process(payload)
        elif modality == "textual":
            vector = self.textual_processor.process(payload)
        else:
            logger.warning(f"Unknown modality '{modality}'. Falling back to raw normalization.")
            raw_data = payload.data if isinstance(payload.data, list) else [0.0]
            vector = FeatureVector(
                payload_id=payload.source_id,
                modality="generic",
                confidence=0.50,
                features=NoiseFilter.normalize_signals(raw_data),
                tags=["uncalibrated_modality"]
            )

        self.perception_history.append(vector)
        return vector

    def get_unified_perception_frame(self, limit: int = 5) -> Dict[str, Any]:
        """Synthesizes recent sensory features into a cohesive perception frame."""
        recent_frames = self.perception_history[-limit:]
        return {
            "total_percepts": len(recent_frames),
            "modalities_present": list({fv.modality for fv in recent_frames}),
            "avg_confidence": sum(fv.confidence for fv in recent_frames) / max(len(recent_frames), 1),
            "percept_summary": [
                {"id": fv.payload_id, "modality": fv.modality, "tags": fv.tags}
                for fv in recent_frames
            ]
        }


# Quick Demonstration Execution
if __name__ == "__main__":
    engine = DynamicPerceptionEngine()

    # Simulated Inputs
    sample_text = SensoryPayload(source_id="text_stream_1", modality="textual", data="System status is operational. All parameters normal.")
    sample_visual = SensoryPayload(source_id="cam_front_1", modality="visual", data=[12.5, 45.0, 100.2, 5.8])

    res_text = engine.ingest(sample_text)
    res_visual = engine.ingest(sample_visual)

    print("\n--- Unified Perception Frame ---")
    print(engine.get_unified_perception_frame())
