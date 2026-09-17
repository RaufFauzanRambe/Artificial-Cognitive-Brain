"""
Artificial Cognitive Brain - Attention Model Module.

This module implements the Attention Model Architecture for the Artificial Cognitive Brain.
Modeled after landmark cognitive psychology and neuroscience frameworks (e.g., Posner's Attentional
Networks, Treisman's Feature Integration Theory, Corbetta & Shulman's Dual-System Model, and 
Desimone & Duncan's Biased Competition Model), this module regulates top-down (goal-driven, endogenous) 
and bottom-up (stimulus-driven, exogenous) attentional allocation.

Key Features:
    - AttentionalItem: Converts multi-modal sensory input signals into structured candidate items.
    - BiasedCompetitionEngine: Integrates endogenous top-down goals, exogenous sensory salience, 
      and Inhibition of Return (IOR) into a dynamic priority map.
    - AttentionalFocus: Tracks the active spotlight of attention, spotlight intensity, and target dwell duration.
    - AttentionModelSystem: Primary module coordinating attentional shifts, spotlight tracking, and workspace gating.

Author: Artificial Cognitive Brain Team
License: MIT
Version: 0.1.0
"""

import time
import math
import logging
from typing import Dict, Any, List, Optional, Tuple, Set, Union
from dataclasses import dataclass, field

from cognitive_models import BaseCognitiveModel

# Configure module logger
logger = logging.getLogger(__name__)


@dataclass
class AttentionalItem:
    """
    Represents an individual sensory object, feature vector, or cognitive candidate competing for attention.

    Attributes:
        item_id (str): Unique string identifier for the candidate item.
        modality (str): Sensory modality ('visual', 'auditory', 'tactile', 'conceptual').
        features (Dict[str, float]): Normalized feature dimensions (e.g., {'intensity': 0.8, 'contrast': 0.9}).
        salience_score (float): Bottom-up physical salience computed from raw feature properties [0.0, 1.0].
        goal_relevance (float): Top-down task-relevance bias assigned by current goal priorities [0.0, 1.0].
        combined_priority (float): Final dynamic competition score calculated via biased competition.
        location (Dict[str, float]): Spatial coordinates or relative position mapping.
        timestamp (float): UNIX creation or detection timestamp.
        metadata (Dict[str, Any]): Additional domain-specific key-value pairs.
    """
    item_id: str
    modality: str = "visual"
    features: Dict[str, float] = field(default_factory=dict)
    salience_score: float = 0.0
    goal_relevance: float = 0.0
    combined_priority: float = 0.0
    location: Dict[str, float] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def calculate_bottom_up_salience(self, feature_weights: Optional[Dict[str, float]] = None) -> float:
        """
        Computes exogenous bottom-up salience based on weighted intrinsic feature intensities.

        Args:
            feature_weights (Optional[Dict[str, float]]): Modality feature importance weights.

        Returns:
            float: Computed raw salience score in range [0.0, 1.0].
        """
        if not self.features:
            self.salience_score = 0.0
            return 0.0

        weights = feature_weights or {
            "intensity": 0.3,
            "contrast": 0.3,
            "novelty": 0.2,
            "motion": 0.2
        }

        total_weight = 0.0
        weighted_sum = 0.0

        for feat_name, feat_val in self.features.items():
            w = weights.get(feat_name, 0.1)
            weighted_sum += feat_val * w
            total_weight += w

        self.salience_score = min(1.0, max(0.0, weighted_sum / max(total_weight, 1e-5)))
        return self.salience_score


@dataclass
class AttentionalFocus:
    """
    Data structure defining the current state of the "Attentional Spotlight".

    Attributes:
        focused_item_id (Optional[str]): Item ID currently residing within the attentional focus.
        intensity (float): Current focus strength or gain factor [0.0, 1.0].
        spotlight_width (float): Dynamic spatial or conceptual width of attentional filter.
        dwell_time (float): Time duration (in seconds) focus has remained on the current item.
        last_fired_timestamp (float): UNIX timestamp when focus last shifted.
    """
    focused_item_id: Optional[str] = None
    intensity: float = 1.0
    spotlight_width: float = 1.0
    dwell_time: float = 0.0
    last_fired_timestamp: float = field(default_factory=time.time)

    def update_dwell(self, current_time: Optional[float] = None) -> float:
        """Updates and returns the dwell duration on the current focal target."""
        now = current_time or time.time()
        self.dwell_time = max(0.0, now - self.last_fired_timestamp)
        return self.dwell_time

    def shift_to(self, item_id: Optional[str]) -> None:
        """Shifts attentional spotlight to a new target and resets dwell parameters."""
        self.focused_item_id = item_id
        self.dwell_time = 0.0
        self.last_fired_timestamp = time.time()
        logger.info(f"Attentional Focus shifted to item: '{item_id}'")


class BiasedCompetitionEngine:
    """
    Implements Desimone & Duncan's Biased Competition Model of Attention.
    
    Integrates bottom-up physical salience (exogenous) and top-down goal relevance (endogenous)
    with Inhibition of Return (IOR) to prevent attentional perseveration/locking.
    """

    def __init__(
        self,
        top_down_weight: float = 0.6,
        bottom_up_weight: float = 0.4,
        ior_decay_rate: float = 0.05
    ) -> None:
        """
        Initializes the Biased Competition Engine.

        Args:
            top_down_weight (float): Relative bias factor given to goal-driven endogenous targets [0.0, 1.0].
            bottom_up_weight (float): Relative bias factor given to physical sensory salience [0.0, 1.0].
            ior_decay_rate (float): Decay rate per second for Inhibition of Return (IOR) memory tags.
        """
        self.top_down_weight = top_down_weight
        self.bottom_up_weight = bottom_up_weight
        self.ior_decay_rate = ior_decay_rate
        
        # Inhibition of Return (IOR) map: item_id -> (inhibition_level, timestamp)
        self.ior_map: Dict[str, Tuple[float, float]] = {}

    def apply_inhibition_of_return(self, item_id: str, inhibition_strength: float = 0.8) -> None:
        """
        Applies Inhibition of Return (IOR) to a recently focused item to prevent immediate re-selection.

        Args:
            item_id (str): Item ID to inhibit.
            inhibition_strength (float): Degree of negative priority penalty applied.
        """
        self.ior_map[item_id] = (inhibition_strength, time.time())
        logger.debug(f"Applied Inhibition of Return (IOR) to item '{item_id}' (Strength: {inhibition_strength:.2f})")

    def decay_ior(self, current_time: Optional[float] = None) -> None:
        """Decays active IOR tags over time."""
        now = current_time or time.time()
        expired_keys = []

        for item_id, (strength, ts) in self.ior_map.items():
            elapsed = max(0.0, now - ts)
            updated_strength = strength * math.exp(-self.ior_decay_rate * elapsed)
            
            if updated_strength < 0.05:
                expired_keys.append(item_id)
            else:
                self.ior_map[item_id] = (updated_strength, ts)

        for k in expired_keys:
            del self.ior_map[k]

    def compute_priority_map(
        self,
        candidates: List[AttentionalItem],
        active_goals: Dict[str, float]
    ) -> List[AttentionalItem]:
        """
        Calculates biased competition scores across all candidate sensory items.

        Formula:
            Priority = (W_td * Goal_Relevance) + (W_bu * Salience) - IOR_Penalty

        Args:
            candidates (List[AttentionalItem]): Candidate items competing for focus.
            active_goals (Dict[str, float]): Active task targets and feature bias expectations.

        Returns:
            List[AttentionalItem]: Sorted candidates ordered by combined priority descending.
        """
        self.decay_ior()

        for item in candidates:
            # 1. Compute Bottom-Up Salience
            salience = item.calculate_bottom_up_salience()

            # 2. Compute Top-Down Goal Relevance
            relevance = 0.0
            if active_goals:
                for target_feat, goal_bias in active_goals.items():
                    if target_feat in item.features:
                        relevance += item.features[target_feat] * goal_bias
                    elif target_feat == item.modality or target_feat in item.metadata.get("tags", []):
                        relevance += goal_bias
                item.goal_relevance = min(1.0, relevance)
            else:
                item.goal_relevance = 0.0

            # 3. Retrieve IOR Penalty
            ior_penalty = 0.0
            if item.item_id in self.ior_map:
                ior_penalty, _ = self.ior_map[item.item_id]

            # 4. Integrate Biased Priority Score
            raw_priority = (self.top_down_weight * item.goal_relevance) + (self.bottom_up_weight * salience)
            item.combined_priority = max(0.0, raw_priority - ior_penalty)

        # Sort descending by priority
        candidates.sort(key=lambda x: x.combined_priority, reverse=True)
        return candidates


class AttentionModelSystem(BaseCognitiveModel):
    """
    Attention Model Cognitive System for the Artificial Cognitive Brain.

    Coordinates sensory filtering, biased competition, top-down goal orientation, 
    spotlight selection, and workspace gating.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initializes the Attention Model System.

        Args:
            config (Optional[Dict[str, Any]]): Configuration dictionary for weights, 
                                               dwell thresholds, and spotlight limits.
        """
        super().__init__(model_name="AttentionModelSystem", config=config)

        cfg = self.config or {}
        top_down = cfg.get("top_down_weight", 0.6)
        bottom_up = cfg.get("bottom_up_weight", 0.4)
        ior_decay = cfg.get("ior_decay_rate", 0.05)
        
        self.max_dwell_time = cfg.get("max_dwell_time", 3.0)  # Forced shift threshold in seconds
        self.competition_engine = BiasedCompetitionEngine(
            top_down_weight=top_down,
            bottom_up_weight=bottom_up,
            ior_decay_rate=ior_decay
        )

        self.spotlight = AttentionalFocus()
        self.active_goals: Dict[str, float] = {}  # target_feature -> bias_weight

    def set_active_goals(self, goal_biases: Dict[str, float]) -> None:
        """
        Configures top-down endogenous goal biases (e.g., {'red': 0.8, 'audio': 0.5}).

        Args:
            goal_biases (Dict[str, float]): Key-value pairs mapping feature keys to target biases.
        """
        self.active_goals = goal_biases
        logger.info(f"Top-Down Goal Biases updated: {self.active_goals}")

    def process_sensory_candidates(self, raw_candidates: List[Dict[str, Any]]) -> AttentionalFocus:
        """
        Evaluates competing sensory candidates, runs biased competition, and updates spotlight focus.

        Args:
            raw_candidates (List[Dict[str, Any]]): Candidate items from perceptual inputs.

        Returns:
            AttentionalFocus: Updated attentional focus spotlight state.
        """
        # Convert raw inputs to AttentionalItem instances
        items: List[AttentionalItem] = []
        for raw in raw_candidates:
            item = AttentionalItem(
                item_id=raw.get("id", f"item_{len(items)}"),
                modality=raw.get("modality", "visual"),
                features=raw.get("features", {}),
                location=raw.get("location", {}),
                metadata=raw.get("metadata", {})
            )
            items.append(item)

        if not items:
            return self.spotlight

        # Run Biased Competition
        ranked_candidates = self.competition_engine.compute_priority_map(items, self.active_goals)
        winning_candidate = ranked_candidates[0]

        # Check dwell time constraint on current focus
        self.spotlight.update_dwell()
        if (
            self.spotlight.focused_item_id == winning_candidate.item_id and
            self.spotlight.dwell_time > self.max_dwell_time
        ):
            # Apply IOR to force attentional shift away from perseverated target
            logger.info(f"Max dwell time ({self.max_dwell_time}s) exceeded for '{self.spotlight.focused_item_id}'. Applying IOR.")
            self.competition_engine.apply_inhibition_of_return(self.spotlight.focused_item_id, inhibition_strength=0.9)
            
            # Re-compete excluding or penalizing the perseverated item
            ranked_candidates = self.competition_engine.compute_priority_map(items, self.active_goals)
            winning_candidate = ranked_candidates[0]

        # Shift focus if winner changes or initial focus
        if self.spotlight.focused_item_id != winning_candidate.item_id:
            if self.spotlight.focused_item_id:
                # Apply mild IOR to previous focus
                self.competition_engine.apply_inhibition_of_return(self.spotlight.focused_item_id, inhibition_strength=0.5)
            self.spotlight.shift_to(winning_candidate.item_id)

        self.spotlight.intensity = round(winning_candidate.combined_priority, 3)
        return self.spotlight

    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Cognitive Architecture step handler.

        Interacts with Global Workspace state to read incoming perceptual inputs and top-down goal commands.

        Args:
            inputs (Dict[str, Any]): Global workspace state snapshot.

        Returns:
            Dict[str, Any]: Attentional allocation status and gated spotlight output.
        """
        # 1. Update top-down goals if commanded by Executive Control or Workspace
        att_command = inputs.get("attention_command", {})
        if isinstance(att_command, dict) and "set_goals" in att_command:
            self.set_active_goals(att_command["set_goals"])

        # 2. Extract perceptual inputs / sensory candidates
        perceptual_data = inputs.get("sensory_input", {})
        raw_candidates = []

        if isinstance(perceptual_data, dict):
            raw_candidates = perceptual_data.get("candidates", [])
        elif isinstance(perceptual_data, list):
            raw_candidates = perceptual_data

        # 3. Compute dynamic spotlight focus
        spotlight_state = self.process_sensory_candidates(raw_candidates)

        return {
            "focused_item_id": spotlight_state.focused_item_id,
            "focus_intensity": spotlight_state.intensity,
            "dwell_time": round(spotlight_state.dwell_time, 2),
            "active_top_down_goals": self.active_goals,
            "active_ior_entries": len(self.competition_engine.ior_map)
        }

    def reset_state(self) -> None:
        """Resets spotlight focus, goal biases, and IOR maps."""
        self.spotlight = AttentionalFocus()
        self.active_goals.clear()
        self.competition_engine.ior_map.clear()
        super().reset_state()
