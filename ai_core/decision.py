"""
decision.py - Artificial Cognitive Brain Decision Engine & Voice Automation

Evaluates multiple candidate choices against cognitive memory data, 
computes risk-reward matrices, resolves execution ambiguities, 
and announces decisions via automated text-to-speech feedback.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
import math
import uuid
import pyttsx3

# Fallback & integration imports for memory modules
try:
    from memory import (
        CognitiveMemoryRepository,
        MemoryNode,
        StorageDataCognitive,
        SystemDashboard,
    )
except ImportError:
    # Fallback mock for standalone syntax checking
    pass


class RiskTolerance(Enum):
    """Defines the risk profile mode of the decision engine."""
    CONSERVATIVE = "CONSERVATIVE"  # Heavily penalizes risk scores
    BALANCED = "BALANCED"          # Standard risk-reward tradeoff
    AGGRESSIVE = "AGGRESSIVE"      # Prioritizes high expected rewards despite risk


@dataclass
class OptionCandidate:
    """Represents a selectable action candidate for evaluation."""
    option_id: str
    description: str
    risk_score: float        # Range: 0.0 (Safe) to 1.0 (Critical Risk)
    expected_reward: float   # Range: 0.0 (No Value) to 1.0 (High Value)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DecisionOutcome:
    """Encapsulates the final selected decision, confidence, and reasoning trail."""
    decision_id: str
    prompt: str
    selected_option: OptionCandidate
    utility_score: float
    confidence: float
    reasoning: str
    retrieved_memory_ids: List[str]
    timestamp: datetime = field(default_factory=datetime.utcnow)


class TTSVoiceHandler:
    """Handles audio configuration and safe speech synthesis execution."""

    def __init__(self, speech_rate: int = 150, volume: float = 0.9):
        self.speech_rate = speech_rate
        self.volume = volume
        self.engine: Optional[pyttsx3.Engine] = None
        self._init_tts()

    def _init_tts(self) -> None:
        """Initializes pyttsx3 audio hardware bindings with fallback handling."""
        try:
            self.engine = pyttsx3.init()
            if self.engine:
                self.engine.setProperty('rate', self.speech_rate)
                self.engine.setProperty('volume', self.volume)
        except Exception as err:
            print(f"⚠️ [TTS Init Notice]: Audio output disabled (Hardware unbindable). Details: {err}")
            self.engine = None

    def speak(self, text: str) -> bool:
        """Broadcasts audio output safely without crashing the main thread."""
        print(f"🔊 [AI Voice Decision Output]: {text}")
        if not self.engine:
            return False

        try:
            self.engine.say(text)
            self.engine.runAndWait()
            return True
        except Exception as err:
            print(f"❌ [TTS Execution Error]: {err}")
            return False


class CognitiveDecisionEngine:
    """
    Evaluates alternative candidate choices, resolves decision ambiguity, 
    computes utility matrices using memory bias, and vocalizes results.
    """

    def __init__(
        self, 
        storage_engine: StorageDataCognitive,
        risk_profile: RiskTolerance = RiskTolerance.BALANCED,
        min_confidence_threshold: float = 0.40
    ):
        """
        Initialize the Cognitive Decision Engine.

        :param storage_engine: The active StorageDataCognitive instance binding memory.
        :param risk_profile: The decision risk strategy (CONSERVATIVE, BALANCED, AGGRESSIVE).
        :param min_confidence_threshold: Minimum acceptable confidence score to execute a decision.
        """
        self.storage_engine = storage_engine
        self.memory_repo: CognitiveMemoryRepository = storage_engine.memory_repository[cite: 2]
        self.risk_profile = risk_profile
        self.min_confidence_threshold = min_confidence_threshold
        self.voice_handler = TTSVoiceHandler()
        self.decision_history: List[DecisionOutcome] = []

    # -------------------------------------------------------------------------
    # Core Decision Matrix Engine
    # -------------------------------------------------------------------------

    def make_decision(
        self,
        prompt: str,
        candidates: List[OptionCandidate],
        query_embedding: List[float],
        speak: bool = True
    ) -> DecisionOutcome:
        """
        Calculates utility scores for candidate options by balancing expected reward, 
        risk penalty matrices, and historical cognitive memory bias.
        """
        if not candidates:
            raise ValueError("Decision Engine failed: Candidate options list cannot be empty.")

        decision_id = f"dec-{str(uuid.uuid4())[:8]}"

        # Step 1: Fetch relevant contextual engrams from working & long-term memory
        relevant_memories = self.memory_repo.query_memory([cite: 1, 2]
            query_embedding=query_embedding,
            top_k=3,
            similarity_weight=0.6,[cite: 1, 2]
            recency_weight=0.2,[cite: 1, 2]
            importance_weight=0.2[cite: 1, 2]
        )

        # Compute memory bias factor based on historical retrieval similarity
        memory_bias = (
            sum(m[1] for m in relevant_memories) / len(relevant_memories)
            if relevant_memories else 0.5
        )
        memory_ids = [m[0].id for m in relevant_memories] if relevant_memories else []

        # Step 2: Weight adjustments according to the selected risk profile
        reward_weight, risk_weight = self._get_profile_weights()

        # Step 3: Evaluate utility scores for each candidate choice
        best_candidate: Optional[OptionCandidate] = None
        highest_utility = -float("inf")

        for candidate in candidates:
            # Multi-Criteria Utility Equation:
            # Utility = (Reward * RewardWeight) + (MemoryBias * 0.25) - (Risk * RiskWeight)
            utility = (
                (candidate.expected_reward * reward_weight) +
                (memory_bias * 0.25) -
                (candidate.risk_score * risk_weight)
            )

            if utility > highest_utility:
                highest_utility = utility
                best_candidate = candidate

        # Step 4: Calculate Confidence Rating & Formulate Reasoning
        confidence = self._calculate_confidence(highest_utility, memory_bias)
        
        reasoning_str = (
            f"Evaluated {len(candidates)} options under '{self.risk_profile.value}' profile. "
            f"Selected candidate '{best_candidate.option_id}' ({best_candidate.description}) "
            f"with Utility Score = {highest_utility:.4f} and Memory Bias = {memory_bias:.4f}."
        )

        # Step 5: Handle Low-Confidence Fallback Scenario
        if confidence < self.min_confidence_threshold:
            reasoning_str += " [WARNING: Confidence below safety threshold. Execution flagged.]"

        outcome = DecisionOutcome(
            decision_id=decision_id,
            prompt=prompt,
            selected_option=best_candidate,
            utility_score=round(highest_utility, 4),
            confidence=confidence,
            reasoning=reasoning_str,
            retrieved_memory_ids=memory_ids
        )

        # Step 6: Write decision memory node into memory repository
        self.memory_repo.write_memory([cite: 1, 2]
            content=f"Decision Outcome ({decision_id}): Selected '{best_candidate.description}' for query '{prompt}'",
            embedding=query_embedding,
            importance=confidence,
            tags=["decision", "automated_choice", self.risk_profile.value.lower()]
        )

        self.decision_history.append(outcome)

        # Step 7: Broadcast Voice Output
        if speak:
            self.speak_decision(outcome)

        return outcome

    # -------------------------------------------------------------------------
    # Auxiliary Decision Utilities
    # -------------------------------------------------------------------------

    def speak_decision(self, outcome: DecisionOutcome) -> None:
        """Formats and broadcasts selected decisions via automated speech synthesis."""
        speech_text = (
            f"Decision finalized for task: {outcome.prompt}. "
            f"Option selected: {outcome.selected_option.description}. "
            f"Confidence rating is {int(outcome.confidence * 100)} percent."
        )
        self.voice_handler.speak(speech_text)

    def _get_profile_weights(self) -> Tuple[float, float]:
        """Returns (reward_weight, risk_weight) tuples based on active risk profile."""
        if self.risk_profile == RiskTolerance.CONSERVATIVE:
            return 0.50, 0.85
        elif self.risk_profile == RiskTolerance.AGGRESSIVE:
            return 0.85, 0.25
        # Default: BALANCED Profile
        return 0.70, 0.50

    def _calculate_confidence(self, utility: float, memory_bias: float) -> float:
        """Sigmoid-inspired normalizer for confidence ratings (clamped between 0.10 and 0.99)."""
        raw_confidence = (utility * 0.6) + (memory_bias * 0.4)
        normalized = 1.0 / (1.0 + math.exp(-raw_confidence))
        return round(max(0.10, min(0.99, normalized)), 2)


# -----------------------------------------------------------------------------
# Standalone Integration Routine & Verification
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("==========================================================")
    print("   Artificial Cognitive Brain - Decision Engine Execution")
    print("==========================================================")

    # 1. Initialize Storage infrastructure & memory engine
    storage = StorageDataCognitive(working_memory_capacity=5)[cite: 1, 2]
    if not storage.georboard:
        storage.brain_intelligence("ax-8i-+1hhxb_1biwdh")[cite: 1, 2]

    # 2. Seed memory with contextual history
    storage.memory_repository.write_memory([cite: 1, 2]
        content="Prior System Maintenance: Full optimization produced 35% speed boost.",
        embedding=[0.5, 0.5, 0.2],
        importance=0.88,
        tags=["maintenance", "history"]
    )

    # 3. Instantiate Decision Engine
    decider = CognitiveDecisionEngine(
        storage_engine=storage,
        risk_profile=RiskTolerance.BALANCED,
        min_confidence_threshold=0.50
    )

    # 4. Define Candidate Options
    options = [
        OptionCandidate(
            option_id="opt_1",
            description="Execute low-risk backup protocol",
            risk_score=0.10,
            expected_reward=0.60
        ),
        OptionCandidate(
            option_id="opt_2",
            description="Perform full system optimization routine",
            risk_score=0.35,
            expected_reward=0.92
        ),
        OptionCandidate(
            option_id="opt_3",
            description="Re-index vector database while live",
            risk_score=0.80,
            expected_reward=0.95
        )
    ]

    # 5. Run Decision Routine
    prompt_query = "Automated System Maintenance Request"
    query_vec = [0.5, 0.5, 0.2]

    print(f"\n[Evaluating Options for Query]: '{prompt_query}'")
    decision_result = decider.make_decision(
        prompt=prompt_query,
        candidates=options,
        query_embedding=query_vec,
        speak=True
    )

    # 6. Display Output Summary
    print("\n--- Decision Evaluation Summary ---")
    print(f"Decision ID      : {decision_result.decision_id}")
    print(f"Selected Option  : [{decision_result.selected_option.option_id}] {decision_result.selected_option.description}")
    print(f"Utility Score    : {decision_result.utility_score}")
    print(f"Confidence Level : {decision_result.confidence * 100}%")
    print(f"Reasoning Trail  : {decision_result.reasoning}")
    print("==========================================================")
