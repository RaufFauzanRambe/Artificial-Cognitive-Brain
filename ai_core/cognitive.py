"""
cognition.py - Central Orchestrator for Artificial Cognitive Brain Architecture

Unifies Memory, Learning, Decision, Planning, and Reasoning modules into a single
cognitive pipeline with full text-to-speech voice automation, system metrics monitoring,
state context persistence, and robust fallback error management.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple
import time
import uuid
import pyttsx3

# Fallback & integration setup for system modules
try:
    from memory import (
        CognitiveMemoryRepository,
        MemoryNode,
        StorageDataCognitive,
        SystemDashboard,
    )
    from learning import CognitiveLearningEngine, LearningFeedback, LearningMode
    from decision import CognitiveDecisionEngine, OptionCandidate, DecisionOutcome, RiskTolerance
    from planning import CognitivePlanningEngine, ExecutivePlan, StepStatus, PlanPriority
except ImportError:
    # Standalone verification mock classes
    pass


class SystemState(Enum):
    """Operational state of the master cognitive brain."""
    UNINITIALIZED = "UNINITIALIZED"
    READY = "READY"
    PROCESSING = "PROCESSING"
    DEGRADED = "DEGRADED"
    ERROR = "ERROR"


@dataclass
class CognitiveCycleTelemetry:
    """Detailed telemetry record tracking performance during a single cognitive loop."""
    cycle_id: str
    prompt: str
    state: SystemState
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    execution_duration_sec: float = 0.0
    retrieved_memory_count: int = 0
    decision_confidence: float = 0.0
    plan_steps_completed: int = 0
    learning_reward_applied: float = 0.0
    error_logs: List[str] = field(default_factory=list)


class CentralVoiceSynthesizer:
    """Manages audio text-to-speech output across the centralized brain orchestrator."""

    def __init__(self, speech_rate: int = 165, volume: float = 1.0):
        self.speech_rate = speech_rate
        self.volume = volume
        self.engine: Optional[pyttsx3.Engine] = None
        self._initialize_master_voice()

    def _initialize_master_voice(self) -> None:
        """Configures master voice characteristics with exception handling."""
        try:
            self.engine = pyttsx3.init()
            if self.engine:
                self.engine.setProperty('rate', self.speech_rate)
                self.engine.setProperty('volume', self.volume)
                
                # Attempt natural voice selection
                voices = self.engine.getProperty('voices')
                for v in voices:
                    if "english" in v.name.lower() or "zira" in v.name.lower() or "david" in v.name.lower():
                        self.engine.setProperty('voice', v.id)
                        break
        except Exception as err:
            print(f"⚠️ [Master Voice Setup Notice]: Audio hardware binding failed. Output muted. Details: {err}")
            self.engine = None

    def speak(self, phrase: str) -> bool:
        """Central voice synthesizer output for the Cognitive Brain."""
        print(f"\n🗣️ [AI Voice Master]: \"{phrase}\"")
        if not self.engine:
            return False

        try:
            self.engine.say(phrase)
            self.engine.runAndWait()
            return True
        except Exception as err:
            print(f"❌ [Master TTS Error]: {err}")
            return False


class CognitiveBrainOrchestrator:
    """
    The master consciousness unit binding memory retention, dynamic decision-making,
    sequential planning, learning loops, and real-time audio output.
    """

    def __init__(
        self, 
        system_key: str = "ax-8i-+1hhxb_1biwdh", 
        enable_voice: bool = True,
        risk_profile: RiskTolerance = RiskTolerance.BALANCED
    ):
        """
        Initialize the Master Cognitive Brain Orchestrator.

        :param system_key: Secret authentication hardware board key.
        :param enable_voice: Toggle centralized audio text-to-speech feedback.
        :param risk_profile: Default risk profile mode for decision sub-engine.
        """
        print("🧠 [Cognitive Brain Core]: Initializing Subsystems...")
        self.system_key = system_key
        self.enable_voice = enable_voice
        self.state = SystemState.UNINITIALIZED
        self.telemetry_history: List[CognitiveCycleTelemetry] = []

        # 1. Base Storage & Authentication System
        self.storage = StorageDataCognitive()[cite: 1, 2]
        self.dashboard: SystemDashboard = self.storage.brain_intelligence(system_key)[cite: 1, 2]

        # 2. Subsystem Cognitive Engines
        self.learning = CognitiveLearningEngine(
            storage_engine=self.storage, 
            learning_rate=0.12, 
            enable_speech=False
        )
        self.decision = CognitiveDecisionEngine(
            storage_engine=self.storage, 
            risk_profile=risk_profile
        )
        self.planning = CognitivePlanningEngine(
            storage_engine=self.storage, 
            enable_speech=False,
            execution_delay_scale=0.2
        )

        # 3. Voice Synthesizer Setup
        self.voice = CentralVoiceSynthesizer(speech_rate=165, volume=1.0)
        self.state = SystemState.READY
        print(f"✨ [Cognitive Brain Core]: Orchestrator Online. Board Status: '{self.dashboard.status}'")[cite: 1, 2]

    def speak(self, phrase: str) -> None:
        """Delegates phrase vocalization to the master voice synthesizer."""
        if self.enable_voice:
            self.voice.speak(phrase)

    # -------------------------------------------------------------------------
    # Central Cognitive Execution Loop
    # -------------------------------------------------------------------------

    def run_cognitive_cycle(
        self,
        prompt: str,
        query_embedding: List[float],
        candidates: List[OptionCandidate],
        plan_actions: List[str],
        reward_expectation: float = 0.90
    ) -> Dict[str, Any]:
        """
        Executes a complete cognitive pipeline loop:
        1. Contextual Retrieval (Memory Read)
        2. Utility Matrix & Risk Evaluation (Decision Engine)
        3. Step-by-Step Task Construction & Execution (Planning Engine)
        4. Audio Output Broadcast (Voice Automation)
        5. Hebbian Association & Engram Update (Learning Integration)
        """
        cycle_id = f"cyc-{str(uuid.uuid4())[:8]}"
        telemetry = CognitiveCycleTelemetry(
            cycle_id=cycle_id,
            prompt=prompt,
            state=SystemState.PROCESSING
        )
        self.state = SystemState.PROCESSING
        start_timestamp = time.time()

        print(f"\n==========================================================")
        print(f"🚀 [Cognitive Cycle Initiated]: ID={cycle_id}")
        print(f"   Prompt: '{prompt}'")
        print(f"==========================================================")

        self.speak(f"Cognitive cycle initiated for query: {prompt}")

        try:
            # Step 1: Context Retrieval from Memory Repository
            pre_memories = self.storage.memory_repository.query_memory([cite: 1, 2]
                query_embedding=query_embedding, 
                top_k=3
            )
            telemetry.retrieved_memory_count = len(pre_memories)

            # Step 2: Decision Evaluation Matrix
            print("\n🔍 [Step 1: Decision Evaluation]")
            decision_result: DecisionOutcome = self.decision.make_decision(
                prompt=prompt,
                candidates=candidates,
                query_embedding=query_embedding,
                speak=False
            )
            telemetry.decision_confidence = decision_result.confidence
            self.speak(f"Option selected: {decision_result.selected_option.description}")

            # Step 3: Executive Plan Construction & Execution
            print("\n📋 [Step 2: Executive Task Planning]")
            plan: ExecutivePlan = self.planning.create_plan(
                goal=decision_result.selected_option.description,
                action_items=plan_actions,
                goal_embedding=query_embedding,
                priority=PlanPriority.HIGH
            )
            
            # Execute sequential actions
            executed_plan = self.planning.execute_plan(plan, speak_progress=self.enable_voice)
            telemetry.plan_steps_completed = sum(1 for s in executed_plan.steps if s.status == StepStatus.COMPLETED)

            # Step 4: Self-Learning Reinforcement Loop
            print("\n🧠 [Step 3: Self-Learning & Association Reinforcement]")
            target_mems = [m.id for m in self.storage.memory_repository.working_memory][cite: 1, 2]
            
            # Include decision node in learning feedback if available
            recent_nodes = self.storage.memory_repository.query_memory(query_embedding, top_k=1)[cite: 1, 2]
            if recent_nodes:
                target_mems.append(recent_nodes[0][0].id)

            feedback = LearningFeedback(
                task_id=executed_plan.plan_id,
                target_memory_ids=target_mems,
                reward_signal=reward_expectation,
                feedback_notes=f"Cycle {cycle_id} executed successfully. Plan: {executed_plan.goal}",
                learning_mode=LearningMode.HEBBIAN_REINFORCEMENT
            )
            
            updated_metrics = self.learning.process_feedback(feedback, speak_output=False)
            telemetry.learning_reward_applied = reward_expectation

            # Finalize Telemetry & State
            elapsed_time = round(time.time() - start_timestamp, 2)
            telemetry.end_time = datetime.utcnow()
            telemetry.execution_duration_sec = elapsed_time
            telemetry.state = SystemState.READY
            self.state = SystemState.READY

            self.speak("Cognitive cycle completed. System state updated.")
            self.telemetry_history.append(telemetry)

            return {
                "status": "SUCCESS",
                "cycle_id": cycle_id,
                "elapsed_seconds": elapsed_time,
                "decision": decision_result,
                "plan": executed_plan,
                "learning_metrics": updated_metrics,
                "georboard_status": self.dashboard.status[cite: 1, 2]
            }

        except Exception as err:
            error_msg = f"Critical breakdown during cognitive cycle: {str(err)}"
            print(f"❌ [Orchestrator Failure]: {error_msg}")
            telemetry.error_logs.append(error_msg)
            telemetry.state = SystemState.ERROR
            self.state = SystemState.ERROR
            self.speak("Warning: Cognitive cycle encountered a system error.")
            self.telemetry_history.append(telemetry)

            return {
                "status": "ERROR",
                "cycle_id": cycle_id,
                "error": str(err),
                "georboard_status": self.dashboard.status[cite: 1, 2]
            }

    # -------------------------------------------------------------------------
    # Diagnostics & System Reporting Methods
    # -------------------------------------------------------------------------

    def get_system_health(self) -> Dict[str, Any]:
        """Gathers aggregated diagnostics across storage, memory, and cognitive modules."""
        working_count = len(self.storage.memory_repository.working_memory)[cite: 1, 2]
        long_term_count = len(self.storage.memory_repository.long_term_memory)[cite: 1, 2]

        return {
            "orchestrator_state": self.state.value,
            "georboard_board_key": self.system_key,[cite: 1, 2]
            "georboard_status": self.dashboard.status,[cite: 1, 2]
            "working_memory_nodes": working_count,
            "long_term_memory_nodes": long_term_count,
            "total_cycles_executed": len(self.telemetry_history),
            "learning_avg_reward": self.learning.metrics.average_reward
        }


# -----------------------------------------------------------------------------
# Demonstration / Main System Run Execution
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # 1. Initialize the complete Artificial Cognitive Brain System
    brain = CognitiveBrainOrchestrator(
        system_key="ax-8i-+1hhxb_1biwdh",
        enable_voice=True,
        risk_profile=RiskTolerance.BALANCED
    )

    # 2. Define Query Prompt, Vector Embeddings & Candidate Options
    test_prompt = "Optimize core cognitive memory retention"
    mock_vector = [0.85, 0.15, 0.45]

    test_options = [
        OptionCandidate(
            option_id="c1", 
            description="Trigger Sleep-like Memory Consolidation Routine", 
            risk_score=0.10, 
            expected_reward=0.92
        ),
        OptionCandidate(
            option_id="c2", 
            description="Purge Short-term Working Memory Engrams", 
            risk_score=0.65, 
            expected_reward=0.40
        ),
        OptionCandidate(
            option_id="c3", 
            description="Re-cluster Graph Node Vector Embeddings", 
            risk_score=0.30, 
            expected_reward=0.85
        )
    ]

    action_workflow = [
        "Analyze working memory node density",
        "Transfer high-importance engrams to long-term store",
        "Re-index vector similarities across associative graph"
    ]

    # 3. Execute Master Cognitive Pipeline Loop
    cycle_summary = brain.run_cognitive_cycle(
        prompt=test_prompt,
        query_embedding=mock_vector,
        candidates=test_options,
        plan_actions=action_workflow,
        reward_expectation=0.95
    )

    # 4. Display System Diagnostics & Execution Results
    print("\n==========================================================")
    print("   Cognitive Brain Orchestration Results Summary")
    print("==========================================================")
    print(f"Cycle Execution Status : {cycle_summary.get('status')}")
    print(f"Time Taken             : {cycle_summary.get('elapsed_seconds')} seconds")
    
    if cycle_summary.get('status') == "SUCCESS":
        dec = cycle_summary['decision']
        pln = cycle_summary['plan']
        print(f"Selected Decision      : [{dec.selected_option.option_id}] {dec.selected_option.description}")
        print(f"Decision Confidence    : {dec.confidence * 100}%")
        print(f"Plan ID Created        : {pln.plan_id}")
        print(f"Plan Execution Status  : Fully Completed = {pln.is_fully_completed}")

    # Display Brain Diagnostics
    print("\n--- Central System Diagnostics ---")
    health = brain.get_system_health()
    for key, value in health.items():
        print(f"  {key:<25}: {value}")
    print("==========================================================")
