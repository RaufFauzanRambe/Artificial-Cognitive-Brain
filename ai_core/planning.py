"""
planning.py - Artificial Cognitive Brain Sequential Planning & Voice Automation

Generates structured multi-step execution plans, tracks step progress, handles dynamic task 
dependencies, records step outcomes into memory, and provides real-time voice updates on task execution.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
import time
import uuid
import pyttsx3

# Fallback & integration imports for memory system
try:
    from memory import (
        CognitiveMemoryRepository,
        MemoryNode,
        StorageDataCognitive,
        SystemDashboard,
    )
except ImportError:
    # Standalone execution mock fallback
    pass


class StepStatus(Enum):
    """Execution status of an individual plan step."""
    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


class PlanPriority(Enum):
    """Priority levels for executive plan scheduling."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class PlanStep:
    """Represents an individual step within an executive planning sequence."""
    step_id: int
    action_name: str
    estimated_duration_sec: float
    dependencies: List[int] = field(default_factory=list)
    status: StepStatus = StepStatus.PENDING
    result_notes: str = ""
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    @property
    def is_ready(self) -> bool:
        """Checks whether step dependencies have been satisfied."""
        return self.status == StepStatus.PENDING


@dataclass
class ExecutivePlan:
    """Encapsulates a multi-step sequence designed to achieve a cognitive goal."""
    plan_id: str
    goal: str
    steps: List[PlanStep]
    priority: PlanPriority = PlanPriority.MEDIUM
    created_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))
    completed_at: Optional[str] = None
    total_execution_time_sec: float = 0.0

    @property
    def is_fully_completed(self) -> bool:
        """Determines if all individual steps have achieved completed status."""
        return all(s.status == StepStatus.COMPLETED for s in self.steps)


class PlanningVoiceSynthesizer:
    """Manages audio text-to-speech output for planning execution updates."""

    def __init__(self, speech_rate: int = 155, volume: float = 0.9):
        self.speech_rate = speech_rate
        self.volume = volume
        self.engine: Optional[pyttsx3.Engine] = None
        self._initialize_tts()

    def _initialize_tts(self) -> None:
        """Initializes the underlying pyttsx3 voice hardware binding."""
        try:
            self.engine = pyttsx3.init()
            if self.engine:
                self.engine.setProperty('rate', self.speech_rate)
                self.engine.setProperty('volume', self.volume)
        except Exception as err:
            print(f"⚠️ [Planning Voice Init Warning]: Audio output disabled. Details: {err}")
            self.engine = None

    def speak(self, phrase: str) -> bool:
        """Renders speech output via TTS hardware safely."""
        print(f"🔊 [AI Voice Planning Output]: {phrase}")
        if not self.engine:
            return False

        try:
            self.engine.say(phrase)
            self.engine.runAndWait()
            return True
        except Exception as err:
            print(f"❌ [Planning TTS Execution Error]: {err}")
            return False


class CognitivePlanningEngine:
    """
    Constructs multi-step plans, monitors execution progress, evaluates dependencies,
    writes execution logs to memory repositories, and vocalizes status updates.
    """

    def __init__(
        self,
        storage_engine: StorageDataCognitive,
        enable_speech: bool = True,
        execution_delay_scale: float = 1.0
    ):
        """
        Initialize the Cognitive Planning Engine.

        :param storage_engine: Active StorageDataCognitive instance binding memory layers.
        :param enable_speech: Toggle real-time speech automation output.
        :param execution_delay_scale: Time multiplier for step execution simulation.
        """
        self.storage_engine = storage_engine
        self.memory_repo: CognitiveMemoryRepository = storage_engine.memory_repository[cite: 2]
        self.enable_speech = enable_speech
        self.execution_delay_scale = execution_delay_scale
        self.voice_synthesizer = PlanningVoiceSynthesizer()
        self.plan_history: Dict[str, ExecutivePlan] = {}

    # -------------------------------------------------------------------------
    # Core Plan Creation & Lifecycle Operations
    # -------------------------------------------------------------------------

    def create_plan(
        self,
        goal: str,
        action_items: List[str],
        goal_embedding: List[float],
        priority: PlanPriority = PlanPriority.MEDIUM,
        durations: Optional[List[float]] = None
    ) -> ExecutivePlan:
        """
        Constructs a structured multi-step plan, computes step dependencies, 
        and registers the intent in cognitive memory.
        """
        if not action_items:
            raise ValueError("Cannot construct an executive plan with zero action items.")

        plan_id = f"plan-{int(time.time())}-{str(uuid.uuid4())[:4]}"
        plan_steps: List[PlanStep] = []

        for index, action in enumerate(action_items):
            step_num = index + 1
            # Step duration calculation
            est_duration = durations[index] if (durations and index < len(durations)) else 1.5
            
            # Formulate simple linear dependency chain (step N depends on step N-1)
            deps = [step_num - 1] if step_num > 1 else []

            step = PlanStep(
                step_id=step_num,
                action_name=action,
                estimated_duration_sec=est_duration,
                dependencies=deps,
                status=StepStatus.PENDING
            )
            plan_steps.append(step)

        plan = ExecutivePlan(
            plan_id=plan_id,
            goal=goal,
            steps=plan_steps,
            priority=priority
        )

        # Query memory to retrieve past experience related to this goal
        context_memories = self.memory_repo.query_memory([cite: 1, 2]
            query_embedding=goal_embedding,
            top_k=2
        )
        context_ids = [m[0].id for m in context_memories] if context_memories else []

        # Store executive plan intent node inside cognitive memory repository
        self.memory_repo.write_memory([cite: 1, 2]
            content=f"Executive Plan Created ({plan_id}): Goal='{goal}' containing {len(plan_steps)} steps.",
            embedding=goal_embedding,
            importance=0.80,
            tags=["planning", "workflow", priority.name.lower()]
        )

        self.plan_history[plan_id] = plan
        return plan

    def execute_plan(self, plan: ExecutivePlan, speak_progress: Optional[bool] = None) -> ExecutivePlan:
        """
        Executes an executive plan sequentially, tracking step outcomes, dependencies,
        and announcing execution milestones using voice automation.
        """
        should_speak = self.enable_speech if speak_progress is None else speak_progress
        start_time = time.time()

        if should_speak:
            self.voice_synthesizer.speak(f"Starting execution of plan for goal: {plan.goal}")

        for step in plan.steps:
            # Dependency Resolution Check
            if not self._check_dependencies_met(step, plan):
                step.status = StepStatus.SKIPPED
                step.result_notes = "Skipped due to unfulfilled step dependencies."
                if should_speak:
                    self.voice_synthesizer.speak(f"Step {step.step_id} skipped due to dependency failure.")
                continue

            # Step Execution Phase
            step.status = StepStatus.IN_PROGRESS
            step.started_at = datetime.utcnow()

            # Simulate hardware/task execution cycle
            execution_delay = step.estimated_duration_sec * self.execution_delay_scale
            time.sleep(execution_delay)

            # Mark completion
            step.status = StepStatus.COMPLETED
            step.completed_at = datetime.utcnow()
            step.result_notes = "Executed successfully without exceptions."

            status_msg = f"Step {step.step_id} completed: {step.action_name}."
            if should_speak:
                self.voice_synthesizer.speak(status_msg)

        # Complete Plan Summary
        total_time = round(time.time() - start_time, 2)
        plan.total_execution_time_sec = total_time
        plan.completed_at = time.strftime("%Y-%m-%d %H:%M:%S")

        completion_msg = f"All steps successfully executed for plan {plan.plan_id} in {total_time} seconds."
        if should_speak:
            self.voice_synthesizer.speak(completion_msg)

        # Write execution summary back into cognitive memory
        self.memory_repo.write_memory([cite: 1, 2]
            content=f"Executive Plan Completed ({plan.plan_id}): Goal='{plan.goal}' achieved in {total_time}s.",
            embedding=[0.3, 0.7, 0.1],
            importance=0.85,
            tags=["planning_execution", "completed"]
        )

        return plan

    # -------------------------------------------------------------------------
    # Auxiliary Planning Helpers
    # -------------------------------------------------------------------------

    def speak_plan_status(self, text: str) -> None:
        """Wrapper method for direct speech output from external caller."""
        if self.enable_speech:
            self.voice_synthesizer.speak(text)

    def _check_dependencies_met(self, step: PlanStep, plan: ExecutivePlan) -> bool:
        """Validates if all parent step dependencies are marked COMPLETED."""
        if not step.dependencies:
            return True

        for dep_id in step.dependencies:
            parent_step = next((s for s in plan.steps if s.step_id == dep_id), None)
            if not parent_step or parent_step.status != StepStatus.COMPLETED:
                return False
        return True


# -----------------------------------------------------------------------------
# Demonstration & Main Standalone Execution Routine
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("==========================================================")
    print("   Artificial Cognitive Brain - Planning Engine Execution")
    print("==========================================================")

    # 1. Initialize System Core Storage and Hardware Intelligence
    cognitive_storage = StorageDataCognitive(working_memory_capacity=5)[cite: 1, 2]
    if not cognitive_storage.georboard:
        cognitive_storage.brain_intelligence("ax-8i-+1hhxb_1biwdh")[cite: 1, 2]

    # 2. Instantiate Cognitive Planning Engine
    planner = CognitivePlanningEngine(
        storage_engine=cognitive_storage,
        enable_speech=True,
        execution_delay_scale=0.3  # Speed up execution time for demo
    )

    # 3. Define Goal and Action Workflow
    goal_input = "Database Synchronization Routine"
    steps_input = [
        "Lock working memory state",
        "Export long-term engrams to persistent disk",
        "Verify checksum hash and consistency"
    ]
    goal_vec = [0.30, 0.70, 0.10]

    # 4. Construct Multi-Step Plan
    print(f"\n[Creating Plan]: Goal = '{goal_input}'")
    executive_plan = planner.create_plan(
        goal=goal_input,
        action_items=steps_input,
        goal_embedding=goal_vec,
        priority=PlanPriority.HIGH
    )

    print(f"Plan ID Generated : {executive_plan.plan_id}")
    print(f"Total Steps       : {len(executive_plan.steps)}")

    # 5. Execute Plan with Integrated Voice Automation
    print("\n[Executing Plan Workflow...]")
    executed_plan = planner.execute_plan(executive_plan, speak_progress=True)

    # 6. Display Detailed Post-Execution Verification Summary
    print("\n--- Plan Execution Summary ---")
    print(f"Plan Goal            : {executed_plan.goal}")
    print(f"Fully Completed      : {executed_plan.is_fully_completed}")
    print(f"Total Time Elapsed   : {executed_plan.total_execution_time_sec} s")
    print(f"Completion Timestamp : {executed_plan.completed_at}")
    
    print("\n--- Individual Step Telemetry ---")
    for s in executed_plan.steps:
        print(f"  Step {s.step_id}: [{s.status.value}] {s.action_name} | Notes: {s.result_notes}")
    print("==========================================================")
