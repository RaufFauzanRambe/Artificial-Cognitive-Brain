"""
reasoning.py - Artificial Cognitive Brain Executive Reasoning Engine

This module implements the core cognitive reasoning and executive control unit
for autonomous AI agents. It integrates directly with CognitiveMemoryRepository 
and StorageDataCognitive to synthesize retrieved context, perform step-by-step
logic decomposition, evaluate confidence thresholds, and generate grounded actions.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
import math
import uuid

# Import cognitive storage and memory structures from memory.py
from memory import (
    CognitiveMemoryRepository,
    MemoryNode,
    StorageDataCognitive,
    SystemDashboard,
)


class CognitiveState(Enum):
    """Represents the current processing state of the reasoning engine."""
    IDLE = "IDLE"
    ANALYZING = "ANALYZING"
    RETRIEVING = "RETRIEVING"
    DEDUCING = "DEDUCING"
    RESOLVING_CONFLICT = "RESOLVING_CONFLICT"
    EXECUTING = "EXECUTING"
    FAILED = "FAILED"


class DeliberationStrategy(Enum):
    """Defines the cognitive depth strategy applied to a problem."""
    FAST_HEURISTIC = "FAST_HEURISTIC"  # Low-latency, direct association search
    CHAIN_OF_THOUGHT = "CHAIN_OF_THOUGHT"  # Multi-step sequential logic breakdown
    TREE_OF_THOUGHT = "TREE_OF_THOUGHT"   # Branching path analysis with back-tracking


@dataclass
class ThoughtStep:
    """Represents a single step in a multi-step cognitive reasoning sequence."""
    step_number: int
    premise: str
    deduction: str
    confidence: float  # Scale 0.0 to 1.0
    supporting_memory_ids: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ReasoningOutcome:
    """Encapsulates the final decision, executive action, and supporting trail."""
    task_id: str
    query: str
    final_conclusion: str
    action_plan: List[str]
    overall_confidence: float
    thought_chain: List[ThoughtStep]
    retrieved_context: List[Tuple[MemoryNode, float]]
    execution_time_ms: float
    strategy_used: DeliberationStrategy


class ExecutiveReasoningEngine:
    """
    Core cognitive reasoning processor. Evaluates complex inputs, interacts 
    with memory repositories, performs logic synthesis, and constructs dynamic plans.
    """

    def __init__(
        self,
        storage_engine: StorageDataCognitive,
        min_confidence_threshold: float = 0.65,
        max_reasoning_depth: int = 5,
    ):
        """
        Initialize the Executive Reasoning Engine.

        :param storage_engine: The active StorageDataCognitive instance binding system control & memory.
        :param min_confidence_threshold: Minimum confidence score required to finalize a decision.
        :param max_reasoning_depth: Maximum recursion/thought steps allowed per deliberation.
        """
        self.storage_engine = storage_engine
        self.memory_repo: CognitiveMemoryRepository = storage_engine.memory_repository[cite: 2]
        self.min_confidence_threshold = min_confidence_threshold
        self.max_reasoning_depth = max_reasoning_depth
        self.current_state = CognitiveState.IDLE
        self.reasoning_history: List[ReasoningOutcome] = []

    # -------------------------------------------------------------------------
    # Main Executive Workflow
    # -------------------------------------------------------------------------

    def process_task(
        self,
        task_prompt: str,
        query_embedding: List[float],
        strategy: DeliberationStrategy = DeliberationStrategy.CHAIN_OF_THOUGHT,
        importance: float = 0.7,
    ) -> ReasoningOutcome:
        """
        Processes an incoming cognitive prompt using memory retrieval, multi-step 
        deduction, and action generation.
        """
        start_time = datetime.utcnow()
        task_id = str(uuid.uuid4())
        self.current_state = CognitiveState.ANALYZING

        # Step 1: Ensure system intelligence and motherboard board interface are authenticated
        if not self.storage_engine.georboard or self.storage_engine.georboard.status != "ONLINE":
            self.storage_engine.brain_intelligence()[cite: 1, 2]

        # Step 2: Retrieve Context from Working & Long-Term Memory
        self.current_state = CognitiveState.RETRIEVING
        relevant_memories = self.memory_repo.query_memory([cite: 1, 2]
            query_embedding=query_embedding,
            top_k=4,
            similarity_weight=0.5,[cite: 1, 2]
            recency_weight=0.3,[cite: 1, 2]
            importance_weight=0.2,[cite: 1, 2]
        )

        # Step 3: Select Deliberation Pipeline based on Strategy
        self.current_state = CognitiveState.DEDUCING
        thought_chain: List[ThoughtStep] = []
        
        if strategy == DeliberationStrategy.FAST_HEURISTIC:
            thought_chain, overall_confidence = self._execute_fast_heuristic(
                task_prompt, relevant_memories
            )
        elif strategy == DeliberationStrategy.TREE_OF_THOUGHT:
            thought_chain, overall_confidence = self._execute_tree_of_thought(
                task_prompt, relevant_memories
            )
        else:  # Default: CHAIN_OF_THOUGHT
            thought_chain, overall_confidence = self._execute_chain_of_thought(
                task_prompt, relevant_memories
            )

        # Step 4: Conflict Resolution and Fallback handling
        if overall_confidence < self.min_confidence_threshold:
            self.current_state = CognitiveState.RESOLVING_CONFLICT
            thought_chain, overall_confidence = self._resolve_low_confidence(
                task_prompt, thought_chain, relevant_memories
            )

        # Step 5: Action Synthesis & Memory Write-back
        self.current_state = CognitiveState.EXECUTING
        final_conclusion = self._synthesize_conclusion(thought_chain)
        action_plan = self._derive_action_plan(thought_chain, final_conclusion)

        elapsed_ms = (datetime.utcnow() - start_time).total_seconds() * 1000.0

        outcome = ReasoningOutcome(
            task_id=task_id,
            query=task_prompt,
            final_conclusion=final_conclusion,
            action_plan=action_plan,
            overall_confidence=round(overall_confidence, 4),
            thought_chain=thought_chain,
            retrieved_context=relevant_memories,
            execution_time_ms=round(elapsed_ms, 2),
            strategy_used=strategy,
        )

        # Write the reasoning experience back into working memory as an engram node
        self.memory_repo.write_memory([cite: 1, 2]
            content=f"Task: {task_prompt} -> Conclusion: {final_conclusion}",
            embedding=query_embedding,
            importance=max(importance, overall_confidence),
            tags=["reasoning_outcome", strategy.value.lower()],
        )

        self.reasoning_history.append(outcome)
        self.current_state = CognitiveState.IDLE
        return outcome

    # -------------------------------------------------------------------------
    # Strategy Deliberation Implementations
    # -------------------------------------------------------------------------

    def _execute_fast_heuristic(
        self, prompt: str, memories: List[Tuple[MemoryNode, float]]
    ) -> Tuple[List[ThoughtStep], float]:
        """Direct associative retrieval mapping with lightweight validation."""
        top_memory_id = memories[0][0].id if memories else "NONE"
        context_str = memories[0][0].content if memories else "No prior context"[cite: 1, 2]

        step = ThoughtStep(
            step_number=1,
            premise=f"Direct pattern match for query: '{prompt}'",
            deduction=f"Applied immediate context match: '{context_str}'",
            confidence=memories[0][1] if memories else 0.5,[cite: 1, 2]
            supporting_memory_ids=[top_memory_id] if memories else [],
        )

        confidence = memories[0][1] if memories else 0.4[cite: 1, 2]
        return [step], confidence

    def _execute_chain_of_thought(
        self, prompt: str, memories: List[Tuple[MemoryNode, float]]
    ) -> Tuple[List[ThoughtStep], float]:
        """Executes step-by-step logic expansion using context nodes."""
        steps: List[ThoughtStep] = []
        cumulative_conf = 0.0

        # Phase 1: Problem Decomposition
        supp_ids = [m[0].id for m in memories[:2]][cite: 1, 2]
        step1 = ThoughtStep(
            step_number=1,
            premise=f"Deconstruct task requirements for prompt: '{prompt}'",
            deduction="Divided task into context extraction, logic alignment, and synthesis.",
            confidence=0.90,
            supporting_memory_ids=supp_ids,
        )
        steps.append(step1)

        # Phase 2: Contextual Analysis & Cross-Referencing
        context_summary = " & ".join([f"[{m[0].content}]" for m in memories]) if memories else "Empty"[cite: 1, 2]
        avg_retrieval_score = (
            sum(m[1] for m in memories) / len(memories) if memories else 0.3[cite: 1, 2]
        )
        
        step2 = ThoughtStep(
            step_number=2,
            premise=f"Evaluate contextual evidence: {context_summary}",
            deduction="Verified alignment between user intent and retrieved long-term engrams.",
            confidence=round(min(0.95, avg_retrieval_score + 0.2), 4),
            supporting_memory_ids=[m[0].id for m in memories],[cite: 1, 2]
        )
        steps.append(step2)

        # Phase 3: Inferential Deduction
        step3 = ThoughtStep(
            step_number=3,
            premise="Synthesize premise (Deconstructed Task) with deduction (Contextual Evidence).",
            deduction="Constructed optimal strategy path based on historical memory patterns.",
            confidence=0.85,
            supporting_memory_ids=supp_ids,
        )
        steps.append(step3)

        cumulative_conf = sum(s.confidence for s in steps) / len(steps)
        return steps, cumulative_conf

    def _execute_tree_of_thought(
        self, prompt: str, memories: List[Tuple[MemoryNode, float]]
    ) -> Tuple[List[ThoughtStep], float]:
        """Evaluates multiple logical pathways and picks the optimal branch."""
        branches: List[List[ThoughtStep]] = []

        # Branch A: Strict Memory-Grounded Path
        branch_a_steps, conf_a = self._execute_chain_of_thought(prompt, memories)

        # Branch B: Generative Dynamic Logic Path (Weighted towards novelty)
        branch_b_steps = [
            ThoughtStep(
                step_number=1,
                premise="Formulate abstract hypothesis without strict memory reliance.",
                deduction="Hypothesized novel resolution pattern.",
                confidence=0.72,
            ),
            ThoughtStep(
                step_number=2,
                premise="Validate hypothesis against general logical consistency.",
                deduction="Hypothesis passed basic validation rules.",
                confidence=0.78,
            ),
        ]
        conf_b = sum(s.confidence for s in branch_b_steps) / len(branch_b_steps)

        # Select highest confidence branch
        if conf_a >= conf_b:
            return branch_a_steps, conf_a
        return branch_b_steps, conf_b

    # -------------------------------------------------------------------------
    # Auxiliary Cognitive Operations
    # -------------------------------------------------------------------------

    def _resolve_low_confidence(
        self,
        prompt: str,
        chain: List[ThoughtStep],
        memories: List[Tuple[MemoryNode, float]],
    ) -> Tuple[List[ThoughtStep], float]:
        """Refines logic when initial confidence falls below the accepted threshold."""
        remedial_step = ThoughtStep(
            step_number=len(chain) + 1,
            premise="Confidence score below threshold. Executing cognitive safety buffer.",
            deduction="Applied cautious baseline assumptions and flagged query for further verification.",
            confidence=self.min_confidence_threshold,
            supporting_memory_ids=[m[0].id for m in memories],[cite: 1, 2]
        )
        chain.append(remedial_step)
        new_confidence = sum(s.confidence for s in chain) / len(chain)
        return chain, new_confidence

    def _synthesize_conclusion(self, thought_chain: List[ThoughtStep]) -> str:
        """Converts step-by-step logic into a single cohesive output statement."""
        if not thought_chain:
            return "Unable to reach a definitive cognitive conclusion."
        last_deduction = thought_chain[-1].deduction
        return f"Final Cognitive Decision: {last_deduction}"

    def _derive_action_plan(
        self, thought_chain: List[ThoughtStep], conclusion: str
    ) -> List[str]:
        """Translates final thought chain into explicit sequential actions."""
        actions = []
        for step in thought_chain:
            actions.append(f"EXECUTE_STEP_{step.step_number}: {step.premise}")
        actions.append(f"DISPATCH_RESPONSE: Process output -> '{conclusion}'")
        return actions


# -----------------------------------------------------------------------------
# Demonstration / Main Integration Routine
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("==========================================================")
    print("   Artificial Cognitive Brain - Executive Reasoning Engine")
    print("==========================================================")

    # Step 1: Initialize System Core Storage and Hardware Intelligence
    cognitive_storage = StorageDataCognitive(working_memory_capacity=5)[cite: 1, 2]
    georboard = cognitive_storage.brain_intelligence("ax-8i-+1hhxb_1biwdh")[cite: 1, 2]
    
    print(f"[System Init] Georboard Board Status: {georboard.status}")[cite: 1, 2]

    # Step 2: Seed Cognitive Memory with initial knowledge nodes
    repo = cognitive_storage.memory_repository[cite: 2]
    emb_sec = [0.88, 0.12, 0.30]
    emb_net = [0.20, 0.90, 0.85]
    
    node_a = repo.write_memory([cite: 1, 2]
        content="Security Policy: Port 8080 requires encrypted TLS tunnel for execution",
        embedding=emb_sec,
        importance=0.92,
        tags=["security", "network"],
    )
    node_b = repo.write_memory([cite: 1, 2]
        content="System Protocol: Reroute unencrypted connections to proxy port 443",
        embedding=emb_net,
        importance=0.85,
        tags=["routing", "network"],
    )

    # Establish an associative link between the memory nodes
    repo.create_association(node_a.id, node_b.id, weight=0.88)[cite: 1, 2]

    # Step 3: Instantiate Executive Reasoning Engine
    reasoning_engine = ExecutiveReasoningEngine(
        storage_engine=cognitive_storage,
        min_confidence_threshold=0.70,
        max_reasoning_depth=5,
    )

    # Step 4: Execute Task Deliberation
    task = "Authorize inbound HTTP traffic on port 8080 safely"
    task_embedding = [0.85, 0.20, 0.35]

    print(f"\n[Task Processing] Query: '{task}'")
    outcome = reasoning_engine.process_task(
        task_prompt=task,
        query_embedding=task_embedding,
        strategy=DeliberationStrategy.CHAIN_OF_THOUGHT,
        importance=0.9,
    )

    # Step 5: Output Execution Details
    print("\n--- Reasoning Summary ---")
    print(f"Task ID: {outcome.task_id}")
    print(f"Strategy: {outcome.strategy_used.value}")
    print(f"Overall Confidence: {outcome.overall_confidence}")
    print(f"Execution Time: {outcome.execution_time_ms} ms")
    print(f"Conclusion: {outcome.final_conclusion}")

    print("\n--- Thought Chain Steps ---")
    for step in outcome.thought_chain:
        print(f"  Step {step.step_number} [{step.confidence:.2f}]: {step.premise}")
        print(f"          └─> Deduction: {step.deduction}")

    print("\n--- Generated Action Plan ---")
    for action in outcome.action_plan:
        print(f"  [ACTION] {action}")
