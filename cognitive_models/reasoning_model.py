"""
Artificial Cognitive Brain - Reasoning Model Module.

This module implements the Reasoning Model Architecture for the Artificial Cognitive Brain.
Modeled after dual-process theory (System 1 fast heuristic intuition vs. System 2 slow deliberative 
logical reasoning), mental model theory (Johnson-Laird), and formal logic inference engines, this 
module handles deductive, inductive, abductive, and security/identity verification processes.

Key Features:
    - LogicFact & LogicRule: Structured representation of logical knowledge propositions and conditional rules.
    - MentalModelEngine: Simulates situational hypothesis generation and counterfactual inconsistency verification.
    - InferenceEngine: Core engine executing Deductive (Modus Ponens), Inductive, and Abductive inferences.
    - ReasoningModelSystem: Primary module coordinating working memory inputs, dual-process logical cycles,
      and rigorous identity/safety authorization pipelines via complete_verified().

Author: Artificial Cognitive Brain Team
License: MIT
Version: 0.2.0
"""

import time
import math
import logging
from typing import Dict, Any, List, Optional, Set, Tuple, Union
from dataclasses import dataclass, field

from cognitive_models import BaseCognitiveModel

# Configure module logger
logger = logging.getLogger(__name__)


@dataclass
class LogicFact:
    """
    Represents a discrete atomic proposition or fact in the cognitive knowledge state.

    Attributes:
        predicate (str): The subject/attribute relationship identifier (e.g., 'is_identity_verified').
        subject (str): Entity or item topic (e.g., 'user_123', 'agent_01').
        object_val (Any): Value, attribute, or target entity bound to the predicate.
        certainty (float): Epistemic belief probability or confidence score [0.0, 1.0].
        timestamp (float): UNIX creation timestamp.
    """
    predicate: str
    subject: str
    object_val: Any
    certainty: float = 1.0
    timestamp: float = field(default_factory=time.time)

    def to_symbolic_str(self) -> str:
        """Returns symbolic representation format: Predicate(Subject, Object)."""
        return f"{self.predicate}({self.subject}, {self.object_val})"


@dataclass
class LogicRule:
    """
    Represents a conditional rule/implication in the knowledge network.
    Format: IF premises THEN conclusion (Confidence: weight)

    Attributes:
        rule_id (str): Unique rule identifier.
        premises (List[Tuple[str, str, Any]]): List of conditions (predicate, subject, object).
        conclusion (Tuple[str, str, Any]): Implied outcome tuple (predicate, subject, object).
        confidence (float): Structural validity or rule probability [0.0, 1.0].
    """
    rule_id: str
    premises: List[Tuple[str, str, Any]]
    conclusion: Tuple[str, str, Any]
    confidence: float = 0.95


@dataclass
class ReasoningOutcome:
    """
    Structured response containing conclusions derived during a reasoning cycle.

    Attributes:
        deduced_facts (List[LogicFact]): Facts strictly derived through deductive rules.
        hypotheses (List[LogicFact]): Inferred candidates via inductive/abductive reasoning.
        inference_chain (List[str]): Audit trail of active rules and facts applied.
        system_type (str): Execution path used ('System1_Heuristic' or 'System2_Deliberative').
        execution_time (float): Computational duration of inference loop in seconds.
    """
    deduced_facts: List[LogicFact] = field(default_factory=list)
    hypotheses: List[LogicFact] = field(default_factory=list)
    inference_chain: List[str] = field(default_factory=list)
    system_type: str = "System1_Heuristic"
    execution_time: float = 0.0


class MentalModelEngine:
    """
    Simulates Johnson-Laird's Mental Model Theory by creating temporary spatial/logical 
    counterfactual models to validate hypotheses and detect contradictions.
    """

    def __init__(self) -> None:
        """Initializes empty temporary simulation worlds."""
        self.active_models: List[Dict[str, Any]] = []

    def verify_consistency(self, facts: List[LogicFact], candidate: LogicFact) -> bool:
        """
        Validates if adding a candidate fact introduces contradictions with current facts.

        Args:
            facts (List[LogicFact]): Known fact base.
            candidate (LogicFact): Candidate fact to test.

        Returns:
            bool: True if consistent, False if a direct logical contradiction is found.
        """
        for fact in facts:
            # Check direct negation / contradictory property assignment
            if (
                fact.subject == candidate.subject and
                fact.predicate == candidate.predicate and
                fact.object_val != candidate.object_val
            ):
                if fact.certainty > 0.8 and candidate.certainty > 0.8:
                    logger.warning(
                        f"Contradiction detected: {fact.to_symbolic_str()} vs {candidate.to_symbolic_str()}"
                    )
                    return False
        return True


class InferenceEngine:
    """
    Core logical processor capable of Deductive, Inductive, and Abductive operations.
    """

    def __init__(self) -> None:
        """Initializes inference rule bases and mental models."""
        self.rules: List[LogicRule] = []
        self.mental_model_engine = MentalModelEngine()

    def add_rule(self, rule: LogicRule) -> None:
        """Registers a new inference rule into the logical knowledge base."""
        self.rules.append(rule)

    def run_deductive_inference(self, fact_base: List[LogicFact]) -> Tuple[List[LogicFact], List[str]]:
        """
        Executes Forward Chaining Deductive Reasoning (Modus Ponens).

        Args:
            fact_base (List[LogicFact]): Current set of validated facts.

        Returns:
            Tuple[List[LogicFact], List[str]]: Newly derived facts and audit trail strings.
        """
        new_facts: List[LogicFact] = []
        audit_trail: List[str] = []

        # Convert fact base into lookup set for quick matching
        fact_tuples = {(f.predicate, f.subject, f.object_val): f.certainty for f in fact_base}

        for rule in self.rules:
            # Check if all premises are satisfied
            all_premises_met = True
            min_certainty = 1.0

            for p_pred, p_sub, p_obj in rule.premises:
                if (p_pred, p_sub, p_obj) in fact_tuples:
                    min_certainty = min(min_certainty, fact_tuples[(p_pred, p_sub, p_obj)])
                else:
                    all_premises_met = False
                    break

            if all_premises_met:
                c_pred, c_sub, c_obj = rule.conclusion
                conclusion_certainty = min_certainty * rule.confidence
                
                derived_fact = LogicFact(
                    predicate=c_pred,
                    subject=c_sub,
                    object_val=c_obj,
                    certainty=conclusion_certainty
                )

                if self.mental_model_engine.verify_consistency(fact_base, derived_fact):
                    new_facts.append(derived_fact)
                    audit_msg = f"Rule [{rule.rule_id}] triggered -> Derived: {derived_fact.to_symbolic_str()}"
                    audit_trail.append(audit_msg)
                    logger.debug(audit_msg)

        return new_facts, audit_trail

    def run_abductive_inference(self, observation: LogicFact) -> List[LogicFact]:
        """
        Executes Abductive Inference (Inference to the Best Explanation).
        Given an observation B, finds potential causes A where A -> B.

        Args:
            observation (LogicFact): Observed target state.

        Returns:
            List[LogicFact]: Candidate explanatory hypotheses.
        """
        hypotheses: List[LogicFact] = []

        for rule in self.rules:
            c_pred, c_sub, c_obj = rule.conclusion
            # Match conclusion with observation
            if (c_pred, c_sub) == (observation.predicate, observation.subject):
                for p_pred, p_sub, p_obj in rule.premises:
                    candidate_hypothesis = LogicFact(
                        predicate=p_pred,
                        subject=p_sub,
                        object_val=p_obj,
                        certainty=observation.certainty * rule.confidence * 0.7  # Explanation penalty
                    )
                    hypotheses.append(candidate_hypothesis)

        return hypotheses


class ReasoningModelSystem(BaseCognitiveModel):
    """
    Reasoning Model Cognitive System for the Artificial Cognitive Brain.

    Coordinates Dual-Process Reasoning (System 1 fast pattern matching vs System 2 deliberative logic),
    executes user identity/safety verification pipelines, and interacts with global working memory inputs.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initializes the Reasoning Model System.

        Args:
            config (Optional[Dict[str, Any]]): System configuration parameters.
        """
        super().__init__(model_name="ReasoningModelSystem", config=config)

        cfg = self.config or {}
        self.system2_threshold = cfg.get("system2_complexity_threshold", 0.7)
        self.risk_threshold = cfg.get("max_acceptable_risk_score", 0.5)
        self.inference_engine = InferenceEngine()
        self.fact_memory: List[LogicFact] = []

        # Bootstrap default basic logic rules
        self._bootstrap_default_rules()

    def _bootstrap_default_rules(self) -> None:
        """Loads baseline domain-agnostic inference rules."""
        rule_1 = LogicRule(
            rule_id="R001_Goal_Obstacle",
            premises=[("status", "goal", "blocked"), ("type", "obstacle", "physical")],
            conclusion=("action_required", "system", "path_replanning"),
            confidence=0.9
        )
        rule_auth = LogicRule(
            rule_id="R002_User_Grant_Access",
            premises=[
                ("is_identity_verified", "user", True),
                ("is_safety_cleared", "user", True),
                ("has_acceptable_risk", "user", True)
            ],
            conclusion=("access_permission", "user", "granted"),
            confidence=1.0
        )
        self.inference_engine.add_rule(rule_1)
        self.inference_engine.add_rule(rule_auth)

    def process_workspace_facts(self, raw_facts: List[Dict[str, Any]]) -> List[LogicFact]:
        """Converts raw input dictionary representations into structured LogicFact objects."""
        parsed_facts: List[LogicFact] = []
        for rf in raw_facts:
            fact = LogicFact(
                predicate=rf.get("predicate", "unknown"),
                subject=rf.get("subject", "entity"),
                object_val=rf.get("object", None),
                certainty=rf.get("certainty", 1.0)
            )
            parsed_facts.append(fact)
        return parsed_facts

    def complete_verified(
        self,
        user_id: str,
        is_identity_verified: bool,
        is_safety_passed: bool,
        verification_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Executes a rigorous cognitive verification pipeline for user identity and safety validation.

        This method injects formal logic facts regarding a user's verification status and safety 
        clearance into the system's working memory, updates the internal truth maintenance graph, 
        evaluates mental model consistency, and synthesizes authorization decisions.

        Args:
            user_id (str): The unique string identifier for the target user entity.
            is_identity_verified (bool): Flag indicating if the user's primary identity claims are validated.
            is_safety_passed (bool): Flag indicating if the user passed all safety and constraint checks.
            verification_metadata (Optional[Dict[str, Any]]): Contextual key-value pairs (e.g., auth_level,
                                                               risk_score, source_session).

        Returns:
            Dict[str, Any]: Detailed logical report containing evaluation status, generated LogicFacts,
                            and the finalized system authorization state.
        """
        start_timestamp = time.time()
        metadata = verification_metadata or {}

        logger.info(f"Initiating complete_verified evaluation for user entity: '{user_id}'")

        # 1. Instantiate symbolic LogicFact representations for Identity and Safety
        identity_fact = LogicFact(
            predicate="is_identity_verified",
            subject=user_id,
            object_val=is_identity_verified,
            certainty=1.0 if is_identity_verified else 0.0,
            timestamp=start_timestamp
        )

        safety_fact = LogicFact(
            predicate="is_safety_cleared",
            subject=user_id,
            object_val=is_safety_passed,
            certainty=1.0 if is_safety_passed else 0.0,
            timestamp=start_timestamp
        )

        # 2. Extract contextual metrics and evaluate risk score bounds
        auth_level = metadata.get("auth_level", "standard")
        risk_score = float(metadata.get("risk_score", 0.0))
        is_risk_acceptable = risk_score < self.risk_threshold

        risk_fact = LogicFact(
            predicate="has_acceptable_risk",
            subject=user_id,
            object_val=is_risk_acceptable,
            certainty=max(0.0, min(1.0, 1.0 - risk_score)),
            timestamp=start_timestamp
        )

        session_fact = LogicFact(
            predicate="active_session_type",
            subject=user_id,
            object_val=metadata.get("source_session", "anonymous_session"),
            certainty=1.0,
            timestamp=start_timestamp
        )

        # 3. Store generated facts in transient memory for mental model consistency checking
        staged_facts = [identity_fact, safety_fact, risk_fact, session_fact]
        consistent_facts: List[LogicFact] = []
        contradiction_occurred = False

        for fact in staged_facts:
            if self.inference_engine.mental_model_engine.verify_consistency(self.fact_memory, fact):
                consistent_facts.append(fact)
                self.fact_memory.append(fact)
            else:
                contradiction_occurred = True
                logger.error(
                    f"Contradiction encountered while registering fact: {fact.to_symbolic_str()}. "
                    "Verification state flagged as compromised."
                )

        # 4. Evaluate authorization logic
        # Rule: User is fully authorized ONLY IF identity is verified AND safety is cleared 
        # AND risk is acceptable AND no logical contradictions exist in working memory.
        fully_verified = (
            is_identity_verified and
            is_safety_passed and
            is_risk_acceptable and
            not contradiction_occurred and
            (len(consistent_facts) == len(staged_facts))
        )

        # 5. Formulate final cognitive decision LogicFact
        authorization_fact = LogicFact(
            predicate="is_fully_authorized",
            subject=user_id,
            object_val=fully_verified,
            certainty=1.0 if fully_verified else 0.0,
            timestamp=time.time()
        )
        self.fact_memory.append(authorization_fact)

        # 6. Execute Deductive Forward Chaining to derive downstream permissions
        deduced_facts, deductive_audit = self.inference_engine.run_deductive_inference(self.fact_memory)
        self.fact_memory.extend(deduced_facts)

        # 7. Audit Trail and System Response Construction
        audit_log = [
            f"Fact Staged: {identity_fact.to_symbolic_str()} (Certainty: {identity_fact.certainty})",
            f"Fact Staged: {safety_fact.to_symbolic_str()} (Certainty: {safety_fact.certainty})",
            f"Fact Staged: {risk_fact.to_symbolic_str()} (Certainty: {risk_fact.certainty})",
            f"Fact Staged: {session_fact.to_symbolic_str()} (Certainty: {session_fact.certainty})",
            f"Mental Model Contradiction Detected: {contradiction_occurred}",
            f"Decision Synthesized: {authorization_fact.to_symbolic_str()}"
        ] + deductive_audit

        logger.info(f"User '{user_id}' verification finished. Fully Authorized: {fully_verified}")

        return {
            "user_id": user_id,
            "status": "VERIFIED_SUCCESS" if fully_verified else "VERIFICATION_FAILED",
            "is_authorized": fully_verified,
            "auth_level": auth_level,
            "risk_score": risk_score,
            "confidence_score": authorization_fact.certainty,
            "asserted_facts": [f.to_symbolic_str() for f in staged_facts],
            "derived_permissions": [f.to_symbolic_str() for f in deduced_facts],
            "decision_fact": authorization_fact.to_symbolic_str(),
            "audit_trail": audit_log,
            "evaluation_time_sec": round(time.time() - start_timestamp, 4)
        }

    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Cognitive Architecture step handler.

        Evaluates active focus items from workspace inputs, triggers System 1 or System 2 
        reasoning pipelines, handles verification requests, and outputs newly deduced logical facts.

        Args:
            inputs (Dict[str, Any]): Global workspace state snapshot.

        Returns:
            Dict[str, Any]: Inferred facts, active hypotheses, and audit logs.
        """
        start_time = time.time()

        # 1. Check for verification command triggers in global workspace
        verification_request = inputs.get("verification_request")
        verification_report = None

        if isinstance(verification_request, dict):
            target_user = verification_request.get("user_id", "unknown_user")
            id_verified = verification_request.get("is_identity_verified", False)
            safety_passed = verification_request.get("is_safety_passed", False)
            v_meta = verification_request.get("metadata", {})

            verification_report = self.complete_verified(
                user_id=target_user,
                is_identity_verified=id_verified,
                is_safety_passed=safety_passed,
                verification_metadata=v_meta
            )

        # 2. Parsing Working Memory facts from global inputs
        wm_inputs = inputs.get("working_memory", {})
        raw_facts = wm_inputs.get("facts", [])

        current_facts = self.process_workspace_facts(raw_facts)
        self.fact_memory.extend(current_facts)

        # 3. Dual-Process Decision Gate (System 1 vs System 2)
        complexity_score = inputs.get("task_complexity", 0.3)
        use_system_2 = complexity_score >= self.system2_threshold or inputs.get("force_deliberation", False)

        outcome = ReasoningOutcome()

        if not use_system_2:
            # System 1: Heuristic / Direct Matching Path
            outcome.system_type = "System1_Heuristic"
            new_deductions, audit = self.inference_engine.run_deductive_inference(self.fact_memory)
            outcome.deduced_facts = new_deductions
            outcome.inference_chain = audit
        else:
            # System 2: Deliberative Reasoning & Abductive Hypothesis Generation
            outcome.system_type = "System2_Deliberative"

            # Step A: Deductive forward chaining
            new_deductions, audit = self.inference_engine.run_deductive_inference(self.fact_memory)
            outcome.deduced_facts = new_deductions
            outcome.inference_chain = audit

            # Step B: Abductive reasoning on unhandled workspace goals
            target_goal = inputs.get("unresolved_goal")
            if target_goal and isinstance(target_goal, dict):
                obs_fact = LogicFact(
                    predicate=target_goal.get("predicate", ""),
                    subject=target_goal.get("subject", ""),
                    object_val=target_goal.get("object", None)
                )
                hypotheses = self.inference_engine.run_abductive_inference(obs_fact)
                outcome.hypotheses = hypotheses
                outcome.inference_chain.append(f"Abductive inference generated {len(hypotheses)} hypotheses.")

        outcome.execution_time = round(time.time() - start_time, 4)

        # 4. Format response for Global Workspace broadcast
        response = {
            "system_mode": outcome.system_type,
            "derived_facts": [f.to_symbolic_str() for f in outcome.deduced_facts],
            "hypotheses": [h.to_symbolic_str() for h in outcome.hypotheses],
            "inference_chain": outcome.inference_chain,
            "execution_time_sec": outcome.execution_time
        }

        if verification_report:
            response["verification_report"] = verification_report

        return response

    def reset_state(self) -> None:
        """Clears transient fact memory and resets model state."""
        self.fact_memory.clear()
        super().reset_state()
