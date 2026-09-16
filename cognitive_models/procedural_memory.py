"""
Artificial Cognitive Brain - Procedural Memory Module.

This module implements the Procedural Memory System for the Artificial Cognitive Brain.
Grounded in cognitive science architectures (such as ACT-R's production system and the 
basal ganglia-striatal motor skill loops), this system stores, retrieves, compiles, 
and executes rule-based procedures, motor schemas, and cognitive skills ("How-To" knowledge).

Key Components:
    - ProductionRule: Condition-action IF-THEN unit with execution utility and priority scores.
    - MotorSchema: Encapsulated parameterized action sequence for motor or behavioral execution.
    - ProductionSystem: Pattern-matching engine handling conflict resolution and rule firing.
    - ProceduralMemorySystem: Core cognitive model managing procedural learning, compilation, 
      and dynamic execution within the Global Workspace.

Author: Artificial Cognitive Brain Team
License: MIT
Version: 0.1.0
"""

import time
import math
import logging
from typing import Dict, Any, List, Optional, Callable, Union, Set
from dataclasses import dataclass, field

from cognitive_models import BaseCognitiveModel

# Configure module logger
logger = logging.getLogger(__name__)


@dataclass
class ProductionRule:
    """
    Represents an individual production rule (IF conditions THEN actions).

    Attributes:
        rule_id (str): Unique identifier for the production rule.
        name (str): Human-readable name describing the cognitive or motor skill.
        conditions (Dict[str, Any]): Pattern dictionary that must match current state/workspace.
        actions (List[Dict[str, Any]]): Sequence of state transitions, internal commands, or external actions.
        utility (float): Reinforcement learning utility score estimating success probability [0.0, 1.0].
        execution_count (int): Number of times this rule has successfully fired.
        latency (float): Estimated execution duration in seconds.
        last_fired_timestamp (float): UNIX timestamp of the last execution.
        metadata (Dict[str, Any]): Operational tags, such as skill domain or compiler source.
    """
    rule_id: str
    name: str
    conditions: Dict[str, Any]
    actions: List[Dict[str, Any]]
    utility: float = 0.5
    execution_count: int = 0
    latency: float = 0.05
    last_fired_timestamp: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def matches(self, state: Dict[str, Any]) -> bool:
        """
        Evaluates whether the provided state matches the production rule's conditions.

        Supports nested dictionary checks, exact value matches, and set inclusions.

        Args:
            state (Dict[str, Any]): Current snapshot of internal cognitive state or Global Workspace.

        Returns:
            bool: True if all condition predicates are satisfied, False otherwise.
        """
        if not self.conditions:
            return False

        for key, expected_val in self.conditions.items():
            actual_val = self._extract_nested_value(state, key)
            if actual_val is None:
                return False

            if isinstance(expected_val, dict) and "operator" in expected_val:
                op = expected_val["operator"]
                val = expected_val["value"]
                if op == "==" and actual_val != val:
                    return False
                elif op == "!=" and actual_val == val:
                    return False
                elif op == ">" and not (actual_val > val):
                    return False
                elif op == "<" and not (actual_val < val):
                    return False
                elif op == "in" and actual_val not in val:
                    return False
            elif actual_val != expected_val:
                return False

        return True

    def update_utility(self, reward: float, learning_rate: float = 0.1) -> float:
        """
        Updates the rule utility score using a Temporal Difference (TD) reinforcement learning update.

        Args:
            reward (float): Environmental or internal reward scalar [-1.0, 1.0].
            learning_rate (float): Step size factor for reinforcement updating.

        Returns:
            float: Updated utility score.
        """
        self.utility += learning_rate * (reward - self.utility)
        self.utility = max(0.0, min(1.0, self.utility))
        return self.utility

    def fire(self) -> List[Dict[str, Any]]:
        """
        Executes the production rule, updating internal metrics and returning the action sequence.

        Returns:
            List[Dict[str, Any]]: The instantiated action sequence to be executed.
        """
        self.execution_count += 1
        self.last_fired_timestamp = time.time()
        logger.info(f"Production Rule Fired: '{self.name}' (ID: {self.rule_id}, Utility: {self.utility:.3f})")
        return self.actions

    def _extract_nested_value(self, data: Dict[str, Any], path: str) -> Any:
        """Helper to extract dot-separated key paths from nested dictionaries."""
        keys = path.split(".")
        current = data
        for k in keys:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                return None
        return current


@dataclass
class MotorSchema:
    """
    Represents an encapsulated motor schema or automated movement sequence.

    Attributes:
        schema_id (str): Unique identifier for the motor schema.
        label (str): Human-readable motor label (e.g., 'reach_and_grasp', 'press_button').
        trajectory_points (List[Dict[str, float]]): Parameterized kinesthetic or motor trajectory points.
        default_parameters (Dict[str, Any]): Default execution variables (e.g., speed, force, precision).
        is_automated (bool): True if schema can execute without continuous working memory supervision.
    """
    schema_id: str
    label: str
    trajectory_points: List[Dict[str, float]] = field(default_factory=list)
    default_parameters: Dict[str, Any] = field(default_factory=dict)
    is_automated: bool = False

    def generate_motor_commands(self, dynamic_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Instantiates specific motor force vectors or movement payloads based on dynamic parameters.

        Args:
            dynamic_params (Optional[Dict[str, Any]]): Overriding parameters for runtime context.

        Returns:
            Dict[str, Any]: Executable motor parameter payload.
        """
        params = {**self.default_parameters, **(dynamic_params or {})}
        return {
            "schema_id": self.schema_id,
            "label": self.label,
            "executed_trajectory": self.trajectory_points,
            "runtime_parameters": params,
            "automated_execution": self.is_automated
        }


class ProductionConflictResolver:
    """
    Conflict Resolution Engine deciding which production rule fires when multiple rules match.
    """

    @staticmethod
    def select_rule(
        candidate_rules: List[ProductionRule], 
        strategy: str = "utility_actr",
        noise_temp: float = 0.05
    ) -> Optional[ProductionRule]:
        """
        Selects a single winning production rule from a list of matching candidates.

        Args:
            candidate_rules (List[ProductionRule]): Rules whose conditions match current state.
            strategy (str): Strategy selection ('utility_actr', 'specificity', or 'recency').
            noise_temp (float): Stochastic noise temperature for ACT-R utility evaluation.

        Returns:
            Optional[ProductionRule]: The winning production rule, or None if candidate list is empty.
        """
        if not candidate_rules:
            return None

        if len(candidate_rules) == 1:
            return candidate_rules[0]

        if strategy == "utility_actr":
            # ACT-R Style Softmax / Boltzmann selection over expected utilities with stochastic noise
            import random
            scored_rules = []
            for r in candidate_rules:
                # Add Boltzmann noise: E = Utility + Noise
                noise = random.gauss(0, noise_temp)
                evaluation_score = r.utility + noise
                scored_rules.append((r, evaluation_score))
            
            scored_rules.sort(key=lambda x: x[1], reverse=True)
            return scored_rules[0][0]

        elif strategy == "specificity":
            # Prefers rules with the highest number of specific condition constraints
            return max(candidate_rules, key=lambda r: (len(r.conditions), r.utility))

        elif strategy == "recency":
            # Prefers rules that have fired least recently to avoid infinite looping
            return min(candidate_rules, key=lambda r: (r.last_fired_timestamp, -r.utility))

        # Default fallback
        return max(candidate_rules, key=lambda r: r.utility)


class ProceduralMemorySystem(BaseCognitiveModel):
    """
    Procedural Memory Cognitive Module for the Artificial Cognitive Brain.
    
    Coordinates the storage, matching, compilation, and execution of production rules 
    and motor schemas. Interacts directly with Global Workspace state to drive automated 
    behavioral cycles.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initializes the Procedural Memory System.

        Args:
            config (Optional[Dict[str, Any]]): Configuration parameters including selection strategy,
                                               learning rates, and procedural compilation thresholds.
        """
        super().__init__(model_name="ProceduralMemorySystem", config=config)

        cfg = self.config or {}
        self.selection_strategy = cfg.get("selection_strategy", "utility_actr")
        self.learning_rate = cfg.get("learning_rate", 0.1)
        self.compilation_threshold = cfg.get("compilation_threshold", 5)

        # Storage Repositories
        self.rules: Dict[str, ProductionRule] = {}
        self.motor_schemas: Dict[str, MotorSchema] = {}
        
        # Execution History for procedural compilation / chunking
        self.firing_history: List[str] = []
        
        # Bootstrap base default skills
        self._bootstrap_default_skills()

    def add_production_rule(
        self, 
        rule_id: str, 
        name: str, 
        conditions: Dict[str, Any], 
        actions: List[Dict[str, Any]], 
        utility: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ProductionRule:
        """
        Registers a new production rule in procedural memory.

        Args:
            rule_id (str): Unique rule string identifier.
            name (str): Readable name of skill or process.
            conditions (Dict[str, Any]): Pattern conditions dictionary.
            actions (List[Dict[str, Any]]): Action payloads to execute upon firing.
            utility (float): Initial utility estimate.
            metadata (Optional[Dict[str, Any]]): Optional diagnostic tags.

        Returns:
            ProductionRule: The created production rule instance.
        """
        rule = ProductionRule(
            rule_id=rule_id,
            name=name,
            conditions=conditions,
            actions=actions,
            utility=utility,
            metadata=metadata or {}
        )
        self.rules[rule_id] = rule
        logger.info(f"Registered Production Rule: '{name}' [{rule_id}]")
        return rule

    def register_motor_schema(self, schema: MotorSchema) -> None:
        """Registers a motor schema for physical or fine motor execution."""
        self.motor_schemas[schema.schema_id] = schema
        logger.info(f"Registered Motor Schema: '{schema.label}' [{schema.schema_id}]")

    def match_and_select(self, current_state: Dict[str, Any]) -> Optional[ProductionRule]:
        """
        Evaluates all rules against current cognitive state and selects one rule via conflict resolution.

        Args:
            current_state (Dict[str, Any]): Current Global Workspace state.

        Returns:
            Optional[ProductionRule]: Winning production rule ready for execution, or None.
        """
        matching_rules = [
            rule for rule in self.rules.values()
            if rule.matches(current_state)
        ]

        if not matching_rules:
            logger.debug("No production rules matched the current workspace state.")
            return None

        selected_rule = ProductionConflictResolver.select_rule(
            matching_rules, 
            strategy=self.selection_strategy
        )
        return selected_rule

    def compile_procedural_chunks((self) -> Optional[ProductionRule]:
        """
        Simulates Production Compilation (ACT-R Chunking).
        
        Combines two frequently co-occurring sequential production rules into a single,
        highly efficient macro production rule, reducing cognitive load.

        Returns:
            Optional[ProductionRule]: Newly compiled macro production rule if compilation occurred.
        """
        if len(self.firing_history) < 2:
            return None

        r1_id, r2_id = self.firing_history[-2], self.firing_history[-1]
        if r1_id not in self.rules or r2_id not in self.rules:
            return None

        r1 = self.rules[r1_id]
        r2 = self.rules[r2_id]

        # Check if sequence repeated enough times to justify compilation
        if r1.execution_count < self.compilation_threshold or r2.execution_count < self.compilation_threshold:
            return None

        macro_id = f"macro_{r1_id}_{r2_id}"
        if macro_id in self.rules:
            return None  # Already compiled

        # Combine conditions and actions
        combined_conditions = {**r1.conditions, **r2.conditions}
        combined_actions = r1.actions + r2.actions
        macro_name = f"Compiled Macro: {r1.name} + {r2.name}"
        initial_utility = min(1.0, max(r1.utility, r2.utility) + 0.1)

        macro_rule = self.add_production_rule(
            rule_id=macro_id,
            name=macro_name,
            conditions=combined_conditions,
            actions=combined_actions,
            utility=initial_utility,
            metadata={"compiled_from": [r1_id, r2_id]}
        )
        logger.info(f"Procedural Compilation Successful! Created Macro Rule '{macro_name}'")
        return macro_rule

    def reward_rule(self, rule_id: str, reward: float) -> None:
        """
        Applies a reinforcement signal reward to a specific rule based on outcome feedback.

        Args:
            rule_id (str): Identifier of the rule to reward.
            reward (float): Reward scalar [-1.0, 1.0].
        """
        if rule_id in self.rules:
            updated_utility = self.rules[rule_id].update_utility(reward, self.learning_rate)
            logger.info(f"Rule '{rule_id}' utility updated to {updated_utility:.3f} (Reward: {reward})")

    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Cognitive Architecture step integration handler.

        Evaluates workspace state, triggers matching procedural rules, executes motor schemas if requested,
        and manages procedural compilation.

        Args:
            inputs (Dict[str, Any]): Global workspace state.

        Returns:
            Dict[str, Any]: Output dictionary containing selected actions, motor outputs, and status.
        """
        # 1. Handle incoming reinforcement learning signals if provided
        procedural_feedback = inputs.get("procedural_reward", {})
        if isinstance(procedural_feedback, dict) and "rule_id" in procedural_feedback:
            self.reward_rule(
                rule_id=procedural_feedback["rule_id"],
                reward=procedural_feedback.get("reward", 0.0)
            )

        # 2. Match and fire production rule based on current state
        selected_rule = self.match_and_select(inputs)
        fired_actions: List[Dict[str, Any]] = []
        fired_rule_info: Optional[Dict[str, Any]] = None
        motor_output: Optional[Dict[str, Any]] = None

        if selected_rule:
            fired_actions = selected_rule.fire()
            self.firing_history.append(selected_rule.rule_id)
            fired_rule_info = {
                "rule_id": selected_rule.rule_id,
                "name": selected_rule.name,
                "utility": round(selected_rule.utility, 3)
            }

            # Check for motor schema execution within fired actions
            for action in fired_actions:
                if action.get("type") == "execute_motor_schema":
                    schema_id = action.get("schema_id")
                    if schema_id in self.motor_schemas:
                        motor_output = self.motor_schemas[schema_id].generate_motor_commands(
                            dynamic_params=action.get("parameters")
                        )

            # 3. Attempt procedural compilation over recent rule sequences
            self.compile_procedural_chunks()

        return {
            "selected_rule": fired_rule_info,
            "actions_to_execute": fired_actions,
            "motor_payload": motor_output,
            "total_rules": len(self.rules),
            "total_motor_schemas": len(self.motor_schemas)
        }

    def reset_state(self) -> None:
        """Resets execution history and transient rule firing statistics."""
        self.firing_history.clear()
        for rule in self.rules.values():
            rule.last_fired_timestamp = 0.0
        super().reset_state()

    def _bootstrap_default_skills((self) -> None:
        """Populates default cognitive skills and motor schemas."""
        # Default Motor Schema
        default_reach = MotorSchema(
            schema_id="schema_reach_default",
            label="Default Reach Trajectory",
            trajectory_points=[{"x": 0.0, "y": 0.0}, {"x": 0.5, "y": 0.5}, {"x": 1.0, "y": 1.0}],
            default_parameters={"speed": 1.0, "force": 0.5},
            is_automated=True
        )
        self.register_motor_schema(default_reach)

        # Default Production Rule: Goal Orienting
        self.add_production_rule(
            rule_id="rule_orient_to_goal",
            name="Orient Attention to Active Goal",
            conditions={
                "sensory_input.has_unprocessed_target": True
            },
            actions=[
                {"type": "wm_command", "action": "focus", "id": "target_object"},
                {
                    "type": "execute_motor_schema", 
                    "schema_id": "schema_reach_default", 
                    "parameters": {"speed": 1.2}
                }
            ],
            utility=0.8
        )
