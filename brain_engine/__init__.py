"""
Artificial Cognitive Brain - Brain Engine Module
================================================
This module serves as the core orchestration layer for the Artificial Cognitive Brain.
It initializes key cognitive subsystems including perception, memory, reasoning, 
and action selection interfaces.
"""

import logging
from typing import Dict, Any, Optional

# Set up module-level logging
logger = logging.getLogger(__name__)

# Package metadata
__version__ = "1.0.0"
__author__ = "Cognitive AI Architecture Team"

# Define public API for the brain_engine package
__all__ = [
    "CognitiveEngine",
    "EngineConfig",
    "initialize_engine",
]

class EngineConfig:
    """Configuration class for setting up cognitive processing parameters."""
    def __init__(self, processing_mode: str = "balanced", max_threads: int = 4):
        self.processing_mode = processing_mode
        self.max_threads = max_threads

class CognitiveEngine:
    """
    Main controller class for the Cognitive Brain Engine.
    Orchestrates data flow between perception, memory, and reasoning units.
    """
    def __init__(self, config: Optional[EngineConfig] = None):
        self.config = config or EngineConfig()
        self.is_active = False
        logger.info(f"Cognitive Engine initialized with mode: {self.config.processing_mode}")

    def start(self) -> None:
        """Activates the cognitive processing loop."""
        self.is_active = True
        logger.info("Cognitive Engine loop started.")

    def stop(self) -> None:
        """Gracefully shuts down the cognitive processing loop."""
        self.is_active = False
        logger.info("Cognitive Engine loop stopped.")

def initialize_engine(config_dict: Optional[Dict[str, Any]] = None) -> CognitiveEngine:
    """Helper factory function to quickly instantiate the CognitiveEngine."""
    if config_dict:
        config = EngineConfig(**config_dict)
    else:
        config = EngineConfig()
    return CognitiveEngine(config)
