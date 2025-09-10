"""
Training Scripts for RLlib Trading Demo

This package contains training scripts that showcase RLlib's latest features
including the new API stack, distributed training, and multi-agent capabilities.
"""

from .multi_agent_trainer import MultiAgentTrainer
from .single_agent_trainer import SingleAgentTrainer
from .config_manager import ConfigManager

__all__ = ["MultiAgentTrainer", "SingleAgentTrainer", "ConfigManager"]

