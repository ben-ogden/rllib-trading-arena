"""
Training Scripts for RLlib Trading Demo

This package contains training scripts that showcase RLlib's latest features
including the new API stack, distributed training, and single-agent capabilities.
"""

from .single_agent_demo import run_single_agent_demo
from .single_agent_evaluation import run_single_agent_evaluation

__all__ = ["run_single_agent_demo", "run_single_agent_evaluation"]

