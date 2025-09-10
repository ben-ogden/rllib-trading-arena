"""
Trading Agent Implementations for RLlib Demo

This package contains different types of trading agents that showcase
various RLlib capabilities including different policy architectures,
action spaces, and learning strategies.
"""

from .market_maker import MarketMakerAgent
from .momentum_trader import MomentumTraderAgent
from .arbitrageur import ArbitrageurAgent
from .base_agent import BaseTradingAgent

__all__ = [
    "BaseTradingAgent",
    "MarketMakerAgent", 
    "MomentumTraderAgent",
    "ArbitrageurAgent"
]

