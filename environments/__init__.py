"""
Multi-Agent Trading Environment for RLlib Demo

This package contains the core trading environment that showcases RLlib's
multi-agent reinforcement learning capabilities in a realistic financial
market simulation.
"""

from .trading_environment import TradingEnvironment
from .market_simulator import MarketSimulator
from .order_book import OrderBook

__all__ = ["TradingEnvironment", "MarketSimulator", "OrderBook"]

