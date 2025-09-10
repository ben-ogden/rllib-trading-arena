"""
Base Trading Agent for RLlib Demo

This module provides the base class for all trading agents, defining
common interfaces and utilities that all agent types will inherit.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from dataclasses import dataclass


@dataclass
class AgentConfig:
    """Configuration for a trading agent."""
    agent_type: str
    initial_capital: float
    risk_tolerance: float
    max_position_size: float
    learning_rate: float = 0.0003
    exploration_rate: float = 0.1
    memory_size: int = 10000
    batch_size: int = 32


@dataclass
class TradingAction:
    """Represents a trading action taken by an agent."""
    action_type: int  # 0=hold, 1=buy, 2=sell, 3=cancel
    quantity: float
    price: Optional[float]
    order_type: int  # 0=market, 1=limit
    confidence: float = 1.0
    metadata: Optional[Dict] = None


@dataclass
class AgentState:
    """Current state of an agent."""
    cash: float
    position: float
    pnl: float
    total_trades: int
    active_orders: List[str]
    performance_metrics: Dict[str, float]
    last_action: Optional[TradingAction] = None


class BaseTradingAgent(ABC):
    """
    Base class for all trading agents in the RLlib demo.
    
    This class defines the common interface and utilities that all
    trading agents must implement, providing a foundation for different
    trading strategies and RLlib integration patterns.
    """
    
    def __init__(self, agent_id: str, config: AgentConfig):
        """
        Initialize the base trading agent.
        
        Args:
            agent_id: Unique identifier for this agent
            config: Agent configuration parameters
        """
        self.agent_id = agent_id
        self.config = config
        self.state = AgentState(
            cash=config.initial_capital,
            position=0.0,
            pnl=0.0,
            total_trades=0,
            active_orders=[],
            performance_metrics={}
        )
        
        # Learning and memory
        self.experience_buffer = []
        self.learning_step = 0
        self.episode_rewards = []
        self.episode_actions = []
        
        # Performance tracking
        self.total_reward = 0.0
        self.best_performance = -np.inf
        self.performance_history = []
        
    @abstractmethod
    def select_action(self, observation: np.ndarray, market_data: Dict) -> TradingAction:
        """
        Select an action based on current observation and market data.
        
        Args:
            observation: Current environment observation
            market_data: Current market conditions
            
        Returns:
            Trading action to execute
        """
        pass
    
    @abstractmethod
    def update_policy(self, experiences: List[Dict]) -> Dict[str, float]:
        """
        Update the agent's policy based on recent experiences.
        
        Args:
            experiences: List of experience tuples (state, action, reward, next_state)
            
        Returns:
            Dictionary of training metrics
        """
        pass
    
    def process_reward(self, reward: float, info: Dict) -> float:
        """
        Process and potentially modify the reward signal.
        
        Args:
            reward: Raw reward from environment
            info: Additional environment information
            
        Returns:
            Processed reward value
        """
        # Add risk-based reward shaping
        risk_penalty = self._calculate_risk_penalty()
        performance_bonus = self._calculate_performance_bonus()
        
        processed_reward = reward + risk_penalty + performance_bonus
        
        # Update tracking
        self.total_reward += processed_reward
        self.episode_rewards.append(processed_reward)
        
        return processed_reward
    
    def _calculate_risk_penalty(self) -> float:
        """Calculate risk-based penalty for current position."""
        position_risk = abs(self.state.position) / self.config.max_position_size
        return -self.config.risk_tolerance * (position_risk ** 2)
    
    def _calculate_performance_bonus(self) -> float:
        """Calculate performance-based bonus."""
        if len(self.episode_rewards) < 10:
            return 0.0
        
        # Bonus for consistent positive performance
        recent_rewards = self.episode_rewards[-10:]
        if all(r > 0 for r in recent_rewards):
            return 0.1
        
        return 0.0
    
    def update_state(self, new_state: Dict):
        """
        Update agent state with new information.
        
        Args:
            new_state: Dictionary containing updated state information
        """
        if "cash" in new_state:
            self.state.cash = new_state["cash"]
        if "position" in new_state:
            self.state.position = new_state["position"]
        if "pnl" in new_state:
            self.state.pnl = new_state["pnl"]
        if "total_trades" in new_state:
            self.state.total_trades = new_state["total_trades"]
        if "active_orders" in new_state:
            self.state.active_orders = new_state["active_orders"]
        if "performance_metrics" in new_state:
            self.state.performance_metrics.update(new_state["performance_metrics"])
    
    def get_observation_features(self, market_data: Dict) -> np.ndarray:
        """
        Extract relevant features for the agent's observation.
        
        Args:
            market_data: Current market data
            
        Returns:
            Feature vector for the agent
        """
        features = []
        
        # Market features
        features.extend([
            market_data.get("price", 0.0) / 100.0,  # Normalized price
            market_data.get("volatility", 0.0) * 100,  # Volatility
            market_data.get("liquidity", 0.0),  # Liquidity
            market_data.get("volume", 0.0) / 10000.0,  # Normalized volume
            market_data.get("spread", 0.0) / 100.0,  # Normalized spread
        ])
        
        # Agent-specific features
        features.extend([
            self.state.cash / 100000.0,  # Normalized cash
            self.state.position / 1000.0,  # Normalized position
            self.state.pnl / 1000.0,  # Normalized PnL
            len(self.state.active_orders) / 10.0,  # Normalized active orders
        ])
        
        # Agent-type specific features
        features.extend(self._get_agent_specific_features(market_data))
        
        return np.array(features, dtype=np.float32)
    
    @abstractmethod
    def _get_agent_specific_features(self, market_data: Dict) -> List[float]:
        """
        Get agent-type specific features.
        
        Args:
            market_data: Current market data
            
        Returns:
            List of additional features specific to this agent type
        """
        pass
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """
        Get comprehensive performance metrics for this agent.
        
        Returns:
            Dictionary of performance metrics
        """
        metrics = {
            "total_reward": self.total_reward,
            "episode_reward": sum(self.episode_rewards) if self.episode_rewards else 0.0,
            "average_reward": np.mean(self.episode_rewards) if self.episode_rewards else 0.0,
            "total_trades": self.state.total_trades,
            "current_pnl": self.state.pnl,
            "cash": self.state.cash,
            "position": self.state.position,
            "active_orders": len(self.state.active_orders),
            "learning_step": self.learning_step,
        }
        
        # Add agent-specific metrics
        metrics.update(self._get_agent_specific_metrics())
        
        return metrics
    
    @abstractmethod
    def _get_agent_specific_metrics(self) -> Dict[str, float]:
        """
        Get agent-type specific performance metrics.
        
        Returns:
            Dictionary of additional metrics specific to this agent type
        """
        pass
    
    def reset_episode(self):
        """Reset agent state for a new episode."""
        self.episode_rewards = []
        self.episode_actions = []
        self.learning_step += 1
        
        # Update performance history
        if self.episode_rewards:
            episode_performance = sum(self.episode_rewards)
            self.performance_history.append(episode_performance)
            self.best_performance = max(self.best_performance, episode_performance)
    
    def save_checkpoint(self, checkpoint_path: str):
        """
        Save agent state to checkpoint.
        
        Args:
            checkpoint_path: Path to save checkpoint
        """
        checkpoint_data = {
            "agent_id": self.agent_id,
            "config": self.config,
            "state": self.state,
            "total_reward": self.total_reward,
            "learning_step": self.learning_step,
            "performance_history": self.performance_history,
            "best_performance": self.best_performance,
        }
        
        # Add agent-specific checkpoint data
        checkpoint_data.update(self._get_checkpoint_data())
        
        import pickle
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint_data, f)
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        Load agent state from checkpoint.
        
        Args:
            checkpoint_path: Path to load checkpoint from
        """
        import pickle
        with open(checkpoint_path, 'rb') as f:
            checkpoint_data = pickle.load(f)
        
        self.agent_id = checkpoint_data["agent_id"]
        self.config = checkpoint_data["config"]
        self.state = checkpoint_data["state"]
        self.total_reward = checkpoint_data["total_reward"]
        self.learning_step = checkpoint_data["learning_step"]
        self.performance_history = checkpoint_data["performance_history"]
        self.best_performance = checkpoint_data["best_performance"]
        
        # Load agent-specific checkpoint data
        self._load_checkpoint_data(checkpoint_data)
    
    @abstractmethod
    def _get_checkpoint_data(self) -> Dict:
        """Get agent-specific data for checkpointing."""
        pass
    
    @abstractmethod
    def _load_checkpoint_data(self, checkpoint_data: Dict):
        """Load agent-specific data from checkpoint."""
        pass

