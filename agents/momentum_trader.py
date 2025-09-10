"""
Momentum Trader Agent for RLlib Demo

This agent implements a momentum trading strategy that identifies and follows
price trends. It showcases RLlib's ability to learn complex temporal patterns
and market timing strategies.
"""

import numpy as np
from typing import Dict, List, Optional, Any
from .base_agent import BaseTradingAgent, AgentConfig, TradingAction, AgentState


class MomentumTraderAgent(BaseTradingAgent):
    """
    Momentum Trader Agent that learns to identify and follow price trends.
    
    This agent learns to:
    - Identify momentum signals from price movements
    - Time entries and exits based on trend strength
    - Manage position sizes based on momentum confidence
    - Adapt to changing market conditions
    """
    
    def __init__(self, agent_id: str, config: AgentConfig):
        """
        Initialize the momentum trader agent.
        
        Args:
            agent_id: Unique identifier for this agent
            config: Agent configuration parameters
        """
        super().__init__(agent_id, config)
        
        # Momentum trading specific parameters
        self.lookback_period = 20
        self.momentum_threshold = 0.05
        self.trend_strength_threshold = 0.02
        
        # Learning parameters
        self.momentum_learning_rate = 0.001
        self.timing_learning_rate = 0.001
        
        # State tracking
        self.price_history = []
        self.momentum_signals = []
        self.trend_directions = []
        self.entry_times = []
        self.exit_times = []
        
        # Momentum trading metrics
        self.trends_identified = 0
        self.successful_trades = 0
        self.momentum_accuracy = 0.0
        
    def select_action(self, observation: np.ndarray, market_data: Dict) -> TradingAction:
        """
        Select momentum trading action based on current market conditions.
        
        Args:
            observation: Current environment observation
            market_data: Current market data
            
        Returns:
            Momentum trading action
        """
        # Extract relevant market information
        current_price = market_data.get("price", 100.0)
        volatility = market_data.get("volatility", 0.02)
        volume = market_data.get("volume", 1000.0)
        
        # Update price history
        self.price_history.append(current_price)
        if len(self.price_history) > self.lookback_period * 2:
            self.price_history.pop(0)
        
        # Calculate momentum signals
        momentum_signal = self._calculate_momentum_signal()
        trend_strength = self._calculate_trend_strength()
        trend_direction = self._determine_trend_direction()
        
        # Store signals for learning
        self.momentum_signals.append(momentum_signal)
        self.trend_directions.append(trend_direction)
        
        # Determine action based on momentum analysis
        if self._should_enter_long(momentum_signal, trend_strength, trend_direction):
            return self._enter_long_position(current_price, trend_strength)
        elif self._should_enter_short(momentum_signal, trend_strength, trend_direction):
            return self._enter_short_position(current_price, trend_strength)
        elif self._should_exit_position(momentum_signal, trend_strength):
            return self._exit_position(current_price)
        else:
            return TradingAction(action_type=0, quantity=0.0, price=None, order_type=0)
    
    def _calculate_momentum_signal(self) -> float:
        """
        Calculate momentum signal from price history.
        
        Returns:
            Momentum signal value (-1 to 1)
        """
        if len(self.price_history) < self.lookback_period:
            return 0.0
        
        # Calculate rate of change
        recent_prices = self.price_history[-self.lookback_period:]
        price_change = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
        
        # Calculate momentum using multiple timeframes
        short_momentum = self._calculate_short_momentum()
        long_momentum = self._calculate_long_momentum()
        
        # Combine momentum signals
        combined_momentum = 0.6 * short_momentum + 0.4 * long_momentum
        
        # Normalize to [-1, 1] range
        return np.tanh(combined_momentum * 10)
    
    def _calculate_short_momentum(self) -> float:
        """Calculate short-term momentum (5 periods)."""
        if len(self.price_history) < 5:
            return 0.0
        
        short_prices = self.price_history[-5:]
        return (short_prices[-1] - short_prices[0]) / short_prices[0]
    
    def _calculate_long_momentum(self) -> float:
        """Calculate long-term momentum (lookback period)."""
        if len(self.price_history) < self.lookback_period:
            return 0.0
        
        long_prices = self.price_history[-self.lookback_period:]
        return (long_prices[-1] - long_prices[0]) / long_prices[0]
    
    def _calculate_trend_strength(self) -> float:
        """
        Calculate the strength of the current trend.
        
        Returns:
            Trend strength value (0 to 1)
        """
        if len(self.price_history) < self.lookback_period:
            return 0.0
        
        # Calculate price volatility
        recent_prices = self.price_history[-self.lookback_period:]
        returns = np.diff(recent_prices) / recent_prices[:-1]
        volatility = np.std(returns)
        
        # Calculate trend consistency
        positive_returns = np.sum(returns > 0)
        negative_returns = np.sum(returns < 0)
        trend_consistency = abs(positive_returns - negative_returns) / len(returns)
        
        # Combine volatility and consistency
        trend_strength = trend_consistency * (1.0 - min(volatility * 10, 1.0))
        
        return min(trend_strength, 1.0)
    
    def _determine_trend_direction(self) -> int:
        """
        Determine the direction of the current trend.
        
        Returns:
            1 for uptrend, -1 for downtrend, 0 for sideways
        """
        if len(self.price_history) < self.lookback_period:
            return 0
        
        recent_prices = self.price_history[-self.lookback_period:]
        price_change = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
        
        if price_change > self.momentum_threshold:
            return 1  # Uptrend
        elif price_change < -self.momentum_threshold:
            return -1  # Downtrend
        else:
            return 0  # Sideways
    
    def _should_enter_long(self, momentum_signal: float, trend_strength: float, 
                          trend_direction: int) -> bool:
        """
        Determine if we should enter a long position.
        
        Args:
            momentum_signal: Current momentum signal
            trend_strength: Current trend strength
            trend_direction: Current trend direction
            
        Returns:
            True if we should enter long
        """
        # Don't enter if we already have a position
        if self.state.position > 0:
            return False
        
        # Check momentum and trend conditions
        momentum_condition = momentum_signal > self.momentum_threshold
        trend_condition = trend_direction == 1 and trend_strength > self.trend_strength_threshold
        
        # Check position size limits
        position_limit = abs(self.state.position) < self.config.max_position_size * 0.8
        
        return momentum_condition and trend_condition and position_limit
    
    def _should_enter_short(self, momentum_signal: float, trend_strength: float, 
                           trend_direction: int) -> bool:
        """
        Determine if we should enter a short position.
        
        Args:
            momentum_signal: Current momentum signal
            trend_strength: Current trend strength
            trend_direction: Current trend direction
            
        Returns:
            True if we should enter short
        """
        # Don't enter if we already have a position
        if self.state.position < 0:
            return False
        
        # Check momentum and trend conditions
        momentum_condition = momentum_signal < -self.momentum_threshold
        trend_condition = trend_direction == -1 and trend_strength > self.trend_strength_threshold
        
        # Check position size limits
        position_limit = abs(self.state.position) < self.config.max_position_size * 0.8
        
        return momentum_condition and trend_condition and position_limit
    
    def _should_exit_position(self, momentum_signal: float, trend_strength: float) -> bool:
        """
        Determine if we should exit current position.
        
        Args:
            momentum_signal: Current momentum signal
            trend_strength: Current trend strength
            
        Returns:
            True if we should exit position
        """
        # No position to exit
        if abs(self.state.position) < 1e-6:
            return False
        
        # Exit if momentum reverses
        if self.state.position > 0 and momentum_signal < -self.momentum_threshold * 0.5:
            return True
        elif self.state.position < 0 and momentum_signal > self.momentum_threshold * 0.5:
            return True
        
        # Exit if trend weakens significantly
        if trend_strength < self.trend_strength_threshold * 0.3:
            return True
        
        return False
    
    def _enter_long_position(self, current_price: float, trend_strength: float) -> TradingAction:
        """
        Enter a long position based on momentum.
        
        Args:
            current_price: Current market price
            trend_strength: Current trend strength
            
        Returns:
            Trading action to enter long position
        """
        # Calculate position size based on trend strength and confidence
        base_size = 100.0
        size_multiplier = min(trend_strength * 2.0, 1.0)
        position_size = base_size * size_multiplier
        
        # Limit position size
        max_size = min(position_size, self.config.max_position_size - abs(self.state.position))
        
        self.entry_times.append(len(self.price_history))
        self.trends_identified += 1
        
        return TradingAction(
            action_type=1,  # Buy
            quantity=max_size,
            price=None,  # Market order
            order_type=0,  # Market order
            confidence=trend_strength,
            metadata={"strategy": "momentum_long", "trend_strength": trend_strength}
        )
    
    def _enter_short_position(self, current_price: float, trend_strength: float) -> TradingAction:
        """
        Enter a short position based on momentum.
        
        Args:
            current_price: Current market price
            trend_strength: Current trend strength
            
        Returns:
            Trading action to enter short position
        """
        # Calculate position size based on trend strength and confidence
        base_size = 100.0
        size_multiplier = min(trend_strength * 2.0, 1.0)
        position_size = base_size * size_multiplier
        
        # Limit position size
        max_size = min(position_size, self.config.max_position_size - abs(self.state.position))
        
        self.entry_times.append(len(self.price_history))
        self.trends_identified += 1
        
        return TradingAction(
            action_type=2,  # Sell
            quantity=max_size,
            price=None,  # Market order
            order_type=0,  # Market order
            confidence=trend_strength,
            metadata={"strategy": "momentum_short", "trend_strength": trend_strength}
        )
    
    def _exit_position(self, current_price: float) -> TradingAction:
        """
        Exit current position.
        
        Args:
            current_price: Current market price
            
        Returns:
            Trading action to exit position
        """
        exit_quantity = abs(self.state.position)
        self.exit_times.append(len(self.price_history))
        
        if self.state.position > 0:
            # Exit long position
            return TradingAction(
                action_type=2,  # Sell
                quantity=exit_quantity,
                price=None,  # Market order
                order_type=0,  # Market order
                confidence=1.0,
                metadata={"strategy": "momentum_exit_long"}
            )
        else:
            # Exit short position
            return TradingAction(
                action_type=1,  # Buy
                quantity=exit_quantity,
                price=None,  # Market order
                order_type=0,  # Market order
                confidence=1.0,
                metadata={"strategy": "momentum_exit_short"}
            )
    
    def update_policy(self, experiences: List[Dict]) -> Dict[str, float]:
        """
        Update momentum trading policy based on recent experiences.
        
        Args:
            experiences: List of experience tuples
            
        Returns:
            Training metrics
        """
        if not experiences:
            return {"loss": 0.0, "momentum_learning": 0.0, "timing_learning": 0.0}
        
        # Extract relevant information from experiences
        rewards = [exp.get("reward", 0.0) for exp in experiences]
        actions = [exp.get("action") for exp in experiences]
        
        # Update momentum detection based on performance
        momentum_learning = self._update_momentum_detection(rewards, actions)
        
        # Update timing strategy based on trade outcomes
        timing_learning = self._update_timing_strategy(rewards, actions)
        
        # Calculate overall loss
        avg_reward = np.mean(rewards) if rewards else 0.0
        loss = -avg_reward
        
        return {
            "loss": loss,
            "momentum_learning": momentum_learning,
            "timing_learning": timing_learning,
            "avg_reward": avg_reward,
            "trends_identified": self.trends_identified,
            "successful_trades": self.successful_trades,
            "momentum_accuracy": self.momentum_accuracy,
        }
    
    def _update_momentum_detection(self, rewards: List[float], actions: List[Dict]) -> float:
        """
        Update momentum detection parameters based on performance.
        
        Args:
            rewards: Recent reward values
            actions: Recent actions taken
            
        Returns:
            Learning progress metric
        """
        if not rewards:
            return 0.0
        
        avg_reward = np.mean(rewards)
        
        # Adjust momentum threshold based on performance
        if avg_reward > 0:
            # Good performance, can be more sensitive
            self.momentum_threshold *= 0.99
        else:
            # Poor performance, need stronger signals
            self.momentum_threshold *= 1.01
        
        # Clamp to valid range
        self.momentum_threshold = max(0.01, min(0.1, self.momentum_threshold))
        
        return abs(avg_reward) * self.momentum_learning_rate
    
    def _update_timing_strategy(self, rewards: List[float], actions: List[Dict]) -> float:
        """
        Update timing strategy based on trade outcomes.
        
        Args:
            rewards: Recent reward values
            actions: Recent actions taken
            
        Returns:
            Learning progress metric
        """
        if not rewards:
            return 0.0
        
        # Update trend strength threshold based on performance
        avg_reward = np.mean(rewards)
        
        if avg_reward > 0:
            # Good performance, can enter on weaker trends
            self.trend_strength_threshold *= 0.99
        else:
            # Poor performance, need stronger trends
            self.trend_strength_threshold *= 1.01
        
        # Clamp to valid range
        self.trend_strength_threshold = max(0.005, min(0.05, self.trend_strength_threshold))
        
        return abs(avg_reward) * self.timing_learning_rate
    
    def _get_agent_specific_features(self, market_data: Dict) -> List[float]:
        """
        Get momentum trader specific features.
        
        Args:
            market_data: Current market data
            
        Returns:
            List of additional features
        """
        features = []
        
        # Momentum features
        if self.momentum_signals:
            features.append(self.momentum_signals[-1])  # Latest momentum signal
            features.append(np.mean(self.momentum_signals[-5:]) if len(self.momentum_signals) >= 5 else 0.0)  # Recent momentum
        else:
            features.extend([0.0, 0.0])
        
        # Trend features
        if self.trend_directions:
            features.append(float(self.trend_directions[-1]))  # Latest trend direction
        else:
            features.append(0.0)
        
        # Trading activity features
        features.append(self.trends_identified / 1000.0)  # Trends identified (normalized)
        features.append(self.successful_trades / max(1, self.trends_identified))  # Success rate
        
        # Position features
        features.append(self.state.position / self.config.max_position_size)  # Normalized position
        
        return features
    
    def _get_agent_specific_metrics(self) -> Dict[str, float]:
        """
        Get momentum trader specific performance metrics.
        
        Returns:
            Dictionary of additional metrics
        """
        return {
            "trends_identified": self.trends_identified,
            "successful_trades": self.successful_trades,
            "momentum_accuracy": self.momentum_accuracy,
            "momentum_threshold": self.momentum_threshold,
            "trend_strength_threshold": self.trend_strength_threshold,
            "lookback_period": self.lookback_period,
            "position_ratio": abs(self.state.position) / self.config.max_position_size,
        }
    
    def _get_checkpoint_data(self) -> Dict:
        """Get momentum trader specific checkpoint data."""
        return {
            "lookback_period": self.lookback_period,
            "momentum_threshold": self.momentum_threshold,
            "trend_strength_threshold": self.trend_strength_threshold,
            "trends_identified": self.trends_identified,
            "successful_trades": self.successful_trades,
            "momentum_accuracy": self.momentum_accuracy,
            "price_history": self.price_history[-100:],  # Keep last 100 prices
            "momentum_signals": self.momentum_signals[-50:],  # Keep last 50 signals
            "trend_directions": self.trend_directions[-50:],  # Keep last 50 directions
        }
    
    def _load_checkpoint_data(self, checkpoint_data: Dict):
        """Load momentum trader specific checkpoint data."""
        self.lookback_period = checkpoint_data.get("lookback_period", 20)
        self.momentum_threshold = checkpoint_data.get("momentum_threshold", 0.05)
        self.trend_strength_threshold = checkpoint_data.get("trend_strength_threshold", 0.02)
        self.trends_identified = checkpoint_data.get("trends_identified", 0)
        self.successful_trades = checkpoint_data.get("successful_trades", 0)
        self.momentum_accuracy = checkpoint_data.get("momentum_accuracy", 0.0)
        self.price_history = checkpoint_data.get("price_history", [])
        self.momentum_signals = checkpoint_data.get("momentum_signals", [])
        self.trend_directions = checkpoint_data.get("trend_directions", [])

