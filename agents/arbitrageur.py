"""
Arbitrageur Agent for RLlib Demo

This agent implements an arbitrage trading strategy that identifies and exploits
price discrepancies. It showcases RLlib's ability to learn optimal execution
timing and risk management in high-frequency trading scenarios.
"""

import numpy as np
from typing import Dict, List, Optional, Any
from .base_agent import BaseTradingAgent, AgentConfig, TradingAction, AgentState


class ArbitrageurAgent(BaseTradingAgent):
    """
    Arbitrageur Agent that learns to identify and exploit price discrepancies.
    
    This agent learns to:
    - Identify arbitrage opportunities across different price levels
    - Execute trades with optimal timing and sizing
    - Manage execution risk and slippage
    - Adapt to changing market microstructure
    """
    
    def __init__(self, agent_id: str, config: AgentConfig):
        """
        Initialize the arbitrageur agent.
        
        Args:
            agent_id: Unique identifier for this agent
            config: Agent configuration parameters
        """
        super().__init__(agent_id, config)
        
        # Arbitrage specific parameters
        self.profit_threshold = 0.01
        self.max_position_size = 500
        self.execution_delay = 1  # Steps to wait before execution
        self.slippage_tolerance = 0.005
        
        # Learning parameters
        self.opportunity_learning_rate = 0.001
        self.execution_learning_rate = 0.001
        
        # State tracking
        self.price_levels = []
        self.arbitrage_opportunities = []
        self.execution_history = []
        self.profit_history = []
        
        # Arbitrage metrics
        self.opportunities_identified = 0
        self.opportunities_executed = 0
        self.total_profit = 0.0
        self.execution_success_rate = 0.0
        
    def select_action(self, observation: np.ndarray, market_data: Dict) -> TradingAction:
        """
        Select arbitrage action based on current market conditions.
        
        Args:
            observation: Current environment observation
            market_data: Current market data
            
        Returns:
            Arbitrage trading action
        """
        # Extract relevant market information
        current_price = market_data.get("price", 100.0)
        best_bid = market_data.get("best_bid")
        best_ask = market_data.get("best_ask")
        spread = market_data.get("spread", 0.01)
        volume = market_data.get("volume", 1000.0)
        
        # Update price levels
        self.price_levels.append({
            "price": current_price,
            "bid": best_bid,
            "ask": best_ask,
            "spread": spread,
            "volume": volume
        })
        
        # Keep only recent price levels
        if len(self.price_levels) > 100:
            self.price_levels.pop(0)
        
        # Identify arbitrage opportunities
        opportunity = self._identify_arbitrage_opportunity()
        
        if opportunity:
            self.opportunities_identified += 1
            self.arbitrage_opportunities.append(opportunity)
            
            # Execute arbitrage if profitable and within risk limits
            if self._should_execute_arbitrage(opportunity):
                return self._execute_arbitrage(opportunity)
        
        # Check for position management
        if abs(self.state.position) > 0:
            return self._manage_position(current_price, spread)
        
        return TradingAction(action_type=0, quantity=0.0, price=None, order_type=0)
    
    def _identify_arbitrage_opportunity(self) -> Optional[Dict[str, Any]]:
        """
        Identify potential arbitrage opportunities.
        
        Returns:
            Dictionary describing the arbitrage opportunity, or None
        """
        if len(self.price_levels) < 2:
            return None
        
        current_level = self.price_levels[-1]
        previous_level = self.price_levels[-2]
        
        # Check for price discrepancies
        price_discrepancy = abs(current_level["price"] - previous_level["price"]) / previous_level["price"]
        
        # Check for spread arbitrage
        if current_level["bid"] and current_level["ask"]:
            spread_arbitrage = current_level["ask"] - current_level["bid"]
            spread_profit = spread_arbitrage / current_level["price"]
        else:
            spread_profit = 0.0
        
        # Check for volume-based opportunities
        volume_change = (current_level["volume"] - previous_level["volume"]) / max(previous_level["volume"], 1.0)
        
        # Identify opportunity type
        opportunity = None
        
        if price_discrepancy > self.profit_threshold:
            # Price discrepancy arbitrage
            opportunity = {
                "type": "price_discrepancy",
                "profit_potential": price_discrepancy,
                "direction": 1 if current_level["price"] > previous_level["price"] else -1,
                "confidence": min(price_discrepancy * 10, 1.0),
                "execution_price": current_level["price"],
                "target_price": previous_level["price"]
            }
        
        elif spread_profit > self.profit_threshold * 0.5:
            # Spread arbitrage
            opportunity = {
                "type": "spread_arbitrage",
                "profit_potential": spread_profit,
                "direction": 0,  # Market making
                "confidence": min(spread_profit * 20, 1.0),
                "execution_price": (current_level["bid"] + current_level["ask"]) / 2,
                "target_price": current_level["ask"] if current_level["bid"] > current_level["ask"] else current_level["bid"]
            }
        
        elif abs(volume_change) > 0.5:
            # Volume-based opportunity
            opportunity = {
                "type": "volume_arbitrage",
                "profit_potential": abs(volume_change) * 0.01,
                "direction": 1 if volume_change > 0 else -1,
                "confidence": min(abs(volume_change), 1.0),
                "execution_price": current_level["price"],
                "target_price": current_level["price"] * (1 + volume_change * 0.01)
            }
        
        return opportunity
    
    def _should_execute_arbitrage(self, opportunity: Dict[str, Any]) -> bool:
        """
        Determine if we should execute the arbitrage opportunity.
        
        Args:
            opportunity: Arbitrage opportunity details
            
        Returns:
            True if we should execute
        """
        # Check profit potential
        if opportunity["profit_potential"] < self.profit_threshold:
            return False
        
        # Check confidence level
        if opportunity["confidence"] < 0.3:
            return False
        
        # Check position limits
        if abs(self.state.position) >= self.max_position_size * 0.8:
            return False
        
        # Check execution risk
        execution_risk = self._calculate_execution_risk(opportunity)
        if execution_risk > self.slippage_tolerance:
            return False
        
        return True
    
    def _calculate_execution_risk(self, opportunity: Dict[str, Any]) -> float:
        """
        Calculate execution risk for the arbitrage opportunity.
        
        Args:
            opportunity: Arbitrage opportunity details
            
        Returns:
            Execution risk value
        """
        # Base risk from market volatility
        if len(self.price_levels) >= 10:
            recent_prices = [level["price"] for level in self.price_levels[-10:]]
            price_volatility = np.std(recent_prices) / np.mean(recent_prices)
        else:
            price_volatility = 0.01
        
        # Risk from opportunity type
        if opportunity["type"] == "price_discrepancy":
            type_risk = 0.002
        elif opportunity["type"] == "spread_arbitrage":
            type_risk = 0.001
        else:  # volume_arbitrage
            type_risk = 0.003
        
        # Risk from position size
        position_risk = abs(self.state.position) / self.max_position_size * 0.001
        
        total_risk = price_volatility + type_risk + position_risk
        
        return min(total_risk, 0.01)  # Cap at 1%
    
    def _execute_arbitrage(self, opportunity: Dict[str, Any]) -> TradingAction:
        """
        Execute the arbitrage opportunity.
        
        Args:
            opportunity: Arbitrage opportunity details
            
        Returns:
            Trading action to execute arbitrage
        """
        # Calculate position size based on opportunity and risk
        base_size = 50.0  # Conservative base size
        confidence_multiplier = opportunity["confidence"]
        risk_multiplier = 1.0 - self._calculate_execution_risk(opportunity) * 100
        
        position_size = base_size * confidence_multiplier * risk_multiplier
        
        # Limit position size
        max_size = min(position_size, self.max_position_size - abs(self.state.position))
        
        # Determine action based on opportunity type
        if opportunity["type"] == "spread_arbitrage":
            # Market making - place both buy and sell orders
            # For simplicity, we'll place a buy order
            action_type = 1  # Buy
            order_type = 1  # Limit order
            price = opportunity["execution_price"] - 0.001  # Slightly better price
        else:
            # Directional arbitrage
            if opportunity["direction"] > 0:
                action_type = 1  # Buy
            else:
                action_type = 2  # Sell
            
            order_type = 0  # Market order for faster execution
            price = None
        
        self.opportunities_executed += 1
        
        return TradingAction(
            action_type=action_type,
            quantity=max_size,
            price=price,
            order_type=order_type,
            confidence=opportunity["confidence"],
            metadata={
                "opportunity_type": opportunity["type"],
                "profit_potential": opportunity["profit_potential"],
                "execution_risk": self._calculate_execution_risk(opportunity)
            }
        )
    
    def _manage_position(self, current_price: float, spread: float) -> TradingAction:
        """
        Manage existing position to lock in profits or limit losses.
        
        Args:
            current_price: Current market price
            spread: Current bid-ask spread
            
        Returns:
            Trading action to manage position
        """
        # Calculate unrealized P&L
        if self.state.position > 0:
            # Long position
            unrealized_pnl = (current_price - self._get_average_entry_price()) * self.state.position
        else:
            # Short position
            unrealized_pnl = (self._get_average_entry_price() - current_price) * abs(self.state.position)
        
        # Check for profit taking
        profit_threshold = self.profit_threshold * 2  # Take profit at 2x threshold
        if unrealized_pnl > profit_threshold * self.state.position:
            # Take profit
            return self._close_position(current_price, "profit_taking")
        
        # Check for stop loss
        loss_threshold = -self.profit_threshold * self.state.position
        if unrealized_pnl < loss_threshold:
            # Stop loss
            return self._close_position(current_price, "stop_loss")
        
        # Check for spread-based exit
        if spread > self.profit_threshold * 3:
            # Spread too wide, exit position
            return self._close_position(current_price, "spread_exit")
        
        return TradingAction(action_type=0, quantity=0.0, price=None, order_type=0)
    
    def _get_average_entry_price(self) -> float:
        """Get average entry price for current position."""
        # Simplified - in reality, this would track individual trade prices
        if len(self.price_levels) > 0:
            return self.price_levels[-1]["price"]
        return 100.0
    
    def _close_position(self, current_price: float, reason: str) -> TradingAction:
        """
        Close current position.
        
        Args:
            current_price: Current market price
            reason: Reason for closing position
            
        Returns:
            Trading action to close position
        """
        close_quantity = abs(self.state.position)
        
        if self.state.position > 0:
            # Close long position
            return TradingAction(
                action_type=2,  # Sell
                quantity=close_quantity,
                price=None,  # Market order
                order_type=0,  # Market order
                confidence=1.0,
                metadata={"close_reason": reason}
            )
        else:
            # Close short position
            return TradingAction(
                action_type=1,  # Buy
                quantity=close_quantity,
                price=None,  # Market order
                order_type=0,  # Market order
                confidence=1.0,
                metadata={"close_reason": reason}
            )
    
    def update_policy(self, experiences: List[Dict]) -> Dict[str, float]:
        """
        Update arbitrage policy based on recent experiences.
        
        Args:
            experiences: List of experience tuples
            
        Returns:
            Training metrics
        """
        if not experiences:
            return {"loss": 0.0, "opportunity_learning": 0.0, "execution_learning": 0.0}
        
        # Extract relevant information from experiences
        rewards = [exp.get("reward", 0.0) for exp in experiences]
        actions = [exp.get("action") for exp in experiences]
        
        # Update opportunity detection based on performance
        opportunity_learning = self._update_opportunity_detection(rewards, actions)
        
        # Update execution strategy based on trade outcomes
        execution_learning = self._update_execution_strategy(rewards, actions)
        
        # Calculate overall loss
        avg_reward = np.mean(rewards) if rewards else 0.0
        loss = -avg_reward
        
        # Update success rate
        if self.opportunities_executed > 0:
            self.execution_success_rate = self.successful_trades / self.opportunities_executed
        
        return {
            "loss": loss,
            "opportunity_learning": opportunity_learning,
            "execution_learning": execution_learning,
            "avg_reward": avg_reward,
            "opportunities_identified": self.opportunities_identified,
            "opportunities_executed": self.opportunities_executed,
            "execution_success_rate": self.execution_success_rate,
            "total_profit": self.total_profit,
        }
    
    def _update_opportunity_detection(self, rewards: List[float], actions: List[Dict]) -> float:
        """
        Update opportunity detection parameters based on performance.
        
        Args:
            rewards: Recent reward values
            actions: Recent actions taken
            
        Returns:
            Learning progress metric
        """
        if not rewards:
            return 0.0
        
        avg_reward = np.mean(rewards)
        
        # Adjust profit threshold based on performance
        if avg_reward > 0:
            # Good performance, can be more selective
            self.profit_threshold *= 1.01
        else:
            # Poor performance, need more opportunities
            self.profit_threshold *= 0.99
        
        # Clamp to valid range
        self.profit_threshold = max(0.005, min(0.05, self.profit_threshold))
        
        return abs(avg_reward) * self.opportunity_learning_rate
    
    def _update_execution_strategy(self, rewards: List[float], actions: List[Dict]) -> float:
        """
        Update execution strategy based on trade outcomes.
        
        Args:
            rewards: Recent reward values
            actions: Recent actions taken
            
        Returns:
            Learning progress metric
        """
        if not rewards:
            return 0.0
        
        # Update slippage tolerance based on performance
        avg_reward = np.mean(rewards)
        
        if avg_reward > 0:
            # Good performance, can accept more slippage
            self.slippage_tolerance *= 1.01
        else:
            # Poor performance, need tighter execution
            self.slippage_tolerance *= 0.99
        
        # Clamp to valid range
        self.slippage_tolerance = max(0.001, min(0.01, self.slippage_tolerance))
        
        return abs(avg_reward) * self.execution_learning_rate
    
    def _get_agent_specific_features(self, market_data: Dict) -> List[float]:
        """
        Get arbitrageur specific features.
        
        Args:
            market_data: Current market data
            
        Returns:
            List of additional features
        """
        features = []
        
        # Opportunity features
        features.append(self.opportunities_identified / 1000.0)  # Opportunities identified (normalized)
        features.append(self.opportunities_executed / max(1, self.opportunities_identified))  # Execution rate
        
        # Profit features
        features.append(self.total_profit / 1000.0)  # Total profit (normalized)
        features.append(self.execution_success_rate)  # Success rate
        
        # Risk features
        features.append(self.profit_threshold)  # Current profit threshold
        features.append(self.slippage_tolerance)  # Current slippage tolerance
        
        # Position features
        features.append(self.state.position / self.max_position_size)  # Normalized position
        
        return features
    
    def _get_agent_specific_metrics(self) -> Dict[str, float]:
        """
        Get arbitrageur specific performance metrics.
        
        Returns:
            Dictionary of additional metrics
        """
        return {
            "opportunities_identified": self.opportunities_identified,
            "opportunities_executed": self.opportunities_executed,
            "execution_success_rate": self.execution_success_rate,
            "total_profit": self.total_profit,
            "profit_threshold": self.profit_threshold,
            "slippage_tolerance": self.slippage_tolerance,
            "max_position_size": self.max_position_size,
            "position_ratio": abs(self.state.position) / self.max_position_size,
        }
    
    def _get_checkpoint_data(self) -> Dict:
        """Get arbitrageur specific checkpoint data."""
        return {
            "profit_threshold": self.profit_threshold,
            "max_position_size": self.max_position_size,
            "slippage_tolerance": self.slippage_tolerance,
            "opportunities_identified": self.opportunities_identified,
            "opportunities_executed": self.opportunities_executed,
            "total_profit": self.total_profit,
            "execution_success_rate": self.execution_success_rate,
            "price_levels": self.price_levels[-50:],  # Keep last 50 price levels
            "arbitrage_opportunities": self.arbitrage_opportunities[-20:],  # Keep last 20 opportunities
        }
    
    def _load_checkpoint_data(self, checkpoint_data: Dict):
        """Load arbitrageur specific checkpoint data."""
        self.profit_threshold = checkpoint_data.get("profit_threshold", 0.01)
        self.max_position_size = checkpoint_data.get("max_position_size", 500)
        self.slippage_tolerance = checkpoint_data.get("slippage_tolerance", 0.005)
        self.opportunities_identified = checkpoint_data.get("opportunities_identified", 0)
        self.opportunities_executed = checkpoint_data.get("opportunities_executed", 0)
        self.total_profit = checkpoint_data.get("total_profit", 0.0)
        self.execution_success_rate = checkpoint_data.get("execution_success_rate", 0.0)
        self.price_levels = checkpoint_data.get("price_levels", [])
        self.arbitrage_opportunities = checkpoint_data.get("arbitrage_opportunities", [])

