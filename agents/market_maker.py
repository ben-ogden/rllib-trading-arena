"""
Market Maker Agent for RLlib Demo

This agent implements a market making strategy that provides liquidity
by continuously placing buy and sell orders around the current market price.
It showcases RLlib's ability to learn optimal bid-ask spreads and inventory management.
"""

import numpy as np
from typing import Dict, List, Optional, Any
from .base_agent import BaseTradingAgent, AgentConfig, TradingAction, AgentState


class MarketMakerAgent(BaseTradingAgent):
    """
    Market Maker Agent that learns to provide liquidity optimally.
    
    This agent learns to:
    - Set optimal bid-ask spreads based on market conditions
    - Manage inventory risk by adjusting position sizes
    - Respond to volatility and liquidity changes
    - Compete with other market makers
    """
    
    def __init__(self, agent_id: str, config: AgentConfig):
        """
        Initialize the market maker agent.
        
        Args:
            agent_id: Unique identifier for this agent
            config: Agent configuration parameters
        """
        super().__init__(agent_id, config)
        
        # Market making specific parameters
        self.inventory_target = 0.0
        self.max_inventory = config.max_position_size * 0.5
        self.min_spread = 0.01
        self.max_spread = 0.05
        
        # Learning parameters
        self.spread_learning_rate = 0.001
        self.inventory_learning_rate = 0.001
        
        # State tracking
        self.recent_spreads = []
        self.recent_inventory = []
        self.profit_history = []
        
        # Market making metrics
        self.quotes_placed = 0
        self.quotes_filled = 0
        self.inventory_risk = 0.0
        
    def select_action(self, observation: np.ndarray, market_data: Dict) -> TradingAction:
        """
        Select market making action based on current market conditions.
        
        Args:
            observation: Current environment observation
            market_data: Current market data
            
        Returns:
            Market making action
        """
        # Extract relevant market information
        current_price = market_data.get("price", 100.0)
        volatility = market_data.get("volatility", 0.02)
        liquidity = market_data.get("liquidity", 0.1)
        spread = market_data.get("spread", 0.01)
        
        # Calculate optimal spread based on market conditions
        optimal_spread = self._calculate_optimal_spread(volatility, liquidity)
        
        # Calculate inventory adjustment
        inventory_adjustment = self._calculate_inventory_adjustment()
        
        # Determine action based on current state
        if self._should_place_quotes(current_price, optimal_spread):
            return self._place_quotes(current_price, optimal_spread, inventory_adjustment)
        elif self._should_adjust_inventory():
            return self._adjust_inventory(current_price)
        else:
            return TradingAction(action_type=0, quantity=0.0, price=None, order_type=0)
    
    def _calculate_optimal_spread(self, volatility: float, liquidity: float) -> float:
        """
        Calculate optimal bid-ask spread based on market conditions.
        
        Args:
            volatility: Current market volatility
            liquidity: Current market liquidity
            
        Returns:
            Optimal spread value
        """
        # Base spread increases with volatility
        base_spread = self.min_spread + volatility * 2.0
        
        # Adjust for liquidity (lower liquidity = higher spread)
        liquidity_adjustment = 1.0 / (1.0 + liquidity * 10)
        
        # Adjust for inventory risk
        inventory_risk = abs(self.state.position) / self.max_inventory
        inventory_adjustment = 1.0 + inventory_risk * 0.5
        
        optimal_spread = base_spread * liquidity_adjustment * inventory_adjustment
        
        # Clamp to valid range
        return max(self.min_spread, min(self.max_spread, optimal_spread))
    
    def _calculate_inventory_adjustment(self) -> float:
        """
        Calculate how much to adjust inventory based on current position.
        
        Returns:
            Inventory adjustment factor (-1 to 1)
        """
        if abs(self.state.position) < self.max_inventory * 0.1:
            return 0.0  # No adjustment needed
        
        # Calculate desired adjustment
        inventory_ratio = self.state.position / self.max_inventory
        adjustment = -np.sign(inventory_ratio) * min(abs(inventory_ratio), 0.5)
        
        return adjustment
    
    def _should_place_quotes(self, current_price: float, optimal_spread: float) -> bool:
        """
        Determine if we should place new quotes.
        
        Args:
            current_price: Current market price
            optimal_spread: Calculated optimal spread
            
        Returns:
            True if we should place quotes
        """
        # Don't place quotes if we have too many active orders
        if len(self.state.active_orders) >= 4:
            return False
        
        # Don't place quotes if inventory is too extreme
        if abs(self.state.position) > self.max_inventory * 0.8:
            return False
        
        # Don't place quotes if spread is too tight (not profitable)
        if optimal_spread < self.min_spread * 1.5:
            return False
        
        return True
    
    def _place_quotes(self, current_price: float, optimal_spread: float, 
                     inventory_adjustment: float) -> TradingAction:
        """
        Place buy and sell quotes around current price.
        
        Args:
            current_price: Current market price
            optimal_spread: Calculated optimal spread
            inventory_adjustment: Inventory adjustment factor
            
        Returns:
            Trading action to place quotes
        """
        # Calculate quote prices
        half_spread = optimal_spread / 2.0
        bid_price = current_price - half_spread
        ask_price = current_price + half_spread
        
        # Adjust for inventory (favor the side that reduces inventory)
        if inventory_adjustment > 0:  # Want to sell more
            ask_price -= half_spread * 0.3  # Make ask more attractive
        elif inventory_adjustment < 0:  # Want to buy more
            bid_price += half_spread * 0.3  # Make bid more attractive
        
        # Calculate quote size based on inventory and market conditions
        base_size = 100.0
        inventory_factor = 1.0 - abs(self.state.position) / self.max_inventory
        quote_size = base_size * inventory_factor
        
        # For simplicity, we'll place a buy order (bid)
        # In a full implementation, we'd place both buy and sell orders
        self.quotes_placed += 1
        
        return TradingAction(
            action_type=1,  # Buy
            quantity=quote_size,
            price=bid_price,
            order_type=1,  # Limit order
            confidence=0.8,
            metadata={"quote_type": "bid", "spread": optimal_spread}
        )
    
    def _should_adjust_inventory(self) -> bool:
        """
        Determine if we should actively adjust our inventory.
        
        Returns:
            True if we should adjust inventory
        """
        # Adjust if inventory is too extreme
        if abs(self.state.position) > self.max_inventory * 0.7:
            return True
        
        # Adjust if we've been holding a position too long
        if len(self.recent_inventory) > 10:
            recent_avg = np.mean(self.recent_inventory[-10:])
            if abs(recent_avg) > self.max_inventory * 0.5:
                return True
        
        return False
    
    def _adjust_inventory(self, current_price: float) -> TradingAction:
        """
        Adjust inventory by placing a market order.
        
        Args:
            current_price: Current market price
            
        Returns:
            Trading action to adjust inventory
        """
        # Calculate how much to adjust
        target_reduction = self.state.position * 0.5  # Reduce by half
        
        if target_reduction > 0:
            # Sell to reduce long position
            return TradingAction(
                action_type=2,  # Sell
                quantity=abs(target_reduction),
                price=None,  # Market order
                order_type=0,  # Market order
                confidence=0.9,
                metadata={"adjustment": "inventory_reduction"}
            )
        else:
            # Buy to reduce short position
            return TradingAction(
                action_type=1,  # Buy
                quantity=abs(target_reduction),
                price=None,  # Market order
                order_type=0,  # Market order
                confidence=0.9,
                metadata={"adjustment": "inventory_reduction"}
            )
    
    def update_policy(self, experiences: List[Dict]) -> Dict[str, float]:
        """
        Update market making policy based on recent experiences.
        
        Args:
            experiences: List of experience tuples
            
        Returns:
            Training metrics
        """
        if not experiences:
            return {"loss": 0.0, "spread_learning": 0.0, "inventory_learning": 0.0}
        
        # Extract relevant information from experiences
        rewards = [exp.get("reward", 0.0) for exp in experiences]
        actions = [exp.get("action") for exp in experiences]
        
        # Update spread learning based on profitability
        spread_learning = self._update_spread_policy(rewards, actions)
        
        # Update inventory management based on risk
        inventory_learning = self._update_inventory_policy(rewards, actions)
        
        # Calculate overall loss (simplified)
        avg_reward = np.mean(rewards) if rewards else 0.0
        loss = -avg_reward  # Negative reward as loss
        
        return {
            "loss": loss,
            "spread_learning": spread_learning,
            "inventory_learning": inventory_learning,
            "avg_reward": avg_reward,
            "quotes_placed": self.quotes_placed,
            "quotes_filled": self.quotes_filled,
        }
    
    def _update_spread_policy(self, rewards: List[float], actions: List[Dict]) -> float:
        """
        Update spread setting policy based on recent performance.
        
        Args:
            rewards: Recent reward values
            actions: Recent actions taken
            
        Returns:
            Learning progress metric
        """
        if not rewards or not actions:
            return 0.0
        
        # Simple policy update: adjust spread based on profitability
        avg_reward = np.mean(rewards)
        
        if avg_reward > 0:
            # Profitable, can afford tighter spreads
            self.min_spread *= 0.99
        else:
            # Unprofitable, need wider spreads
            self.min_spread *= 1.01
        
        # Clamp to valid range
        self.min_spread = max(0.005, min(0.05, self.min_spread))
        
        return abs(avg_reward) * self.spread_learning_rate
    
    def _update_inventory_policy(self, rewards: List[float], actions: List[Dict]) -> float:
        """
        Update inventory management policy based on recent performance.
        
        Args:
            rewards: Recent reward values
            actions: Recent actions taken
            
        Returns:
            Learning progress metric
        """
        if not rewards:
            return 0.0
        
        # Update inventory risk based on recent performance
        recent_inventory = [abs(self.state.position) for _ in rewards]
        avg_inventory_risk = np.mean(recent_inventory) / self.max_inventory
        
        # Adjust max inventory based on performance
        if np.mean(rewards) > 0:
            # Good performance, can handle more inventory
            self.max_inventory *= 1.001
        else:
            # Poor performance, reduce inventory risk
            self.max_inventory *= 0.999
        
        # Clamp to valid range
        self.max_inventory = max(100.0, min(1000.0, self.max_inventory))
        
        return avg_inventory_risk * self.inventory_learning_rate
    
    def _get_agent_specific_features(self, market_data: Dict) -> List[float]:
        """
        Get market maker specific features.
        
        Args:
            market_data: Current market data
            
        Returns:
            List of additional features
        """
        features = []
        
        # Inventory features
        features.append(self.state.position / self.max_inventory)  # Normalized inventory
        features.append(len(self.state.active_orders) / 10.0)  # Active orders
        
        # Market making specific features
        features.append(self.min_spread)  # Current min spread
        features.append(self.quotes_placed / 1000.0)  # Quotes placed (normalized)
        features.append(self.quotes_filled / max(1, self.quotes_placed))  # Fill rate
        
        # Risk features
        features.append(self.inventory_risk)  # Current inventory risk
        
        return features
    
    def _get_agent_specific_metrics(self) -> Dict[str, float]:
        """
        Get market maker specific performance metrics.
        
        Returns:
            Dictionary of additional metrics
        """
        return {
            "quotes_placed": self.quotes_placed,
            "quotes_filled": self.quotes_filled,
            "fill_rate": self.quotes_filled / max(1, self.quotes_placed),
            "inventory_risk": self.inventory_risk,
            "min_spread": self.min_spread,
            "max_inventory": self.max_inventory,
            "inventory_ratio": abs(self.state.position) / self.max_inventory,
        }
    
    def _get_checkpoint_data(self) -> Dict:
        """Get market maker specific checkpoint data."""
        return {
            "inventory_target": self.inventory_target,
            "max_inventory": self.max_inventory,
            "min_spread": self.min_spread,
            "max_spread": self.max_spread,
            "quotes_placed": self.quotes_placed,
            "quotes_filled": self.quotes_filled,
            "recent_spreads": self.recent_spreads,
            "recent_inventory": self.recent_inventory,
            "profit_history": self.profit_history,
        }
    
    def _load_checkpoint_data(self, checkpoint_data: Dict):
        """Load market maker specific checkpoint data."""
        self.inventory_target = checkpoint_data.get("inventory_target", 0.0)
        self.max_inventory = checkpoint_data.get("max_inventory", 500.0)
        self.min_spread = checkpoint_data.get("min_spread", 0.01)
        self.max_spread = checkpoint_data.get("max_spread", 0.05)
        self.quotes_placed = checkpoint_data.get("quotes_placed", 0)
        self.quotes_filled = checkpoint_data.get("quotes_filled", 0)
        self.recent_spreads = checkpoint_data.get("recent_spreads", [])
        self.recent_inventory = checkpoint_data.get("recent_inventory", [])
        self.profit_history = checkpoint_data.get("profit_history", [])

