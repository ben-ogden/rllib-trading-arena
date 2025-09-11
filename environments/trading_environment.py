"""
Multi-Agent Trading Environment for RLlib

This module implements the main trading environment that integrates the order book,
market simulator, and trading agents. It provides the RLlib-compatible interface
for multi-agent reinforcement learning in a realistic trading scenario.
"""

import gymnasium as gym
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import IntEnum
import uuid
import time

from .order_book import OrderBook, Order, OrderType, OrderSide
from .market_simulator import MarketSimulator, MarketEvent


class ActionType(IntEnum):
    """Action types for trading agents."""
    HOLD = 0
    BUY = 1
    SELL = 2
    CANCEL = 3


class OrderTypeAction(IntEnum):
    """Order type for agent actions (different from OrderType to avoid conflicts)."""
    MARKET = 0
    LIMIT = 1


@dataclass
class AgentState:
    """State of a trading agent."""
    agent_id: str
    agent_type: str
    cash: float
    position: float
    pnl: float
    total_trades: int
    active_orders: List[str]
    last_action: Optional[Dict] = None
    performance_metrics: Optional[Dict] = None


class TradingEnvironment(gym.Env):
    """
    Multi-agent trading environment compatible with RLlib.
    
    This environment simulates a realistic financial market where multiple
    trading agents compete or cooperate to maximize their profits. Each agent
    can place buy/sell orders, manage their positions, and respond to market
    conditions.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the trading environment.
        
        Args:
            config: Configuration dictionary containing environment parameters
        """
        super().__init__()
        
        # Environment configuration
        self.config = config
        self.market_config = config.get("market", {})
        self.agents_config = config.get("agents", {})
        
        # Initialize components
        self.order_book = OrderBook(
            tick_size=self.market_config.get("tick_size", 0.01),
            max_depth=self.market_config.get("order_book_depth", 10)
        )
        
        self.market_simulator = MarketSimulator(
            initial_price=self.market_config.get("initial_price", 100.0),
            volatility=self.market_config.get("volatility", 0.02),
            liquidity_factor=self.market_config.get("liquidity_factor", 0.1),
            mean_reversion=self.market_config.get("mean_reversion", 0.1),
            event_probability=self.market_config.get("event_probability", 0.05)
        )
        
        # Environment state
        self.current_step = 0
        self.max_steps = config.get("max_steps_per_episode", 1000)
        self.episode_rewards: Dict[str, List[float]] = {}
        self.episode_metrics: Dict[str, List[Dict]] = {}
        
        # Agent management
        self.agents: Dict[str, AgentState] = {}
        self.agent_types: List[str] = []
        self._initialize_agents()
        
        # Action and observation spaces
        self._setup_spaces()
        
        # Performance tracking
        self.total_trades = 0
        self.total_volume = 0.0
        self.market_efficiency_metrics = {}
        
    def _initialize_agents(self):
        """Initialize trading agents based on configuration."""
        for agent_type, agent_config in self.agents_config.items():
            count = agent_config.get("count", 1)
            initial_capital = agent_config.get("initial_capital", 100000)
            
            for i in range(count):
                agent_id = f"{agent_type}_{i}"
                self.agents[agent_id] = AgentState(
                    agent_id=agent_id,
                    agent_type=agent_type,
                    cash=initial_capital,
                    position=0.0,
                    pnl=0.0,
                    total_trades=0,
                    active_orders=[],
                    performance_metrics={}
                )
                
                if agent_type not in self.agent_types:
                    self.agent_types.append(agent_type)
                
                # Initialize episode tracking
                self.episode_rewards[agent_id] = []
                self.episode_metrics[agent_id] = []
    
    def _setup_spaces(self):
        """Setup action and observation spaces for RLlib compatibility."""
        # Action space: [action_type, quantity, price, order_type]
        # action_type: ActionType enum (HOLD=0, BUY=1, SELL=2, CANCEL=3)
        # quantity: normalized quantity (0-1)
        # price: normalized price offset from mid price (-1 to 1)
        # order_type: OrderTypeAction enum (MARKET=0, LIMIT=1)
        self.action_space = gym.spaces.Box(
            low=np.array([0, 0, -1, 0]),
            high=np.array([3, 1, 1, 1]),
            dtype=np.float32
        )
        
        # Observation space: market data + agent state
        # Market: [price, volatility, liquidity, volume, spread, depth, event]
        # Agent: [cash, position, pnl, active_orders_count]
        market_features = 7
        agent_features = 4
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(market_features + agent_features,),
            dtype=np.float32
        )
    
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to initial state.
        
        Returns:
            Tuple of (observation, info) for single-agent mode
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Reset components
        self.order_book = OrderBook(
            tick_size=self.market_config.get("tick_size", 0.01),
            max_depth=self.market_config.get("order_book_depth", 10)
        )
        self.market_simulator.reset()
        
        # Add initial liquidity to the order book
        self._add_initial_liquidity()
        
        # Reset agent states
        for agent_id, agent in self.agents.items():
            agent_config = self.agents_config[agent.agent_type]
            agent.cash = agent_config.get("initial_capital", 100000)
            agent.position = 0.0
            agent.pnl = 0.0
            agent.total_trades = 0
            agent.active_orders = []
            agent.last_action = None
            agent.performance_metrics = {}
            
            # Reset episode tracking
            self.episode_rewards[agent_id] = []
            self.episode_metrics[agent_id] = []
        
        # Reset environment state
        self.current_step = 0
        self.total_trades = 0
        self.total_volume = 0.0
        self.market_efficiency_metrics = {}
        
        # Get initial observations
        observations = self._get_observations()
        info = self._get_info()
        
        # For single-agent mode, return the observation for the first agent
        if len(observations) == 1:
            agent_id = list(observations.keys())[0]
            return observations[agent_id], info
        else:
            return observations, info
    
    def _add_initial_liquidity(self):
        """Add initial liquidity to the order book to enable trading."""
        current_price = self.market_simulator.current_price
        tick_size = self.order_book.tick_size
        
        # Add some buy orders below current price
        for i in range(5):
            price = current_price - (i + 1) * tick_size * 10  # 10 ticks below
            quantity = 100 + i * 50  # Varying quantities
            
            order = Order(
                order_id=f"liquidity_buy_{i}",
                agent_id="market_maker_liquidity",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=quantity,
                price=price,
                timestamp=time.time()
            )
            self.order_book.add_order(order)
        
        # Add some sell orders above current price
        for i in range(5):
            price = current_price + (i + 1) * tick_size * 10  # 10 ticks above
            quantity = 100 + i * 50  # Varying quantities
            
            order = Order(
                order_id=f"liquidity_sell_{i}",
                agent_id="market_maker_liquidity",
                side=OrderSide.SELL,
                order_type=OrderType.LIMIT,
                quantity=quantity,
                price=price,
                timestamp=time.time()
            )
            self.order_book.add_order(order)
    
    def step(self, actions) -> Tuple:
        """
        Execute one step of the environment.
        
        Args:
            actions: Dictionary mapping agent IDs to their actions (multi-agent) or single action array (single-agent)
            
        Returns:
            Tuple of (observations, rewards, terminated, truncated, info)
        """
        self.current_step += 1
        
        # Handle both single-agent and multi-agent modes
        if isinstance(actions, dict):
            # Multi-agent mode
            for agent_id, action in actions.items():
                if agent_id in self.agents:
                    self._process_agent_action(agent_id, action)
        else:
            # Single-agent mode - actions is a numpy array
            agent_id = list(self.agents.keys())[0]
            self._process_agent_action(agent_id, actions)
        
        # Advance market simulation
        market_state = self.market_simulator.step()
        
        # Update market price in order book (for reference)
        # In a real implementation, this would be driven by actual trades
        
        # Calculate rewards and update agent states
        rewards = self._calculate_rewards()
        
        # Check termination conditions
        terminated = self._check_termination()
        truncated = self.current_step >= self.max_steps
        
        # Get observations and info
        observations = self._get_observations()
        info = self._get_info()
        
        # For single-agent mode, return single values instead of dictionaries
        if len(self.agents) == 1:
            agent_id = list(self.agents.keys())[0]
            return (
                observations[agent_id],  # Single observation array
                rewards[agent_id],       # Single reward value
                terminated[agent_id],    # Single terminated bool
                truncated,               # Single truncated bool
                info                     # Info dict
            )
        else:
            return observations, rewards, terminated, truncated, info
    
    def _process_agent_action(self, agent_id: str, action: np.ndarray):
        """Process an agent's action and update the order book."""
        agent = self.agents[agent_id]
        
        # Parse action
        action_type = ActionType(int(action[0]))
        quantity = float(action[1])
        price_offset = float(action[2])
        order_type = OrderTypeAction(int(action[3]))
        
        # Get current market conditions
        market_data = self.order_book.get_market_data()
        mid_price = market_data.get("mid_price") or self.market_simulator.current_price
        
        # Convert normalized values to actual values
        max_quantity = min(agent.cash / mid_price, 1000) if action_type == ActionType.BUY else abs(agent.position)
        actual_quantity = quantity * max_quantity
        
        if order_type == OrderTypeAction.MARKET:  # Market order
            actual_price = None
        else:  # Limit order
            price_range = mid_price * 0.1  # 10% price range
            actual_price = mid_price + (price_offset * price_range)
            actual_price = round(actual_price / self.order_book.tick_size) * self.order_book.tick_size
        
        # Execute action
        if action_type == ActionType.BUY and actual_quantity > 0:  # Buy
            self._place_buy_order(agent_id, actual_quantity, actual_price)
        elif action_type == ActionType.SELL and actual_quantity > 0:  # Sell
            self._place_sell_order(agent_id, actual_quantity, actual_price)
        elif action_type == ActionType.CANCEL:  # Cancel orders
            self._cancel_agent_orders(agent_id)
        
        # Store action for analysis
        agent.last_action = {
            "action_type": action_type,
            "quantity": actual_quantity,
            "price": actual_price,
            "order_type": order_type
        }
        
        # Debug: Print action details occasionally
        if self.current_step % 100 == 0:
            # Format price properly for display
            price_str = f"{actual_price:.2f}" if actual_price is not None else "None"
            print(f"Step {self.current_step}: Agent {agent_id} action: type={action_type.name}, order={order_type.name}, qty={actual_quantity:.2f}, price={price_str}, cumulative_trades={agent.total_trades}")
    
    def _place_buy_order(self, agent_id: str, quantity: float, price: Optional[float]):
        """Place a buy order for an agent."""
        agent = self.agents[agent_id]
        
        # Check if agent has enough cash
        if price is None:  # Market order
            max_quantity = agent.cash / self.market_simulator.current_price
            quantity = min(quantity, max_quantity)
        
        if quantity <= 0:
            return
        
        # Create order
        order_id = f"{agent_id}_{uuid.uuid4().hex[:8]}"
        order = Order(
            order_id=order_id,
            agent_id=agent_id,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET if price is None else OrderType.LIMIT,
            quantity=quantity,
            price=price,
            timestamp=time.time()
        )
        
        # Add to order book
        trades = self.order_book.add_order(order)
        
        # Process trades
        for trade in trades:
            self._process_trade(trade)
        
        # Update agent state
        if not order.is_filled:
            agent.active_orders.append(order_id)
    
    def _place_sell_order(self, agent_id: str, quantity: float, price: Optional[float]):
        """Place a sell order for an agent."""
        agent = self.agents[agent_id]
        
        # Check if agent has enough position
        max_quantity = abs(agent.position)
        quantity = min(quantity, max_quantity)
        
        if quantity <= 0:
            return
        
        # Create order
        order_id = f"{agent_id}_{uuid.uuid4().hex[:8]}"
        order = Order(
            order_id=order_id,
            agent_id=agent_id,
            side=OrderSide.SELL,
            order_type=OrderType.MARKET if price is None else OrderType.LIMIT,
            quantity=quantity,
            price=price,
            timestamp=time.time()
        )
        
        # Add to order book
        trades = self.order_book.add_order(order)
        
        # Process trades
        for trade in trades:
            self._process_trade(trade)
        
        # Update agent state
        if not order.is_filled:
            agent.active_orders.append(order_id)
    
    def _cancel_agent_orders(self, agent_id: str):
        """Cancel all active orders for an agent."""
        agent = self.agents[agent_id]
        
        for order_id in agent.active_orders[:]:
            if self.order_book.cancel_order(order_id):
                agent.active_orders.remove(order_id)
    
    def _process_trade(self, trade):
        """Process a completed trade and update agent states."""
        self.total_trades += 1
        self.total_volume += trade.quantity
        
        # Update buyer (only if it's a real agent, not liquidity)
        if trade.buy_agent_id != "market_maker_liquidity":
            buyer = self.agents[trade.buy_agent_id]
            buyer.cash -= trade.quantity * trade.price
            buyer.position += trade.quantity
            buyer.total_trades += 1
            
            # Remove filled orders from active orders
            if trade.buy_order_id in buyer.active_orders:
                buyer.active_orders.remove(trade.buy_order_id)
        
        # Update seller (only if it's a real agent, not liquidity)
        if trade.sell_agent_id != "market_maker_liquidity":
            seller = self.agents[trade.sell_agent_id]
            seller.cash += trade.quantity * trade.price
            seller.position -= trade.quantity
            seller.total_trades += 1
            
            # Remove filled orders from active orders
            if trade.sell_order_id in seller.active_orders:
                seller.active_orders.remove(trade.sell_order_id)
    
    def _calculate_rewards(self) -> Dict[str, float]:
        """Calculate rewards for all agents."""
        rewards = {}
        
        for agent_id, agent in self.agents.items():
            # Calculate current PnL
            current_price = self.market_simulator.current_price
            unrealized_pnl = agent.position * current_price
            total_pnl = agent.cash + unrealized_pnl - 100000  # Subtract initial capital
            
            # PROFIT-FOCUSED REWARD FUNCTION
            # Primary reward: PnL (make this the dominant component)
            pnl_reward = total_pnl / 1000.0  # Increased weight for PnL (was 10000.0)
            
            # Risk penalty (moderate penalty for large positions)
            position_penalty = -(abs(agent.position) / 1000.0) ** 2 * 0.1  # Reduced penalty for large positions
            
            # Trading cost penalty (realistic but not prohibitive)
            trading_penalty = -agent.total_trades * 0.005  # Reduced trading cost
            
            # Market making bonus (only for profitable market making)
            market_making_bonus = 0.0
            if agent.agent_type == "market_maker" and agent.total_trades > 0 and total_pnl > 0:
                spread = self.order_book.get_spread()
                if spread is not None and spread > 0:
                    market_making_bonus = spread * 0.05  # Reduced bonus
            
            # BALANCED EXPLORATION REWARDS (encourage learning)
            # Reward for placing orders (encourage exploration)
            order_placement_reward = 0.01 if agent.last_action and agent.last_action.get("action_type") in [ActionType.BUY, ActionType.SELL] else 0.0
            
            # Reward for being active (prevent complete inactivity)
            activity_reward = 0.005 if agent.last_action and agent.last_action.get("action_type") != ActionType.HOLD else 0.0
            
            # Small penalty for doing nothing (encourage some activity)
            inactivity_penalty = -0.005 if agent.last_action and agent.last_action.get("action_type") == ActionType.HOLD else 0.0
            
            # Reward for profitable trades (not just any trades)
            profitable_trade_reward = 0.0
            if agent.total_trades > 0 and total_pnl > 0:
                profitable_trade_reward = min(total_pnl / 1000.0, 0.1)  # Reward proportional to profit
            
            # Reward for maintaining reasonable position (risk management)
            position_reward = 0.02 if abs(agent.position) < 100 else 0.0  # Increased reward for smaller positions
            
            # Reward for having active orders (market making behavior)
            active_orders_reward = len(agent.active_orders) * 0.001 if len(agent.active_orders) > 0 else 0.0
            
            # Total reward - PnL focused with minimal exploration incentives
            total_reward = (pnl_reward + position_penalty + trading_penalty + market_making_bonus + 
                          activity_reward + inactivity_penalty + profitable_trade_reward + 
                          position_reward + order_placement_reward + active_orders_reward)
            
            # Ensure minimum reward for exploration (prevent total stagnation)
            if total_reward < -0.1:  # Cap large negative rewards
                total_reward = -0.1
            
            rewards[agent_id] = total_reward
            
            # Update episode rewards
            self.episode_rewards[agent_id].append(total_reward)
            
            # Update agent PnL
            agent.pnl = total_pnl
        
        return rewards
    
    def _check_termination(self) -> Dict[str, bool]:
        """Check if any agents should be terminated."""
        terminated = {}
        
        for agent_id, agent in self.agents.items():
            # Terminate if agent runs out of cash and has no position
            if agent.cash <= 0 and agent.position <= 0:
                terminated[agent_id] = True
            else:
                terminated[agent_id] = False
        
        return terminated
    
    def _get_observations(self) -> Dict[str, np.ndarray]:
        """Get observations for all agents."""
        observations = {}
        
        # Get market data
        market_data = self.order_book.get_market_data()
        market_conditions = self.market_simulator.get_market_conditions()
        
        for agent_id, agent in self.agents.items():
            # Market features
            market_features = np.array([
                market_conditions["price"] / 100.0,  # Normalized price
                market_conditions["volatility"] * 100,  # Volatility
                market_conditions["liquidity"],
                market_conditions["volume"] / 10000.0,  # Normalized volume
                (market_data.get("spread") or 0.0) / 100.0,  # Normalized spread
                len(market_data.get("buy_depth", [])),  # Buy depth
                float(market_conditions["event"] != "normal")  # Event indicator
            ], dtype=np.float32)
            
            # Agent features
            agent_features = np.array([
                agent.cash / 100000.0,  # Normalized cash
                agent.position / 1000.0,  # Normalized position
                agent.pnl / 1000.0,  # Normalized PnL
                len(agent.active_orders) / 10.0  # Normalized active orders
            ], dtype=np.float32)
            
            # Combine features
            observation = np.concatenate([market_features, agent_features])
            observations[agent_id] = observation
        
        return observations
    
    def _get_info(self) -> Dict[str, Any]:
        """Get additional information about the environment state."""
        info = {
            "step": self.current_step,
            "market_conditions": self.market_simulator.get_market_conditions(),
            "market_data": self.order_book.get_market_data(),
            "total_trades": self.total_trades,
            "total_volume": self.total_volume,
            "agents": {}
        }
        
        for agent_id, agent in self.agents.items():
            info["agents"][agent_id] = {
                "cash": agent.cash,
                "position": agent.position,
                "pnl": agent.pnl,
                "total_trades": agent.total_trades,
                "active_orders": len(agent.active_orders),
                "agent_type": agent.agent_type
            }
        
        return info
    
    def render(self, mode: str = "human"):
        """Render the environment (placeholder for visualization)."""
        if mode == "human":
            print(f"Step: {self.current_step}")
            print(f"Price: {self.market_simulator.current_price:.2f}")
            print(f"Volatility: {self.market_simulator.current_volatility:.4f}")
            print(f"Total Trades: {self.total_trades}")
            print("Agent States:")
            for agent_id, agent in self.agents.items():
                print(f"  {agent_id}: Cash={agent.cash:.2f}, Position={agent.position:.2f}, PnL={agent.pnl:.2f}")
    
    def close(self):
        """Clean up environment resources."""
        pass

