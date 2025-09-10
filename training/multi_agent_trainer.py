"""
Multi-Agent Training Script for RLlib Demo

This script showcases RLlib's latest multi-agent capabilities using Ray 2.49.1,
including the new API stack, distributed training, and advanced multi-agent scenarios.
"""

import ray
import numpy as np
from typing import Dict, List, Any, Optional
import yaml
import os
from pathlib import Path
import time
import logging
import gymnasium as gym
import uuid

from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
# A3C removed in Ray 2.49.1 - using PPO instead
from ray.rllib.algorithms.impala import ImpalaConfig
from ray.rllib.algorithms.sac import SACConfig
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.utils.typing import PolicyID

from environments.trading_environment import TradingEnvironment
from environments.order_book import OrderBook, Order, OrderType, OrderSide
from environments.market_simulator import MarketSimulator
from agents.market_maker import MarketMakerAgent
from agents.momentum_trader import MomentumTraderAgent
from agents.arbitrageur import ArbitrageurAgent


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultiAgentTradingEnv(MultiAgentEnv):
    """
    Proper multi-agent trading environment for RLlib.
    
    This class creates a true multi-agent environment that works with RLlib's
    multi-agent training framework, showcasing the latest RLlib features.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the multi-agent trading environment.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__()
        
        self.config = config
        self.market_config = config.get("market", {})
        self.agents_config = config.get("agents", {})
        
        # Initialize components directly (not wrapping single-agent env)
        
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
        
        # Agent management
        self.agent_states = {}
        self.agent_ids = []
        self.agent_types = {}
        self._initialize_agents()
        
        # Environment spaces - multi-agent environments need dict spaces
        self.observation_space = {
            agent_id: gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(11,), dtype=np.float32
            ) for agent_id in self.agent_ids
        }
        self.action_space = {
            agent_id: gym.spaces.Box(
                low=np.array([0, 0, -1, 0]),
                high=np.array([3, 1, 1, 1]),
                dtype=np.float32
            ) for agent_id in self.agent_ids
        }
        
        # Episode tracking
        self.episode_count = 0
        self.episode_rewards = {agent_id: [] for agent_id in self.agent_ids}
        
        # Required attributes for Ray 2.49.1 multi-agent environments
        self.agents = self.agent_ids  # Currently active agents
        self.possible_agents = self.agent_ids  # All possible agents
    
    def _initialize_agents(self):
        """Initialize trading agents based on configuration."""
        for agent_type, agent_config in self.agents_config.items():
            count = agent_config.get("count", 1)
            initial_capital = agent_config.get("initial_capital", 100000)
            
            for i in range(count):
                agent_id = f"{agent_type}_{i}"
                self.agent_states[agent_id] = {
                    "agent_id": agent_id,
                    "agent_type": agent_type,
                    "cash": initial_capital,
                    "position": 0.0,
                    "pnl": 0.0,
                    "total_trades": 0,
                    "active_orders": [],
                    "last_action": None,
                    "performance_metrics": {}
                }
                
                self.agent_ids.append(agent_id)
                self.agent_types[agent_id] = agent_type
    
    def get_action_space(self, agent_id: str):
        """Get action space for a specific agent."""
        return self.action_space[agent_id]
    
    def get_observation_space(self, agent_id: str):
        """Get observation space for a specific agent."""
        return self.observation_space[agent_id]
        
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        """
        Reset the environment for a new episode.
        
        Returns:
            Dictionary mapping agent IDs to their initial observations
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
        for agent_id, agent_state in self.agent_states.items():
            agent_config = self.agents_config[agent_state["agent_type"]]
            agent_state["cash"] = agent_config.get("initial_capital", 100000)
            agent_state["position"] = 0.0
            agent_state["pnl"] = 0.0
            agent_state["total_trades"] = 0
            agent_state["active_orders"] = []
            agent_state["last_action"] = None
            agent_state["performance_metrics"] = {}
            
            # Reset episode tracking
            self.episode_rewards[agent_id] = []
        
        # Reset environment state
        self.current_step = 0
        self.episode_count += 1
        
        # Get initial observations
        observations = self._get_observations()
        info = {}
        
        return observations, info
    
    def step(self, action_dict: Dict[PolicyID, Any]):
        """
        Execute one step of the multi-agent environment.
        
        Args:
            action_dict: Dictionary mapping agent IDs to their actions
            
        Returns:
            Tuple of (observations, rewards, terminated, truncated, info)
        """
        self.current_step += 1
        
        # Process actions for all agents
        for agent_id, action in action_dict.items():
            if agent_id in self.agent_states:
                self._process_agent_action(agent_id, action)
        
        # Advance market simulation
        market_state = self.market_simulator.step()
        
        # Calculate rewards and update agent states
        rewards = self._calculate_rewards()
        
        # Check termination conditions
        terminated = self._check_termination()
        truncated = {agent_id: self.current_step >= self.max_steps for agent_id in self.agent_ids}
        
        # Get observations and info
        observations = self._get_observations()
        info = self._get_info()
        
        # Add '__all__' key required for multi-agent environments
        terminated['__all__'] = all(terminated.values())
        truncated['__all__'] = all(truncated.values())
        
        # Convert info to per-agent format for multi-agent environments
        agent_info = {agent_id: info for agent_id in self.agent_ids}
        
        return observations, rewards, terminated, truncated, agent_info
    
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
    
    def _process_agent_action(self, agent_id: str, action: np.ndarray):
        """Process an agent's action and update the order book."""
        agent_state = self.agent_states[agent_id]
        
        # Parse action
        action_type = int(action[0])
        quantity = float(action[1])
        price_offset = float(action[2])
        order_type = int(action[3])
        
        # Get current market conditions
        market_data = self.order_book.get_market_data()
        mid_price = market_data.get("mid_price") or self.market_simulator.current_price
        
        # Convert normalized values to actual values
        max_quantity = min(agent_state["cash"] / mid_price, 1000) if action_type == 1 else abs(agent_state["position"])
        actual_quantity = quantity * max_quantity
        
        if order_type == 0:  # Market order
            actual_price = None
        else:  # Limit order
            price_range = mid_price * 0.1  # 10% price range
            actual_price = mid_price + (price_offset * price_range)
            actual_price = round(actual_price / self.order_book.tick_size) * self.order_book.tick_size
        
        # Execute action
        if action_type == 1 and actual_quantity > 0:  # Buy
            self._place_buy_order(agent_id, actual_quantity, actual_price)
        elif action_type == 2 and actual_quantity > 0:  # Sell
            self._place_sell_order(agent_id, actual_quantity, actual_price)
        elif action_type == 3:  # Cancel orders
            self._cancel_agent_orders(agent_id)
        
        # Store action for analysis
        agent_state["last_action"] = {
            "action_type": action_type,
            "quantity": actual_quantity,
            "price": actual_price,
            "order_type": order_type
        }
    
    def _place_buy_order(self, agent_id: str, quantity: float, price: Optional[float]):
        """Place a buy order for an agent."""
        agent_state = self.agent_states[agent_id]
        
        # Check if agent has enough cash
        if price is None:  # Market order
            max_quantity = agent_state["cash"] / self.market_simulator.current_price
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
            agent_state["active_orders"].append(order_id)
    
    def _place_sell_order(self, agent_id: str, quantity: float, price: Optional[float]):
        """Place a sell order for an agent."""
        agent_state = self.agent_states[agent_id]
        
        # Check if agent has enough position
        max_quantity = abs(agent_state["position"])
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
            agent_state["active_orders"].append(order_id)
    
    def _cancel_agent_orders(self, agent_id: str):
        """Cancel all active orders for an agent."""
        agent_state = self.agent_states[agent_id]
        
        for order_id in agent_state["active_orders"][:]:
            if self.order_book.cancel_order(order_id):
                agent_state["active_orders"].remove(order_id)
    
    def _process_trade(self, trade):
        """Process a completed trade and update agent states."""
        # Update buyer (only if it's a real agent, not liquidity)
        if trade.buy_agent_id != "market_maker_liquidity" and trade.buy_agent_id in self.agent_states:
            buyer = self.agent_states[trade.buy_agent_id]
            buyer["cash"] -= trade.quantity * trade.price
            buyer["position"] += trade.quantity
            buyer["total_trades"] += 1
            
            # Remove filled orders from active orders
            if trade.buy_order_id in buyer["active_orders"]:
                buyer["active_orders"].remove(trade.buy_order_id)
        
        # Update seller (only if it's a real agent, not liquidity)
        if trade.sell_agent_id != "market_maker_liquidity" and trade.sell_agent_id in self.agent_states:
            seller = self.agent_states[trade.sell_agent_id]
            seller["cash"] += trade.quantity * trade.price
            seller["position"] -= trade.quantity
            seller["total_trades"] += 1
            
            # Remove filled orders from active orders
            if trade.sell_order_id in seller["active_orders"]:
                seller["active_orders"].remove(trade.sell_order_id)
    
    def _calculate_rewards(self) -> Dict[str, float]:
        """Calculate rewards for all agents."""
        rewards = {}
        
        for agent_id, agent_state in self.agent_states.items():
            # Calculate current PnL
            current_price = self.market_simulator.current_price
            unrealized_pnl = agent_state["position"] * current_price
            total_pnl = agent_state["cash"] + unrealized_pnl - 100000  # Subtract initial capital
            
            # Reward components
            pnl_reward = total_pnl / 1000.0  # Normalize
            
            # Risk penalty (penalize large positions)
            position_penalty = -(abs(agent_state["position"]) / 1000.0) ** 2
            
            # Trading cost penalty
            trading_penalty = -agent_state["total_trades"] * 0.01
            
            # Add small positive reward for being active (encourage exploration)
            activity_reward = 0.05 if agent_state["total_trades"] > 0 else 0.0
            
            # Penalty for doing nothing (action type 0) - encourage trading
            inactivity_penalty = -0.001 if agent_state["last_action"] and agent_state["last_action"].get("action_type") == 0 else 0.0
            
            # Total reward
            total_reward = pnl_reward + position_penalty + trading_penalty + activity_reward + inactivity_penalty
            
            rewards[agent_id] = total_reward
            
            # Update episode rewards
            self.episode_rewards[agent_id].append(total_reward)
            
            # Update agent PnL
            agent_state["pnl"] = total_pnl
        
        return rewards
    
    def _check_termination(self) -> Dict[str, bool]:
        """Check if any agents should be terminated."""
        terminated = {}
        
        for agent_id, agent_state in self.agent_states.items():
            # Terminate if agent runs out of cash and has no position
            if agent_state["cash"] <= 0 and agent_state["position"] <= 0:
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
        
        for agent_id, agent_state in self.agent_states.items():
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
                agent_state["cash"] / 100000.0,  # Normalized cash
                agent_state["position"] / 1000.0,  # Normalized position
                agent_state["pnl"] / 1000.0,  # Normalized PnL
                len(agent_state["active_orders"]) / 10.0  # Normalized active orders
            ], dtype=np.float32)
            
            # Combine features
            observation = np.concatenate([market_features, agent_features])
            observations[agent_id] = observation
        
        return observations
    
    def _get_info(self) -> Dict[str, Any]:
        """Get additional information about the environment state."""
        # Return empty info dict to avoid validation issues
        # RLlib expects info to be per-agent or empty
        return {}
    
    def get_agent_ids(self) -> List[PolicyID]:
        """Get list of agent IDs."""
        return self.agent_ids
    
    def render(self, mode: str = "human"):
        """Render the environment."""
        self.env.render(mode)
    
    def close(self):
        """Close the environment."""
        self.env.close()


class MultiAgentTrainer:
    """
    Multi-agent trainer that showcases RLlib's latest capabilities.
    
    This trainer demonstrates:
    - Multi-agent training with different algorithms
    - Distributed training across multiple workers
    - Policy sharing and independent policies
    - Advanced monitoring and logging
    - Anyscale cloud integration
    """
    
    def __init__(self, config_path: str = "configs/trading_config.yaml"):
        """
        Initialize the multi-agent trainer.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        
        # Initialize Ray
        self._initialize_ray()
        
        # Training state
        self.trainers = {}
        self.results = {}
        self.best_models = {}
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def _initialize_ray(self):
        """Initialize Ray with optimal settings for the demo."""
        # Initialize Ray with distributed training settings
        ray.init(
            num_cpus=self.config.get("distributed", {}).get("num_workers", 4),
            num_gpus=self.config.get("distributed", {}).get("num_gpus", 0),
            ignore_reinit_error=True,
            logging_level=logging.INFO,
            # Anyscale cloud settings
            runtime_env={
                "working_dir": ".",
                "pip": ["ray[rllib]==2.49.1"]
            }
        )
        
        logger.info("Ray initialized successfully")
        logger.info(f"Ray cluster resources: {ray.cluster_resources()}")
    
    def create_algorithm_config(self, algorithm: str) -> Any:
        """
        Create algorithm configuration using RLlib's latest API.
        
        Args:
            algorithm: Algorithm name (ppo, a3c, impala, sac)
            
        Returns:
            Algorithm configuration object
        """
        # Base configuration
        base_config = {
            "env": MultiAgentTradingEnv,
            "env_config": self.config,
            "framework": "torch",
            "num_env_runners": self.config.get("distributed", {}).get("num_workers", 4),
            "num_cpus_per_env_runner": self.config.get("distributed", {}).get("num_cpus_per_worker", 1),
            "num_gpus": self.config.get("distributed", {}).get("num_gpus", 0),
            "use_gpu": self.config.get("distributed", {}).get("use_gpu", False),
            "log_level": self.config.get("monitoring", {}).get("log_level", "INFO"),
            "metrics_num_episodes_for_smoothing": 10,
            "min_sample_timesteps_per_iteration": 1000,
            "train_batch_size": self.config.get("training", {}).get("batch_size", 32),
            "lr": self.config.get("training", {}).get("learning_rate", 0.0003),
            "gamma": self.config.get("training", {}).get("gamma", 0.99),
            "lambda": 0.95,
            "use_gae": True,
            "clip_param": 0.2,
            "grad_clip": 0.5,
            "entropy_coeff": 0.01,
            "vf_loss_coeff": 0.5,
            "kl_coeff": 0.2,
            "kl_target": 0.01,
            "rollout_fragment_length": 200,
            "num_epochs": 10,
            "batch_mode": "truncate_episodes",
            "observation_filter": "NoFilter",
            "replay_buffer_config": {
                "type": "MultiAgentReplayBuffer",
                "capacity": self.config.get("training", {}).get("buffer_size", 10000),
            },
            "exploration_config": {
                "type": "EpsilonGreedy",
                "initial_epsilon": 1.0,
                "final_epsilon": 0.02,
                "epsilon_timesteps": 10000,
            },
            "callbacks": self._get_callbacks(),
        }
        
        # Algorithm-specific configurations
        if algorithm.lower() == "ppo":
            config = PPOConfig().update_from_dict(base_config)
            # PPO-specific settings
            ppo_config = self.config.get("algorithms", {}).get("ppo", {})
            config.update_from_dict(ppo_config)
            
        elif algorithm.lower() == "a3c":
            # A3C removed in Ray 2.49.1 - using PPO instead
            config = PPOConfig().update_from_dict(base_config)
            # Use PPO settings for A3C replacement
            ppo_config = self.config.get("algorithms", {}).get("ppo", {})
            config.update_from_dict(ppo_config)
            
        elif algorithm.lower() == "impala":
            config = ImpalaConfig().update_from_dict(base_config)
            # IMPALA-specific settings
            impala_config = self.config.get("algorithms", {}).get("impala", {})
            config.update_from_dict(impala_config)
            
        elif algorithm.lower() == "sac":
            config = SACConfig().update_from_dict(base_config)
            # SAC-specific settings
            config.update_from_dict({
                "tau": self.config.get("training", {}).get("tau", 0.005),
                "target_network_update_freq": 1,
                "use_state_preprocessor": True,
                "replay_buffer_config": {
                    "type": "MultiAgentReplayBuffer",
                    "capacity": self.config.get("training", {}).get("buffer_size", 10000),
                },
            })
            
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
        
        # Multi-agent specific configuration
        config.multi_agent(
            policies=self._get_policy_specs(),
            policy_mapping_fn=self._policy_mapping_fn,
            policies_to_train=list(self.config.get("agents", {}).keys()),
        )
        
        return config
    
    def _get_policy_specs(self) -> Dict[str, PolicySpec]:
        """
        Define policy specifications for different agent types.
        
        Returns:
            Dictionary mapping policy names to PolicySpec objects
        """
        policies = {}
        
        for agent_type in self.config.get("agents", {}).keys():
            policies[agent_type] = PolicySpec(
                policy_class=None,  # Use default policy
                observation_space=None,  # Will be inferred
                action_space=None,  # Will be inferred
                config={
                    "agent_type": agent_type,
                    "model": {
                        "fcnet_hiddens": [256, 256],
                        "fcnet_activation": "relu",
                        "vf_share_layers": False,
                        "use_lstm": False,
                        "max_seq_len": 20,
                        "lstm_cell_size": 256,
                        "lstm_use_prev_action": True,
                        "lstm_use_prev_reward": True,
                    },
                    "exploration_config": {
                        "type": "EpsilonGreedy",
                        "initial_epsilon": 1.0,
                        "final_epsilon": 0.02,
                        "epsilon_timesteps": 10000,
                    },
                }
            )
        
        return policies
    
    def _policy_mapping_fn(self, agent_id: str, episode, worker, **kwargs) -> str:
        """
        Map agent IDs to policy names.
        
        Args:
            agent_id: Agent identifier
            episode: Current episode
            worker: Worker instance
            **kwargs: Additional arguments
            
        Returns:
            Policy name for the agent
        """
        # Extract agent type from agent ID
        agent_type = agent_id.split("_")[0]
        return agent_type
    
    def _get_callbacks(self):
        """Get custom callbacks for monitoring and logging."""
        from ray.rllib.algorithms.callbacks import DefaultCallbacks
        
        class TradingCallbacks(DefaultCallbacks):
            """Custom callbacks for trading environment monitoring."""
            
            def on_episode_start(self, *, worker, base_env, policies, episode, env_index, **kwargs):
                """Called when an episode starts."""
                episode.user_data["episode_start_time"] = time.time()
                episode.user_data["agent_rewards"] = {}
                episode.user_data["agent_actions"] = {}
            
            def on_episode_step(self, *, worker, base_env, episode, env_index, **kwargs):
                """Called on each episode step."""
                # Track agent performance
                for agent_id in episode.get_agents():
                    if agent_id not in episode.user_data["agent_rewards"]:
                        episode.user_data["agent_rewards"][agent_id] = []
                        episode.user_data["agent_actions"][agent_id] = []
                    
                    # Get latest reward and action
                    if hasattr(episode, 'last_reward_for') and episode.last_reward_for(agent_id) is not None:
                        episode.user_data["agent_rewards"][agent_id].append(episode.last_reward_for(agent_id))
            
            def on_episode_end(self, *, worker, base_env, policies, episode, env_index, **kwargs):
                """Called when an episode ends."""
                episode_duration = time.time() - episode.user_data["episode_start_time"]
                
                # Log episode statistics
                for agent_id, rewards in episode.user_data["agent_rewards"].items():
                    if rewards:
                        total_reward = sum(rewards)
                        avg_reward = np.mean(rewards)
                        
                        episode.custom_metrics[f"{agent_id}_total_reward"] = total_reward
                        episode.custom_metrics[f"{agent_id}_avg_reward"] = avg_reward
                        episode.custom_metrics[f"{agent_id}_episode_length"] = len(rewards)
                
                episode.custom_metrics["episode_duration"] = episode_duration
                
                # Log market efficiency metrics
                if hasattr(base_env, 'get_unwrapped'):
                    env = base_env.get_unwrapped()[0]
                    if hasattr(env, 'env'):
                        market_data = env.env.get_market_data()
                        episode.custom_metrics["total_trades"] = market_data.get("total_trades", 0)
                        episode.custom_metrics["total_volume"] = market_data.get("total_volume", 0.0)
                        episode.custom_metrics["market_spread"] = market_data.get("spread", 0.0)
            
            def on_train_result(self, *, algorithm, result, **kwargs):
                """Called when training results are available."""
                # Log training progress
                logger.info(f"Training iteration {result['training_iteration']}")
                logger.info(f"Episode reward mean: {result['episode_reward_mean']:.2f}")
                logger.info(f"Episode length mean: {result['episode_len_mean']:.2f}")
                
                # Log agent-specific metrics
                for key, value in result.items():
                    if "agent" in key.lower() and isinstance(value, (int, float)):
                        logger.info(f"{key}: {value:.4f}")
        
        return TradingCallbacks
    
    def train_single_algorithm(self, algorithm: str, iterations: int = 100) -> Dict[str, Any]:
        """
        Train a single algorithm on the multi-agent trading environment.
        
        Args:
            algorithm: Algorithm name to train
            iterations: Number of training iterations
            
        Returns:
            Training results
        """
        logger.info(f"Starting training with {algorithm.upper()}")
        
        # Create algorithm configuration
        config = self.create_algorithm_config(algorithm)
        
        # Build the algorithm
        trainer = config.build_algo()
        
        # Training loop
        results = []
        start_time = time.time()
        
        for i in range(iterations):
            result = trainer.train()
            results.append(result)
            
            # Log progress
            if i % 10 == 0:
                logger.info(f"Iteration {i}: Episode reward mean = {result['episode_reward_mean']:.2f}")
                
                # Log agent-specific metrics
                for agent_type in self.config.get("agents", {}).keys():
                    agent_reward_key = f"policy_{agent_type}_reward_mean"
                    if agent_reward_key in result:
                        logger.info(f"  {agent_type} reward: {result[agent_reward_key]:.2f}")
            
            # Save checkpoint
            if i % 50 == 0 and i > 0:
                checkpoint_path = trainer.save(f"checkpoints/{algorithm}_iter_{i}")
                logger.info(f"Checkpoint saved: {checkpoint_path}")
        
        training_time = time.time() - start_time
        
        # Store results
        self.trainers[algorithm] = trainer
        self.results[algorithm] = {
            "results": results,
            "training_time": training_time,
            "final_metrics": results[-1] if results else {},
        }
        
        logger.info(f"Training completed in {training_time:.2f} seconds")
        
        return self.results[algorithm]
    
    def train_all_algorithms(self, iterations: int = 100) -> Dict[str, Any]:
        """
        Train all configured algorithms and compare performance.
        
        Args:
            iterations: Number of training iterations per algorithm
            
        Returns:
            Dictionary of results for all algorithms
        """
        algorithms = ["ppo", "a3c", "impala"]  # Add more as needed
        
        logger.info("Starting multi-algorithm training comparison")
        
        all_results = {}
        
        for algorithm in algorithms:
            try:
                logger.info(f"\n{'='*50}")
                logger.info(f"Training {algorithm.upper()}")
                logger.info(f"{'='*50}")
                
                result = self.train_single_algorithm(algorithm, iterations)
                all_results[algorithm] = result
                
            except Exception as e:
                logger.error(f"Error training {algorithm}: {e}")
                all_results[algorithm] = {"error": str(e)}
        
        # Compare results
        self._compare_algorithm_performance(all_results)
        
        return all_results
    
    def _compare_algorithm_performance(self, results: Dict[str, Any]):
        """
        Compare performance across different algorithms.
        
        Args:
            results: Dictionary of results from all algorithms
        """
        logger.info("\n" + "="*60)
        logger.info("ALGORITHM PERFORMANCE COMPARISON")
        logger.info("="*60)
        
        for algorithm, result in results.items():
            if "error" in result:
                logger.info(f"{algorithm.upper()}: ERROR - {result['error']}")
                continue
            
            final_metrics = result.get("final_metrics", {})
            training_time = result.get("training_time", 0)
            
            logger.info(f"\n{algorithm.upper()}:")
            logger.info(f"  Training Time: {training_time:.2f} seconds")
            logger.info(f"  Episode Reward Mean: {final_metrics.get('episode_reward_mean', 0):.2f}")
            logger.info(f"  Episode Length Mean: {final_metrics.get('episode_len_mean', 0):.2f}")
            
            # Agent-specific metrics
            for agent_type in self.config.get("agents", {}).keys():
                agent_reward_key = f"policy_{agent_type}_reward_mean"
                if agent_reward_key in final_metrics:
                    logger.info(f"  {agent_type} Reward: {final_metrics[agent_reward_key]:.2f}")
    
    def evaluate_models(self, num_episodes: int = 10) -> Dict[str, Any]:
        """
        Evaluate trained models on the trading environment.
        
        Args:
            num_episodes: Number of episodes to evaluate
            
        Returns:
            Evaluation results
        """
        logger.info("Starting model evaluation")
        
        evaluation_results = {}
        
        for algorithm, trainer in self.trainers.items():
            logger.info(f"Evaluating {algorithm.upper()} model")
            
            # Create evaluation environment
            eval_env = MultiAgentTradingEnv(self.config)
            
            episode_rewards = []
            episode_lengths = []
            
            for episode in range(num_episodes):
                obs, info = eval_env.reset()
                episode_reward = 0
                episode_length = 0
                
                while True:
                    # Get actions from trained policies
                    actions = {}
                    for agent_id, observation in obs.items():
                        policy_id = self._policy_mapping_fn(agent_id, None, None)
                        action = trainer.compute_single_action(observation, policy_id=policy_id)
                        actions[agent_id] = action
                    
                    # Step environment
                    obs, rewards, terminated, truncated, info = eval_env.step(actions)
                    
                    episode_reward += sum(rewards.values())
                    episode_length += 1
                    
                    # Check if episode is done
                    if all(terminated.values()) or all(truncated.values()):
                        break
                
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
            
            evaluation_results[algorithm] = {
                "mean_reward": np.mean(episode_rewards),
                "std_reward": np.std(episode_rewards),
                "mean_length": np.mean(episode_lengths),
                "std_length": np.std(episode_lengths),
            }
            
            logger.info(f"{algorithm.upper()} Evaluation:")
            logger.info(f"  Mean Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
            logger.info(f"  Mean Length: {np.mean(episode_lengths):.2f} ± {np.std(episode_lengths):.2f}")
        
        return evaluation_results
    
    def cleanup(self):
        """Clean up Ray resources."""
        ray.shutdown()
        logger.info("Ray resources cleaned up")


def main():
    """Main function to run the multi-agent training demo."""
    import argparse
    
    parser = argparse.ArgumentParser(description="RLlib Multi-Agent Trading Demo")
    parser.add_argument("--config", type=str, default="configs/trading_config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--algorithm", type=str, default="all",
                       choices=["ppo", "a3c", "impala", "sac", "all"],
                       help="Algorithm to train")
    parser.add_argument("--iterations", type=int, default=100,
                       help="Number of training iterations")
    parser.add_argument("--eval-episodes", type=int, default=10,
                       help="Number of evaluation episodes")
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = MultiAgentTrainer(args.config)
    
    try:
        if args.algorithm == "all":
            # Train all algorithms
            results = trainer.train_all_algorithms(args.iterations)
        else:
            # Train single algorithm
            results = trainer.train_single_algorithm(args.algorithm, args.iterations)
        
        # Evaluate models
        eval_results = trainer.evaluate_models(args.eval_episodes)
        
        logger.info("\nTraining and evaluation completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    finally:
        trainer.cleanup()


if __name__ == "__main__":
    main()

