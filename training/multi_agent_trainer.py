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

from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
# A3C removed in Ray 2.49.1 - using PPO instead
from ray.rllib.algorithms.impala import ImpalaConfig
from ray.rllib.algorithms.sac import SACConfig
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.utils.typing import PolicyID

from environments.trading_environment import TradingEnvironment
from agents.market_maker import MarketMakerAgent
from agents.momentum_trader import MomentumTraderAgent
from agents.arbitrageur import ArbitrageurAgent


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultiAgentTradingEnv(MultiAgentEnv):
    """
    Multi-agent wrapper for the trading environment.
    
    This class adapts our trading environment to work with RLlib's multi-agent
    training framework, showcasing the latest RLlib features.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the multi-agent trading environment.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__()
        
        self.config = config
        self.env = TradingEnvironment(config)
        
        # Define agent IDs and types
        self.agent_ids = list(self.env.agents.keys())
        self.agent_types = {}
        
        for agent_id in self.agent_ids:
            agent = self.env.agents[agent_id]
            self.agent_types[agent_id] = agent.agent_type
        
        # Environment spaces - multi-agent environments need dict spaces
        self.observation_space = {
            agent_id: self.env.observation_space for agent_id in self.agent_ids
        }
        self.action_space = {
            agent_id: self.env.action_space for agent_id in self.agent_ids
        }
        
        # Episode tracking
        self.episode_count = 0
        self.episode_rewards = {agent_id: [] for agent_id in self.agent_ids}
        
        # Required attributes for Ray 2.49.1 multi-agent environments
        self.agents = self.agent_ids  # Currently active agents
        self.possible_agents = self.agent_ids  # All possible agents
        
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        """
        Reset the environment for a new episode.
        
        Returns:
            Dictionary mapping agent IDs to their initial observations
        """
        # Reset the underlying single-agent environment
        single_obs, info = self.env.reset(seed=seed, options=options)
        self.episode_count += 1
        
        # Initialize episode rewards
        for agent_id in self.agent_ids:
            self.episode_rewards[agent_id] = []
        
        # Convert single observation to multi-agent observations
        # Each agent gets the same observation in this simple setup
        observations = {agent_id: single_obs for agent_id in self.agent_ids}
        
        return observations, info
    
    def step(self, action_dict: Dict[PolicyID, Any]):
        """
        Execute one step of the multi-agent environment.
        
        Args:
            action_dict: Dictionary mapping agent IDs to their actions
            
        Returns:
            Tuple of (observations, rewards, terminated, truncated, info)
        """
        # For simplicity, use the first agent's action to step the environment
        # In a real multi-agent setup, you'd need more sophisticated action aggregation
        first_agent_id = list(action_dict.keys())[0]
        action = action_dict[first_agent_id]
        
        # Convert action to numpy array if needed
        if isinstance(action, (list, tuple)):
            action = np.array(action, dtype=np.float32)
        
        # Step the single-agent environment
        single_obs, single_reward, terminated, truncated, info = self.env.step(action)
        
        # Convert to multi-agent format
        observations = {agent_id: single_obs for agent_id in self.agent_ids}
        rewards = {agent_id: single_reward for agent_id in self.agent_ids}
        terminated_dict = {agent_id: terminated for agent_id in self.agent_ids}
        truncated_dict = {agent_id: truncated for agent_id in self.agent_ids}
        
        # Update episode rewards
        for agent_id, reward in rewards.items():
            self.episode_rewards[agent_id].append(reward)
        
        return observations, rewards, terminated_dict, truncated_dict, info
    
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

