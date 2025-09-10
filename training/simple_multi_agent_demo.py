#!/usr/bin/env python3
"""
Simple Multi-Agent Trading Demo for Ray 2.49.1

This is a simplified multi-agent demo that works with Ray 2.49.1's new API stack.
"""

import logging
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from gymnasium import spaces
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.utils.typing import PolicyID
from ray.tune.logger import JsonLoggerCallback, CSVLoggerCallback, TBXLoggerCallback
from ray.rllib.core.rl_module.rl_module import RLModuleSpec

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleMultiAgentTradingEnv(MultiAgentEnv):
    """
    Simple multi-agent trading environment that works with Ray 2.49.1.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        # Simple agent IDs
        self.agent_ids = ["agent_0", "agent_1", "agent_2"]
        self.possible_agents = self.agent_ids
        self.agents = self.agent_ids
        
        # Multi-agent observation and action spaces (dictionaries)
        self.observation_space = {
            agent_id: spaces.Box(
                low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32
            ) for agent_id in self.agent_ids
        }
        self.action_space = {
            agent_id: spaces.Box(
                low=-1, high=1, shape=(2,), dtype=np.float32
            ) for agent_id in self.agent_ids
        }
        
        # Add get_action_space method required by RLlib
        def get_action_space(agent_id):
            return self.action_space[agent_id]
        
        self.get_action_space = get_action_space
        
        # Environment state
        self.current_step = 0
        self.max_steps = 100
        
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        """Reset the environment."""
        self.current_step = 0
        
        # Return observations for all agents
        observations = {}
        for agent_id in self.agent_ids:
            observations[agent_id] = np.random.randn(4).astype(np.float32)
        
        # Return empty info dict
        info = {}
        
        return observations, info
    
    def step(self, action_dict: Dict[PolicyID, Any]):
        """Step the environment."""
        self.current_step += 1
        
        # Calculate rewards for each agent
        observations = {}
        rewards = {}
        terminated = {}
        truncated = {}
        info = {}
        
        for agent_id in self.agent_ids:
            # Simple observation
            observations[agent_id] = np.random.randn(4).astype(np.float32)
            
            # Simple reward based on action
            if agent_id in action_dict:
                action = action_dict[agent_id]
                rewards[agent_id] = float(np.sum(action)) * 0.1
            else:
                rewards[agent_id] = 0.0
            
            # Termination conditions
            terminated[agent_id] = self.current_step >= self.max_steps
            truncated[agent_id] = False
        
        # Add '__all__' key required for multi-agent environments
        terminated['__all__'] = all(terminated.values())
        truncated['__all__'] = all(truncated.values())
        
        return observations, rewards, terminated, truncated, info


def run_simple_multi_agent_demo():
    """Run a simple multi-agent demo."""
    logger.info("Starting Simple Multi-Agent Trading Demo")
    logger.info("=" * 60)
    
    try:
        # Create configuration
        config = {
            "training": {
                "learning_rate": 0.0003,
                "batch_size": 32,
                "gamma": 0.99,
            },
            "distributed": {
                "num_workers": 1,
                "num_cpus_per_worker": 1,
                "num_gpus": 0,
            }
        }
        
        # Create PPO configuration
        ppo_config = (
            PPOConfig()
            .environment(SimpleMultiAgentTradingEnv, env_config=config)
            .framework("torch")
            .api_stack(
                enable_rl_module_and_learner=True,
                enable_env_runner_and_connector_v2=True
            )
            .rl_module(
                rl_module_spec=RLModuleSpec(
                    model_config={
                        "fcnet_hiddens": [64, 64],
                        "fcnet_activation": "tanh",
                    }
                )
            )
            .training(
                lr=config["training"]["learning_rate"],
                train_batch_size=config["training"]["batch_size"],
                gamma=config["training"]["gamma"],
                num_epochs=5,
                clip_param=0.2,
                vf_clip_param=10.0,
                entropy_coeff=0.01,
                minibatch_size=16,
            )
            .env_runners(
                num_env_runners=config["distributed"]["num_workers"],
                num_cpus_per_env_runner=config["distributed"]["num_cpus_per_worker"],
                rollout_fragment_length="auto",
            )
            .resources(
                num_gpus=config["distributed"]["num_gpus"],
            )
            .multi_agent(
                policies={
                    "policy_0": (None, None, None, {}),
                    "policy_1": (None, None, None, {}),
                    "policy_2": (None, None, None, {}),
                },
                policy_mapping_fn=lambda agent_id, episode, **kwargs: f"policy_{agent_id.split('_')[1]}",
                policies_to_train=["policy_0", "policy_1", "policy_2"],
            )
            .debugging(
                log_level="INFO",
            )
            .callbacks(None)
            .experimental(
                _validate_config=False
            )
        )
        
        # Create trainer using the new API stack
        trainer = ppo_config.build_algo()
        
        logger.info("Starting simple multi-agent training...")
        
        # Training loop
        for i in range(10):
            result = trainer.train()
            
            if i % 5 == 0:
                logger.info(f"Iteration {i}:")
                logger.info(f"  Training iteration: {result.get('training_iteration', 0)}")
        
        logger.info("Simple multi-agent demo completed successfully!")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise
    finally:
        if 'trainer' in locals():
            trainer.stop()


if __name__ == "__main__":
    run_simple_multi_agent_demo()
