#!/usr/bin/env python3
"""
Single Agent Trading Demo

A simplified demo script that showcases RLlib's capabilities with a single
trading agent. This is perfect for quick demonstrations and testing.
"""

import os
import warnings
from pathlib import Path

# Set environment variables BEFORE importing Ray
os.environ["PYTHONWARNINGS"] = "ignore::DeprecationWarning"
os.environ["RAY_PYTHON"] = "/Users/ben/github/rllib-hackathon/.venv/bin/python"

# Suppress specific Ray deprecation warnings
warnings.filterwarnings("ignore", message=".*UnifiedLogger.*", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*RLModule.*", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*JsonLogger.*", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*CSVLogger.*", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*TBXLogger.*", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*Ray 2.7.*", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning, module="ray.*")

import ray
import numpy as np
import yaml
import logging

from ray.rllib.algorithms.ppo import PPOConfig
from environments.trading_environment import TradingEnvironment

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_single_agent_config():
    """Create a simplified configuration for single agent training."""
    return {
        "market": {
            "initial_price": 100.0,
            "volatility": 0.02,
            "liquidity_factor": 0.1,
            "tick_size": 0.01,
            "order_book_depth": 10,
        },
        "agents": {
            "market_maker": {
                "count": 1,
                "initial_capital": 100000,
                "risk_tolerance": 0.1,
                "max_position_size": 1000,
            }
        },
        "training": {
            "max_steps_per_episode": 500,
            "learning_rate": 0.0003,
            "batch_size": 32,
            "gamma": 0.99,
        },
        "distributed": {
            "num_workers": 2,
            "num_cpus_per_worker": 1,
            "num_gpus": 0,
        }
    }


def run_single_agent_demo():
    """Run a simple single agent trading demo."""
    logger.info("Starting Single Agent Trading Demo")
    logger.info("=" * 50)
    
    # Initialize Ray with dashboard
    ray.init(
        num_cpus=4,
        ignore_reinit_error=True,
        logging_level=logging.WARNING,  
        dashboard_host="127.0.0.1",
        dashboard_port=8265,
        include_dashboard=True,
        runtime_env={
            "working_dir": "/Users/ben/github/rllib-hackathon",
            "py_modules": ["/Users/ben/github/rllib-hackathon"]
        }
    )
    
    # Print dashboard URL and check if it's running
    dashboard_url = "http://127.0.0.1:8265"
    logger.info(f"Ray Dashboard available at: {dashboard_url}")
    
    # Check if dashboard is actually running
    import time
    time.sleep(2)  # Give Ray time to start the dashboard
    try:
        import requests
        response = requests.get(dashboard_url, timeout=5)
        if response.status_code == 200:
            logger.info("✅ Ray Dashboard is running and accessible!")
        else:
            logger.warning(f"⚠️  Dashboard returned status code: {response.status_code}")
    except Exception as e:
        logger.warning(f"⚠️  Could not access dashboard: {e}")
        logger.info("Try opening the dashboard manually in your browser")
    
    try:
        # Create configuration
        config = create_single_agent_config()
        
        # Create PPO configuration
        ppo_config = (
            PPOConfig()
            .environment(TradingEnvironment, env_config=config)
            .framework("torch")
            .api_stack(
                enable_rl_module_and_learner=True,
                enable_env_runner_and_connector_v2=True
            )
            .rl_module(
                model_config={
                    "fcnet_hiddens": [256, 256],
                    "fcnet_activation": "tanh",
                }
            )
            .training(
                lr=config["training"]["learning_rate"] * 2,  # Higher learning rate
                train_batch_size=config["training"]["batch_size"],
                gamma=config["training"]["gamma"],
                num_epochs=15,  # More epochs
                clip_param=0.3,  # Slightly higher clip
                vf_clip_param=10.0,
                entropy_coeff=0.05,  # Higher entropy for more exploration
                minibatch_size=32,  # Fix the minibatch size warning
            )
            .env_runners(
                num_env_runners=config["distributed"]["num_workers"],
                num_cpus_per_env_runner=config["distributed"]["num_cpus_per_worker"],
                rollout_fragment_length="auto",
            )
            .resources(
                num_gpus=config["distributed"]["num_gpus"],
            )
            .debugging(
                log_level="WARNING",  # Reduce log verbosity
            )
            .callbacks(
                # Use new logging callbacks instead of deprecated UnifiedLogger
                callbacks_class=None,
            )
            .experimental(
                _validate_config=False
            )
        )
        
        # Build the algorithm
        trainer = ppo_config.build_algo()
        
        logger.info("Starting training...")
        
        # Training loop
        for i in range(100):  # More iterations for better results  # Reduced iterations for demo
            result = trainer.train()
            
            if i % 10 == 0:
                logger.info(f"Iteration {i}:")
                # Handle different result key structures in Ray 2.49.1
                episode_reward = result.get('episode_reward_mean', result.get('env_runners', {}).get('episode_reward_mean', 0.0))
                episode_length = result.get('episode_len_mean', result.get('env_runners', {}).get('episode_len_mean', 0.0))
                logger.info(f"  Episode reward mean: {episode_reward:.2f}")
                logger.info(f"  Episode length mean: {episode_length:.2f}")
                
                # Policy loss logging removed for cleaner output
        
        # Save the trained model
        import os
        checkpoint_dir = os.path.abspath("checkpoints/single_agent_demo")
        checkpoint_path = trainer.save(checkpoint_dir)
        logger.info(f"Model saved to: {checkpoint_path}")
        
        # Quick evaluation
        logger.info("\nRunning evaluation...")
        eval_env = TradingEnvironment(config)
        
        total_rewards = []
        for episode in range(5):
            obs, info = eval_env.reset()
            episode_reward = 0
            step = 0
            
            while step < 200:  # Limit evaluation steps
                # Use the new API for getting actions
                import torch
                module = trainer.get_module("default_policy")
                obs_tensor = torch.tensor(obs.reshape(1, -1), dtype=torch.float32)
                action = module.forward_inference({"obs": obs_tensor})["action_dist_inputs"]
                action = action.numpy().flatten()
                
                obs, reward, terminated, truncated, info = eval_env.step(action)
                episode_reward += reward
                step += 1
                
                if terminated or truncated:
                    break
            
            total_rewards.append(episode_reward)
            logger.info(f"Episode {episode + 1}: Reward = {episode_reward:.2f}")
        
        logger.info(f"\nEvaluation Results:")
        logger.info(f"  Mean reward: {np.mean(total_rewards):.2f}")
        logger.info(f"  Std reward: {np.std(total_rewards):.2f}")
        
        logger.info("\nDemo completed successfully!")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise
    finally:
        ray.shutdown()


if __name__ == "__main__":
    run_single_agent_demo()

