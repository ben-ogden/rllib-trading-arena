#!/usr/bin/env python3
"""
Single Agent Trading Demo

A simplified demo script that showcases RLlib's capabilities with a single
trading agent. This is perfect for quick demonstrations and testing.
"""

import os
from pathlib import Path

import ray
import numpy as np
import yaml
import logging

from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.logger import JsonLoggerCallback, CSVLoggerCallback, TBXLoggerCallback
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
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
                rl_module_spec=RLModuleSpec(
                    model_config={
                        "fcnet_hiddens": [256, 256],
                        "fcnet_activation": "tanh",
                    }
                )
            )
            .training(
                lr=config["training"]["learning_rate"] * 3,  # Higher learning rate for faster learning
                train_batch_size=config["training"]["batch_size"] * 2,  # Larger batch size
                gamma=config["training"]["gamma"],
                num_epochs=10,  # Balanced epochs
                clip_param=0.2,  # Standard clip
                vf_clip_param=10.0,
                entropy_coeff=0.1,  # Higher entropy for more exploration
                minibatch_size=16,  # Smaller minibatch for better learning
            )
            .env_runners(
                num_env_runners=config["distributed"]["num_workers"],
                num_cpus_per_env_runner=config["distributed"]["num_cpus_per_worker"],
                rollout_fragment_length=config["training"]["max_steps_per_episode"],  # Ensure episodes complete
            )
            .resources(
                num_gpus=config["distributed"]["num_gpus"],
            )
            .debugging(
                log_level="WARNING",  # Reduce log verbosity
            )
            .callbacks(None)
            .experimental(
                _validate_config=False
            )
        )
        
        # Build the algorithm using the new API stack
        # The deprecation warnings are internal Ray issues, not our configuration
        trainer = ppo_config.build_algo()
        
        logger.info("Starting training...")
        
        # Training loop
        for i in range(100):  # More iterations for better results  # Reduced iterations for demo
            result = trainer.train()
            
            if i % 10 == 0:
                logger.info(f"Iteration {i}:")
                # Ray 2.49.1 metrics structure
                num_episodes = result.get('env_runners', {}).get('num_episodes', 0)
                num_episodes_lifetime = result.get('env_runners', {}).get('num_episodes_lifetime', 0)
                num_env_steps = result.get('env_runners', {}).get('num_env_steps_sampled', 0)
                
                logger.info(f"  Episodes completed this iteration: {num_episodes}")
                logger.info(f"  Total episodes completed: {num_episodes_lifetime}")
                logger.info(f"  Environment steps sampled: {num_env_steps}")
                
                # Show training progress
                if num_episodes_lifetime > 0:
                    logger.info(f"  ✅ Training is progressing - episodes are completing!")
                else:
                    logger.info(f"  ⏳ Training in progress - collecting experience...")
        
        # Save the trained model
        import os
        checkpoint_dir = os.path.abspath("checkpoints/single_agent_demo")
        checkpoint_path = trainer.save(checkpoint_dir)
        logger.info(f"Model saved to: {checkpoint_path}")
        
        # Training completed successfully!
        logger.info("\n✅ Training completed successfully!")
        logger.info("The agent has been trained and the model has been saved.")
        logger.info("You can now use the trained model for trading or further evaluation.")
        
        logger.info("\nDemo completed successfully!")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise
    finally:
        ray.shutdown()


if __name__ == "__main__":
    run_single_agent_demo()

