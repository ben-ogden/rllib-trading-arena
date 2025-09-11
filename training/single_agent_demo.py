#!/usr/bin/env python3
"""
Single Agent Trading Demo

A simplified demo script that showcases RLlib's capabilities with a single
trading agent. This is perfect for quick demonstrations and testing.
"""

import os
import json
import time
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


def load_config(config_path: str = "configs/trading_config.yaml"):
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Override for single agent training
        config["agents"]["market_maker"]["count"] = 1
        config["training"]["max_steps_per_episode"] = 500  # Shorter episodes for demo
        
        return config
    except FileNotFoundError:
        logger.warning(f"Config file {config_path} not found, using default config")
        return create_default_config()
    except Exception as e:
        logger.warning(f"Error loading config: {e}, using default config")
        return create_default_config()


def create_default_config():
    """Create a default configuration if YAML file is not available."""
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
        config = load_config()
        
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
        training_results = []  # Collect training results for metrics saving
        for i in range(100):  # More iterations for better results  # Reduced iterations for demo
            result = trainer.train()
            training_results.append(result)  # Store each training result
            
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
        
        # Save training metrics for dashboard
        _save_training_metrics(training_results, checkpoint_dir)
        
        # Training completed successfully!
        logger.info("\n✅ Training completed successfully!")
        logger.info("The agent has been trained and the model has been saved.")
        logger.info("Training metrics have been saved for the dashboard.")
        logger.info("You can now use the trained model for trading or further evaluation.")
        
        logger.info("\nDemo completed successfully!")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise
    finally:
        ray.shutdown()


def _save_training_metrics(training_results: list, checkpoint_dir: str):
    """Save training metrics to JSON file for dashboard consumption."""
    try:
        # Extract metrics from training results
        metrics = {
            "training_progress": {
                "iterations": [],
                "episode_rewards": [],
                "episode_lengths": [],
                "policy_losses": [],
            },
            "performance_summary": {
                "total_episodes": 0,
                "average_reward": 0.0,
                "best_reward": 0.0,
                "total_trades": 0,
                "success_rate": 0.0,
                "total_pnl": 0.0,
            },
            "checkpoint_info": {
                "model_path": checkpoint_dir,
                "model_exists": True,
                "last_modified": time.time(),
                "has_training_data": True,
            }
        }
        
        # Process training results
        if training_results:
            for i, result in enumerate(training_results):
                metrics["training_progress"]["iterations"].append(i)
                
                # Extract episode metrics and convert to Python floats for JSON serialization
                episode_return_mean = float(result.get("env_runners", {}).get("episode_return_mean", 0.0))
                episode_len_mean = float(result.get("env_runners", {}).get("episode_len_mean", 0.0))
                policy_loss = float(result.get("learners", {}).get("default_policy", {}).get("policy_loss", 0.0))
                
                metrics["training_progress"]["episode_rewards"].append(episode_return_mean)
                metrics["training_progress"]["episode_lengths"].append(episode_len_mean)
                metrics["training_progress"]["policy_losses"].append(policy_loss)
            
            # Calculate summary metrics
            if metrics["training_progress"]["episode_rewards"]:
                rewards = metrics["training_progress"]["episode_rewards"]
                metrics["performance_summary"]["total_episodes"] = len(rewards)
                metrics["performance_summary"]["average_reward"] = sum(rewards) / len(rewards)
                metrics["performance_summary"]["best_reward"] = max(rewards)
        
        # Save to JSON file
        metrics_file = os.path.join(checkpoint_dir, "training_metrics.json")
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Training metrics saved to: {metrics_file}")
        
    except Exception as e:
        logger.warning(f"Could not save training metrics: {e}")


if __name__ == "__main__":
    run_single_agent_demo()

