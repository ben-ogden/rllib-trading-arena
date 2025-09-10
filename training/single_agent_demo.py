#!/usr/bin/env python3
"""
Single Agent Trading Demo

A simplified demo script that showcases RLlib's capabilities with a single
trading agent. This is perfect for quick demonstrations and testing.
"""

import ray
import numpy as np
import yaml
import logging
from pathlib import Path

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
    
    # Initialize Ray
    ray.init(
        num_cpus=4,
        ignore_reinit_error=True,
        logging_level=logging.INFO
    )
    
    try:
        # Create configuration
        config = create_single_agent_config()
        
        # Create PPO configuration
        ppo_config = (
            PPOConfig()
            .environment(TradingEnvironment, env_config=config)
            .framework("torch")
            .training(
                lr=config["training"]["learning_rate"],
                train_batch_size=config["training"]["batch_size"],
                gamma=config["training"]["gamma"],
                num_epochs=10,
                clip_param=0.2,
                vf_clip_param=10.0,
                entropy_coeff=0.01,
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
                log_level="INFO",
            )
            .experimental(
                _validate_config=False
            )
        )
        
        # Build the algorithm
        trainer = ppo_config.build()
        
        logger.info("Starting training...")
        
        # Training loop
        for i in range(50):  # Reduced iterations for demo
            result = trainer.train()
            
            if i % 10 == 0:
                logger.info(f"Iteration {i}:")
                logger.info(f"  Episode reward mean: {result['episode_reward_mean']:.2f}")
                logger.info(f"  Episode length mean: {result['episode_len_mean']:.2f}")
                logger.info(f"  Policy loss: {result['info']['learner']['default_policy']['policy_loss']:.4f}")
        
        # Save the trained model
        checkpoint_path = trainer.save("checkpoints/single_agent_demo")
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
                action = trainer.compute_single_action(obs)
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

