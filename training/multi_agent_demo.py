#!/usr/bin/env python3
"""
Multi-Agent Trading Demo

A comprehensive demo script that showcases RLlib's multi-agent capabilities
with different trading strategies competing in the same market.
"""

import ray
import numpy as np
import yaml
import logging
from pathlib import Path

from ray.rllib.algorithms.ppo import PPOConfig
from training.multi_agent_trainer import MultiAgentTradingEnv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_multi_agent_config():
    """Create configuration for multi-agent training."""
    return {
        "market": {
            "initial_price": 100.0,
            "volatility": 0.02,
            "liquidity_factor": 0.1,
            "tick_size": 0.01,
            "order_book_depth": 10,
            "event_probability": 0.05,
        },
        "agents": {
            "market_maker": {
                "count": 2,
                "initial_capital": 100000,
                "risk_tolerance": 0.1,
                "max_position_size": 1000,
            },
            "momentum_trader": {
                "count": 2,
                "initial_capital": 100000,
                "risk_tolerance": 0.15,
                "max_position_size": 500,
            },
            "arbitrageur": {
                "count": 1,
                "initial_capital": 100000,
                "risk_tolerance": 0.05,
                "max_position_size": 300,
            }
        },
        "training": {
            "max_steps_per_episode": 1000,
            "learning_rate": 0.0003,
            "batch_size": 32,
            "gamma": 0.99,
        },
        "distributed": {
            "num_workers": 4,
            "num_cpus_per_worker": 1,
            "num_gpus": 0,
        }
    }


def run_multi_agent_demo():
    """Run a comprehensive multi-agent trading demo."""
    logger.info("Starting Multi-Agent Trading Demo")
    logger.info("=" * 60)
    
    # Initialize Ray
    ray.init(
        num_cpus=8,
        ignore_reinit_error=True,
        logging_level=logging.INFO
    )
    
    try:
        # Create configuration
        config = create_multi_agent_config()
        
        # Create multi-agent PPO configuration
        ppo_config = (
            PPOConfig()
            .environment(MultiAgentTradingEnv, env_config=config)
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
            .multi_agent(
                policies={
                    "market_maker": None,
                    "momentum_trader": None,
                    "arbitrageur": None,
                },
                policy_mapping_fn=lambda agent_id, episode, worker, **kwargs: agent_id.split("_")[0],
                policies_to_train=["market_maker", "momentum_trader", "arbitrageur"],
            )
            .debugging(
                log_level="INFO",
            )
        )
        
        # Build the algorithm
        trainer = ppo_config.build()
        
        logger.info("Starting multi-agent training...")
        logger.info(f"Agent types: {list(config['agents'].keys())}")
        
        # Training loop
        for i in range(100):  # More iterations for multi-agent
            result = trainer.train()
            
            if i % 20 == 0:
                logger.info(f"Iteration {i}:")
                logger.info(f"  Episode reward mean: {result['episode_reward_mean']:.2f}")
                logger.info(f"  Episode length mean: {result['episode_len_mean']:.2f}")
                
                # Log agent-specific metrics
                for agent_type in config["agents"].keys():
                    policy_key = f"policy_{agent_type}_reward_mean"
                    if policy_key in result:
                        logger.info(f"  {agent_type} reward: {result[policy_key]:.2f}")
        
        # Save the trained model
        checkpoint_path = trainer.save("checkpoints/multi_agent_demo")
        logger.info(f"Model saved to: {checkpoint_path}")
        
        # Multi-agent evaluation
        logger.info("\nRunning multi-agent evaluation...")
        eval_env = MultiAgentTradingEnv(config)
        
        for episode in range(3):
            obs, info = eval_env.reset()
            episode_reward = 0
            step = 0
            agent_rewards = {agent_id: 0 for agent_id in obs.keys()}
            
            logger.info(f"\nEpisode {episode + 1}:")
            
            while step < 500:  # Limit evaluation steps
                # Get actions from all agents
                actions = {}
                for agent_id, observation in obs.items():
                    policy_id = agent_id.split("_")[0]
                    action = trainer.compute_single_action(observation, policy_id=policy_id)
                    actions[agent_id] = action
                
                # Step environment
                obs, rewards, terminated, truncated, info = eval_env.step(actions)
                
                # Accumulate rewards
                for agent_id, reward in rewards.items():
                    agent_rewards[agent_id] += reward
                    episode_reward += reward
                
                step += 1
                
                # Check if episode is done
                if all(terminated.values()) or all(truncated.values()):
                    break
            
            logger.info(f"  Total episode reward: {episode_reward:.2f}")
            logger.info(f"  Episode length: {step}")
            
            # Log individual agent performance
            for agent_id, reward in agent_rewards.items():
                agent_type = agent_id.split("_")[0]
                logger.info(f"  {agent_id} ({agent_type}): {reward:.2f}")
        
        logger.info("\nMulti-agent demo completed successfully!")
        
        # Showcase Anyscale features
        logger.info("\n" + "=" * 60)
        logger.info("ANYSYCALE CLOUD FEATURES DEMONSTRATED:")
        logger.info("=" * 60)
        logger.info("✓ Distributed multi-agent training")
        logger.info("✓ Policy sharing and independent learning")
        logger.info("✓ Scalable training across multiple workers")
        logger.info("✓ Real-time monitoring and metrics")
        logger.info("✓ Checkpoint management")
        logger.info("✓ Multi-algorithm support")
        logger.info("✓ Cloud-native deployment ready")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise
    finally:
        ray.shutdown()


if __name__ == "__main__":
    run_multi_agent_demo()

