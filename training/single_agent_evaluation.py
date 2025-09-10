#!/usr/bin/env python3
"""
Single Agent Evaluation Demo

This script evaluates a trained market maker agent, demonstrating its trading
performance and decision-making capabilities.

Prerequisites:
- Must have run single_agent_demo.py first to train and save a model
- Model should be saved in checkpoints/single_agent_demo/

Usage:
    uv run python training/single_agent_evaluation.py
"""

import os
import sys
import logging
import numpy as np
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environments.trading_environment import TradingEnvironment

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
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
        },
    }


def load_trained_model(checkpoint_path):
    """Load a trained PPO model from checkpoint"""
    logger.info(f"Loading trained model from: {checkpoint_path}")
    
    # Load configuration
    config = create_single_agent_config()
    
    # Create trainer with same configuration as training
    trainer = (
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
            lr=config["training"]["learning_rate"] * 3,
            train_batch_size=64,  # Fixed batch size for evaluation
            gamma=config["training"]["gamma"],
            num_epochs=10,
            clip_param=0.2,
            vf_clip_param=10.0,
            entropy_coeff=0.1,
            minibatch_size=16,
        )
        .env_runners(
            num_env_runners=1,  # Only need 1 for evaluation
            num_cpus_per_env_runner=1,
            rollout_fragment_length=64,  # Match batch size
        )
        .build()
    )
    
    # Restore from checkpoint
    trainer.restore(checkpoint_path)
    logger.info("✅ Model loaded successfully!")
    
    return trainer


def run_evaluation(trainer, config, num_episodes=5):
    """Run evaluation episodes and demonstrate agent performance"""
    logger.info(f"\n🎯 Running {num_episodes} Evaluation Episodes...")
    
    eval_env = TradingEnvironment(config)
    
    total_rewards = []
    total_trades = []
    total_pnl = []
    episode_details = []
    
    for episode in range(num_episodes):
        logger.info(f"\n{'='*50}")
        logger.info(f"Episode {episode + 1}/{num_episodes}")
        logger.info(f"{'='*50}")
        
        obs, info = eval_env.reset()
        episode_reward = 0
        step = 0
        actions_taken = []
        
        # Get the first (and only) agent
        agent_id = list(eval_env.agents.keys())[0]
        agent = eval_env.agents[agent_id]
        
        logger.info(f"Initial State:")
        logger.info(f"  Agent ID: {agent_id}")
        logger.info(f"  Cash: ${agent.cash:,.2f}")
        logger.info(f"  Position: {agent.position:.2f}")
        logger.info(f"  Market Price: ${eval_env.market_simulator.current_price:.2f}")
        
        while step < 200:  # Limit evaluation steps
            # Use the trained model to get actions (new API stack)
            import torch
            module = trainer.get_module("default_policy")
            obs_tensor = torch.tensor(obs.reshape(1, -1), dtype=torch.float32)
            action_dist_inputs = module.forward_inference({"obs": obs_tensor})["action_dist_inputs"]
            
            # Create proper action distribution and sample from it
            # For continuous actions, we need to create a Normal distribution
            from torch.distributions import Normal
            
            # Split the action_dist_inputs into mean and log_std
            # Assuming the first half is mean, second half is log_std
            action_dim = len(action_dist_inputs[0]) // 2
            mean = action_dist_inputs[0, :action_dim]
            log_std = action_dist_inputs[0, action_dim:]
            
            # Ensure std is positive and not too small
            std = torch.exp(log_std).clamp(min=1e-6, max=1.0)
            
            # Create and sample from the distribution
            action_dist = Normal(mean, std)
            action = action_dist.sample().numpy()
            obs, reward, terminated, truncated, info = eval_env.step(action)
            episode_reward += reward
            step += 1
            
            # Track actions for analysis
            action_names = ["HOLD", "BUY", "SELL", "CANCEL"]
            action_name = action_names[int(action[0])]
            qty = action[1]
            price = action[2] if action[2] else "market"
            
            actions_taken.append({
                'step': step,
                'action': action_name,
                'qty': qty,
                'price': price,
                'reward': reward,
                'market_price': eval_env.market_simulator.current_price
            })
            
            # Show agent's decision every 25 steps
            if step % 25 == 0:
                logger.info(f"  Step {step:3d}: {action_name:6s} (qty={qty:6.2f}, price={str(price):>8s}) | "
                           f"Reward: {reward:6.3f} | Market: ${eval_env.market_simulator.current_price:6.2f}")
            
            if terminated or truncated:
                break
        
        # Get final agent state
        agent = eval_env.agents[agent_id]
        total_rewards.append(episode_reward)
        total_trades.append(agent.total_trades)
        total_pnl.append(agent.pnl)
        
        # Store episode details
        episode_details.append({
            'episode': episode + 1,
            'reward': episode_reward,
            'trades': agent.total_trades,
            'pnl': agent.pnl,
            'position': agent.position,
            'cash': agent.cash,
            'actions': actions_taken
        })
        
        logger.info(f"\nEpisode {episode + 1} Results:")
        logger.info(f"  Total Reward: {episode_reward:8.2f}")
        logger.info(f"  Total Trades: {agent.total_trades:8d}")
        logger.info(f"  Final P&L:    ${agent.pnl:8.2f}")
        logger.info(f"  Final Position: {agent.position:8.2f}")
        logger.info(f"  Final Cash:     ${agent.cash:8.2f}")
        logger.info(f"  Market Price:   ${eval_env.market_simulator.current_price:8.2f}")
    
    return episode_details, total_rewards, total_trades, total_pnl


def analyze_performance(episode_details, total_rewards, total_trades, total_pnl):
    """Analyze and display performance statistics"""
    logger.info(f"\n{'='*60}")
    logger.info("📊 PERFORMANCE ANALYSIS")
    logger.info(f"{'='*60}")
    
    # Basic statistics
    logger.info(f"Average Reward:     {np.mean(total_rewards):8.2f} ± {np.std(total_rewards):6.2f}")
    logger.info(f"Average Trades:     {np.mean(total_trades):8.1f} ± {np.std(total_trades):6.1f}")
    logger.info(f"Average P&L:        ${np.mean(total_pnl):8.2f} ± ${np.std(total_pnl):6.2f}")
    logger.info(f"Best Episode Reward: {max(total_rewards):8.2f}")
    logger.info(f"Best Episode P&L:    ${max(total_pnl):8.2f}")
    logger.info(f"Worst Episode P&L:   ${min(total_pnl):8.2f}")
    
    # Action analysis
    all_actions = []
    for episode in episode_details:
        all_actions.extend([action['action'] for action in episode['actions']])
    
    if all_actions:
        action_counts = {action: all_actions.count(action) for action in set(all_actions)}
        total_actions = len(all_actions)
        
        logger.info(f"\nAction Distribution:")
        for action, count in sorted(action_counts.items()):
            percentage = (count / total_actions) * 100
            logger.info(f"  {action:6s}: {count:4d} ({percentage:5.1f}%)")
    
    # Trading performance
    profitable_episodes = sum(1 for pnl in total_pnl if pnl > 0)
    logger.info(f"\nTrading Performance:")
    logger.info(f"  Profitable Episodes: {profitable_episodes}/{len(total_pnl)} ({profitable_episodes/len(total_pnl)*100:.1f}%)")
    logger.info(f"  Total Profit:        ${sum(total_pnl):8.2f}")
    logger.info(f"  Average Profit/Episode: ${np.mean(total_pnl):8.2f}")


def run_single_agent_evaluation():
    """Main evaluation function"""
    try:
        logger.info("🎯 Starting Single Agent Evaluation Demo")
        logger.info("=" * 50)
        
        # Initialize Ray
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
            logger.info("✅ Ray initialized successfully!")
        
        # Check if checkpoint exists
        checkpoint_path = os.path.abspath("checkpoints/single_agent_demo")
        if not os.path.exists(checkpoint_path):
            logger.error(f"❌ Checkpoint not found at: {checkpoint_path}")
            logger.error("Please run 'uv run python training/single_agent_demo.py' first to train a model.")
            return
        
        # Load configuration
        config = create_single_agent_config()
        
        # Load trained model
        trainer = load_trained_model(checkpoint_path)
        
        # Run evaluation
        episode_details, total_rewards, total_trades, total_pnl = run_evaluation(
            trainer, config, num_episodes=5
        )
        
        # Analyze performance
        analyze_performance(episode_details, total_rewards, total_trades, total_pnl)
        
        logger.info(f"\n🎉 Evaluation completed successfully!")
        logger.info("The trained agent has been evaluated and its performance analyzed.")
        
    except Exception as e:
        logger.error(f"❌ Evaluation failed: {e}")
        raise
    finally:
        ray.shutdown()


if __name__ == "__main__":
    run_single_agent_evaluation()
