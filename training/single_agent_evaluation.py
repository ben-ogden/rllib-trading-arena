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
import json
import time
import yaml
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


def load_config(config_path: str = "configs/trading_config.yaml"):
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Override for single agent training
        config["agents"] = {
            "market_maker": {
                "count": 1,
                "initial_capital": 100000,
                "risk_tolerance": 0.1,
                "inventory_target": 0,
                "max_inventory": 1000,
                "min_spread": 0.02,
            }
        }
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
            "num_workers": 1,  # Minimal workers for evaluation
            "num_cpus_per_worker": 1,
            "num_gpus": 0,
        },
    }


def load_trained_model(checkpoint_path):
    """Load a trained PPO model from checkpoint"""
    logger.info(f"Loading trained model from: {checkpoint_path}")
    
    # Load configuration
    config = load_config()
    
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
        .build_algo()
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
            try:
                action_name = action_names[int(action[0])] if len(action) > 0 else "UNKNOWN"
                qty = action[1] if len(action) > 1 else 0.0
                price = action[2] if len(action) > 2 and action[2] is not None else "market"
            except (IndexError, ValueError, TypeError):
                action_name = "UNKNOWN"
                qty = 0.0
                price = "market"
            
            actions_taken.append({
                'step': int(step),
                'action': str(action_name),
                'qty': float(qty),
                'price': str(price) if price != "market" else "market",
                'reward': float(reward),
                'market_price': float(eval_env.market_simulator.current_price)
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
        
        # Store episode details (convert to Python types for JSON serialization)
        # Calculate portfolio metrics
        current_price = eval_env.market_simulator.current_price
        position_value = agent.position * current_price
        total_portfolio_value = agent.cash + position_value
        
        episode_details.append({
            'episode': int(episode + 1),
            'reward': float(episode_reward),
            'trades': int(agent.total_trades),
            'pnl': float(agent.pnl),
            'position': float(agent.position),
            'cash': float(agent.cash),
            'position_value': float(position_value),
            'total_portfolio_value': float(total_portfolio_value),
            'market_price': float(current_price),
            'actions': actions_taken
        })
        
        logger.info(f"\nEpisode {episode + 1} Results:")
        logger.info(f"  Total Reward: {episode_reward:8.2f}")
        # Calculate total portfolio value
        current_price = eval_env.market_simulator.current_price
        position_value = agent.position * current_price
        total_portfolio_value = agent.cash + position_value
        
        logger.info(f"  Total Trades: {agent.total_trades:8d}")
        logger.info(f"  Final P&L:    ${agent.pnl:8.2f}")
        logger.info(f"  Final Position: {agent.position:8.2f}")
        logger.info(f"  Final Cash:     ${agent.cash:8.2f}")
        logger.info(f"  Position Value: ${position_value:8.2f}")
        logger.info(f"  Total Portfolio: ${total_portfolio_value:8.2f}")
        logger.info(f"  Market Price:   ${current_price:8.2f}")
    
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
    
    # Portfolio value statistics
    portfolio_values = [ep['total_portfolio_value'] for ep in episode_details]
    initial_capital = 100000  # Assuming this is the starting capital
    portfolio_returns = [(pv - initial_capital) / initial_capital * 100 for pv in portfolio_values]
    
    logger.info(f"\nPortfolio Performance:")
    logger.info(f"Average Portfolio Value: ${np.mean(portfolio_values):8.2f} ± ${np.std(portfolio_values):6.2f}")
    logger.info(f"Best Portfolio Value:    ${max(portfolio_values):8.2f}")
    logger.info(f"Worst Portfolio Value:   ${min(portfolio_values):8.2f}")
    logger.info(f"Average Return:          {np.mean(portfolio_returns):6.2f}% ± {np.std(portfolio_returns):5.2f}%")
    logger.info(f"Best Return:             {max(portfolio_returns):6.2f}%")
    logger.info(f"Worst Return:            {min(portfolio_returns):6.2f}%")
    
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


def save_evaluation_results(episode_details, total_rewards, total_trades, total_pnl, checkpoint_path):
    """Save evaluation results to JSON file for dashboard consumption."""
    try:
        # Calculate summary statistics and convert to Python floats for JSON serialization
        avg_reward = float(np.mean(total_rewards))
        avg_trades = float(np.mean(total_trades))
        avg_pnl = float(np.mean(total_pnl))
        best_reward = float(max(total_rewards))
        best_pnl = float(max(total_pnl))
        worst_pnl = float(min(total_pnl))
        
        # Count action distribution
        all_actions = []
        for episode in episode_details:
            all_actions.extend([action['action'] for action in episode['actions']])
        
        action_counts = {}
        if all_actions:
            action_counts = {action: all_actions.count(action) for action in set(all_actions)}
        
        # Create evaluation results structure
        evaluation_results = {
            "evaluation_summary": {
                "total_episodes": len(episode_details),
                "average_reward": avg_reward,
                "average_trades": avg_trades,
                "average_pnl": avg_pnl,
                "best_reward": best_reward,
                "best_pnl": best_pnl,
                "worst_pnl": worst_pnl,
                "profitable_episodes": sum(1 for pnl in total_pnl if pnl > 0),
                "success_rate": sum(1 for pnl in total_pnl if pnl > 0) / len(total_pnl) * 100,
            },
            "episode_details": episode_details,
            "action_distribution": action_counts,
            "evaluation_info": {
                "timestamp": time.time(),
                "model_path": checkpoint_path,
                "evaluation_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
        }
        
        # Save to JSON file
        eval_file = os.path.join(checkpoint_path, "evaluation_results.json")
        with open(eval_file, 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        
        logger.info(f"📊 Evaluation results saved to: {eval_file}")
        
    except Exception as e:
        logger.warning(f"Could not save evaluation results: {e}")


def run_single_agent_evaluation(checkpoint_path: str = "checkpoints/single_agent_demo", episodes: int = 5, render: bool = False, reuse_ray: bool = False):
    """Main evaluation function"""
    try:
        logger.info("🎯 Starting Single Agent Evaluation Demo")
        logger.info("=" * 50)
        
        # Initialize Ray (reuse existing cluster if requested)
        if not reuse_ray:
            if ray.is_initialized():
                ray.shutdown()
            ray.init(ignore_reinit_error=True, num_cpus=2)  # Limit CPU usage for evaluation
            logger.info("✅ Ray initialized successfully!")
        else:
            logger.info("✅ Reusing existing Ray cluster from training")
        
        # Check if checkpoint exists
        checkpoint_path_abs = os.path.abspath(checkpoint_path)
        if not os.path.exists(checkpoint_path_abs):
            logger.error(f"❌ Checkpoint not found at: {checkpoint_path}")
            logger.error("Please run 'uv run python training/single_agent_demo.py' first to train a model.")
            return
        
        # Load configuration
        config = load_config()
        
        # Load trained model
        trainer = load_trained_model(checkpoint_path_abs)
        
        # Run evaluation
        episode_details, total_rewards, total_trades, total_pnl = run_evaluation(
            trainer, config, num_episodes=episodes
        )
        
        # Analyze performance
        analyze_performance(episode_details, total_rewards, total_trades, total_pnl)
        
        # Save evaluation results to JSON file for dashboard
        save_evaluation_results(episode_details, total_rewards, total_trades, total_pnl, checkpoint_path)
        
        logger.info(f"\n🎉 Evaluation completed successfully!")
        logger.info("The trained agent has been evaluated and its performance analyzed.")
        
    except Exception as e:
        logger.error(f"❌ Evaluation failed: {e}")
        raise
    finally:
        # Only shutdown Ray if we started it (not reusing from training)
        if not reuse_ray and ray.is_initialized():
            ray.shutdown()
            logger.info("🧹 Ray cluster cleaned up")


if __name__ == "__main__":
    run_single_agent_evaluation()
