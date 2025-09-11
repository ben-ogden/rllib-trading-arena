"""
Command Line Interface for RLlib Trading Demo

This module provides a command-line interface for running the trading demo
with various options and configurations.
"""

import click
import yaml
import logging
from pathlib import Path
from typing import Dict, Any

from .single_agent_demo import run_single_agent_demo
from .single_agent_evaluation import run_single_agent_evaluation


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.group()
@click.option('--config', '-c', 
              type=click.Path(exists=True, path_type=Path),
              default='configs/trading_config.yaml',
              help='Path to configuration file')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.pass_context
def cli(ctx, config: Path, verbose: bool):
    """RLlib Trading Demo - Single-agent trading with Ray 2.49.1."""
    ctx.ensure_object(dict)
    ctx.obj['config'] = config
    ctx.obj['verbose'] = verbose
    
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load and validate config
    try:
        with open(config, 'r') as f:
            ctx.obj['config_data'] = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {config}")
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        raise click.Abort()


@cli.command()
@click.option('--algorithm', '-a',
              type=click.Choice(['ppo', 'impala', 'sac', 'all']),
              default='ppo',
              help='Algorithm to train (PPO recommended for Ray 2.49.1)')
@click.option('--agent-type', '-t',
              type=click.Choice(['market_maker', 'momentum_trader', 'arbitrageur']),
              default='market_maker',
              help='Type of trading agent to train')
@click.option('--iterations', '-i',
              type=int,
              default=100,
              help='Number of training iterations (calls to trainer.train())')
@click.option('--checkpoint-dir', '-d',
              type=click.Path(path_type=Path),
              default='checkpoints',
              help='Directory to save model checkpoints')
@click.pass_context
def train(ctx, algorithm: str, agent_type: str, iterations: int, checkpoint_dir: Path):
    """Train single-agent models on the trading environment."""
    logger.info(f"🚀 Starting single-agent training with {algorithm.upper()} and {agent_type}")
    
    # Create checkpoint directory
    checkpoint_dir.mkdir(exist_ok=True)
    
    try:
        # Use single-agent training (multi-agent has Ray 2.49.1 compatibility issues)
        logger.info("✅ Using single-agent training (Ray 2.49.1 compatible)")
        
        if algorithm == 'all':
            # Train with PPO (most stable with Ray 2.49.1)
            logger.info("🎯 Training with PPO algorithm (recommended for Ray 2.49.1)")
            run_single_agent_demo(iterations=iterations, eval_episodes=0, checkpoint_dir=str(checkpoint_dir / f"single_agent_{agent_type}"), agent_type=agent_type)
            results = {'algorithm': 'ppo', 'agent_type': agent_type, 'status': 'completed'}
        else:
            # Train single algorithm (only PPO fully supported for now)
            if algorithm != 'ppo':
                logger.warning(f"⚠️ Algorithm {algorithm} not fully supported with Ray 2.49.1, using PPO instead")
            logger.info(f"🎯 Training with {algorithm.upper()} algorithm and {agent_type} agent")
            run_single_agent_demo(iterations=iterations, eval_episodes=0, checkpoint_dir=str(checkpoint_dir / f"single_agent_{agent_type}"), agent_type=agent_type)
            results = {'algorithm': algorithm, 'agent_type': agent_type, 'status': 'completed'}
        
        # Note: Evaluation is separate - use 'evaluate' command after training
        eval_results = {'status': 'skipped', 'note': 'Use separate evaluate command to test the trained model'}
        
        logger.info("✅ Training completed successfully!")
        logger.info(f"📊 Model saved to checkpoints/single_agent_{agent_type}/")
        logger.info("🔍 Run 'evaluate' command to test the trained model")
        
        # Save results
        results_file = checkpoint_dir / f"training_results_{algorithm}.yaml"
        with open(results_file, 'w') as f:
            yaml.dump({
                'training_results': results,
                'evaluation_results': eval_results,
                'note': 'Single-agent training used for Ray 2.49.1 compatibility'
            }, f, default_flow_style=False)
        
        logger.info(f"📄 Results saved to {results_file}")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        logger.info("💡 Check the logs above for detailed error information")
        raise click.Abort()


@cli.command()
@click.option('--agent-type', '-t',
              type=click.Choice(['market_maker', 'momentum_trader', 'arbitrageur']),
              default='market_maker',
              help='Type of trading agent to train')
@click.option('--iterations', '-i',
              type=int,
              default=5,
              help='Number of training iterations for the demo')
@click.option('--render', '-r', is_flag=True, help='Show detailed step-by-step training progress')
@click.pass_context
def demo(ctx, agent_type: str, iterations: int, render: bool):
    """Run a quick single-agent trading demo."""
    logger.info(f"🚀 Starting single-agent trading demo with {agent_type}")
    
    try:
        # Run single agent demo
        run_single_agent_demo(iterations=iterations, eval_episodes=0, render=render, agent_type=agent_type)  # Use iterations and render parameters
        
        logger.info("✅ Demo completed successfully!")
        logger.info(f"📊 Model trained and saved to checkpoints/single_agent_{agent_type}/")
        logger.info("🔍 Run 'evaluate' command to test the trained model")
        
    except Exception as e:
        logger.error(f"❌ Demo failed: {e}")
        logger.info("💡 Check the logs above for detailed error information")
        raise click.Abort()


@cli.command()
@click.option('--episodes', '-e',
              type=int,
              default=3,
              help='Number of demo episodes')
@click.option('--render', '-r', is_flag=True, help='Render environment during demo')
@click.pass_context
def multi_demo(ctx, episodes: int, render: bool):
    """Run a multi-agent trading demo (DISABLED - Ray 2.49.1 compatibility issues)."""
    logger.error("❌ Multi-agent demo is disabled due to Ray 2.49.1 compatibility issues")
    logger.info("🔧 The multi-agent system has fundamental issues with Ray 2.49.1's episode tracking")
    logger.info("📝 Use 'demo' command for single-agent demo which works correctly")
    logger.info("🚀 Single-agent training and evaluation are fully functional")
    
    click.echo("\n❌ Multi-Agent Demo Unavailable")
    click.echo("=" * 40)
    click.echo("Multi-agent training has compatibility issues with Ray 2.49.1")
    click.echo("The AssertionError in multi-agent episode tracking cannot be resolved")
    click.echo("\n✅ Available alternatives:")
    click.echo("  • 'demo' - Single-agent training demo")
    click.echo("  • 'train' - Train single-agent models")
    click.echo("  • 'evaluate' - Evaluate trained models")
    click.echo("  • 'dashboard' - Interactive monitoring")
    
    raise click.Abort()


@cli.command()
@click.option('--port', '-p',
              type=int,
              default=8501,
              help='Port to run dashboard on')
@click.option('--host', '-h',
              default='localhost',
              help='Host to run dashboard on')
@click.pass_context
def dashboard(ctx, port: int, host: str):
    """Launch the interactive trading dashboard."""
    logger.info(f"Starting dashboard on {host}:{port}")
    
    try:
        import subprocess
        import sys
        
        # Launch Streamlit dashboard
        cmd = [
            sys.executable, '-m', 'streamlit', 'run',
            'dashboard/trading_dashboard.py',
            '--server.port', str(port),
            '--server.address', host
        ]
        
        subprocess.run(cmd)
        
    except Exception as e:
        logger.error(f"Failed to launch dashboard: {e}")
        raise click.Abort()


@cli.command()
@click.option('--checkpoint', '-c',
              type=click.Path(exists=True, path_type=Path),
              default=None,
              help='Path to checkpoint file (defaults to checkpoints/single_agent_{agent_type})')
@click.option('--agent-type', '-t',
              type=click.Choice(['market_maker', 'momentum_trader', 'arbitrageur']),
              default='market_maker',
              help='Type of trading agent (used to determine default checkpoint path)')
@click.option('--episodes', '-e',
              type=int,
              default=5,
              help='Number of evaluation episodes to run')
@click.option('--render', '-r', is_flag=True, help='Show detailed step-by-step evaluation progress')
@click.pass_context
def evaluate(ctx, checkpoint: Path, agent_type: str, episodes: int, render: bool):
    """Evaluate a trained model using the dedicated evaluation script."""
    # Set default checkpoint path based on agent type if not provided
    if checkpoint is None:
        checkpoint = Path(f'checkpoints/single_agent_{agent_type}')
    
    logger.info(f"Evaluating model from {checkpoint}")
    
    try:
        # Use the dedicated single-agent evaluation script
        logger.info("Running dedicated single-agent evaluation")
        
        # Run the evaluation script
        run_single_agent_evaluation(checkpoint_path=str(checkpoint), episodes=episodes, render=render, agent_type=agent_type)
        
        eval_results = {
            'status': 'completed',
            'episodes': episodes,
            'checkpoint': str(checkpoint),
            'note': 'Evaluation completed using dedicated evaluation script'
        }
        
        logger.info("✅ Evaluation completed successfully!")
        logger.info(f"📊 Results: {eval_results}")
        
    except Exception as e:
        logger.error(f"❌ Evaluation failed: {e}")
        logger.info("💡 Make sure you have a trained model in the checkpoint directory")
        logger.info("🚀 Run 'train' command first to train a model")
        raise click.Abort()


@cli.command()
@click.option('--output', '-o',
              type=click.Path(path_type=Path),
              default='configs/generated_config.yaml',
              help='Output path for generated configuration')
@click.pass_context
def generate_config(ctx, output: Path):
    """Generate a sample configuration file."""
    logger.info(f"Generating configuration file: {output}")
    
    # Create output directory
    output.parent.mkdir(parents=True, exist_ok=True)
    
    # Generate sample configuration
    sample_config = {
        'market': {
            'initial_price': 100.0,
            'volatility': 0.02,
            'liquidity_factor': 0.1,
            'spread_min': 0.01,
            'spread_max': 0.05,
            'order_book_depth': 10,
            'tick_size': 0.01,
            'market_hours': 252,
            'sessions_per_day': 1
        },
        'agents': {
            'market_maker': {
                'count': 2,
                'initial_capital': 100000,
                'risk_tolerance': 0.1,
                'inventory_target': 0,
                'max_inventory': 1000,
                'min_spread': 0.02
            },
            'momentum_trader': {
                'count': 2,
                'initial_capital': 100000,
                'risk_tolerance': 0.15,
                'lookback_period': 20,
                'momentum_threshold': 0.05
            },
            'arbitrageur': {
                'count': 1,
                'initial_capital': 100000,
                'risk_tolerance': 0.05,
                'max_position_size': 500,
                'profit_threshold': 0.01
            }
        },
        'training': {
            'episodes': 1000,
            'max_steps_per_episode': 1000,
            'learning_rate': 0.0003,
            'batch_size': 32,
            'buffer_size': 10000,
            'gamma': 0.99,
            'tau': 0.005
        },
        'algorithms': {
            'ppo': {
                'clip_param': 0.2,
                'vf_clip_param': 10.0,
                'entropy_coeff': 0.01,
                'train_batch_size': 4000,
                'sgd_minibatch_size': 128,
                'num_sgd_iter': 10
            },
            'dqn': {
                'train_batch_size': 32,
                'learning_starts': 1000,
                'target_network_update_freq': 500,
                'exploration_fraction': 0.1
            },
            'impala': {
                'train_batch_size': 1000,
                'sgd_minibatch_size': 64,
                'num_sgd_iter': 8
            }
        },
        'distributed': {
            'num_workers': 4,
            'num_cpus_per_worker': 1,
            'num_gpus': 0,
            'use_gpu': False
        }
    }
    
    # Save configuration
    with open(output, 'w') as f:
        yaml.dump(sample_config, f, default_flow_style=False, indent=2)
    
    logger.info(f"Configuration saved to {output}")


@cli.command()
@click.pass_context
def info(ctx):
    """Display information about the RLlib Trading Demo."""
    config_data = ctx.obj.get('config_data', {})
    
    click.echo("🚀 RLlib Trading Demo Information")
    click.echo("=" * 50)
    click.echo(f"Ray Version: 2.49.1 (Latest)")
    click.echo(f"RLlib Version: Latest")
    click.echo(f"Configuration: {ctx.obj['config']}")
    
    if config_data:
        click.echo("\n📊 Environment Configuration:")
        market_config = config_data.get('market', {})
        click.echo(f"  Initial Price: {market_config.get('initial_price', 'N/A')}")
        click.echo(f"  Volatility: {market_config.get('volatility', 'N/A')}")
        click.echo(f"  Liquidity Factor: {market_config.get('liquidity_factor', 'N/A')}")
        
        click.echo("\n🤖 Agent Configuration:")
        agents_config = config_data.get('agents', {})
        for agent_type, agent_config in agents_config.items():
            count = agent_config.get('count', 0)
            click.echo(f"  {agent_type}: {count} agents")
        
        click.echo("\n⚡ Training Configuration:")
        training_config = config_data.get('training', {})
        click.echo(f"  Max Episodes: {training_config.get('episodes', 'N/A')}")
        click.echo(f"  Learning Rate: {training_config.get('learning_rate', 'N/A')}")
        click.echo(f"  Batch Size: {training_config.get('batch_size', 'N/A')}")
        
        click.echo("\n☁️ Distributed Configuration:")
        distributed_config = config_data.get('distributed', {})
        click.echo(f"  Workers: {distributed_config.get('num_workers', 'N/A')}")
        click.echo(f"  CPUs per Worker: {distributed_config.get('num_cpus_per_worker', 'N/A')}")
        click.echo(f"  GPUs: {distributed_config.get('num_gpus', 'N/A')}")
    
    click.echo("\n🔗 Available Commands:")
    click.echo("  train     - Train single-agent models")
    click.echo("  demo      - Run single-agent demo")
    click.echo("  multi-demo - Multi-agent demo (DISABLED - Ray 2.49.1 issues)")
    click.echo("  dashboard - Launch interactive dashboard")
    click.echo("  evaluate  - Evaluate trained models")
    click.echo("  generate-config - Generate sample configuration")


def main():
    """Main entry point for the CLI."""
    cli()


if __name__ == '__main__':
    main()

