import streamlit as st
import yaml
import json
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from typing import Dict, Any, Optional

# Set page config
st.set_page_config(
    page_title="RLlib Trading Demo Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

class TradingDashboard:
    def __init__(self):
        """Initialize the dashboard."""
        self.config = self._load_config()
        self.metrics_data = self._load_metrics_data()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        config_path = Path("configs/trading_config.yaml")
        if config_path.exists():
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        return {}
    
    def _load_metrics_data(self) -> Dict[str, Any]:
        """Load metrics data from training runs."""
        # Only return real data if it exists, otherwise return empty data
        real_data = self._load_real_training_data()
        if real_data and real_data.get("checkpoint_info", {}).get("has_training_data", False):
            return real_data
        # Return empty data structure - no fake metrics
        return self._get_empty_metrics()
    
    def _load_real_training_data(self) -> Optional[Dict[str, Any]]:
        """Load real training data from checkpoints and logs."""
        try:
            # Look for recent training results
            checkpoint_dir = Path("checkpoints")
            if not checkpoint_dir.exists():
                return None
            
            # Check if we have a trained model
            single_agent_checkpoint = checkpoint_dir / "single_agent_demo"
            if single_agent_checkpoint.exists():
                # Check for training metrics file
                metrics_file = single_agent_checkpoint / "training_metrics.json"
                eval_file = single_agent_checkpoint / "evaluation_results.json"
                
                if metrics_file.exists():
                    # Load actual training metrics
                    with open(metrics_file, 'r') as f:
                        data = json.load(f)
                    
                    # Also load evaluation results if available
                    if eval_file.exists():
                        with open(eval_file, 'r') as f:
                            eval_data = json.load(f)
                        data["evaluation_results"] = eval_data
                    
                    return data
                else:
                    # We have a trained model, but no training metrics file
                    data = {
                        "training_progress": {
                            "iterations": [],
                            "episode_rewards": [],
                            "episode_lengths": [],
                            "policy_losses": [],
                        },
                        "agent_performance": {
                            "market_maker": {
                                "rewards": [],
                                "trades": [],
                                "pnl": [],
                            }
                        },
                        "market_metrics": {
                            "prices": [],
                            "volatility": [],
                            "volume": [],
                            "spread": [],
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
                            "model_path": str(single_agent_checkpoint),
                            "model_exists": True,
                            "last_modified": single_agent_checkpoint.stat().st_mtime if single_agent_checkpoint.exists() else None,
                            "has_training_data": False,  # No training metrics file
                        }
                    }
                    return data
            else:
                return None
        except Exception as e:
            st.warning(f"Could not load real training data: {e}")
            return None
    
    def _get_empty_metrics(self) -> Dict[str, Any]:
        """Return empty metrics structure - no fake data."""
        return {
            "training_progress": {
                "iterations": [],
                "episode_rewards": [],
                "episode_lengths": [],
                "policy_losses": [],
            },
            "agent_performance": {
                "market_maker": {
                    "rewards": [],
                    "trades": [],
                    "pnl": [],
                }
            },
            "market_metrics": {
                "prices": [],
                "volatility": [],
                "volume": [],
                "spread": [],
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
                "model_path": "",
                "model_exists": False,
                "last_modified": None,
                "has_training_data": False,
            }
        }
    
    def render_header(self):
        """Render the dashboard header."""
        st.markdown('<h1 class="main-header">🚀 RLlib Trading Demo Dashboard</h1>', unsafe_allow_html=True)
        
        checkpoint_dir = Path("checkpoints")
        single_agent_checkpoint = checkpoint_dir / "single_agent_demo"
        has_model = single_agent_checkpoint.exists()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="Ray Version",
                value="2.49.1",
                delta="Latest"
            )
        
        with col2:
            if has_model:
                st.metric(
                    label="Model Status",
                    value="Trained",
                    delta="Ready for evaluation"
                )
            else:
                st.metric(
                    label="Model Status",
                    value="Not Trained",
                    delta="Run training first"
                )
        
        with col3:
            if has_model:
                st.metric(
                    label="Dashboard Mode",
                    value="Real Data",
                    delta="Showing trained model"
                )
            else:
                st.metric(
                    label="Dashboard Mode",
                    value="Demo Mode",
                    delta="Showing sample data"
                )
    
    def render_training_progress(self):
        """Render training progress visualization."""
        st.header("📊 Training Progress")
        
        data = self.metrics_data["training_progress"]
        
        # Check if we have actual training data
        if not data["iterations"] or len(data["iterations"]) == 0:
            st.info("📝 **No training metrics available**")
            st.markdown("""
            **Why no training data?**
            - RLlib checkpoints don't store training metrics by default
            - Training logs are typically stored separately during training
            - The model was trained but metrics weren't saved to disk
            
            **To see training progress:**
            1. Run training with detailed logging: `uv run rllib-trading-arena train --iterations 100 --render`
            2. Watch the terminal output for real-time metrics
            3. Use Ray Dashboard at http://127.0.0.1:8265 during training
            """)
            return
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Episode Rewards", "Episode Lengths", "Policy Loss", "Learning Curve"),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Episode rewards
        fig.add_trace(
            go.Scatter(
                x=data["iterations"],
                y=data["episode_rewards"],
                mode='lines',
                name='Episode Rewards',
                line=dict(color='#1f77b4', width=2)
            ),
            row=1, col=1
        )
        
        # Episode lengths
        fig.add_trace(
            go.Scatter(
                x=data["iterations"],
                y=data["episode_lengths"],
                mode='lines',
                name='Episode Lengths',
                line=dict(color='#ff7f0e', width=2)
            ),
            row=1, col=2
        )
        
        # Policy loss
        fig.add_trace(
            go.Scatter(
                x=data["iterations"],
                y=data["policy_losses"],
                mode='lines',
                name='Policy Loss',
                line=dict(color='#2ca02c', width=2)
            ),
            row=2, col=1
        )
        
        # Learning curve (smoothed rewards)
        if len(data["episode_rewards"]) > 0:
            smoothed_rewards = []
            window = 10
            for i in range(len(data["episode_rewards"])):
                start = max(0, i - window)
                smoothed_rewards.append(sum(data["episode_rewards"][start:i+1]) / (i - start + 1))
            
            fig.add_trace(
                go.Scatter(
                    x=data["iterations"],
                    y=smoothed_rewards,
                    mode='lines',
                    name='Smoothed Rewards',
                    line=dict(color='#d62728', width=3)
                ),
                row=2, col=2
            )
        
        fig.update_layout(height=600, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    def render_checkpoint_info(self):
        """Render checkpoint and model information."""
        st.header("💾 Model Checkpoint Information")
        
        checkpoint_dir = Path("checkpoints")
        single_agent_checkpoint = checkpoint_dir / "single_agent_demo"
        
        if single_agent_checkpoint.exists():
            st.success("✅ Trained model found!")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Model Status", "Trained", "Ready for evaluation")
            
            with col2:
                total_size = sum(f.stat().st_size for f in single_agent_checkpoint.rglob('*') if f.is_file())
                size_mb = total_size / (1024 * 1024)
                st.metric("Model Size", f"{size_mb:.1f} MB", "Checkpoint files")
            
            with col3:
                last_modified = single_agent_checkpoint.stat().st_mtime
                import datetime
                mod_time = datetime.datetime.fromtimestamp(last_modified)
                st.metric("Last Modified", mod_time.strftime("%Y-%m-%d %H:%M"), "Training completed")
            
            st.subheader("📁 Checkpoint Structure")
            checkpoint_files = []
            for file_path in single_agent_checkpoint.rglob('*'):
                if file_path.is_file():
                    relative_path = file_path.relative_to(single_agent_checkpoint)
                    size = file_path.stat().st_size
                    checkpoint_files.append({
                        "File": str(relative_path),
                        "Size (bytes)": size,
                        "Type": "Model" if "module_state" in str(relative_path) else "Config" if "metadata" in str(relative_path) else "Other"
                    })
            
            if checkpoint_files:
                df = pd.DataFrame(checkpoint_files)
                st.dataframe(df, use_container_width=True)
            
            st.subheader("🔍 Model Evaluation")
            st.info("💡 To evaluate your trained model, run: `uv run python training/single_agent_evaluation.py`")
            
            st.markdown("**Evaluation will show:**")
            st.markdown("- 🎯 Trading behavior (BUY, SELL, HOLD, CANCEL actions)")
            st.markdown("- 📊 Performance metrics (rewards, P&L, trading frequency)")
            st.markdown("- 📈 Action distribution analysis")
            st.markdown("- 🛡️ Risk management assessment")
            
        else:
            st.warning("⚠️ No trained model found")
            st.info("💡 To train a model, run: `uv run rllib-trading-arena train --iterations 100`")
            
            st.markdown("**Training will:**")
            st.markdown("- 🚀 Train a market maker agent using PPO")
            st.markdown("- 📊 Save the model to `checkpoints/single_agent_demo/`")
            st.markdown("- ⏱️ Take 5-10 minutes to complete")
            st.markdown("- 🎯 Show training progress and metrics")
    
    def render_performance_summary(self):
        """Render training performance summary."""
        st.header("📈 Training Performance Summary")
        
        summary = self.metrics_data["performance_summary"]
        
        # Check if we have any data
        if summary["total_episodes"] == 0:
            st.info("📝 **No training data available**")
            st.markdown("""
            **To get training metrics:**
            1. Run training: `uv run rllib-trading-arena train --iterations 100`
            2. Run evaluation: `uv run rllib-trading-arena evaluate --episodes 5`
            3. Refresh this dashboard
            """)
            return
        
        st.info("""
        **Note:** Training metrics show high-level learning progress. For detailed trading performance, 
        see the **Evaluation Results** section below which contains actual agent behavior data.
        """)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Training Episodes", summary["total_episodes"])
        with col2:
            st.metric("Average Reward", f"{summary['average_reward']:.2f}")
        with col3:
            st.metric("Best Reward", f"{summary['best_reward']:.2f}")
        
        # Show training progress chart if we have data
        training_data = self.metrics_data.get("training_progress", {})
        if training_data.get("iterations") and training_data.get("episode_rewards"):
            st.subheader("📊 Training Progress")
            
            import plotly.graph_objects as go
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=training_data["iterations"],
                y=training_data["episode_rewards"],
                mode='lines',
                name='Episode Rewards',
                line=dict(color='#1f77b4', width=2)
            ))
            fig.update_layout(
                title="Episode Rewards Over Training",
                xaxis_title="Training Iteration",
                yaxis_title="Average Episode Reward",
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)

    def render_evaluation_results(self):
        """Render evaluation results section."""
        st.header("🎯 Evaluation Results")
        
        eval_data = self.metrics_data.get("evaluation_results")
        
        if not eval_data:
            st.info("📝 **No evaluation results available**")
            st.markdown("""
            **To see evaluation results:**
            1. Run training: `uv run rllib-trading-arena train --iterations 100`
            2. Run evaluation: `uv run rllib-trading-arena evaluate --episodes 5`
            3. Refresh this dashboard
            """)
            return
        
        eval_summary = eval_data.get("evaluation_summary", {})
        action_dist = eval_data.get("action_distribution", {})
        
        # Evaluation summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Episodes Evaluated", eval_summary.get("total_episodes", 0))
            st.metric("Average Reward", f"{eval_summary.get('average_reward', 0):.2f}")
        
        with col2:
            st.metric("Average P&L", f"${eval_summary.get('average_pnl', 0):,.2f}")
            st.metric("Best P&L", f"${eval_summary.get('best_pnl', 0):,.2f}")
        
        with col3:
            st.metric("Worst P&L", f"${eval_summary.get('worst_pnl', 0):,.2f}")
            st.metric("Success Rate", f"{eval_summary.get('success_rate', 0):.1f}%")
        
        with col4:
            st.metric("Profitable Episodes", f"{eval_summary.get('profitable_episodes', 0)}/{eval_summary.get('total_episodes', 0)}")
            st.metric("Average Trades", f"{eval_summary.get('average_trades', 0):.1f}")
        
        # Portfolio Performance Section
        portfolio_perf = eval_data.get("portfolio_performance", {})
        if portfolio_perf:
            st.subheader("💼 Portfolio Performance")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                avg_portfolio = portfolio_perf.get('average_portfolio_value', 0)
                portfolio_std = portfolio_perf.get('portfolio_value_std', 0)
                st.metric("Average Portfolio Value", f"${avg_portfolio:,.2f}", f"±${portfolio_std:,.2f}")
            
            with col2:
                best_portfolio = portfolio_perf.get('best_portfolio_value', 0)
                worst_portfolio = portfolio_perf.get('worst_portfolio_value', 0)
                st.metric("Best Portfolio Value", f"${best_portfolio:,.2f}")
                st.metric("Worst Portfolio Value", f"${worst_portfolio:,.2f}")
            
            with col3:
                avg_return = portfolio_perf.get('average_return_pct', 0)
                return_std = portfolio_perf.get('return_std_pct', 0)
                st.metric("Average Return", f"{avg_return:.2f}%", f"±{return_std:.2f}%")
            
            with col4:
                best_return = portfolio_perf.get('best_return_pct', 0)
                worst_return = portfolio_perf.get('worst_return_pct', 0)
                st.metric("Best Return", f"{best_return:.2f}%")
                st.metric("Worst Return", f"{worst_return:.2f}%")
        
        # Action distribution
        st.subheader("📊 Action Distribution")
        if action_dist:
            col1, col2, col3, col4 = st.columns(4)
            total_actions = sum(action_dist.values())
            
            with col1:
                buy_pct = (action_dist.get("BUY", 0) / total_actions * 100) if total_actions > 0 else 0
                st.metric("BUY", f"{action_dist.get('BUY', 0)} ({buy_pct:.1f}%)")
            
            with col2:
                sell_pct = (action_dist.get("SELL", 0) / total_actions * 100) if total_actions > 0 else 0
                st.metric("SELL", f"{action_dist.get('SELL', 0)} ({sell_pct:.1f}%)")
            
            with col3:
                hold_pct = (action_dist.get("HOLD", 0) / total_actions * 100) if total_actions > 0 else 0
                st.metric("HOLD", f"{action_dist.get('HOLD', 0)} ({hold_pct:.1f}%)")
            
            with col4:
                cancel_pct = (action_dist.get("CANCEL", 0) / total_actions * 100) if total_actions > 0 else 0
                st.metric("CANCEL", f"{action_dist.get('CANCEL', 0)} ({cancel_pct:.1f}%)")
        
        # Episode details
        episode_details = eval_data.get("episode_details", [])
        if episode_details:
            st.subheader("📋 Episode Summary")
            
            # Create a DataFrame for better display
            df_data = []
            for episode in episode_details:
                df_data.append({
                    "Episode": episode.get("episode", 0),
                    "Reward": f"{episode.get('reward', 0):.2f}",
                    "P&L": f"${episode.get('pnl', 0):,.2f}",
                    "Trades": episode.get("trades", 0),
                    "Position": f"{episode.get('position', 0):.2f}",
                    "Cash": f"${episode.get('cash', 0):,.2f}",
                    "Portfolio Value": f"${episode.get('total_portfolio_value', 0):,.2f}"
                })
            
            if df_data:
                df = pd.DataFrame(df_data)
                st.dataframe(df, use_container_width=True)
            
            # Detailed step-by-step actions
            st.subheader("🔍 Detailed Episode Actions")
            
            # Episode selector
            episode_options = [f"Episode {ep.get('episode', 0)}" for ep in episode_details]
            selected_episode = st.selectbox("Select Episode to View:", episode_options)
            
            if selected_episode:
                episode_num = int(selected_episode.split()[-1]) - 1
                if 0 <= episode_num < len(episode_details):
                    episode = episode_details[episode_num]
                    actions = episode.get("actions", [])
                    
                    if actions:
                        # Create detailed actions DataFrame
                        actions_df = pd.DataFrame(actions)
                        actions_df = actions_df.rename(columns={
                            'step': 'Step',
                            'action': 'Action',
                            'qty': 'Quantity',
                            'price': 'Price',
                            'reward': 'Reward',
                            'market_price': 'Market Price'
                        })
                        
                        # Format the DataFrame
                        actions_df['Market Price'] = actions_df['Market Price'].apply(lambda x: f"${x:.2f}")
                        actions_df['Reward'] = actions_df['Reward'].apply(lambda x: f"{x:.3f}")
                        
                        st.dataframe(actions_df, use_container_width=True, height=400)
                        
                        # Show episode summary
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Steps", len(actions))
                        with col2:
                            st.metric("Total Reward", f"{episode.get('reward', 0):.2f}")
                        with col3:
                            st.metric("Final P&L", f"${episode.get('pnl', 0):,.2f}")
                        with col4:
                            st.metric("Total Trades", episode.get("trades", 0))
                    else:
                        st.info("No detailed actions available for this episode.")
    
    def run(self):
        """Run the complete dashboard."""
        # Render header
        self.render_header()
        
        # Render main content
        self.render_training_progress()
        #self.render_checkpoint_info()
        self.render_performance_summary()
        self.render_evaluation_results()
        
        # Footer
        st.markdown("---")
        checkpoint_dir = Path("checkpoints")
        single_agent_checkpoint = checkpoint_dir / "single_agent_demo"
        has_model = single_agent_checkpoint.exists()
        st.markdown(f"**Ray Version:** 2.49.1 | **Dashboard Mode:** {'Real Data' if has_model else 'Demo Mode'}")

def main():
    """Main function to run the dashboard."""
    dashboard = TradingDashboard()
    dashboard.run()

if __name__ == "__main__":
    main()
