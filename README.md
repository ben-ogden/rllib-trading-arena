# 🏟️ RLlib Trading Arena

A comprehensive trading arena showcasing RLlib's latest capabilities. Train and evaluate trading strategies in realistic financial markets using Ray 2.49.1.

## 🚀 Features

- **Trading Environment**: Realistic order book simulation with market dynamics
- **Distributed Training**: Leverages Ray's distributed computing for scalable RL training
- **Algorithm Support**: PPO with Ray 2.49.1's new API stack
- **Interactive Dashboard**: Training metrics and progress charts
- **Cloud Ready**: Optimized for cloud deployment and scaling

## 🏗️ Architecture

```
├── environments/          # Trading environment implementations
├── agents/               # Agent-specific configurations and policies
├── training/             # Training scripts and CLI
├── dashboard/            # Interactive monitoring dashboard
└── configs/              # Configuration files
```

## ⚡ Quick Start (5 Minutes)

### 1. Install Dependencies
   ```bash
   # Install uv if you haven't already
   curl -LsSf https://astral.sh/uv/install.sh | sh
   
   # Install project dependencies
   uv sync
   ```

### 2. Train & Evaluate
   ```bash
# Train a trading agent
uv run rllib-trading-arena train --iterations 100

# Evaluate the trained model
uv run rllib-trading-arena evaluate --episodes 5

# View results in dashboard
uv run trading-dashboard
```

## 🎯 Complete Demo (15 Minutes)

### Step 1: Train the Agent
   ```bash
# Start training (this will take 5-10 minutes)
uv run rllib-trading-arena train --iterations 100

# Watch the training progress in your terminal
# The agent will learn to make trading decisions
```

### Step 2: Evaluate Performance
   ```bash
# Test the trained agent
uv run rllib-trading-arena evaluate --episodes 5

# This will show you how well the agent performs
# Look for positive P&L and reasonable trading activity
```

### Step 3: View Results in Dashboard
   ```bash
# Launch the interactive dashboard
   uv run trading-dashboard

# Open your browser to http://localhost:8501
# View training metrics and progress charts
```

## 📊 Dashboard Screenshots

### Training Progress
![Training Progress](assets/training-progress.png)

### Training Performance
![Training Performance](assets/training-performance.png)

### Evaluation Results
![Evaluation Results](assets/evaluation-results.png)

### Evaluation Episodes
![Evaluation Episodes](assets/eval-episodes.png)

## 🎛️ Configuration

The demo uses `configs/trading_config.yaml` for all settings. You can customize these parameters to experiment with different trading strategies and training configurations.

### Market Parameters
```yaml
market:
  initial_price: 100.0        # Starting stock price
  volatility: 0.02           # Daily price volatility (2%)
  max_steps_per_episode: 128 # Must match batch size requirements
  tick_size: 0.01            # Minimum price increment
  liquidity_factor: 0.1      # Market liquidity (higher = more liquid)
  spread_min: 0.01           # Minimum bid-ask spread
  spread_max: 0.05           # Maximum bid-ask spread
```

### Agent Configurations
```yaml
agents:
  market_maker:
    initial_capital: 100000   # Starting capital
    risk_tolerance: 0.1       # Lower = more conservative
    inventory_target: 0       # Target inventory level
    max_inventory: 1000       # Maximum position size
    min_spread: 0.02         # Minimum spread to maintain
    
  momentum_trader:
    initial_capital: 100000   # Starting capital
    risk_tolerance: 0.15      # Higher = more aggressive
    lookback_period: 20       # Period for momentum calculation
    momentum_threshold: 0.05  # Minimum momentum to trade
    
  arbitrageur:
    initial_capital: 100000   # Starting capital
    risk_tolerance: 0.05      # Very conservative
    max_position_size: 500    # Maximum position per trade
    profit_threshold: 0.01    # Minimum profit to execute
```

### Training Parameters
```yaml
training:
  episodes: 1000             # Number of training episodes
  max_steps_per_episode: 128 # Must match batch size requirements
  learning_rate: 0.0003      # Base learning rate (multiplied by 3 in code)
  batch_size: 256            # Will be doubled to 512 in training
  gamma: 0.99                # Discount factor for future rewards
  tau: 0.005                 # Soft update parameter
```

### Distributed Training
```yaml
distributed:
  num_workers: 4             # Number of parallel workers
  num_cpus_per_worker: 1     # CPUs per worker
  num_gpus: 0                # GPUs (0 for CPU-only)
  use_gpu: false             # Enable GPU training
```

### Customization Tips
- **Learning Rate**: Start with 0.0003, increase to 0.0009 for faster learning
- **Batch Size**: Use 256 for good performance, 128 for low memory
- **Workers**: Use 4-8 for good performance, 2 for testing
- **Agent Risk**: Lower risk_tolerance = more conservative trading
- **Episode Length**: Keep at 128 to match batch size requirements

## 📈 Understanding the Results

### Key Metrics to Watch
- **Episode Reward**: Should increase over time (learning)
- **Episode Length**: Should stabilize around 128 steps
- **Policy Loss**: Should decrease (better decision making)
- **P&L**: Should be positive in evaluation (profitable trading)

### What Good Performance Looks Like
- **Training**: Episode rewards trending upward
- **Evaluation**: Positive P&L, reasonable trade frequency
- **Stability**: Consistent performance across episodes
- **Learning**: Clear improvement from random to strategic behavior

## ⚠️ Single-Stock Environment

This demo uses a **single stock** for simplicity. The agent trades one asset (no stock symbols or diversification). This keeps the demo focused and easy to understand, but in real trading you'd typically want multiple assets for portfolio diversification.

## 🎯 What the Agent Can Learn

### ✅ Learnable Trading Patterns
- **Mean Reversion**: Buy low, sell high when prices deviate from trend
- **Event-Based Trading**: React to market events (crashes, rallies)
- **Risk Management**: Control position sizes and stop losses
- **Market Making**: Provide liquidity and capture spreads
- **Timing**: When to enter/exit positions

### ❌ Limitations (Not Learnable)
- **Fundamental Analysis**: No company financials or news
- **Cross-Asset Relationships**: Only single stock trading
- **Market Microstructure**: Simplified order book
- **External Factors**: No economic indicators or sentiment
- **Competition**: No other agents in the market

## 🔧 Advanced Usage

### Training Different Agent Types
```bash
# Train a Market Maker agent (default)
uv run rllib-trading-arena train --agent-type market_maker --iterations 1000

# Train a Momentum Trader agent
uv run rllib-trading-arena train --agent-type momentum_trader --iterations 1000

# Train an Arbitrageur agent
uv run rllib-trading-arena train --agent-type arbitrageur --iterations 1000

# Demo with different agents
uv run rllib-trading-arena demo --agent-type momentum_trader --iterations 10 --render

# Evaluate specific agent type
uv run rllib-trading-arena evaluate --agent-type arbitrageur --episodes 10 --render
```

### CLI Commands
```bash
# Training options
uv run rllib-trading-arena train --iterations 200 --checkpoint-dir custom_path
uv run rllib-trading-arena demo --iterations 10 --render

# Evaluation options
uv run rllib-trading-arena evaluate --episodes 10 --render

# Dashboard options
uv run trading-dashboard --port 8502
```

### Agent Type Comparison
| Agent Type | Strategy | Risk Tolerance | Best For |
|------------|----------|----------------|----------|
| **Market Maker** | Provide liquidity, capture spreads | Low (0.1) | Stable markets, consistent profits |
| **Momentum Trader** | Follow trends, ride momentum | Medium (0.15) | Trending markets, breakout strategies |
| **Arbitrageur** | Exploit price discrepancies | Very Low (0.05) | Efficient markets, quick profits |

### Environment Customization
```python
# Modify environments/trading_environment.py
class TradingEnvironment(gym.Env):
    def __init__(self, config):
        # Market parameters
        self.initial_price = config["market"]["initial_price"]
        self.volatility = config["market"]["volatility"]
        self.liquidity_factor = config["market"]["liquidity_factor"]
        
        # Agent parameters
        self.agent_config = config["agents"]["market_maker"]  # or other agent
        
    def _calculate_reward(self, action, market_state):
        # Customize reward function here
        # Current: PnL-based with risk penalties
        pass
        
    def _get_observation(self):
        # Customize observation space
        # Current: [price, volume, position, cash, market_events]
        pass
```

### Market Dynamics
```python
# Modify environments/market_simulator.py
class MarketSimulator:
    def __init__(self, config):
        # Add new market events
        self.event_types = ["volatility_spike", "liquidity_crisis", 
                           "news_event", "flash_crash", "flash_rally"]
        
    def _generate_market_event(self):
        # Customize market event generation
        # Add new event types or modify probabilities
        pass
```

## 🐛 Troubleshooting

### Training Issues
- **Low Episode Rewards**: Try increasing learning rate (0.0009) or training iterations (1000+)
- **High Policy Loss**: Reduce learning rate or increase batch size (512)
- **No Learning**: Check if rewards are properly scaled, try different agent types
- **Memory Issues**: Reduce batch size (128) or number of workers (2-4)
- **Slow Training**: Increase number of workers (4-8) or use GPU if available
- **Ray Deprecation Warnings**: These are internal Ray issues, not your code - ignore them

### Evaluation Issues
- **All HOLD Actions**: Agent may not be trained enough, try more iterations
- **Negative P&L**: This is normal - trading is hard! Try different agent types
- **Episode Length 0**: Check if environment is properly configured
- **Import Errors**: Make sure you're using `uv run` for all commands

### Dashboard Issues
- **"Not Trained" Status**: Dashboard looks for `checkpoints/single_agent_*` directories
- **No Training Data**: Make sure training completed successfully and saved metrics
- **Empty Charts**: Check if `training_metrics.json` exists in checkpoint directory
- **Dashboard Won't Start**: Try different port: `uv run trading-dashboard --port 8502`

### CLI Issues
- **Command Not Found**: Use `uv run rllib-trading-arena` not just `rllib-trading-arena`
- **Checkpoint Not Found**: Use `--agent-type` parameter to match your training
- **Evaluation Fails**: Make sure you trained a model with the same agent type

### Performance Optimization
- **Batch Size**: Start with 256, adjust based on memory (128 for low memory)
- **Workers**: Use 4-8 workers for good performance (2 for testing)
- **Learning Rate**: 0.0009 works well for most cases (0.0003 for stability)
- **Episode Length**: 128 steps provides good balance (matches batch requirements)
- **Agent Type**: Try different agents - some learn faster than others

## 📚 Learning Resources

### RLlib Documentation
- [RLlib Overview](https://docs.ray.io/en/latest/rllib/index.html)
- [PPO Algorithm](https://docs.ray.io/en/latest/rllib/algorithms/ppo.html)
- [Environment API](https://docs.ray.io/en/latest/rllib/rllib-env.html)

### Ray Documentation
- [Ray Core](https://docs.ray.io/en/latest/ray-core/walkthrough.html)
- [Distributed Training](https://docs.ray.io/en/latest/ray-core/actors.html)

### Distributed Training Resources
- [Ray Cluster Setup](https://docs.ray.io/en/latest/cluster/getting-started.html)
- [Performance Tuning](https://docs.ray.io/en/latest/ray-core/performance-tips.html)

## 🎉 Next Steps

### 🚀 Quick Wins (Start Here)
- **Try Different Agents**: Train and compare all three agent types
- **Experiment with Parameters**: Adjust risk tolerance, learning rates, batch sizes
- **Run Longer Training**: Train for 2000+ iterations to see better results
- **Customize Rewards**: Modify the reward function in `trading_environment.py`
- **Add Technical Indicators**: Include RSI, MACD, or moving averages in observations

### 🔧 Environment Enhancements
- **Multi-Asset Trading**: Add multiple stocks with correlation modeling
- **Realistic Order Book**: Implement depth-of-market simulation
- **Transaction Costs**: Add trading fees, slippage, and market impact
- **Market Events**: Add earnings announcements, news events, sector rotation
- **Trading Sessions**: Implement market hours, pre/post-market trading

### 🤖 Agent Improvements
- **Custom Strategies**: Implement mean reversion, pairs trading, or momentum strategies
- **Multi-Agent Competition**: Enable multiple agents trading simultaneously
- **Risk Management**: Add portfolio-level risk controls and position limits
- **News Integration**: Incorporate sentiment analysis and news events
- **Ensemble Trading**: Combine multiple agent types for robust performance

### 📊 Data & Training
- **Real Market Data**: Connect to Yahoo Finance, Alpha Vantage, or other data feeds
- **Historical Backtesting**: Use real historical data for training and validation
- **Curriculum Learning**: Start with simple markets, gradually increase complexity
- **Hyperparameter Tuning**: Use Ray Tune for automated parameter optimization
- **Transfer Learning**: Pre-train on historical data, fine-tune on recent data

### 🏭 Production Deployment

#### Model Serving
- **Ray Serve**: Deploy trained models as REST APIs
- **Model Versioning**: Implement A/B testing for different model versions
- **Load Balancing**: Handle multiple concurrent trading requests
- **Model Monitoring**: Track model performance and drift in production

#### Trading Infrastructure
- **Real-Time Execution**: Connect to Interactive Brokers, Alpaca, or other brokers
- **Order Management**: Implement sophisticated order routing and execution
- **Risk Controls**: Add real-time position monitoring and risk limits
- **Compliance**: Ensure regulatory compliance for automated trading

#### Monitoring & Operations
- **Performance Tracking**: Monitor P&L, Sharpe ratio, maximum drawdown
- **Alerting**: Set up alerts for unusual trading behavior or system issues
- **Logging**: Comprehensive logging for audit trails and debugging
- **Backup & Recovery**: Implement robust backup and disaster recovery

#### Backtesting Framework
- **Historical Simulation**: Test strategies on years of historical data
- **Walk-Forward Analysis**: Validate strategies on out-of-sample data
- **Monte Carlo Simulation**: Test robustness across different market scenarios
- **Performance Attribution**: Analyze which factors drive trading performance

### ☁️ Cloud Deployment
- **AWS/GCP/Azure**: Configure distributed training on cloud instances
- **Ray Cluster**: Set up multi-node Ray clusters for scaling
- **Container Deployment**: Docker/Kubernetes deployment options
- **Cost Optimization**: Resource management and auto-scaling
- **Monitoring**: Cloud-native monitoring and alerting setup

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.