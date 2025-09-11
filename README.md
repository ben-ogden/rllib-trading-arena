# 🏟️ RLlib Trading Arena

🏟️ A comprehensive trading arena showcasing RLlib's latest capabilities. Train and evaluate trading strategies in realistic financial markets using Ray 2.49.1.

## 🚀 Features

- **Trading Environment**: Realistic order book simulation with market dynamics
- **Distributed Training**: Leverages Ray's distributed computing for scalable RL training
- **Multiple Algorithms**: Supports PPO, A3C, IMPALA, and other RLlib algorithms
- **Interactive Dashboard**: Real-time monitoring of training progress and agent performance
- **Cloud Ready**: Optimized for cloud deployment and scaling

## 🏗️ Architecture

```
├── environments/          # Trading environment implementations
├── agents/               # Agent-specific configurations and policies
├── training/             # Training scripts and configurations
├── dashboard/            # Interactive monitoring dashboard
├── utils/                # Utility functions and helpers
└── configs/              # Configuration files for different scenarios
```

## 🎯 Demo Scenarios

1. **Single Agent Training & Evaluation**: Train and evaluate individual trading strategies
2. **Algorithm Comparison**: Compare different RL algorithms (PPO, A3C, IMPALA)
3. **Hyperparameter Tuning**: Optimize training parameters for better performance
4. **Scalability Demo**: Show distributed training across multiple nodes

## ⚠️ Important Note

This demo uses a **single-stock trading environment** for simplicity. The agent trades one asset (no stock symbols or diversification). This keeps the demo focused and easy to understand, but in real trading you'd typically want multiple assets for portfolio diversification.

## 🎯 What the Agent Can Learn

### ✅ Learnable Trading Patterns

- **Mean Reversion Strategy**: Price tends to revert toward $100, agent can learn to buy low/sell high
- **Event-Based Trading**: React to market events (volatility spikes, flash crashes, news events)
- **Risk Management**: Learn optimal position sizes and cash management
- **Market Making**: Provide liquidity and earn bid-ask spreads
- **Timing**: Learn when to be active vs. passive based on market conditions

### ❌ Limitations (Not Learnable)

- **No Fundamental Analysis**: No company earnings, news, or business fundamentals
- **No Cross-Asset Relationships**: Only one stock, no portfolio diversification
- **No Market Microstructure**: Simplified order book without real market depth
- **No External Factors**: No economic indicators, interest rates, or macro events
- **No Competition**: Only one agent, no other market participants to learn from

### 🎯 Expected Learning Outcomes

The agent should learn to be a **single-stock day trader** with market making capabilities, focusing on mean reversion and event-based trading strategies.

## 🚀 Quick Start

1. **Install Dependencies with uv**:
   ```bash
   # Install uv if you haven't already
   curl -LsSf https://astral.sh/uv/install.sh | sh
   
   # Install project dependencies
   uv sync
   ```

2. **Train Single Agent** (5-10 minutes):
   ```bash
   uv run python training/single_agent_demo.py
   ```

3. **Evaluate Trained Agent** (2-3 minutes):
   ```bash
   uv run python training/single_agent_evaluation.py
   ```

4. **Launch Interactive Dashboard**:
   ```bash
   uv run streamlit run dashboard/trading_dashboard.py
   ```

5. **Alternative: Use CLI commands**:
   ```bash
   uv run rllib-trading-arena --help
   uv run rllib-trading-arena train --algorithm ppo --iterations 100
   uv run trading-dashboard
   ```

6. **For detailed instructions, troubleshooting, and advanced usage, see [DEMO_GUIDE.md](DEMO_GUIDE.md)**

## 🌟 Key RLlib Features Demonstrated

- **Single-Agent Training**: Robust training with latest API stack
- **Distributed Training**: Ray 2.49.1's enhanced distributed computing capabilities
- **Algorithm Flexibility**: Easy switching between RL algorithms including new API stack
- **Scalability**: Horizontal scaling across multiple machines with improved performance
- **Monitoring**: Built-in metrics, custom callbacks, and enhanced observability
- **Cloud Integration**: Cloud-native deployment with latest Ray features
- **New API Stack**: Leverages RLlib's latest modular architecture (alpha features)
- **Enhanced Performance**: Improved training speed and memory efficiency
- **Separate Evaluation**: Dedicated evaluation scripts for comprehensive agent testing

## 🎯 Evaluation Capabilities

The project includes comprehensive evaluation tools:

- **`single_agent_evaluation.py`**: Detailed evaluation of trained single agents
  - Shows diverse trading behavior (BUY, SELL, HOLD, CANCEL)
  - Performance analysis with P&L tracking
  - Action distribution analysis
  - Risk management assessment
  - Realistic trading simulation with proper stochastic sampling

- **Performance Metrics**:
  - Average reward and standard deviation
  - Trading frequency and success rate
  - Profit/Loss analysis
  - Action distribution breakdown
  - Episode-by-episode performance comparison

## 📊 Performance Metrics

- Training time reduction with distributed computing
- Agent performance comparison across algorithms
- Market efficiency metrics
- Scalability benchmarks

## 🔧 Configuration

The demo supports various configuration options. See [DEMO_GUIDE.md](DEMO_GUIDE.md) for detailed configuration examples and options.

## 🌐 Cloud Deployment

This demo is optimized for cloud deployment. See [DEMO_GUIDE.md](DEMO_GUIDE.md) for deployment options and cloud benefits.

## 📈 Results and Insights

The demo showcases RLlib's capabilities in financial markets. See [DEMO_GUIDE.md](DEMO_GUIDE.md) for detailed results analysis and performance insights.

## 🤝 Contributing

This is a demonstration project showcasing RLlib capabilities. Feel free to extend and modify for your own use cases.

## 📄 License

MIT License - See LICENSE file for details.
