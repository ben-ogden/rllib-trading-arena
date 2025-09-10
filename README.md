# 🏟️ RLlib Trading Arena

🏟️ A competitive multi-agent trading arena showcasing RLlib's latest capabilities. Watch market makers, momentum traders, and arbitrageurs compete in realistic financial markets using Ray 2.49.1 and Anyscale Cloud.

## 🚀 Features

- **Multi-Agent Trading Environment**: Three distinct agent types (Market Maker, Momentum Trader, Arbitrageur)
- **Realistic Market Dynamics**: Order book simulation with realistic price movements and liquidity
- **Distributed Training**: Leverages Ray's distributed computing for scalable RL training
- **Multiple Algorithms**: Supports PPO, A3C, IMPALA, and other RLlib algorithms
- **Interactive Dashboard**: Real-time monitoring of training progress and agent performance
- **Anyscale Integration**: Optimized for Anyscale Cloud deployment

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

1. **Single Agent Training**: Train individual trading strategies
2. **Multi-Agent Competition**: Competing agents in the same market
3. **Cooperative Trading**: Agents working together for optimal market making
4. **Scalability Demo**: Show distributed training across multiple nodes

## 🚀 Quick Start

1. **Install Dependencies with uv**:
   ```bash
   # Install uv if you haven't already
   curl -LsSf https://astral.sh/uv/install.sh | sh
   
   # Install project dependencies
   uv sync
   ```

2. **Run Single Agent Demo** (5 minutes):
   ```bash
   uv run python training/single_agent_demo.py
   ```

3. **Run Multi-Agent Demo** (15 minutes):
   ```bash
   uv run python training/multi_agent_demo.py
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

6. **For detailed instructions, see [DEMO_GUIDE.md](DEMO_GUIDE.md)**

## 🌟 Key RLlib Features Demonstrated

- **Multi-Agent Support**: Independent and shared policies with latest API stack
- **Distributed Training**: Ray 2.49.1's enhanced distributed computing capabilities
- **Algorithm Flexibility**: Easy switching between RL algorithms including new API stack
- **Scalability**: Horizontal scaling across multiple machines with improved performance
- **Monitoring**: Built-in metrics, custom callbacks, and enhanced observability
- **Anyscale Integration**: Cloud-native deployment with latest Ray features
- **New API Stack**: Leverages RLlib's latest modular architecture (alpha features)
- **Enhanced Performance**: Improved training speed and memory efficiency

## 📊 Performance Metrics

- Training time reduction with distributed computing
- Agent performance comparison across algorithms
- Market efficiency metrics
- Scalability benchmarks

## 🔧 Configuration

The demo supports various configuration options:
- Market parameters (volatility, liquidity, etc.)
- Agent behavior parameters
- Training hyperparameters
- Distributed computing settings

## 🌐 Anyscale Cloud Deployment

This demo is optimized for Anyscale Cloud with:
- Pre-configured cluster settings
- Cloud storage integration
- Automatic scaling
- Cost optimization

## 📈 Results and Insights

The demo showcases:
- How different RL algorithms perform in financial markets
- The benefits of multi-agent systems
- Scalability advantages of distributed RL training
- Real-world applicability of RLlib in finance

## 🤝 Contributing

This is a demonstration project showcasing RLlib capabilities. Feel free to extend and modify for your own use cases.

## 📄 License

MIT License - See LICENSE file for details.
