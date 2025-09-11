# 🤖 Finance Agent - LLM-Agentic AI for Financial Analysis

A sophisticated finance analysis agent with planning, reflection, memory, and tool-use capabilities. This agent can analyze stocks, calculate technical indicators, train ML models, assess risk, and provide investment recommendations with intelligent reasoning.

## 🌟 Features

### Core Capabilities
- **🧠 Planning**: Intelligent task decomposition and execution planning
- **⚡ Acting**: Execution of analysis tasks using various tools
- **📊 Evaluating**: Comprehensive performance evaluation and quality checks
- **🔄 Reflecting**: Learning from results and adjusting strategies
- **💾 Memory**: Short-term and long-term memory for learning and adaptation
 - **🧭 Decisioning (Planner+)**: INDEX_SET vs SINGLE_STOCK vs NO_TRADE with rationale
 - **📝 Reporting (TH/EN)**: Bilingual, non‑promissory executive summaries and rationale

### Analysis Tools
- **📈 Data Loading**: Fetch historical price data from Yahoo Finance
- **🔧 Technical Indicators**: RSI, MACD, Bollinger Bands, SMA, EMA, and more
- **📰 Sentiment Analysis**: News sentiment analysis (extensible)
- **🤖 Machine Learning**: Random Forest, Gradient Boosting, Linear Regression
- **⚠️ Risk Assessment**: Volatility, drawdown, and risk level analysis
- **🔮 Predictions**: Price prediction with confidence intervals

### Interfaces
- **🖥️ CLI Interface**: Command-line tool for batch analysis
- **🌐 Streamlit Web App**: Interactive web interface with real-time charts
- **🔌 API Ready**: Modular design for easy integration

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- API Key for LLM (OpenRouter or OpenAI)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/patcharaph/finance-agent.git
   cd finance-agent
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up API keys**
   ```bash
   # For OpenRouter (recommended)
   export OPENROUTER_API_KEY="sk-or-..."
   export OPENROUTER_SITE_URL="https://github.com/patcharaph/finance-agent.git"
   export OPENROUTER_APP_NAME="Finance Agent Demo"
   export OPENROUTER_MODEL="openrouter/auto"
   
   # Or for OpenAI
   export OPENAI_API_KEY="sk-..."
   ```

### Usage

#### 🌐 Streamlit Web Interface (Recommended)
```bash
streamlit run app_demo.py
```
Then open your browser to `http://localhost:8501`

#### 🖥️ Command Line Interface
```bash
# Basic analysis
python app_demo.py --symbol PTT.BK --horizon 5

# Advanced analysis
python app_demo.py --symbol PTT.BK --horizon 10 --period 2y --plan-type comprehensive_analysis

# Help
python app_demo.py --help
```

#### ⚡ Optimized (Lite) Agent
Quick, streamlined run with the optimized agent (smaller codepath, same core flow):
```bash
python demo_optimized.py
```

#### 🐍 Python API (Optimized)
```python
from agent.agent_optimized import OptimizedFinanceAgent, AgentConfig

config = AgentConfig(max_loops=3, confidence_threshold=0.7, enable_llm_planning=True)
agent = OptimizedFinanceAgent(config, llm_client)

goal = "ช่วยประเมินว่าควรลงทุนใน SET หรือเลือกหุ้นรายตัวใน 1-3 เดือนข้างหน้า"
context = {"budget_thb": 200000, "risk_tolerance": "medium", "time_horizon": "1-3m"}
result = agent.run(goal, context)
```

#### 🐍 Python API
```python
from agent.agent import FinanceAgent, AgentConfig

# Create agent
config = AgentConfig(max_loops=3, enable_llm_planning=True)
agent = FinanceAgent(config)

# Run analysis
goal = "Analyze PTT.BK stock and provide investment recommendation"
context = {"symbol": "PTT.BK", "horizon": 5, "period": "2y"}
result = agent.run(goal, context)

print(f"Success: {result['success']}")
print(f"Recommendations: {result['final_report']['recommendations']}")
```

## 📁 Project Structure

```
finance-agent/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── Makefile                 # Build and run commands
├── app_demo.py              # Main demo application (CLI + Streamlit)
├── app_streamlit_news.py    # Legacy Streamlit app
└── agent/                   # Core agent modules
    ├── __init__.py          # Package initialization
    ├── agent.py             # Main agent orchestrator
    ├── agent_optimized.py   # Optimized (lite) agent orchestrator
    ├── tools.py             # Data loading, indicators, ML tools
    ├── memory.py            # Short-term and long-term memory
    ├── evaluator.py         # Performance evaluation system
    ├── planner.py           # Task planning and decomposition
    └── planner_optimized.py # Optimized (lite) planner
├── demo_optimized.py        # Optimized agent demo (quick start)
├── test_enhanced_features.py# E2E test for Planner/Loop/Reflect/Report
├── ENHANCED_FEATURES_IMPLEMENTATION.md # Details of the 4 new features
└── OPTIMIZATION_SUMMARY.md  # Summary of code size/perf optimization
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENROUTER_API_KEY` | OpenRouter API key | None |
| `OPENROUTER_SITE_URL` | Your site URL | `https://github.com/patcharaph/finance-agent.git` |
| `OPENROUTER_APP_NAME` | App name for OpenRouter | `Finance Agent Demo` |
| `OPENROUTER_MODEL` | Model to use | `openrouter/auto` |
| `OPENAI_API_KEY` | OpenAI API key (alternative) | None |

### Agent Configuration

```python
config = AgentConfig(
    max_loops=3,                    # Maximum agent loops
    max_execution_time=300.0,       # Max execution time (seconds)
    confidence_threshold=0.7,       # Minimum confidence threshold
    enable_llm_planning=True,       # Enable LLM-based planning
    enable_learning=True,           # Enable learning and memory
    log_level="INFO",              # Logging level
    storage_path="agent_storage"    # Memory storage path
)
```

## 📊 Analysis Types

### 1. Basic Analysis
- Data fetching and quality check
- Feature engineering with technical indicators
- Model training and evaluation
- Basic predictions and recommendations

### 2. Comprehensive Analysis
- All basic analysis features
- News sentiment analysis
- Risk assessment
- Advanced recommendations

### 3. Model Optimization
- Parameter tuning
- Model comparison
- Performance optimization
- Best model selection

## 🎯 Example Analysis Flow

```mermaid
flowchart TD
    A[User Goal: 'SET50: ควร DCA ไหม?'] --> B[Planner: แปลงเป็น Subtasks]
    B --> C[Act: Tools Execution]
    C -->|Price/Features| D[Short-term Memory]
    C -->|News/Sentiment| D
    C -->|Indicators/Factors| D
    D --> E[Reasoner: เลือก/จูนโมเดล]
    E --> F[Train/Infer]
    F --> G[Evaluator: metrics + constraints (Sharpe/MDD/RelPerf)]
    G -->|Pass| H[Report (TH/EN): executive summaries + rationale]
    G -->|Fail| I[Reflector: วิเคราะห์สาเหตุ, ปรับแผน + store lesson]
    I --> B
    H --> J[Long-term Memory: store playbook/lessons]
```

## 📈 Supported Symbols

### Thai Stocks
- `PTT.BK` - PTT Public Company Limited
- `DELTA.BK` - Delta Electronics (Thailand)
- `ADVANC.BK` - Advanced Info Service
- `KBANK.BK` - Kasikornbank
- And many more...

### Thai Indices
- `^SETI` - SET Index
- `^SET50` - SET50 Index
- `^SET100` - SET100 Index

### International
- Any symbol supported by Yahoo Finance

## 🔍 Technical Details

### Machine Learning Models
- **Random Forest**: Default model for price prediction
- **Gradient Boosting**: Alternative ensemble method
- **Linear Regression**: Simple baseline model

### Technical Indicators
- **RSI**: Relative Strength Index
- **MACD**: Moving Average Convergence Divergence
- **Bollinger Bands**: Volatility bands
- **SMA/EMA**: Simple and Exponential Moving Averages
- **Stochastic**: Momentum oscillator
- **ATR**: Average True Range
- **ADX**: Average Directional Index
- **CCI**: Commodity Channel Index
- **Williams %R**: Momentum indicator
- **OBV**: On-Balance Volume

### Evaluation Metrics
- **MAE**: Mean Absolute Error
- **R²**: Coefficient of determination
- **Relative Performance**: vs naive baseline
- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Worst peak-to-trough decline
- **Directional Accuracy**: Trend prediction accuracy

## 🛠️ Development

### Running Tests
```bash
# Test individual components
python -m agent.tools
python -m agent.memory
python -m agent.evaluator
python -m agent.planner
python -m agent.agent

# Test full system
python app_demo.py --symbol PTT.BK --horizon 5

# Test enhanced features end-to-end
python test_enhanced_features.py
```

### Adding New Tools
1. Extend the `Tools` class in `agent/tools.py`
2. Add task type in `agent/planner.py`
3. Implement execution in `agent/agent.py`

### Adding New Indicators
1. Add indicator calculation in `IndicatorCalculator` class
2. Update `available_indicators` list
3. Test with sample data

## 📝 API Reference

### FinanceAgent Class
```python
class FinanceAgent:
    def __init__(self, config: AgentConfig, llm_client=None, logger=None)
    def run(self, goal: str, context: Dict[str, Any]) -> Dict[str, Any]
    def get_agent_status(self) -> Dict[str, Any]
```

### Key Methods
- `run()`: Execute complete analysis
- `get_agent_status()`: Get current agent state
- `setup_agent()`: Initialize agent with configuration

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Yahoo Finance**: For providing free financial data
- **OpenRouter**: For LLM API access
- **Streamlit**: For the web interface framework
- **scikit-learn**: For machine learning capabilities
- **TA-Lib**: For technical analysis indicators

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/patcharaph/finance-agent/issues)
- **Discussions**: [GitHub Discussions](https://github.com/patcharaph/finance-agent/discussions)
- **Email**: [Your Email]

## 🔮 Roadmap

- [ ] **News API Integration**: Real-time news sentiment analysis
- [ ] **Portfolio Optimization**: Multi-asset portfolio analysis
- [ ] **Backtesting Framework**: Historical strategy testing
- [ ] **Real-time Trading**: Live market data integration
- [ ] **Advanced ML Models**: LSTM, Transformer models
- [ ] **Risk Management**: Advanced risk metrics and hedging
- [ ] **API Endpoints**: REST API for external integration
- [ ] **Mobile App**: React Native mobile application

---

**⚠️ Disclaimer**: This tool is for educational and research purposes only. It is not financial advice. Always do your own research and consult with financial professionals before making investment decisions.
