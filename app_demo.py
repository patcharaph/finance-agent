"""
Finance Agent Demo Application
==============================

This application demonstrates the Finance Agent capabilities with both CLI and Streamlit interfaces.
It provides a user-friendly way to interact with the agent and see real-time reasoning.

Usage:
    CLI: python app_demo.py --symbol PTT.BK --horizon 5
    Streamlit: streamlit run app_demo.py
"""

import os
import sys
import argparse
import time
import json
from typing import Dict, Any, Optional
from datetime import datetime

# Add the current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import our agent components
from agent.agent import FinanceAgent, AgentConfig
from agent.tools import DataLoader


class LLMClient:
    """
    Simple LLM client that works with both OpenRouter and OpenAI
    """
    def __init__(self, model: str = None):
        # Auto-detect provider
        self.provider = "openrouter" if os.getenv("OPENROUTER_API_KEY") else ("openai" if os.getenv("OPENAI_API_KEY") else "none")
        
        # Choose model
        env_model = os.getenv("OPENROUTER_MODEL") or os.getenv("LLM_MODEL")
        if self.provider == "openrouter":
            self.model = model or env_model or "openrouter/auto"
            self._api_key = os.getenv("OPENROUTER_API_KEY")
            self._base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
            self._site_url = os.getenv("OPENROUTER_SITE_URL", "https://github.com/patcharaph/finance-agent.git")
            self._app_name = os.getenv("OPENROUTER_APP_NAME", "Finance Agent Demo")
            self.available = True if self._api_key else False
        elif self.provider == "openai":
            self.model = model or env_model or "gpt-4o-mini"
            try:
                import openai
                self._openai = openai
                self._openai.api_key = os.getenv("OPENAI_API_KEY")
                self.available = True
            except Exception:
                self._openai = None
                self.available = False
        else:
            self.model = model or env_model or ""
            self.available = False

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        if not self.available:
            return None

        if self.provider == "openrouter":
            import requests
            headers = {
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": self._site_url,
                "X-Title": self._app_name,
            }
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.2,
            }
            try:
                r = requests.post(f"{self._base_url}/chat/completions", headers=headers, json=payload, timeout=60)
                r.raise_for_status()
                data = r.json()
                return data["choices"][0]["message"]["content"].strip()
            except Exception:
                return None

        # OpenAI legacy ChatCompletion path
        try:
            resp = self._openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
            )
            return resp["choices"][0]["message"]["content"].strip()
        except Exception:
            return None


class FinanceAgentDemo:
    """
    Demo application for the Finance Agent
    """
    
    def __init__(self):
        self.llm_client = LLMClient()
        self.agent = None
        self.execution_logs = []
    
    def setup_agent(self, config: AgentConfig = None) -> FinanceAgent:
        """Setup the finance agent with configuration"""
        if config is None:
            config = AgentConfig(
                max_loops=3,
                max_execution_time=300,
                confidence_threshold=0.7,
                enable_llm_planning=self.llm_client.available,
                enable_learning=True
            )
        
        def logger(level: str, message: str, data: Dict[str, Any] = None):
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "level": level,
                "message": message,
                "data": data or {}
            }
            self.execution_logs.append(log_entry)
            print(f"[{level}] {message}")
            if data:
                print(f"  Data: {json.dumps(data, indent=2, default=str)}")
        
        self.agent = FinanceAgent(config, self.llm_client, logger)
        return self.agent
    
    def run_analysis(self, symbol: str, horizon: int = 5, period: str = "2y", 
                    plan_type: str = "comprehensive_analysis") -> Dict[str, Any]:
        """
        Run financial analysis for a given symbol
        
        Args:
            symbol: Stock symbol to analyze
            horizon: Prediction horizon in days
            period: Data period to fetch
            plan_type: Type of analysis plan
        
        Returns:
            Analysis results
        """
        try:
            if self.agent is None:
                self.setup_agent()
            
            # Clear previous logs
            self.execution_logs = []
            
            # Define goal and context
            goal = f"Analyze {symbol} stock and provide investment recommendation"
            context = {
                "symbol": symbol,
                "horizon": horizon,
                "period": period,
                "plan_type": plan_type
            }
            
            print(f"🚀 Starting analysis for {symbol}")
            print(f"   Goal: {goal}")
            print(f"   Context: {context}")
            print("-" * 50)
            
            # Run the agent
            start_time = time.time()
            result = self.agent.run(goal, context)
            execution_time = time.time() - start_time
            
            print("-" * 50)
            print(f"✅ Analysis completed in {execution_time:.2f} seconds")
            
            return {
                "success": result.get("success", False),
                "result": result,
                "execution_time": execution_time,
                "logs": self.execution_logs
            }
            
        except Exception as e:
            error_msg = f"Analysis failed: {str(e)}"
            print(f"❌ {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "execution_time": 0,
                "logs": self.execution_logs
            }
    
    def get_price_chart_data(self, symbol: str, period: str = "2y") -> Optional[pd.DataFrame]:
        """Get price data for charting"""
        try:
            loader = DataLoader()
            result = loader.fetch_price_data(symbol, period)
            if result.success:
                return result.data
            return None
        except Exception as e:
            print(f"Error fetching price data: {str(e)}")
            return None


def cli_main():
    """Command line interface"""
    parser = argparse.ArgumentParser(description="Finance Agent CLI Demo")
    parser.add_argument("--symbol", default="PTT.BK", help="Stock symbol to analyze")
    parser.add_argument("--horizon", type=int, default=5, help="Prediction horizon in days")
    parser.add_argument("--period", default="2y", help="Data period (1y, 2y, 5y)")
    parser.add_argument("--plan-type", default="comprehensive_analysis", 
                       choices=["basic_analysis", "comprehensive_analysis", "model_optimization"],
                       help="Type of analysis plan")
    parser.add_argument("--max-loops", type=int, default=3, help="Maximum agent loops")
    parser.add_argument("--max-time", type=int, default=300, help="Maximum execution time in seconds")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🤖 Finance Agent CLI Demo")
    print("=" * 60)
    
    # Check API keys
    if not os.getenv("OPENROUTER_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Warning: No API keys found. LLM features will be disabled.")
        print("   Set OPENROUTER_API_KEY or OPENAI_API_KEY environment variable.")
        print()
    
    # Create demo
    demo = FinanceAgentDemo()
    
    # Setup agent config
    config = AgentConfig(
        max_loops=args.max_loops,
        max_execution_time=args.max_time,
        enable_llm_planning=bool(os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY"))
    )
    demo.setup_agent(config)
    
    # Run analysis
    result = demo.run_analysis(
        symbol=args.symbol,
        horizon=args.horizon,
        period=args.period,
        plan_type=args.plan_type
    )
    
    # Display results
    print("\n" + "=" * 60)
    print("📊 ANALYSIS RESULTS")
    print("=" * 60)
    
    if result["success"]:
        final_report = result["result"].get("final_report")
        if final_report:
            print(f"Symbol: {final_report.get('symbol', 'Unknown')}")
            print(f"Data Points: {final_report.get('data_summary', {}).get('price_data_points', 0)}")
            print(f"Last Price: {final_report.get('data_summary', {}).get('last_price', 'N/A')}")
            print(f"Confidence: {final_report.get('confidence', 0):.2f}")
            
            print("\n📈 RECOMMENDATIONS:")
            recommendations = final_report.get('recommendations', [])
            for i, rec in enumerate(recommendations, 1):
                print(f"  {i}. {rec}")
            
            # Model performance
            model_perf = final_report.get('model_performance', {})
            if model_perf:
                print(f"\n🎯 MODEL PERFORMANCE:")
                metrics = model_perf.get('metrics', {})
                print(f"  MAE: {metrics.get('mae', 0):.4f}")
                print(f"  R²: {metrics.get('r2', 0):.4f}")
                print(f"  Relative Performance: {metrics.get('rel_performance', 1):.4f}")
            
            # Risk assessment
            risk = final_report.get('risk_assessment', {})
            if risk:
                print(f"\n⚠️  RISK ASSESSMENT:")
                print(f"  Risk Level: {risk.get('risk_level', 'Unknown')}")
                print(f"  Volatility: {risk.get('volatility', 0):.4f}")
                print(f"  Max Drawdown: {risk.get('max_drawdown', 0):.4f}")
        
        print(f"\n⏱️  Execution Time: {result['execution_time']:.2f} seconds")
        print(f"🔄 Loops Completed: {result['result'].get('loops_completed', 0)}")
        
    else:
        print(f"❌ Analysis failed: {result.get('error', 'Unknown error')}")
    
    print("\n" + "=" * 60)
    print("✅ CLI Demo completed!")


def streamlit_main():
    """Streamlit web interface"""
    st.set_page_config(
        page_title="Finance Agent Demo",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🤖 Finance Agent Demo")
    st.caption("LLM-Agentic AI for Financial Analysis with Planning, Reflection, Memory, and Tool-use")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key status
        has_openrouter = bool(os.getenv("OPENROUTER_API_KEY"))
        has_openai = bool(os.getenv("OPENAI_API_KEY"))
        
        if has_openrouter or has_openai:
            st.success("✅ LLM API Key configured")
            if has_openrouter:
                st.info("Using OpenRouter API")
            else:
                st.info("Using OpenAI API")
        else:
            st.warning("⚠️ No LLM API Key found")
            st.caption("Set OPENROUTER_API_KEY or OPENAI_API_KEY for full functionality")
        
        # Input parameters
        symbol = st.text_input("Stock Symbol", value="PTT.BK", 
                              help="Thai stocks: PTT.BK, DELTA.BK | Index: ^SETI")
        horizon = st.selectbox("Prediction Horizon (days)", options=[5, 10, 20], index=0)
        period = st.selectbox("Data Period", options=["1y", "2y", "5y"], index=1)
        plan_type = st.selectbox("Analysis Type", 
                                options=["basic_analysis", "comprehensive_analysis", "model_optimization"],
                                index=1)
        
        # Agent settings
        st.subheader("Agent Settings")
        max_loops = st.slider("Max Loops", 1, 5, 3)
        max_time = st.slider("Max Time (seconds)", 60, 600, 300)
        enable_learning = st.checkbox("Enable Learning", value=True)
        
        # Run button
        run_analysis = st.button("🚀 Run Analysis", type="primary")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📊 Analysis Results")
        
        if run_analysis:
            # Initialize demo
            demo = FinanceAgentDemo()
            
            # Setup agent config
            config = AgentConfig(
                max_loops=max_loops,
                max_execution_time=max_time,
                enable_llm_planning=has_openrouter or has_openai,
                enable_learning=enable_learning
            )
            demo.setup_agent(config)
            
            # Show progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Run analysis
            with st.spinner("Running analysis..."):
                status_text.text("Initializing agent...")
                progress_bar.progress(10)
                
                status_text.text("Fetching data...")
                progress_bar.progress(30)
                
                result = demo.run_analysis(symbol, horizon, period, plan_type)
                
                status_text.text("Analysis completed!")
                progress_bar.progress(100)
            
            # Display results
            if result["success"]:
                final_report = result["result"].get("final_report")
                
                if final_report:
                    # Summary metrics
                    col_metric1, col_metric2, col_metric3, col_metric4 = st.columns(4)
                    
                    with col_metric1:
                        st.metric("Symbol", final_report.get('symbol', 'Unknown'))
                    
                    with col_metric2:
                        last_price = final_report.get('data_summary', {}).get('last_price')
                        if last_price:
                            st.metric("Last Price", f"฿{last_price:.2f}")
                    
                    with col_metric3:
                        confidence = final_report.get('confidence', 0)
                        st.metric("Confidence", f"{confidence:.1%}")
                    
                    with col_metric4:
                        data_points = final_report.get('data_summary', {}).get('price_data_points', 0)
                        st.metric("Data Points", data_points)
                    
                    # Recommendations
                    st.subheader("📈 Investment Recommendations")
                    recommendations = final_report.get('recommendations', [])
                    for i, rec in enumerate(recommendations, 1):
                        st.write(f"{i}. {rec}")
                    
                    # Model Performance
                    model_perf = final_report.get('model_performance', {})
                    if model_perf:
                        st.subheader("🎯 Model Performance")
                        
                        col_perf1, col_perf2, col_perf3 = st.columns(3)
                        
                        with col_perf1:
                            mae = model_perf.get('metrics', {}).get('mae', 0)
                            st.metric("MAE", f"{mae:.4f}")
                        
                        with col_perf2:
                            r2 = model_perf.get('metrics', {}).get('r2', 0)
                            st.metric("R² Score", f"{r2:.4f}")
                        
                        with col_perf3:
                            rel_perf = model_perf.get('metrics', {}).get('rel_performance', 1)
                            st.metric("Relative Performance", f"{rel_perf:.4f}")
                    
                    # Risk Assessment
                    risk = final_report.get('risk_assessment', {})
                    if risk:
                        st.subheader("⚠️ Risk Assessment")
                        
                        col_risk1, col_risk2, col_risk3 = st.columns(3)
                        
                        with col_risk1:
                            risk_level = risk.get('risk_level', 'Unknown')
                            st.metric("Risk Level", risk_level)
                        
                        with col_risk2:
                            volatility = risk.get('volatility', 0)
                            st.metric("Volatility", f"{volatility:.1%}")
                        
                        with col_risk3:
                            max_dd = risk.get('max_drawdown', 0)
                            st.metric("Max Drawdown", f"{max_dd:.1%}")
                    
                    # Predictions
                    predictions = final_report.get('predictions', {})
                    if predictions and predictions.get('values'):
                        st.subheader("🔮 Price Predictions")
                        pred_values = predictions['values']
                        pred_horizon = predictions.get('horizon', 5)
                        
                        # Create prediction chart
                        dates = pd.date_range(start=pd.Timestamp.now(), periods=len(pred_values), freq='D')
                        pred_df = pd.DataFrame({
                            'date': dates,
                            'predicted_return': pred_values
                        })
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=pred_df['date'],
                            y=pred_df['predicted_return'],
                            mode='lines+markers',
                            name=f'{pred_horizon}-day predictions',
                            line=dict(color='blue', width=2)
                        ))
                        
                        fig.update_layout(
                            title=f"Predicted Returns for {symbol}",
                            xaxis_title="Date",
                            yaxis_title="Predicted Return",
                            hovermode='x unified'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                
                # Execution summary
                st.subheader("⏱️ Execution Summary")
                col_sum1, col_sum2, col_sum3 = st.columns(3)
                
                with col_sum1:
                    st.metric("Execution Time", f"{result['execution_time']:.2f}s")
                
                with col_sum2:
                    loops = result['result'].get('loops_completed', 0)
                    st.metric("Loops Completed", loops)
                
                with col_sum3:
                    plan_progress = result['result'].get('plan_progress', {})
                    completion = plan_progress.get('completion_percentage', 0)
                    st.metric("Plan Completion", f"{completion:.1f}%")
            
            else:
                st.error(f"❌ Analysis failed: {result.get('error', 'Unknown error')}")
    
    with col2:
        st.subheader("📈 Price Chart")
        
        if run_analysis:
            # Get price data for chart
            demo = FinanceAgentDemo()
            price_data = demo.get_price_chart_data(symbol, period)
            
            if price_data is not None:
                # Create price chart
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=price_data.index,
                    y=price_data['close'],
                    mode='lines',
                    name=symbol,
                    line=dict(color='green', width=2)
                ))
                
                fig.update_layout(
                    title=f"{symbol} Price Chart",
                    xaxis_title="Date",
                    yaxis_title="Price (฿)",
                    hovermode='x unified',
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Price statistics
                st.subheader("📊 Price Statistics")
                latest_price = price_data['close'].iloc[-1]
                price_change = price_data['close'].pct_change().iloc[-1]
                volatility = price_data['close'].pct_change().std() * (252 ** 0.5)
                
                st.metric("Latest Price", f"฿{latest_price:.2f}")
                st.metric("Daily Change", f"{price_change:.2%}")
                st.metric("Annual Volatility", f"{volatility:.1%}")
            else:
                st.error("Failed to fetch price data")
        
        st.subheader("📝 Execution Logs")
        
        if run_analysis and 'demo' in locals():
            logs = demo.execution_logs
            if logs:
                # Show recent logs
                recent_logs = logs[-10:]  # Last 10 logs
                for log in recent_logs:
                    level = log['level']
                    message = log['message']
                    timestamp = log['timestamp']
                    
                    if level == "ERROR":
                        st.error(f"[{timestamp}] {message}")
                    elif level == "WARNING":
                        st.warning(f"[{timestamp}] {message}")
                    elif level == "INFO":
                        st.info(f"[{timestamp}] {message}")
                    else:
                        st.text(f"[{timestamp}] {message}")
            else:
                st.info("No execution logs available")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    ### 🚀 About Finance Agent
    
    This is a demonstration of an LLM-Agentic AI system for financial analysis that includes:
    
    - **Planning**: Intelligent task decomposition and execution planning
    - **Acting**: Execution of analysis tasks using various tools
    - **Evaluating**: Comprehensive performance evaluation and quality checks
    - **Reflecting**: Learning from results and adjusting strategies
    - **Memory**: Short-term and long-term memory for learning and adaptation
    
    The agent can analyze stocks, calculate technical indicators, train ML models, 
    assess risk, and provide investment recommendations with reasoning.
    """)


if __name__ == "__main__":
    # Check if running in Streamlit
    if "streamlit" in sys.modules:
        streamlit_main()
    else:
        cli_main()
