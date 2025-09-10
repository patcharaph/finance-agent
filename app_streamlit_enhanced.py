"""
Enhanced Streamlit Finance Agent App
====================================

This app demonstrates the enhanced finance agent with:
- Investment goal analysis
- LSTM support
- Structured logging with reasoning
- Evaluator presets
- Think-Act-Evaluate-Reflect loop
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json
import time

# Import our enhanced agent
from agent.agent import FinanceAgent, AgentConfig
from settings import get_agent_config, get_streamlit_config, get_investment_goal_template

# Page configuration
st.set_page_config(
    page_title="Enhanced Finance Agent",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Space Theme
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Exo+2:wght@300;400;600&display=swap');
    
    /* Main theme - Space Dark */
    .stApp {
        background: linear-gradient(135deg, #0c0c0c 0%, #1a1a2e 50%, #16213e 100%);
        color: #ffffff;
    }
    
    /* Header styling */
    .main-header {
        font-family: 'Orbitron', monospace;
        font-size: 2.5rem;
        font-weight: 900;
        background: linear-gradient(45deg, #00d4ff, #ff00ff, #00ff88);
        background-size: 200% 200%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 2rem;
        letter-spacing: 2px;
        animation: gradientShift 3s ease-in-out infinite;
        text-shadow: 0 0 20px rgba(0, 212, 255, 0.5);
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
        border-right: 2px solid #00d4ff;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, rgba(0, 212, 255, 0.1) 0%, rgba(255, 0, 255, 0.1) 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border: 1px solid rgba(0, 212, 255, 0.3);
        box-shadow: 0 8px 32px rgba(0, 212, 255, 0.2);
        margin-bottom: 1rem;
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(0, 212, 255, 0.3);
    }
    
    /* Reasoning log */
    .reasoning-log {
        background: linear-gradient(135deg, rgba(26, 26, 46, 0.9) 0%, rgba(22, 33, 62, 0.9) 100%);
        padding: 1rem;
        border-radius: 15px;
        border: 1px solid rgba(0, 212, 255, 0.3);
        max-height: 400px;
        overflow-y: auto;
        box-shadow: 0 8px 32px rgba(0, 212, 255, 0.2);
        backdrop-filter: blur(10px);
    }
    
    /* Log entries */
    .log-entry {
        margin-bottom: 0.75rem;
        padding: 0.75rem;
        border-radius: 10px;
        border-left: 4px solid #00d4ff;
        background: linear-gradient(90deg, rgba(0, 212, 255, 0.1) 0%, rgba(0, 212, 255, 0.05) 100%);
        font-family: 'Exo 2', monospace;
        font-size: 0.9rem;
        color: #ffffff;
        transition: all 0.3s ease;
    }
    
    .log-entry:hover {
        background: linear-gradient(90deg, rgba(0, 212, 255, 0.2) 0%, rgba(0, 212, 255, 0.1) 100%);
        transform: translateX(5px);
    }
    
    .log-info { 
        border-left-color: #00ff88; 
        background: linear-gradient(90deg, rgba(0, 255, 136, 0.1) 0%, rgba(0, 255, 136, 0.05) 100%);
    }
    .log-warning { 
        border-left-color: #ffaa00; 
        background: linear-gradient(90deg, rgba(255, 170, 0, 0.1) 0%, rgba(255, 170, 0, 0.05) 100%);
    }
    .log-error { 
        border-left-color: #ff3366; 
        background: linear-gradient(90deg, rgba(255, 51, 102, 0.1) 0%, rgba(255, 51, 102, 0.05) 100%);
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(45deg, #00d4ff, #ff00ff);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-family: 'Exo 2', sans-serif;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 212, 255, 0.3);
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton > button:hover {
        background: linear-gradient(45deg, #ff00ff, #00d4ff);
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(0, 212, 255, 0.5);
    }
    
    /* Select boxes and inputs */
    .stSelectbox > div > div {
        background: linear-gradient(135deg, rgba(0, 212, 255, 0.1) 0%, rgba(255, 0, 255, 0.1) 100%);
        border: 1px solid rgba(0, 212, 255, 0.3);
        border-radius: 10px;
        color: #ffffff;
    }
    
    .stTextInput > div > div > input {
        background: linear-gradient(135deg, rgba(0, 212, 255, 0.1) 0%, rgba(255, 0, 255, 0.1) 100%);
        border: 1px solid rgba(0, 212, 255, 0.3);
        border-radius: 10px;
        color: #ffffff;
    }
    
    /* Success/Error messages */
    .stSuccess {
        background: linear-gradient(135deg, rgba(0, 255, 136, 0.2) 0%, rgba(0, 255, 136, 0.1) 100%);
        border: 1px solid rgba(0, 255, 136, 0.5);
        border-radius: 10px;
        color: #00ff88;
        font-family: 'Exo 2', sans-serif;
    }
    
    .stError {
        background: linear-gradient(135deg, rgba(255, 51, 102, 0.2) 0%, rgba(255, 51, 102, 0.1) 100%);
        border: 1px solid rgba(255, 51, 102, 0.5);
        border-radius: 10px;
        color: #ff3366;
        font-family: 'Exo 2', sans-serif;
    }
    
    .stInfo {
        background: linear-gradient(135deg, rgba(0, 212, 255, 0.2) 0%, rgba(0, 212, 255, 0.1) 100%);
        border: 1px solid rgba(0, 212, 255, 0.5);
        border-radius: 10px;
        color: #00d4ff;
        font-family: 'Exo 2', sans-serif;
    }
    
    .stWarning {
        background: linear-gradient(135deg, rgba(255, 170, 0, 0.2) 0%, rgba(255, 170, 0, 0.1) 100%);
        border: 1px solid rgba(255, 170, 0, 0.5);
        border-radius: 10px;
        color: #ffaa00;
        font-family: 'Exo 2', sans-serif;
    }
    
    /* Data fetching status */
    .data-status {
        padding: 1rem;
        border-radius: 15px;
        margin: 1rem 0;
        border: 1px solid rgba(0, 212, 255, 0.3);
        backdrop-filter: blur(10px);
    }
    
    .data-success {
        background: linear-gradient(135deg, rgba(0, 255, 136, 0.1) 0%, rgba(0, 255, 136, 0.05) 100%);
        border-left: 4px solid #00ff88;
    }
    
    .data-error {
        background: linear-gradient(135deg, rgba(255, 51, 102, 0.1) 0%, rgba(255, 51, 102, 0.05) 100%);
        border-left: 4px solid #ff3366;
    }
    
    .data-loading {
        background: linear-gradient(135deg, rgba(0, 212, 255, 0.1) 0%, rgba(0, 212, 255, 0.05) 100%);
        border-left: 4px solid #00d4ff;
    }
    
    /* Sliders */
    .stSlider > div > div > div > div {
        background: linear-gradient(90deg, #00d4ff, #ff00ff);
    }
    
    /* Checkboxes */
    .stCheckbox > label > div {
        background: linear-gradient(135deg, rgba(0, 212, 255, 0.1) 0%, rgba(255, 0, 255, 0.1) 100%);
        border: 1px solid rgba(0, 212, 255, 0.3);
        border-radius: 5px;
    }
    
    /* Text areas */
    .stTextArea > div > div > textarea {
        background: linear-gradient(135deg, rgba(0, 212, 255, 0.1) 0%, rgba(255, 0, 255, 0.1) 100%);
        border: 1px solid rgba(0, 212, 255, 0.3);
        border-radius: 10px;
        color: #ffffff;
    }
    
    /* Expander headers */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, rgba(0, 212, 255, 0.1) 0%, rgba(255, 0, 255, 0.1) 100%);
        border: 1px solid rgba(0, 212, 255, 0.3);
        border-radius: 10px;
        color: #ffffff;
        font-family: 'Exo 2', sans-serif;
        font-weight: 600;
    }
    
    /* Metrics */
    .metric-container {
        background: linear-gradient(135deg, rgba(0, 212, 255, 0.1) 0%, rgba(255, 0, 255, 0.1) 100%);
        border: 1px solid rgba(0, 212, 255, 0.3);
        border-radius: 15px;
        padding: 1rem;
        margin: 0.5rem;
        backdrop-filter: blur(10px);
    }
    
    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(26, 26, 46, 0.5);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(45deg, #00d4ff, #ff00ff);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(45deg, #ff00ff, #00d4ff);
    }
    
    /* Animated background stars */
    .stApp::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: 
            radial-gradient(2px 2px at 20px 30px, #00d4ff, transparent),
            radial-gradient(2px 2px at 40px 70px, #ff00ff, transparent),
            radial-gradient(1px 1px at 90px 40px, #00ff88, transparent),
            radial-gradient(1px 1px at 130px 80px, #ffaa00, transparent),
            radial-gradient(2px 2px at 160px 30px, #00d4ff, transparent);
        background-repeat: repeat;
        background-size: 200px 100px;
        animation: sparkle 20s linear infinite;
        pointer-events: none;
        z-index: -1;
    }
    
    @keyframes sparkle {
        from { transform: translateY(0px); }
        to { transform: translateY(-100px); }
    }
</style>
""", unsafe_allow_html=True)

def test_data_fetching(symbol: str) -> dict:
    """Test data fetching and return status with logs"""
    from agent.tools import DataLoader
    
    status = {
        "success": False,
        "data_points": 0,
        "period": "",
        "error": "",
        "logs": []
    }
    
    try:
        # Initialize data loader
        loader = DataLoader()
        
        # Test fetching data
        result = loader.fetch_price_data(symbol, period="1mo")
        
        if result.success:
            status["success"] = True
            status["data_points"] = len(result.data)
            status["period"] = f"{result.data.index[0].date()} to {result.data.index[-1].date()}"
            status["logs"].append(f"✅ Successfully fetched {len(result.data)} data points")
            status["logs"].append(f"📅 Period: {status['period']}")
            status["logs"].append(f"💰 Latest price: {result.data['close'].iloc[-1]:.2f}")
            
            # Check data quality
            if len(result.data) < 20:
                status["logs"].append("⚠️ Warning: Limited data points available")
            else:
                status["logs"].append("✅ Data quality: Good")
                
        else:
            status["error"] = result.error
            status["logs"].append(f"❌ Failed to fetch data: {result.error}")
            
    except Exception as e:
        status["error"] = str(e)
        status["logs"].append(f"❌ Exception occurred: {str(e)}")
    
    return status

def main():
    # Header
    st.markdown('<h1 class="main-header">🚀 SPACE FINANCE AGENT</h1>', unsafe_allow_html=True)
    
    # Yahoo Finance API Status Notice
    st.info("""
    🌌 **Yahoo Finance API Status**: Currently experiencing connectivity issues. 
    The system will automatically use mock data for testing purposes. 
    This is normal and the analysis will still work correctly.
    """)
    
    # Sidebar configuration
    with st.sidebar:
        st.header("🚀 SPACE COMMAND CENTER")
        
        # Investment goal selection
        st.subheader("🌌 MISSION OBJECTIVE")
        investment_goal_type = st.selectbox(
            "Analysis Type",
            ["INDIVIDUAL_STOCK", "SET_INDEX", "PORTFOLIO_OPTIMIZATION"],
            help="Choose the type of investment analysis to perform"
        )
        
        # Get template for selected type
        template = get_investment_goal_template(investment_goal_type)
        
        # Symbol input based on type
        if investment_goal_type == "INDIVIDUAL_STOCK":
            symbol = st.text_input("Stock Symbol", value="PTT.BK", help="Enter stock symbol (e.g., PTT.BK, KBANK.BK)")
            st.caption("🌟 Recommended: PTT.BK, KBANK.BK, SCB.BK, CPALL.BK, ADVANC.BK")
        elif investment_goal_type == "SET_INDEX":
            symbol = st.text_input("Index Symbol", value="^SETI", help="Enter index symbol")
            st.caption("🌟 Recommended: ^SETI, ^SET.BK")
        else:  # PORTFOLIO_OPTIMIZATION
            symbols_input = st.text_input("Stock Symbols", value="PTT.BK,KBANK.BK,SCB.BK", 
                                        help="Enter comma-separated stock symbols")
            symbols = [s.strip() for s in symbols_input.split(",")]
            symbol = symbols[0] if symbols else "PTT.BK"
            st.caption("🌟 Recommended: PTT.BK, KBANK.BK, SCB.BK, CPALL.BK, ADVANC.BK")
        
        # Data fetching test
        st.subheader("🛰️ DATA SCANNER")
        if st.button("SCAN DATA SOURCE", help="Test if we can fetch data for the symbol"):
            with st.spinner("Scanning data source..."):
                data_status = test_data_fetching(symbol)
                
                if data_status["success"]:
                    st.success("✅ Data source connected!")
                    for log in data_status["logs"]:
                        st.text(log)
                    
                    # Check if using mock data
                    if "mock" in str(data_status.get("logs", [])).lower():
                        st.warning("⚠️ Using simulation data - Real-time feed unavailable")
                        st.info("💫 This is normal for testing. Live data will be used when connection is restored.")
                else:
                    st.error("❌ Data source connection failed!")
                    for log in data_status["logs"]:
                        st.text(log)
        
        # Model selection
        st.subheader("🤖 AI BRAIN CONFIG")
        model_type = st.selectbox(
            "Model Type",
            ["random_forest", "gradient_boosting", "linear_regression", "lstm"],
            help="Select machine learning model type"
        )
        
        # Evaluation preset
        evaluation_preset = st.selectbox(
            "Evaluation Preset",
            ["forecasting_preset", "strategy_preset"],
            help="Select evaluation criteria preset"
        )
        
        # Agent configuration
        st.subheader("⚡ MISSION PARAMETERS")
        max_loops = st.slider("Max Loops", 1, 10, 5, help="Maximum number of agent loops")
        max_execution_time = st.slider("Max Execution Time (seconds)", 60, 600, 300, help="Maximum execution time")
        confidence_threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.7, help="Minimum confidence to stop")
        
        # Investment goal description
        investment_goal = st.text_area(
            "Mission Brief",
            value=f"Analyze {symbol} for investment decision using {model_type} model",
            help="Describe your investment goal"
        )
    
    # Main content area
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.header("🌌 MISSION CONTROL")
        
        # Run analysis button
        if st.button("🚀 LAUNCH MISSION", type="primary", use_container_width=True):
            with st.spinner("Initiating space mission..."):
                # Create agent configuration
                config = AgentConfig(
                    max_loops=max_loops,
                    max_execution_time=max_execution_time,
                    confidence_threshold=confidence_threshold,
                    enable_structured_logging=True,
                    log_level="INFO"
                )
                
                # Create agent
                agent = FinanceAgent(config)
                
                # Prepare context
                context = {
                    "symbol": symbol,
                    "symbols": symbols if investment_goal_type == "PORTFOLIO_OPTIMIZATION" else [symbol],
                    "investment_goal": investment_goal,
                    "analysis_type": investment_goal_type,
                    "model_type": model_type,
                    "evaluation_preset": evaluation_preset,
                    "horizon": 5,
                    "period": "2y"
                }
                
                # Run agent
                start_time = time.time()
                result = agent.run(investment_goal, context)
                execution_time = time.time() - start_time
                
                # Store results in session state
                st.session_state.agent_result = result
                st.session_state.agent = agent
                st.session_state.execution_time = execution_time
    
    with col2:
        st.header("🛰️ MISSION STATUS")
        
        if 'agent' in st.session_state:
            agent = st.session_state.agent
            status = agent.get_agent_status()
            
            # Display key metrics in a clean format
            st.metric("Mission ID", status['run_id'][:8])
            st.metric("Cycles", status['loop_count'])
            st.metric("Duration", f"{status['execution_time']:.1f}s")
            st.metric("Data Logs", status['structured_logs_count'])
        else:
            st.info("No mission launched yet")
    
    # Reasoning Log Section - Always visible (moved up)
    st.markdown("---")
    
    # Refresh logs button
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("🔄 REFRESH LOGS", help="Reload logs from file"):
            st.rerun()
    
    with st.expander("🧠 AI BRAIN LOGS", expanded=True):
        if 'agent' in st.session_state:
            agent = st.session_state.agent
            reasoning_log = agent._get_reasoning_log()
            
            if reasoning_log:
                # Filter options
                col1, col2 = st.columns(2)
                with col1:
                    log_level_filter = st.selectbox("Filter", ["All", "INFO", "WARNING", "ERROR"], key="log_filter")
                with col2:
                    show_details = st.checkbox("Details", value=False, key="log_details")
                
                # Display log entries
                st.markdown('<div class="reasoning-log">', unsafe_allow_html=True)
                
                for entry in reasoning_log:
                    level = entry['level']
                    
                    # Apply filter
                    if log_level_filter != "All" and level != log_level_filter:
                        continue
                    
                    # Format timestamp
                    timestamp = datetime.fromisoformat(entry['timestamp'].replace('Z', '+00:00'))
                    time_str = timestamp.strftime("%H:%M:%S")
                    
                    # Create log entry
                    log_class = f"log-{level.lower()}"
                    st.markdown(f'''
                    <div class="log-entry {log_class}">
                        <strong>[{time_str}]</strong> {level} - {entry['message']}
                        {f"<br><small>Loop: {entry['loop']} | Time: {entry['execution_time']:.2f}s</small>" if show_details else ""}
                    </div>
                    ''', unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.info("No reasoning log available yet.")
        else:
            # Show existing logs from file
            st.info("Run an analysis to see the agent's reasoning process here.")
            
            # Try to load existing logs from file
            try:
                import os
                log_file = "agent_logs.json"
                if os.path.exists(log_file):
                    st.subheader("📄 Recent Logs from File")
                    
                    # Read and parse log file
                    with open(log_file, 'r', encoding='utf-8') as f:
                        log_lines = f.readlines()
                    
                    # Show last 10 log entries
                    recent_logs = log_lines[-10:] if len(log_lines) > 10 else log_lines
                    
                    if recent_logs:
                        st.markdown('<div class="reasoning-log">', unsafe_allow_html=True)
                        for line in recent_logs:
                            if line.strip():
                                # Parse log line (simplified)
                                parts = line.split(' - ')
                                if len(parts) >= 3:
                                    timestamp_part = parts[0].split(' ')[1]  # Extract time
                                    level_part = parts[1].split(' ')[0]      # Extract level
                                    message_part = ' - '.join(parts[2:])     # Extract message
                                    
                                    # Clean up message
                                    message_part = message_part.split(' | ')[0]  # Remove extra data
                                    
                                    log_class = f"log-{level_part.lower()}"
                                    st.markdown(f'''
                                    <div class="log-entry {log_class}">
                                        <strong>[{timestamp_part}]</strong> {level_part} - {message_part}
                                    </div>
                                    ''', unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Show log file info
                        st.caption(f"📁 Showing last {len(recent_logs)} entries from {log_file}")
                        
                        # Button to show all logs
                        if st.button("📖 Show All Logs", help="Display all log entries from file"):
                            st.session_state.show_all_logs = True
                        
                        # Show all logs if requested
                        if st.session_state.get('show_all_logs', False):
                            st.subheader("📄 All Logs from File")
                            st.markdown('<div class="reasoning-log">', unsafe_allow_html=True)
                            for line in log_lines:
                                if line.strip():
                                    # Parse log line (simplified)
                                    parts = line.split(' - ')
                                    if len(parts) >= 3:
                                        timestamp_part = parts[0].split(' ')[1]  # Extract time
                                        level_part = parts[1].split(' ')[0]      # Extract level
                                        message_part = ' - '.join(parts[2:])     # Extract message
                                        
                                        # Clean up message
                                        message_part = message_part.split(' | ')[0]  # Remove extra data
                                        
                                        log_class = f"log-{level_part.lower()}"
                                        st.markdown(f'''
                                        <div class="log-entry {log_class}">
                                            <strong>[{timestamp_part}]</strong> {level_part} - {message_part}
                                        </div>
                                        ''', unsafe_allow_html=True)
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                            if st.button("🔼 Hide All Logs"):
                                st.session_state.show_all_logs = False
                                st.rerun()
                    else:
                        st.info("No recent logs found in file.")
                else:
                    st.info("No log file found. Run an analysis to create logs.")
                    
            except Exception as e:
                st.warning(f"Could not load log file: {str(e)}")
                
                # Show sample log entries for demonstration
                st.subheader("📝 Sample Log Structure")
                sample_logs = [
                    {"timestamp": "14:30:15", "level": "INFO", "message": "Finance Agent initialized", "loop": 0, "execution_time": 0.0},
                    {"timestamp": "14:30:16", "level": "INFO", "message": "Starting agent execution", "loop": 0, "execution_time": 0.0},
                    {"timestamp": "14:30:17", "level": "INFO", "message": "THINK: Creating execution plan", "loop": 1, "execution_time": 0.5},
                    {"timestamp": "14:30:18", "level": "INFO", "message": "ACT: Executing data fetch task", "loop": 1, "execution_time": 1.2},
                    {"timestamp": "14:30:19", "level": "WARNING", "message": "Data fetch failed - retrying", "loop": 1, "execution_time": 2.1},
                    {"timestamp": "14:30:20", "level": "INFO", "message": "EVALUATE: Assessing task results", "loop": 1, "execution_time": 2.5},
                    {"timestamp": "14:30:21", "level": "INFO", "message": "REFLECT: Learning from results", "loop": 1, "execution_time": 3.0},
                ]
                
                st.markdown('<div class="reasoning-log">', unsafe_allow_html=True)
                for entry in sample_logs:
                    log_class = f"log-{entry['level'].lower()}"
                    st.markdown(f'''
                    <div class="log-entry {log_class}">
                        <strong>[{entry['timestamp']}]</strong> {entry['level']} - {entry['message']}
                        <br><small>Loop: {entry['loop']} | Time: {entry['execution_time']:.1f}s</small>
                    </div>
                    ''', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

    # Display results if available
    if 'agent_result' in st.session_state:
        result = st.session_state.agent_result
        
        if result.get('success'):
            st.success("✅ Mission completed successfully!")
            
            # Display final report
            if 'final_report' in result:
                report = result['final_report']
                
                # Investment decision - Clean layout
                if 'investment_decision' in report:
                    decision = report['investment_decision']
                    
                    # Decision metrics in a clean row
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Decision", decision.get('action', 'UNKNOWN'))
                    with col2:
                        st.metric("Confidence", f"{decision.get('confidence', 0):.2f}")
                    with col3:
                        st.metric("Risk", decision.get('risk_level', 'UNKNOWN'))
                    with col4:
                        st.metric("Symbol", report.get('symbol', 'N/A'))
                    
                    # Reasoning in expandable section
                    with st.expander("💭 MISSION ANALYSIS", expanded=False):
                        if 'reasoning' in decision:
                            for reason in decision['reasoning']:
                                st.write(f"• {reason}")
                
                # LLM Explanation - Clean format
                if 'llm_explanation' in report:
                    explanation = report['llm_explanation']
                    
                    with st.expander("🤖 AI COMMANDER REPORT", expanded=True):
                        st.write("**Mission Summary:**")
                        st.write(explanation.get('executive_summary', 'No summary available'))
                        
                        if 'investment_recommendation' in explanation:
                            st.write("**Commander's Recommendation:**")
                            st.write(explanation['investment_recommendation'])
                        
                        if 'key_findings' in explanation and explanation['key_findings']:
                            st.write("**Key Discoveries:**")
                            for finding in explanation['key_findings']:
                                st.write(f"• {finding}")
                
                # Model performance - Compact display
                if 'model_performance' in report and report['model_performance']:
                    with st.expander("📊 AI BRAIN PERFORMANCE", expanded=False):
                        perf = report['model_performance']
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("MAE", f"{perf.get('mae', 0):.4f}")
                        with col2:
                            st.metric("R²", f"{perf.get('r2', 0):.4f}")
                        with col3:
                            st.metric("Confidence", f"{perf.get('confidence', 0):.2f}")
                        with col4:
                            st.metric("Rel. Performance", f"{perf.get('rel_performance', 0):.4f}")
                
                # Predictions chart - Clean and minimal
                if 'predictions' in report and report['predictions']['values']:
                    with st.expander("🔮 FUTURE PREDICTIONS", expanded=False):
                        predictions = report['predictions']['values']
                        horizon = report['predictions']['horizon']
                        
                        # Create prediction chart
                        dates = pd.date_range(start=datetime.now(), periods=len(predictions), freq='D')
                        pred_df = pd.DataFrame({
                            'Date': dates,
                            'Predicted_Return': predictions
                        })
                        
                        fig = px.line(pred_df, x='Date', y='Predicted_Return', 
                                    title=f'Future Predictions (Next {horizon} Days)')
                        fig.update_layout(
                            height=300,
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='#ffffff'),
                            xaxis=dict(color='#ffffff'),
                            yaxis=dict(color='#ffffff')
                        )
                        st.plotly_chart(fig, use_container_width=True)
        else:
            st.error(f"❌ Mission failed: {result.get('error', 'Unknown error')}")
    
    # Download options - Minimal footer
    if 'agent' in st.session_state:
        st.markdown("---")
        st.subheader("💾 MISSION DATA")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if 'agent_result' in st.session_state:
                report_data = st.session_state.agent_result.get('final_report', {})
                json_str = json.dumps(report_data, indent=2, default=str)
                st.download_button(
                    label="📄 MISSION REPORT",
                    data=json_str,
                    file_name=f"space_finance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )
        
        with col2:
            agent = st.session_state.agent
            logs = agent.get_structured_logs()
            json_str = json.dumps(logs, indent=2, default=str)
            st.download_button(
                label="📋 MISSION LOGS",
                data=json_str,
                file_name=f"space_finance_logs_{agent.run_id[:8]}.json",
                mime="application/json",
                use_container_width=True
            )
        
        with col3:
            agent = st.session_state.agent
            status = agent.get_agent_status()
            json_str = json.dumps(status, indent=2, default=str)
            st.download_button(
                label="📊 MISSION STATUS",
                data=json_str,
                file_name=f"space_finance_status_{agent.run_id[:8]}.json",
                mime="application/json",
                use_container_width=True
            )

if __name__ == "__main__":
    main()
