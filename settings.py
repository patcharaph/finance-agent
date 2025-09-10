"""
Settings and Configuration for Finance Agent
============================================

This module contains all configuration settings, presets, and constants
used throughout the finance agent system.
"""

from typing import Dict, Any, List
from dataclasses import dataclass
from enum import Enum


class EvaluationPreset(Enum):
    """Available evaluation presets"""
    FORECASTING = "forecasting_preset"
    STRATEGY = "strategy_preset"
    CUSTOM = "custom_preset"


@dataclass
class ForecastingPreset:
    """Preset for forecasting evaluation metrics"""
    name: str = "Forecasting Preset"
    description: str = "Metrics optimized for time series forecasting tasks"
    
    # Primary metrics for forecasting
    primary_metrics: List[str] = None
    thresholds: Dict[str, float] = None
    weights: Dict[str, float] = None
    
    def __post_init__(self):
        if self.primary_metrics is None:
            self.primary_metrics = [
                "mae",           # Mean Absolute Error
                "r2",            # R-squared
                "directional_accuracy",  # Directional Accuracy
                "rel_performance"        # Relative to naive baseline
            ]
        
        if self.thresholds is None:
            self.thresholds = {
                "mae_max": 0.05,                    # Max 5% MAE
                "r2_min": 0.15,                     # Min 15% R²
                "directional_accuracy_min": 0.55,   # Min 55% directional accuracy
                "rel_performance_max": 0.95,        # Max 95% of naive baseline
                "confidence_min": 0.65              # Min 65% confidence
            }
        
        if self.weights is None:
            self.weights = {
                "mae": 0.3,
                "r2": 0.25,
                "directional_accuracy": 0.25,
                "rel_performance": 0.2
            }


@dataclass
class StrategyPreset:
    """Preset for trading strategy evaluation metrics"""
    name: str = "Strategy Preset"
    description: str = "Metrics optimized for trading strategy performance"
    
    # Primary metrics for strategy
    primary_metrics: List[str] = None
    thresholds: Dict[str, float] = None
    weights: Dict[str, float] = None
    
    def __post_init__(self):
        if self.primary_metrics is None:
            self.primary_metrics = [
                "sharpe_ratio",     # Risk-adjusted returns
                "max_drawdown",     # Maximum drawdown
                "win_rate",         # Win rate
                "profit_factor"     # Profit factor
            ]
        
        if self.thresholds is None:
            self.thresholds = {
                "sharpe_ratio_min": 0.5,            # Min Sharpe ratio 0.5
                "max_drawdown_max": 0.15,           # Max 15% drawdown
                "win_rate_min": 0.45,               # Min 45% win rate
                "profit_factor_min": 1.2,           # Min 1.2 profit factor
                "confidence_min": 0.7               # Min 70% confidence
            }
        
        if self.weights is None:
            self.weights = {
                "sharpe_ratio": 0.3,
                "max_drawdown": 0.3,
                "win_rate": 0.2,
                "profit_factor": 0.2
            }


@dataclass
class AgentSettings:
    """Main agent configuration settings"""
    
    # Execution settings
    max_loops: int = 5
    max_execution_time: float = 600.0  # 10 minutes
    confidence_threshold: float = 0.7
    
    # Learning settings
    enable_learning: bool = True
    enable_llm_planning: bool = True
    enable_reflection: bool = True
    
    # Logging settings
    log_level: str = "INFO"
    enable_structured_logging: bool = True
    log_file: str = "agent_logs.json"
    
    # Storage settings
    storage_path: str = "agent_storage"
    memory_retention_days: int = 30
    
    # Model settings
    default_model_type: str = "random_forest"
    lstm_sequence_length: int = 10
    lstm_units: int = 50
    lstm_epochs: int = 50
    
    # Evaluation settings
    default_evaluation_preset: EvaluationPreset = EvaluationPreset.FORECASTING
    enable_risk_gates: bool = True
    risk_threshold: float = 0.2
    
    # Data settings
    default_period: str = "2y"
    default_horizon: int = 5
    min_data_points: int = 100


@dataclass
class LLMSettings:
    """LLM configuration settings"""
    
    # Model settings
    model_name: str = "gpt-3.5-turbo"
    max_tokens: int = 1000
    temperature: float = 0.7
    
    # API settings
    api_key: str = ""
    base_url: str = ""
    timeout: int = 30
    
    # Retry settings
    max_retries: int = 3
    retry_delay: float = 1.0


@dataclass
class StreamlitSettings:
    """Streamlit app configuration"""
    
    # UI settings
    page_title: str = "Finance Agent"
    page_icon: str = "📈"
    layout: str = "wide"
    
    # Display settings
    show_reasoning_log: bool = True
    show_confidence_scores: bool = True
    show_risk_metrics: bool = True
    
    # Chart settings
    default_chart_height: int = 400
    enable_interactive_charts: bool = True
    
    # Performance settings
    cache_ttl: int = 300  # 5 minutes


# Global settings instances
FORECASTING_PRESET = ForecastingPreset()
STRATEGY_PRESET = StrategyPreset()
AGENT_SETTINGS = AgentSettings()
LLM_SETTINGS = LLMSettings()
STREAMLIT_SETTINGS = StreamlitSettings()


# Preset registry
EVALUATION_PRESETS = {
    EvaluationPreset.FORECASTING: FORECASTING_PRESET,
    EvaluationPreset.STRATEGY: STRATEGY_PRESET
}


def get_evaluation_preset(preset_name: str) -> Dict[str, Any]:
    """
    Get evaluation preset configuration
    
    Args:
        preset_name: Name of the preset to retrieve
    
    Returns:
        Dictionary with preset configuration
    """
    try:
        preset_enum = EvaluationPreset(preset_name)
        preset = EVALUATION_PRESETS[preset_enum]
        
        return {
            "name": preset.name,
            "description": preset.description,
            "primary_metrics": preset.primary_metrics,
            "thresholds": preset.thresholds,
            "weights": preset.weights
        }
    except (ValueError, KeyError):
        # Return default forecasting preset
        return {
            "name": FORECASTING_PRESET.name,
            "description": FORECASTING_PRESET.description,
            "primary_metrics": FORECASTING_PRESET.primary_metrics,
            "thresholds": FORECASTING_PRESET.thresholds,
            "weights": FORECASTING_PRESET.weights
        }


def get_agent_config() -> Dict[str, Any]:
    """Get agent configuration as dictionary"""
    return {
        "max_loops": AGENT_SETTINGS.max_loops,
        "max_execution_time": AGENT_SETTINGS.max_execution_time,
        "confidence_threshold": AGENT_SETTINGS.confidence_threshold,
        "enable_learning": AGENT_SETTINGS.enable_learning,
        "enable_llm_planning": AGENT_SETTINGS.enable_llm_planning,
        "enable_reflection": AGENT_SETTINGS.enable_reflection,
        "log_level": AGENT_SETTINGS.log_level,
        "enable_structured_logging": AGENT_SETTINGS.enable_structured_logging,
        "storage_path": AGENT_SETTINGS.storage_path,
        "default_model_type": AGENT_SETTINGS.default_model_type,
        "default_evaluation_preset": AGENT_SETTINGS.default_evaluation_preset.value,
        "enable_risk_gates": AGENT_SETTINGS.enable_risk_gates
    }


def get_llm_config() -> Dict[str, Any]:
    """Get LLM configuration as dictionary"""
    return {
        "model_name": LLM_SETTINGS.model_name,
        "max_tokens": LLM_SETTINGS.max_tokens,
        "temperature": LLM_SETTINGS.temperature,
        "api_key": LLM_SETTINGS.api_key,
        "base_url": LLM_SETTINGS.base_url,
        "timeout": LLM_SETTINGS.timeout,
        "max_retries": LLM_SETTINGS.max_retries,
        "retry_delay": LLM_SETTINGS.retry_delay
    }


def get_streamlit_config() -> Dict[str, Any]:
    """Get Streamlit configuration as dictionary"""
    return {
        "page_title": STREAMLIT_SETTINGS.page_title,
        "page_icon": STREAMLIT_SETTINGS.page_icon,
        "layout": STREAMLIT_SETTINGS.layout,
        "show_reasoning_log": STREAMLIT_SETTINGS.show_reasoning_log,
        "show_confidence_scores": STREAMLIT_SETTINGS.show_confidence_scores,
        "show_risk_metrics": STREAMLIT_SETTINGS.show_risk_metrics,
        "default_chart_height": STREAMLIT_SETTINGS.default_chart_height,
        "enable_interactive_charts": STREAMLIT_SETTINGS.enable_interactive_charts,
        "cache_ttl": STREAMLIT_SETTINGS.cache_ttl
    }


# Investment goal templates
INVESTMENT_GOAL_TEMPLATES = {
    "SET_INDEX": {
        "name": "SET Index Analysis",
        "description": "Analyze SET Index for market-wide investment decision",
        "symbol": "^SETI",
        "analysis_type": "market_analysis",
        "preset": "strategy_preset"
    },
    "INDIVIDUAL_STOCK": {
        "name": "Individual Stock Analysis",
        "description": "Analyze individual stock for investment decision",
        "symbol": "PTT.BK",
        "analysis_type": "stock_analysis",
        "preset": "forecasting_preset"
    },
    "PORTFOLIO_OPTIMIZATION": {
        "name": "Portfolio Optimization",
        "description": "Optimize portfolio allocation across multiple assets",
        "symbols": ["PTT.BK", "KBANK.BK", "SCB.BK"],
        "analysis_type": "portfolio_analysis",
        "preset": "strategy_preset"
    }
}


def get_investment_goal_template(template_name: str) -> Dict[str, Any]:
    """
    Get investment goal template
    
    Args:
        template_name: Name of the template
    
    Returns:
        Template configuration
    """
    return INVESTMENT_GOAL_TEMPLATES.get(template_name, INVESTMENT_GOAL_TEMPLATES["INDIVIDUAL_STOCK"])


# Risk assessment thresholds
RISK_THRESHOLDS = {
    "LOW": {
        "volatility_max": 0.15,
        "max_drawdown_max": 0.05,
        "sharpe_min": 1.0
    },
    "MEDIUM": {
        "volatility_max": 0.25,
        "max_drawdown_max": 0.15,
        "sharpe_min": 0.5
    },
    "HIGH": {
        "volatility_max": 0.4,
        "max_drawdown_max": 0.3,
        "sharpe_min": 0.2
    }
}


def get_risk_level(volatility: float, max_drawdown: float, sharpe: float) -> str:
    """
    Determine risk level based on metrics
    
    Args:
        volatility: Volatility metric
        max_drawdown: Maximum drawdown
        sharpe: Sharpe ratio
    
    Returns:
        Risk level string
    """
    for level, thresholds in RISK_THRESHOLDS.items():
        if (volatility <= thresholds["volatility_max"] and
            abs(max_drawdown) <= thresholds["max_drawdown_max"] and
            sharpe >= thresholds["sharpe_min"]):
            return level
    
    return "HIGH"  # Default to high risk if no criteria met


# Example usage
if __name__ == "__main__":
    print("Finance Agent Settings")
    print("=====================")
    
    # Test preset retrieval
    forecasting_config = get_evaluation_preset("forecasting_preset")
    print(f"Forecasting Preset: {forecasting_config['name']}")
    print(f"Primary Metrics: {forecasting_config['primary_metrics']}")
    
    # Test agent config
    agent_config = get_agent_config()
    print(f"Agent Max Loops: {agent_config['max_loops']}")
    
    # Test investment templates
    stock_template = get_investment_goal_template("INDIVIDUAL_STOCK")
    print(f"Stock Template: {stock_template['name']}")
    
    # Test risk assessment
    risk_level = get_risk_level(0.1, -0.03, 1.2)
    print(f"Risk Level: {risk_level}")
