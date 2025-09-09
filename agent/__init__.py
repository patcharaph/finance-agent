"""
Finance Analyst Agent - Modular LLM Agentic AI
==============================================

A sophisticated finance analysis agent with planning, reflection, memory, and tool-use capabilities.

Components:
- tools: Data loading, indicators, sentiment analysis
- memory: Short-term and long-term memory systems
- evaluator: Metrics and rule-based evaluation
- planner: Task decomposition and plan generation
- agent: Main agent loop (think-act-eval-reflect)
"""

__version__ = "1.0.0"
__author__ = "Finance Agent Team"

from .agent import FinanceAgent
from .tools import DataLoader, IndicatorCalculator, SentimentAnalyzer
from .memory import ShortTermMemory, LongTermMemory
from .evaluator import Evaluator
from .planner import Planner

__all__ = [
    "FinanceAgent",
    "DataLoader", 
    "IndicatorCalculator",
    "SentimentAnalyzer",
    "ShortTermMemory",
    "LongTermMemory", 
    "Evaluator",
    "Planner"
]
