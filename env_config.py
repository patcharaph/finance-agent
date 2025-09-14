#!/usr/bin/env python3
"""
Environment Configuration
========================

This file contains environment variables and configuration settings
for the Finance Agent application.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# OpenRouter API Configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "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")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openrouter/auto")
OPENROUTER_SITE_URL = os.getenv("OPENROUTER_SITE_URL", "https://github.com/patcharaph/finance-agent.git")
OPENROUTER_APP_NAME = os.getenv("OPENROUTER_APP_NAME", "Finance Agent Demo")

# OpenAI Configuration (Alternative)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

# Finance Agent Configuration
AGENT_MAX_LOOPS = int(os.getenv("AGENT_MAX_LOOPS", "10"))
AGENT_MAX_EXECUTION_TIME = int(os.getenv("AGENT_MAX_EXECUTION_TIME", "300"))
AGENT_CONFIDENCE_THRESHOLD = float(os.getenv("AGENT_CONFIDENCE_THRESHOLD", "0.7"))
AGENT_ENABLE_LLM_PLANNING = os.getenv("AGENT_ENABLE_LLM_PLANNING", "true").lower() == "true"
AGENT_ENABLE_LEARNING = os.getenv("AGENT_ENABLE_LEARNING", "true").lower() == "true"

# Data Sources
YFINANCE_TIMEOUT = int(os.getenv("YFINANCE_TIMEOUT", "30"))
YFINANCE_RETRIES = int(os.getenv("YFINANCE_RETRIES", "3"))

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = os.getenv("LOG_FILE", "agent_logs.json")
ENABLE_STRUCTURED_LOGGING = os.getenv("ENABLE_STRUCTURED_LOGGING", "true").lower() == "true"

# Storage
STORAGE_PATH = os.getenv("STORAGE_PATH", "agent_storage")
MEMORY_ENABLED = os.getenv("MEMORY_ENABLED", "true").lower() == "true"

# Streamlit Configuration
STREAMLIT_SERVER_PORT = int(os.getenv("STREAMLIT_SERVER_PORT", "8501"))
STREAMLIT_SERVER_ADDRESS = os.getenv("STREAMLIT_SERVER_ADDRESS", "localhost")
STREAMLIT_THEME_BASE = os.getenv("STREAMLIT_THEME_BASE", "light")

def print_env_status():
    """Print current environment configuration status"""
    print("🔧 Environment Configuration Status")
    print("=" * 50)
    
    # OpenRouter Configuration
    print("📡 OpenRouter API:")
    print(f"   API Key: {'✅ Set' if OPENROUTER_API_KEY else '❌ Not set'}")
    print(f"   Base URL: {OPENROUTER_BASE_URL}")
    print(f"   Model: {OPENROUTER_MODEL}")
    print(f"   Site URL: {OPENROUTER_SITE_URL}")
    print(f"   App Name: {OPENROUTER_APP_NAME}")
    
    # OpenAI Configuration
    print("\n🤖 OpenAI API:")
    print(f"   API Key: {'✅ Set' if OPENAI_API_KEY else '❌ Not set'}")
    print(f"   Base URL: {OPENAI_BASE_URL}")
    
    # Agent Configuration
    print("\n🤖 Agent Configuration:")
    print(f"   Max Loops: {AGENT_MAX_LOOPS}")
    print(f"   Max Execution Time: {AGENT_MAX_EXECUTION_TIME}s")
    print(f"   Confidence Threshold: {AGENT_CONFIDENCE_THRESHOLD}")
    print(f"   LLM Planning: {'✅ Enabled' if AGENT_ENABLE_LLM_PLANNING else '❌ Disabled'}")
    print(f"   Learning: {'✅ Enabled' if AGENT_ENABLE_LEARNING else '❌ Disabled'}")
    
    # Data Sources
    print("\n📊 Data Sources:")
    print(f"   YFinance Timeout: {YFINANCE_TIMEOUT}s")
    print(f"   YFinance Retries: {YFINANCE_RETRIES}")
    
    # Logging
    print("\n📝 Logging:")
    print(f"   Log Level: {LOG_LEVEL}")
    print(f"   Log File: {LOG_FILE}")
    print(f"   Structured Logging: {'✅ Enabled' if ENABLE_STRUCTURED_LOGGING else '❌ Disabled'}")
    
    # Storage
    print("\n💾 Storage:")
    print(f"   Storage Path: {STORAGE_PATH}")
    print(f"   Memory: {'✅ Enabled' if MEMORY_ENABLED else '❌ Disabled'}")
    
    # Streamlit
    print("\n🌐 Streamlit:")
    print(f"   Server Port: {STREAMLIT_SERVER_PORT}")
    print(f"   Server Address: {STREAMLIT_SERVER_ADDRESS}")
    print(f"   Theme: {STREAMLIT_THEME_BASE}")

if __name__ == "__main__":
    print_env_status()
