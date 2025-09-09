# Finance Agent Makefile
# =====================
# This Makefile provides convenient commands for development, testing, and deployment

.PHONY: help install test run-cli run-web clean lint format check-deps

# Default target
help:
	@echo "🤖 Finance Agent - Available Commands:"
	@echo ""
	@echo "📦 Setup & Installation:"
	@echo "  install          Install dependencies"
	@echo "  install-dev      Install development dependencies"
	@echo "  check-deps       Check if all dependencies are installed"
	@echo ""
	@echo "🚀 Running the Application:"
	@echo "  run-cli          Run CLI demo with default settings"
	@echo "  run-web          Run Streamlit web interface"
	@echo "  run-test         Run with test symbol (PTT.BK)"
	@echo ""
	@echo "🧪 Testing:"
	@echo "  test             Run all tests"
	@echo "  test-tools       Test tools module"
	@echo "  test-memory      Test memory module"
	@echo "  test-evaluator   Test evaluator module"
	@echo "  test-planner     Test planner module"
	@echo "  test-agent       Test agent module"
	@echo ""
	@echo "🔧 Development:"
	@echo "  lint             Run code linting"
	@echo "  format           Format code with black"
	@echo "  clean            Clean temporary files"
	@echo "  clean-memory     Clean agent memory storage"
	@echo ""
	@echo "📊 Examples:"
	@echo "  example-basic    Run basic analysis example"
	@echo "  example-advanced Run advanced analysis example"
	@echo "  example-thai     Run Thai stock analysis example"
	@echo ""
	@echo "🌐 Deployment:"
	@echo "  docker-build     Build Docker image"
	@echo "  docker-run       Run in Docker container"
	@echo "  deploy-check     Check deployment readiness"

# Variables
PYTHON := python
PIP := pip
STREAMLIT := streamlit
DOCKER := docker

# Installation targets
install:
	@echo "📦 Installing Finance Agent dependencies..."
	$(PIP) install -r requirements.txt
	@echo "✅ Installation completed!"

install-dev: install
	@echo "📦 Installing development dependencies..."
	$(PIP) install black flake8 pytest pytest-cov
	@echo "✅ Development dependencies installed!"

check-deps:
	@echo "🔍 Checking dependencies..."
	$(PYTHON) -c "import streamlit, pandas, numpy, sklearn, yfinance, requests, plotly, ta; print('✅ All dependencies are installed!')" || echo "❌ Some dependencies are missing. Run 'make install' to install them."

# Running targets
run-cli:
	@echo "🚀 Running Finance Agent CLI..."
	$(PYTHON) app_demo.py --symbol PTT.BK --horizon 5 --period 2y

run-web:
	@echo "🌐 Starting Streamlit web interface..."
	@echo "   Open your browser to: http://localhost:8501"
	$(STREAMLIT) run app_demo.py

run-test:
	@echo "🧪 Running test analysis..."
	$(PYTHON) app_demo.py --symbol PTT.BK --horizon 5 --period 1y --plan-type basic_analysis

# Testing targets
test: test-tools test-memory test-evaluator test-planner test-agent
	@echo "✅ All tests completed!"

test-tools:
	@echo "🧪 Testing tools module..."
	$(PYTHON) -m agent.tools

test-memory:
	@echo "🧪 Testing memory module..."
	$(PYTHON) -m agent.memory

test-evaluator:
	@echo "🧪 Testing evaluator module..."
	$(PYTHON) -m agent.evaluator

test-planner:
	@echo "🧪 Testing planner module..."
	$(PYTHON) -m agent.planner

test-agent:
	@echo "🧪 Testing agent module..."
	$(PYTHON) -m agent.agent

# Development targets
lint:
	@echo "🔍 Running code linting..."
	flake8 agent/ app_demo.py --max-line-length=100 --ignore=E203,W503

format:
	@echo "🎨 Formatting code..."
	black agent/ app_demo.py --line-length=100

clean:
	@echo "🧹 Cleaning temporary files..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type f -name "*.log" -delete
	find . -type f -name ".DS_Store" -delete
	@echo "✅ Cleanup completed!"

clean-memory:
	@echo "🧹 Cleaning agent memory storage..."
	rm -rf agent_storage/
	rm -rf memory_storage/
	@echo "✅ Memory storage cleaned!"

# Example targets
example-basic:
	@echo "📊 Running basic analysis example..."
	$(PYTHON) app_demo.py --symbol PTT.BK --horizon 5 --period 1y --plan-type basic_analysis

example-advanced:
	@echo "📊 Running advanced analysis example..."
	$(PYTHON) app_demo.py --symbol PTT.BK --horizon 10 --period 2y --plan-type comprehensive_analysis

example-thai:
	@echo "📊 Running Thai stock analysis example..."
	$(PYTHON) app_demo.py --symbol DELTA.BK --horizon 5 --period 2y --plan-type comprehensive_analysis

example-optimization:
	@echo "📊 Running model optimization example..."
	$(PYTHON) app_demo.py --symbol PTT.BK --horizon 5 --period 2y --plan-type model_optimization

# Docker targets
docker-build:
	@echo "🐳 Building Docker image..."
	$(DOCKER) build -t finance-agent .

docker-run:
	@echo "🐳 Running in Docker container..."
	$(DOCKER) run -p 8501:8501 -e OPENROUTER_API_KEY=$(OPENROUTER_API_KEY) finance-agent

# Deployment targets
deploy-check:
	@echo "🔍 Checking deployment readiness..."
	@echo "Checking Python version..."
	$(PYTHON) --version
	@echo "Checking dependencies..."
	$(PYTHON) -c "import streamlit, pandas, numpy, sklearn, yfinance, requests, plotly, ta; print('✅ All dependencies available')"
	@echo "Checking API keys..."
	@if [ -z "$$OPENROUTER_API_KEY" ] && [ -z "$$OPENAI_API_KEY" ]; then \
		echo "⚠️  Warning: No API keys found. Set OPENROUTER_API_KEY or OPENAI_API_KEY"; \
	else \
		echo "✅ API keys configured"; \
	fi
	@echo "✅ Deployment check completed!"

# Development workflow
dev-setup: install-dev
	@echo "🔧 Setting up development environment..."
	@echo "Creating .env file template..."
	@echo "# Finance Agent Environment Variables" > .env.template
	@echo "OPENROUTER_API_KEY=sk-or-..." >> .env.template
	@echo "OPENROUTER_SITE_URL=https://github.com/patcharaph/finance-agent.git" >> .env.template
	@echo "OPENROUTER_APP_NAME=Finance Agent Demo" >> .env.template
	@echo "OPENROUTER_MODEL=openrouter/auto" >> .env.template
	@echo "✅ Development environment ready!"
	@echo "📝 Copy .env.template to .env and add your API keys"

# Quick start
quick-start: check-deps
	@echo "🚀 Quick start - running basic analysis..."
	@if [ -z "$$OPENROUTER_API_KEY" ] && [ -z "$$OPENAI_API_KEY" ]; then \
		echo "⚠️  No API keys found. Running without LLM features..."; \
		$(PYTHON) app_demo.py --symbol PTT.BK --horizon 5 --period 1y; \
	else \
		echo "✅ API keys found. Running with full features..."; \
		$(PYTHON) app_demo.py --symbol PTT.BK --horizon 5 --period 2y --plan-type comprehensive_analysis; \
	fi

# Performance testing
perf-test:
	@echo "⚡ Running performance tests..."
	@echo "Testing with different symbols and horizons..."
	$(PYTHON) app_demo.py --symbol PTT.BK --horizon 5 --period 1y --max-loops 2
	$(PYTHON) app_demo.py --symbol DELTA.BK --horizon 10 --period 1y --max-loops 2
	$(PYTHON) app_demo.py --symbol ^SETI --horizon 5 --period 1y --max-loops 2
	@echo "✅ Performance tests completed!"

# Documentation
docs:
	@echo "📚 Generating documentation..."
	@echo "README.md is the main documentation file."
	@echo "For API documentation, see the docstrings in each module."
	@echo "Run 'make help' to see all available commands."

# All-in-one setup
setup: install check-deps
	@echo "🎉 Finance Agent setup completed!"
	@echo ""
	@echo "Next steps:"
	@echo "1. Set your API keys:"
	@echo "   export OPENROUTER_API_KEY='sk-or-...'"
	@echo "2. Run the web interface:"
	@echo "   make run-web"
	@echo "3. Or run CLI:"
	@echo "   make run-cli"
	@echo ""
	@echo "For more commands, run: make help"
