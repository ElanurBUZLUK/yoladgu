# Yoladgu - AI-Powered Question Recommendation System
# Development and deployment automation (mevcut app/ yapısı için)

.PHONY: help install install-dev setup-dev clean test lint format check run

# === HELP ===
help: ## Show this help message
	@echo "🚀 Yoladgu Development Commands"
	@echo ""
	@awk 'BEGIN {FS = ":.*##"; printf "Usage: make <target>\n\nTargets:\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  %-15s %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

# === SETUP AND INSTALLATION ===
install: ## Install production dependencies
	cd backend && pip install -r requirements.txt

install-dev: ## Install development dependencies
	pip install black isort ruff pre-commit mypy pytest

setup-dev: install-dev ## Setup development environment with pre-commit
	pre-commit install
	@echo "✅ Development environment setup complete!"

# === CODE QUALITY ===
lint: ## Run linting with ruff
	ruff check app/ --fix

format: ## Format code with black and isort
	black app/ --line-length 100
	isort app/ --profile black --line-length 100

check: ## Run all code quality checks
	ruff check app/
	black app/ --check --line-length 100
	isort app/ --check-only --profile black

pre-commit: ## Run pre-commit hooks on all files
	pre-commit run --all-files

# === TESTING ===
test: ## Run tests
	cd backend && python -m pytest

test-cov: ## Run tests with coverage
	cd backend && python -m pytest --cov=app --cov-report=html --cov-report=term-missing

# === DEVELOPMENT ===
run: ## Start development server
	cd backend && python run.py

run-prod: ## Start production server
	uvicorn app.main:app --host 0.0.0.0 --port 8000

debug: ## Start server in debug mode
	uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload --log-level debug

# === MONITORING ===
health: ## Check application health
	curl -s http://localhost:8000/health | jq . || curl -s http://localhost:8000/health

performance: ## Run performance benchmark
	curl -s http://localhost:8000/api/v1/performance/benchmark | jq . || curl -s http://localhost:8000/api/v1/performance/benchmark

dashboard: ## View performance dashboard
	curl -s http://localhost:8000/api/v1/performance/dashboard | jq . || curl -s http://localhost:8000/api/v1/performance/dashboard

# === UTILITIES ===
clean: ## Clean temporary files and caches
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type f -name ".coverage" -delete
	find . -type d -name "htmlcov" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +

# === AI/ML SPECIFIC ===
update-embeddings: ## Update all embeddings
	curl -X POST http://localhost:8000/api/v1/embeddings/vector/batch-store

sync-vector-store: ## Sync vector store with database
	curl -X POST http://localhost:8000/api/v1/scheduler/trigger/daily-sync

reset-cache: ## Clear all caches
	curl -X POST http://localhost:8000/api/v1/embeddings/cache/clear

# === PROJECT INFO ===
info: ## Show project information
	@echo "🎓 Yoladgu - AI-Powered Question Recommendation System"
	@echo "📍 Main code: app/"
	@echo "📍 Backend utils: backend/"
	@echo "🐍 Python version: $(shell python --version)"
	@echo "🌐 FastAPI docs: http://localhost:8000/docs"
	@echo "📊 Performance: http://localhost:8000/api/v1/performance/dashboard"
	@echo "📈 Metrics: http://localhost:8000/metrics"

# === QUICK FORMATTING ===
format-quick: ## Quick format current changes only
	@echo "🎨 Formatting modified files..."
	@git diff --name-only --cached | grep '\.py$$' | xargs -r black --line-length 100
	@git diff --name-only --cached | grep '\.py$$' | xargs -r isort --profile black
	@echo "✅ Formatting complete!"

fix-all: format lint ## Run formatting and linting together
