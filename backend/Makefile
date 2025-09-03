# Adaptive Question Recommendation System - Backend Makefile

.PHONY: help install dev test lint format clean docker-build docker-up docker-down migrate

# Default target
help:
	@echo "Available commands:"
	@echo "  install     - Install dependencies"
	@echo "  dev         - Run development server"
	@echo "  test        - Run tests"
	@echo "  lint        - Run linting"
	@echo "  format      - Format code"
	@echo "  clean       - Clean cache and temp files"
	@echo "  docker-build - Build Docker image"
	@echo "  docker-up   - Start Docker services"
	@echo "  docker-down - Stop Docker services"
	@echo "  migrate     - Run database migrations"

# Install dependencies
install:
	pip install -r requirements.txt

# Run development server
dev:
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Run tests
test:
	pytest tests/ -v

# Run tests with coverage
test-cov:
	pytest tests/ -v --cov=app --cov-report=html --cov-report=term

# Run linting
lint:
	flake8 app/
	mypy app/

# Format code
format:
	black app/ tests/
	isort app/ tests/

# Clean cache and temp files
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name "htmlcov" -exec rm -rf {} +

# Docker commands
docker-build:
	docker build -t adaptive-questions-backend .

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f app

# Database migrations
migrate:
	alembic upgrade head

migrate-create:
	alembic revision --autogenerate -m "$(name)"

migrate-downgrade:
	alembic downgrade -1

# Development setup
setup-dev: install
	cp .env.example .env
	@echo "Please edit .env file with your configuration"

# Production deployment
deploy-prod:
	docker-compose -f docker-compose.prod.yml up -d

# Backup database
backup-db:
	docker-compose exec db pg_dump -U postgres adaptive_questions > backup_$(shell date +%Y%m%d_%H%M%S).sql

# Restore database
restore-db:
	docker-compose exec -T db psql -U postgres adaptive_questions < $(file)

# Load test data
load-test-data:
	python scripts/load_test_data.py

# Generate API documentation
docs:
	mkdocs build

# Serve documentation
serve-docs:
	mkdocs serve