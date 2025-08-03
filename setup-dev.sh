#!/bin/bash

# 🚀 Yoladgu Development Environment Setup Script
# Mevcut app/ yapısı için code quality tools setup

set -e  # Exit on any error

echo "🎓 Setting up Yoladgu Development Environment..."
echo "=================================================="

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Python 3.9+ is available
print_status "Checking Python version..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    if python3 -c 'import sys; exit(0 if sys.version_info >= (3, 9) else 1)'; then
        print_success "Python $PYTHON_VERSION found"
    else
        print_error "Python 3.9+ required, found $PYTHON_VERSION"
        exit 1
    fi
else
    print_error "Python 3 not found. Please install Python 3.9+"
    exit 1
fi

# Check if we're in the right directory
if [ ! -f "app/main.py" ]; then
    print_error "Please run this script from the project root directory (where app/ exists)"
    exit 1
fi

# Install code quality tools
print_status "Installing code quality tools..."
pip install black isort ruff pre-commit mypy pytest

# Install backend dependencies if backend exists
if [ -d "backend" ]; then
    print_status "Installing backend dependencies..."
    cd backend
    if [ ! -d "venv" ]; then
        python3 -m venv venv
        print_success "Backend virtual environment created"
    fi
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    cd ..
    print_success "Backend dependencies installed"
fi

# Install pre-commit hooks
print_status "Setting up pre-commit hooks..."
if command -v pre-commit &> /dev/null; then
    pre-commit install
    print_success "Pre-commit hooks installed"
else
    print_warning "pre-commit not found, installing..."
    pip install pre-commit
    pre-commit install
    print_success "Pre-commit installed and configured"
fi

# Run initial pre-commit check
print_status "Running initial code quality checks..."
if pre-commit run --all-files; then
    print_success "Code quality checks passed"
else
    print_warning "Some code quality issues found and fixed"
fi

# Create useful directories
print_status "Creating project directories..."
mkdir -p logs
mkdir -p data
mkdir -p models
print_success "Project directories created"

# Display helpful information
echo ""
echo "🎉 Development Environment Setup Complete!"
echo "==========================================="
echo ""
echo "📂 Project Structure:"
echo "   app/              - Main FastAPI application"
echo "   backend/          - Backend utilities and configurations"
echo ""
echo "🛠️  Development Commands:"
echo "   make help         - Show all available commands"
echo "   make run          - Start development server"
echo "   make format       - Format code with black & isort"
echo "   make lint         - Run linting with ruff"
echo "   make check        - Run all quality checks"
echo "   make fix-all      - Format + lint together"
echo ""
echo "🌐 URLs (after starting server):"
echo "   API Docs:         http://localhost:8000/docs"
echo "   Health Check:     http://localhost:8000/health"
echo "   Performance:      http://localhost:8000/api/v1/performance/dashboard"
echo "   Metrics:          http://localhost:8000/metrics"
echo ""
echo "🚀 Quick Start:"
echo "   make run          # Start the application"
echo "   make format       # Format your code"
echo ""
echo "💡 Tips:"
echo "   - Use 'make help' to see all available commands"
echo "   - Pre-commit hooks will run automatically on git commit"
echo "   - Run 'make format' before committing code"
echo "   - Use 'make fix-all' for quick formatting + linting"
echo ""
print_success "Ready to start developing! 🚀"
