#!/bin/bash

# PgVector Installation Script for Adaptive Learning Platform
# This script installs pgvector extension for PostgreSQL

set -e

echo "🔧 Installing pgvector extension..."

# Detect OS
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux
    if command -v apt-get &> /dev/null; then
        # Ubuntu/Debian
        echo "📦 Detected Ubuntu/Debian system"
        echo "Installing pgvector..."
        sudo apt-get update
        sudo apt-get install -y postgresql-13-pgvector
    elif command -v yum &> /dev/null; then
        # CentOS/RHEL
        echo "📦 Detected CentOS/RHEL system"
        echo "Installing pgvector..."
        sudo yum install -y pgvector_13
    else
        echo "❌ Unsupported Linux distribution"
        echo "Please install pgvector manually:"
        echo "  https://github.com/pgvector/pgvector#installation"
        exit 1
    fi
elif [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    echo "📦 Detected macOS system"
    if command -v brew &> /dev/null; then
        echo "Installing pgvector via Homebrew..."
        brew install pgvector
    else
        echo "❌ Homebrew not found"
        echo "Please install Homebrew first: https://brew.sh/"
        exit 1
    fi
else
    echo "❌ Unsupported operating system: $OSTYPE"
    echo "Please install pgvector manually:"
    echo "  https://github.com/pgvector/pgvector#installation"
    exit 1
fi

echo "✅ pgvector installation completed!"

# Verify installation
echo "🔍 Verifying installation..."
if command -v psql &> /dev/null; then
    echo "PostgreSQL client found"
else
    echo "⚠️  PostgreSQL client not found"
    echo "Please install PostgreSQL client tools"
fi

echo ""
echo "🎉 PgVector installation completed successfully!"
echo ""
echo "Next steps:"
echo "1. Restart PostgreSQL service: sudo systemctl restart postgresql"
echo "2. Run database migrations: alembic upgrade head"
echo "3. Run setup script: python scripts/setup_system.py"
echo ""
echo "For manual installation instructions, visit:"
echo "https://github.com/pgvector/pgvector#installation"
