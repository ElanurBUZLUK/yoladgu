#!/usr/bin/env python3
"""
Run script for the Question Recommendation System Backend
"""

import os
import sys
import uvicorn
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def main():
    """Main function to start the server"""
    try:
        print("🚀 Starting Question Recommendation System Backend...")
        print(f"📁 Project root: {project_root}")
        print(f"🐍 Python path: {sys.path[0]}")

        # Load environment variables
        from dotenv import load_dotenv
        env_path = Path(__file__).parent / ".env"
        load_dotenv(env_path)

        print(f"📋 Environment loaded from: {env_path}")

        # Start the server (updated path for new structure)
        uvicorn.run(
            "app.main:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info",
            reload_dirs=["app"]
        )
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure you're in the correct directory and dependencies are installed:")
        print("   cd backend")
        print("   pip install -r requirements.txt")
        print("   python run.py")
    except Exception as e:
        print(f"❌ Error starting server: {e}")

if __name__ == "__main__":
    main()
