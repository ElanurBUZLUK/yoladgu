"""
Setup script for the adaptive learning backend.
This ensures proper Python path configuration.
"""

import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Add app directory to Python path
app_path = project_root / "app"
sys.path.insert(0, str(app_path))

# Set environment variables
os.environ.setdefault("PYTHONPATH", f"{project_root}:{app_path}")

if __name__ == "__main__":
    print(f"Python path configured:")
    print(f"Project root: {project_root}")
    print(f"App path: {app_path}")
    print(f"Python path: {sys.path[:3]}")
