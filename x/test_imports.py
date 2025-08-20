#!/usr/bin/env python3
"""
Import Test Script
Bu script import'ların çalışıp çalışmadığını test eder.
"""

import sys
import os
from pathlib import Path

def main():
    print("🧪 Import Test Script")
    print("=" * 25)
    
    # Add current directory to Python path
    current_dir = Path(__file__).parent
    sys.path.insert(0, str(current_dir))
    sys.path.insert(0, str(current_dir / "app"))
    
    print(f"📍 Working directory: {current_dir}")
    print(f"🐍 Python version: {sys.version}")
    print(f"📦 Python path: {sys.path[:3]}...")
    print()
    
    # Test SQLAlchemy imports
    print("🔍 Testing SQLAlchemy imports:")
    try:
        from sqlalchemy import select, and_, or_, func
        print("  ✅ sqlalchemy core: select, and_, or_, func")
    except Exception as e:
        print(f"  ❌ sqlalchemy core: {e}")
    
    try:
        from sqlalchemy.ext.asyncio import AsyncSession
        print("  ✅ sqlalchemy.ext.asyncio: AsyncSession")
    except Exception as e:
        print(f"  ❌ sqlalchemy.ext.asyncio: {e}")
    
    try:
        from sqlalchemy.orm import joinedload
        print("  ✅ sqlalchemy.orm: joinedload")
    except Exception as e:
        print(f"  ❌ sqlalchemy.orm: {e}")
    
    # Test other problematic imports
    print("\n🔍 Testing other imports:")
    imports_to_test = [
        ("redis.asyncio", "import redis.asyncio"),
        ("numpy", "import numpy"),
        ("boto3", "import boto3"),
        ("scipy", "import scipy"),
        ("PyPDF2", "import PyPDF2"),
    ]
    
    for name, import_statement in imports_to_test:
        try:
            exec(import_statement)
            print(f"  ✅ {name}")
        except Exception as e:
            print(f"  ❌ {name}: {e}")
    
    # Test app imports
    print("\n🔍 Testing app imports:")
    try:
        from app.core.cache import cache_service
        print("  ✅ app.core.cache")
    except Exception as e:
        print(f"  ❌ app.core.cache: {e}")
    
    try:
        from app.services.math_selector import MathSelector
        print("  ✅ app.services.math_selector")
    except Exception as e:
        print(f"  ❌ app.services.math_selector: {e}")
    
    try:
        from app.main import app
        print("  ✅ app.main (FastAPI)")
    except Exception as e:
        print(f"  ❌ app.main: {e}")
    
    print("\n🎯 Test completed!")
    print("\nEğer terminal'de tüm import'lar ✅ ise, problem IDE'de.")
    print("VS Code'u yeniden başlatın ve Python interpreter'ını seçin.")

if __name__ == "__main__":
    main()
