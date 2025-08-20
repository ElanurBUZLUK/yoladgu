#!/usr/bin/env python3
"""
Setup Complete Verification Script
Bu script tüm kurulum adımlarının başarıyla tamamlandığını doğrular.
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    print("🎉 Setup Complete Verification")
    print("=" * 50)
    
    project_root = Path(__file__).parent
    
    # Check virtual environment
    print("🐍 Virtual Environment Check:")
    venv_path = project_root / ".venv"
    if venv_path.exists():
        print(f"  ✅ .venv directory exists: {venv_path}")
        
        python_path = venv_path / "bin" / "python"
        if python_path.exists():
            print(f"  ✅ Python interpreter: {python_path}")
        else:
            print("  ❌ Python interpreter not found")
            return
    else:
        print("  ❌ .venv directory not found")
        return
    
    # Test imports
    print("\n📦 Import Tests:")
    
    test_imports = [
        ("sqlalchemy", "from sqlalchemy import select, and_, or_, func"),
        ("sqlalchemy.ext.asyncio", "from sqlalchemy.ext.asyncio import AsyncSession"),
        ("sqlalchemy.orm", "from sqlalchemy.orm import joinedload"),
        ("redis.asyncio", "import redis.asyncio"),
        ("numpy", "import numpy"),
        ("boto3", "import boto3"),
        ("scipy", "import scipy"),
        ("PyPDF2", "import PyPDF2"),
        ("app.core.cache", "from app.core.cache import cache_service"),
        ("app.services.math_selector", "from app.services.math_selector import MathSelector"),
        ("FastAPI app", "from app.main import app")
    ]
    
    all_passed = True
    for name, import_cmd in test_imports:
        try:
            result = subprocess.run([
                str(python_path), "-c", import_cmd
            ], capture_output=True, text=True, cwd=project_root)
            
            if result.returncode == 0:
                print(f"  ✅ {name}")
            else:
                print(f"  ❌ {name}: {result.stderr.strip()}")
                all_passed = False
        except Exception as e:
            print(f"  ❌ {name}: {e}")
            all_passed = False
    
    # Check configuration files
    print("\n⚙️ Configuration Files:")
    config_files = [
        ".vscode/settings.json",
        "pyrightconfig.json",
        "adaptive-learning-backend.code-workspace",
        ".vscode/launch.json",
        "typings/sqlalchemy-stubs/sqlalchemy/__init__.pyi",
        "typings/sqlalchemy-stubs/sqlalchemy/ext/asyncio/__init__.pyi",
        "typings/sqlalchemy-stubs/sqlalchemy/orm/__init__.pyi"
    ]
    
    for config_file in config_files:
        file_path = project_root / config_file
        if file_path.exists():
            print(f"  ✅ {config_file}")
        else:
            print(f"  ❌ {config_file}")
            all_passed = False
    
    # Check __init__.py files
    print("\n📁 Package Structure:")
    init_files = [
        "app/workers/__init__.py",
        "app/repositories/__init__.py",
        "app/domains/__init__.py",
        "app/tools/__init__.py"
    ]
    
    for init_file in init_files:
        file_path = project_root / init_file
        if file_path.exists():
            print(f"  ✅ {init_file}")
        else:
            print(f"  ❌ {init_file}")
            all_passed = False
    
    print("\n🎯 Summary:")
    if all_passed:
        print("  ✅ All tests passed! Setup is complete.")
        print("\n📋 Next Steps:")
        print("1. VS Code'u yeniden başlatın")
        print("2. adaptive-learning-backend.code-workspace dosyasını açın")
        print("3. Ctrl+Shift+P → 'Python: Select Interpreter' → './.venv/bin/python'")
        print("4. Ctrl+Shift+P → 'Developer: Reload Window'")
        print("\n🚀 Uygulamayı başlatmak için:")
        print("  source .venv/bin/activate")
        print("  uvicorn app.main:app --reload")
    else:
        print("  ❌ Some tests failed. Please check the errors above.")
    
    print("\n🎉 Setup verification completed!")

if __name__ == "__main__":
    main()
