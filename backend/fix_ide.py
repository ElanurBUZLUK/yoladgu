#!/usr/bin/env python3
"""
IDE Import Fix Script
Bu script IDE'nin import hatalarını çözmek için gerekli ayarları yapar.
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    print("🔧 IDE Import Fix Script")
    print("=" * 50)
    
    # Project root
    project_root = Path(__file__).parent
    print(f"📁 Project root: {project_root}")
    
    # Check virtual environment
    venv_path = project_root / "venv"
    if not venv_path.exists():
        print("❌ Virtual environment bulunamadı!")
        return
    
    print(f"✅ Virtual environment: {venv_path}")
    
    # Check Python path
    python_path = venv_path / "bin" / "python"
    if not python_path.exists():
        print("❌ Python interpreter bulunamadı!")
        return
    
    print(f"✅ Python interpreter: {python_path}")
    
    # Test imports
    print("\n🧪 Import testleri:")
    
    test_imports = [
        "sqlalchemy",
        "sqlalchemy.ext.asyncio",
        "sqlalchemy.orm", 
        "redis.asyncio",
        "numpy",
        "boto3",
        "scipy",
        "PyPDF2"
    ]
    
    for module in test_imports:
        try:
            result = subprocess.run([
                str(python_path), "-c", f"import {module}; print('OK')"
            ], capture_output=True, text=True, cwd=project_root)
            
            if result.returncode == 0:
                print(f"  ✅ {module}")
            else:
                print(f"  ❌ {module}: {result.stderr.strip()}")
        except Exception as e:
            print(f"  ❌ {module}: {e}")
    
    # App imports test
    print("\n🧪 App import testleri:")
    try:
        result = subprocess.run([
            str(python_path), "-c", 
            "import sys; sys.path.append('.'); from app.core.cache import cache_service; print('OK')"
        ], capture_output=True, text=True, cwd=project_root)
        
        if result.returncode == 0:
            print("  ✅ app.core.cache")
        else:
            print(f"  ❌ app.core.cache: {result.stderr.strip()}")
    except Exception as e:
        print(f"  ❌ app.core.cache: {e}")
    
    print("\n📋 IDE Ayarları:")
    print("1. VS Code: Ctrl+Shift+P → 'Python: Select Interpreter' → './venv/bin/python'")
    print("2. VS Code: Ctrl+Shift+P → 'Developer: Reload Window'")
    print("3. PyCharm: File → Settings → Project → Python Interpreter → './venv/bin/python'")
    print("4. PyCharm: File → Invalidate Caches and Restart")
    
    print("\n📁 Oluşturulan dosyalar:")
    files = [
        ".vscode/settings.json",
        "pyrightconfig.json", 
        "app/core/cache.py",
        "typings/sqlalchemy-stubs/__init__.pyi",
        "IDE_SETUP.md"
    ]
    
    for file in files:
        file_path = project_root / file
        if file_path.exists():
            print(f"  ✅ {file}")
        else:
            print(f"  ❌ {file}")
    
    print("\n🎉 Script tamamlandı! IDE'nizi yeniden başlatın.")

if __name__ == "__main__":
    main()
