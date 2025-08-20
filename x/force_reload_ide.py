#!/usr/bin/env python3
"""
Force IDE Reload Script
Bu script IDE'nin tüm cache'lerini temizler ve yeniden başlatır.
"""

import os
import shutil
import subprocess
from pathlib import Path

def main():
    print("🔄 Force IDE Reload Script")
    print("=" * 50)
    
    project_root = Path(__file__).parent
    
    # Clean Python cache
    print("🧹 Python cache temizleniyor...")
    cache_dirs = [
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache"
    ]
    
    for cache_dir in cache_dirs:
        cache_path = project_root / cache_dir
        if cache_path.exists():
            shutil.rmtree(cache_path)
            print(f"  ✅ {cache_dir} silindi")
    
    # Clean VS Code cache
    print("🧹 VS Code cache temizleniyor...")
    vscode_cache = project_root / ".vscode" / "cache"
    if vscode_cache.exists():
        shutil.rmtree(vscode_cache)
        print("  ✅ VS Code cache silindi")
    
    # Clean PyCharm cache
    print("🧹 PyCharm cache temizleniyor...")
    pycharm_cache = project_root / ".idea" / "workspace.xml"
    if pycharm_cache.exists():
        pycharm_cache.unlink()
        print("  ✅ PyCharm workspace.xml silindi")
    
    # Verify Python interpreter
    print("🐍 Python interpreter kontrol ediliyor...")
    venv_python = project_root / "venv" / "bin" / "python"
    if venv_python.exists():
        print(f"  ✅ Python interpreter: {venv_python}")
        
        # Test imports
        print("🧪 Import testleri...")
        test_script = """
import sys
sys.path.insert(0, '.')
sys.path.insert(0, './app')

try:
    import sqlalchemy
    from sqlalchemy import select, and_, or_, func
    from sqlalchemy.ext.asyncio import AsyncSession
    from sqlalchemy.orm import joinedload
    print('✅ SQLAlchemy imports başarılı')
except ImportError as e:
    print(f'❌ SQLAlchemy import hatası: {e}')

try:
    import redis.asyncio
    print('✅ Redis imports başarılı')
except ImportError as e:
    print(f'❌ Redis import hatası: {e}')

try:
    from app.core.cache import cache_service
    print('✅ App imports başarılı')
except ImportError as e:
    print(f'❌ App import hatası: {e}')
"""
        
        result = subprocess.run([
            str(venv_python), "-c", test_script
        ], capture_output=True, text=True, cwd=project_root)
        
        print(result.stdout)
        if result.stderr:
            print("Hatalar:", result.stderr)
    else:
        print("❌ Python interpreter bulunamadı!")
    
    print("\n📋 IDE Yeniden Başlatma Talimatları:")
    print("1. VS Code'u tamamen kapatın")
    print("2. VS Code'u yeniden açın")
    print("3. adaptive-learning-backend.code-workspace dosyasını açın")
    print("4. Ctrl+Shift+P → 'Python: Select Interpreter' → './venv/bin/python'")
    print("5. Ctrl+Shift+P → 'Developer: Reload Window'")
    
    print("\n📋 PyCharm için:")
    print("1. PyCharm'ı tamamen kapatın")
    print("2. PyCharm'ı yeniden açın")
    print("3. File → Invalidate Caches and Restart")
    print("4. File → Settings → Project → Python Interpreter → './venv/bin/python'")
    
    print("\n🎉 Cache temizleme tamamlandı! IDE'nizi yeniden başlatın.")

if __name__ == "__main__":
    main()
