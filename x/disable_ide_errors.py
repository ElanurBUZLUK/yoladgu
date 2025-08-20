#!/usr/bin/env python3
"""
IDE Error Disabler Script
Bu script tüm IDE hatalarını tamamen kapatır.
"""

import json
import os
from pathlib import Path

def main():
    print("🔧 IDE Error Disabler")
    print("=" * 30)
    
    project_root = Path(__file__).parent
    
    # Create minimal VS Code settings
    vscode_dir = project_root / ".vscode"
    vscode_dir.mkdir(exist_ok=True)
    
    minimal_settings = {
        "python.defaultInterpreterPath": "./.venv/bin/python",
        "python.terminal.activateEnvironment": True,
        "python.languageServer": "None",
        "python.analysis.disabled": True,
        "python.analysis.typeCheckingMode": "off",
        "python.linting.enabled": False,
        "python.linting.pylintEnabled": False,
        "python.linting.flake8Enabled": False,
        "python.linting.mypyEnabled": False,
        "python.analysis.autoImportCompletions": False,
        "files.associations": {
            "*.py": "python"
        },
        "editor.quickSuggestions": {
            "other": False,
            "comments": False,
            "strings": False
        },
        "editor.parameterHints.enabled": False,
        "editor.suggest.showKeywords": False,
        "editor.suggest.showSnippets": False,
        "python.envFile": "${workspaceFolder}/.env"
    }
    
    settings_file = vscode_dir / "settings.json"
    with open(settings_file, 'w') as f:
        json.dump(minimal_settings, f, indent=2)
    print(f"✅ VS Code settings updated: {settings_file}")
    
    # Disable Pylance
    extensions_file = vscode_dir / "extensions.json"
    extensions_config = {
        "recommendations": ["ms-python.python"],
        "unwantedRecommendations": [
            "ms-python.pylance",
            "ms-python.mypy-type-checker",
            "ms-python.flake8",
            "ms-python.pylint"
        ]
    }
    
    with open(extensions_file, 'w') as f:
        json.dump(extensions_config, f, indent=2)
    print(f"✅ VS Code extensions config updated: {extensions_file}")
    
    # Create minimal pyrightconfig.json that disables everything
    pyright_config = {
        "typeCheckingMode": "off",
        "reportMissingImports": "none",
        "reportMissingTypeStubs": "none",
        "reportGeneralTypeIssues": "none",
        "useLibraryCodeForTypes": False,
        "autoImportCompletions": False,
        "include": [],
        "exclude": ["**/*"]
    }
    
    pyright_file = project_root / "pyrightconfig.json"
    with open(pyright_file, 'w') as f:
        json.dump(pyright_config, f, indent=2)
    print(f"✅ Pyright config updated: {pyright_file}")
    
    # Create .pylintrc that disables everything
    pylintrc_content = """[MAIN]
disable=all

[MESSAGES CONTROL]
disable=all

[REPORTS]
reports=no
"""
    
    pylintrc_file = project_root / ".pylintrc"
    with open(pylintrc_file, 'w') as f:
        f.write(pylintrc_content)
    print(f"✅ Pylint config updated: {pylintrc_file}")
    
    print("\n🎯 Tüm IDE hataları devre dışı bırakıldı!")
    print("\n📋 Şimdi yapmanız gerekenler:")
    print("1. VS Code'u tamamen kapatın")
    print("2. VS Code'u yeniden açın")
    print("3. Eğer Pylance extension'ı yüklüyse, devre dışı bırakın")
    print("4. Python interpreter'ı seçin: ./.venv/bin/python")
    print("5. Ctrl+Shift+P → 'Developer: Reload Window'")
    
    print("\n✨ Bu ayarlarla hiçbir import hatası görmeyeceksiniz!")

if __name__ == "__main__":
    main()

