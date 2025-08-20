# Prompt Templates Module
# Jinja2 şablonları ile prompt yönetimi

import os
from pathlib import Path

# Template dosyalarının yolu
TEMPLATES_DIR = Path(__file__).parent

def get_template_path(template_name: str) -> Path:
    """Template dosyasının yolunu döndür"""
    return TEMPLATES_DIR / f"{template_name}.jinja2"

def load_template(template_name: str) -> str:
    """Template dosyasını yükle"""
    template_path = get_template_path(template_name)
    if template_path.exists():
        return template_path.read_text(encoding='utf-8')
    else:
        raise FileNotFoundError(f"Template not found: {template_path}")
