#!/usr/bin/env python3
"""
LLM API Endpoint Test Scripti
Bu script LLM API endpoint'lerini test eder.
"""

import os
import sys
import asyncio
import requests
import json
from pathlib import Path

# Backend dizinini Python path'ine ekle
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))

# App dizinini de ekle
app_path = Path(__file__).parent / "app"
sys.path.insert(0, str(app_path))

BASE_URL = "http://localhost:8000/api/v1"

def test_llm_status():
    """LLM durum endpoint'ini test et"""
    print("🔍 LLM Status Endpoint Testi...")
    
    try:
        response = requests.get(f"{BASE_URL}/ai/llm-status")
        print(f"   Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"   OpenAI Configured: {data.get('openai_configured')}")
            print(f"   HuggingFace Configured: {data.get('huggingface_configured')}")
            print(f"   Status: {data.get('status')}")
            return data
        else:
            print(f"   ❌ Hata: {response.text}")
            return None
    except Exception as e:
        print(f"   ❌ Exception: {str(e)}")
        return None

def test_generate_hint():
    """İpucu üretme endpoint'ini test et"""
    print("\n🎯 Generate Hint Endpoint Testi...")
    
    payload = {
        "question": "2x + 5 = 13 denklemini çöz"
    }
    
    try:
        response = requests.post(f"{BASE_URL}/ai/generate-hint", json=payload)
        print(f"   Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"   İpucu: {data.get('hint', '')[:100]}...")
            return data
        else:
            print(f"   ❌ Hata: {response.text}")
            return None
    except Exception as e:
        print(f"   ❌ Exception: {str(e)}")
        return None

def test_generate_explanation():
    """Açıklama üretme endpoint'ini test et"""
    print("\n📚 Generate Explanation Endpoint Testi...")
    
    payload = {
        "question": "2x + 5 = 13 denklemini çöz",
        "answer": "x = 4"
    }
    
    try:
        response = requests.post(f"{BASE_URL}/ai/generate-explanation", json=payload)
        print(f"   Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"   Açıklama: {data.get('explanation', '')[:100]}...")
            return data
        else:
            print(f"   ❌ Hata: {response.text}")
            return None
    except Exception as e:
        print(f"   ❌ Exception: {str(e)}")
        return None

def test_analyze_difficulty():
    """Zorluk analizi endpoint'ini test et"""
    print("\n📊 Analyze Difficulty Endpoint Testi...")
    
    payload = {
        "question": "2x + 5 = 13 denklemini çöz"
    }
    
    try:
        response = requests.post(f"{BASE_URL}/ai/analyze-question-difficulty", json=payload)
        print(f"   Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"   Zorluk: {data.get('difficulty', '')}")
            return data
        else:
            print(f"   ❌ Hata: {response.text}")
            return None
    except Exception as e:
        print(f"   ❌ Exception: {str(e)}")
        return None

def test_legacy_endpoints():
    """Eski endpoint'leri test et"""
    print("\n🔄 Legacy Endpoints Testi...")
    
    # Eski ipucu endpoint'i
    payload = {"question": "2x + 5 = 13 denklemini çöz"}
    try:
        response = requests.post(f"{BASE_URL}/ai/hint", json=payload)
        print(f"   Legacy Hint Status: {response.status_code}")
    except Exception as e:
        print(f"   Legacy Hint Error: {str(e)}")
    
    # Eski açıklama endpoint'i
    payload = {"question": "2x + 5 = 13 denklemini çöz", "answer": "x = 4"}
    try:
        response = requests.post(f"{BASE_URL}/ai/explanation", json=payload)
        print(f"   Legacy Explanation Status: {response.status_code}")
    except Exception as e:
        print(f"   Legacy Explanation Error: {str(e)}")
    
    # Eski zorluk endpoint'i
    payload = {"question": "2x + 5 = 13 denklemini çöz"}
    try:
        response = requests.post(f"{BASE_URL}/ai/difficulty", json=payload)
        print(f"   Legacy Difficulty Status: {response.status_code}")
    except Exception as e:
        print(f"   Legacy Difficulty Error: {str(e)}")

def check_server_status():
    """Server durumunu kontrol et"""
    print("🌐 Server Durumu Kontrolü...")
    
    try:
        response = requests.get("http://localhost:8000/health")
        print(f"   Health Check Status: {response.status_code}")
        
        if response.status_code == 200:
            print("   ✅ Server çalışıyor!")
            return True
        else:
            print("   ❌ Server yanıt vermiyor")
            return False
    except Exception as e:
        print(f"   ❌ Server erişilemiyor: {str(e)}")
        return False

def main():
    """Ana fonksiyon"""
    print("=" * 60)
    print("🤖 LLM API Endpoint Test Aracı")
    print("=" * 60)
    
    # Server durumunu kontrol et
    if not check_server_status():
        print("\n💡 Öneriler:")
        print("   1. Backend server'ı başlatın: cd backend && python run.py")
        print("   2. Veya: ./start-backend.sh")
        print("   3. Server'ın 8000 portunda çalıştığından emin olun")
        return
    
    # LLM durumunu kontrol et
    status = test_llm_status()
    
    if status and status.get('status') == 'ready':
        print("\n✅ LLM servisi hazır, endpoint'ler test ediliyor...")
        
        # Endpoint'leri test et
        test_generate_hint()
        test_generate_explanation()
        test_analyze_difficulty()
        test_legacy_endpoints()
        
    else:
        print("\n❌ LLM servisi hazır değil!")
        print("💡 Öneriler:")
        print("   1. .env dosyasında API anahtarlarını kontrol edin")
        print("   2. OPENAI_API_KEY veya HUGGINGFACE_API_TOKEN ekleyin")
        print("   3. Server'ı yeniden başlatın")
    
    print("\n" + "=" * 60)
    print("✅ Endpoint testi tamamlandı!")
    print("=" * 60)

if __name__ == "__main__":
    main() 