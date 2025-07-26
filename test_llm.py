#!/usr/bin/env python3
"""
LLM Servis Test Scripti
Bu script LLM servisinin çalışıp çalışmadığını test eder.
"""

import os
import sys
import asyncio
from pathlib import Path

# Backend dizinini Python path'ine ekle
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))

# App dizinini de ekle
app_path = Path(__file__).parent / "app"
sys.path.insert(0, str(app_path))

async def test_llm_service():
    """LLM servisini test et"""
    print("🔍 LLM Servis Testi Başlatılıyor...")
    
    try:
        from app.services.llm_service import LLMService, LLMProvider
        
        # Environment variables kontrolü
        print("\n📋 Environment Variables Kontrolü:")
        openai_key = os.getenv("OPENAI_API_KEY")
        hf_token = os.getenv("HUGGINGFACE_API_TOKEN")
        hf_model = os.getenv("HUGGINGFACE_MODEL", "gpt2")
        
        print(f"   OpenAI API Key: {'✅ Mevcut' if openai_key else '❌ Eksik'}")
        print(f"   HuggingFace Token: {'✅ Mevcut' if hf_token else '❌ Eksik'}")
        print(f"   HuggingFace Model: {hf_model}")
        
        # LLM Service instance oluştur
        print("\n🤖 LLM Service Instance Oluşturuluyor...")
        llm_service = LLMService()
        
        # API key kontrolü
        print("\n🔑 API Key Kontrolü:")
        openai_configured = bool(llm_service._get_api_key())
        print(f"   OpenAI Configured: {'✅' if openai_configured else '❌'}")
        
        # Test prompt
        test_prompt = "Merhaba, bu bir test mesajıdır. Lütfen 'Test başarılı' yanıtını ver."
        
        print(f"\n🧪 Test Prompt: {test_prompt}")
        
        if openai_configured:
            print("\n🚀 OpenAI ile test ediliyor...")
            try:
                result = await llm_service.generate_text(test_prompt)
                print(f"   Sonuç: {result}")
                if result.get("success"):
                    print("   ✅ OpenAI LLM çalışıyor!")
                else:
                    print(f"   ❌ OpenAI LLM hatası: {result.get('error')}")
            except Exception as e:
                print(f"   ❌ OpenAI LLM exception: {str(e)}")
        else:
            print("   ⚠️ OpenAI API key eksik, test atlanıyor")
        
        # HuggingFace test
        if hf_token:
            print("\n🚀 HuggingFace ile test ediliyor...")
            try:
                hf_service = LLMService(LLMProvider.HUGGINGFACE)
                result = await hf_service.generate_text(test_prompt)
                print(f"   Sonuç: {result}")
                if result.get("success"):
                    print("   ✅ HuggingFace LLM çalışıyor!")
                else:
                    print(f"   ❌ HuggingFace LLM hatası: {result.get('error')}")
            except Exception as e:
                print(f"   ❌ HuggingFace LLM exception: {str(e)}")
        else:
            print("   ⚠️ HuggingFace token eksik, test atlanıyor")
        
        # Özel fonksiyonlar testi
        print("\n🎯 Özel Fonksiyonlar Testi:")
        
        # İpucu üretme testi
        try:
            hint = await llm_service.generate_question_hint(
                "2x + 5 = 13 denklemini çöz", "mathematics"
            )
            print(f"   İpucu Üretimi: {'✅' if hint else '❌'}")
            print(f"   İpucu: {hint[:100]}...")
        except Exception as e:
            print(f"   ❌ İpucu üretimi hatası: {str(e)}")
        
        # Açıklama üretme testi
        try:
            explanation = await llm_service.generate_question_explanation(
                "2x + 5 = 13 denklemini çöz", "x = 4", "mathematics"
            )
            print(f"   Açıklama Üretimi: {'✅' if explanation else '❌'}")
            print(f"   Açıklama: {explanation[:100]}...")
        except Exception as e:
            print(f"   ❌ Açıklama üretimi hatası: {str(e)}")
        
        # Zorluk analizi testi
        try:
            analysis = await llm_service.analyze_question_difficulty(
                "2x + 5 = 13 denklemini çöz", "mathematics"
            )
            print(f"   Zorluk Analizi: {'✅' if analysis else '❌'}")
            print(f"   Analiz: {analysis}")
        except Exception as e:
            print(f"   ❌ Zorluk analizi hatası: {str(e)}")
        
    except ImportError as e:
        print(f"❌ Import hatası: {str(e)}")
        print("   Backend dizininde doğru çalıştırıldığından emin olun")
    except Exception as e:
        print(f"❌ Genel hata: {str(e)}")

def check_environment():
    """Environment dosyası kontrolü"""
    print("📁 Environment Dosyası Kontrolü:")
    
    env_files = [
        ".env",
        "backend/.env",
        "app/.env"
    ]
    
    for env_file in env_files:
        if os.path.exists(env_file):
            print(f"   ✅ {env_file} mevcut")
        else:
            print(f"   ❌ {env_file} eksik")
    
    print("\n💡 Öneriler:")
    print("   1. backend/.env dosyası oluşturun")
    print("   2. OPENAI_API_KEY ve HUGGINGFACE_API_TOKEN ekleyin")
    print("   3. HUGGINGFACE_MODEL belirtin (varsayılan: gpt2)")

def main():
    """Ana fonksiyon"""
    print("=" * 60)
    print("🤖 LLM Servis Test ve Diagnostik Aracı")
    print("=" * 60)
    
    check_environment()
    asyncio.run(test_llm_service())
    
    print("\n" + "=" * 60)
    print("✅ Test tamamlandı!")
    print("=" * 60)

if __name__ == "__main__":
    main() 