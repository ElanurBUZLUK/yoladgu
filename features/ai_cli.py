#!/usr/bin/env python3
"""
AI Servisleri CLI Tool
LLM entegrasyonu ve soru içe aktarma için komut satırı aracı
"""

import asyncio
import argparse
import sys
import os
from typing import List, Dict, Any

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.llm_service import LLMService, LLMProvider
from app.services.question_ingestion_service import QuestionIngestionService
from app.db.database import SessionLocal

class AICLI:
    def __init__(self):
        self.llm_service = LLMService()
        self.ingestion_service = QuestionIngestionService()
    
    async def test_llm_connection(self, provider: str = "openai"):
        """LLM bağlantısını test et"""
        print(f"🔍 {provider.upper()} bağlantısı test ediliyor...")
        
        try:
            if provider == "openai":
                test_service = LLMService(LLMProvider.OPENAI)
            elif provider == "huggingface":
                test_service = LLMService(LLMProvider.HUGGINGFACE)
            else:
                print("❌ Desteklenmeyen provider")
                return False
            
            result = await test_service.generate_text(
                "Merhaba! Bu bir test mesajıdır. Lütfen 'Test başarılı' yanıtını ver.",
                max_tokens=50,
                temperature=0.1
            )
            
            if result["success"]:
                print(f"✅ {provider.upper()} bağlantısı başarılı!")
                print(f"📝 Yanıt: {result['text']}")
                return True
            else:
                print(f"❌ {provider.upper()} bağlantısı başarısız: {result.get('error', 'Bilinmeyen hata')}")
                return False
                
        except Exception as e:
            print(f"❌ {provider.upper()} bağlantı hatası: {str(e)}")
            return False
    
    async def generate_hint(self, question: str, subject: str):
        """Soru için ipucu üret (Batch)"""
        print(f"💡 '{subject}' konusunda ipucu üretiliyor...")
        
        try:
            hint = await self.llm_service.generate_question_hint(question, subject)
            print(f"✅ İpucu üretildi:\n{hint}")
            return hint
        except Exception as e:
            print(f"❌ İpucu üretme hatası: {str(e)}")
            return None
    
    async def generate_adaptive_hint(self, question: str, student_level: int, struggling: bool = False):
        """Öğrenci durumuna göre adaptif ipucu üret (Runtime)"""
        print(f"🤖 Öğrenci seviyesi {student_level} için adaptif ipucu üretiliyor...")
        
        try:
            hint = await self.llm_service.generate_adaptive_hint(
                question, student_level, 0, struggling
            )
            print(f"✅ Adaptif ipucu üretildi:\n{hint}")
            return hint
        except Exception as e:
            print(f"❌ Adaptif ipucu üretme hatası: {str(e)}")
            return None
    
    async def generate_explanation(self, question: str, answer: str, subject: str):
        """Soru için açıklama üret"""
        print(f"📖 '{subject}' konusunda açıklama üretiliyor...")
        
        try:
            explanation = await self.llm_service.generate_question_explanation(question, answer, subject)
            print(f"✅ Açıklama üretildi:\n{explanation}")
            return explanation
        except Exception as e:
            print(f"❌ Açıklama üretme hatası: {str(e)}")
            return None
    
    async def generate_feedback(self, is_correct: bool, topic: str, level: int):
        """AI geri bildirimi üret"""
        print(f"🤖 AI geri bildirimi üretiliyor...")
        
        try:
            feedback = await self.llm_service.generate_ai_feedback(is_correct, topic, level)
            print(f"✅ Geri bildirim üretildi:\n{feedback}")
            return feedback
        except Exception as e:
            print(f"❌ Geri bildirim üretme hatası: {str(e)}")
            return None
    
    async def ingest_from_website(self, url: str, subject: str, topic: str):
        """Web sitesinden soru içe aktar"""
        print(f"🌐 '{url}' adresinden soru içe aktarılıyor...")
        
        try:
            questions = await self.ingestion_service.scrape_from_website(url, subject, topic)
            
            if questions:
                print(f"✅ {len(questions)} soru bulundu")
                
                # Veritabanına kaydet
                db = SessionLocal()
                try:
                    saved_count = await self.ingestion_service.save_questions_to_database(db, questions)
                    print(f"💾 {saved_count} soru veritabanına kaydedildi")
                finally:
                    db.close()
                
                return questions
            else:
                print("❌ Soru bulunamadı")
                return []
                
        except Exception as e:
            print(f"❌ Soru içe aktarma hatası: {str(e)}")
            return []
    
    async def ingest_from_csv(self, file_path: str, subject: str):
        """CSV dosyasından soru içe aktar"""
        print(f"📄 '{file_path}' dosyasından soru içe aktarılıyor...")
        
        try:
            questions = await self.ingestion_service.ingest_from_csv(file_path, subject)
            
            if questions:
                print(f"✅ {len(questions)} soru bulundu")
                
                # Veritabanına kaydet
                db = SessionLocal()
                try:
                    saved_count = await self.ingestion_service.save_questions_to_database(db, questions)
                    print(f"💾 {saved_count} soru veritabanına kaydedildi")
                finally:
                    db.close()
                
                return questions
            else:
                print("❌ Soru bulunamadı")
                return []
                
        except Exception as e:
            print(f"❌ CSV içe aktarma hatası: {str(e)}")
            return []
    
    async def analyze_difficulty(self, question: str, subject: str):
        """Soru zorluğunu analiz et"""
        print(f"📊 '{subject}' konusunda zorluk analizi yapılıyor...")
        
        try:
            analysis = await self.llm_service.analyze_question_difficulty(question, subject)
            print(f"✅ Zorluk analizi tamamlandı:")
            print(f"   Zorluk Seviyesi: {analysis.get('difficulty_level', 'N/A')}/5")
            print(f"   Gerekli Bilgiler: {', '.join(analysis.get('required_knowledge', []))}")
            print(f"   Çözüm Adımları: {analysis.get('solution_steps', 'N/A')}")
            print(f"   Sınıf Seviyesi: {analysis.get('grade_level', 'N/A')}")
            print(f"   Açıklama: {analysis.get('explanation', 'N/A')}")
            return analysis
        except Exception as e:
            print(f"❌ Zorluk analizi hatası: {str(e)}")
            return None

async def main():
    parser = argparse.ArgumentParser(description="AI Servisleri CLI Tool")
    parser.add_argument("command", choices=[
        "test-connection", "generate-hint", "generate-adaptive-hint", "generate-explanation", 
        "generate-feedback", "ingest-website", "ingest-csv", "analyze-difficulty", "batch-enrich"
    ], help="Çalıştırılacak komut")
    
    parser.add_argument("--provider", choices=["openai", "huggingface"], default="openai",
                       help="LLM provider (test-connection için)")
    parser.add_argument("--question", help="Soru metni")
    parser.add_argument("--subject", help="Konu adı")
    parser.add_argument("--answer", help="Doğru cevap")
    parser.add_argument("--topic", help="Konu başlığı")
    parser.add_argument("--level", type=int, help="Öğrenci seviyesi")
    parser.add_argument("--correct", action="store_true", help="Cevap doğru mu?")
    parser.add_argument("--url", help="Web sitesi URL'i")
    parser.add_argument("--file", help="CSV dosya yolu")
    
    args = parser.parse_args()
    
    cli = AICLI()
    
    if args.command == "test-connection":
        await cli.test_llm_connection(args.provider)
    
    elif args.command == "generate-hint":
        if not args.question or not args.subject:
            print("❌ --question ve --subject parametreleri gerekli")
            return
        await cli.generate_hint(args.question, args.subject)
    
    elif args.command == "generate-adaptive-hint":
        if not args.question or args.level is None:
            print("❌ --question ve --level parametreleri gerekli")
            return
        await cli.generate_adaptive_hint(args.question, args.level, args.correct)
    
    elif args.command == "generate-explanation":
        if not args.question or not args.answer or not args.subject:
            print("❌ --question, --answer ve --subject parametreleri gerekli")
            return
        await cli.generate_explanation(args.question, args.answer, args.subject)
    
    elif args.command == "generate-feedback":
        if not args.topic or args.level is None or args.correct is None:
            print("❌ --topic, --level ve --correct parametreleri gerekli")
            return
        await cli.generate_feedback(args.correct, args.topic, args.level)
    
    elif args.command == "ingest-website":
        if not args.url or not args.subject or not args.topic:
            print("❌ --url, --subject ve --topic parametreleri gerekli")
            return
        await cli.ingest_from_website(args.url, args.subject, args.topic)
    
    elif args.command == "ingest-csv":
        if not args.file or not args.subject:
            print("❌ --file ve --subject parametreleri gerekli")
            return
        await cli.ingest_from_csv(args.file, args.subject)
    
    elif args.command == "analyze-difficulty":
        if not args.question or not args.subject:
            print("❌ --question ve --subject parametreleri gerekli")
            return
        await cli.analyze_difficulty(args.question, args.subject)

if __name__ == "__main__":
    asyncio.run(main()) 