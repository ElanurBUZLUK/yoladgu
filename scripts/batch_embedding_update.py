#!/usr/bin/env python3
"""
Toplu Embedding Güncelleme Scripti
Mevcut soruların embedding'lerini SBERT ile hesaplar ve PostgreSQL'e kaydeder
"""

import os
import sys
import asyncio
import time
from typing import Dict, List
import structlog

# Proje root'unu Python path'ine ekle
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# from app.services.embedding_service import embedding_service  # Removed - use enhanced_embedding_service if needed

# All embedding_service usage below should be replaced with enhanced_embedding_service or removed.
# For now, raise NotImplementedError for all embedding_service calls.

def not_implemented(*args, **kwargs):
    raise NotImplementedError('embedding_service is removed. Use enhanced_embedding_service instead.')

embedding_service = type('Dummy', (), {
    'batch_update_embeddings': staticmethod(not_implemented),
    'get_questions_without_embeddings': staticmethod(not_implemented),
    'compute_embeddings_batch': staticmethod(not_implemented),
    'save_embeddings_batch': staticmethod(not_implemented),
    'update_question_embedding': staticmethod(not_implemented),
    'get_embedding_stats': staticmethod(not_implemented),
})()

from app.core.config import settings

logger = structlog.get_logger()

async def update_all_embeddings(batch_size: int = 50, max_batches: int = None):
    """Tüm soruların embedding'lerini güncelle"""
    print("🚀 Toplu Embedding Güncelleme Başlıyor...")
    
    total_start_time = time.time()
    total_processed = 0
    total_success = 0
    total_failed = 0
    batch_count = 0
    
    try:
        while True:
            # Batch güncelleme
            result = await embedding_service.batch_update_embeddings(batch_size)
            
            batch_count += 1
            total_processed += result.get("processed", 0)
            total_success += result.get("success", 0)
            total_failed += result.get("failed", 0)
            
            print(f"📦 Batch {batch_count}: {result.get('processed', 0)} işlendi, "
                  f"{result.get('success', 0)} başarılı, "
                  f"{result.get('failed', 0)} başarısız "
                  f"({result.get('duration', 0):.2f}s)")
            
            # Eğer işlenecek soru kalmadıysa dur
            if result.get("processed", 0) == 0:
                break
            
            # Maksimum batch sayısı kontrolü
            if max_batches and batch_count >= max_batches:
                print(f"⚠️  Maksimum batch sayısına ({max_batches}) ulaşıldı.")
                break
            
            # Kısa bir bekleme (rate limiting)
            await asyncio.sleep(0.1)
    
    except KeyboardInterrupt:
        print("\n⏹️  Kullanıcı tarafından durduruldu.")
    except Exception as e:
        logger.error("batch_update_error", error=str(e))
        print(f"❌ Toplu güncelleme hatası: {e}")
    
    total_duration = time.time() - total_start_time
    
    print(f"\n📊 Toplu Güncelleme Tamamlandı:")
    print(f"   Toplam Batch: {batch_count}")
    print(f"   Toplam İşlenen: {total_processed}")
    print(f"   Başarılı: {total_success}")
    print(f"   Başarısız: {total_failed}")
    print(f"   Toplam Süre: {total_duration:.2f}s")
    print(f"   Ortalama Hız: {total_processed/total_duration:.1f} soru/saniye")

async def update_embeddings_for_subject(subject_id: int, batch_size: int = 50):
    """Belirli bir ders için embedding'leri güncelle"""
    print(f"📚 Ders {subject_id} için embedding güncelleme başlıyor...")
    
    try:
        # Ders için embedding'i olmayan soruları getir
        questions = embedding_service.get_questions_without_embeddings(1000)  # Büyük limit
        subject_questions = [q for q in questions if q['subject_id'] == subject_id]
        
        if not subject_questions:
            print(f"✅ Ders {subject_id} için güncellenecek soru bulunamadı.")
            return
        
        print(f"📝 Ders {subject_id} için {len(subject_questions)} soru bulundu.")
        
        # Batch'ler halinde işle
        for i in range(0, len(subject_questions), batch_size):
            batch = subject_questions[i:i + batch_size]
            
            question_ids = [q['id'] for q in batch]
            texts = [q['content'] for q in batch]
            
            # Embedding'leri hesapla
            embeddings = embedding_service.compute_embeddings_batch(texts)
            
            # Veritabanına kaydet
            question_embeddings = list(zip(question_ids, embeddings))
            saved_count = embedding_service.save_embeddings_batch(question_embeddings)
            
            print(f"   Batch {i//batch_size + 1}: {len(batch)} soru, {saved_count} kaydedildi")
        
        print(f"✅ Ders {subject_id} için embedding güncelleme tamamlandı.")
        
    except Exception as e:
        logger.error("subject_embedding_update_error", subject_id=subject_id, error=str(e))
        print(f"❌ Ders {subject_id} güncelleme hatası: {e}")

async def update_single_question_embedding(question_id: int):
    """Tek soru için embedding güncelle"""
    try:
        # Soruyu getir
        from app.db.database import SessionLocal
        from app.db.models import Question
        
        db = SessionLocal()
        question = db.query(Question).filter(Question.id == question_id).first()
        db.close()
        
        if not question:
            print(f"❌ Soru {question_id} bulunamadı.")
            return False
        
        # Embedding güncelle
        success = await embedding_service.update_question_embedding(question_id, question.content)
        
        if success:
            print(f"✅ Soru {question_id} embedding'i güncellendi.")
        else:
            print(f"❌ Soru {question_id} embedding güncelleme başarısız.")
        
        return success
        
    except Exception as e:
        logger.error("single_question_update_error", question_id=question_id, error=str(e))
        print(f"❌ Soru {question_id} güncelleme hatası: {e}")
        return False

def show_embedding_stats():
    """Embedding istatistiklerini göster"""
    try:
        stats = embedding_service.get_embedding_stats()
        
        print("📊 Embedding İstatistikleri:")
        print(f"   Toplam Soru: {stats.get('total_questions', 0)}")
        print(f"   Embedding'li: {stats.get('questions_with_embedding', 0)}")
        print(f"   Embedding'siz: {stats.get('questions_without_embedding', 0)}")
        print(f"   Kapsama Oranı: {stats.get('embedding_coverage', 0):.1f}%")
        
        if stats.get('questions_without_embedding', 0) > 0:
            print(f"\n💡 {stats.get('questions_without_embedding', 0)} soru için embedding hesaplanması gerekiyor.")
        
    except Exception as e:
        logger.error("stats_error", error=str(e))
        print(f"❌ İstatistik hatası: {e}")

async def main():
    """Ana fonksiyon"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Toplu Embedding Güncelleme")
    parser.add_argument("--batch-size", type=int, default=50, help="Batch boyutu")
    parser.add_argument("--max-batches", type=int, help="Maksimum batch sayısı")
    parser.add_argument("--subject", type=int, help="Belirli ders ID'si")
    parser.add_argument("--question", type=int, help="Tek soru ID'si")
    parser.add_argument("--stats", action="store_true", help="İstatistikleri göster")
    
    args = parser.parse_args()
    
    if args.stats:
        show_embedding_stats()
        return
    
    if args.question:
        await update_single_question_embedding(args.question)
        return
    
    if args.subject:
        await update_embeddings_for_subject(args.subject, args.batch_size)
        return
    
    # Tüm soruları güncelle
    await update_all_embeddings(args.batch_size, args.max_batches)

if __name__ == "__main__":
    asyncio.run(main()) 