#!/usr/bin/env python3
"""
PostgreSQL pgvector kurulumu ve embedding sütunu ekleme scripti
"""

import os
import sys
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from app.core.config import settings
import structlog

logger = structlog.get_logger()

def get_postgres_connection():
    """PostgreSQL bağlantısı oluştur"""
    try:
        # Parse DATABASE_URL
        db_url = settings.DATABASE_URL
        if db_url.startswith('postgresql+psycopg2://'):
            db_url = db_url.replace('postgresql+psycopg2://', 'postgresql://')
        
        conn = psycopg2.connect(db_url)
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        return conn
    except Exception as e:
        logger.error("postgres_connection_error", error=str(e))
        raise

def install_pgvector_extension():
    """pgvector extension'ını kur"""
    conn = get_postgres_connection()
    cursor = conn.cursor()
    
    try:
        # pgvector extension'ını kur
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        logger.info("pgvector_extension_created")
        print("✅ pgvector extension başarıyla kuruldu!")
        
    except Exception as e:
        logger.error("pgvector_installation_error", error=str(e))
        print(f"❌ pgvector kurulum hatası: {e}")
        print("💡 Çözüm: PostgreSQL superuser olarak giriş yapın:")
        print("   sudo -u postgres psql -d yoladgu -c 'CREATE EXTENSION vector;'")
        raise
    finally:
        cursor.close()
        conn.close()

def add_embedding_column():
    """questions tablosuna embedding sütunu ekle"""
    conn = get_postgres_connection()
    cursor = conn.cursor()
    
    try:
        # Embedding sütununu ekle (384 boyutlu vector)
        cursor.execute("""
            ALTER TABLE questions 
            ADD COLUMN IF NOT EXISTS embedding vector(384);
        """)
        
        # Index oluştur (cosine similarity için)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS questions_embedding_idx 
            ON questions 
            USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100);
        """)
        
        logger.info("embedding_column_added")
        print("✅ Embedding sütunu ve index başarıyla eklendi!")
        
    except Exception as e:
        logger.error("embedding_column_error", error=str(e))
        print(f"❌ Embedding sütunu ekleme hatası: {e}")
        raise
    finally:
        cursor.close()
        conn.close()

def verify_embedding_setup():
    """Embedding kurulumunu doğrula"""
    conn = get_postgres_connection()
    cursor = conn.cursor()
    
    try:
        # Extension'ın kurulu olup olmadığını kontrol et
        cursor.execute("SELECT * FROM pg_extension WHERE extname = 'vector';")
        extension_exists = cursor.fetchone()
        
        # Embedding sütununun var olup olmadığını kontrol et
        cursor.execute("""
            SELECT column_name, data_type 
            FROM information_schema.columns 
            WHERE table_name = 'questions' AND column_name = 'embedding';
        """)
        column_exists = cursor.fetchone()
        
        # Index'in var olup olmadığını kontrol et
        cursor.execute("""
            SELECT indexname 
            FROM pg_indexes 
            WHERE tablename = 'questions' AND indexname = 'questions_embedding_idx';
        """)
        index_exists = cursor.fetchone()
        
        print("\n📊 Kurulum Durumu:")
        print(f"   pgvector Extension: {'✅' if extension_exists else '❌'}")
        print(f"   Embedding Sütunu: {'✅' if column_exists else '❌'}")
        print(f"   Embedding Index: {'✅' if index_exists else '❌'}")
        
        if extension_exists and column_exists and index_exists:
            print("\n🎉 Tüm embedding bileşenleri başarıyla kuruldu!")
            return True
        else:
            print("\n⚠️  Bazı bileşenler eksik. Lütfen kurulumu tamamlayın.")
            return False
            
    except Exception as e:
        logger.error("verification_error", error=str(e))
        print(f"❌ Doğrulama hatası: {e}")
        return False
    finally:
        cursor.close()
        conn.close()

def create_embedding_functions():
    """Embedding işlemleri için yardımcı fonksiyonlar oluştur"""
    conn = get_postgres_connection()
    cursor = conn.cursor()
    
    try:
        # Cosine similarity fonksiyonu
        cursor.execute("""
            CREATE OR REPLACE FUNCTION cosine_similarity(a vector, b vector)
            RETURNS float
            LANGUAGE plpgsql
            AS $$
            BEGIN
                RETURN 1 - (a <=> b);
            END;
            $$;
        """)
        
        # En yakın embedding'leri bulma fonksiyonu
        cursor.execute("""
            CREATE OR REPLACE FUNCTION find_similar_questions(
                query_embedding vector(384),
                match_threshold float DEFAULT 0.8,
                match_count int DEFAULT 10
            )
            RETURNS TABLE (
                id int,
                content text,
                similarity float
            )
            LANGUAGE plpgsql
            AS $$
            BEGIN
                RETURN QUERY
                SELECT 
                    q.id,
                    q.content,
                    1 - (q.embedding <=> query_embedding) as similarity
                FROM questions q
                WHERE q.embedding IS NOT NULL
                AND 1 - (q.embedding <=> query_embedding) > match_threshold
                ORDER BY q.embedding <=> query_embedding
                LIMIT match_count;
            END;
            $$;
        """)
        
        logger.info("embedding_functions_created")
        print("✅ Embedding yardımcı fonksiyonları oluşturuldu!")
        
    except Exception as e:
        logger.error("embedding_functions_error", error=str(e))
        print(f"❌ Embedding fonksiyonları oluşturma hatası: {e}")
        raise
    finally:
        cursor.close()
        conn.close()

def main():
    """Ana fonksiyon"""
    print("🚀 PostgreSQL pgvector Kurulumu Başlıyor...")
    
    try:
        # 1. pgvector extension'ını kur
        install_pgvector_extension()
        
        # 2. Embedding sütununu ekle
        add_embedding_column()
        
        # 3. Yardımcı fonksiyonları oluştur
        create_embedding_functions()
        
        # 4. Kurulumu doğrula
        success = verify_embedding_setup()
        
        if success:
            print("\n🎯 Kurulum Tamamlandı!")
            print("📝 Sonraki adımlar:")
            print("   1. SBERT modelini kurun: pip install sentence-transformers")
            print("   2. Toplu embedding güncelleme scriptini çalıştırın")
            print("   3. Ensemble skor sistemini aktifleştirin")
        else:
            print("\n⚠️  Kurulum tamamlanamadı. Lütfen hataları kontrol edin.")
            sys.exit(1)
            
    except Exception as e:
        logger.error("setup_failed", error=str(e))
        print(f"\n❌ Kurulum başarısız: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 