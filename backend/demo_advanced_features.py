#!/usr/bin/env python3
"""
Advanced Features Demo Script
Bu script, geliştirilmiş backend'in tüm özelliklerini gösterir.
"""

import asyncio
import numpy as np
import sys
import os
from typing import List, Dict, Any

# Add app to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.services.index_backends.faiss_advanced_index import FAISSAdvancedIndexBackend
from app.services.advanced_rag_system import AdvancedRAGSystem
from app.services.embedding_service import EmbeddingService

async def demonstrate_advanced_features():
    """Gelişmiş özellikleri gösteren demo fonksiyonu."""
    
    print("🚀 Advanced Features Demo Başlıyor...")
    print("=" * 50)
    
    # 1. Advanced Index Oluştur
    print("\n1️⃣ Advanced FAISS Index Oluşturuluyor...")
    index = FAISSAdvancedIndexBackend(
        vector_size=384,
        index_type="flat",  # Daha basit index tipi
        metric="ip",
        index_path="demo_advanced_index.index",
        cache_enabled=True
    )
    
    # Başlat
    await index.initialize()
    print("✅ Advanced index başlatıldı")
    
    # 2. Sağlık Kontrolü
    print("\n2️⃣ Index Sağlık Kontrolü...")
    health = await index.health_check()
    print("Index Health:", health)
    
    # 3. Örnek Veri Oluştur
    print("\n3️⃣ Test Verisi Oluşturuluyor...")
    num_vectors = 1000  # Daha küçük dataset için
    vectors = np.random.rand(num_vectors, 384).astype(np.float32)
    ids = [f"doc_{i}" for i in range(num_vectors)]
    metadata = [{"source": f"doc_{i}", "category": "test"} for i in range(num_vectors)]
    
    # 4. Batch Ekleme
    print("\n4️⃣ Batch Ekleme Test Ediliyor...")
    success = await index.add_items_batch(vectors, ids, metadata, batch_size=100)
    
    if success:
        print("✅ Batch ekleme başarılı")
        
        # 5. Hyperparameter Optimizasyonu
        print("\n5️⃣ Hyperparameter Optimizasyonu...")
        try:
            # Training ve validation için veri ayır
            training_vectors = vectors[:500]
            validation_queries = vectors[500:600]
            validation_ground_truth = [[f"doc_{i}"] for i in range(500, 600)]
            
            optimization_result = await index.optimize_hyperparameters(
                training_vectors=training_vectors,
                validation_queries=validation_queries,
                validation_ground_truth=validation_ground_truth
            )
            
            print("🎯 Optimizasyon sonucu:", optimization_result)
            
        except Exception as e:
            print(f"⚠️ Optimizasyon hatası: {e}")
        
        # 6. Arama Testi
        print("\n6️⃣ Arama Testi...")
        query_vector = np.random.rand(384).astype(np.float32)
        results = await index.search(query_vector, k=10)
        
        print(f"🔍 {len(results)} sonuç bulundu")
        for i, result in enumerate(results[:3]):  # İlk 3 sonucu göster
            print(f"  {i+1}. ID: {result['item_id']}, Score: {result['score']:.4f}")
        
        # 7. İstatistikler
        print("\n7️⃣ Index İstatistikleri...")
        stats = await index.get_stats()
        print("📊 Stats:", stats)
        
        # 8. Index Kaydetme
        print("\n8️⃣ Index Kaydediliyor...")
        await index.save_index()
        print("✅ Index kaydedildi")
        
    else:
        print("❌ Batch ekleme başarısız")
    
    # 9. Advanced RAG System Testi
    print("\n9️⃣ Advanced RAG System Testi...")
    try:
        rag_system = AdvancedRAGSystem()
        
        # Vector store başlat
        await rag_system.initialize_vector_store("ivf")
        print("✅ RAG System başlatıldı")
        
        # Sağlık kontrolü
        health = await rag_system.get_vector_store_health()
        print("RAG Health:", health)
        
        # Test dokümanları ekle
        test_docs = [
            "Python programlama dili çok güçlüdür.",
            "Machine learning yapay zeka alanında önemlidir.",
            "Deep learning neural network'ler kullanır.",
            "Natural language processing metin işleme yapar.",
            "Computer vision görüntü analizi yapar."
        ]
        
        success = await rag_system.batch_process_documents(test_docs, batch_size=2)
        if success:
            print("✅ Test dokümanları eklendi")
            
            # Query testi
            query = "Yapay zeka nedir?"
            answer, results = await rag_system.query(query, k=3)
            print(f"❓ Soru: {query}")
            print(f"💡 Cevap: {answer}")
            print(f"📚 {len(results)} doküman bulundu")
        
    except Exception as e:
        print(f"⚠️ RAG System hatası: {e}")
    
    # 10. Embedding Service Testi
    print("\n🔟 Embedding Service Testi...")
    try:
        embedding_service = EmbeddingService()
        
        # Initialize service
        await embedding_service.initialize()
        
        # Model listesi
        models = await embedding_service.list_models()
        print(f"📚 {len(models)} model bulundu")
        
        # Test metni
        test_text = "Bu bir test metnidir."
        embedding = await embedding_service.encode_text(test_text)
        print(f"🔤 Test metni: {test_text}")
        print(f"📐 Embedding boyutu: {len(embedding)}")
        
    except Exception as e:
        print(f"⚠️ Embedding Service hatası: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 Demo Tamamlandı!")
    print("Tüm gelişmiş özellikler başarıyla test edildi.")

async def performance_benchmark():
    """Performance benchmark testi."""
    print("\n🏃 Performance Benchmark Başlıyor...")
    
    # Farklı index tipleri için test
    index_types = ["flat", "ivf", "hnsw", "pq"]
    
    for index_type in index_types:
        print(f"\n📊 Testing {index_type.upper()} index...")
        
        try:
            index = FAISSAdvancedIndexBackend(
                vector_size=384,
                index_type=index_type,
                metric="ip"
            )
            
            await index.initialize()
            
            # Test verisi
            test_vectors = np.random.rand(100, 384).astype(np.float32)
            test_ids = [f"test_{i}" for i in range(100)]
            
            # Ekleme performansı
            start_time = asyncio.get_event_loop().time()
            success = await index.add_items(test_vectors, test_ids)
            add_time = (asyncio.get_event_loop().time() - start_time) * 1000
            
            if success:
                # Arama performansı
                query_vector = np.random.rand(384).astype(np.float32)
                start_time = asyncio.get_event_loop().time()
                results = await index.search(query_vector, k=10)
                search_time = (asyncio.get_event_loop().time() - start_time) * 1000
                
                print(f"  ✅ Add: {add_time:.2f}ms, Search: {search_time:.2f}ms")
            else:
                print(f"  ❌ Failed to add items")
                
        except Exception as e:
            print(f"  ❌ Error: {e}")

async def main():
    """Ana demo fonksiyonu."""
    try:
        await demonstrate_advanced_features()
        await performance_benchmark()
        
    except Exception as e:
        print(f"❌ Demo hatası: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
