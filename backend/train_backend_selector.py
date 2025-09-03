#!/usr/bin/env python3
"""
Training Script for ML-based Backend Selector

This script trains the backend selector model using synthetic data
and saves it for use in the vector service.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent
sys.path.insert(0, str(backend_path))

from app.services.vector_index_manager import vector_index_manager
from app.services.ml.backend_selector import train_backend_selector
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def main():
    """Main training function."""
    print("🚀 ML-based Backend Selector Training")
    print("=" * 50)
    
    try:
        # Initialize vector index manager
        logger.info("Initializing vector index manager...")
        success = await vector_index_manager.initialize()
        
        if not success:
            print("❌ Failed to initialize vector index manager")
            return
        
        print(f"✅ Vector index manager initialized with {len(vector_index_manager.backends)} backends:")
        for name in vector_index_manager.backends:
            print(f"   - {name}")
        
        # Check if FAISS is available (required for training)
        if "faiss" not in vector_index_manager.backends:
            print("❌ FAISS backend is required for training data generation")
            print("   Please ensure FAISS is properly configured")
            return
        
        # Training parameters
        n_examples = 200
        save_path = "data/ml/backend_selector.joblib"
        
        print(f"\n🧪 Starting training with {n_examples} examples...")
        print(f"📁 Model will be saved to: {save_path}")
        
        # Train the model
        result = await train_backend_selector(
            vector_index_manager=vector_index_manager,
            n_examples=n_examples,
            save_path=save_path
        )
        
        if result["ok"]:
            print("\n🎉 Training completed successfully!")
            print(f"   - Training examples: {result['n']}")
            print(f"   - Backend labels: {', '.join(result['labels'])}")
            print(f"   - Model accuracy: {result['accuracy']:.3f}")
            print(f"   - Model saved to: {save_path}")
            
            # Test the trained model
            print("\n🧪 Testing trained model...")
            await test_trained_model(save_path)
            
        else:
            print(f"\n❌ Training failed: {result['msg']}")
            
    except Exception as e:
        logger.error(f"Training script failed: {e}")
        print(f"❌ Error: {e}")


async def test_trained_model(model_path: str):
    """Test the trained model with sample queries."""
    try:
        from app.services.ml.backend_selector import BackendSelector
        from app.services.ml.feature_extractor import extract_query_features
        
        # Load the trained model
        selector = BackendSelector.load(model_path)
        if not selector:
            print("❌ Failed to load trained model")
            return
        
        print("✅ Model loaded successfully")
        
        # Test with sample queries
        test_queries = [
            ("matematik kesirler", 10, None),
            ("algebra equations", 50, {"type": "math"}),
            ("geometry problems", 5, {"difficulty": "easy"}),
            ("calculus integration", 100, {"subject": "advanced"}),
        ]
        
        print("\n🔍 Testing model predictions:")
        print("-" * 60)
        
        for query_text, k, filters in test_queries:
            try:
                # Generate mock features (in real usage, these would come from actual query)
                mock_features = {
                    'query_norm': 1.0,
                    'query_std': 0.5,
                    'query_max': 0.8,
                    'query_min': -0.8,
                    'k_value': float(k),
                    'has_filters': 1.0 if filters else 0.0,
                    'filter_count': float(len(filters)) if filters else 0.0,
                    'total_items': 50000.0,
                    'hnsw_available': 1.0,
                    'faiss_available': 1.0,
                    'qdrant_available': 1.0,
                    'recent_hnsw_performance': 1.2,
                    'recent_faiss_performance': 0.8,
                    'recent_qdrant_performance': 2.1
                }
                
                # Get prediction
                prediction = selector.predict_backend(mock_features)
                probabilities = selector.predict_proba(mock_features)
                
                # Format output
                query_display = query_text[:30] + "..." if len(query_text) > 30 else query_text
                filters_display = str(filters) if filters else "None"
                
                print(f"Query: {query_display:<33} | k={k:<3} | Filters: {filters_display:<20}")
                print(f"Prediction: {prediction:<8} | Probabilities: ", end="")
                
                if probabilities:
                    for backend, prob in sorted(probabilities.items(), key=lambda x: x[1], reverse=True):
                        print(f"{backend}:{prob:.2f} ", end="")
                print()
                print()
                
            except Exception as e:
                print(f"❌ Error testing query '{query_text}': {e}")
        
        # Show model statistics
        stats = selector.get_performance_stats()
        print(f"\n📊 Model Statistics:")
        print(f"   - Model type: {stats['model_type']}")
        print(f"   - Feature count: {stats['feature_count']}")
        print(f"   - Prediction counts: {stats['prediction_counts']}")
        
    except Exception as e:
        logger.error(f"Model testing failed: {e}")
        print(f"❌ Model testing error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
