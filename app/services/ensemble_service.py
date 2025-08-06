"""
Ensemble Service
Multiple ML model'larını birleştiren ensemble servisi
"""

import structlog
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import numpy as np

logger = structlog.get_logger()


@dataclass
class ModelPrediction:
    """Model tahmini"""
    model_name: str
    prediction: float
    confidence: float
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class EnsemblePrediction:
    """Ensemble tahmini"""
    final_prediction: float
    confidence: float
    model_predictions: List[ModelPrediction]
    ensemble_method: str
    weights: Optional[Dict[str, float]] = None


class EnsembleService:
    """Ensemble model servisi"""
    
    def __init__(self):
        self.models = {}
        self.weights = {}
        self.ensemble_method = "weighted_average"
        self.initialized = False
    
    async def initialize(self):
        """Ensemble servisini başlat"""
        try:
            # Model ağırlıklarını ayarla
            self.weights = {
                "cf_model": 0.4,
                "bandit_model": 0.3,
                "online_model": 0.3
            }
            
            self.initialized = True
            logger.info("ensemble_service_initialized")
            
        except Exception as e:
            logger.error("ensemble_service_initialization_error", error=str(e))
            raise
    
    async def combine_predictions(
        self,
        predictions: List[ModelPrediction],
        method: str = "weighted_average"
    ) -> EnsemblePrediction:
        """Tahminleri birleştir"""
        try:
            if not predictions:
                raise ValueError("No predictions provided")
            
            if method == "weighted_average":
                return await self._weighted_average(predictions)
            elif method == "simple_average":
                return await self._simple_average(predictions)
            elif method == "max_confidence":
                return await self._max_confidence(predictions)
            else:
                raise ValueError(f"Unknown ensemble method: {method}")
                
        except Exception as e:
            logger.error("combine_predictions_error", error=str(e))
            raise
    
    async def _weighted_average(self, predictions: List[ModelPrediction]) -> EnsemblePrediction:
        """Ağırlıklı ortalama"""
        try:
            total_weight = 0.0
            weighted_sum = 0.0
            
            for pred in predictions:
                weight = self.weights.get(pred.model_name, 1.0)
                weighted_sum += pred.prediction * weight
                total_weight += weight
            
            if total_weight == 0:
                final_prediction = np.mean([p.prediction for p in predictions])
            else:
                final_prediction = weighted_sum / total_weight
            
            # Confidence hesapla
            confidence = np.mean([p.confidence for p in predictions])
            
            return EnsemblePrediction(
                final_prediction=final_prediction,
                confidence=confidence,
                model_predictions=predictions,
                ensemble_method="weighted_average",
                weights=self.weights
            )
            
        except Exception as e:
            logger.error("weighted_average_error", error=str(e))
            raise
    
    async def _simple_average(self, predictions: List[ModelPrediction]) -> EnsemblePrediction:
        """Basit ortalama"""
        try:
            final_prediction = np.mean([p.prediction for p in predictions])
            confidence = np.mean([p.confidence for p in predictions])
            
            return EnsemblePrediction(
                final_prediction=final_prediction,
                confidence=confidence,
                model_predictions=predictions,
                ensemble_method="simple_average"
            )
            
        except Exception as e:
            logger.error("simple_average_error", error=str(e))
            raise
    
    async def _max_confidence(self, predictions: List[ModelPrediction]) -> EnsemblePrediction:
        """En yüksek confidence'lı tahmini seç"""
        try:
            # En yüksek confidence'lı tahmini bul
            best_prediction = max(predictions, key=lambda x: x.confidence)
            
            return EnsemblePrediction(
                final_prediction=best_prediction.prediction,
                confidence=best_prediction.confidence,
                model_predictions=predictions,
                ensemble_method="max_confidence"
            )
            
        except Exception as e:
            logger.error("max_confidence_error", error=str(e))
            raise
    
    async def update_weights(self, new_weights: Dict[str, float]):
        """Model ağırlıklarını güncelle"""
        try:
            self.weights.update(new_weights)
            logger.info("ensemble_weights_updated", weights=new_weights)
            
        except Exception as e:
            logger.error("update_weights_error", error=str(e))
            raise
    
    async def get_ensemble_stats(self) -> Dict[str, Any]:
        """Ensemble istatistiklerini getir"""
        try:
            return {
                "initialized": self.initialized,
                "ensemble_method": self.ensemble_method,
                "weights": self.weights,
                "models_count": len(self.weights)
            }
            
        except Exception as e:
            logger.error("get_ensemble_stats_error", error=str(e))
            return {"error": str(e)}


# Global instance
ensemble_service = EnsembleService() 