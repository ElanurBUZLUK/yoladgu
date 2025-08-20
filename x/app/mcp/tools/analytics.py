from typing import Dict, Any, List
from .base import BaseMCPTool


class AnalyticsTool(BaseMCPTool):
    """Öğrenci performans analizi için MCP tool"""
    
    def get_name(self) -> str:
        return "analyze_performance"
    
    def get_description(self) -> str:
        return "Öğrenci performansını analiz eder ve seviye önerileri yapar"
    
    def get_input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "user_id": {
                    "type": "string",
                    "description": "Öğrenci ID'si"
                },
                "subject": {
                    "type": "string",
                    "enum": ["math", "english"],
                    "description": "Analiz edilecek ders"
                },
                "recent_attempts": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "is_correct": {"type": "boolean"},
                            "time_spent": {"type": "integer"},
                            "difficulty_level": {"type": "integer"},
                            "score": {"type": "number"},
                            "error_types": {
                                "type": "array",
                                "items": {"type": "string"}
                            }
                        },
                        "required": ["is_correct", "difficulty_level"]
                    },
                    "description": "Son denemeler"
                },
                "analysis_type": {
                    "type": "string",
                    "enum": ["level_assessment", "error_pattern", "progress_tracking", "similarity_analysis"],
                    "default": "level_assessment",
                    "description": "Analiz tipi"
                },
                "time_period": {
                    "type": "string",
                    "enum": ["last_week", "last_month", "last_3_months", "all_time"],
                    "default": "last_month",
                    "description": "Analiz periyodu"
                }
            },
            "required": ["user_id", "subject", "recent_attempts"]
        }
    
    async def execute(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Performans analizi mantığı"""
        
        try:
            user_id = arguments["user_id"]
            subject = arguments["subject"]
            recent_attempts = arguments["recent_attempts"]
            analysis_type = arguments.get("analysis_type", "level_assessment")
            time_period = arguments.get("time_period", "last_month")
            
            # Gerçek analytics service'i kullan
            from app.services.analytics_service import analytics_service
            
            # Analytics service'e yönlendir
            analysis_result = await analytics_service.analyze_student_performance(
                user_id=user_id,
                subject=subject,
                recent_attempts=recent_attempts,
                analysis_type=analysis_type,
                time_period=time_period
            )
            
            return {
                "success": True,
                "analysis": analysis_result,
                "metadata": {
                    "tool": "analytics",
                    "method": "real_service",
                    "timestamp": "2024-01-01T00:00:00Z"
                }
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "analysis": None,
                "metadata": {
                    "tool": "analytics",
                    "method": "error_fallback",
                    "timestamp": "2024-01-01T00:00:00Z"
                }
            }
    
    def _calculate_basic_stats(self, attempts: List[Dict]) -> Dict[str, Any]:
        """Temel istatistikleri hesapla"""
        if not attempts:
            return {
                "total_attempts": 0,
                "accuracy_rate": 0.0,
                "average_score": 0.0,
                "average_time": 0.0,
                "difficulty_distribution": {}
            }
        
        total_attempts = len(attempts)
        correct_attempts = sum(1 for attempt in attempts if attempt["is_correct"])
        accuracy_rate = correct_attempts / total_attempts
        
        scores = [attempt.get("score", 0) for attempt in attempts]
        average_score = sum(scores) / len(scores) if scores else 0
        
        times = [attempt.get("time_spent", 0) for attempt in attempts if attempt.get("time_spent")]
        average_time = sum(times) / len(times) if times else 0
        
        # Zorluk seviyesi dağılımı
        difficulty_dist = {}
        for attempt in attempts:
            level = attempt["difficulty_level"]
            difficulty_dist[level] = difficulty_dist.get(level, 0) + 1
        
        return {
            "total_attempts": total_attempts,
            "accuracy_rate": round(accuracy_rate, 3),
            "average_score": round(average_score, 2),
            "average_time": round(average_time, 2),
            "difficulty_distribution": difficulty_dist
        }
    
    async def _assess_level(self, attempts: List[Dict], subject: str) -> Dict[str, Any]:
        """Seviye değerlendirmesi"""
        if not attempts:
            return {"current_level": 1, "confidence": 0.0}
        
        # Son 10 denemeye odaklan
        recent_attempts = attempts[-10:]
        
        # Zorluk seviyelerine göre başarı oranları
        level_performance = {}
        for attempt in recent_attempts:
            level = attempt["difficulty_level"]
            if level not in level_performance:
                level_performance[level] = {"correct": 0, "total": 0}
            
            level_performance[level]["total"] += 1
            if attempt["is_correct"]:
                level_performance[level]["correct"] += 1
        
        # Seviye önerisi
        suggested_level = 1
        confidence = 0.0
        
        for level, perf in level_performance.items():
            accuracy = perf["correct"] / perf["total"]
            if accuracy >= 0.8 and perf["total"] >= 3:  # En az 3 deneme ve %80 başarı
                suggested_level = max(suggested_level, level + 1)
                confidence = max(confidence, accuracy)
            elif accuracy >= 0.6:
                suggested_level = max(suggested_level, level)
                confidence = max(confidence, accuracy * 0.8)
        
        return {
            "current_level": min(suggested_level, 5),
            "confidence": round(confidence, 3),
            "level_performance": level_performance,
            "recommendation": self._get_level_recommendation(suggested_level, confidence)
        }
    
    async def _analyze_error_patterns(self, attempts: List[Dict], subject: str) -> Dict[str, Any]:
        """Hata pattern analizi"""
        error_frequency = {}
        error_trends = {}
        
        for i, attempt in enumerate(attempts):
            if not attempt["is_correct"]:
                error_types = attempt.get("error_types", ["unknown_error"])
                for error_type in error_types:
                    if error_type not in error_frequency:
                        error_frequency[error_type] = 0
                        error_trends[error_type] = []
                    
                    error_frequency[error_type] += 1
                    error_trends[error_type].append(i)  # Attempt index for trend analysis
        
        # En sık yapılan hatalar
        most_common_errors = sorted(error_frequency.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Hata trendleri (artıyor mu azalıyor mu?)
        trend_analysis = {}
        for error_type, occurrences in error_trends.items():
            if len(occurrences) >= 3:
                # Son 3 occurrence'ın trend'i
                recent_trend = occurrences[-3:]
                if all(recent_trend[i] < recent_trend[i+1] for i in range(len(recent_trend)-1)):
                    trend_analysis[error_type] = "increasing"
                elif all(recent_trend[i] > recent_trend[i+1] for i in range(len(recent_trend)-1)):
                    trend_analysis[error_type] = "decreasing"
                else:
                    trend_analysis[error_type] = "stable"
        
        return {
            "error_frequency": error_frequency,
            "most_common_errors": most_common_errors,
            "trend_analysis": trend_analysis,
            "total_unique_errors": len(error_frequency),
            "error_diversity": len(error_frequency) / len(attempts) if attempts else 0
        }
    
    async def _track_progress(self, attempts: List[Dict], subject: str) -> Dict[str, Any]:
        """İlerleme takibi"""
        if len(attempts) < 5:
            return {"progress_trend": "insufficient_data"}
        
        # Zaman içinde accuracy trend'i
        window_size = 5
        accuracy_trend = []
        
        for i in range(window_size, len(attempts) + 1):
            window = attempts[i-window_size:i]
            window_accuracy = sum(1 for a in window if a["is_correct"]) / len(window)
            accuracy_trend.append(window_accuracy)
        
        # Trend analizi
        if len(accuracy_trend) >= 3:
            recent_trend = accuracy_trend[-3:]
            if all(recent_trend[i] <= recent_trend[i+1] for i in range(len(recent_trend)-1)):
                progress_direction = "improving"
            elif all(recent_trend[i] >= recent_trend[i+1] for i in range(len(recent_trend)-1)):
                progress_direction = "declining"
            else:
                progress_direction = "stable"
        else:
            progress_direction = "unknown"
        
        # İyileşme hızı
        if len(accuracy_trend) >= 2:
            improvement_rate = accuracy_trend[-1] - accuracy_trend[0]
        else:
            improvement_rate = 0
        
        return {
            "progress_trend": progress_direction,
            "improvement_rate": round(improvement_rate, 3),
            "accuracy_history": accuracy_trend,
            "consistency_score": self._calculate_consistency(accuracy_trend)
        }
    
    async def _find_similar_students(self, user_id: str, attempts: List[Dict], subject: str) -> Dict[str, Any]:
        """Benzer öğrenci analizi"""
        # Mock implementation - gerçekte veritabanından benzer öğrenciler bulunacak
        
        user_profile = {
            "accuracy_rate": sum(1 for a in attempts if a["is_correct"]) / len(attempts) if attempts else 0,
            "avg_difficulty": sum(a["difficulty_level"] for a in attempts) / len(attempts) if attempts else 1,
            "common_errors": ["past_tense", "vocabulary"] if subject == "english" else ["arithmetic", "algebra"]
        }
        
        # Simulated similar students
        similar_students = [
            {
                "student_id": "student_123",
                "similarity_score": 0.85,
                "common_patterns": ["past_tense", "vocabulary"],
                "performance_comparison": "slightly_better"
            },
            {
                "student_id": "student_456", 
                "similarity_score": 0.78,
                "common_patterns": ["past_tense"],
                "performance_comparison": "similar"
            }
        ]
        
        return {
            "user_profile": user_profile,
            "similar_students": similar_students,
            "similarity_factors": ["error_patterns", "difficulty_preference", "learning_pace"],
            "group_recommendations": [
                "Bu seviyedeki öğrenciler genellikle past tense konusunda zorlanıyor",
                "Benzer öğrenciler vocabulary çalışmasından fayda görüyor"
            ]
        }
    
    def _calculate_consistency(self, accuracy_trend: List[float]) -> float:
        """Tutarlılık puanı hesapla"""
        if len(accuracy_trend) < 2:
            return 0.0
        
        # Standart sapma ile tutarlılık ölç
        mean_accuracy = sum(accuracy_trend) / len(accuracy_trend)
        variance = sum((x - mean_accuracy) ** 2 for x in accuracy_trend) / len(accuracy_trend)
        std_dev = variance ** 0.5
        
        # Tutarlılık puanı (düşük std_dev = yüksek tutarlılık)
        consistency = max(0, 1 - (std_dev * 2))
        return round(consistency, 3)
    
    def _get_level_recommendation(self, level: int, confidence: float) -> str:
        """Seviye önerisi metni"""
        if confidence >= 0.8:
            return f"Seviye {level} için hazırsınız. Güvenle geçebilirsiniz."
        elif confidence >= 0.6:
            return f"Seviye {level} için neredeyse hazırsınız. Biraz daha pratik yapın."
        else:
            return f"Mevcut seviyede daha fazla pratik yapmanız önerilir."
    
    def _generate_recommendations(self, basic_stats: Dict, detailed_analysis: Dict, subject: str) -> List[str]:
        """Öneriler üret"""
        recommendations = []
        
        accuracy = basic_stats["accuracy_rate"]
        
        if accuracy < 0.6:
            recommendations.append("Temel konuları tekrar gözden geçirin")
            recommendations.append("Daha kolay sorularla başlayın")
        elif accuracy < 0.8:
            recommendations.append("İyi ilerliyorsunuz, çalışmaya devam edin")
            recommendations.append("Zayıf olduğunuz konulara odaklanın")
        else:
            recommendations.append("Mükemmel performans! Daha zor sorulara geçebilirsiniz")
        
        # Hata pattern'larına göre öneriler
        if "error_frequency" in detailed_analysis:
            most_common = detailed_analysis.get("most_common_errors", [])
            if most_common:
                error_type = most_common[0][0]
                recommendations.append(f"{error_type} konusunda ekstra çalışma yapın")
        
        return recommendations
    
    def _suggest_next_actions(self, basic_stats: Dict, detailed_analysis: Dict) -> List[str]:
        """Sonraki adım önerileri"""
        actions = []
        
        accuracy = basic_stats["accuracy_rate"]
        
        if accuracy >= 0.8:
            actions.append("Zorluk seviyesini artırın")
        elif accuracy < 0.5:
            actions.append("Temel konuları tekrar edin")
        
        actions.append("Günlük 15 dakika pratik yapın")
        actions.append("Hata yaptığınız soruları tekrar çözün")
        
        return actions