import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, or_
from sqlalchemy.orm import joinedload

from app.database import database_manager
from app.models.user import User
from app.models.student_attempt import StudentAttempt
from app.models.error_pattern import ErrorPattern
from app.models.question import Question, Subject
from app.services.llm_gateway import llm_gateway
from app.core.config import settings

logger = logging.getLogger(__name__)


class AnalyticsService:
    """Analytics service - Öğrenci performans analizi ve trend hesaplama"""
    
    def __init__(self):
        self.cache = {}
        self.cache_ttl = 300  # 5 dakika
    
    async def analyze_student_performance(
        self,
        db: AsyncSession,
        user_id: str,
        subject: Subject,
        days: int = 30
    ) -> Dict[str, Any]:
        """Öğrenci performans analizi"""
        
        try:
            # Öğrenci denemelerini al
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            result = await db.execute(
                select(StudentAttempt).where(
                    and_(
                        StudentAttempt.user_id == user_id,
                        StudentAttempt.subject == subject,
                        StudentAttempt.created_at >= cutoff_date
                    )
                ).order_by(StudentAttempt.created_at.desc())
            )
            attempts = result.scalars().all()
            
            if not attempts:
                return {
                    "user_id": user_id,
                    "subject": subject.value,
                    "analysis_period_days": days,
                    "total_attempts": 0,
                    "message": "No attempts found for the specified period"
                }
            
            # Temel istatistikler
            total_attempts = len(attempts)
            correct_attempts = len([a for a in attempts if a.is_correct])
            accuracy_rate = (correct_attempts / total_attempts) * 100 if total_attempts > 0 else 0
            
            # Zorluk seviyesi analizi
            difficulty_analysis = await self._analyze_difficulty_performance(attempts)
            
            # Hata kalıpları analizi
            error_analysis = await self._analyze_error_patterns(db, user_id, subject, days)
            
            # Trend analizi
            trend_analysis = await self._analyze_performance_trend(attempts)
            
            # Güçlü ve zayıf yönler
            strengths_weaknesses = await self._identify_strengths_weaknesses(attempts)
            
            # Öneriler
            recommendations = await self._generate_recommendations(
                accuracy_rate, difficulty_analysis, error_analysis, strengths_weaknesses
            )
            
            return {
                "user_id": user_id,
                "subject": subject.value,
                "analysis_period_days": days,
                "total_attempts": total_attempts,
                "correct_attempts": correct_attempts,
                "accuracy_rate": round(accuracy_rate, 2),
                "difficulty_analysis": difficulty_analysis,
                "error_analysis": error_analysis,
                "trend_analysis": trend_analysis,
                "strengths_weaknesses": strengths_weaknesses,
                "recommendations": recommendations,
                "analysis_timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing student performance: {e}")
            raise
    
    async def find_similar_students(
        self,
        db: AsyncSession,
        user_id: str,
        subject: Subject,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Benzer öğrencileri bul"""
        
        try:
            # Hedef öğrencinin performansını al
            target_performance = await self._get_student_performance_profile(db, user_id, subject)
            
            if not target_performance:
                return []
            
            # Diğer öğrencileri bul
            result = await db.execute(
                select(User).where(
                    and_(
                        User.id != user_id,
                        User.role == "student",
                        User.is_active == "true"
                    )
                )
            )
            other_students = result.scalars().all()
            
            similar_students = []
            
            for student in other_students:
                try:
                    # Öğrenci performansını al
                    student_performance = await self._get_student_performance_profile(
                        db, str(student.id), subject
                    )
                    
                    if student_performance:
                        # Benzerlik skoru hesapla
                        similarity_score = await self._calculate_similarity_score(
                            target_performance, student_performance
                        )
                        
                        if similarity_score > 0.6:  # %60'dan fazla benzerlik
                            similar_students.append({
                                "user_id": str(student.id),
                                "username": student.username,
                                "similarity_score": round(similarity_score, 3),
                                "performance_profile": student_performance
                            })
                
                except Exception as e:
                    logger.warning(f"Error analyzing student {student.id}: {e}")
                    continue
            
            # Benzerlik skoruna göre sırala
            similar_students.sort(key=lambda x: x["similarity_score"], reverse=True)
            
            return similar_students[:limit]
            
        except Exception as e:
            logger.error(f"Error finding similar students: {e}")
            raise
    
    async def calculate_performance_trends(
        self,
        db: AsyncSession,
        user_id: str,
        subject: Subject,
        days: int = 30
    ) -> Dict[str, Any]:
        """Performans trendlerini hesapla"""
        
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            # Günlük performans verilerini al
            result = await db.execute(
                select(
                    func.date(StudentAttempt.created_at).label('date'),
                    func.count(StudentAttempt.id).label('total_attempts'),
                    func.sum(func.case((StudentAttempt.is_correct == True, 1), else_=0)).label('correct_attempts'),
                    func.avg(StudentAttempt.difficulty_level).label('avg_difficulty'),
                    func.avg(StudentAttempt.time_spent).label('avg_time_spent')
                ).where(
                    and_(
                        StudentAttempt.user_id == user_id,
                        StudentAttempt.subject == subject,
                        StudentAttempt.created_at >= cutoff_date
                    )
                ).group_by(func.date(StudentAttempt.created_at))
                .order_by(func.date(StudentAttempt.created_at))
            )
            
            daily_stats = result.all()
            
            if not daily_stats:
                return {
                    "user_id": user_id,
                    "subject": subject.value,
                    "trend_period_days": days,
                    "message": "No data available for trend analysis"
                }
            
            # Trend hesaplamaları
            accuracy_trend = []
            difficulty_trend = []
            time_trend = []
            
            for stat in daily_stats:
                date_str = stat.date.isoformat()
                accuracy = (stat.correct_attempts / stat.total_attempts * 100) if stat.total_attempts > 0 else 0
                
                accuracy_trend.append({
                    "date": date_str,
                    "accuracy": round(accuracy, 2),
                    "attempts": stat.total_attempts
                })
                
                difficulty_trend.append({
                    "date": date_str,
                    "avg_difficulty": round(stat.avg_difficulty, 2),
                    "attempts": stat.total_attempts
                })
                
                time_trend.append({
                    "date": date_str,
                    "avg_time_spent": round(stat.avg_time_spent, 2),
                    "attempts": stat.total_attempts
                })
            
            # Trend yönü analizi
            trend_direction = await self._analyze_trend_direction(accuracy_trend)
            
            return {
                "user_id": user_id,
                "subject": subject.value,
                "trend_period_days": days,
                "accuracy_trend": accuracy_trend,
                "difficulty_trend": difficulty_trend,
                "time_trend": time_trend,
                "trend_direction": trend_direction,
                "analysis_timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error calculating performance trends: {e}")
            raise
    
    async def identify_weaknesses_strengths(
        self,
        db: AsyncSession,
        user_id: str,
        subject: Subject,
        days: int = 30
    ) -> Dict[str, Any]:
        """Güçlü ve zayıf yönleri belirle"""
        
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            # Öğrenci denemelerini al
            result = await db.execute(
                select(StudentAttempt).where(
                    and_(
                        StudentAttempt.user_id == user_id,
                        StudentAttempt.subject == subject,
                        StudentAttempt.created_at >= cutoff_date
                    )
                ).options(joinedload(StudentAttempt.question))
            )
            attempts = result.scalars().all()
            
            if not attempts:
                return {
                    "user_id": user_id,
                    "subject": subject.value,
                    "analysis_period_days": days,
                    "message": "No attempts found for analysis"
                }
            
            # Konu bazında performans analizi
            topic_performance = {}
            
            for attempt in attempts:
                if attempt.question:
                    topic = attempt.question.topic_category
                    if topic not in topic_performance:
                        topic_performance[topic] = {
                            "total": 0,
                            "correct": 0,
                            "difficulties": []
                        }
                    
                    topic_performance[topic]["total"] += 1
                    if attempt.is_correct:
                        topic_performance[topic]["correct"] += 1
                    topic_performance[topic]["difficulties"].append(attempt.difficulty_level)
            
            # Güçlü ve zayıf yönleri belirle
            strengths = []
            weaknesses = []
            
            for topic, stats in topic_performance.items():
                accuracy = (stats["correct"] / stats["total"]) * 100
                avg_difficulty = sum(stats["difficulties"]) / len(stats["difficulties"])
                
                topic_analysis = {
                    "topic": topic,
                    "accuracy": round(accuracy, 2),
                    "total_attempts": stats["total"],
                    "avg_difficulty": round(avg_difficulty, 2)
                }
                
                if accuracy >= 70 and avg_difficulty >= 3:
                    strengths.append(topic_analysis)
                elif accuracy < 50 or (accuracy < 60 and avg_difficulty < 3):
                    weaknesses.append(topic_analysis)
            
            # Sıralama
            strengths.sort(key=lambda x: x["accuracy"], reverse=True)
            weaknesses.sort(key=lambda x: x["accuracy"])
            
            return {
                "user_id": user_id,
                "subject": subject.value,
                "analysis_period_days": days,
                "strengths": strengths,
                "weaknesses": weaknesses,
                "topic_performance": topic_performance,
                "analysis_timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error identifying weaknesses/strengths: {e}")
            raise
    
    async def _analyze_difficulty_performance(self, attempts: List[StudentAttempt]) -> Dict[str, Any]:
        """Zorluk seviyesi performans analizi"""
        
        difficulty_stats = {}
        
        for attempt in attempts:
            difficulty = attempt.difficulty_level
            if difficulty not in difficulty_stats:
                difficulty_stats[difficulty] = {"total": 0, "correct": 0, "times": []}
            
            difficulty_stats[difficulty]["total"] += 1
            if attempt.is_correct:
                difficulty_stats[difficulty]["correct"] += 1
            difficulty_stats[difficulty]["times"].append(attempt.time_spent)
        
        # İstatistikleri hesapla
        analysis = {}
        for difficulty, stats in difficulty_stats.items():
            accuracy = (stats["correct"] / stats["total"]) * 100
            avg_time = sum(stats["times"]) / len(stats["times"])
            
            analysis[difficulty] = {
                "total_attempts": stats["total"],
                "correct_attempts": stats["correct"],
                "accuracy_rate": round(accuracy, 2),
                "average_time_spent": round(avg_time, 2)
            }
        
        return analysis
    
    async def _analyze_error_patterns(
        self,
        db: AsyncSession,
        user_id: str,
        subject: Subject,
        days: int
    ) -> Dict[str, Any]:
        """Hata kalıpları analizi"""
        
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        result = await db.execute(
            select(ErrorPattern).where(
                and_(
                    ErrorPattern.user_id == user_id,
                    ErrorPattern.subject == subject,
                    ErrorPattern.created_at >= cutoff_date
                )
            )
        )
        error_patterns = result.scalars().all()
        
        if not error_patterns:
            return {"total_patterns": 0, "patterns": []}
        
        # Hata kalıplarını analiz et
        pattern_analysis = []
        for pattern in error_patterns:
            pattern_analysis.append({
                "pattern_type": pattern.pattern_type,
                "frequency": pattern.frequency,
                "severity": pattern.severity,
                "last_occurrence": pattern.last_occurrence.isoformat() if pattern.last_occurrence else None,
                "suggested_intervention": pattern.suggested_intervention
            })
        
        # Sıklığa göre sırala
        pattern_analysis.sort(key=lambda x: x["frequency"], reverse=True)
        
        return {
            "total_patterns": len(error_patterns),
            "patterns": pattern_analysis
        }
    
    async def _analyze_performance_trend(self, attempts: List[StudentAttempt]) -> Dict[str, Any]:
        """Performans trend analizi"""
        
        if len(attempts) < 2:
            return {"trend": "insufficient_data", "message": "Need more attempts for trend analysis"}
        
        # Son 10 denemeyi al
        recent_attempts = attempts[:10]
        
        # Doğruluk oranı trendi
        recent_accuracy = len([a for a in recent_attempts if a.is_correct]) / len(recent_attempts) * 100
        
        # Zorluk seviyesi trendi
        recent_difficulty = sum(a.difficulty_level for a in recent_attempts) / len(recent_attempts)
        
        # Trend yönü belirle
        if recent_accuracy > 70 and recent_difficulty > 3:
            trend = "improving"
            message = "Performance is improving with higher difficulty levels"
        elif recent_accuracy > 60:
            trend = "stable"
            message = "Performance is stable"
        else:
            trend = "declining"
            message = "Performance needs attention"
        
        return {
            "trend": trend,
            "message": message,
            "recent_accuracy": round(recent_accuracy, 2),
            "recent_avg_difficulty": round(recent_difficulty, 2)
        }
    
    async def _identify_strengths_weaknesses(self, attempts: List[StudentAttempt]) -> Dict[str, Any]:
        """Güçlü ve zayıf yönleri belirle"""
        
        if not attempts:
            return {"strengths": [], "weaknesses": []}
        
        # Konu bazında performans
        topic_performance = {}
        
        for attempt in attempts:
            if attempt.question:
                topic = attempt.question.topic_category
                if topic not in topic_performance:
                    topic_performance[topic] = {"total": 0, "correct": 0}
                
                topic_performance[topic]["total"] += 1
                if attempt.is_correct:
                    topic_performance[topic]["correct"] += 1
        
        strengths = []
        weaknesses = []
        
        for topic, stats in topic_performance.items():
            accuracy = (stats["correct"] / stats["total"]) * 100
            
            if accuracy >= 70:
                strengths.append({
                    "topic": topic,
                    "accuracy": round(accuracy, 2),
                    "attempts": stats["total"]
                })
            elif accuracy < 50:
                weaknesses.append({
                    "topic": topic,
                    "accuracy": round(accuracy, 2),
                    "attempts": stats["total"]
                })
        
        return {
            "strengths": strengths,
            "weaknesses": weaknesses
        }
    
    async def _generate_recommendations(
        self,
        accuracy_rate: float,
        difficulty_analysis: Dict[str, Any],
        error_analysis: Dict[str, Any],
        strengths_weaknesses: Dict[str, Any]
    ) -> List[str]:
        """Öneriler oluştur"""
        
        recommendations = []
        
        # Genel performans önerileri
        if accuracy_rate < 50:
            recommendations.append("Temel konuları tekrar gözden geçirmenizi öneriyoruz")
        elif accuracy_rate < 70:
            recommendations.append("Orta seviye konulara odaklanmanızı öneriyoruz")
        else:
            recommendations.append("Zorlu konulara geçebilirsiniz")
        
        # Hata kalıpları önerileri
        if error_analysis.get("total_patterns", 0) > 0:
            recommendations.append("Hata kalıplarınızı incelemenizi öneriyoruz")
        
        # Zayıf yönler için öneriler
        if strengths_weaknesses.get("weaknesses"):
            weak_topics = [w["topic"] for w in strengths_weaknesses["weaknesses"]]
            recommendations.append(f"Şu konulara odaklanın: {', '.join(weak_topics[:3])}")
        
        return recommendations
    
    async def _get_student_performance_profile(
        self,
        db: AsyncSession,
        user_id: str,
        subject: Subject
    ) -> Optional[Dict[str, Any]]:
        """Öğrenci performans profilini al"""
        
        try:
            # Son 30 günlük performans
            cutoff_date = datetime.utcnow() - timedelta(days=30)
            
            result = await db.execute(
                select(StudentAttempt).where(
                    and_(
                        StudentAttempt.user_id == user_id,
                        StudentAttempt.subject == subject,
                        StudentAttempt.created_at >= cutoff_date
                    )
                )
            )
            attempts = result.scalars().all()
            
            if not attempts:
                return None
            
            # Profil hesapla
            total_attempts = len(attempts)
            correct_attempts = len([a for a in attempts if a.is_correct])
            accuracy = (correct_attempts / total_attempts) * 100
            avg_difficulty = sum(a.difficulty_level for a in attempts) / total_attempts
            avg_time = sum(a.time_spent for a in attempts) / total_attempts
            
            return {
                "accuracy": accuracy,
                "avg_difficulty": avg_difficulty,
                "avg_time": avg_time,
                "total_attempts": total_attempts
            }
            
        except Exception as e:
            logger.error(f"Error getting performance profile: {e}")
            return None
    
    async def _calculate_similarity_score(
        self,
        profile1: Dict[str, Any],
        profile2: Dict[str, Any]
    ) -> float:
        """İki profil arasındaki benzerlik skorunu hesapla"""
        
        # Ağırlıklı benzerlik hesaplama
        accuracy_diff = abs(profile1["accuracy"] - profile2["accuracy"]) / 100
        difficulty_diff = abs(profile1["avg_difficulty"] - profile2["avg_difficulty"]) / 5
        time_diff = abs(profile1["avg_time"] - profile2["avg_time"]) / 300  # 5 dakika normalize
        
        # Benzerlik skoru (0-1 arası)
        similarity = 1 - (accuracy_diff * 0.5 + difficulty_diff * 0.3 + time_diff * 0.2)
        
        return max(0, min(1, similarity))
    
    async def _analyze_trend_direction(self, accuracy_trend: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Trend yönünü analiz et"""
        
        if len(accuracy_trend) < 3:
            return {"direction": "insufficient_data", "slope": 0}
        
        # Son 3 günün ortalaması
        recent_avg = sum(day["accuracy"] for day in accuracy_trend[-3:]) / 3
        
        # İlk 3 günün ortalaması
        early_avg = sum(day["accuracy"] for day in accuracy_trend[:3]) / 3
        
        slope = recent_avg - early_avg
        
        if slope > 5:
            direction = "improving"
        elif slope < -5:
            direction = "declining"
        else:
            direction = "stable"
        
        return {
            "direction": direction,
            "slope": round(slope, 2),
            "recent_avg": round(recent_avg, 2),
            "early_avg": round(early_avg, 2)
        }


# Global analytics service instance
analytics_service = AnalyticsService()
