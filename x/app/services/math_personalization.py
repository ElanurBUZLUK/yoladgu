import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from collections import defaultdict, deque
import json

from app.models.math_profile import MathProfile
from app.models.question import Question

logger = logging.getLogger(__name__)


@dataclass
class LearningPreference:
    """Öğrenme tercihi"""
    topic_preferences: Dict[str, float]  # Konu tercihleri
    difficulty_preferences: Dict[str, float]  # Zorluk tercihleri
    question_type_preferences: Dict[str, float]  # Soru tipi tercihleri
    learning_style: str  # visual, auditory, kinesthetic, reading
    pace_preference: str  # slow, moderate, fast
    feedback_preference: str  # immediate, delayed, minimal


@dataclass
class PersonalizationContext:
    """Kişiselleştirme context'i"""
    time_of_day: int  # 0-23
    day_of_week: int  # 0-6
    session_duration: int  # dakika
    recent_performance: List[float]  # Son performans skorları
    current_mood: Optional[str] = None  # tired, focused, distracted
    learning_goals: List[str] = None  # Öğrenme hedefleri


class MathPersonalization:
    """Matematik kişiselleştirme servisi"""
    
    def __init__(self):
        self.config = {
            # Öğrenme tercihi parametreleri
            "preference_decay_rate": 0.95,  # Tercih azalma oranı
            "min_preference_weight": 0.1,  # Minimum tercih ağırlığı
            "max_preference_weight": 2.0,  # Maksimum tercih ağırlığı
            
            # Zorluk adaptasyonu parametreleri
            "difficulty_adaptation_rate": 0.1,  # Zorluk adaptasyon oranı
            "confidence_threshold": 0.8,  # Güven eşiği
            "frustration_threshold": 0.3,  # Frustrasyon eşiği
            
            # Konu tercihi parametreleri
            "topic_exploration_rate": 0.2,  # Konu keşif oranı
            "topic_exploitation_rate": 0.8,  # Konu sömürme oranı
            
            # Öğrenme stili parametreleri
            "style_adaptation_rate": 0.15,  # Stil adaptasyon oranı
            "style_exploration_rate": 0.1,  # Stil keşif oranı
            
            # Context-aware parametreleri
            "time_weight": 0.3,  # Zaman ağırlığı
            "performance_weight": 0.4,  # Performans ağırlığı
            "mood_weight": 0.2,  # Ruh hali ağırlığı
            "goal_weight": 0.1,  # Hedef ağırlığı
        }
        
        # Kullanıcı tercihleri cache'i
        self.user_preferences: Dict[str, LearningPreference] = {}
        self.preference_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
    
    async def learn_user_preferences(
        self, 
        user_id: str,
        profile: MathProfile,
        question: Question,
        user_answer: str,
        is_correct: bool,
        response_time: float,
        context: PersonalizationContext
    ) -> Dict[str, Any]:
        """Kullanıcı tercihlerini öğren"""
        
        # Mevcut tercihleri al veya oluştur
        preferences = self.user_preferences.get(user_id, self._create_default_preferences())
        
        # Konu tercihlerini güncelle
        self._update_topic_preferences(preferences, question, is_correct, response_time)
        
        # Zorluk tercihlerini güncelle
        self._update_difficulty_preferences(preferences, question, is_correct, response_time)
        
        # Soru tipi tercihlerini güncelle
        self._update_question_type_preferences(preferences, question, is_correct, response_time)
        
        # Öğrenme stilini güncelle
        self._update_learning_style(preferences, question, user_answer, response_time, context)
        
        # Hız tercihlerini güncelle
        self._update_pace_preferences(preferences, response_time, context)
        
        # Feedback tercihlerini güncelle
        self._update_feedback_preferences(preferences, context)
        
        # Tercihleri kaydet
        self.user_preferences[user_id] = preferences
        self.preference_history[user_id].append({
            "timestamp": datetime.utcnow().isoformat(),
            "preferences": self._preferences_to_dict(preferences),
            "context": self._context_to_dict(context)
        })
        
        return {
            "updated_preferences": self._preferences_to_dict(preferences),
            "learning_insights": self._generate_learning_insights(preferences, context),
            "recommendations": self._generate_personalization_recommendations(preferences, context)
        }
    
    async def personalize_question_selection(
        self, 
        user_id: str,
        profile: MathProfile,
        candidate_questions: List[Question],
        context: PersonalizationContext
    ) -> List[Tuple[Question, float]]:
        """Soru seçimini kişiselleştir"""
        
        # Kullanıcı tercihlerini al
        preferences = self.user_preferences.get(user_id, self._create_default_preferences())
        
        # Aday soruları skorla
        scored_questions = []
        for question in candidate_questions:
            personalization_score = self._calculate_personalization_score(
                question, preferences, context, profile
            )
            scored_questions.append((question, personalization_score))
        
        # Skora göre sırala
        scored_questions.sort(key=lambda x: x[1], reverse=True)
        
        return scored_questions
    
    async def adapt_difficulty(
        self, 
        user_id: str,
        profile: MathProfile,
        current_difficulty: float,
        recent_performance: List[float],
        context: PersonalizationContext
    ) -> float:
        """Zorluğu kişiselleştirilmiş şekilde adapte et"""
        
        preferences = self.user_preferences.get(user_id, self._create_default_preferences())
        
        # Performans analizi
        avg_performance = np.mean(recent_performance) if recent_performance else 0.5
        performance_trend = self._calculate_performance_trend(recent_performance)
        
        # Zorluk tercihlerini analiz et
        difficulty_pref = preferences.difficulty_preferences
        preferred_difficulty = self._get_preferred_difficulty(difficulty_pref)
        
        # Context-aware ayarlamalar
        time_adjustment = self._calculate_time_adjustment(context.time_of_day)
        mood_adjustment = self._calculate_mood_adjustment(context.current_mood)
        goal_adjustment = self._calculate_goal_adjustment(context.learning_goals)
        
        # Yeni zorluk hesapla
        base_difficulty = current_difficulty
        performance_adjustment = (avg_performance - 0.5) * self.config["difficulty_adaptation_rate"]
        preference_adjustment = (preferred_difficulty - current_difficulty) * 0.1
        
        new_difficulty = (
            base_difficulty + 
            performance_adjustment + 
            preference_adjustment +
            time_adjustment +
            mood_adjustment +
            goal_adjustment
        )
        
        # Sınırları uygula
        new_difficulty = max(0.0, min(5.0, new_difficulty))
        
        return new_difficulty
    
    async def generate_personalized_feedback(
        self, 
        user_id: str,
        question: Question,
        user_answer: str,
        is_correct: bool,
        context: PersonalizationContext
    ) -> Dict[str, Any]:
        """Kişiselleştirilmiş feedback oluştur"""
        
        preferences = self.user_preferences.get(user_id, self._create_default_preferences())
        
        # Feedback stilini belirle
        feedback_style = self._determine_feedback_style(preferences, context)
        
        # Feedback içeriğini oluştur
        feedback_content = self._generate_feedback_content(
            question, user_answer, is_correct, feedback_style, preferences
        )
        
        # Motivasyon mesajı ekle
        motivation_message = self._generate_motivation_message(
            is_correct, context, preferences
        )
        
        return {
            "feedback_style": feedback_style,
            "feedback_content": feedback_content,
            "motivation_message": motivation_message,
            "personalization_insights": self._generate_feedback_insights(preferences, context)
        }
    
    async def get_learning_recommendations(
        self, 
        user_id: str,
        profile: MathProfile,
        context: PersonalizationContext
    ) -> Dict[str, Any]:
        """Kişiselleştirilmiş öğrenme önerileri al"""
        
        preferences = self.user_preferences.get(user_id, self._create_default_preferences())
        
        # Konu önerileri
        topic_recommendations = self._generate_topic_recommendations(preferences, profile, context)
        
        # Zorluk önerileri
        difficulty_recommendations = self._generate_difficulty_recommendations(preferences, profile, context)
        
        # Öğrenme stili önerileri
        style_recommendations = self._generate_style_recommendations(preferences, context)
        
        # Zaman önerileri
        timing_recommendations = self._generate_timing_recommendations(preferences, context)
        
        return {
            "topic_recommendations": topic_recommendations,
            "difficulty_recommendations": difficulty_recommendations,
            "style_recommendations": style_recommendations,
            "timing_recommendations": timing_recommendations,
            "overall_recommendations": self._generate_overall_recommendations(
                preferences, profile, context
            )
        }
    
    def _create_default_preferences(self) -> LearningPreference:
        """Varsayılan öğrenme tercihleri oluştur"""
        
        return LearningPreference(
            topic_preferences={
                "algebra": 1.0,
                "geometry": 1.0,
                "calculus": 1.0,
                "arithmetic": 1.0,
                "statistics": 1.0
            },
            difficulty_preferences={
                "easy": 1.0,
                "medium": 1.0,
                "hard": 1.0
            },
            question_type_preferences={
                "multiple_choice": 1.0,
                "open_ended": 1.0,
                "fill_blank": 1.0,
                "true_false": 1.0
            },
            learning_style="visual",
            pace_preference="moderate",
            feedback_preference="immediate"
        )
    
    def _update_topic_preferences(
        self, 
        preferences: LearningPreference,
        question: Question,
        is_correct: bool,
        response_time: float
    ):
        """Konu tercihlerini güncelle"""
        
        topic = question.topic_category
        if topic not in preferences.topic_preferences:
            preferences.topic_preferences[topic] = 1.0
        
        # Performans bazlı güncelleme
        performance_score = 1.0 if is_correct else 0.0
        time_score = min(1.0, 120.0 / max(response_time, 1.0))  # Normalize response time
        
        combined_score = (performance_score + time_score) / 2.0
        
        # Tercihi güncelle
        current_pref = preferences.topic_preferences[topic]
        new_pref = current_pref * self.config["preference_decay_rate"] + combined_score * (1 - self.config["preference_decay_rate"])
        
        preferences.topic_preferences[topic] = max(
            self.config["min_preference_weight"],
            min(self.config["max_preference_weight"], new_pref)
        )
    
    def _update_difficulty_preferences(
        self, 
        preferences: LearningPreference,
        question: Question,
        is_correct: bool,
        response_time: float
    ):
        """Zorluk tercihlerini güncelle"""
        
        difficulty = question.estimated_difficulty or question.difficulty_level
        
        # Zorluk kategorisini belirle
        if difficulty <= 1.5:
            difficulty_category = "easy"
        elif difficulty <= 3.5:
            difficulty_category = "medium"
        else:
            difficulty_category = "hard"
        
        # Performans bazlı güncelleme
        performance_score = 1.0 if is_correct else 0.0
        time_score = min(1.0, 120.0 / max(response_time, 1.0))
        
        combined_score = (performance_score + time_score) / 2.0
        
        # Tercihi güncelle
        current_pref = preferences.difficulty_preferences[difficulty_category]
        new_pref = current_pref * self.config["preference_decay_rate"] + combined_score * (1 - self.config["preference_decay_rate"])
        
        preferences.difficulty_preferences[difficulty_category] = max(
            self.config["min_preference_weight"],
            min(self.config["max_preference_weight"], new_pref)
        )
    
    def _update_question_type_preferences(
        self, 
        preferences: LearningPreference,
        question: Question,
        is_correct: bool,
        response_time: float
    ):
        """Soru tipi tercihlerini güncelle"""
        
        question_type = question.question_type.value
        
        # Performans bazlı güncelleme
        performance_score = 1.0 if is_correct else 0.0
        time_score = min(1.0, 120.0 / max(response_time, 1.0))
        
        combined_score = (performance_score + time_score) / 2.0
        
        # Tercihi güncelle
        current_pref = preferences.question_type_preferences[question_type]
        new_pref = current_pref * self.config["preference_decay_rate"] + combined_score * (1 - self.config["preference_decay_rate"])
        
        preferences.question_type_preferences[question_type] = max(
            self.config["min_preference_weight"],
            min(self.config["max_preference_weight"], new_pref)
        )
    
    def _update_learning_style(
        self, 
        preferences: LearningPreference,
        question: Question,
        user_answer: str,
        response_time: float,
        context: PersonalizationContext
    ):
        """Öğrenme stilini güncelle"""
        
        # Cevap uzunluğu ve detayına göre stil analizi
        answer_length = len(user_answer)
        answer_detail = self._analyze_answer_detail(user_answer)
        
        # Stil belirleme
        if answer_length > 50 and answer_detail > 0.7:
            detected_style = "reading"
        elif answer_length > 20:
            detected_style = "kinesthetic"
        elif response_time > 60:
            detected_style = "auditory"
        else:
            detected_style = "visual"
        
        # Stil güncelleme (yavaş adaptasyon)
        if detected_style != preferences.learning_style:
            # Rastgele stil değişimi (keşif)
            if np.random.random() < self.config["style_exploration_rate"]:
                preferences.learning_style = detected_style
    
    def _update_pace_preferences(
        self, 
        preferences: LearningPreference,
        response_time: float,
        context: PersonalizationContext
    ):
        """Hız tercihlerini güncelle"""
        
        # Ortalama yanıt süresine göre hız belirleme
        if response_time < 30:
            detected_pace = "fast"
        elif response_time < 90:
            detected_pace = "moderate"
        else:
            detected_pace = "slow"
        
        # Hız güncelleme
        if detected_pace != preferences.pace_preference:
            preferences.pace_preference = detected_pace
    
    def _update_feedback_preferences(
        self, 
        preferences: LearningPreference,
        context: PersonalizationContext
    ):
        """Feedback tercihlerini güncelle"""
        
        # Context'e göre feedback tercihi
        if context.session_duration > 60:  # Uzun oturum
            preferences.feedback_preference = "minimal"
        elif context.current_mood == "focused":
            preferences.feedback_preference = "immediate"
        else:
            preferences.feedback_preference = "delayed"
    
    def _calculate_personalization_score(
        self, 
        question: Question,
        preferences: LearningPreference,
        context: PersonalizationContext,
        profile: MathProfile
    ) -> float:
        """Kişiselleştirme skoru hesapla"""
        
        score = 1.0
        
        # Konu tercihi skoru
        topic_score = preferences.topic_preferences.get(question.topic_category, 1.0)
        score *= topic_score
        
        # Zorluk tercihi skoru
        difficulty = question.estimated_difficulty or question.difficulty_level
        if difficulty <= 1.5:
            difficulty_category = "easy"
        elif difficulty <= 3.5:
            difficulty_category = "medium"
        else:
            difficulty_category = "hard"
        
        difficulty_score = preferences.difficulty_preferences.get(difficulty_category, 1.0)
        score *= difficulty_score
        
        # Soru tipi tercihi skoru
        type_score = preferences.question_type_preferences.get(question.question_type.value, 1.0)
        score *= type_score
        
        # Context-aware ayarlamalar
        time_score = self._calculate_time_preference_score(context.time_of_day, preferences)
        score *= time_score
        
        # Performans bazlı ayarlama
        performance_score = self._calculate_performance_preference_score(context.recent_performance, profile)
        score *= performance_score
        
        return score
    
    def _calculate_time_preference_score(self, time_of_day: int, preferences: LearningPreference) -> float:
        """Zaman tercihi skoru hesapla"""
        
        # Sabah (6-12): Daha zor konular
        if 6 <= time_of_day < 12:
            return 1.2
        # Öğleden sonra (12-18): Orta zorluk
        elif 12 <= time_of_day < 18:
            return 1.0
        # Akşam (18-24): Kolay konular
        elif 18 <= time_of_day < 24:
            return 0.8
        # Gece (0-6): Çok kolay konular
        else:
            return 0.6
    
    def _calculate_performance_preference_score(self, recent_performance: List[float], profile: MathProfile) -> float:
        """Performans tercihi skoru hesapla"""
        
        if not recent_performance:
            return 1.0
        
        avg_performance = np.mean(recent_performance)
        
        # Düşük performans: Daha kolay sorular
        if avg_performance < 0.4:
            return 0.7
        # Yüksek performans: Daha zor sorular
        elif avg_performance > 0.8:
            return 1.3
        else:
            return 1.0
    
    def _calculate_performance_trend(self, recent_performance: List[float]) -> float:
        """Performans trendi hesapla"""
        
        if len(recent_performance) < 2:
            return 0.0
        
        # Linear regression slope
        x = np.arange(len(recent_performance))
        y = np.array(recent_performance)
        
        slope = np.polyfit(x, y, 1)[0]
        return slope
    
    def _get_preferred_difficulty(self, difficulty_preferences: Dict[str, float]) -> float:
        """Tercih edilen zorluğu hesapla"""
        
        # Ağırlıklı ortalama
        total_weight = 0.0
        weighted_sum = 0.0
        
        for category, weight in difficulty_preferences.items():
            if category == "easy":
                difficulty_value = 1.0
            elif category == "medium":
                difficulty_value = 3.0
            else:  # hard
                difficulty_value = 4.5
            
            weighted_sum += difficulty_value * weight
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 2.5
    
    def _calculate_time_adjustment(self, time_of_day: int) -> float:
        """Zaman bazlı ayarlama"""
        
        # Sabah: Daha zor
        if 6 <= time_of_day < 12:
            return 0.2
        # Öğleden sonra: Normal
        elif 12 <= time_of_day < 18:
            return 0.0
        # Akşam: Daha kolay
        elif 18 <= time_of_day < 24:
            return -0.2
        # Gece: Çok kolay
        else:
            return -0.3
    
    def _calculate_mood_adjustment(self, mood: Optional[str]) -> float:
        """Ruh hali bazlı ayarlama"""
        
        if mood == "tired":
            return -0.3
        elif mood == "focused":
            return 0.2
        elif mood == "distracted":
            return -0.2
        else:
            return 0.0
    
    def _calculate_goal_adjustment(self, learning_goals: Optional[List[str]]) -> float:
        """Hedef bazlı ayarlama"""
        
        if not learning_goals:
            return 0.0
        
        # Hedef türüne göre ayarlama
        if "mastery" in learning_goals:
            return 0.3
        elif "speed" in learning_goals:
            return -0.1
        elif "accuracy" in learning_goals:
            return 0.1
        else:
            return 0.0
    
    def _analyze_answer_detail(self, answer: str) -> float:
        """Cevap detayını analiz et"""
        
        # Basit detay analizi
        words = answer.split()
        if len(words) < 3:
            return 0.0
        
        # Matematiksel ifadeler
        math_symbols = sum(1 for char in answer if char in "+-*/=()[]{}")
        detail_score = min(1.0, math_symbols / len(words))
        
        return detail_score
    
    def _determine_feedback_style(
        self, 
        preferences: LearningPreference,
        context: PersonalizationContext
    ) -> str:
        """Feedback stilini belirle"""
        
        # Öğrenme stiline göre feedback
        if preferences.learning_style == "visual":
            return "visual_detailed"
        elif preferences.learning_style == "auditory":
            return "verbal_explanation"
        elif preferences.learning_style == "kinesthetic":
            return "step_by_step"
        else:  # reading
            return "text_detailed"
    
    def _generate_feedback_content(
        self, 
        question: Question,
        user_answer: str,
        is_correct: bool,
        feedback_style: str,
        preferences: LearningPreference
    ) -> str:
        """Feedback içeriği oluştur"""
        
        if is_correct:
            if feedback_style == "visual_detailed":
                return "Excellent! Your solution shows clear understanding. Here's a visual representation of the concept."
            elif feedback_style == "verbal_explanation":
                return "Great job! You've mastered this concept. Let me explain why this approach works."
            elif feedback_style == "step_by_step":
                return "Perfect! Your step-by-step approach was correct. Here's how to apply this method to similar problems."
            else:
                return "Well done! Your answer is correct. Here's a detailed explanation of the underlying concepts."
        else:
            if feedback_style == "visual_detailed":
                return "Let's work through this step by step with visual aids to understand the concept better."
            elif feedback_style == "verbal_explanation":
                return "Let me explain the correct approach and why your answer needs adjustment."
            elif feedback_style == "step_by_step":
                return "Let's break this down into smaller steps to identify where the confusion occurred."
            else:
                return "Here's a detailed explanation of the correct solution and the concepts involved."
    
    def _generate_motivation_message(
        self, 
        is_correct: bool,
        context: PersonalizationContext,
        preferences: LearningPreference
    ) -> str:
        """Motivasyon mesajı oluştur"""
        
        if is_correct:
            messages = [
                "Keep up the great work!",
                "You're making excellent progress!",
                "Your understanding is growing stronger!",
                "Fantastic effort!",
                "You're on the right track!"
            ]
        else:
            messages = [
                "Don't worry, learning takes time!",
                "Every mistake is a learning opportunity!",
                "You're getting closer to the solution!",
                "Keep trying, you'll get it!",
                "Learning is a journey, not a destination!"
            ]
        
        return np.random.choice(messages)
    
    def _generate_topic_recommendations(
        self, 
        preferences: LearningPreference,
        profile: MathProfile,
        context: PersonalizationContext
    ) -> List[Dict[str, Any]]:
        """Konu önerileri oluştur"""
        
        recommendations = []
        
        # En çok tercih edilen konular
        sorted_topics = sorted(
            preferences.topic_preferences.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        for topic, preference in sorted_topics[:3]:
            recommendations.append({
                "topic": topic,
                "preference_score": preference,
                "reason": f"High preference score ({preference:.2f})",
                "suggested_difficulty": self._get_topic_difficulty(topic, profile)
            })
        
        return recommendations
    
    def _generate_difficulty_recommendations(
        self, 
        preferences: LearningPreference,
        profile: MathProfile,
        context: PersonalizationContext
    ) -> List[Dict[str, Any]]:
        """Zorluk önerileri oluştur"""
        
        recommendations = []
        
        # Tercih edilen zorluk seviyeleri
        for difficulty, preference in preferences.difficulty_preferences.items():
            recommendations.append({
                "difficulty": difficulty,
                "preference_score": preference,
                "reason": f"Preferred difficulty level",
                "suggested_range": self._get_difficulty_range(difficulty)
            })
        
        return recommendations
    
    def _generate_style_recommendations(
        self, 
        preferences: LearningPreference,
        context: PersonalizationContext
    ) -> List[Dict[str, Any]]:
        """Stil önerileri oluştur"""
        
        style_tips = {
            "visual": [
                "Use diagrams and charts to visualize problems",
                "Draw out the problem before solving",
                "Use color coding for different concepts"
            ],
            "auditory": [
                "Read problems aloud",
                "Explain your solution to someone else",
                "Use verbal mnemonics"
            ],
            "kinesthetic": [
                "Use physical objects to model problems",
                "Write out each step clearly",
                "Practice with hands-on activities"
            ],
            "reading": [
                "Read problems carefully multiple times",
                "Take detailed notes",
                "Summarize concepts in your own words"
            ]
        }
        
        current_style = preferences.learning_style
        tips = style_tips.get(current_style, [])
        
        return [{"tip": tip, "style": current_style} for tip in tips]
    
    def _generate_timing_recommendations(
        self, 
        preferences: LearningPreference,
        context: PersonalizationContext
    ) -> List[Dict[str, Any]]:
        """Zaman önerileri oluştur"""
        
        recommendations = []
        
        # Optimal çalışma zamanları
        if context.time_of_day < 12:
            recommendations.append({
                "timing": "morning",
                "reason": "Peak cognitive performance",
                "suggestion": "Focus on challenging topics"
            })
        elif context.time_of_day < 18:
            recommendations.append({
                "timing": "afternoon",
                "reason": "Good energy levels",
                "suggestion": "Balanced mix of topics"
            })
        else:
            recommendations.append({
                "timing": "evening",
                "reason": "Lower energy levels",
                "suggestion": "Review and practice"
            })
        
        return recommendations
    
    def _generate_overall_recommendations(
        self, 
        preferences: LearningPreference,
        profile: MathProfile,
        context: PersonalizationContext
    ) -> List[str]:
        """Genel öneriler oluştur"""
        
        recommendations = []
        
        # Performans bazlı öneriler
        if context.recent_performance:
            avg_performance = np.mean(context.recent_performance)
            if avg_performance < 0.5:
                recommendations.append("Consider reviewing fundamental concepts")
            elif avg_performance > 0.8:
                recommendations.append("Ready for more challenging problems")
        
        # Stil bazlı öneriler
        if preferences.learning_style == "visual":
            recommendations.append("Try using visual aids for complex problems")
        elif preferences.learning_style == "kinesthetic":
            recommendations.append("Practice with hands-on problem solving")
        
        # Hız bazlı öneriler
        if preferences.pace_preference == "fast":
            recommendations.append("Focus on accuracy over speed")
        elif preferences.pace_preference == "slow":
            recommendations.append("Gradually increase your problem-solving pace")
        
        return recommendations
    
    def _get_topic_difficulty(self, topic: str, profile: MathProfile) -> float:
        """Konu zorluğunu hesapla"""
        
        # Basit konu zorluk mapping'i
        topic_difficulties = {
            "arithmetic": 1.5,
            "algebra": 2.5,
            "geometry": 3.0,
            "calculus": 4.0,
            "statistics": 3.5
        }
        
        base_difficulty = topic_difficulties.get(topic, 2.5)
        return max(0.0, min(5.0, base_difficulty + profile.global_skill - 2.5))
    
    def _get_difficulty_range(self, difficulty_category: str) -> Tuple[float, float]:
        """Zorluk aralığını hesapla"""
        
        if difficulty_category == "easy":
            return (0.0, 2.0)
        elif difficulty_category == "medium":
            return (1.5, 3.5)
        else:  # hard
            return (3.0, 5.0)
    
    def _preferences_to_dict(self, preferences: LearningPreference) -> Dict[str, Any]:
        """Tercihleri dict'e çevir"""
        
        return {
            "topic_preferences": preferences.topic_preferences,
            "difficulty_preferences": preferences.difficulty_preferences,
            "question_type_preferences": preferences.question_type_preferences,
            "learning_style": preferences.learning_style,
            "pace_preference": preferences.pace_preference,
            "feedback_preference": preferences.feedback_preference
        }
    
    def _context_to_dict(self, context: PersonalizationContext) -> Dict[str, Any]:
        """Context'i dict'e çevir"""
        
        return {
            "time_of_day": context.time_of_day,
            "day_of_week": context.day_of_week,
            "session_duration": context.session_duration,
            "recent_performance": context.recent_performance,
            "current_mood": context.current_mood,
            "learning_goals": context.learning_goals
        }
    
    def _generate_learning_insights(
        self, 
        preferences: LearningPreference,
        context: PersonalizationContext
    ) -> List[str]:
        """Öğrenme içgörüleri oluştur"""
        
        insights = []
        
        # En çok tercih edilen konu
        top_topic = max(preferences.topic_preferences.items(), key=lambda x: x[1])
        insights.append(f"Your strongest preference is for {top_topic[0]} topics")
        
        # Öğrenme stili
        insights.append(f"You prefer {preferences.learning_style} learning style")
        
        # Hız tercihi
        insights.append(f"Your preferred pace is {preferences.pace_preference}")
        
        return insights
    
    def _generate_personalization_recommendations(
        self, 
        preferences: LearningPreference,
        context: PersonalizationContext
    ) -> List[str]:
        """Kişiselleştirme önerileri oluştur"""
        
        recommendations = []
        
        # Konu çeşitliliği önerisi
        if len(preferences.topic_preferences) < 3:
            recommendations.append("Try exploring different mathematical topics")
        
        # Zorluk çeşitliliği önerisi
        if len(preferences.difficulty_preferences) < 2:
            recommendations.append("Consider trying problems of varying difficulty levels")
        
        return recommendations
    
    def _generate_feedback_insights(
        self, 
        preferences: LearningPreference,
        context: PersonalizationContext
    ) -> List[str]:
        """Feedback içgörüleri oluştur"""
        
        insights = []
        
        # Feedback tercihi
        insights.append(f"Feedback style: {preferences.feedback_preference}")
        
        # Öğrenme stili
        insights.append(f"Learning style: {preferences.learning_style}")
        
        return insights


# Global instance
math_personalization = MathPersonalization()
