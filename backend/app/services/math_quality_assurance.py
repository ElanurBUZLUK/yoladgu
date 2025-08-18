import re
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass

from app.models.math_profile import MathProfile
from app.models.question import Question

logger = logging.getLogger(__name__)


@dataclass
class QualityCheckResult:
    """Kalite kontrol sonucu"""
    passed: bool
    score: float  # 0.0 - 1.0
    issues: List[str]
    warnings: List[str]
    recommendations: List[str]


class MathQualityAssurance:
    """Matematik kalite güvencesi servisi"""
    
    def __init__(self):
        self.config = {
            # Zorluk sınırları
            "min_difficulty": 0.0,
            "max_difficulty": 5.0,
            "difficulty_step": 0.1,
            
            # Burnout koruması
            "max_questions_per_session": 50,
            "max_wrong_streak": 5,
            "recovery_threshold": 0.3,
            "session_timeout_minutes": 120,
            
            # Duplicate/Leakage koruması
            "similarity_threshold": 0.8,
            "recent_questions_window": 20,
            "duplicate_check_enabled": True,
            
            # Timeout/Guess koruması
            "min_response_time": 5,  # saniye
            "max_response_time": 600,  # saniye
            "guess_detection_threshold": 0.1,
            
            # Kısmi puan hesaplama
            "partial_credit_enabled": True,
            "min_partial_credit": 0.1,
            "max_partial_credit": 0.9,
            
            # Kalite skorları
            "content_quality_weight": 0.3,
            "difficulty_appropriateness_weight": 0.25,
            "diversity_weight": 0.2,
            "freshness_weight": 0.15,
            "accessibility_weight": 0.1,
        }
    
    def validate_question_quality(
        self, 
        question: Question, 
        user_profile: MathProfile,
        recent_questions: List[Question]
    ) -> QualityCheckResult:
        """Soru kalitesini doğrula"""
        
        issues = []
        warnings = []
        recommendations = []
        score = 1.0
        
        # İçerik kalitesi kontrolü
        content_score, content_issues = self._check_content_quality(question)
        score *= content_score
        issues.extend(content_issues)
        
        # Zorluk uygunluğu kontrolü
        difficulty_score, difficulty_issues = self._check_difficulty_appropriateness(
            question, user_profile
        )
        score *= difficulty_score
        issues.extend(difficulty_issues)
        
        # Çeşitlilik kontrolü
        diversity_score, diversity_issues = self._check_diversity(question, recent_questions)
        score *= diversity_score
        issues.extend(diversity_issues)
        
        # Tazelik kontrolü
        freshness_score, freshness_issues = self._check_freshness(question, recent_questions)
        score *= freshness_score
        issues.extend(freshness_issues)
        
        # Erişilebilirlik kontrolü
        accessibility_score, accessibility_issues = self._check_accessibility(question)
        score *= accessibility_score
        issues.extend(accessibility_issues)
        
        # Duplicate kontrolü
        if self.config["duplicate_check_enabled"]:
            duplicate_score, duplicate_issues = self._check_duplicates(question, recent_questions)
            score *= duplicate_score
            issues.extend(duplicate_issues)
        
        # Öneriler oluştur
        recommendations = self._generate_quality_recommendations(score, issues)
        
        return QualityCheckResult(
            passed=score >= 0.7,  # %70 üzeri geçer
            score=score,
            issues=issues,
            warnings=warnings,
            recommendations=recommendations
        )
    
    def validate_user_session(
        self, 
        user_profile: MathProfile,
        session_data: Dict[str, Any]
    ) -> QualityCheckResult:
        """Kullanıcı oturumunu doğrula"""
        
        issues = []
        warnings = []
        recommendations = []
        score = 1.0
        
        # Burnout kontrolü
        burnout_score, burnout_issues = self._check_burnout_risk(user_profile, session_data)
        score *= burnout_score
        issues.extend(burnout_issues)
        
        # Zorluk sınırları kontrolü
        difficulty_score, difficulty_issues = self._check_difficulty_bounds(user_profile)
        score *= difficulty_score
        issues.extend(difficulty_issues)
        
        # Oturum süresi kontrolü
        session_score, session_issues = self._check_session_duration(session_data)
        score *= session_score
        issues.extend(session_issues)
        
        # Öneriler oluştur
        recommendations = self._generate_session_recommendations(score, issues)
        
        return QualityCheckResult(
            passed=score >= 0.8,  # %80 üzeri geçer
            score=score,
            issues=issues,
            warnings=warnings,
            recommendations=recommendations
        )
    
    def validate_answer_quality(
        self, 
        student_answer: str,
        question: Question,
        response_time: float,
        session_context: Dict[str, Any]
    ) -> QualityCheckResult:
        """Cevap kalitesini doğrula"""
        
        issues = []
        warnings = []
        recommendations = []
        score = 1.0
        
        # Cevap uzunluğu kontrolü
        length_score, length_issues = self._check_answer_length(student_answer, question)
        score *= length_score
        issues.extend(length_issues)
        
        # Yanıt süresi kontrolü
        time_score, time_issues = self._check_response_time(response_time, question)
        score *= time_score
        issues.extend(time_issues)
        
        # Tahmin tespiti
        guess_score, guess_issues = self._detect_guessing(student_answer, response_time, question)
        score *= guess_score
        issues.extend(guess_issues)
        
        # İçerik uygunluğu kontrolü
        content_score, content_issues = self._check_answer_content(student_answer, question)
        score *= content_score
        issues.extend(content_issues)
        
        # Öneriler oluştur
        recommendations = self._generate_answer_recommendations(score, issues)
        
        return QualityCheckResult(
            passed=score >= 0.6,  # %60 üzeri geçer
            score=score,
            issues=issues,
            warnings=warnings,
            recommendations=recommendations
        )
    
    def calculate_partial_credit(
        self, 
        student_answer: str,
        correct_answer: str,
        question: Question,
        response_time: float
    ) -> float:
        """Kısmi puan hesapla"""
        
        if not self.config["partial_credit_enabled"]:
            return 1.0 if student_answer == correct_answer else 0.0
        
        # Temel doğruluk kontrolü
        if student_answer.strip().lower() == correct_answer.strip().lower():
            return 1.0
        
        # Kısmi puan hesaplama
        partial_score = 0.0
        
        # Sayısal yakınlık kontrolü
        numeric_score = self._calculate_numeric_similarity(student_answer, correct_answer)
        partial_score = max(partial_score, numeric_score)
        
        # String benzerlik kontrolü
        string_score = self._calculate_string_similarity(student_answer, correct_answer)
        partial_score = max(partial_score, string_score)
        
        # Adım bazlı kontrol
        step_score = self._calculate_step_based_credit(student_answer, correct_answer, question)
        partial_score = max(partial_score, step_score)
        
        # Zaman bazlı ayarlama
        time_adjustment = self._calculate_time_adjustment(response_time, question)
        partial_score *= time_adjustment
        
        # Sınırları uygula
        partial_score = max(
            self.config["min_partial_credit"],
            min(self.config["max_partial_credit"], partial_score)
        )
        
        return partial_score
    
    def _check_content_quality(self, question: Question) -> Tuple[float, List[str]]:
        """İçerik kalitesi kontrolü"""
        
        issues = []
        score = 1.0
        
        # İçerik uzunluğu kontrolü
        if len(question.content) < 10:
            issues.append("Question content is too short")
            score *= 0.7
        
        if len(question.content) > 500:
            issues.append("Question content is too long")
            score *= 0.8
        
        # İçerik netliği kontrolü
        if "?" not in question.content:
            issues.append("Question lacks clear question mark")
            score *= 0.9
        
        # Matematiksel ifade kontrolü
        if not self._contains_mathematical_content(question.content):
            issues.append("Question lacks mathematical content")
            score *= 0.6
        
        # Seçenek kontrolü (çoktan seçmeli için)
        if question.question_type.value == "multiple_choice":
            if not question.options or len(question.options) < 2:
                issues.append("Multiple choice question lacks sufficient options")
                score *= 0.5
        
        return score, issues
    
    def _check_difficulty_appropriateness(
        self, 
        question: Question, 
        user_profile: MathProfile
    ) -> Tuple[float, List[str]]:
        """Zorluk uygunluğu kontrolü"""
        
        issues = []
        score = 1.0
        
        # Zorluk sınırları kontrolü
        difficulty = question.estimated_difficulty or question.difficulty_level
        
        if difficulty < self.config["min_difficulty"]:
            issues.append(f"Question difficulty ({difficulty}) is below minimum")
            score *= 0.8
        
        if difficulty > self.config["max_difficulty"]:
            issues.append(f"Question difficulty ({difficulty}) is above maximum")
            score *= 0.8
        
        # Kullanıcı profiline uygunluk
        user_skill = user_profile.global_skill
        skill_gap = abs(difficulty - user_skill)
        
        if skill_gap > 2.0:
            issues.append(f"Question difficulty ({difficulty}) is too far from user skill ({user_skill})")
            score *= 0.7
        
        if skill_gap < 0.1:
            issues.append(f"Question difficulty ({difficulty}) is too close to user skill ({user_skill})")
            score *= 0.9
        
        return score, issues
    
    def _check_diversity(self, question: Question, recent_questions: List[Question]) -> Tuple[float, List[str]]:
        """Çeşitlilik kontrolü"""
        
        issues = []
        score = 1.0
        
        if not recent_questions:
            return score, issues
        
        # Konu çeşitliliği
        recent_topics = [q.topic_category for q in recent_questions[-5:]]
        if question.topic_category in recent_topics:
            issues.append("Question topic is too similar to recent questions")
            score *= 0.8
        
        # Zorluk çeşitliliği
        recent_difficulties = [q.estimated_difficulty or q.difficulty_level for q in recent_questions[-3:]]
        if recent_difficulties:
            avg_recent_difficulty = sum(recent_difficulties) / len(recent_difficulties)
            current_difficulty = question.estimated_difficulty or question.difficulty_level
            
            if abs(current_difficulty - avg_recent_difficulty) < 0.5:
                issues.append("Question difficulty is too similar to recent questions")
                score *= 0.9
        
        return score, issues
    
    def _check_freshness(self, question: Question, recent_questions: List[Question]) -> Tuple[float, List[str]]:
        """Tazelik kontrolü"""
        
        issues = []
        score = 1.0
        
        # Son görülme zamanı kontrolü
        if question.last_seen_at:
            days_since_seen = (datetime.utcnow() - question.last_seen_at).days
            if days_since_seen < 1:
                issues.append("Question was seen very recently")
                score *= 0.7
            elif days_since_seen < 7:
                issues.append("Question was seen recently")
                score *= 0.9
        
        # Freshness score kontrolü
        if question.freshness_score is not None:
            if question.freshness_score < 0.3:
                issues.append("Question has low freshness score")
                score *= 0.8
        
        return score, issues
    
    def _check_accessibility(self, question: Question) -> Tuple[float, List[str]]:
        """Erişilebilirlik kontrolü"""
        
        issues = []
        score = 1.0
        
        # İçerik okunabilirliği
        if len(question.content) > 200:
            issues.append("Question content may be too long for some users")
            score *= 0.9
        
        # Karmaşık matematiksel ifadeler
        complex_patterns = [r'\\frac\{.*?\}\{.*?\}', r'\\sqrt\{.*?\}', r'\\sum_\{.*?\}']
        for pattern in complex_patterns:
            if re.search(pattern, question.content):
                issues.append("Question contains complex mathematical expressions")
                score *= 0.95
        
        return score, issues
    
    def _check_duplicates(self, question: Question, recent_questions: List[Question]) -> Tuple[float, List[str]]:
        """Duplicate kontrolü"""
        
        issues = []
        score = 1.0
        
        for recent_q in recent_questions[-self.config["recent_questions_window"]:]:
            similarity = self._calculate_question_similarity(question, recent_q)
            if similarity > self.config["similarity_threshold"]:
                issues.append(f"Question is too similar to recent question (similarity: {similarity:.2f})")
                score *= 0.5
                break
        
        return score, issues
    
    def _check_burnout_risk(self, user_profile: MathProfile, session_data: Dict[str, Any]) -> Tuple[float, List[str]]:
        """Burnout risk kontrolü"""
        
        issues = []
        score = 1.0
        
        # Yanlış cevap serisi kontrolü
        if user_profile.streak_wrong >= self.config["max_wrong_streak"]:
            issues.append(f"User has {user_profile.streak_wrong} consecutive wrong answers")
            score *= 0.6
        
        # Doğruluk oranı kontrolü
        if user_profile.ema_accuracy < self.config["recovery_threshold"]:
            issues.append(f"User accuracy ({user_profile.ema_accuracy:.2f}) is below recovery threshold")
            score *= 0.7
        
        # Oturum sorusu sayısı kontrolü
        questions_answered = session_data.get("questions_answered", 0)
        if questions_answered >= self.config["max_questions_per_session"]:
            issues.append(f"User has answered {questions_answered} questions in this session")
            score *= 0.8
        
        return score, issues
    
    def _check_difficulty_bounds(self, user_profile: MathProfile) -> Tuple[float, List[str]]:
        """Zorluk sınırları kontrolü"""
        
        issues = []
        score = 1.0
        
        # Global skill sınırları
        if user_profile.global_skill < 0.0:
            issues.append("User global skill is below minimum")
            score *= 0.5
        
        if user_profile.global_skill > 5.0:
            issues.append("User global skill is above maximum")
            score *= 0.5
        
        # Difficulty factor sınırları
        if user_profile.difficulty_factor < 0.1:
            issues.append("User difficulty factor is too low")
            score *= 0.7
        
        if user_profile.difficulty_factor > 1.5:
            issues.append("User difficulty factor is too high")
            score *= 0.7
        
        return score, issues
    
    def _check_session_duration(self, session_data: Dict[str, Any]) -> Tuple[float, List[str]]:
        """Oturum süresi kontrolü"""
        
        issues = []
        score = 1.0
        
        session_start = session_data.get("session_start")
        if session_start:
            session_duration = datetime.utcnow() - session_start
            if session_duration > timedelta(minutes=self.config["session_timeout_minutes"]):
                issues.append(f"Session duration ({session_duration}) exceeds timeout")
                score *= 0.8
        
        return score, issues
    
    def _check_answer_length(self, student_answer: str, question: Question) -> Tuple[float, List[str]]:
        """Cevap uzunluğu kontrolü"""
        
        issues = []
        score = 1.0
        
        # Çok kısa cevap kontrolü
        if len(student_answer.strip()) < 1:
            issues.append("Answer is too short")
            score *= 0.3
        
        # Çok uzun cevap kontrolü
        if len(student_answer) > 1000:
            issues.append("Answer is too long")
            score *= 0.8
        
        return score, issues
    
    def _check_response_time(self, response_time: float, question: Question) -> Tuple[float, List[str]]:
        """Yanıt süresi kontrolü"""
        
        issues = []
        score = 1.0
        
        if response_time < self.config["min_response_time"]:
            issues.append(f"Response time ({response_time}s) is too fast")
            score *= 0.6
        
        if response_time > self.config["max_response_time"]:
            issues.append(f"Response time ({response_time}s) is too slow")
            score *= 0.8
        
        return score, issues
    
    def _detect_guessing(self, student_answer: str, response_time: float, question: Question) -> Tuple[float, List[str]]:
        """Tahmin tespiti"""
        
        issues = []
        score = 1.0
        
        # Çok hızlı cevap kontrolü
        if response_time < self.config["min_response_time"]:
            issues.append("Response time suggests guessing")
            score *= 0.5
        
        # Çok kısa cevap kontrolü
        if len(student_answer.strip()) < 2:
            issues.append("Very short answer suggests guessing")
            score *= 0.7
        
        return score, issues
    
    def _check_answer_content(self, student_answer: str, question: Question) -> Tuple[float, List[str]]:
        """Cevap içeriği kontrolü"""
        
        issues = []
        score = 1.0
        
        # Boş cevap kontrolü
        if not student_answer.strip():
            issues.append("Answer is empty")
            score *= 0.2
        
        # Uygunsuz içerik kontrolü
        inappropriate_words = ["idk", "dunno", "guess", "random"]
        if any(word in student_answer.lower() for word in inappropriate_words):
            issues.append("Answer contains inappropriate content")
            score *= 0.6
        
        return score, issues
    
    def _calculate_numeric_similarity(self, student_answer: str, correct_answer: str) -> float:
        """Sayısal benzerlik hesapla"""
        
        try:
            student_num = float(student_answer)
            correct_num = float(correct_answer)
            difference = abs(student_num - correct_num)
            
            if difference == 0:
                return 1.0
            elif difference <= 1:
                return 0.8
            elif difference <= 5:
                return 0.6
            elif difference <= 10:
                return 0.4
            else:
                return 0.2
        except ValueError:
            return 0.0
    
    def _calculate_string_similarity(self, student_answer: str, correct_answer: str) -> float:
        """String benzerlik hesapla"""
        
        # Basit string benzerlik
        student_clean = student_answer.strip().lower()
        correct_clean = correct_answer.strip().lower()
        
        if student_clean == correct_clean:
            return 1.0
        
        # Kısmi eşleşme
        if student_clean in correct_clean or correct_clean in student_clean:
            return 0.7
        
        # Karakter benzerliği
        common_chars = set(student_clean) & set(correct_clean)
        total_chars = set(student_clean) | set(correct_clean)
        
        if total_chars:
            return len(common_chars) / len(total_chars)
        
        return 0.0
    
    def _calculate_step_based_credit(self, student_answer: str, correct_answer: str, question: Question) -> float:
        """Adım bazlı kısmi puan hesapla"""
        
        # Basit adım kontrolü (gerçek implementasyonda daha karmaşık olacak)
        if "solve for x" in question.content.lower():
            if "=" in student_answer and "x" in student_answer:
                return 0.5
            if "x=" in student_answer:
                return 0.7
        
        return 0.0
    
    def _calculate_time_adjustment(self, response_time: float, question: Question) -> float:
        """Zaman bazlı ayarlama"""
        
        # Normal yanıt süresi (60-120 saniye)
        if 30 <= response_time <= 180:
            return 1.0
        elif response_time < 30:
            return 0.8  # Çok hızlı
        elif response_time > 300:
            return 0.9  # Çok yavaş
        else:
            return 0.95
    
    def _calculate_question_similarity(self, question1: Question, question2: Question) -> float:
        """Soru benzerliği hesapla"""
        
        # İçerik benzerliği
        content_similarity = self._calculate_string_similarity(question1.content, question2.content)
        
        # Konu benzerliği
        topic_similarity = 1.0 if question1.topic_category == question2.topic_category else 0.0
        
        # Zorluk benzerliği
        diff1 = question1.estimated_difficulty or question1.difficulty_level
        diff2 = question2.estimated_difficulty or question2.difficulty_level
        difficulty_similarity = 1.0 - (abs(diff1 - diff2) / 5.0)
        
        # Ağırlıklı ortalama
        return (content_similarity * 0.6 + topic_similarity * 0.3 + difficulty_similarity * 0.1)
    
    def _contains_mathematical_content(self, content: str) -> bool:
        """Matematiksel içerik kontrolü"""
        
        math_patterns = [
            r'\d+',  # Sayılar
            r'[+\-*/=]',  # Operatörler
            r'[a-zA-Z]\s*=',  # Değişkenler
            r'\(.*\)',  # Parantezler
            r'[²³]',  # Üsler
            r'√',  # Kök
        ]
        
        return any(re.search(pattern, content) for pattern in math_patterns)
    
    def _generate_quality_recommendations(self, score: float, issues: List[str]) -> List[str]:
        """Kalite önerileri oluştur"""
        
        recommendations = []
        
        if score < 0.7:
            recommendations.append("Consider reviewing question content for clarity")
            recommendations.append("Check difficulty appropriateness for target users")
        
        if any("similar" in issue.lower() for issue in issues):
            recommendations.append("Add more diversity to question selection")
        
        if any("freshness" in issue.lower() for issue in issues):
            recommendations.append("Consider using newer questions")
        
        return recommendations
    
    def _generate_session_recommendations(self, score: float, issues: List[str]) -> List[str]:
        """Oturum önerileri oluştur"""
        
        recommendations = []
        
        if score < 0.8:
            recommendations.append("Consider taking a short break")
            recommendations.append("Review recent mistakes before continuing")
        
        if any("burnout" in issue.lower() for issue in issues):
            recommendations.append("Switch to easier questions for confidence building")
        
        if any("timeout" in issue.lower() for issue in issues):
            recommendations.append("Consider shorter study sessions")
        
        return recommendations
    
    def _generate_answer_recommendations(self, score: float, issues: List[str]) -> List[str]:
        """Cevap önerileri oluştur"""
        
        recommendations = []
        
        if score < 0.6:
            recommendations.append("Take more time to think through the problem")
            recommendations.append("Show your work step by step")
        
        if any("guessing" in issue.lower() for issue in issues):
            recommendations.append("Avoid random guessing, try to solve systematically")
        
        if any("short" in issue.lower() for issue in issues):
            recommendations.append("Provide more detailed answers")
        
        return recommendations


# Global instance
math_quality_assurance = MathQualityAssurance()
