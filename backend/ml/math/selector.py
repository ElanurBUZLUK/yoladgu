"""
Math Question Selector with IRT + Multi-Skill Elo Integration
Optimal question selection for maximizing learning gains
"""

import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import structlog

from .irt import IRTModel, ItemParams, StudentAbility, irt_model
from .multiskill_elo import MultiSkillElo, multiskill_elo

logger = structlog.get_logger()


@dataclass
class QuestionCandidate:
    """Candidate question for selection."""
    item_id: str
    item_params: ItemParams
    required_skills: List[str]
    skill_weights: Dict[str, float]
    elo_rating: float
    difficulty: float
    expected_probability: float
    learning_gain: float
    skill_diversity: float
    metadata: Dict[str, Any]


class MathQuestionSelector:
    """
    Intelligent question selector using IRT + Multi-Skill Elo.

    Katmanlar:
    - Çekirdek skorlayıcı: P(correct | user, item) ~ IRT + Elo
    - Learning gain: belirsizlik + weakness-based + skill diversity
    - Level matching: target probability aralığı
    - Bandit: exploration / exploitation (epsilon-greedy veya UCB)
    """

    def __init__(
        self,
        target_prob_low: float = 0.55,
        target_prob_high: float = 0.8,
        learning_gain_weight: float = 1.0,
        difficulty_balance_weight: float = 0.3,
        skill_diversity_weight: float = 0.2,
        exploration_strategy: str = "epsilon_greedy",  # "none" | "epsilon_greedy" | "ucb"
        epsilon: float = 0.1,
        top_k_explore: int = 5,
        ucb_c: float = 1.0,
        irt_model: IRTModel = irt_model,
        multiskill_elo: MultiSkillElo = multiskill_elo,
        irt_weight: float = 0.6,
        elo_weight: float = 0.4,
    ) -> None:
        self.target_prob_low = target_prob_low
        self.target_prob_high = target_prob_high
        self.learning_gain_weight = learning_gain_weight
        self.difficulty_balance_weight = difficulty_balance_weight
        self.skill_diversity_weight = skill_diversity_weight

        self.exploration_strategy = exploration_strategy
        self.epsilon = epsilon
        self.top_k_explore = top_k_explore
        self.ucb_c = ucb_c

        self.irt_model = irt_model
        self.elo_system = multiskill_elo

        # IRT vs Elo ağırlıkları
        total = max(irt_weight + elo_weight, 1e-6)
        self.irt_weight = irt_weight / total
        self.elo_weight = elo_weight / total

        logger.info(
            "Math question selector initialized",
            target_prob=(target_prob_low, target_prob_high),
            exploration_strategy=self.exploration_strategy,
            epsilon=self.epsilon,
            ucb_c=self.ucb_c,
            irt_weight=self.irt_weight,
            elo_weight=self.elo_weight,
        )

    # ------------------------------------------------------------------
    # Çekirdek skorlayıcı: P(correct | user, item)
    # ------------------------------------------------------------------
    def _get_student_ability(self, user_id: str) -> StudentAbility:
        """Get or create a StudentAbility object."""
        if user_id in self.irt_model.students:
            return self.irt_model.students[user_id]
        # Varsayılan sıfır yetenek
        return StudentAbility(theta=0.0)

    def calculate_expected_probability(
        self,
        user_id: str,
        item_id: str,
        item_params: ItemParams,
        required_skills: List[str],
    ) -> float:
        """
        P(correct | user, item) tahmini.
        - IRT: p_irt
        - Elo: p_elo (user_skill_rating vs item_rating)
        Sonuç: p = w_irt * p_irt + w_elo * p_elo
        """
        # IRT tarafı
        try:
            student = self._get_student_ability(user_id)
            p_irt = self.irt_model.p_correct(student.theta, item_params)
        except Exception:
            p_irt = 0.6  # nötr fallback

        # Elo tarafı
        try:
            # Kullanıcının ilgili skill'lerdeki ortalama rating'i
            if user_id in self.elo_system.user_skills and required_skills:
                user_skill_ratings: List[float] = []
                for skill in required_skills:
                    if skill in self.elo_system.user_skills[user_id]:
                        user_skill_ratings.append(
                            self.elo_system.user_skills[user_id][skill].rating
                        )
                    else:
                        user_skill_ratings.append(self.elo_system.initial_rating)
                user_rating = float(np.mean(user_skill_ratings))
            else:
                user_rating = self.elo_system.initial_rating

            item_rating = self.elo_system.get_item_rating(item_id)

            # Klasik Elo logistic fonksiyonu
            rating_diff = item_rating - user_rating
            p_elo = 1.0 / (1.0 + 10.0 ** (rating_diff / 400.0))
        except Exception:
            p_elo = 0.6

        combined = self.irt_weight * p_irt + self.elo_weight * p_elo
        # Aşırı uçları keselim
        return float(np.clip(combined, 1e-3, 1.0 - 1e-3))

    # ------------------------------------------------------------------
    # Adaptif zorluk aralığı (probability-based)
    # ------------------------------------------------------------------
    def get_adaptive_difficulty_range(self, user_id: str) -> Tuple[float, float]:
        """
        Kullanıcının genel skill mastery seviyesine göre
        hedef başarı olasılığı aralığını hafifçe kaydırır.
        """
        base_low, base_high = self.target_prob_low, self.target_prob_high

        if user_id not in self.elo_system.user_skills:
            return base_low, base_high

        # Tüm skill'ler için mastery ortalaması
        masteries: List[float] = []
        for skill in self.elo_system.user_skills[user_id].keys():
            try:
                m = self.elo_system.get_skill_mastery(user_id, skill)
            except Exception:
                m = 0.5
            masteries.append(m)

        if not masteries:
            return base_low, base_high

        mean_mastery = float(np.mean(masteries))

        # İyi gidiyorsa (mastery yüksek) → daha zor sorular (daha düşük başarı hedefi)
        if mean_mastery > 0.75:
            low = max(0.4, base_low - 0.1)
            high = max(low + 0.1, base_high - 0.05)
        # Zorlanıyorsa (mastery düşük) → daha kolay sorular (daha yüksek başarı hedefi)
        elif mean_mastery < 0.4:
            low = min(0.8, base_low + 0.1)
            high = min(0.95, base_high + 0.1)
        else:
            low, high = base_low, base_high

        # Biraz güvenlik: low < high olsun
        if low >= high:
            low, high = base_low, base_high

        return low, high

    def filter_questions_by_difficulty(
        self,
        questions: List[Dict[str, Any]],
        user_id: str,
    ) -> List[Dict[str, Any]]:
        """
        Kullanıcının seviyesi için uygun probability band'inde kalan soruları filtreler.
        """
        if not questions:
            return []

        filtered_questions: List[Dict[str, Any]] = []

        # Adaptif aralık
        min_prob, max_prob = self.get_adaptive_difficulty_range(user_id)

        for question in questions:
            item_id = question["id"]
            item_params = ItemParams.from_dict(question.get("irt_params", {}))
            required_skills = question.get("required_skills", [])

            expected_prob = self.calculate_expected_probability(
                user_id, item_id, item_params, required_skills
            )

            if min_prob <= expected_prob <= max_prob:
                filtered_questions.append(question)

        # Hiç soru yoksa, daha geniş ve sabit bir aralığa geri dön
        if not filtered_questions:
            logger.warning(
                "No questions in target difficulty range, falling back to wide band",
                user_id=user_id,
                min_prob=min_prob,
                max_prob=max_prob,
            )
            for question in questions:
                item_id = question["id"]
                item_params = ItemParams.from_dict(question.get("irt_params", {}))
                required_skills = question.get("required_skills", [])

                expected_prob = self.calculate_expected_probability(
                    user_id, item_id, item_params, required_skills
                )

                if 0.3 <= expected_prob <= 0.9:
                    filtered_questions.append(question)

        # Yine de boş ise, orijinal listeyi dön (son çare)
        return filtered_questions or questions

    # ------------------------------------------------------------------
    # Learning gain & weakness-based
    # ------------------------------------------------------------------
    def calculate_learning_gain(
        self,
        user_id: str,
        item_params: ItemParams,
        required_skills: List[str],
        skill_weights: Dict[str, float],
    ) -> float:
        """
        Beklenen öğrenme kazancı:
        - IRT belirsizliği (p*(1-p))
        - Skill importance (teacher-weight)
        - Weakness-based (1 - mastery)
        - Skill diversity (birden çok skill içeren sorulara hafif bonus)
        """
        # 1) IRT belirsizliği
        try:
            if user_id in self.irt_model.students:
                student = self.irt_model.students[user_id]
                current_theta = student.theta
            else:
                current_theta = 0.0
            current_prob = self.irt_model.p_correct(current_theta, item_params)
        except Exception:
            current_prob = 0.6

        uncertainty = current_prob * (1.0 - current_prob)  # [0, 0.25]

        # 2) Skill importance (öğretmen tarafından verilen ağırlıklar)
        if skill_weights:
            base_importance = sum(skill_weights.values()) / len(skill_weights)
        else:
            base_importance = 1.0

        # 3) Weakness-based ağırlık (Elo skill mastery)
        weakness_scores: List[float] = []
        for skill in required_skills:
            try:
                mastery = self.elo_system.get_skill_mastery(user_id, skill)
            except Exception:
                mastery = 0.5  # nötr
            weakness = 1.0 - mastery  # mastery düşükse weakness yüksek
            weakness_scores.append(weakness)

        avg_weakness = float(np.mean(weakness_scores)) if weakness_scores else 0.0
        # 1 (nötr) ile 2 (çok zayıf) aralığında çarpan
        weakness_factor = 1.0 + avg_weakness

        # 4) Skill çeşitliliği: farklı skill sayısı
        skill_diversity = len(set(required_skills))
        diversity_factor = 1.0 + 0.5 * min(skill_diversity / 5.0, 1.0)  # max +0.5

        learning_gain = uncertainty * base_importance * weakness_factor * diversity_factor
        return learning_gain

    def calculate_skill_diversity_score(
        self,
        user_id: str,
        required_skills: List[str],
    ) -> float:
        """
        Skill çeşitliliğini 0-1 aralığında skorlar.
        Şu an basit: benzersiz skill sayısını normalize ediyor.
        """
        if not required_skills:
            return 0.0
        unique_skills = len(set(required_skills))
        return min(unique_skills / 5.0, 1.0)  # 5 skill ve üzeri ≈ 1.0

    # ------------------------------------------------------------------
    # Soru seçim politikası + bandit
    # ------------------------------------------------------------------
    def select_next_question(
        self,
        user_id: str,
        questions: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """
        Verilen soru havuzundan, bu kullanıcı için bir sonraki soruyu seçer.

        Adımlar:
        1) Zorluk aralığına göre filtreleme (P(correct) band'i)
        2) Her soru için candidate obje oluşturma
        3) Learning gain + skill diversity + difficulty balance ile skor hesaplama
        4) Bandit (epsilon-greedy veya UCB) ile exploration/exploitation kararı
        """
        if not questions:
            logger.warning("No questions provided to selector", user_id=user_id)
            return None

        filtered_questions = self.filter_questions_by_difficulty(questions, user_id)
        if not filtered_questions:
            logger.warning("No questions after difficulty filtering", user_id=user_id)
            return None

        candidates: List[QuestionCandidate] = []

        for question in filtered_questions:
            item_id = question["id"]
            item_params = ItemParams.from_dict(question.get("irt_params", {}))
            required_skills = question.get("required_skills", [])
            skill_weights = question.get("skill_weights", {})

            elo_rating = self.elo_system.get_item_rating(item_id)
            difficulty = self.elo_system.get_item_difficulty(item_id)

            expected_prob = self.calculate_expected_probability(
                user_id, item_id, item_params, required_skills
            )
            learning_gain = self.calculate_learning_gain(
                user_id, item_params, required_skills, skill_weights
            )
            skill_diversity = self.calculate_skill_diversity_score(
                user_id, required_skills
            )

            candidate = QuestionCandidate(
                item_id=item_id,
                item_params=item_params,
                required_skills=required_skills,
                skill_weights=skill_weights,
                elo_rating=elo_rating,
                difficulty=difficulty,
                expected_probability=expected_prob,
                learning_gain=learning_gain,
                skill_diversity=skill_diversity,
                metadata=question,
            )
            candidates.append(candidate)

        if not candidates:
            logger.warning("No candidates built for selector", user_id=user_id)
            return None

        # 3) Skorla ve sıralama
        scored_candidates: List[Tuple[QuestionCandidate, float]] = []

        for candidate in candidates:
            # Ana skor: learning gain
            score = candidate.learning_gain * self.learning_gain_weight

            # Skill çeşitliliği bonusu
            score += candidate.skill_diversity * self.skill_diversity_weight

            # Kullanıcının skill dağılımını dengelemeye çalışan bonus
            if user_id in self.elo_system.user_skills:
                user_skills = self.elo_system.user_skills[user_id]

                skill_ratings: List[float] = []
                for skill in candidate.required_skills:
                    if skill in user_skills:
                        skill_ratings.append(user_skills[skill].rating)
                    else:
                        skill_ratings.append(self.elo_system.initial_rating)

                if skill_ratings:
                    mean_rating = float(np.mean(skill_ratings))
                    std_rating = float(np.std(skill_ratings))
                    # std / mean ne kadar küçükse, o kadar dengeli
                    if mean_rating > 0:
                        balance = 1.0 - (std_rating / mean_rating)
                        score += balance * self.difficulty_balance_weight

            scored_candidates.append((candidate, score))

        if not scored_candidates:
            return None

        # Deterministik sıralama (exploitation için baz)
        scored_candidates.sort(key=lambda x: x[1], reverse=True)

        # 4) Bandit: exploration / exploitation
        best_candidate: QuestionCandidate

        if self.exploration_strategy == "epsilon_greedy" and len(scored_candidates) > 1:
            top_k = min(self.top_k_explore, len(scored_candidates))
            explore_pool = [c for (c, _) in scored_candidates[:top_k]]

            if random.random() < self.epsilon:
                # Explore: top-k içerisinden rastgele
                best_candidate = random.choice(explore_pool)
                logger.info(
                    "Exploration step (epsilon-greedy)",
                    user_id=user_id,
                    item_id=best_candidate.item_id,
                )
            else:
                best_candidate = scored_candidates[0][0]

        elif self.exploration_strategy == "ucb":
            # Basit UCB: az çözülmüş sorulara bonus
            total_attempts = sum(
                elo_item.attempts
                for elo_item in self.elo_system.item_ratings.values()
            )
            total_attempts = max(total_attempts, 1)

            ucb_scored: List[Tuple[QuestionCandidate, float]] = []
            for candidate, base_score in scored_candidates:
                elo_item = self.elo_system.item_ratings.get(candidate.item_id)
                n_i = elo_item.attempts if elo_item is not None else 0
                bonus = self.ucb_c * math.sqrt(
                    math.log(total_attempts + 1.0) / (n_i + 1.0)
                )
                ucb_scored.append((candidate, base_score + bonus))

            ucb_scored.sort(key=lambda x: x[1], reverse=True)
            best_candidate = ucb_scored[0][0]

        else:
            # exploration_strategy = "none"
            best_candidate = scored_candidates[0][0]

        logger.info(
            "Selected question",
            user_id=user_id,
            item_id=best_candidate.item_id,
            expected_prob=best_candidate.expected_probability,
            learning_gain=best_candidate.learning_gain,
        )

        return best_candidate.metadata


# Varsayılan global selector instance
math_question_selector = MathQuestionSelector()
