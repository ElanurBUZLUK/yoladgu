import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from collections import defaultdict
import re
import json

from app.models.math_profile import MathProfile
from app.models.question import Question

logger = logging.getLogger(__name__)


@dataclass
class RetrievalQuery:
    """Retrieval sorgusu"""
    user_id: str
    profile: MathProfile
    target_difficulty: float
    topic_category: Optional[str] = None
    question_type: Optional[str] = None
    exclude_question_ids: Optional[List[str]] = None
    context: Optional[Dict[str, Any]] = None


@dataclass
class RetrievalResult:
    """Retrieval sonucu"""
    question: Question
    relevance_score: float
    diversity_score: float
    freshness_score: float
    difficulty_match: float
    overall_score: float
    retrieval_method: str
    metadata: Dict[str, Any]


class MathAdvancedRetrieval:
    """Matematik gelişmiş retrieval servisi"""
    
    def __init__(self):
        self.config = {
            # Query expansion parametreleri
            "expansion_threshold": 0.7,
            "max_expansion_terms": 5,
            "synonym_weight": 0.8,
            "related_term_weight": 0.6,
            
            # Reranking parametreleri
            "relevance_weight": 0.4,
            "diversity_weight": 0.2,
            "freshness_weight": 0.15,
            "difficulty_weight": 0.25,
            
            # MMR diversification parametreleri
            "mmr_lambda": 0.5,  # Relevance vs diversity balance
            "diversity_threshold": 0.3,
            "max_similar_questions": 3,
            
            # Hybrid search parametreleri
            "semantic_weight": 0.6,
            "keyword_weight": 0.4,
            "context_weight": 0.2,
            
            # Context-aware parametreleri
            "recent_questions_window": 10,
            "performance_weight": 0.3,
            "preference_weight": 0.4,
            "goal_weight": 0.3,
        }
        
        # Matematik terimleri ve eşanlamlıları
        self.math_synonyms = {
            "solve": ["find", "calculate", "compute", "determine", "evaluate"],
            "equation": ["formula", "expression", "function", "relation"],
            "variable": ["unknown", "parameter", "quantity"],
            "function": ["mapping", "transformation", "operation"],
            "derivative": ["rate of change", "slope", "gradient"],
            "integral": ["antiderivative", "area", "summation"],
            "limit": ["bound", "threshold", "approximation"],
            "matrix": ["array", "table", "grid"],
            "vector": ["direction", "magnitude", "component"],
            "probability": ["chance", "likelihood", "odds"],
        }
        
        # Matematik konu ilişkileri
        self.topic_relations = {
            "algebra": ["equations", "functions", "polynomials", "matrices"],
            "geometry": ["shapes", "angles", "areas", "volumes", "coordinates"],
            "calculus": ["derivatives", "integrals", "limits", "functions"],
            "arithmetic": ["numbers", "operations", "fractions", "decimals"],
            "statistics": ["probability", "data", "averages", "distributions"],
        }
    
    async def advanced_retrieve_questions(
        self, 
        query: RetrievalQuery,
        candidate_questions: List[Question],
        max_results: int = 10
    ) -> List[RetrievalResult]:
        """Gelişmiş soru retrieval"""
        
        # Query expansion
        expanded_query = self._expand_query(query)
        
        # Hybrid search
        hybrid_results = await self._hybrid_search(expanded_query, candidate_questions)
        
        # Reranking
        reranked_results = await self._rerank_questions(hybrid_results, query)
        
        # MMR diversification
        diversified_results = await self._mmr_diversification(reranked_results, query)
        
        # Context-aware filtering
        context_filtered = await self._context_aware_filtering(diversified_results, query)
        
        return context_filtered[:max_results]
    
    async def query_expansion(
        self, 
        base_query: str,
        topic_category: Optional[str] = None
    ) -> Dict[str, float]:
        """Query expansion"""
        
        expanded_terms = {}
        
        # Temel terimleri ekle
        base_terms = base_query.lower().split()
        for term in base_terms:
            expanded_terms[term] = 1.0
        
        # Eşanlamlıları ekle
        for term in base_terms:
            if term in self.math_synonyms:
                synonyms = self.math_synonyms[term]
                for synonym in synonyms:
                    expanded_terms[synonym] = self.config["synonym_weight"]
        
        # Konu ilişkili terimleri ekle
        if topic_category and topic_category in self.topic_relations:
            related_terms = self.topic_relations[topic_category]
            for term in related_terms:
                if term not in expanded_terms:
                    expanded_terms[term] = self.config["related_term_weight"]
        
        # Ağırlıkları normalize et
        max_weight = max(expanded_terms.values()) if expanded_terms else 1.0
        normalized_terms = {
            term: weight / max_weight 
            for term, weight in expanded_terms.items()
        }
        
        return normalized_terms
    
    async def semantic_search(
        self, 
        query_terms: Dict[str, float],
        questions: List[Question]
    ) -> List[Tuple[Question, float]]:
        """Semantic search"""
        
        results = []
        
        for question in questions:
            # Basit semantic similarity (gerçek implementasyonda embedding kullanılacak)
            question_terms = self._extract_question_terms(question.content)
            
            # Cosine similarity
            similarity = self._calculate_cosine_similarity(query_terms, question_terms)
            
            if similarity > 0.1:  # Minimum similarity threshold
                results.append((question, similarity))
        
        # Skora göre sırala
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results
    
    async def keyword_search(
        self, 
        query_terms: Dict[str, float],
        questions: List[Question]
    ) -> List[Tuple[Question, float]]:
        """Keyword search"""
        
        results = []
        
        for question in questions:
            # Exact match scoring
            exact_matches = 0
            total_terms = len(query_terms)
            
            for term, weight in query_terms.items():
                if term.lower() in question.content.lower():
                    exact_matches += weight
            
            # Normalize score
            keyword_score = exact_matches / total_terms if total_terms > 0 else 0.0
            
            if keyword_score > 0.0:
                results.append((question, keyword_score))
        
        # Skora göre sırala
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results
    
    async def context_aware_search(
        self, 
        query: RetrievalQuery,
        questions: List[Question]
    ) -> List[Tuple[Question, float]]:
        """Context-aware search"""
        
        results = []
        
        for question in questions:
            context_score = 0.0
            
            # Performans bazlı scoring
            if query.context and "recent_performance" in query.context:
                performance_score = self._calculate_performance_context_score(
                    question, query.context["recent_performance"]
                )
                context_score += performance_score * self.config["performance_weight"]
            
            # Tercih bazlı scoring
            if query.context and "preferences" in query.context:
                preference_score = self._calculate_preference_context_score(
                    question, query.context["preferences"]
                )
                context_score += preference_score * self.config["preference_weight"]
            
            # Hedef bazlı scoring
            if query.context and "learning_goals" in query.context:
                goal_score = self._calculate_goal_context_score(
                    question, query.context["learning_goals"]
                )
                context_score += goal_score * self.config["goal_weight"]
            
            if context_score > 0.0:
                results.append((question, context_score))
        
        # Skora göre sırala
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results
    
    def _expand_query(self, query: RetrievalQuery) -> Dict[str, float]:
        """Query expansion"""
        
        # Temel query terimleri
        base_terms = {
            "math": 1.0,
            "question": 1.0,
            "problem": 1.0,
        }
        
        # Konu terimleri ekle
        if query.topic_category:
            base_terms[query.topic_category] = 1.0
            if query.topic_category in self.topic_relations:
                for related_term in self.topic_relations[query.topic_category]:
                    base_terms[related_term] = 0.8
        
        # Zorluk terimleri ekle
        if query.target_difficulty <= 1.5:
            base_terms["easy"] = 0.9
            base_terms["basic"] = 0.8
        elif query.target_difficulty <= 3.5:
            base_terms["medium"] = 0.9
            base_terms["intermediate"] = 0.8
        else:
            base_terms["hard"] = 0.9
            base_terms["advanced"] = 0.8
        
        # Eşanlamlıları ekle
        expanded_terms = base_terms.copy()
        for term in list(base_terms.keys()):
            if term in self.math_synonyms:
                synonyms = self.math_synonyms[term]
                for synonym in synonyms:
                    expanded_terms[synonym] = self.config["synonym_weight"]
        
        return expanded_terms
    
    async def _hybrid_search(
        self, 
        expanded_query: Dict[str, float],
        questions: List[Question]
    ) -> List[Tuple[Question, float]]:
        """Hybrid search"""
        
        # Semantic search
        semantic_results = await self.semantic_search(expanded_query, questions)
        
        # Keyword search
        keyword_results = await self.keyword_search(expanded_query, questions)
        
        # Sonuçları birleştir
        combined_results = {}
        
        # Semantic sonuçları ekle
        for question, score in semantic_results:
            combined_results[question.id] = {
                "question": question,
                "semantic_score": score,
                "keyword_score": 0.0,
                "combined_score": score * self.config["semantic_weight"]
            }
        
        # Keyword sonuçları ekle/güncelle
        for question, score in keyword_results:
            if question.id in combined_results:
                combined_results[question.id]["keyword_score"] = score
                combined_results[question.id]["combined_score"] += score * self.config["keyword_weight"]
            else:
                combined_results[question.id] = {
                    "question": question,
                    "semantic_score": 0.0,
                    "keyword_score": score,
                    "combined_score": score * self.config["keyword_weight"]
                }
        
        # Birleştirilmiş sonuçları döndür
        results = [
            (item["question"], item["combined_score"])
            for item in combined_results.values()
        ]
        
        # Skora göre sırala
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results
    
    async def _rerank_questions(
        self, 
        search_results: List[Tuple[Question, float]],
        query: RetrievalQuery
    ) -> List[RetrievalResult]:
        """Soru reranking"""
        
        reranked_results = []
        
        for question, search_score in search_results:
            # Relevance score (search score)
            relevance_score = search_score
            
            # Diversity score
            diversity_score = self._calculate_diversity_score(question, query)
            
            # Freshness score
            freshness_score = self._calculate_freshness_score(question)
            
            # Difficulty match score
            difficulty_match = self._calculate_difficulty_match(question, query.target_difficulty)
            
            # Overall score
            overall_score = (
                relevance_score * self.config["relevance_weight"] +
                diversity_score * self.config["diversity_weight"] +
                freshness_score * self.config["freshness_weight"] +
                difficulty_match * self.config["difficulty_weight"]
            )
            
            result = RetrievalResult(
                question=question,
                relevance_score=relevance_score,
                diversity_score=diversity_score,
                freshness_score=freshness_score,
                difficulty_match=difficulty_match,
                overall_score=overall_score,
                retrieval_method="hybrid_reranked",
                metadata={
                    "search_score": search_score,
                    "topic_category": question.topic_category,
                    "question_type": question.question_type.value
                }
            )
            
            reranked_results.append(result)
        
        # Overall score'a göre sırala
        reranked_results.sort(key=lambda x: x.overall_score, reverse=True)
        
        return reranked_results
    
    async def _mmr_diversification(
        self, 
        reranked_results: List[RetrievalResult],
        query: RetrievalQuery
    ) -> List[RetrievalResult]:
        """MMR diversification"""
        
        if not reranked_results:
            return []
        
        # İlk sonucu seç
        selected = [reranked_results[0]]
        remaining = reranked_results[1:]
        
        # MMR ile diğer sonuçları seç
        while remaining and len(selected) < len(reranked_results):
            mmr_scores = []
            
            for result in remaining:
                # Relevance score
                relevance = result.relevance_score
                
                # Diversity score (seçili sonuçlarla benzerlik)
                max_similarity = 0.0
                for selected_result in selected:
                    similarity = self._calculate_question_similarity(
                        result.question, selected_result.question
                    )
                    max_similarity = max(max_similarity, similarity)
                
                diversity = 1.0 - max_similarity
                
                # MMR score
                mmr_score = (
                    self.config["mmr_lambda"] * relevance +
                    (1 - self.config["mmr_lambda"]) * diversity
                )
                
                mmr_scores.append((result, mmr_score))
            
            # En yüksek MMR score'lu sonucu seç
            if mmr_scores:
                best_result, _ = max(mmr_scores, key=lambda x: x[1])
                selected.append(best_result)
                remaining = [r for r, _ in mmr_scores if r != best_result]
            else:
                break
        
        # MMR method'unu güncelle
        for result in selected:
            result.retrieval_method = "mmr"
        
        return selected
    
    async def _context_aware_filtering(
        self, 
        diversified_results: List[RetrievalResult],
        query: RetrievalQuery
    ) -> List[RetrievalResult]:
        """Context-aware filtering"""
        
        filtered_results = []
        
        for result in diversified_results:
            # Context score hesapla
            context_score = 0.0
            
            # Performans context'i
            if query.context and "recent_performance" in query.context:
                performance_context = self._calculate_performance_context_score(
                    result.question, query.context["recent_performance"]
                )
                context_score += performance_context * self.config["performance_weight"]
            
            # Tercih context'i
            if query.context and "preferences" in query.context:
                preference_context = self._calculate_preference_context_score(
                    result.question, query.context["preferences"]
                )
                context_score += preference_context * self.config["preference_weight"]
            
            # Hedef context'i
            if query.context and "learning_goals" in query.context:
                goal_context = self._calculate_goal_context_score(
                    result.question, query.context["learning_goals"]
                )
                context_score += goal_context * self.config["goal_weight"]
            
            # Context score'u overall score'a ekle
            result.overall_score = (
                result.overall_score * 0.8 + context_score * 0.2
            )
            
            filtered_results.append(result)
        
        # Yeni overall score'a göre sırala
        filtered_results.sort(key=lambda x: x.overall_score, reverse=True)
        
        return filtered_results
    
    def _extract_question_terms(self, content: str) -> Dict[str, float]:
        """Soru terimlerini çıkar"""
        
        # Basit term extraction
        words = re.findall(r'\b\w+\b', content.lower())
        
        # Stop words'leri filtrele
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}
        filtered_words = [word for word in words if word not in stop_words and len(word) > 2]
        
        # Term frequency hesapla
        term_freq = defaultdict(int)
        for word in filtered_words:
            term_freq[word] += 1
        
        # Normalize et
        max_freq = max(term_freq.values()) if term_freq else 1
        normalized_terms = {
            term: freq / max_freq 
            for term, freq in term_freq.items()
        }
        
        return normalized_terms
    
    def _calculate_cosine_similarity(
        self, 
        query_terms: Dict[str, float],
        question_terms: Dict[str, float]
    ) -> float:
        """Cosine similarity hesapla"""
        
        # Ortak terimleri bul
        common_terms = set(query_terms.keys()) & set(question_terms.keys())
        
        if not common_terms:
            return 0.0
        
        # Dot product
        dot_product = sum(
            query_terms[term] * question_terms[term]
            for term in common_terms
        )
        
        # Magnitudes
        query_magnitude = np.sqrt(sum(score ** 2 for score in query_terms.values()))
        question_magnitude = np.sqrt(sum(score ** 2 for score in question_terms.values()))
        
        # Cosine similarity
        if query_magnitude == 0 or question_magnitude == 0:
            return 0.0
        
        return dot_product / (query_magnitude * question_magnitude)
    
    def _calculate_diversity_score(self, question: Question, query: RetrievalQuery) -> float:
        """Çeşitlilik skoru hesapla"""
        
        # Basit diversity scoring
        diversity_score = 1.0
        
        # Konu çeşitliliği
        if query.topic_category and question.topic_category != query.topic_category:
            diversity_score += 0.2
        
        # Soru tipi çeşitliliği
        if query.question_type and question.question_type.value != query.question_type:
            diversity_score += 0.1
        
        return min(1.0, diversity_score)
    
    def _calculate_freshness_score(self, question: Question) -> float:
        """Tazelik skoru hesapla"""
        
        if not question.last_seen_at:
            return 1.0
        
        # Son görülme zamanına göre tazelik
        days_since_seen = (datetime.utcnow() - question.last_seen_at).days
        
        if days_since_seen < 1:
            return 0.3  # Çok yeni
        elif days_since_seen < 7:
            return 0.7  # Yeni
        elif days_since_seen < 30:
            return 0.9  # Orta
        else:
            return 1.0  # Eski
    
    def _calculate_difficulty_match(self, question: Question, target_difficulty: float) -> float:
        """Zorluk uyumu hesapla"""
        
        question_difficulty = question.estimated_difficulty or question.difficulty_level
        difficulty_diff = abs(question_difficulty - target_difficulty)
        
        # Zorluk farkına göre skor
        if difficulty_diff <= 0.5:
            return 1.0
        elif difficulty_diff <= 1.0:
            return 0.8
        elif difficulty_diff <= 1.5:
            return 0.6
        else:
            return 0.3
    
    def _calculate_question_similarity(self, question1: Question, question2: Question) -> float:
        """Soru benzerliği hesapla"""
        
        # İçerik benzerliği
        terms1 = self._extract_question_terms(question1.content)
        terms2 = self._extract_question_terms(question2.content)
        
        content_similarity = self._calculate_cosine_similarity(terms1, terms2)
        
        # Konu benzerliği
        topic_similarity = 1.0 if question1.topic_category == question2.topic_category else 0.0
        
        # Zorluk benzerliği
        diff1 = question1.estimated_difficulty or question1.difficulty_level
        diff2 = question2.estimated_difficulty or question2.difficulty_level
        difficulty_similarity = 1.0 - (abs(diff1 - diff2) / 5.0)
        
        # Ağırlıklı ortalama
        return (
            content_similarity * 0.6 +
            topic_similarity * 0.3 +
            difficulty_similarity * 0.1
        )
    
    def _calculate_performance_context_score(
        self, 
        question: Question,
        recent_performance: List[float]
    ) -> float:
        """Performans context skoru hesapla"""
        
        if not recent_performance:
            return 0.5
        
        avg_performance = np.mean(recent_performance)
        question_difficulty = question.estimated_difficulty or question.difficulty_level
        
        # Düşük performans: Kolay sorular
        if avg_performance < 0.4:
            return 1.0 if question_difficulty <= 2.0 else 0.3
        # Yüksek performans: Zor sorular
        elif avg_performance > 0.8:
            return 1.0 if question_difficulty >= 3.0 else 0.3
        else:
            return 1.0 if 2.0 <= question_difficulty <= 3.5 else 0.5
    
    def _calculate_preference_context_score(
        self, 
        question: Question,
        preferences: Dict[str, Any]
    ) -> float:
        """Tercih context skoru hesapla"""
        
        score = 0.5  # Default score
        
        # Konu tercihi
        if "topic_preferences" in preferences:
            topic_pref = preferences["topic_preferences"].get(question.topic_category, 1.0)
            score += (topic_pref - 1.0) * 0.2
        
        # Zorluk tercihi
        if "difficulty_preferences" in preferences:
            difficulty = question.estimated_difficulty or question.difficulty_level
            if difficulty <= 1.5:
                difficulty_category = "easy"
            elif difficulty <= 3.5:
                difficulty_category = "medium"
            else:
                difficulty_category = "hard"
            
            diff_pref = preferences["difficulty_preferences"].get(difficulty_category, 1.0)
            score += (diff_pref - 1.0) * 0.2
        
        return max(0.0, min(1.0, score))
    
    def _calculate_goal_context_score(
        self, 
        question: Question,
        learning_goals: List[str]
    ) -> float:
        """Hedef context skoru hesapla"""
        
        if not learning_goals:
            return 0.5
        
        score = 0.5
        
        # Hedef türüne göre scoring
        if "mastery" in learning_goals:
            # Mastery için zor sorular
            difficulty = question.estimated_difficulty or question.difficulty_level
            if difficulty >= 3.0:
                score += 0.3
        
        if "speed" in learning_goals:
            # Speed için çoktan seçmeli sorular
            if question.question_type.value == "multiple_choice":
                score += 0.2
        
        if "accuracy" in learning_goals:
            # Accuracy için açık uçlu sorular
            if question.question_type.value == "open_ended":
                score += 0.2
        
        return max(0.0, min(1.0, score))


# Global instance
math_advanced_retrieval = MathAdvancedRetrieval()
