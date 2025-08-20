from typing import List, Dict, Any, Optional, Union, Tuple
import logging
import json
import re
from collections import defaultdict
from datetime import datetime

from app.core.config import settings

logger = logging.getLogger(__name__)


class CEFRValidator:
    """Comprehensive CEFR validation service"""
    
    def __init__(self):
        self.cache_ttl = 3600  # 1 hour
        
        # CEFR level definitions
        self.cefr_levels = ["A1", "A2", "B1", "B2", "C1", "C2"]
        
        # CEFR wordlists (simplified - in production, use comprehensive lists)
        self.cefr_wordlists = self._initialize_cefr_wordlists()
        
        # Grammar complexity patterns
        self.grammar_patterns = self._initialize_grammar_patterns()
        
        # Vocabulary complexity indicators
        self.vocabulary_indicators = self._initialize_vocabulary_indicators()
        
        # Text complexity metrics
        self.complexity_metrics = {
            "avg_sentence_length": {"A1": 8, "A2": 12, "B1": 15, "B2": 18, "C1": 20, "C2": 25},
            "avg_word_length": {"A1": 4.0, "A2": 4.5, "B1": 5.0, "B2": 5.5, "C1": 6.0, "C2": 6.5},
            "complex_word_ratio": {"A1": 0.1, "A2": 0.2, "B1": 0.3, "B2": 0.4, "C1": 0.5, "C2": 0.6}
        }
    
    def _initialize_cefr_wordlists(self) -> Dict[str, Dict[str, List[str]]]:
        """Initialize CEFR wordlists"""
        
        return {
            "A1": {
                "basic_words": [
                    "hello", "goodbye", "yes", "no", "please", "thank you", "sorry",
                    "name", "age", "family", "friend", "house", "car", "book", "food",
                    "water", "time", "day", "night", "morning", "evening", "today",
                    "tomorrow", "yesterday", "big", "small", "good", "bad", "hot", "cold"
                ],
                "grammar_words": [
                    "is", "are", "am", "was", "were", "have", "has", "had", "do", "does",
                    "can", "will", "would", "should", "may", "might", "the", "a", "an",
                    "and", "or", "but", "in", "on", "at", "to", "from", "with", "by"
                ]
            },
            "A2": {
                "basic_words": [
                    "work", "school", "study", "learn", "teach", "read", "write",
                    "speak", "listen", "watch", "see", "hear", "feel", "think", "know",
                    "understand", "remember", "forget", "like", "love", "hate", "want",
                    "need", "help", "use", "make", "take", "give", "get", "go", "come"
                ],
                "grammar_words": [
                    "be", "been", "being", "have", "having", "do", "doing", "did",
                    "can", "could", "will", "would", "shall", "should", "may", "might",
                    "must", "ought", "used", "going", "want", "need", "like", "love"
                ]
            },
            "B1": {
                "intermediate_words": [
                    "achieve", "accomplish", "accomplish", "acquire", "adapt", "adjust",
                    "admit", "adopt", "advance", "affect", "agree", "allow", "announce",
                    "appear", "apply", "approach", "argue", "arrange", "arrive", "ask",
                    "assume", "avoid", "become", "begin", "believe", "belong", "build"
                ],
                "grammar_structures": [
                    "present_perfect", "past_perfect", "future_perfect", "conditionals",
                    "passive_voice", "reported_speech", "relative_clauses", "gerunds",
                    "infinitives", "participles", "modal_verbs", "phrasal_verbs"
                ]
            },
            "B2": {
                "advanced_words": [
                    "accomplish", "acquire", "adapt", "adequate", "adjust", "administer",
                    "advocate", "allocate", "analyze", "anticipate", "appreciate", "approach",
                    "appropriate", "approximate", "articulate", "assemble", "assert",
                    "assess", "assign", "assume", "assure", "attain", "attribute", "authorize"
                ],
                "grammar_structures": [
                    "complex_conditionals", "inverted_conditionals", "mixed_conditionals",
                    "perfect_gerunds", "perfect_infinitives", "causative_forms",
                    "subjunctive_mood", "inversion", "cleft_sentences", "emphasis"
                ]
            },
            "C1": {
                "sophisticated_words": [
                    "accomplish", "acquire", "adapt", "adequate", "adjust", "administer",
                    "advocate", "allocate", "analyze", "anticipate", "appreciate", "approach",
                    "appropriate", "approximate", "articulate", "assemble", "assert",
                    "assess", "assign", "assume", "assure", "attain", "attribute", "authorize"
                ],
                "academic_structures": [
                    "academic_vocabulary", "complex_sentences", "formal_register",
                    "impersonal_structures", "nominalization", "passive_voice",
                    "subordination", "coordination", "discourse_markers"
                ]
            },
            "C2": {
                "expert_words": [
                    "accomplish", "acquire", "adapt", "adequate", "adjust", "administer",
                    "advocate", "allocate", "analyze", "anticipate", "appreciate", "approach",
                    "appropriate", "approximate", "articulate", "assemble", "assert",
                    "assess", "assign", "assume", "assure", "attain", "attribute", "authorize"
                ],
                "expert_structures": [
                    "complex_academic_vocabulary", "sophisticated_grammar",
                    "nuanced_expressions", "idiomatic_language", "literary_devices",
                    "rhetorical_techniques", "advanced_discourse", "stylistic_variation"
                ]
            }
        }
    
    def _initialize_grammar_patterns(self) -> Dict[str, List[str]]:
        """Initialize grammar complexity patterns"""
        
        return {
            "A1": [
                r"\b(am|is|are)\b",  # Basic be verbs
                r"\b(have|has)\b",   # Basic have verbs
                r"\b(can|will)\b",   # Basic modals
                r"\b(the|a|an)\b",   # Basic articles
                r"\b(and|or|but)\b"  # Basic conjunctions
            ],
            "A2": [
                r"\b(was|were)\b",   # Past be verbs
                r"\b(had)\b",        # Past have
                r"\b(could|would)\b", # Past modals
                r"\b(because|when|if)\b", # Basic subordinators
                r"\b(very|really|quite)\b" # Basic intensifiers
            ],
            "B1": [
                r"\b(have|has)\s+\w+ed\b",  # Present perfect
                r"\b(had)\s+\w+ed\b",       # Past perfect
                r"\b(will|would)\s+have\b", # Future perfect
                r"\b(if|unless|although)\b", # Complex subordinators
                r"\b(however|therefore|furthermore)\b" # Discourse markers
            ],
            "B2": [
                r"\b(had)\s+been\s+\w+ing\b",  # Past perfect continuous
                r"\b(would)\s+have\s+been\b",  # Conditional perfect
                r"\b(not\s+only|but\s+also)\b", # Inversion patterns
                r"\b(it\s+is|there\s+is)\b",   # Cleft sentences
                r"\b(nevertheless|consequently|subsequently)\b" # Advanced markers
            ],
            "C1": [
                r"\b(should|might)\s+have\s+been\b", # Modal perfect
                r"\b(were|had)\s+it\s+not\b",       # Inverted conditionals
                r"\b(notwithstanding|furthermore|moreover)\b", # Academic markers
                r"\b(thus|hence|thereby)\b",        # Logical connectors
                r"\b(consequently|subsequently|accordingly)\b" # Causal markers
            ],
            "C2": [
                r"\b(notwithstanding|furthermore|moreover)\b", # Complex academic
                r"\b(thus|hence|thereby)\b",        # Sophisticated connectors
                r"\b(consequently|subsequently|accordingly)\b", # Advanced causal
                r"\b(nevertheless|nonetheless|however)\b",     # Nuanced contrast
                r"\b(accordingly|consequently|therefore)\b"    # Formal conclusion
            ]
        }
    
    def _initialize_vocabulary_indicators(self) -> Dict[str, Dict[str, float]]:
        """Initialize vocabulary complexity indicators"""
        
        return {
            "word_length": {
                "A1": 4.0, "A2": 4.5, "B1": 5.0, "B2": 5.5, "C1": 6.0, "C2": 6.5
            },
            "complex_word_ratio": {
                "A1": 0.1, "A2": 0.2, "B1": 0.3, "B2": 0.4, "C1": 0.5, "C2": 0.6
            },
            "unique_word_ratio": {
                "A1": 0.6, "A2": 0.7, "B1": 0.75, "B2": 0.8, "C1": 0.85, "C2": 0.9
            },
            "academic_word_ratio": {
                "A1": 0.0, "A2": 0.05, "B1": 0.1, "B2": 0.15, "C1": 0.25, "C2": 0.35
            }
        }
    
    async def validate_cefr_compliance(
        self,
        text: str,
        target_cefr: str,
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """Comprehensive CEFR compliance validation"""
        
        try:
            # Basic validation
            if target_cefr not in self.cefr_levels:
                return {
                    "is_compliant": False,
                    "issues": [f"Invalid CEFR level: {target_cefr}"],
                    "suggestions": ["Use valid CEFR level (A1, A2, B1, B2, C1, C2)"],
                    "overall_score": 0.0
                }
            
            # Analyze text complexity
            complexity_analysis = self._analyze_text_complexity(text)
            
            # Check vocabulary compliance
            vocabulary_check = self._check_vocabulary_compliance(text, target_cefr)
            
            # Check grammar compliance
            grammar_check = self._check_grammar_compliance(text, target_cefr)
            
            # Check structural compliance
            structural_check = self._check_structural_compliance(text, target_cefr)
            
            # Calculate overall compliance
            compliance_score = self._calculate_compliance_score(
                complexity_analysis, vocabulary_check, grammar_check, structural_check
            )
            
            # Determine if compliant
            is_compliant = compliance_score >= 0.7
            
            # Generate issues and suggestions
            issues, suggestions = self._generate_issues_and_suggestions(
                complexity_analysis, vocabulary_check, grammar_check, structural_check, target_cefr
            )
            
            return {
                "is_compliant": is_compliant,
                "overall_score": compliance_score,
                "complexity_analysis": complexity_analysis,
                "vocabulary_check": vocabulary_check,
                "grammar_check": grammar_check,
                "structural_check": structural_check,
                "issues": issues,
                "suggestions": suggestions,
                "target_cefr": target_cefr,
                "estimated_cefr": complexity_analysis["estimated_cefr"]
            }
            
        except Exception as e:
            logger.error(f"Error in CEFR validation: {e}")
            return {
                "is_compliant": False,
                "issues": [f"Validation error: {str(e)}"],
                "suggestions": ["Check text format and try again"],
                "overall_score": 0.0
            }
    
    def _analyze_text_complexity(self, text: str) -> Dict[str, Any]:
        """Analyze text complexity metrics"""
        
        words = text.lower().split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Basic metrics
        word_count = len(words)
        sentence_count = len(sentences)
        unique_words = len(set(words))
        
        # Calculate averages
        avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
        avg_word_length = sum(len(word) for word in words) / word_count if word_count > 0 else 0
        unique_word_ratio = unique_words / word_count if word_count > 0 else 0
        
        # Complex word analysis
        complex_words = [word for word in words if len(word) > 6]
        complex_word_ratio = len(complex_words) / word_count if word_count > 0 else 0
        
        # Academic word analysis (simplified)
        academic_words = [word for word in words if self._is_academic_word(word)]
        academic_word_ratio = len(academic_words) / word_count if word_count > 0 else 0
        
        # Estimate CEFR level
        estimated_cefr = self._estimate_cefr_from_metrics(
            avg_sentence_length, avg_word_length, complex_word_ratio, academic_word_ratio
        )
        
        return {
            "word_count": word_count,
            "sentence_count": sentence_count,
            "unique_words": unique_words,
            "avg_sentence_length": avg_sentence_length,
            "avg_word_length": avg_word_length,
            "unique_word_ratio": unique_word_ratio,
            "complex_word_ratio": complex_word_ratio,
            "academic_word_ratio": academic_word_ratio,
            "estimated_cefr": estimated_cefr
        }
    
    def _check_vocabulary_compliance(self, text: str, target_cefr: str) -> Dict[str, Any]:
        """Check vocabulary compliance with target CEFR level"""
        
        words = text.lower().split()
        target_index = self.cefr_levels.index(target_cefr)
        
        # Check word length compliance
        avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
        target_word_length = self.vocabulary_indicators["word_length"][target_cefr]
        word_length_score = 1.0 - abs(avg_word_length - target_word_length) / target_word_length
        
        # Check complex word ratio
        complex_words = [word for word in words if len(word) > 6]
        complex_word_ratio = len(complex_words) / len(words) if words else 0
        target_complex_ratio = self.vocabulary_indicators["complex_word_ratio"][target_cefr]
        complex_word_score = 1.0 - abs(complex_word_ratio - target_complex_ratio) / target_complex_ratio
        
        # Check for inappropriate words
        inappropriate_words = []
        for word in words:
            word_level = self._get_word_cefr_level(word)
            if word_level > target_index + 1:  # Allow +1 level tolerance
                inappropriate_words.append(word)
        
        # Calculate vocabulary score
        vocabulary_score = (word_length_score + complex_word_score) / 2
        if inappropriate_words:
            vocabulary_score *= 0.8  # Penalty for inappropriate words
        
        return {
            "vocabulary_score": max(0.0, vocabulary_score),
            "avg_word_length": avg_word_length,
            "complex_word_ratio": complex_word_ratio,
            "inappropriate_words": inappropriate_words,
            "is_compliant": vocabulary_score >= 0.7
        }
    
    def _check_grammar_compliance(self, text: str, target_cefr: str) -> Dict[str, Any]:
        """Check grammar compliance with target CEFR level"""
        
        target_index = self.cefr_levels.index(target_cefr)
        
        # Check grammar patterns
        pattern_scores = {}
        total_patterns = 0
        found_patterns = 0
        
        for level in self.cefr_levels[:target_index + 1]:
            level_patterns = self.grammar_patterns.get(level, [])
            total_patterns += len(level_patterns)
            
            for pattern in level_patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    pattern_scores[pattern] = 1.0
                    found_patterns += 1
                else:
                    pattern_scores[pattern] = 0.0
        
        # Calculate grammar score
        grammar_score = found_patterns / total_patterns if total_patterns > 0 else 0.5
        
        # Check for advanced patterns that shouldn't be used
        inappropriate_patterns = []
        for level in self.cefr_levels[target_index + 2:]:
            level_patterns = self.grammar_patterns.get(level, [])
            for pattern in level_patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    inappropriate_patterns.append(pattern)
        
        if inappropriate_patterns:
            grammar_score *= 0.8  # Penalty for inappropriate patterns
        
        return {
            "grammar_score": max(0.0, grammar_score),
            "pattern_scores": pattern_scores,
            "inappropriate_patterns": inappropriate_patterns,
            "is_compliant": grammar_score >= 0.6
        }
    
    def _check_structural_compliance(self, text: str, target_cefr: str) -> Dict[str, Any]:
        """Check structural compliance (sentence length, complexity)"""
        
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Check sentence length
        sentence_lengths = [len(s.split()) for s in sentences]
        avg_sentence_length = sum(sentence_lengths) / len(sentence_lengths) if sentence_lengths else 0
        
        target_sentence_length = self.complexity_metrics["avg_sentence_length"][target_cefr]
        sentence_length_score = 1.0 - abs(avg_sentence_length - target_sentence_length) / target_sentence_length
        
        # Check sentence variety
        sentence_variety = len(set(sentence_lengths)) / len(sentence_lengths) if sentence_lengths else 0
        variety_score = min(1.0, sentence_variety * 2)  # Normalize to 0-1
        
        # Check for complex structures
        complex_structures = 0
        for sentence in sentences:
            if len(sentence.split()) > target_sentence_length * 1.5:
                complex_structures += 1
        
        complexity_ratio = complex_structures / len(sentences) if sentences else 0
        complexity_score = 1.0 - abs(complexity_ratio - 0.2) / 0.2  # Target 20% complex sentences
        
        # Calculate structural score
        structural_score = (sentence_length_score + variety_score + complexity_score) / 3
        
        return {
            "structural_score": max(0.0, structural_score),
            "avg_sentence_length": avg_sentence_length,
            "sentence_variety": sentence_variety,
            "complexity_ratio": complexity_ratio,
            "is_compliant": structural_score >= 0.6
        }
    
    def _calculate_compliance_score(
        self,
        complexity_analysis: Dict[str, Any],
        vocabulary_check: Dict[str, Any],
        grammar_check: Dict[str, Any],
        structural_check: Dict[str, Any]
    ) -> float:
        """Calculate overall compliance score"""
        
        # Weighted average of all scores
        weights = {
            "vocabulary": 0.3,
            "grammar": 0.3,
            "structural": 0.2,
            "complexity": 0.2
        }
        
        # Complexity score based on estimated vs target CEFR
        complexity_score = 1.0
        if "estimated_cefr" in complexity_analysis:
            estimated = complexity_analysis["estimated_cefr"]
            # This would be compared with target CEFR in the main validation function
        
        total_score = (
            vocabulary_check["vocabulary_score"] * weights["vocabulary"] +
            grammar_check["grammar_score"] * weights["grammar"] +
            structural_check["structural_score"] * weights["structural"] +
            complexity_score * weights["complexity"]
        )
        
        return max(0.0, min(1.0, total_score))
    
    def _generate_issues_and_suggestions(
        self,
        complexity_analysis: Dict[str, Any],
        vocabulary_check: Dict[str, Any],
        grammar_check: Dict[str, Any],
        structural_check: Dict[str, Any],
        target_cefr: str
    ) -> Tuple[List[str], List[str]]:
        """Generate issues and suggestions based on validation results"""
        
        issues = []
        suggestions = []
        
        # Vocabulary issues
        if not vocabulary_check["is_compliant"]:
            issues.append("Vocabulary not appropriate for target CEFR level")
            suggestions.append("Use vocabulary appropriate for " + target_cefr + " level")
        
        if vocabulary_check["inappropriate_words"]:
            issues.append(f"Contains {len(vocabulary_check['inappropriate_words'])} inappropriate words")
            suggestions.append("Replace advanced vocabulary with simpler alternatives")
        
        # Grammar issues
        if not grammar_check["is_compliant"]:
            issues.append("Grammar structures not appropriate for target CEFR level")
            suggestions.append("Use grammar structures appropriate for " + target_cefr + " level")
        
        if grammar_check["inappropriate_patterns"]:
            issues.append(f"Contains {len(grammar_check['inappropriate_patterns'])} inappropriate grammar patterns")
            suggestions.append("Simplify grammar structures")
        
        # Structural issues
        if not structural_check["is_compliant"]:
            issues.append("Sentence structure not appropriate for target CEFR level")
            suggestions.append("Adjust sentence length and complexity for " + target_cefr + " level")
        
        # Complexity issues
        estimated_cefr = complexity_analysis.get("estimated_cefr", "B1")
        if estimated_cefr != target_cefr:
            issues.append(f"Text complexity estimated as {estimated_cefr}, target is {target_cefr}")
            if self.cefr_levels.index(estimated_cefr) > self.cefr_levels.index(target_cefr):
                suggestions.append("Simplify text to match target CEFR level")
            else:
                suggestions.append("Increase text complexity to match target CEFR level")
        
        return issues, suggestions
    
    def _estimate_cefr_from_metrics(
        self,
        avg_sentence_length: float,
        avg_word_length: float,
        complex_word_ratio: float,
        academic_word_ratio: float
    ) -> str:
        """Estimate CEFR level from text metrics"""
        
        # Calculate scores for each level
        level_scores = {}
        
        for level in self.cefr_levels:
            target_sentence_length = self.complexity_metrics["avg_sentence_length"][level]
            target_word_length = self.vocabulary_indicators["word_length"][level]
            target_complex_ratio = self.vocabulary_indicators["complex_word_ratio"][level]
            target_academic_ratio = self.vocabulary_indicators["academic_word_ratio"][level]
            
            # Calculate individual scores
            sentence_score = 1.0 - abs(avg_sentence_length - target_sentence_length) / target_sentence_length
            word_score = 1.0 - abs(avg_word_length - target_word_length) / target_word_length
            complex_score = 1.0 - abs(complex_word_ratio - target_complex_ratio) / target_complex_ratio
            academic_score = 1.0 - abs(academic_word_ratio - target_academic_ratio) / (target_academic_ratio + 0.1)
            
            # Weighted average
            level_scores[level] = (
                sentence_score * 0.3 +
                word_score * 0.3 +
                complex_score * 0.2 +
                academic_score * 0.2
            )
        
        # Return level with highest score
        return max(level_scores, key=level_scores.get)
    
    def _get_word_cefr_level(self, word: str) -> int:
        """Get CEFR level for a specific word"""
        
        for i, level in enumerate(self.cefr_levels):
            wordlists = self.cefr_wordlists.get(level, {})
            for category, words in wordlists.items():
                if word in words:
                    return i
        
        # If not found in wordlists, estimate based on length
        if len(word) <= 4:
            return 0  # A1
        elif len(word) <= 5:
            return 1  # A2
        elif len(word) <= 6:
            return 2  # B1
        elif len(word) <= 7:
            return 3  # B2
        elif len(word) <= 8:
            return 4  # C1
        else:
            return 5  # C2
    
    def _is_academic_word(self, word: str) -> bool:
        """Check if word is academic (simplified)"""
        
        academic_indicators = [
            "tion", "sion", "ment", "ness", "ity", "ance", "ence", "al", "ive", "able",
            "ible", "ous", "ful", "less", "ly", "ize", "ise", "ify", "ate", "en"
        ]
        
        return any(indicator in word for indicator in academic_indicators)
    
    async def get_cefr_statistics(self) -> Dict[str, Any]:
        """Get statistics about CEFR validation"""
        
        return {
            "supported_levels": self.cefr_levels,
            "wordlist_sizes": {
                level: sum(len(words) for words in categories.values())
                for level, categories in self.cefr_wordlists.items()
            },
            "grammar_patterns_count": {
                level: len(patterns) for level, patterns in self.grammar_patterns.items()
            },
            "complexity_metrics": self.complexity_metrics,
            "vocabulary_indicators": self.vocabulary_indicators
        }

    def get_full_rubric_text(self) -> str:
        """Returns a comprehensive CEFR rubric text for LLM context."""
        return """
CEFR Rubric Guidelines:

A1 - Breakthrough:
  - Can understand and use familiar everyday expressions and very basic phrases aimed at the satisfaction of needs of a concrete type.
  - Can introduce him/herself and others and can ask and answer questions about personal details such as where he/she lives, people he/she knows and things he/she has.
  - Can interact in a simple way provided the other person talks slowly and clearly and is prepared to help.

A2 - Waystage:
  - Can understand sentences and frequently used expressions related to areas of most immediate relevance (e.g. very basic personal and family information, shopping, local geography, employment).
  - Can communicate in simple and routine tasks requiring a simple and direct exchange of information on familiar and routine matters.
  - Can describe in simple terms aspects of his/her background, immediate environment and matters in areas of immediate need.

B1 - Threshold:
  - Can understand the main points of clear standard input on familiar matters regularly encountered in work, school, leisure, etc.
  - Can deal with most situations likely to arise whilst travelling in an area where the language is spoken.
  - Can produce simple connected text on topics which are familiar or of personal interest.
  - Can describe experiences and events, dreams, hopes & ambitions and briefly give reasons and explanations for opinions and plans.

B2 - Vantage:
  - Can understand the main ideas of complex text on both concrete and abstract topics, including technical discussions in his/her field of specialisation.
  - Can interact with a degree of fluency and spontaneity that makes regular interaction with native speakers quite possible without strain for either party.
  - Can produce clear, detailed text on a wide range of subjects and explain a viewpoint on a topical issue giving the advantages and disadvantages of various options.

C1 - Effective Operational Proficiency:
  - Can understand a wide range of demanding, longer texts, and recognise implicit meaning.
  - Can express him/herself fluently and spontaneously without much obvious searching for expressions.
  - Can use language flexibly and effectively for social, academic and professional purposes.
  - Can produce clear, well-structured, detailed text on complex subjects, showing controlled use of organisational patterns, connectors and cohesive devices.

C2 - Mastery:
  - Can understand with ease virtually everything heard or read.
  - Can summarise information from different spoken and written sources, reconstructing arguments and accounts in a coherent presentation.
  - Can express him/herself spontaneously, very fluently and precisely, differentiating finer shades of meaning even in more complex situations.
"""


# Global instance
cefr_validator = CEFRValidator()