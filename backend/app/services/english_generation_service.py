"""
English cloze question generation service.

This service implements error taxonomy and confusion sets for generating
targeted English grammar questions based on student error profiles.
"""

import logging
import re
from typing import Dict, List, Set, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ErrorType(Enum):
    """English error types for targeted question generation."""
    PREPOSITIONS = "prepositions"
    ARTICLES = "articles"
    SUBJECT_VERB_AGREEMENT = "subject_verb_agreement"
    COLLOCATIONS = "collocations"
    VERB_TENSES = "verb_tenses"
    MODAL_VERBS = "modal_verbs"
    CONDITIONALS = "conditionals"
    RELATIVE_CLAUSES = "relative_clauses"


class CEFRLevel(Enum):
    """CEFR proficiency levels."""
    A1 = "A1"
    A2 = "A2"
    B1 = "B1"
    B2 = "B2"
    C1 = "C1"


@dataclass
class ConfusionSet:
    """A set of commonly confused words/phrases for a specific error type."""
    error_type: ErrorType
    target_word: str
    confusors: List[str]
    context_patterns: List[str]
    cefr_level: CEFRLevel
    examples: List[str]


@dataclass
class ErrorPattern:
    """Pattern for detecting and generating specific error types."""
    error_type: ErrorType
    pattern_regex: str
    replacement_rules: Dict[str, List[str]]
    context_requirements: List[str]
    difficulty_level: CEFRLevel


class EnglishErrorTaxonomy:
    """
    Comprehensive error taxonomy for English language learning.
    
    This class manages confusion sets and error patterns for different
    types of English grammar errors commonly made by learners.
    """
    
    def __init__(self):
        self.confusion_sets = self._initialize_confusion_sets()
        self.error_patterns = self._initialize_error_patterns()
        self.cefr_vocabulary = self._initialize_cefr_vocabulary()
    
    def _initialize_confusion_sets(self) -> Dict[ErrorType, List[ConfusionSet]]:
        """Initialize confusion sets for different error types."""
        confusion_sets = {
            ErrorType.PREPOSITIONS: [
                # Time prepositions
                ConfusionSet(
                    error_type=ErrorType.PREPOSITIONS,
                    target_word="at",
                    confusors=["in", "on"],
                    context_patterns=[r"\b(at|in|on)\s+\d+:\d+", r"\b(at|in|on)\s+(night|noon|midnight)"],
                    cefr_level=CEFRLevel.A1,
                    examples=[
                        "I wake up at 7 o'clock.",
                        "We meet at noon.",
                        "She works at night."
                    ]
                ),
                ConfusionSet(
                    error_type=ErrorType.PREPOSITIONS,
                    target_word="in",
                    confusors=["at", "on"],
                    context_patterns=[r"\b(in|at|on)\s+(January|February|March|April|May|June|July|August|September|October|November|December)", r"\b(in|at|on)\s+\d{4}"],
                    cefr_level=CEFRLevel.A1,
                    examples=[
                        "I was born in 1990.",
                        "The meeting is in January.",
                        "We go on vacation in summer."
                    ]
                ),
                ConfusionSet(
                    error_type=ErrorType.PREPOSITIONS,
                    target_word="on",
                    confusors=["at", "in"],
                    context_patterns=[r"\b(on|at|in)\s+(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)", r"\b(on|at|in)\s+\w+day"],
                    cefr_level=CEFRLevel.A1,
                    examples=[
                        "The class is on Monday.",
                        "We have a party on Friday.",
                        "I don't work on weekends."
                    ]
                ),
                # Place prepositions
                ConfusionSet(
                    error_type=ErrorType.PREPOSITIONS,
                    target_word="at",
                    confusors=["in", "on"],
                    context_patterns=[r"\b(at|in|on)\s+(school|work|home|university)", r"\b(at|in|on)\s+the\s+(station|airport|hospital)"],
                    cefr_level=CEFRLevel.A2,
                    examples=[
                        "I study at university.",
                        "She works at the hospital.",
                        "We meet at the station."
                    ]
                ),
                ConfusionSet(
                    error_type=ErrorType.PREPOSITIONS,
                    target_word="in",
                    confusors=["at", "on"],
                    context_patterns=[r"\b(in|at|on)\s+(bed|prison|hospital)", r"\b(in|at|on)\s+the\s+(car|bus|train)"],
                    cefr_level=CEFRLevel.A2,
                    examples=[
                        "He is in bed.",
                        "They are in the car.",
                        "She was in prison."
                    ]
                ),
            ],
            
            ErrorType.ARTICLES: [
                # Definite article
                ConfusionSet(
                    error_type=ErrorType.ARTICLES,
                    target_word="the",
                    confusors=["a", "an", ""],
                    context_patterns=[r"\b(the|a|an)?\s+(sun|moon|earth|sky)", r"\b(the|a|an)?\s+(first|second|third|last)"],
                    cefr_level=CEFRLevel.A1,
                    examples=[
                        "The sun is shining.",
                        "She is the first student.",
                        "The moon is bright tonight."
                    ]
                ),
                # Indefinite articles
                ConfusionSet(
                    error_type=ErrorType.ARTICLES,
                    target_word="a",
                    confusors=["an", "the", ""],
                    context_patterns=[r"\b(a|an|the)?\s+[bcdfghjklmnpqrstvwxyz]\w*", r"\b(a|an|the)?\s+(book|car|house|dog)"],
                    cefr_level=CEFRLevel.A1,
                    examples=[
                        "I have a book.",
                        "She drives a car.",
                        "He lives in a house."
                    ]
                ),
                ConfusionSet(
                    error_type=ErrorType.ARTICLES,
                    target_word="an",
                    confusors=["a", "the", ""],
                    context_patterns=[r"\b(a|an|the)?\s+[aeiou]\w*", r"\b(a|an|the)?\s+(apple|orange|elephant|umbrella)"],
                    cefr_level=CEFRLevel.A1,
                    examples=[
                        "I eat an apple.",
                        "She has an umbrella.",
                        "He is an engineer."
                    ]
                ),
                # Zero article
                ConfusionSet(
                    error_type=ErrorType.ARTICLES,
                    target_word="",
                    confusors=["a", "an", "the"],
                    context_patterns=[r"\b(a|an|the)?\s+(water|milk|coffee|tea)", r"\b(a|an|the)?\s+(music|art|science|history)"],
                    cefr_level=CEFRLevel.A2,
                    examples=[
                        "I drink water.",
                        "She studies music.",
                        "He likes art."
                    ]
                ),
            ],
            
            ErrorType.SUBJECT_VERB_AGREEMENT: [
                # Third person singular
                ConfusionSet(
                    error_type=ErrorType.SUBJECT_VERB_AGREEMENT,
                    target_word="goes",
                    confusors=["go"],
                    context_patterns=[r"\b(he|she|it)\s+(go|goes)", r"\b\w+\s+(go|goes)\s+to"],
                    cefr_level=CEFRLevel.A1,
                    examples=[
                        "He goes to school.",
                        "She goes shopping.",
                        "It goes well."
                    ]
                ),
                ConfusionSet(
                    error_type=ErrorType.SUBJECT_VERB_AGREEMENT,
                    target_word="has",
                    confusors=["have"],
                    context_patterns=[r"\b(he|she|it)\s+(have|has)", r"\b\w+\s+(have|has)\s+\w+"],
                    cefr_level=CEFRLevel.A1,
                    examples=[
                        "He has a car.",
                        "She has two cats.",
                        "It has four wheels."
                    ]
                ),
                ConfusionSet(
                    error_type=ErrorType.SUBJECT_VERB_AGREEMENT,
                    target_word="is",
                    confusors=["are"],
                    context_patterns=[r"\b(he|she|it)\s+(is|are)", r"\bthere\s+(is|are)\s+\w+"],
                    cefr_level=CEFRLevel.A1,
                    examples=[
                        "He is happy.",
                        "She is a teacher.",
                        "There is a book on the table."
                    ]
                ),
            ],
            
            ErrorType.COLLOCATIONS: [
                # Make/Do collocations
                ConfusionSet(
                    error_type=ErrorType.COLLOCATIONS,
                    target_word="make",
                    confusors=["do"],
                    context_patterns=[r"\b(make|do)\s+(a\s+)?(mistake|decision|choice|plan)", r"\b(make|do)\s+(money|friends|progress)"],
                    cefr_level=CEFRLevel.A2,
                    examples=[
                        "I make a mistake.",
                        "She makes money.",
                        "They make friends easily."
                    ]
                ),
                ConfusionSet(
                    error_type=ErrorType.COLLOCATIONS,
                    target_word="do",
                    confusors=["make"],
                    context_patterns=[r"\b(make|do)\s+(homework|exercise|work|business)", r"\b(make|do)\s+(your\s+)?(best|job)"],
                    cefr_level=CEFRLevel.A2,
                    examples=[
                        "I do my homework.",
                        "She does exercise.",
                        "They do business together."
                    ]
                ),
                # Take/Have collocations
                ConfusionSet(
                    error_type=ErrorType.COLLOCATIONS,
                    target_word="take",
                    confusors=["have"],
                    context_patterns=[r"\b(take|have)\s+(a\s+)?(shower|bath|break|rest)", r"\b(take|have)\s+(medicine|pills|photos)"],
                    cefr_level=CEFRLevel.A2,
                    examples=[
                        "I take a shower.",
                        "She takes medicine.",
                        "They take photos."
                    ]
                ),
                ConfusionSet(
                    error_type=ErrorType.COLLOCATIONS,
                    target_word="have",
                    confusors=["take"],
                    context_patterns=[r"\b(take|have)\s+(breakfast|lunch|dinner)", r"\b(take|have)\s+(a\s+)?(party|meeting|conversation)"],
                    cefr_level=CEFRLevel.A2,
                    examples=[
                        "I have breakfast.",
                        "She has a meeting.",
                        "They have a party."
                    ]
                ),
            ]
        }
        
        return confusion_sets
    
    def _initialize_error_patterns(self) -> Dict[ErrorType, List[ErrorPattern]]:
        """Initialize error patterns for different error types."""
        patterns = {
            ErrorType.PREPOSITIONS: [
                ErrorPattern(
                    error_type=ErrorType.PREPOSITIONS,
                    pattern_regex=r'\b(at|in|on)\s+(\d+:\d+|night|noon|midnight)\b',
                    replacement_rules={
                        "time_specific": ["at"],
                        "time_general": ["in", "on"]
                    },
                    context_requirements=["time_expression"],
                    difficulty_level=CEFRLevel.A1
                ),
                ErrorPattern(
                    error_type=ErrorType.PREPOSITIONS,
                    pattern_regex=r'\b(at|in|on)\s+(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b',
                    replacement_rules={
                        "day_of_week": ["on"]
                    },
                    context_requirements=["day_expression"],
                    difficulty_level=CEFRLevel.A1
                ),
            ],
            
            ErrorType.ARTICLES: [
                ErrorPattern(
                    error_type=ErrorType.ARTICLES,
                    pattern_regex=r'\b(a|an|the)?\s+([aeiou]\w*)\b',
                    replacement_rules={
                        "vowel_sound": ["an"],
                        "consonant_sound": ["a"]
                    },
                    context_requirements=["countable_noun"],
                    difficulty_level=CEFRLevel.A1
                ),
            ],
            
            ErrorType.SUBJECT_VERB_AGREEMENT: [
                ErrorPattern(
                    error_type=ErrorType.SUBJECT_VERB_AGREEMENT,
                    pattern_regex=r'\b(he|she|it)\s+(go|have|do)\b',
                    replacement_rules={
                        "third_person_singular": ["goes", "has", "does"]
                    },
                    context_requirements=["third_person_subject"],
                    difficulty_level=CEFRLevel.A1
                ),
            ]
        }
        
        return patterns
    
    def _initialize_cefr_vocabulary(self) -> Dict[CEFRLevel, Set[str]]:
        """Initialize vocabulary sets for different CEFR levels."""
        return {
            CEFRLevel.A1: {
                # Basic vocabulary
                "hello", "goodbye", "yes", "no", "please", "thank", "you",
                "I", "you", "he", "she", "it", "we", "they",
                "am", "is", "are", "have", "has", "do", "does",
                "go", "come", "see", "look", "listen", "eat", "drink",
                "house", "car", "book", "table", "chair", "bed",
                "red", "blue", "green", "big", "small", "good", "bad",
                "one", "two", "three", "four", "five",
                "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
                "January", "February", "March", "April", "May", "June",
                "July", "August", "September", "October", "November", "December"
            },
            
            CEFRLevel.A2: {
                # Elementary vocabulary
                "family", "friend", "work", "school", "study", "learn",
                "buy", "sell", "pay", "cost", "money", "expensive", "cheap",
                "travel", "holiday", "hotel", "restaurant", "food", "menu",
                "weather", "rain", "sun", "snow", "hot", "cold", "warm",
                "happy", "sad", "angry", "tired", "hungry", "thirsty",
                "morning", "afternoon", "evening", "night", "today", "tomorrow", "yesterday",
                "always", "sometimes", "never", "often", "usually"
            },
            
            CEFRLevel.B1: {
                # Intermediate vocabulary
                "experience", "opportunity", "advantage", "disadvantage", "problem", "solution",
                "environment", "pollution", "technology", "computer", "internet", "website",
                "education", "university", "degree", "career", "profession", "interview",
                "culture", "tradition", "custom", "festival", "celebration", "ceremony",
                "health", "medicine", "doctor", "hospital", "treatment", "patient",
                "government", "politics", "election", "vote", "democracy", "citizen"
            },
            
            CEFRLevel.B2: {
                # Upper-intermediate vocabulary
                "achievement", "accomplishment", "ambition", "determination", "perseverance",
                "innovation", "creativity", "imagination", "inspiration", "motivation",
                "responsibility", "accountability", "reliability", "trustworthiness",
                "sustainability", "conservation", "preservation", "restoration",
                "globalization", "urbanization", "industrialization", "modernization",
                "controversy", "debate", "argument", "discussion", "negotiation"
            },
            
            CEFRLevel.C1: {
                # Advanced vocabulary
                "sophistication", "complexity", "intricacy", "subtlety", "nuance",
                "phenomenon", "manifestation", "implication", "ramification", "consequence",
                "hypothesis", "assumption", "presumption", "speculation", "conjecture",
                "paradigm", "methodology", "framework", "infrastructure", "architecture",
                "unprecedented", "extraordinary", "remarkable", "exceptional", "outstanding"
            }
        }
    
    def get_confusion_sets_by_error_type(self, error_type: ErrorType) -> List[ConfusionSet]:
        """Get all confusion sets for a specific error type."""
        return self.confusion_sets.get(error_type, [])
    
    def get_confusion_sets_by_cefr_level(self, cefr_level: CEFRLevel) -> List[ConfusionSet]:
        """Get all confusion sets appropriate for a CEFR level."""
        result = []
        for error_type, confusion_sets in self.confusion_sets.items():
            for cs in confusion_sets:
                if cs.cefr_level.value <= cefr_level.value:
                    result.append(cs)
        return result
    
    def get_targeted_confusion_sets(
        self, 
        error_profile: Dict[str, float], 
        cefr_level: CEFRLevel,
        min_error_rate: float = 0.3
    ) -> List[ConfusionSet]:
        """
        Get confusion sets targeted to student's error profile.
        
        Args:
            error_profile: Dictionary mapping error types to error rates
            cefr_level: Student's CEFR level
            min_error_rate: Minimum error rate to consider an error type
            
        Returns:
            List of confusion sets prioritized by error rate
        """
        targeted_sets = []
        
        for error_type_str, error_rate in error_profile.items():
            if error_rate < min_error_rate:
                continue
                
            try:
                error_type = ErrorType(error_type_str)
                confusion_sets = self.get_confusion_sets_by_error_type(error_type)
                
                # Filter by CEFR level
                appropriate_sets = [
                    cs for cs in confusion_sets 
                    if cs.cefr_level.value <= cefr_level.value
                ]
                
                # Add error rate for prioritization
                for cs in appropriate_sets:
                    cs.priority_score = error_rate
                    targeted_sets.append(cs)
                    
            except ValueError:
                logger.warning(f"Unknown error type in profile: {error_type_str}")
                continue
        
        # Sort by error rate (highest first)
        targeted_sets.sort(key=lambda x: getattr(x, 'priority_score', 0), reverse=True)
        
        return targeted_sets
    
    def classify_cefr_level(self, text: str) -> CEFRLevel:
        """
        Classify the CEFR level of a text based on vocabulary complexity.
        
        Args:
            text: Input text to classify
            
        Returns:
            Estimated CEFR level
        """
        words = re.findall(r'\b\w+\b', text.lower())
        total_words = len(words)
        
        if total_words == 0:
            return CEFRLevel.A1
        
        # Count words at each level
        level_counts = {}
        for level, vocabulary in self.cefr_vocabulary.items():
            count = sum(1 for word in words if word in vocabulary)
            level_counts[level] = count / total_words
        
        # Determine level based on vocabulary distribution
        if level_counts[CEFRLevel.C1] > 0.1:
            return CEFRLevel.C1
        elif level_counts[CEFRLevel.B2] > 0.15:
            return CEFRLevel.B2
        elif level_counts[CEFRLevel.B1] > 0.2:
            return CEFRLevel.B1
        elif level_counts[CEFRLevel.A2] > 0.3:
            return CEFRLevel.A2
        else:
            return CEFRLevel.A1
    
    def validate_confusion_set(self, confusion_set: ConfusionSet) -> bool:
        """
        Validate that a confusion set is well-formed.
        
        Args:
            confusion_set: Confusion set to validate
            
        Returns:
            True if valid, False otherwise
        """
        # Check that target word is not in confusors
        if confusion_set.target_word in confusion_set.confusors:
            return False
        
        # Check that examples contain the target word
        target_found = any(
            confusion_set.target_word in example.lower() 
            for example in confusion_set.examples
        )
        
        if not target_found and confusion_set.target_word != "":
            return False
        
        # Check that context patterns are valid regex
        for pattern in confusion_set.context_patterns:
            try:
                re.compile(pattern)
            except re.error:
                return False
        
        return True
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get statistics about the error taxonomy."""
        stats = {
            "total_error_types": len(self.confusion_sets),
            "total_confusion_sets": sum(len(sets) for sets in self.confusion_sets.values()),
            "error_type_distribution": {},
            "cefr_level_distribution": {}
        }
        
        # Count by error type
        for error_type, confusion_sets in self.confusion_sets.items():
            stats["error_type_distribution"][error_type.value] = len(confusion_sets)
        
        # Count by CEFR level
        cefr_counts = {}
        for confusion_sets in self.confusion_sets.values():
            for cs in confusion_sets:
                level = cs.cefr_level.value
                cefr_counts[level] = cefr_counts.get(level, 0) + 1
        
        stats["cefr_level_distribution"] = cefr_counts
        
        return stats


@dataclass
class ClozeBlank:
    """Represents a blank in a cloze question."""
    span: str  # The text span that was blanked out
    answer: str  # The correct answer
    distractors: List[str]  # Wrong answer options
    skill_tag: str  # The error type this blank targets
    position: int  # Position in the text
    rationale: Optional[str] = None  # Explanation for the correct answer


@dataclass
class ClozeQuestion:
    """A complete cloze question."""
    passage: str  # The text with blanks
    blanks: List[ClozeBlank]  # List of blanks in the passage
    level_cefr: CEFRLevel  # CEFR level of the passage
    topic: Optional[str] = None  # Topic/theme of the passage
    error_tags: List[str] = None  # Error types targeted
    source: Optional[str] = None  # Source of the passage


class GrammarValidator:
    """
    Grammar validation and ambiguity checking for English cloze questions.
    
    This class provides grammar checking integration, single-answer guarantee
    checking, and ambiguity detection and resolution.
    """
    
    def __init__(self):
        self.grammar_rules = self._initialize_grammar_rules()
        self.ambiguity_patterns = self._initialize_ambiguity_patterns()
    
    def _initialize_grammar_rules(self) -> Dict[str, List[Dict[str, Any]]]:
        """Initialize grammar validation rules."""
        return {
            "articles": [
                {
                    "rule": "indefinite_article_vowel",
                    "pattern": r'\ban\s+[bcdfghjklmnpqrstvwxyz]',
                    "description": "Use 'a' before consonant sounds, not 'an'",
                    "severity": "error"
                },
                {
                    "rule": "indefinite_article_consonant", 
                    "pattern": r'\ba\s+[aeiou]',
                    "description": "Use 'an' before vowel sounds, not 'a'",
                    "severity": "error"
                },
                {
                    "rule": "definite_article_unique",
                    "pattern": r'\bthe\s+(sun|moon|earth|sky|world)',
                    "description": "Correct use of 'the' with unique objects",
                    "severity": "correct"
                }
            ],
            
            "prepositions": [
                {
                    "rule": "time_at_specific",
                    "pattern": r'\bat\s+\d+:\d+',
                    "description": "Correct use of 'at' with specific times",
                    "severity": "correct"
                },
                {
                    "rule": "time_in_general",
                    "pattern": r'\bin\s+(January|February|March|April|May|June|July|August|September|October|November|December|\d{4})',
                    "description": "Correct use of 'in' with months and years",
                    "severity": "correct"
                },
                {
                    "rule": "time_on_days",
                    "pattern": r'\bon\s+(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)',
                    "description": "Correct use of 'on' with days of the week",
                    "severity": "correct"
                }
            ],
            
            "subject_verb_agreement": [
                {
                    "rule": "third_person_singular_s",
                    "pattern": r'\b(he|she|it)\s+(go|have|do)\b',
                    "description": "Third person singular requires -s ending",
                    "severity": "error"
                },
                {
                    "rule": "plural_no_s",
                    "pattern": r'\b(they|we|you)\s+(goes|has|does)\b',
                    "description": "Plural subjects don't take -s ending",
                    "severity": "error"
                }
            ]
        }
    
    def _initialize_ambiguity_patterns(self) -> List[Dict[str, Any]]:
        """Initialize patterns that commonly lead to ambiguity."""
        return [
            {
                "pattern": r'\b(a|an|the)?\s+(water|milk|coffee|tea|music|art)\b',
                "description": "Uncountable nouns can be ambiguous with articles",
                "ambiguity_type": "article_uncountable",
                "resolution": "Check if noun is used in countable or uncountable sense"
            },
            {
                "pattern": r'\b(in|at|on)\s+(bed|hospital|prison|school|work)\b',
                "description": "Institutional nouns can take different prepositions",
                "ambiguity_type": "preposition_institutional",
                "resolution": "Consider whether referring to the building or the activity"
            },
            {
                "pattern": r'\b(make|do)\s+(homework|exercise|work|business|money|friends)\b',
                "description": "Make/do collocations can be ambiguous",
                "ambiguity_type": "collocation_make_do",
                "resolution": "Check standard collocation patterns"
            },
            {
                "pattern": r'\bthere\s+(is|are)\s+\w+',
                "description": "There is/are agreement depends on following noun",
                "ambiguity_type": "existential_there",
                "resolution": "Check if following noun is singular or plural"
            }
        ]
    
    def validate_grammar(self, text: str) -> Dict[str, Any]:
        """
        Validate grammar in a text passage.
        
        Args:
            text: Text to validate
            
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "suggestions": [],
            "score": 1.0
        }
        
        import re
        
        total_rules_checked = 0
        errors_found = 0
        
        for category, rules in self.grammar_rules.items():
            for rule in rules:
                total_rules_checked += 1
                matches = re.finditer(rule["pattern"], text, re.IGNORECASE)
                
                for match in matches:
                    if rule["severity"] == "error":
                        errors_found += 1
                        validation_results["errors"].append({
                            "rule": rule["rule"],
                            "position": match.span(),
                            "text": match.group(),
                            "description": rule["description"],
                            "category": category
                        })
                        validation_results["valid"] = False
                    elif rule["severity"] == "warning":
                        validation_results["warnings"].append({
                            "rule": rule["rule"],
                            "position": match.span(),
                            "text": match.group(),
                            "description": rule["description"],
                            "category": category
                        })
        
        # Calculate grammar score
        if total_rules_checked > 0:
            validation_results["score"] = max(0.0, 1.0 - (errors_found / total_rules_checked))
        
        return validation_results
    
    def check_single_answer_guarantee(self, question: 'ClozeQuestion') -> Dict[str, Any]:
        """
        Check that each blank has only one correct answer.
        
        Args:
            question: Cloze question to check
            
        Returns:
            Dictionary with single-answer validation results
        """
        results = {
            "guaranteed": True,
            "issues": [],
            "ambiguous_blanks": []
        }
        
        for i, blank in enumerate(question.blanks):
            # Check if multiple answers could be correct
            potential_answers = self._find_potential_answers(question.passage, blank)
            
            if len(potential_answers) > 1:
                results["guaranteed"] = False
                results["ambiguous_blanks"].append({
                    "blank_index": i,
                    "position": blank.position,
                    "potential_answers": potential_answers,
                    "skill_tag": blank.skill_tag
                })
                results["issues"].append(f"Blank {i+1} has multiple valid answers: {potential_answers}")
        
        return results
    
    def _find_potential_answers(self, passage: str, blank: ClozeBlank) -> List[str]:
        """Find all potentially correct answers for a blank."""
        potential_answers = [blank.answer]
        
        # Get the context around the blank
        context_start = max(0, blank.position - 50)
        context_end = min(len(passage), blank.position + len(blank.span) + 50)
        context = passage[context_start:context_end]
        
        # Check if any distractors could also be grammatically correct
        for distractor in blank.distractors:
            if self._is_grammatically_valid_in_context(distractor, context, blank):
                potential_answers.append(distractor)
        
        return potential_answers
    
    def _is_grammatically_valid_in_context(self, word: str, context: str, blank: ClozeBlank) -> bool:
        """Check if a word is grammatically valid in the given context."""
        # This is a simplified check - in a real implementation, this would be more sophisticated
        
        if blank.skill_tag == "articles":
            # For articles, check basic vowel/consonant rules
            if word in ["a", "an"]:
                # Find the word after the article
                import re
                pattern = rf'\b{re.escape(word)}\s+(\w+)'
                match = re.search(pattern, context, re.IGNORECASE)
                if match:
                    next_word = match.group(1)
                    if word == "a" and next_word[0].lower() in "aeiou":
                        return False  # Should be "an"
                    if word == "an" and next_word[0].lower() not in "aeiou":
                        return False  # Should be "a"
            return True
        
        elif blank.skill_tag == "prepositions":
            # For prepositions, most are contextually valid
            return True
        
        elif blank.skill_tag == "subject_verb_agreement":
            # Check subject-verb agreement
            import re
            # Find the subject before the verb
            subject_pattern = r'\b(I|you|he|she|it|we|they)\s+' + re.escape(word)
            match = re.search(subject_pattern, context, re.IGNORECASE)
            if match:
                subject = match.group(1).lower()
                if subject in ["he", "she", "it"] and word in ["go", "have", "do"]:
                    return False  # Should be goes/has/does
                if subject in ["I", "you", "we", "they"] and word in ["goes", "has", "does"]:
                    return False  # Should be go/have/do
            return True
        
        return True  # Default to valid for unknown categories
    
    def detect_ambiguity(self, question: 'ClozeQuestion') -> Dict[str, Any]:
        """
        Detect potential ambiguity in a cloze question.
        
        Args:
            question: Cloze question to analyze
            
        Returns:
            Dictionary with ambiguity analysis results
        """
        results = {
            "ambiguous": False,
            "ambiguity_issues": [],
            "severity": "none",
            "recommendations": []
        }
        
        import re
        
        # Check for known ambiguity patterns
        for pattern_info in self.ambiguity_patterns:
            matches = re.finditer(pattern_info["pattern"], question.passage, re.IGNORECASE)
            
            for match in matches:
                # Check if this match overlaps with any blank
                for blank in question.blanks:
                    if self._overlaps_with_blank(match.span(), blank.position, len(blank.span)):
                        results["ambiguous"] = True
                        results["ambiguity_issues"].append({
                            "type": pattern_info["ambiguity_type"],
                            "description": pattern_info["description"],
                            "position": match.span(),
                            "text": match.group(),
                            "resolution": pattern_info["resolution"],
                            "blank_affected": blank.skill_tag
                        })
        
        # Determine severity
        if len(results["ambiguity_issues"]) > 0:
            if len(results["ambiguity_issues"]) >= 3:
                results["severity"] = "high"
            elif len(results["ambiguity_issues"]) >= 2:
                results["severity"] = "medium"
            else:
                results["severity"] = "low"
        
        # Generate recommendations
        if results["ambiguous"]:
            results["recommendations"] = self._generate_ambiguity_recommendations(results["ambiguity_issues"])
        
        return results
    
    def _overlaps_with_blank(self, match_span: Tuple[int, int], blank_pos: int, blank_len: int) -> bool:
        """Check if a match overlaps with a blank position."""
        match_start, match_end = match_span
        blank_start = blank_pos
        blank_end = blank_pos + blank_len
        
        return not (match_end <= blank_start or match_start >= blank_end)
    
    def _generate_ambiguity_recommendations(self, ambiguity_issues: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations to resolve ambiguity issues."""
        recommendations = []
        
        issue_types = set(issue["type"] for issue in ambiguity_issues)
        
        if "article_uncountable" in issue_types:
            recommendations.append("Consider providing more context to clarify countable vs uncountable usage")
        
        if "preposition_institutional" in issue_types:
            recommendations.append("Add context clues to distinguish between location and activity meanings")
        
        if "collocation_make_do" in issue_types:
            recommendations.append("Use more specific context to make the collocation unambiguous")
        
        if "existential_there" in issue_types:
            recommendations.append("Ensure the noun following 'there is/are' clearly indicates singular/plural")
        
        return recommendations
    
    def resolve_ambiguity(self, question: 'ClozeQuestion', ambiguity_results: Dict[str, Any]) -> 'ClozeQuestion':
        """
        Attempt to resolve ambiguity in a cloze question.
        
        Args:
            question: Original cloze question
            ambiguity_results: Results from ambiguity detection
            
        Returns:
            Modified cloze question with reduced ambiguity
        """
        if not ambiguity_results["ambiguous"]:
            return question
        
        # Create a copy of the question to modify
        import copy
        resolved_question = copy.deepcopy(question)
        
        # Apply resolution strategies based on ambiguity types
        for issue in ambiguity_results["ambiguity_issues"]:
            if issue["type"] == "article_uncountable":
                resolved_question = self._resolve_article_ambiguity(resolved_question, issue)
            elif issue["type"] == "preposition_institutional":
                resolved_question = self._resolve_preposition_ambiguity(resolved_question, issue)
            elif issue["type"] == "collocation_make_do":
                resolved_question = self._resolve_collocation_ambiguity(resolved_question, issue)
        
        return resolved_question
    
    def _resolve_article_ambiguity(self, question: 'ClozeQuestion', issue: Dict[str, Any]) -> 'ClozeQuestion':
        """Resolve article-related ambiguity."""
        # Add context words to make countable/uncountable clear
        context_additions = {
            "water": "some water" if "water" in issue["text"] else issue["text"],
            "music": "classical music" if "music" in issue["text"] else issue["text"],
            "art": "modern art" if "art" in issue["text"] else issue["text"]
        }
        
        for word, replacement in context_additions.items():
            if word in issue["text"]:
                question.passage = question.passage.replace(issue["text"], replacement)
                break
        
        return question
    
    def _resolve_preposition_ambiguity(self, question: 'ClozeQuestion', issue: Dict[str, Any]) -> 'ClozeQuestion':
        """Resolve preposition-related ambiguity."""
        # Add clarifying context
        clarifications = {
            "at school": "at the school building",
            "in school": "in school (as a student)",
            "at hospital": "at the hospital",
            "in hospital": "in hospital (as a patient)"
        }
        
        for phrase, clarification in clarifications.items():
            if phrase in issue["text"]:
                question.passage = question.passage.replace(phrase, clarification)
                break
        
        return question
    
    def _resolve_collocation_ambiguity(self, question: 'ClozeQuestion', issue: Dict[str, Any]) -> 'ClozeQuestion':
        """Resolve collocation-related ambiguity."""
        # This would involve more sophisticated context analysis
        # For now, we'll just flag it for manual review
        return question


class EnglishClozeGenerator:
    """
    English cloze question generation pipeline.
    
    This class implements rule-based blank selection for target error types,
    LLM integration for context-appropriate answer generation, and personalized
    distractor generation based on error history.
    """
    
    def __init__(self, error_taxonomy: EnglishErrorTaxonomy):
        self.error_taxonomy = error_taxonomy
        self.grammar_validator = GrammarValidator()
        self.passage_templates = self._initialize_passage_templates()
        
    def _initialize_passage_templates(self) -> Dict[CEFRLevel, List[str]]:
        """Initialize passage templates for different CEFR levels."""
        return {
            CEFRLevel.A1: [
                "My name is John. I live {in/at/on} London. I go {to/at/in} work every day. I work {in/at/on} a hospital. I am {a/an/the} doctor. I help people who are sick. {In/At/On} the evening, I go home. I {have/has} dinner with my family. We watch TV {in/at/on} the living room. I go {to/in/at} bed {in/at/on} 10 o'clock.",
                
                "Sarah is {a/an/the} student. She {go/goes} to university every day. She {study/studies} medicine. {In/At/On} the morning, she {have/has} breakfast {in/at/on} 7 o'clock. Then she {take/takes} {a/an/the} bus to university. She {have/has} classes {in/at/on} Monday, Tuesday, and Wednesday. {In/At/On} the weekend, she works {in/at/on} a restaurant.",
                
                "Tom and Lisa are friends. They {live/lives} {in/at/on} the same street. Tom {have/has} {a/an/the} dog. The dog's name is Max. Every morning, Tom {take/takes} Max for {a/an/the} walk {in/at/on} the park. Lisa {have/has} {a/an/the} cat. She {love/loves} animals. {In/At/On} Sunday, they go {to/at/in} the zoo together."
            ],
            
            CEFRLevel.A2: [
                "Last summer, I went {to/at/in} Spain for my holidays. I stayed {in/at/on} {a/an/the} hotel near the beach. Every day, I woke up {in/at/on} 8 o'clock and {have/had} breakfast {in/at/on} the hotel restaurant. Then I went {to/at/in} the beach. I {make/do} many new friends there. We played volleyball and swam {in/at/on} the sea. {In/At/On} the evening, we went {to/at/in} restaurants and tried local food.",
                
                "My brother is getting married next month. He {have/has} been planning the wedding for {a/an/the} year. The ceremony will be {in/at/on} {a/an/the} church {in/at/on} Saturday morning. After that, we will {have/has} {a/an/the} party {in/at/on} {a/an/the} hotel. Many relatives are coming from different countries. My mother is very excited. She {make/do} {a/an/the} wedding cake herself.",
                
                "I started learning English three years ago. {In/At/On} the beginning, it was very difficult. I couldn't understand anything. But I {make/do} {a/an/the} decision to study hard. I {take/have} lessons twice {a/an/the} week. I also watch English movies and read books. Now I can {have/make} conversations with native speakers. Learning {a/an/the} language {take/takes} time, but it's worth it."
            ],
            
            CEFRLevel.B1: [
                "Climate change is one of {a/an/the} most serious problems facing our planet today. Scientists {have/has} been studying this issue for decades. They believe that human activities are {a/an/the} main cause of global warming. We need to {make/do} changes {in/at/on} our lifestyle to protect the environment. For example, we should use public transport instead of driving cars. We should also recycle more and use less energy {in/at/on} our homes.",
                
                "Technology {have/has} changed the way we communicate. {In/At/On} the past, people wrote letters and waited weeks for {a/an/the} reply. Now we can send messages instantly. Social media allows us to stay {in/at/on} touch with friends and family around the world. However, some people think that technology is making us less social. They believe we should spend more time talking face to face rather than looking {in/at/on} our phones.",
                
                "Education is very important for personal development. Students should {have/has} access to quality education regardless of their background. Teachers play {a/an/the} crucial role {in/at/on} shaping young minds. They need to be patient and understanding. {In/At/On} many countries, the government invests heavily {in/at/on} education. This includes building new schools and training teachers. Education helps people {make/do} better choices {in/at/on} their lives."
            ]
        }
    
    def generate_cloze_question(
        self,
        level_cefr: CEFRLevel,
        target_error_tags: List[str],
        topic: Optional[str] = None,
        personalization: Optional[Dict[str, Any]] = None,
        passage_text: Optional[str] = None
    ) -> ClozeQuestion:
        """
        Generate a cloze question targeting specific error types.
        
        Args:
            level_cefr: Target CEFR level
            target_error_tags: List of error types to target
            topic: Optional topic/theme
            personalization: User-specific preferences and error history
            passage_text: Optional custom passage text
            
        Returns:
            Generated cloze question
        """
        # Step 1: Select or generate passage
        if passage_text:
            passage = passage_text
        else:
            passage = self._select_passage_template(level_cefr, topic)
        
        # Step 2: Identify potential blanks based on target error types
        potential_blanks = self._identify_potential_blanks(passage, target_error_tags)
        
        # Step 3: Select best blanks based on difficulty and coverage
        selected_blanks = self._select_optimal_blanks(
            potential_blanks, 
            target_error_tags, 
            personalization
        )
        
        # Step 4: Generate distractors for each blank
        blanks_with_distractors = []
        for blank in selected_blanks:
            distractors = self._generate_distractors(blank, personalization)
            blank.distractors = distractors
            blanks_with_distractors.append(blank)
        
        # Step 5: Create final passage with blanks
        final_passage = self._create_blanked_passage(passage, blanks_with_distractors)
        
        # Step 6: Generate rationales
        for blank in blanks_with_distractors:
            blank.rationale = self._generate_rationale(blank)
        
        return ClozeQuestion(
            passage=final_passage,
            blanks=blanks_with_distractors,
            level_cefr=level_cefr,
            topic=topic,
            error_tags=target_error_tags,
            source="generated"
        )
    
    def _select_passage_template(self, level_cefr: CEFRLevel, topic: Optional[str]) -> str:
        """Select an appropriate passage template."""
        templates = self.passage_templates.get(level_cefr, self.passage_templates[CEFRLevel.A1])
        
        # For now, select randomly. In a real implementation, this could be topic-based
        import random
        return random.choice(templates)
    
    def _identify_potential_blanks(self, passage: str, target_error_tags: List[str]) -> List[ClozeBlank]:
        """Identify potential blanks in the passage based on target error types."""
        potential_blanks = []
        
        for error_tag in target_error_tags:
            try:
                error_type = ErrorType(error_tag)
                confusion_sets = self.error_taxonomy.get_confusion_sets_by_error_type(error_type)
                
                for confusion_set in confusion_sets:
                    # Find matches in the passage
                    blanks = self._find_confusion_set_matches(passage, confusion_set)
                    potential_blanks.extend(blanks)
                    
            except ValueError:
                logger.warning(f"Unknown error type: {error_tag}")
                continue
        
        return potential_blanks
    
    def _find_confusion_set_matches(self, passage: str, confusion_set: ConfusionSet) -> List[ClozeBlank]:
        """Find matches for a confusion set in the passage."""
        blanks = []
        
        # Look for the target word in the passage
        import re
        
        # First, check for template placeholders like {in/at/on}
        template_pattern = r'\{([^}]+)\}'
        template_matches = re.finditer(template_pattern, passage)
        
        for match in template_matches:
            options = match.group(1).split('/')
            if confusion_set.target_word in options:
                blank = ClozeBlank(
                    span=match.group(),
                    answer=confusion_set.target_word,
                    distractors=[opt for opt in options if opt != confusion_set.target_word],
                    skill_tag=confusion_set.error_type.value,
                    position=match.start()
                )
                blanks.append(blank)
        
        # Then, look for actual words in the passage
        if confusion_set.target_word == "":
            # Handle zero article case
            pattern = r'\b(a|an|the)\s+([a-zA-Z]+)'
        else:
            # Escape special regex characters
            escaped_target = re.escape(confusion_set.target_word)
            pattern = rf'\b{escaped_target}\b'
        
        matches = re.finditer(pattern, passage, re.IGNORECASE)
        
        for match in matches:
            start, end = match.span()
            
            # Skip if this overlaps with a template placeholder
            overlaps_template = False
            for template_match in re.finditer(template_pattern, passage):
                if (start >= template_match.start() and start <= template_match.end()) or \
                   (end >= template_match.start() and end <= template_match.end()):
                    overlaps_template = True
                    break
            
            if overlaps_template:
                continue
            
            # Check if this matches the context patterns
            context_match = False
            for context_pattern in confusion_set.context_patterns:
                if re.search(context_pattern, passage[max(0, start-50):end+50], re.IGNORECASE):
                    context_match = True
                    break
            
            if context_match or not confusion_set.context_patterns:
                blank = ClozeBlank(
                    span=match.group(),
                    answer=confusion_set.target_word,
                    distractors=[],  # Will be filled later
                    skill_tag=confusion_set.error_type.value,
                    position=start
                )
                blanks.append(blank)
        
        return blanks
    
    def _select_optimal_blanks(
        self, 
        potential_blanks: List[ClozeBlank], 
        target_error_tags: List[str],
        personalization: Optional[Dict[str, Any]]
    ) -> List[ClozeBlank]:
        """Select the optimal set of blanks for the question."""
        if not potential_blanks:
            return []
        
        # Sort by position to avoid overlapping blanks
        potential_blanks.sort(key=lambda x: x.position)
        
        # Remove overlapping blanks
        non_overlapping = []
        last_end = -1
        
        for blank in potential_blanks:
            if blank.position > last_end + 10:  # Minimum 10 character gap
                non_overlapping.append(blank)
                last_end = blank.position + len(blank.span)
        
        # Prioritize based on error profile if available
        if personalization and "error_profile" in personalization:
            error_profile = personalization["error_profile"]
            
            def priority_score(blank):
                error_rate = error_profile.get(blank.skill_tag, 0.0)
                return error_rate
            
            non_overlapping.sort(key=priority_score, reverse=True)
        
        # Select up to 5 blanks (typical for cloze questions)
        max_blanks = min(5, len(non_overlapping))
        return non_overlapping[:max_blanks]
    
    def _generate_distractors(
        self, 
        blank: ClozeBlank, 
        personalization: Optional[Dict[str, Any]]
    ) -> List[str]:
        """Generate plausible distractors for a blank."""
        try:
            error_type = ErrorType(blank.skill_tag)
            confusion_sets = self.error_taxonomy.get_confusion_sets_by_error_type(error_type)
            
            # Find the confusion set that matches this blank
            matching_set = None
            for cs in confusion_sets:
                if cs.target_word == blank.answer:
                    matching_set = cs
                    break
            
            if matching_set:
                # Use confusors from the matching set
                distractors = matching_set.confusors.copy()
                
                # Personalize based on user's common errors if available
                if personalization and "common_errors" in personalization:
                    common_errors = personalization["common_errors"].get(blank.skill_tag, [])
                    # Prioritize user's common errors as distractors
                    for error in common_errors:
                        if error not in distractors and error != blank.answer:
                            distractors.insert(0, error)
                
                # Limit to 3 distractors
                return distractors[:3]
            else:
                # Fallback: generate generic distractors
                return self._generate_generic_distractors(blank)
                
        except ValueError:
            return self._generate_generic_distractors(blank)
    
    def _generate_generic_distractors(self, blank: ClozeBlank) -> List[str]:
        """Generate generic distractors when no specific confusion set is found."""
        # This is a simplified fallback
        generic_distractors = {
            "prepositions": ["in", "on", "at", "to", "for", "with"],
            "articles": ["a", "an", "the", ""],
            "subject_verb_agreement": ["go", "goes", "have", "has", "is", "are"],
            "collocations": ["make", "do", "take", "have", "get", "give"]
        }
        
        skill_distractors = generic_distractors.get(blank.skill_tag, ["option1", "option2", "option3"])
        
        # Remove the correct answer
        distractors = [d for d in skill_distractors if d != blank.answer]
        
        return distractors[:3]
    
    def _create_blanked_passage(self, passage: str, blanks: List[ClozeBlank]) -> str:
        """Create the final passage with blanks replaced by placeholders."""
        # Sort blanks by position (descending) to avoid position shifts
        sorted_blanks = sorted(blanks, key=lambda x: x.position, reverse=True)
        
        blanked_passage = passage
        
        for i, blank in enumerate(sorted_blanks):
            # Replace the span with a numbered blank
            blank_number = len(blanks) - i
            placeholder = f"__{blank_number}__"
            
            start = blank.position
            end = start + len(blank.span)
            
            # Handle template placeholders vs regular words
            if blank.span.startswith('{') and blank.span.endswith('}'):
                # This is a template placeholder, replace entirely
                blanked_passage = (
                    blanked_passage[:start] + 
                    placeholder + 
                    blanked_passage[end:]
                )
            else:
                # This is a regular word, replace it
                blanked_passage = (
                    blanked_passage[:start] + 
                    placeholder + 
                    blanked_passage[end:]
                )
        
        return blanked_passage
    
    def _generate_rationale(self, blank: ClozeBlank) -> str:
        """Generate an explanation for why the answer is correct."""
        rationales = {
            "prepositions": {
                "at": "We use 'at' with specific times (at 3 o'clock) and specific places (at school, at home).",
                "in": "We use 'in' with months, years, and enclosed spaces (in January, in the room).",
                "on": "We use 'on' with days of the week and surfaces (on Monday, on the table)."
            },
            "articles": {
                "a": "We use 'a' before singular countable nouns that start with a consonant sound.",
                "an": "We use 'an' before singular countable nouns that start with a vowel sound.",
                "the": "We use 'the' when we talk about something specific or unique.",
                "": "We don't use an article with uncountable nouns or plural nouns in general."
            },
            "subject_verb_agreement": {
                "goes": "We use 'goes' with third person singular subjects (he, she, it).",
                "has": "We use 'has' with third person singular subjects (he, she, it).",
                "is": "We use 'is' with singular subjects (he, she, it, there is)."
            },
            "collocations": {
                "make": "We 'make' decisions, mistakes, money, and friends.",
                "do": "We 'do' homework, exercise, work, and our best.",
                "take": "We 'take' showers, medicine, photos, and breaks.",
                "have": "We 'have' breakfast, meetings, parties, and conversations."
            }
        }
        
        skill_rationales = rationales.get(blank.skill_tag, {})
        return skill_rationales.get(blank.answer, f"The correct answer is '{blank.answer}'.")
    
    def validate_cloze_question(self, question: ClozeQuestion) -> Dict[str, Any]:
        """
        Validate a cloze question for quality and correctness.
        
        Returns:
            Dictionary with comprehensive validation results
        """
        validation_results = {
            "valid": True,
            "issues": [],
            "quality_score": 0.0,
            "metrics": {},
            "grammar_validation": {},
            "single_answer_check": {},
            "ambiguity_analysis": {}
        }
        
        # Basic structural validation
        num_blanks = len(question.blanks)
        if num_blanks == 0:
            validation_results["valid"] = False
            validation_results["issues"].append("No blanks found")
        elif num_blanks > 5:
            validation_results["issues"].append("Too many blanks (>5)")
        
        # Check blank distribution
        if num_blanks > 0:
            passage_length = len(question.passage)
            blank_density = num_blanks / (passage_length / 100)  # blanks per 100 characters
            
            if blank_density > 10:  # More than 10 blanks per 100 characters
                validation_results["issues"].append("Blanks too dense")
        
        # Check distractor quality
        distractor_issues = 0
        for blank in question.blanks:
            if len(blank.distractors) < 2:
                distractor_issues += 1
            
            # Check for duplicate distractors
            if len(set(blank.distractors)) != len(blank.distractors):
                distractor_issues += 1
            
            # Check that answer is not in distractors
            if blank.answer in blank.distractors:
                distractor_issues += 1
        
        if distractor_issues > 0:
            validation_results["issues"].append(f"{distractor_issues} distractor issues found")
        
        # Grammar validation
        grammar_results = self.grammar_validator.validate_grammar(question.passage)
        validation_results["grammar_validation"] = grammar_results
        
        if not grammar_results["valid"]:
            validation_results["valid"] = False
            validation_results["issues"].extend([
                f"Grammar error: {error['description']}" for error in grammar_results["errors"]
            ])
        
        # Single answer guarantee check
        single_answer_results = self.grammar_validator.check_single_answer_guarantee(question)
        validation_results["single_answer_check"] = single_answer_results
        
        if not single_answer_results["guaranteed"]:
            validation_results["valid"] = False
            validation_results["issues"].extend(single_answer_results["issues"])
        
        # Ambiguity detection
        ambiguity_results = self.grammar_validator.detect_ambiguity(question)
        validation_results["ambiguity_analysis"] = ambiguity_results
        
        if ambiguity_results["ambiguous"] and ambiguity_results["severity"] in ["medium", "high"]:
            validation_results["issues"].append(f"Ambiguity detected (severity: {ambiguity_results['severity']})")
        
        # Calculate comprehensive quality score
        quality_factors = []
        
        # Blank count factor (optimal: 3-5 blanks)
        if 3 <= num_blanks <= 5:
            quality_factors.append(1.0)
        elif 1 <= num_blanks <= 2:
            quality_factors.append(0.7)
        else:
            quality_factors.append(0.3)
        
        # Distractor quality factor
        if distractor_issues == 0:
            quality_factors.append(1.0)
        else:
            quality_factors.append(max(0.0, 1.0 - (distractor_issues / num_blanks)))
        
        # Error type coverage factor
        target_errors = set(question.error_tags or [])
        covered_errors = set(blank.skill_tag for blank in question.blanks)
        coverage = len(covered_errors & target_errors) / len(target_errors) if target_errors else 1.0
        quality_factors.append(coverage)
        
        # Grammar quality factor
        quality_factors.append(grammar_results["score"])
        
        # Single answer factor
        quality_factors.append(1.0 if single_answer_results["guaranteed"] else 0.5)
        
        # Ambiguity factor
        if ambiguity_results["severity"] == "none":
            quality_factors.append(1.0)
        elif ambiguity_results["severity"] == "low":
            quality_factors.append(0.8)
        elif ambiguity_results["severity"] == "medium":
            quality_factors.append(0.6)
        else:  # high
            quality_factors.append(0.3)
        
        validation_results["quality_score"] = sum(quality_factors) / len(quality_factors)
        validation_results["metrics"] = {
            "num_blanks": num_blanks,
            "distractor_issues": distractor_issues,
            "error_coverage": coverage,
            "passage_length": len(question.passage),
            "grammar_score": grammar_results["score"],
            "single_answer_guaranteed": single_answer_results["guaranteed"],
            "ambiguity_severity": ambiguity_results["severity"]
        }
        
        return validation_results
    
    def generate_validated_cloze_question(
        self,
        level_cefr: CEFRLevel,
        target_error_tags: List[str],
        topic: Optional[str] = None,
        personalization: Optional[Dict[str, Any]] = None,
        passage_text: Optional[str] = None,
        max_attempts: int = 3
    ) -> Tuple[ClozeQuestion, Dict[str, Any]]:
        """
        Generate a cloze question with validation and ambiguity resolution.
        
        Args:
            level_cefr: Target CEFR level
            target_error_tags: List of error types to target
            topic: Optional topic/theme
            personalization: User-specific preferences and error history
            passage_text: Optional custom passage text
            max_attempts: Maximum attempts to generate a valid question
            
        Returns:
            Tuple of (generated question, validation results)
        """
        best_question = None
        best_validation = None
        best_score = 0.0
        
        for attempt in range(max_attempts):
            # Generate question
            question = self.generate_cloze_question(
                level_cefr=level_cefr,
                target_error_tags=target_error_tags,
                topic=topic,
                personalization=personalization,
                passage_text=passage_text
            )
            
            # Validate question
            validation = self.validate_cloze_question(question)
            
            # Try to resolve ambiguity if detected
            if validation["ambiguity_analysis"]["ambiguous"]:
                question = self.grammar_validator.resolve_ambiguity(
                    question, validation["ambiguity_analysis"]
                )
                # Re-validate after ambiguity resolution
                validation = self.validate_cloze_question(question)
            
            # Keep track of best question so far
            if validation["quality_score"] > best_score:
                best_question = question
                best_validation = validation
                best_score = validation["quality_score"]
            
            # If we have a valid question with good quality, use it
            if validation["valid"] and validation["quality_score"] > 0.8:
                break
        
        return best_question, best_validation


# Create service instances
english_error_taxonomy = EnglishErrorTaxonomy()
english_cloze_generator = EnglishClozeGenerator(english_error_taxonomy)

# Add convenience methods to the generator instance
def get_available_error_patterns():
    """Get list of available error pattern names."""
    return [error_type.value for error_type in ErrorType]

def get_available_cefr_levels():
    """Get list of available CEFR levels."""
    return list(CEFRLevel)

# Attach methods to the generator instance
english_cloze_generator.get_available_error_patterns = get_available_error_patterns
english_cloze_generator.get_available_cefr_levels = get_available_cefr_levels