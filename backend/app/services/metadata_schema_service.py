from typing import Dict, Any, Optional, List, Union
from datetime import datetime
import logging
from enum import Enum

logger = logging.getLogger(__name__)


class ContentType(Enum):
    """Standardized content types for vector database"""
    ERROR_PATTERN = "error_pattern"
    QUESTION = "question"
    PLACEMENT_TEST = "placement_test"
    CLOZE_QUESTION = "cloze_question"
    GRAMMAR_RULE = "grammar_rule"
    VOCABULARY = "vocabulary"
    CEFR_RUBRIC = "cefr_rubric"
    CEFR_EXAMPLE = "cefr_example"
    USER_ASSESSMENT = "user_assessment"
    MATH_CONCEPT = "math_concept"
    MATH_SOLUTION = "math_solution"


class Domain(Enum):
    """Standardized domains"""
    ENGLISH = "english"
    MATH = "math"
    CEFR = "cefr"
    GENERAL = "general"


class DifficultyLevel(Enum):
    """Standardized difficulty levels"""
    BEGINNER = "beginner"
    ELEMENTARY = "elementary"
    INTERMEDIATE = "intermediate"
    UPPER_INTERMEDIATE = "upper_intermediate"
    ADVANCED = "advanced"


class MetadataSchemaService:
    """Centralized service for standardizing metadata across all services"""
    
    def __init__(self):
        self.required_fields = {
            "domain": str,
            "content_type": str,
            "created_at": str,
            "obj_ref": str
        }
        
        self.optional_fields = {
            "user_id": str,
            "difficulty_level": (str, float, int),
            "topic_category": str,
            "skill_tag": str,
            "similarity_score": float,
            "generation_method": str,
            "source": str,
            "metadata_version": str
        }
    
    def build_standard_metadata(
        self,
        domain: str,
        content_type: str,
        obj_ref: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Build standardized metadata with required and optional fields"""
        try:
            # Validate domain and content type
            if not self._is_valid_domain(domain):
                raise ValueError(f"Invalid domain: {domain}")
            
            if not self._is_valid_content_type(content_type):
                raise ValueError(f"Invalid content_type: {content_type}")
            
            # Build base metadata
            metadata = {
                "domain": domain,
                "content_type": content_type,
                "obj_ref": str(obj_ref),
                "created_at": datetime.utcnow().isoformat(),
                "metadata_version": "2.0.0"
            }
            
            # Add optional fields with validation
            for key, value in kwargs.items():
                if key in self.optional_fields:
                    if self._validate_field_value(key, value):
                        metadata[key] = value
                    else:
                        logger.warning(f"Invalid value for field {key}: {value}")
                else:
                    logger.debug(f"Unknown field {key} will be stored as custom metadata")
                    metadata[f"custom_{key}"] = value
            
            # Validate final metadata
            self._validate_metadata(metadata)
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error building standard metadata: {e}")
            raise
    
    def build_error_pattern_metadata(
        self,
        domain: str,
        error_type: str,
        obj_ref: str,
        user_id: Optional[str] = None,
        topic_category: Optional[str] = None,
        skill_tag: Optional[str] = None,
        error_count: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Build standardized metadata for error patterns"""
        base_metadata = self.build_standard_metadata(
            domain=domain,
            content_type=ContentType.ERROR_PATTERN.value,
            obj_ref=obj_ref,
            **kwargs
        )
        
        # Add error pattern specific fields
        error_metadata = {
            **base_metadata,
            "error_type": error_type,
            "topic_category": topic_category,
            "skill_tag": skill_tag,
            "error_count": error_count
        }
        
        if user_id:
            error_metadata["user_id"] = str(user_id)
        
        return error_metadata
    
    def build_question_metadata(
        self,
        domain: str,
        question_type: str,
        obj_ref: str,
        difficulty_level: Optional[Union[str, float, int]] = None,
        topic_category: Optional[str] = None,
        skill_tag: Optional[str] = None,
        generation_method: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Build standardized metadata for questions"""
        base_metadata = self.build_standard_metadata(
            domain=domain,
            content_type=ContentType.QUESTION.value,
            obj_ref=obj_ref,
            **kwargs
        )
        
        # Add question specific fields
        question_metadata = {
            **base_metadata,
            "question_type": question_type,
            "topic_category": topic_category,
            "skill_tag": skill_tag,
            "generation_method": generation_method
        }
        
        if difficulty_level:
            question_metadata["difficulty_level"] = self._normalize_difficulty(difficulty_level)
        
        return question_metadata
    
    def build_cloze_question_metadata(
        self,
        domain: str,
        obj_ref: str,
        error_type_addressed: Optional[str] = None,
        skill_tag: Optional[str] = None,
        difficulty_level: Optional[Union[str, float, int]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Build standardized metadata for cloze questions"""
        base_metadata = self.build_standard_metadata(
            domain=domain,
            content_type=ContentType.CLOZE_QUESTION.value,
            obj_ref=obj_ref,
            **kwargs
        )
        
        # Add cloze question specific fields
        cloze_metadata = {
            **base_metadata,
            "error_type_addressed": error_type_addressed,
            "skill_tag": skill_tag
        }
        
        if difficulty_level:
            cloze_metadata["difficulty_level"] = self._normalize_difficulty(difficulty_level)
        
        return cloze_metadata
    
    def build_placement_test_metadata(
        self,
        domain: str,
        obj_ref: str,
        user_id: str,
        test_score: Optional[int] = None,
        test_accuracy: Optional[float] = None,
        skill_level: Optional[float] = None,
        difficulty_level: Optional[str] = None,
        topics_covered: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Build standardized metadata for placement tests"""
        base_metadata = self.build_standard_metadata(
            domain=domain,
            content_type=ContentType.PLACEMENT_TEST.value,
            obj_ref=obj_ref,
            **kwargs
        )
        
        # Add placement test specific fields
        placement_metadata = {
            **base_metadata,
            "user_id": str(user_id),
            "test_score": test_score,
            "test_accuracy": test_accuracy,
            "skill_level": skill_level,
            "difficulty_level": difficulty_level
        }
        
        if topics_covered:
            placement_metadata["topics_covered"] = topics_covered
        
        return placement_metadata
    
    def build_cefr_metadata(
        self,
        content_type: str,
        obj_ref: str,
        cefr_level: Optional[str] = None,
        skill_type: Optional[str] = None,
        rubric_type: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Build standardized metadata for CEFR content"""
        base_metadata = self.build_standard_metadata(
            domain=Domain.CEFR.value,
            content_type=content_type,
            obj_ref=obj_ref,
            **kwargs
        )
        
        # Add CEFR specific fields
        cefr_metadata = {
            **base_metadata,
            "cefr_level": cefr_level,
            "skill_type": skill_type,
            "rubric_type": rubric_type
        }
        
        return cefr_metadata
    
    def _is_valid_domain(self, domain: str) -> bool:
        """Validate domain value"""
        try:
            Domain(domain)
            return True
        except ValueError:
            return False
    
    def _is_valid_content_type(self, content_type: str) -> bool:
        """Validate content type value"""
        try:
            ContentType(content_type)
            return True
        except ValueError:
            return False
    
    def _validate_field_value(self, field_name: str, value: Any) -> bool:
        """Validate field value against expected type"""
        expected_type = self.optional_fields.get(field_name)
        if expected_type is None:
            return True
        
        # Handle union types
        if hasattr(expected_type, '__origin__') and expected_type.__origin__ is Union:
            return any(isinstance(value, t) for t in expected_type.__args__)
        
        return isinstance(value, expected_type)
    
    def _normalize_difficulty(self, difficulty: Union[str, float, int]) -> Union[str, float]:
        """Normalize difficulty level to standard format"""
        if isinstance(difficulty, str):
            # Convert string difficulty to numeric
            difficulty_mapping = {
                "beginner": 1.0,
                "elementary": 2.0,
                "intermediate": 3.0,
                "upper_intermediate": 4.0,
                "advanced": 5.0
            }
            return difficulty_mapping.get(difficulty.lower(), 3.0)
        elif isinstance(difficulty, (int, float)):
            # Ensure numeric difficulty is in range 1-5
            return max(1.0, min(5.0, float(difficulty)))
        else:
            return 3.0  # Default difficulty
    
    def _validate_metadata(self, metadata: Dict[str, Any]) -> bool:
        """Validate complete metadata"""
        # Check required fields
        for field, expected_type in self.required_fields.items():
            if field not in metadata:
                raise ValueError(f"Missing required field: {field}")
            
            if not isinstance(metadata[field], expected_type):
                raise ValueError(f"Invalid type for field {field}: expected {expected_type}, got {type(metadata[field])}")
        
        return True
    
    def get_metadata_schema(self, content_type: str) -> Dict[str, Any]:
        """Get metadata schema for specific content type"""
        schemas = {
            ContentType.ERROR_PATTERN.value: {
                "required": ["domain", "content_type", "obj_ref", "created_at", "error_type"],
                "optional": ["user_id", "topic_category", "skill_tag", "error_count", "similarity_score"],
                "example": {
                    "domain": "math",
                    "content_type": "error_pattern",
                    "obj_ref": "error_123",
                    "created_at": "2024-01-15T10:30:00",
                    "error_type": "algebra_equation",
                    "topic_category": "algebra",
                    "skill_tag": "equation_solving",
                    "error_count": 3
                }
            },
            ContentType.QUESTION.value: {
                "required": ["domain", "content_type", "obj_ref", "created_at", "question_type"],
                "optional": ["difficulty_level", "topic_category", "skill_tag", "generation_method"],
                "example": {
                    "domain": "math",
                    "content_type": "question",
                    "obj_ref": "question_456",
                    "created_at": "2024-01-15T10:30:00",
                    "question_type": "multiple_choice",
                    "difficulty_level": 3.5,
                    "topic_category": "geometry",
                    "skill_tag": "area_calculation"
                }
            },
            ContentType.CLOZE_QUESTION.value: {
                "required": ["domain", "content_type", "obj_ref", "created_at"],
                "optional": ["error_type_addressed", "skill_tag", "difficulty_level", "generation_method"],
                "example": {
                    "domain": "english",
                    "content_type": "cloze_question",
                    "obj_ref": "cloze_789",
                    "created_at": "2024-01-15T10:30:00",
                    "error_type_addressed": "grammar",
                    "skill_tag": "verb_tense",
                    "difficulty_level": 3.0,
                    "generation_method": "embedding_enhanced"
                }
            }
        }
        
        return schemas.get(content_type, {
            "required": ["domain", "content_type", "obj_ref", "created_at"],
            "optional": ["user_id", "difficulty_level", "topic_category", "skill_tag"],
            "example": {}
        })
    
    def validate_existing_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and fix existing metadata to match standard schema"""
        try:
            # Ensure required fields exist
            for field in self.required_fields:
                if field not in metadata:
                    if field == "created_at":
                        metadata[field] = datetime.utcnow().isoformat()
                    elif field == "metadata_version":
                        metadata[field] = "2.0.0"
                    else:
                        logger.warning(f"Missing required field: {field}")
            
            # Normalize difficulty level if present
            if "difficulty_level" in metadata:
                metadata["difficulty_level"] = self._normalize_difficulty(metadata["difficulty_level"])
            
            # Ensure domain is valid
            if "domain" in metadata and not self._is_valid_domain(metadata["domain"]):
                logger.warning(f"Invalid domain: {metadata['domain']}, setting to 'general'")
                metadata["domain"] = "general"
            
            # Ensure content_type is valid
            if "content_type" in metadata and not self._is_valid_content_type(metadata["content_type"]):
                logger.warning(f"Invalid content_type: {metadata['content_type']}, setting to 'question'")
                metadata["content_type"] = "question"
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error validating existing metadata: {e}")
            return metadata


# Global instance
metadata_schema_service = MetadataSchemaService()
