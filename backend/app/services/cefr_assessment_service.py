import logging
from typing import Dict, Any, Optional, List
from pydantic import ValidationError, TypeAdapter
import json
import re

from app.services.llm_gateway import llm_gateway
from app.schemas.cefr_assessment import CEFRAssessmentResponse
from app.services.cefr_validator import CEFRValidator
from app.services.prompt_engineering import PromptEngineeringService
from app.repositories.user_repository import UserRepository
from app.services.retriever import HybridRetriever
from sqlalchemy.ext.asyncio import AsyncSession
# --- YENİ EKLENENLER ---
from app.services.embedding_service import embedding_service
from app.services.vector_index_manager import vector_index_manager
# --- BİTİŞ ---

logger = logging.getLogger(__name__)

class CEFRAssessmentService:
    def __init__(self, user_repository: UserRepository, retriever_service: HybridRetriever):
        self.cefr_validator = CEFRValidator()
        self.prompt_engineer = PromptEngineeringService()
        self.user_repository = user_repository
        self.retriever = retriever_service
        
        # CEFR level constraints
        self.valid_cefr_levels = ["A1", "A2", "B1", "B2", "C1", "C2"]
        self.valid_skills = ["grammar", "vocabulary", "reading", "writing", "listening", "speaking"]
        
        # Rubric constraints for RAG
        self.rubric_constraints = {
            "A1": "Basic user - Can understand and use familiar everyday expressions",
            "A2": "Basic user - Can communicate in simple and routine tasks",
            "B1": "Independent user - Can deal with most situations likely to arise",
            "B2": "Independent user - Can interact with a degree of fluency",
            "C1": "Proficient user - Can express ideas fluently and spontaneously",
            "C2": "Proficient user - Can understand with ease virtually everything"
        }
        
        # Embedding-based assessment weights
        self.assessment_weights = {
            "text_complexity": 0.3,
            "vocabulary_richness": 0.25,
            "grammar_accuracy": 0.25,
            "rubric_similarity": 0.2
        }

    async def assess_cefr_level(
        self,
        user_id: str,
        assessment_text: str,
        assessment_type: str,
        session: AsyncSession
    ) -> CEFRAssessmentResponse:
        """Enhanced CEFR assessment using embeddings and semantic search"""
        try:
            logger.info(f"Starting enhanced CEFR assessment for user {user_id}")
            
            # 1. Generate embedding for assessment text
            assessment_embedding = await embedding_service.get_embedding(assessment_text)
            
            # 2. Analyze text characteristics using embeddings
            text_analysis = await self._analyze_text_with_embeddings(assessment_text, assessment_embedding)
            
            # 3. Retrieve relevant CEFR rubrics using semantic search
            cefr_rubrics = await self._get_cefr_rubrics_by_semantic_search(
                assessment_embedding, assessment_type, text_analysis
            )
            
            # 4. Build enhanced prompt with semantic context
            enhanced_prompt = self._build_enhanced_assessment_prompt(
                assessment_text, cefr_rubrics, text_analysis
            )
            
            # 5. Perform assessment with enhanced context
            assessment_result = await self._perform_enhanced_assessment(
                enhanced_prompt, user_id, assessment_text, assessment_type
            )
            
            # 6. Validate and save results
            await self._save_assessment_results(user_id, assessment_result, session)
            
            # 7. Store assessment embedding for future reference
            await self._store_assessment_embedding(
                user_id, assessment_text, assessment_embedding, assessment_result
            )
            
            logger.info(f"Enhanced CEFR assessment completed for user {user_id}")
            return assessment_result
            
        except Exception as e:
            logger.error(f"Error during enhanced CEFR assessment for user {user_id}: {e}", exc_info=True)
            # Fallback to basic assessment
            return await self._fallback_assessment(user_id, assessment_text, assessment_type)

    async def _analyze_text_with_embeddings(
        self, 
        assessment_text: str, 
        text_embedding: List[float]
    ) -> Dict[str, Any]:
        """Analyze assessment text characteristics using embeddings"""
        try:
            analysis = {}
            
            # Text complexity analysis
            word_count = len(assessment_text.split())
            sentence_count = len(assessment_text.split('.'))
            avg_sentence_length = word_count / max(sentence_count, 1)
            
            analysis["text_complexity"] = {
                "word_count": word_count,
                "sentence_count": sentence_count,
                "avg_sentence_length": avg_sentence_length,
                "complexity_score": min(1.0, avg_sentence_length / 20.0)  # Normalize to 0-1
            }
            
            # Vocabulary richness estimation (using embedding variance)
            if len(text_embedding) > 1:
                embedding_variance = sum((x - sum(text_embedding)/len(text_embedding))**2 for x in text_embedding)
                analysis["vocabulary_richness"] = {
                    "embedding_variance": embedding_variance,
                    "richness_score": min(1.0, embedding_variance / 1000.0)  # Normalize
                }
            else:
                analysis["vocabulary_richness"] = {"richness_score": 0.5}
            
            # Grammar accuracy estimation (basic heuristics)
            grammar_indicators = {
                "complex_structures": len(re.findall(r'\b(if|when|although|however|therefore|furthermore)\b', assessment_text, re.IGNORECASE)),
                "passive_voice": len(re.findall(r'\b(am|is|are|was|were|be|been|being)\s+\w+ed\b', assessment_text, re.IGNORECASE)),
                "subjunctive": len(re.findall(r'\b(would|could|should|might)\s+\w+\b', assessment_text, re.IGNORECASE))
            }
            
            grammar_score = min(1.0, sum(grammar_indicators.values()) / 10.0)
            analysis["grammar_accuracy"] = {
                "indicators": grammar_indicators,
                "accuracy_score": grammar_score
            }
            
            # Overall text level estimation
            overall_score = (
                analysis["text_complexity"]["complexity_score"] * self.assessment_weights["text_complexity"] +
                analysis["vocabulary_richness"]["richness_score"] * self.assessment_weights["vocabulary_richness"] +
                analysis["grammar_accuracy"]["accuracy_score"] * self.assessment_weights["grammar_accuracy"]
            )
            
            analysis["estimated_level"] = self._map_score_to_cefr_level(overall_score)
            analysis["overall_score"] = overall_score
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing text with embeddings: {e}")
            return {"estimated_level": "B1", "overall_score": 0.5}

    async def _get_cefr_rubrics_by_semantic_search(
        self,
        assessment_embedding: List[float],
        assessment_type: str,
        text_analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Retrieve CEFR rubrics using semantic search"""
        try:
            # Search for CEFR rubrics in vector index
            similar_rubrics = await vector_index_manager.search_similar_content(
                assessment_embedding,
                namespace="cefr_rubrics",
                similarity_threshold=0.6,
                limit=10
            )
            
            # Filter rubrics by assessment type and estimated level
            filtered_rubrics = []
            estimated_level = text_analysis.get("estimated_level", "B1")
            
            for rubric in similar_rubrics:
                rubric_data = rubric.get("metadata", {})
                rubric_level = rubric_data.get("cefr_level", "")
                rubric_type = rubric_data.get("assessment_type", "")
                
                # Check if rubric matches assessment type and level
                if (rubric_type == assessment_type or not rubric_type) and (
                    rubric_level == estimated_level or 
                    self._is_adjacent_level(rubric_level, estimated_level)
                ):
                    filtered_rubrics.append({
                        "content": rubric.get("content", ""),
                        "cefr_level": rubric_level,
                        "assessment_type": rubric_type,
                        "similarity_score": rubric.get("similarity", 0.0),
                        "metadata": rubric_data
                    })
            
            # Sort by similarity score
            filtered_rubrics.sort(key=lambda x: x["similarity_score"], reverse=True)
            
            # If no rubrics found, use default ones
            if not filtered_rubrics:
                filtered_rubrics = self._get_default_cefr_rubrics(estimated_level, assessment_type)
            
            logger.info(f"Retrieved {len(filtered_rubrics)} CEFR rubrics for {assessment_type}")
            return filtered_rubrics
            
        except Exception as e:
            logger.error(f"Error retrieving CEFR rubrics: {e}")
            return self._get_default_cefr_rubrics("B1", assessment_type)

    def _is_adjacent_level(self, level1: str, level2: str) -> bool:
        """Check if two CEFR levels are adjacent"""
        level_order = ["A1", "A2", "B1", "B2", "C1", "C2"]
        try:
            idx1 = level_order.index(level1)
            idx2 = level_order.index(level2)
            return abs(idx1 - idx2) <= 1
        except ValueError:
            return False

    def _get_default_cefr_rubrics(self, level: str, assessment_type: str) -> List[Dict[str, Any]]:
        """Get default CEFR rubrics when semantic search fails"""
        default_rubrics = []
        
        # Add level-specific rubric
        if level in self.rubric_constraints:
            default_rubrics.append({
                "content": self.rubric_constraints[level],
                "cefr_level": level,
                "assessment_type": assessment_type,
                "similarity_score": 1.0,
                "metadata": {"source": "default"}
            })
        
        # Add adjacent levels
        level_order = ["A1", "A2", "B1", "B2", "C1", "C2"]
        try:
            level_idx = level_order.index(level)
            for offset in [-1, 1]:
                adj_idx = level_idx + offset
                if 0 <= adj_idx < len(level_order):
                    adj_level = level_order[adj_idx]
                    if adj_level in self.rubric_constraints:
                        default_rubrics.append({
                            "content": self.rubric_constraints[adj_level],
                            "cefr_level": adj_level,
                            "assessment_type": assessment_type,
                            "similarity_score": 0.8,
                            "metadata": {"source": "default_adjacent"}
                        })
        except ValueError:
            pass
        
        return default_rubrics

    def _build_enhanced_assessment_prompt(
        self,
        assessment_text: str,
        cefr_rubrics: List[Dict[str, Any]],
        text_analysis: Dict[str, Any]
    ) -> str:
        """Build enhanced assessment prompt with semantic context"""
        
        # Build rubric context
        rubric_context = "\n".join([
            f"Level {r['cefr_level']}: {r['content']} (Similarity: {r['similarity_score']:.2f})"
            for r in cefr_rubrics[:5]  # Top 5 rubrics
        ])
        
        # Build text analysis context
        analysis_context = f"""
        Text Analysis:
        - Word Count: {text_analysis.get('text_complexity', {}).get('word_count', 0)}
        - Sentence Count: {text_analysis.get('text_complexity', {}).get('sentence_count', 0)}
        - Average Sentence Length: {text_analysis.get('text_complexity', {}).get('avg_sentence_length', 0):.1f}
        - Vocabulary Richness Score: {text_analysis.get('vocabulary_richness', {}).get('richness_score', 0):.2f}
        - Grammar Accuracy Score: {text_analysis.get('grammar_accuracy', {}).get('accuracy_score', 0):.2f}
        - Estimated Level: {text_analysis.get('estimated_level', 'Unknown')}
        """
        
        return f"""
        Assess the CEFR level of the following {assessment_type} text using the provided context.
        
        STRICT CONSTRAINTS:
        - Overall level MUST be one of: {self.valid_cefr_levels}
        - Skills MUST include: {self.valid_skills}
        - Confidence MUST be between 0.0 and 1.0
        - All skill levels MUST be valid CEFR levels

        Assessment Text:
        {assessment_text}

        Text Analysis:
        {analysis_context}

        Relevant CEFR Rubrics:
        {rubric_context}

        Consider the text analysis and rubric similarity scores when making your assessment.
        Respond ONLY in the exact JSON schema format provided.
        """

    def _map_score_to_cefr_level(self, score: float) -> str:
        """Map numerical score to CEFR level"""
        if score < 0.2:
            return "A1"
        elif score < 0.4:
            return "A2"
        elif score < 0.6:
            return "B1"
        elif score < 0.8:
            return "B2"
        elif score < 0.9:
            return "C1"
        else:
            return "C2"

    async def _store_assessment_embedding(
        self,
        user_id: str,
        assessment_text: str,
        assessment_embedding: List[float],
        assessment_result: CEFRAssessmentResponse
    ):
        """Store assessment embedding for future reference and analysis"""
        try:
            # Store in vector index for future similarity searches
            await vector_index_manager.upsert_embedding(
                obj_ref=f"assessment_{user_id}_{hash(assessment_text)}",
                namespace="user_assessments",
                embedding=assessment_embedding,
                metadata={
                    "user_id": user_id,
                    "assessment_type": "cefr",
                    "cefr_level": assessment_result.overall_level,
                    "confidence": assessment_result.confidence,
                    "skills": assessment_result.skills,
                    "text_length": len(assessment_text),
                    "timestamp": assessment_result.timestamp
                }
            )
            
            logger.info(f"Stored assessment embedding for user {user_id}")
            
        except Exception as e:
            logger.error(f"Error storing assessment embedding: {e}")

    async def _fallback_assessment(
        self,
        user_id: str,
        assessment_text: str,
        assessment_type: str
    ) -> CEFRAssessmentResponse:
        """Fallback assessment method when embedding-based approach fails"""
        try:
            logger.info(f"Using fallback assessment for user {user_id}")
            
            # Simple text-based assessment
            word_count = len(assessment_text.split())
            if word_count < 20:
                estimated_level = "A1"
            elif word_count < 50:
                estimated_level = "A2"
            elif word_count < 100:
                estimated_level = "B1"
            elif word_count < 200:
                estimated_level = "B2"
            elif word_count < 400:
                estimated_level = "C1"
            else:
                estimated_level = "C2"
            
            # Create basic assessment response
            return CEFRAssessmentResponse(
                overall_level=estimated_level,
                confidence=0.6,  # Lower confidence for fallback
                skills={
                    "grammar": estimated_level,
                    "vocabulary": estimated_level,
                    "reading": estimated_level,
                    "writing": estimated_level,
                    "listening": estimated_level,
                    "speaking": estimated_level
                },
                timestamp="2024-01-01T00:00:00Z",
                assessment_method="fallback_text_analysis"
            )
            
        except Exception as e:
            logger.error(f"Error in fallback assessment: {e}")
            # Return default response
            return CEFRAssessmentResponse(
                overall_level="B1",
                confidence=0.5,
                skills={
                    "grammar": "B1",
                    "vocabulary": "B1",
                    "reading": "B1",
                    "writing": "B1",
                    "listening": "B1",
                    "speaking": "B1"
                },
                timestamp="2024-01-01T00:00:00Z",
                assessment_method="error_fallback"
            )

    async def _assess_with_retry(self, user_prompt: str, system_prompt: str, user_id: str, assessment_text: str, assessment_type: str, max_retries: int = 3) -> CEFRAssessmentResponse:
        """Assesses CEFR level with retry logic and self-repair."""
        for attempt in range(max_retries):
            try:
                # Try MCP first
                from app.core.mcp_utils import mcp_utils
                
                if mcp_utils.is_initialized:
                    try:
                    mcp_response = await mcp_utils.call_tool(
                        tool_name="assess_cefr",
                        arguments={
                            "student_id": user_id,
                            "text": assessment_text,
                            "assessment_type": assessment_type,
                            "language": "english",
                                "detailed_feedback": True,
                                "strict_validation": True
                        }
                    )
                    
                    if mcp_response["success"]:
                        assessment_data = mcp_response["data"]
                        if isinstance(assessment_data, str):
                            assessment_data = json.loads(assessment_data)
                        
                            # Validate and repair if needed
                            validated_data = self._validate_and_repair_assessment(assessment_data)
                            if validated_data:
                                return CEFRAssessmentResponse(**validated_data)
                            else:
                                logger.warning(f"MCP assessment validation failed on attempt {attempt + 1}")
                                continue
                    else:
                        logger.warning(f"MCP CEFR assessment failed: {mcp_response.get('error')}")
                    except Exception as e:
                        logger.warning(f"MCP CEFR assessment failed, using fallback: {e}")
                else:
                    logger.warning("MCP not initialized, using direct LLM")
            
            # Fallback: Direct LLM Gateway call
            llm_response = await llm_gateway.generate_json(
                prompt=user_prompt,
                system_prompt=system_prompt,
                schema=CEFRAssessmentResponse.model_json_schema(),
                    max_retries=2
            )

            if not llm_response["success"]:
                logger.error(f"LLM Gateway failed for CEFR assessment: {llm_response.get('error')}")
                    continue

                # Validate and repair parsed JSON
            parsed_json = llm_response["parsed_json"]
                validated_data = self._validate_and_repair_assessment(parsed_json)
                
                if validated_data:
                    return CEFRAssessmentResponse(**validated_data)
                else:
                    logger.warning(f"Assessment validation failed on attempt {attempt + 1}")
                    continue
                    
            except Exception as e:
                logger.error(f"Assessment attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    raise Exception(f"CEFR assessment failed after {max_retries} attempts")
                continue
        
        raise Exception(f"CEFR assessment failed after {max_retries} attempts")

    def _validate_and_repair_assessment(self, assessment_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Validates and repairs CEFR assessment data."""
        try:
            # Check required fields
            required_fields = ["overall_level", "skills", "confidence"]
            for field in required_fields:
                if field not in assessment_data:
                    logger.warning(f"Missing required field: {field}")
                    return None
            
            # Validate overall level
            if assessment_data["overall_level"] not in self.valid_cefr_levels:
                logger.warning(f"Invalid overall level: {assessment_data['overall_level']}")
                # Try to repair by finding closest valid level
                assessment_data["overall_level"] = self._find_closest_cefr_level(assessment_data["overall_level"])
            
            # Validate skills
            if not isinstance(assessment_data["skills"], dict):
                logger.warning("Skills must be a dictionary")
                return None
            
            # Ensure all required skills are present
            for skill in self.valid_skills:
                if skill not in assessment_data["skills"]:
                    assessment_data["skills"][skill] = assessment_data["overall_level"]  # Default to overall level
                elif assessment_data["skills"][skill] not in self.valid_cefr_levels:
                    assessment_data["skills"][skill] = assessment_data["overall_level"]  # Repair invalid skill level
            
            # Validate confidence
            confidence = assessment_data["confidence"]
            if not isinstance(confidence, (int, float)) or confidence < 0.0 or confidence > 1.0:
                assessment_data["confidence"] = 0.5  # Default confidence
            
            # Validate with Pydantic
            validated_result = CEFRAssessmentResponse(**assessment_data)
            return assessment_data
            
        except ValidationError as e:
            logger.error(f"Validation error: {e.errors()}")
            return None
        except Exception as e:
            logger.error(f"Error in validation and repair: {e}")
            return None

    def _find_closest_cefr_level(self, invalid_level: str) -> str:
        """Finds the closest valid CEFR level for an invalid input."""
        # Simple mapping for common mistakes
        level_mapping = {
            "A0": "A1", "A3": "A2", "B0": "B1", "B3": "B2", "C0": "C1", "C3": "C2",
            "beginner": "A1", "elementary": "A2", "intermediate": "B1", 
            "upper_intermediate": "B2", "advanced": "C1", "proficient": "C2"
        }
        
        if invalid_level in level_mapping:
            return level_mapping[invalid_level]
        
        # Default to B1 for unknown levels
        return "B1"

    async def _save_assessment_results(self, user_id: str, assessment_result: CEFRAssessmentResponse, session: AsyncSession):
        """Saves assessment results to user profile."""
        try:
            user = await self.user_repository.get_by_id(session, user_id)
            if user:
                # Update overall English level
                user.current_english_level = assessment_result.overall_level

                # Update skill scores
                if hasattr(user, 'skill_scores') and isinstance(user.skill_scores, dict):
                    for skill, level in assessment_result.skills.items():
                        user.skill_scores[f'english_{skill}'] = level
                else:
                    user.skill_scores = {f'english_{skill}': level for skill, level in assessment_result.skills.items()}
                
                # Add assessment metadata
                if not hasattr(user, 'assessment_history'):
                    user.assessment_history = []
                
                user.assessment_history.append({
                    "timestamp": assessment_result.timestamp if hasattr(assessment_result, 'timestamp') else None,
                    "overall_level": assessment_result.overall_level,
                    "confidence": assessment_result.confidence,
                    "skills": assessment_result.skills
                })

                await self.user_repository.update(session, user)
                logger.info(f"User {user_id} CEFR level updated to {user.current_english_level}")
            else:
                logger.warning(f"User {user_id} not found for CEFR assessment update.")

        except Exception as e:
            logger.error(f"Error saving assessment results: {e}")

cefr_assessment_service = CEFRAssessmentService(user_repository=UserRepository(), retriever_service=HybridRetriever())