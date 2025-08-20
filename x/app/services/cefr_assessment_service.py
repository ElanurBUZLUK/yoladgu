import logging
from typing import Dict, Any, Optional, List
from pydantic import ValidationError, TypeAdapter
import json
import re

from app.services.llm_gateway import llm_gateway
from app.schemas.cefr_assessment import CEFRAssessmentResponse
from app.services.cefr_validator import CEFRValidator # Assuming this exists for rubric
from app.services.prompt_engineering import PromptEngineeringService # Assuming this exists for prompt adjustments
from app.repositories.user_repository import UserRepository # NEW IMPORT
from app.services.retriever import HybridRetriever # Import HybridRetriever
from sqlalchemy.ext.asyncio import AsyncSession # For session injection

logger = logging.getLogger(__name__)

class CEFRAssessmentService:
    def __init__(self, user_repository: UserRepository, retriever_service: HybridRetriever):
        self.cefr_validator = CEFRValidator()
        self.prompt_engineer = PromptEngineeringService()
        self.user_repository = user_repository # NEW: Inject UserRepository
        self.retriever = retriever_service # NEW: Inject Retriever
        
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

    async def assess_cefr_level(
        self,
        user_id: str,
        assessment_text: str,
        assessment_type: str,
        session: AsyncSession # Add session parameter for DB operations
    ) -> CEFRAssessmentResponse:
        """Assesses the user's CEFR level based on provided text using LLM.
        Integrates CEFR rubrics via RAG and expects structured JSON output.
        """
        try:
            # 1. Retrieve CEFR Rubric and Examples (RAG part) with constraints
            cefr_rubric_context = await self._get_constrained_rubric_context(assessment_type, assessment_text)
            
            # 2. Construct the prompt for the LLM with strict constraints
            system_prompt = self._get_strict_cefr_system_prompt()
            
            user_prompt = f"""
            Assess the CEFR level of the following {assessment_type} provided by the user.
            
            STRICT CONSTRAINTS:
            - Overall level MUST be one of: {self.valid_cefr_levels}
            - Skills MUST include: {self.valid_skills}
            - Confidence MUST be between 0.0 and 1.0
            - All skill levels MUST be valid CEFR levels
            
            Assessment Text:
            {assessment_text}

            CEFR Rubric Context:
            {cefr_rubric_context}

            Respond ONLY in the exact JSON schema format provided.
            """

            # 3. Call MCP for CEFR assessment with retry logic
            assessment_result = await self._assess_with_retry(user_prompt, system_prompt, user_id, assessment_text, assessment_type)
            
            # 4. Validate and save results
            await self._save_assessment_results(user_id, assessment_result, session)
            
            return assessment_result

        except Exception as e:
            logger.error(f"Error during CEFR assessment for user {user_id}: {e}", exc_info=True)
            raise

    async def _get_constrained_rubric_context(self, assessment_type: str, assessment_text: str) -> str:
        """Gets CEFR rubric context with RAG constraints."""
        try:
            # Build constrained query
            query = f"CEFR rubric for {assessment_type} assessment"
            
            # Add text-based constraints
            text_length = len(assessment_text.split())
            if text_length < 50:
                query += " basic level A1-A2"
            elif text_length < 150:
                query += " intermediate level B1-B2"
            else:
                query += " advanced level C1-C2"
            
            retrieved_rubrics = await self.retriever.retrieve_documents(
                query=query,
                top_k=3  # Limit to top 3 for focused context
            )
            
            cefr_rubric_context = "\n".join([doc.content for doc in retrieved_rubrics])
            
            if not cefr_rubric_context:
                # Fallback to full rubric if RAG doesn't return anything
                cefr_rubric_context = self.cefr_validator.get_full_rubric_text()
            
            # Add constraint headers
            constraint_header = "CEFR LEVEL CONSTRAINTS:\n"
            for level, description in self.rubric_constraints.items():
                constraint_header += f"- {level}: {description}\n"
            
            return constraint_header + "\n" + cefr_rubric_context
            
        except Exception as e:
            logger.error(f"Error getting rubric context: {e}")
            return self.cefr_validator.get_full_rubric_text()

    def _get_strict_cefr_system_prompt(self) -> str:
        """Returns a strict system prompt for CEFR assessment."""
        return f"""
        You are a certified CEFR assessor. You must assess English proficiency according to strict CEFR standards.
        
        VALID CEFR LEVELS: {self.valid_cefr_levels}
        REQUIRED SKILLS: {self.valid_skills}
        
        You MUST:
        1. Use ONLY valid CEFR levels (A1, A2, B1, B2, C1, C2)
        2. Assess ALL required skills
        3. Provide confidence score between 0.0 and 1.0
        4. Follow the exact JSON schema format
        5. Base assessment on CEFR rubric criteria only
        
        Any deviation from these constraints will result in assessment failure.
        """

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