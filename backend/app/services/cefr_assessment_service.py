import logging
from typing import Dict, Any, Optional
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
            # 1. Retrieve CEFR Rubric and Examples (RAG part)
            # Use retriever to get relevant rubric sections based on assessment_type or skills.
            query = f"CEFR rubric for {assessment_type} assessment, focusing on grammar, vocabulary, reading, writing, listening, and speaking skills."
            retrieved_rubrics = await self.retriever.retrieve_documents(
                query=query,
                top_k=5 # Retrieve top 5 relevant rubric passages
            )
            cefr_rubric_context = "\n".join([doc.content for doc in retrieved_rubrics])
            
            if not cefr_rubric_context:
                # Fallback to full rubric if RAG doesn't return anything
                cefr_rubric_context = self.cefr_validator.get_full_rubric_text()

            # 2. Construct the prompt for the LLM
            system_prompt = self.prompt_engineer.get_cefr_assessment_system_prompt()
            
            user_prompt = f"""
            Assess the CEFR level of the following {assessment_type} provided by the user.
            Focus on grammar, vocabulary, reading, writing, listening, and speaking skills.
            Provide an overall CEFR level and individual skill levels.
            Also, provide a confidence score for your assessment.

            Assessment Text:
            {assessment_text}

            CEFR Rubric Context:
            {cefr_rubric_context}

            Ensure your response strictly adheres to the provided JSON schema.
            """

            # 3. Call MCP for CEFR assessment
            from app.core.mcp_utils import mcp_utils
            
            try:
                if mcp_utils.is_initialized:
                    # MCP üzerinden CEFR assessment
                    mcp_response = await mcp_utils.call_tool(
                        tool_name="assess_cefr",
                        arguments={
                            "student_id": user_id,
                            "text": assessment_text,
                            "assessment_type": assessment_type,
                            "language": "english",
                            "detailed_feedback": True
                        }
                    )
                    
                    if mcp_response["success"]:
                        # MCP response'u parse et
                        assessment_data = mcp_response["data"]
                        if isinstance(assessment_data, str):
                            import json
                            assessment_data = json.loads(assessment_data)
                        
                        assessment_result = CEFRAssessmentResponse(**assessment_data)
                        return assessment_result
                    else:
                        logger.warning(f"MCP CEFR assessment failed: {mcp_response.get('error')}")
                        # Fallback to direct LLM
                else:
                    logger.warning("MCP not initialized, using direct LLM")
            except Exception as e:
                logger.warning(f"MCP CEFR assessment failed, using fallback: {e}")
            
            # Fallback: Direct LLM Gateway call
            llm_response = await llm_gateway.generate_json(
                prompt=user_prompt,
                system_prompt=system_prompt,
                schema=CEFRAssessmentResponse.model_json_schema(),
                context=cefr_rubric_context # Pass context for LLM to use
            )

            if not llm_response["success"]:
                logger.error(f"LLM Gateway failed for CEFR assessment: {llm_response.get('error')}")
                raise Exception(f"LLM assessment failed: {llm_response.get('error')}")

            # 4. Validate and return the response
            parsed_json = llm_response["parsed_json"]
            assessment_result = CEFRAssessmentResponse(**parsed_json)
            
            # 5. Save assessment_result to user table
            user = await self.user_repository.get_by_id(session, user_id)
            if user:
                # Update overall English level
                # Map CEFR string level (e.g., "B2") to an integer or specific representation if needed
                # For simplicity, let's assume current_english_level can store the string directly or map it.
                # If current_english_level is an int, you'd need a mapping (e.g., A1=1, A2=2, etc.)
                user.current_english_level = assessment_result.overall_level # Assuming direct assignment

                # Update skill scores (if user model supports it)
                if hasattr(user, 'skill_scores') and isinstance(user.skill_scores, dict):
                    for skill, level in assessment_result.skills.items():
                        user.skill_scores[f'english_{skill}'] = level # Store as english_reading, english_writing etc.
                else:
                    user.skill_scores = {f'english_{skill}': level for skill, level in assessment_result.skills.items()}
                
                # Update confidence if needed
                # user.cefr_assessment_confidence = assessment_result.confidence # Assuming such a field exists

                await self.user_repository.update(session, user)
                logger.info(f"User {user_id} CEFR level updated to {user.current_english_level} with skills: {user.skill_scores}")
            else:
                logger.warning(f"User {user_id} not found for CEFR assessment update.")

            return assessment_result

        except Exception as e:
            logger.error(f"Error during CEFR assessment for user {user_id}: {e}", exc_info=True)
            raise

cefr_assessment_service = CEFRAssessmentService(user_repository=UserRepository(), retriever_service=HybridRetriever())