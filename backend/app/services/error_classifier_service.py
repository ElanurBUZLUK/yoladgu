import logging
from typing import Dict, Any, List
from pydantic import ValidationError

from app.services.llm_gateway import llm_gateway
from app.schemas.error_pattern import ClassifiedErrorSchema

logger = logging.getLogger(__name__)

class ErrorClassifierService:
    def __init__(self, llm_gateway_service: llm_gateway):
        self.llm_gateway = llm_gateway_service
        self.few_shot_examples = [
            {
                "input": "He go to the store.",
                "output": {
                    "error_type": "Grammar: Subject-Verb Agreement",
                    "explanation": "The verb 'go' does not agree with the third-person singular subject 'He'. It should be 'goes'.",
                    "confidence": 0.95,
                    "suggested_correction": "He goes to the store.",
                    "relevant_rule": "Subject-verb agreement: singular subjects take singular verbs."
                }
            },
            {
                "input": "I like apple, banana and orange.",
                "output": {
                    "error_type": "Punctuation: Missing Comma",
                    "explanation": "A comma is needed before 'and' in a list of three or more items (Oxford comma).",
                    "confidence": 0.88,
                    "suggested_correction": "I like apple, banana, and orange.",
                    "relevant_rule": "Use commas to separate items in a list."
                }
            },
            {
                "input": "Their is no way.",
                "output": {
                    "error_type": "Spelling: Homophone Confusion",
                    "explanation": "'Their' is a possessive pronoun. The correct word here is 'There', which indicates a place or existence.",
                    "confidence": 0.98,
                    "suggested_correction": "There is no way.",
                    "relevant_rule": "Distinguish between homophones like 'their', 'there', and 'they're'."
                }
            }
        ]

    async def classify_error(self, text: str) -> Optional[ClassifiedErrorSchema]:
        """Classifies the type of error in the given text using LLM few-shot.
        """
        prompt = f"""
        Analyze the following text for grammatical or spelling errors and classify the primary error type.
        Provide an explanation, a confidence score (0.0-1.0), a suggested correction, and a relevant grammar rule.
        
        Text to analyze: "{text}"
        
        Respond in JSON format according to the following schema:
        {ClassifiedErrorSchema.model_json_schema()}
        
        Here are some examples:
        """
        for example in self.few_shot_examples:
            prompt += f"""
        Input: "{example['input']}"
        Output: {example['output']}
        """

        system_prompt = "You are an expert linguistic analyst and grammar checker. You must classify errors and provide corrections in the specified JSON format."

        try:
            # Try MCP first
            from app.core.mcp_utils import mcp_utils
            
            if mcp_utils.is_initialized:
                try:
                    mcp_response = await mcp_utils.call_tool(
                        tool_name="classify_error",
                        arguments={
                            "text": text,
                            "prompt": prompt,
                            "system_prompt": system_prompt,
                            "schema": ClassifiedErrorSchema.model_json_schema()
                        }
                    )
                    
                    if mcp_response["success"]:
                        # Parse MCP response
                        classified_error_data = mcp_response["data"]
                        if isinstance(classified_error_data, str):
                            import json
                            classified_error_data = json.loads(classified_error_data)
                        
                        validated_error = ClassifiedErrorSchema(**classified_error_data)
                        logger.info(f"Successfully classified error via MCP: {validated_error.error_type}")
                        return validated_error
                    else:
                        logger.warning(f"MCP error classification failed: {mcp_response.get('error')}")
                        # Fallback to direct LLM
                except Exception as e:
                    logger.warning(f"MCP error classification failed, using fallback: {e}")
            else:
                logger.warning("MCP not initialized, using direct LLM")
            
            # Fallback: Direct LLM call
            llm_response = await self.llm_gateway.generate_json(
                prompt=prompt,
                system_prompt=system_prompt,
                schema=ClassifiedErrorSchema.model_json_schema(),
                max_retries=3
            )

            if not llm_response["success"]:
                logger.error(f"LLM Gateway failed to classify error: {llm_response.get('error')}")
                return None

            classified_error_data = llm_response["parsed_json"]
            
            try:
                validated_error = ClassifiedErrorSchema(**classified_error_data)
                logger.info(f"Successfully classified error: {validated_error.error_type}")
                return validated_error
            except ValidationError as e:
                logger.error(f"Validation error for classified error: {e.errors()}")
                return None

        except Exception as e:
            logger.error(f"Error in ErrorClassifierService: {e}", exc_info=True)
            return None

# Global instance
error_classifier_service = ErrorClassifierService(llm_gateway_service=llm_gateway)