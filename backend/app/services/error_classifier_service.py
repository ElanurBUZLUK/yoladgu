import logging
from typing import Dict, Any, List, Optional
from pydantic import ValidationError
import re

from app.services.llm_gateway import llm_gateway
from app.schemas.error_pattern import ClassifiedErrorSchema

logger = logging.getLogger(__name__)

class ErrorClassifierService:
    def __init__(self, llm_gateway_service: llm_gateway):
        self.llm_gateway = llm_gateway_service
        
        # Standardized error tags
        self.error_tags = {
            "grammar": ["subject_verb_agreement", "verb_tense", "article_usage", "preposition_usage", "pronoun_usage"],
            "spelling": ["homophone_confusion", "typo", "capitalization", "compound_words"],
            "punctuation": ["missing_comma", "missing_period", "apostrophe_usage", "quotation_marks"],
            "vocabulary": ["word_choice", "collocation", "idiom_usage", "register_mismatch"],
            "syntax": ["word_order", "sentence_structure", "fragment", "run_on"]
        }
        
        # Enhanced few-shot examples with standardized tags
        self.few_shot_examples = [
            {
                "input": "He go to the store.",
                "output": {
                    "error_type": "grammar:subject_verb_agreement",
                    "error_tag": "subject_verb_agreement",
                    "category": "grammar",
                    "explanation": "The verb 'go' does not agree with the third-person singular subject 'He'. It should be 'goes'.",
                    "confidence": 0.95,
                    "suggested_correction": "He goes to the store.",
                    "relevant_rule": "Subject-verb agreement: singular subjects take singular verbs.",
                    "skill_tag": "grammar_basics"
                }
            },
            {
                "input": "I like apple, banana and orange.",
                "output": {
                    "error_type": "punctuation:missing_comma",
                    "error_tag": "missing_comma",
                    "category": "punctuation",
                    "explanation": "A comma is needed before 'and' in a list of three or more items (Oxford comma).",
                    "confidence": 0.88,
                    "suggested_correction": "I like apple, banana, and orange.",
                    "relevant_rule": "Use commas to separate items in a list.",
                    "skill_tag": "punctuation_rules"
                }
            },
            {
                "input": "Their is no way.",
                "output": {
                    "error_type": "spelling:homophone_confusion",
                    "error_tag": "homophone_confusion",
                    "category": "spelling",
                    "explanation": "'Their' is a possessive pronoun. The correct word here is 'There', which indicates a place or existence.",
                    "confidence": 0.98,
                    "suggested_correction": "There is no way.",
                    "relevant_rule": "Distinguish between homophones like 'their', 'there', and 'they're'.",
                    "skill_tag": "homophones"
                }
            },
            {
                "input": "I have been to Paris last year.",
                "output": {
                    "error_type": "grammar:verb_tense",
                    "error_tag": "verb_tense",
                    "category": "grammar",
                    "explanation": "Present perfect 'have been' is used for experiences without specific time. For specific past time, use simple past.",
                    "confidence": 0.92,
                    "suggested_correction": "I went to Paris last year.",
                    "relevant_rule": "Use simple past for actions completed at a specific time in the past.",
                    "skill_tag": "verb_tenses"
                }
            },
            {
                "input": "She is very good in English.",
                "output": {
                    "error_type": "grammar:preposition_usage",
                    "error_tag": "preposition_usage",
                    "category": "grammar",
                    "explanation": "The correct preposition for 'good' when referring to a subject is 'at', not 'in'.",
                    "confidence": 0.85,
                    "suggested_correction": "She is very good at English.",
                    "relevant_rule": "Use 'at' for skills and abilities, 'in' for locations or fields.",
                    "skill_tag": "prepositions"
                }
            }
        ]

    async def classify_error(self, text: str) -> Optional[ClassifiedErrorSchema]:
        """Classifies the type of error in the given text using LLM few-shot and LanguageTool hybrid."""
        try:
            # 1. First, try LanguageTool for basic grammar/spelling detection
            lt_results = await self._check_with_languagetool(text)
            
            # 2. Use LLM for detailed classification
            llm_result = await self._classify_with_llm(text, lt_results)
            
            # 3. Combine and validate results
            final_result = self._combine_results(llm_result, lt_results)
            
            if final_result:
                validated_error = ClassifiedErrorSchema(**final_result)
                logger.info(f"Successfully classified error: {validated_error.error_type}")
                return validated_error
            
            return None

        except Exception as e:
            logger.error(f"Error in ErrorClassifierService: {e}", exc_info=True)
            return None

    async def _check_with_languagetool(self, text: str) -> List[Dict[str, Any]]:
        """Uses LanguageTool for basic grammar/spelling detection."""
        try:
            # LanguageTool API call (if available)
            # For now, we'll use a simplified version
            import requests
            
            try:
                response = requests.post(
                    "https://api.languagetool.org/v2/check",
                    data={
                        "text": text,
                        "language": "en-US"
                    },
                    timeout=5
                )
                
                if response.status_code == 200:
                    return response.json().get("matches", [])
                else:
                    logger.warning(f"LanguageTool API failed: {response.status_code}")
                    return []
                    
            except Exception as e:
                logger.warning(f"LanguageTool API error: {e}")
                return []
                
        except ImportError:
            logger.warning("requests not available for LanguageTool")
            return []

    async def _classify_with_llm(self, text: str, lt_results: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Uses LLM for detailed error classification."""
        # Build enhanced prompt with LanguageTool results
        lt_context = ""
        if lt_results:
            lt_context = f"\nLanguageTool detected {len(lt_results)} potential issues:\n"
            for i, result in enumerate(lt_results[:3]):  # Limit to first 3
                lt_context += f"- {result.get('message', 'Unknown issue')}\n"
        
        prompt = f"""
        Analyze the following text for grammatical or spelling errors and classify the primary error type.
        Provide an explanation, a confidence score (0.0-1.0), a suggested correction, and a relevant grammar rule.
        
        Text to analyze: "{text}"
        {lt_context}
        
        Use standardized error tags from this list:
        {self.error_tags}
        
        Respond in JSON format according to the following schema:
        {ClassifiedErrorSchema.model_json_schema()}
        
        Here are some examples:
        """
        for example in self.few_shot_examples:
            prompt += f"""
        Input: "{example['input']}"
        Output: {example['output']}
        """

        system_prompt = "You are an expert linguistic analyst and grammar checker. You must classify errors using standardized tags and provide corrections in the specified JSON format."

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
                            "schema": ClassifiedErrorSchema.model_json_schema(),
                            "lt_results": lt_results
                        }
                    )
                    
                    if mcp_response["success"]:
                        # Parse MCP response
                        classified_error_data = mcp_response["data"]
                        if isinstance(classified_error_data, str):
                            import json
                            classified_error_data = json.loads(classified_error_data)
                        
                        return classified_error_data
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

            return llm_response["parsed_json"]

        except Exception as e:
            logger.error(f"Error in LLM classification: {e}")
            return None

    def _combine_results(self, llm_result: Optional[Dict[str, Any]], lt_results: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Combines LLM and LanguageTool results for better accuracy."""
        if not llm_result:
            return None
            
        # If LanguageTool found issues, boost confidence
        if lt_results:
            llm_result["confidence"] = min(1.0, llm_result.get("confidence", 0.5) + 0.1)
            
            # Add LanguageTool context if available
            if "lt_context" not in llm_result:
                llm_result["lt_context"] = [r.get("message") for r in lt_results[:2]]
        
        # Ensure standardized error tag
        if "error_tag" not in llm_result and "error_type" in llm_result:
            error_type = llm_result["error_type"]
            if ":" in error_type:
                llm_result["error_tag"] = error_type.split(":")[1]
            else:
                llm_result["error_tag"] = error_type.lower().replace(" ", "_")
        
        # Add category if missing
        if "category" not in llm_result and "error_tag" in llm_result:
            for category, tags in self.error_tags.items():
                if llm_result["error_tag"] in tags:
                    llm_result["category"] = category
                    break
        
        return llm_result

    def get_standardized_error_tags(self) -> Dict[str, List[str]]:
        """Returns the standardized error tags for external use."""
        return self.error_tags

    def validate_error_tag(self, error_tag: str) -> bool:
        """Validates if an error tag is standardized."""
        for tags in self.error_tags.values():
            if error_tag in tags:
                return True
        return False

    def get_skill_tag_for_error(self, error_tag: str) -> str:
        """Maps error tags to skill tags for learning objectives."""
        skill_mapping = {
            "subject_verb_agreement": "grammar_basics",
            "verb_tense": "verb_tenses", 
            "article_usage": "articles",
            "preposition_usage": "prepositions",
            "pronoun_usage": "pronouns",
            "homophone_confusion": "homophones",
            "typo": "spelling_basics",
            "capitalization": "capitalization_rules",
            "missing_comma": "punctuation_rules",
            "missing_period": "punctuation_rules",
            "word_choice": "vocabulary",
            "collocation": "collocations",
            "word_order": "syntax_basics"
        }
        return skill_mapping.get(error_tag, "general_grammar")

# Global instance
error_classifier_service = ErrorClassifierService(llm_gateway_service=llm_gateway)