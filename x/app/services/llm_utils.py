import logging
import json
import re
from typing import Dict, Any, Optional, List, Union
from pydantic import ValidationError, BaseModel
from pydantic.json import pydantic_encoder

from app.services.llm_gateway import llm_gateway
from app.core.config import settings

logger = logging.getLogger(__name__)

class LLMUtils:
    """Utility class for LLM operations with self-repair capabilities"""
    
    def __init__(self):
        self.max_repair_attempts = getattr(settings, 'cloze_generation_params', {}).get('self_repair_attempts', 2)
        self.max_retries = getattr(settings, 'cloze_generation_params', {}).get('max_retries', 3)

    async def generate_json_with_repair(
        self,
        prompt: str,
        schema: Union[Dict[str, Any], BaseModel],
        system_prompt: Optional[str] = None,
        retries: int = 1,
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate JSON response with self-repair capabilities.
        
        Args:
            prompt: The user prompt
            schema: Pydantic model or JSON schema
            system_prompt: Optional system prompt
            retries: Number of retry attempts
            context: Optional context for the LLM
            
        Returns:
            Dict containing success status and parsed JSON or error
        """
        # Convert Pydantic model to JSON schema if needed
        if isinstance(schema, BaseModel):
            json_schema = schema.model_json_schema()
        else:
            json_schema = schema
        
        for attempt in range(retries + 1):
            try:
                # Generate JSON response
                llm_response = await llm_gateway.generate_json(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    schema=json_schema,
                    context=context
                )
                
                if not llm_response.get("success", False):
                    logger.warning(f"LLM generation failed on attempt {attempt + 1}: {llm_response.get('error')}")
                    if attempt < retries:
                        continue
                    return llm_response
                
                # Try to parse and validate the response
                parsed_json = llm_response.get("parsed_json")
                if not parsed_json:
                    logger.warning(f"No parsed JSON in response on attempt {attempt + 1}")
                    if attempt < retries:
                        continue
                    return llm_response
                
                # Attempt self-repair if needed
                repaired_json = self._repair_json_response(parsed_json, json_schema)
                if repaired_json:
                    # Validate against schema if it's a Pydantic model
                    if isinstance(schema, BaseModel):
                        try:
                            validated_data = schema(**repaired_json)
                            return {
                                "success": True,
                                "parsed_json": repaired_json,
                                "validated_data": validated_data,
                                "repair_attempts": attempt + 1
                            }
                        except ValidationError as e:
                            logger.warning(f"Validation failed after repair on attempt {attempt + 1}: {e}")
                            if attempt < retries:
                                continue
                    else:
                        return {
                            "success": True,
                            "parsed_json": repaired_json,
                            "repair_attempts": attempt + 1
                        }
                else:
                    # No repair needed, return original
                    if isinstance(schema, BaseModel):
                        try:
                            validated_data = schema(**parsed_json)
                            return {
                                "success": True,
                                "parsed_json": parsed_json,
                                "validated_data": validated_data
                            }
                        except ValidationError as e:
                            logger.warning(f"Validation failed on attempt {attempt + 1}: {e}")
                            if attempt < retries:
                                continue
                    else:
                        return {
                            "success": True,
                            "parsed_json": parsed_json
                        }
                
            except Exception as e:
                logger.error(f"Error in generate_json_with_repair on attempt {attempt + 1}: {e}")
                if attempt < retries:
                    continue
                return {
                    "success": False,
                    "error": f"Failed after {retries + 1} attempts: {str(e)}"
                }
        
        return {
            "success": False,
            "error": f"Failed to generate valid JSON after {retries + 1} attempts"
        }

    def _repair_json_response(self, response: Union[str, Dict, List], schema: Dict[str, Any]) -> Optional[Union[Dict, List]]:
        """
        Attempt to repair malformed JSON response.
        
        Args:
            response: The response from LLM (could be string, dict, or list)
            schema: JSON schema for validation
            
        Returns:
            Repaired JSON or None if repair failed
        """
        try:
            # If response is already a dict/list, try to repair it
            if isinstance(response, (dict, list)):
                return self._repair_json_structure(response, schema)
            
            # If response is a string, try to parse it
            if isinstance(response, str):
                return self._repair_json_string(response, schema)
            
            return None
            
        except Exception as e:
            logger.error(f"Error in JSON repair: {e}")
            return None

    def _repair_json_string(self, json_string: str, schema: Dict[str, Any]) -> Optional[Union[Dict, List]]:
        """Repair JSON string by cleaning and re-parsing."""
        try:
            # Clean the string
            cleaned_string = self._clean_json_string(json_string)
            
            # Try to parse
            parsed = json.loads(cleaned_string)
            
            # Validate structure
            return self._repair_json_structure(parsed, schema)
            
        except json.JSONDecodeError as e:
            logger.warning(f"JSON decode error: {e}")
            
            # Try to fix common JSON issues
            fixed_string = self._fix_common_json_issues(json_string)
            try:
                parsed = json.loads(fixed_string)
                return self._repair_json_structure(parsed, schema)
            except json.JSONDecodeError:
                return None

    def _clean_json_string(self, json_string: str) -> str:
        """Clean JSON string by removing markdown and extra content."""
        # Remove markdown code blocks
        json_string = re.sub(r'```json\s*', '', json_string)
        json_string = re.sub(r'```\s*$', '', json_string)
        
        # Remove extra text before/after JSON
        json_start = json_string.find('{')
        json_end = json_string.rfind('}') + 1
        
        if json_start != -1 and json_end > json_start:
            json_string = json_string[json_start:json_end]
        
        # Handle array responses
        if json_string.find('[') != -1:
            array_start = json_string.find('[')
            array_end = json_string.rfind(']') + 1
            if array_start != -1 and array_end > array_start:
                json_string = json_string[array_start:array_end]
        
        return json_string.strip()

    def _fix_common_json_issues(self, json_string: str) -> str:
        """Fix common JSON formatting issues."""
        # Fix missing quotes around keys
        json_string = re.sub(r'(\w+):', r'"\1":', json_string)
        
        # Fix trailing commas
        json_string = re.sub(r',(\s*[}\]])', r'\1', json_string)
        
        # Fix missing quotes around string values
        json_string = re.sub(r':\s*([^"][^,}\]]*[^"\s,}\]])', r': "\1"', json_string)
        
        # Fix unescaped quotes in strings
        json_string = re.sub(r'([^\\])"([^"]*?)([^\\])"', r'\1"\2\3"', json_string)
        
        return json_string

    def _repair_json_structure(self, data: Union[Dict, List], schema: Dict[str, Any]) -> Optional[Union[Dict, List]]:
        """Repair JSON structure based on schema requirements."""
        try:
            # Handle different schema types
            if schema.get("type") == "array":
                return self._repair_array_structure(data, schema)
            elif schema.get("type") == "object":
                return self._repair_object_structure(data, schema)
            else:
                return data
                
        except Exception as e:
            logger.error(f"Error repairing JSON structure: {e}")
            return None

    def _repair_array_structure(self, data: Any, schema: Dict[str, Any]) -> Optional[List]:
        """Repair array structure."""
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            # Check if data is wrapped in an object
            for key in ["data", "items", "questions", "results"]:
                if key in data and isinstance(data[key], list):
                    return data[key]
            # If no wrapper found, wrap the dict in an array
            return [data]
        else:
            # Wrap single item in array
            return [data]

    def _repair_object_structure(self, data: Any, schema: Dict[str, Any]) -> Optional[Dict]:
        """Repair object structure."""
        if isinstance(data, dict):
            # Check required properties
            required_props = schema.get("required", [])
            for prop in required_props:
                if prop not in data:
                    # Add default value based on schema
                    prop_schema = schema.get("properties", {}).get(prop, {})
                    data[prop] = self._get_default_value(prop_schema)
            
            return data
        elif isinstance(data, list) and len(data) == 1:
            # If array with single object, extract it
            return data[0]
        else:
            # Wrap in object if needed
            return {"data": data}

    def _get_default_value(self, prop_schema: Dict[str, Any]) -> Any:
        """Get default value based on property schema."""
        prop_type = prop_schema.get("type")
        
        if prop_type == "string":
            return prop_schema.get("default", "")
        elif prop_type == "number":
            return prop_schema.get("default", 0)
        elif prop_type == "integer":
            return prop_schema.get("default", 0)
        elif prop_type == "boolean":
            return prop_schema.get("default", False)
        elif prop_type == "array":
            return prop_schema.get("default", [])
        elif prop_type == "object":
            return prop_schema.get("default", {})
        else:
            return None

    def validate_schema_compliance(self, data: Any, schema: Dict[str, Any]) -> bool:
        """Validate if data complies with schema."""
        try:
            # Basic type validation
            schema_type = schema.get("type")
            
            if schema_type == "array":
                return isinstance(data, list)
            elif schema_type == "object":
                return isinstance(data, dict)
            elif schema_type == "string":
                return isinstance(data, str)
            elif schema_type == "number":
                return isinstance(data, (int, float))
            elif schema_type == "integer":
                return isinstance(data, int)
            elif schema_type == "boolean":
                return isinstance(data, bool)
            
            return True
            
        except Exception as e:
            logger.error(f"Schema validation error: {e}")
            return False

# Global instance
llm_utils = LLMUtils()
