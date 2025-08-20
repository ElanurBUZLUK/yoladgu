import json
import logging
from typing import Dict, Any, List, Optional, Type, Union
from pydantic import BaseModel, ValidationError, create_model
from pydantic.json_schema import GenerateJsonSchema

logger = logging.getLogger(__name__)


class MCPSchemaValidator:
    """MCP Schema Validator for Pydantic models"""
    
    def __init__(self):
        self.registered_schemas: Dict[str, Type[BaseModel]] = {}
        self.validation_cache: Dict[str, Dict[str, Any]] = {}
    
    def register_schema(self, name: str, model: Type[BaseModel]) -> None:
        """Register a Pydantic model for MCP schema validation"""
        self.registered_schemas[name] = model
        logger.info(f"Registered MCP schema: {name}")
    
    def get_json_schema(self, model: Type[BaseModel]) -> Dict[str, Any]:
        """Get JSON schema from Pydantic model"""
        try:
            schema = model.model_json_schema()
            return schema
        except Exception as e:
            logger.error(f"Failed to generate JSON schema for {model.__name__}: {e}")
            return {}
    
    def validate_input(self, schema_name: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate input data against registered schema"""
        if schema_name not in self.registered_schemas:
            raise ValueError(f"Schema '{schema_name}' not registered")
        
        model = self.registered_schemas[schema_name]
        
        try:
            # Validate and convert data
            validated_data = model(**data)
            return validated_data.model_dump()
        except ValidationError as e:
            logger.error(f"Validation error for schema '{schema_name}': {e.errors()}")
            raise ValueError(f"Validation failed: {e.errors()}")
    
    def validate_output(self, schema_name: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate output data against registered schema"""
        return self.validate_input(schema_name, data)
    
    def generate_mcp_schema(self, model: Type[BaseModel], description: str = "") -> Dict[str, Any]:
        """Generate MCP-compatible schema from Pydantic model"""
        try:
            json_schema = self.get_json_schema(model)
            
            # Convert to MCP format
            mcp_schema = {
                "type": "object",
                "properties": json_schema.get("properties", {}),
                "required": json_schema.get("required", []),
                "description": description or model.__doc__ or "",
                "additionalProperties": False
            }
            
            return mcp_schema
        except Exception as e:
            logger.error(f"Failed to generate MCP schema: {e}")
            return {}
    
    def create_dynamic_schema(self, name: str, fields: Dict[str, Any]) -> Type[BaseModel]:
        """Create a dynamic Pydantic model for MCP"""
        try:
            # Convert field definitions to Pydantic field types
            model_fields = {}
            for field_name, field_config in fields.items():
                field_type = self._get_pydantic_type(field_config.get("type", "string"))
                model_fields[field_name] = (field_type, ...)  # Required field
            
            # Create dynamic model
            dynamic_model = create_model(name, **model_fields)
            self.register_schema(name, dynamic_model)
            
            return dynamic_model
        except Exception as e:
            logger.error(f"Failed to create dynamic schema '{name}': {e}")
            raise
    
    def _get_pydantic_type(self, json_type: str) -> Type:
        """Convert JSON schema type to Pydantic type"""
        type_mapping = {
            "string": str,
            "integer": int,
            "number": float,
            "boolean": bool,
            "array": List,
            "object": Dict[str, Any]
        }
        
        return type_mapping.get(json_type, str)
    
    def get_validation_stats(self) -> Dict[str, Any]:
        """Get validation statistics"""
        return {
            "registered_schemas": len(self.registered_schemas),
            "schema_names": list(self.registered_schemas.keys()),
            "cache_size": len(self.validation_cache)
        }


# Pre-defined schemas for common MCP operations
class MCPToolRequest(BaseModel):
    """MCP Tool Request Schema"""
    tool_name: str
    arguments: Dict[str, Any]
    user_id: Optional[str] = None
    session_id: Optional[str] = None


class MCPToolResponse(BaseModel):
    """MCP Tool Response Schema"""
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    method: str = "mcp"
    latency_ms: Optional[int] = None


class MCPContextUpdate(BaseModel):
    """MCP Context Update Schema"""
    context_type: str  # "student" or "session"
    context_data: Dict[str, Any]
    timestamp: str
    user_id: str


class MCPErrorClassification(BaseModel):
    """MCP Error Classification Schema"""
    error_type: str
    explanation: str
    confidence: float
    suggested_correction: str
    relevant_rule: str


class MCPCEFRAssessment(BaseModel):
    """MCP CEFR Assessment Schema"""
    overall_level: str
    skills: Dict[str, str]
    confidence: float
    feedback: str
    recommendations: List[str]


# Global schema validator instance
mcp_schema_validator = MCPSchemaValidator()

# Register common schemas
mcp_schema_validator.register_schema("tool_request", MCPToolRequest)
mcp_schema_validator.register_schema("tool_response", MCPToolResponse)
mcp_schema_validator.register_schema("context_update", MCPContextUpdate)
mcp_schema_validator.register_schema("error_classification", MCPErrorClassification)
mcp_schema_validator.register_schema("cefr_assessment", MCPCEFRAssessment)
